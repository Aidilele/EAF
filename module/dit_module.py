import math
import torch
import einops
import torch.nn as nn
from typing import Optional
from module.helpers import SinusoidalPosEmb
from torch.distributions import Bernoulli


# hidden_size = 384 | 768 | 1024 | 1152
# depth =       12  | 24  | 28
# patch_size =  2   | 4   | 8
# n_heads =     6   | 12  | 16  (hidden_size can be divided by n_heads)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiscreteCondEmbedder(nn.Module):
    def __init__(self,
                 attr_dim: int, hidden_size: int, num_bins: int = 100):
        super().__init__()
        self.num_bins, self.attr_dim = num_bins, attr_dim
        self.embedding = nn.Embedding(attr_dim * num_bins, 128)
        self.attn = nn.MultiheadAttention(128, num_heads=2, batch_first=True)
        self.linear = nn.Linear(128 * attr_dim, hidden_size)

    def forward(self, attr: torch.Tensor, mask: torch.Tensor = None):
        '''
        attr: (batch_size, attr_dim)
        mask: (batch_size, attr_dim) 0 or 1, 0 means ignoring
        '''
        offset = torch.arange(self.attr_dim, device=attr.device, dtype=torch.long)[None,] * self.num_bins
        # e.g. attr=[12, 42, 7] -> [12, 142, 207]
        emb = self.embedding(attr + offset)  # (b, attr_dim, 128)
        if mask is not None: emb *= mask.unsqueeze(-1)  # (b, attr_dim, 128)
        emb, _ = self.attn(emb, emb, emb)  # (b, attr_dim, 128)
        return self.linear(einops.rearrange(emb, 'b c d -> b (c d)'))  # (b, hidden_size)


class CondEmbedding(nn.Module):
    def __init__(self, cond_dim: int, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, dim), nn.Mish(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor):
        return self.mlp(x)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# class TimeEmbedding(nn.Module):
#     def __init__(self, dim: int):
#         super().__init__()
#         self.dim = dim
#         self.mlp = nn.Sequential(
#             nn.Linear(dim // 4, dim), nn.Mish(), nn.Linear(dim, dim))
#     def forward(self, x: torch.Tensor):
#         device = x.device
#         half_dim = self.dim // 8
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
#         emb = x[:, None] * emb[None, :]
#         emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
#         return self.mlp(emb)

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, dim), nn.Mish(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor):
        return self.mlp(x)


class SinPosTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.pos = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.Mish(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor):
        return self.mlp(self.pos(x))


class ObsAttentionEmbedding(nn.Module):

    def __init__(self, x_dim, hidden_size, n_heads, dropout=0.1):
        super().__init__()
        self.mlp1 = nn.Linear(x_dim, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2), approx_gelu(), nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size))
        self.pos_emb = SinusoidalPosEmb(hidden_size)
        self.pos_emb_cache = None

    def forward(self, x: torch.Tensor):
        if self.pos_emb_cache is None or self.pos_emb_cache.shape[0] != x.shape[1]:
            self.pos_emb_cache = self.pos_emb(torch.arange(x.shape[1], device=x.device))
        x = self.norm1(self.mlp1(x) + self.pos_emb_cache)
        x = x + self.attn(x, x, x)[0]
        x = (x + self.mlp2(self.norm2(x)))[:, -1, :]
        return x


class DiTBlock(nn.Module):
    """ A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning. """

    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), approx_gelu(), nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size * 2, hidden_size * 6))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x, x, x)[0]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Finallayer1d(nn.Module):
    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.output_layer = nn.Linear(hidden_size, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(2 * hidden_size, 2 * hidden_size))
        self.liner_layer = nn.Sequential(nn.Linear(hidden_size, 2 * hidden_size),
                                         nn.Tanh(),
                                         nn.Linear(2 * hidden_size, hidden_size))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.liner_layer(x)
        return self.output_layer(x)


class DiT1d(nn.Module):
    def __init__(self,
                 x_dim: int, cond_dim: int, action_dim: int,
                 hidden_dim: int = 128, n_heads: int = 4, depth: int = 8, dropout: float = 0.1,
                 condition_dropout=0.25):
        super().__init__()
        self.x_dim, self.action_dim, self.cond_dim, self.hidden_dim, self.n_heads, self.depth = x_dim, action_dim, cond_dim, hidden_dim, n_heads, depth
        self.x_proj = nn.Linear(x_dim, hidden_dim)
        self.obs_emb = ObsAttentionEmbedding(x_dim, hidden_dim, n_heads, dropout)
        self.t_emb = SinPosTimeEmbedding(hidden_dim)
        self.mask_dist = Bernoulli(probs=1 - condition_dropout)
        self.attr_proj = CondEmbedding(cond_dim, hidden_dim)
        self.pos_emb = SinusoidalPosEmb(hidden_dim)
        self.pos_emb_cache = None
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, n_heads, dropout) for _ in range(depth)])
        self.final_layer = Finallayer1d(hidden_dim, x_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_emb.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.output_layer.weight, 0)
        nn.init.constant_(self.final_layer.output_layer.bias, 0)

    def forward(self,
                x: torch.Tensor,
                obs: torch.Tensor,
                t: torch.Tensor,
                condition: Optional[torch.Tensor] = None,
                use_dropout=True,
                force_dropout=False
                ):
        '''
        Input:
            - x:    (batch, horizon, x_dim)
            - t:    (batch, 1)
            - attr: (batch, attr_dim)
            - mask: (batch, attr_dim)

        Output:
            - y:    (batch, horizon, x_dim)
        '''
        if self.pos_emb_cache is None or self.pos_emb_cache.shape[0] != x.shape[1]:
            self.pos_emb_cache = self.pos_emb(torch.arange(x.shape[1], device=x.device))
        x = self.x_proj(x) + self.pos_emb_cache[None,]
        t = self.t_emb(t)

        # obs_emb = self.obs_emb(obs)
        emb = self.attr_proj(condition)
        if use_dropout:
            mask = self.mask_dist.sample(sample_shape=(t.size(0), 1)).to(t.device)
            emb = mask * emb
            # obs_emb = mask * obs_emb
        if force_dropout:
            emb = 0 * emb
            # obs_emb = 0 * obs_emb
        t = torch.cat([t, emb], dim=-1)
        for block in self.blocks:
            x = block(x, t)
        x = self.final_layer(x, t)
        return x
