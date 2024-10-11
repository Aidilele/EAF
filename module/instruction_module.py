from torch import nn
import torch
from module.generiese_module import SinusoidalPosEmb


class PreferenceEmbedding(nn.Module):
    def __init__(self, x_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean = nn.Linear(hidden_dim, output_dim)
        self.log_var = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.network(x)
        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var


class TaskEmbedding(nn.Module):
    def __init__(self, x_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean = nn.Linear(hidden_dim, output_dim)
        self.log_var = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.network(x)
        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var


class TrajectoryEmbedding(nn.Module):
    def __init__(self, x_dim, hidden_dim, output_dim, n_heads, layers, dropout=0.0):
        super().__init__()
        self.mlp1 = nn.Linear(x_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

        attn_block = nn.ModuleList()
        attn_block.append(nn.MultiheadAttention(hidden_dim, n_heads, dropout, batch_first=True))
        attn_block.append(nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6))
        attn_block.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2),
                                        nn.GELU(approximate="tanh"),
                                        nn.Dropout(dropout),
                                        nn.Linear(hidden_dim * 2, hidden_dim)))
        self.attn_module = nn.ModuleList()
        for i in range(layers):
            self.attn_module.append(attn_block)

        # self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout, batch_first=True)
        # self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        # self.mlp2 = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(approximate="tanh"), nn.Dropout(dropout),
        #     nn.Linear(hidden_dim * 2, hidden_dim))

        self.pos_emb = SinusoidalPosEmb(hidden_dim)
        self.pos_emb_cache = None
        self.mean = nn.Linear(hidden_dim, output_dim)
        self.log_var = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, traj_mask=None):
        if self.pos_emb_cache is None or self.pos_emb_cache.shape[0] != x.shape[1]:
            self.pos_emb_cache = self.pos_emb(torch.arange(x.shape[1], device=x.device))
        x = self.norm1(self.mlp1(x) + self.pos_emb_cache)

        for block in self.attn_module:
            x = x + block[0](x, x, x, key_padding_mask=traj_mask)[0]
            x = (x + block[2](block[1](x)))
        x = x[:, -1, :]
        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    traj_data = torch.randn((128, 50, 39))
    label_data = torch.eye(8).repeat(16, 1)
    label_y = np.where(label_data == 1)[1]
    traj_model = TrajectoryEmbedding(39, 64, 32, 4, 2)
    task_model = TaskEmbedding(8, 64, 32)
    # traj_emb = task_model(label_data).detach().numpy()
    traj_emb = traj_model(traj_data).detach().numpy()
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(traj_emb)
    print('ok')

    # 绘制结果
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=label_y, cmap='viridis', s=20)
    plt.title('t-SNE Visualization of High Dimensional Data')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar(scatter, label='Class Label')
    plt.show()
