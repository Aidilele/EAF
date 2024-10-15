import torch
from dm_control.composer import observable
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import torch.nn.functional as F


class DynamicModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        obs_dim = self.config['environment'].observation_space.shape[0]
        act_dim = self.config['environment'].action_space.shape[0]
        hidden_dim = 256
        self.mlp = nn.Sequential(
            nn.Linear(2*obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )
        self.device = torch.device("cuda:0")
        self.save_freq = 500
        self.dataset = config['dataset']
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.optim_scheduler = StepLR(self.optimizer, step_size=50, gamma=0.9)
        self.mlp.to(self.device)
    def forward(self, x):
        return self.mlp(x)

    def train_model(self, ep_num=50000):
        for ep_index in range(ep_num):
            x, y = self.dataset.dynamic_model_training_sample()
            pred_y = self.forward(x)
            mse_loss = F.mse_loss(pred_y, y)
            self.optimizer.zero_grad()
            mse_loss.backward()
            self.optimizer.step()
            if ep_index % self.save_freq == 0:
                self.save(ep_index)
        self.save(ep_index)
        return 0

    def save(self, episode=0):
        torch.save(self.state_dict(), f'../runs/dynamic_model/model_{episode}.pt')
        return 0

    def load(self, episode=0):
        model_path = f'../runs/dynamic_model/model_{episode}.pt'
        model_dict = torch.load(model_path)
        self.load_state_dict(model_dict)
        return 0
