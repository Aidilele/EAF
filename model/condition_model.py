from module.instruction_module import TrajectoryEmbedding, TaskEmbedding
from dataset.dataset_class import TaskDataset
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from utils.functional import euc_distance, norm_kl_div


class ConditionModel:

    def __init__(self, config):
        self.device = torch.device("cuda:0")
        self.save_freq = 100
        self.trajectory_embedding = TrajectoryEmbedding(39, 128, 64, 4, 2).to(self.device)
        self.task_embedding = TaskEmbedding(10, 128, 64).to(self.device)

        self.dataset = TaskDataset(0)
        self.traj_emb_optimizer = torch.optim.Adam(self.trajectory_embedding.parameters(), lr=1e-3)
        self.task_emb_optimizer = torch.optim.Adam(self.task_embedding.parameters(), lr=1e-3)
        self.traj_optim_scheduler = StepLR(self.traj_emb_optimizer, step_size=50, gamma=0.9)
        self.task_optim_scheduler = StepLR(self.task_emb_optimizer, step_size=50, gamma=0.9)

    def train(self, ep_num=2000):

        for ep_index in range(ep_num):
            batch_sample = self.dataset.sample()
            traj_max = batch_sample[0].to(self.device)
            traj_min = batch_sample[1].to(self.device)
            traj_max_mask = batch_sample[2].to(self.device)
            traj_min_mask = batch_sample[3].to(self.device)
            task = batch_sample[-1].to(self.device)
            obs_traj_max = traj_max[:, :, :39]
            obs_traj_min = traj_min[:, :, :39]

            u_p, lv_p = self.trajectory_embedding(obs_traj_max, traj_max_mask)
            u_m, lv_m = self.trajectory_embedding(obs_traj_min, traj_min_mask)
            u_t, lv_t = self.task_embedding(task)

            kl_p_t = norm_kl_div(u_p, lv_p, u_t.detach(), lv_t.detach())
            kl_m_t = norm_kl_div(u_m, lv_m, u_t.detach(), lv_t.detach())

            kl_m = (kl_m_t + kl_m_t) * 0.5

            d_p_t = euc_distance(u_p, u_t.detach())
            d_m_t = euc_distance(u_m, u_t.detach())

            d_loss = d_p_t - d_m_t + 1e-6
            d_loss = torch.max(d_loss, torch.zeros_like(d_loss))
            loss_traj_emb = (kl_p_t + 1 / kl_m).mean() + d_loss.mean()
            self.traj_emb_optimizer.zero_grad()
            loss_traj_emb.backward()
            torch.nn.utils.clip_grad_norm_(self.trajectory_embedding.parameters(), max_norm=1.0)
            self.traj_emb_optimizer.step()
            self.traj_optim_scheduler.step()

            kl_p_t = norm_kl_div(u_p.detach(), lv_p.detach(), u_t, lv_t)
            kl_m_t = norm_kl_div(u_m.detach(), lv_m.detach(), u_t, lv_t)

            d_p_t = euc_distance(u_p.detach(), u_t)
            d_m_t = euc_distance(u_m.detach(), u_t)

            d_loss = d_p_t - d_m_t + 1e-6
            d_loss = torch.max(d_loss, torch.zeros_like(d_loss))
            loss_task_emb = (kl_p_t + 1 / kl_m_t).mean() + d_loss.mean()
            self.task_emb_optimizer.zero_grad()
            loss_task_emb.backward()
            torch.nn.utils.clip_grad_norm_(self.task_embedding.parameters(), max_norm=1.0)
            self.task_emb_optimizer.step()
            # self.task_optim_scheduler.step()

            print('ep:', ep_index, '  loss:', loss_task_emb)
            if ep_index % self.save_freq == 0:
                self.save(ep_index)

        self.save(ep_index)
        return 0

    def save(self, episode):
        model_dict = {
            "trajectory_embedding": self.trajectory_embedding.state_dict(),
            "task_embedding": self.task_embedding.state_dict()
        }
        torch.save(model_dict, f'../runs/model_{episode}.pt')
        return 0

    def load(self, episode):
        model_path = f'../runs/model_{episode}.pt'
        model_dict = torch.load(model_path)
        self.trajectory_embedding.load_state_dict(model_dict["trajectory_embedding"])
        self.task_embedding.load_state_dict(model_dict["task_embedding"])
        return 0


if __name__ == '__main__':
    model = ConditionModel(0)
    try:
        model.train()
    except:
        model.save(0)
