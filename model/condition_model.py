from module.instruction_module import TrajectoryEmbedding, TaskEmbedding
from dataset.dataset_class import TaskDataset
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


class ConditionModel:

    def __init__(self, config):
        self.device = torch.device("cuda:0")
        self.save_freq = 2000
        self.trajectory_embedding = TrajectoryEmbedding(39, 128, 64, 4, 2).to(self.device)
        self.task_embedding = TaskEmbedding(10, 128, 64).to(self.device)

        self.dataset = TaskDataset(0)
        self.traj_emb_optimizer = torch.optim.Adam(self.trajectory_embedding.parameters(), lr=1e-3)
        self.task_emb_optimizer = torch.optim.Adam(self.task_embedding.parameters(), lr=1e-3)
        self.traj_optim_scheduler = StepLR(self.traj_emb_optimizer, step_size=50, gamma=0.9)
        self.task_optim_scheduler = StepLR(self.task_emb_optimizer, step_size=50, gamma=0.9)

    def train(self, ep_num=10000):
        for ep_index in range(ep_num):
            batch_sample = self.dataset.sample()
            traj_max = batch_sample[0].to(self.device)
            traj_min = batch_sample[1].to(self.device)
            task = batch_sample[2].to(self.device)
            obs_traj_max = traj_max[:, :, :39]
            obs_traj_min = traj_min[:, :, :39]
            max_emb = self.trajectory_embedding(obs_traj_max)
            min_emb = self.trajectory_embedding(obs_traj_min)
            task_emb = self.task_embedding(task)
            kl_max_task = task_emb[1].detach() - max_emb[1] + 0.5 * (
                    (2 * max_emb[1]).exp() + (max_emb[0] - task_emb[0].detach()) ** 2) / (
                                  2 * task_emb[1].detach()).exp() - 0.5
            kl_min_task = task_emb[1].detach() - min_emb[1] + 0.5 * (
                    (2 * min_emb[1]).exp() + (min_emb[0] - task_emb[0].detach()) ** 2) / (
                                  2 * task_emb[1].detach()).exp() - 0.5

            distance1 = (((max_emb[0] - task_emb[0].detach()) ** 2).sum()) ** 0.5
            distance2 = (((min_emb[0] - task_emb[0].detach()) ** 2).sum()) ** 0.5
            distance = distance1 - distance2 + 1e-6
            distance_loss = torch.max(distance, torch.zeros_like(distance))
            loss_traj_emb = (kl_max_task + 1 / kl_min_task + distance_loss).mean()
            self.traj_emb_optimizer.zero_grad()
            loss_traj_emb.backward()
            torch.nn.utils.clip_grad_norm_(self.trajectory_embedding.parameters(), max_norm=1.0)
            self.traj_emb_optimizer.step()
            # self.traj_optim_scheduler.step()

            kl_max_task = task_emb[1] - max_emb[1].detach() + 0.5 * (
                    (2 * max_emb[1].detach()).exp() + (max_emb[0].detach() - task_emb[0]) ** 2) / (
                                  2 * task_emb[1]).exp() - 0.5
            kl_min_task = task_emb[1] - min_emb[1].detach() + 0.5 * (
                    (2 * min_emb[1].detach()).exp() + (min_emb[0].detach() - task_emb[0]) ** 2) / (
                                  2 * task_emb[1]).exp() - 0.5

            distance1 = (((max_emb[0].detach() - task_emb[0]) ** 2).sum()) ** 0.5
            distance2 = (((min_emb[0].detach() - task_emb[0]) ** 2).sum()) ** 0.5
            distance = distance1 - distance2 + 1e-6
            distance_loss = torch.max(distance, torch.zeros_like(distance))
            loss_task_emb = (kl_max_task + 1 / kl_min_task + distance_loss).mean()
            self.task_emb_optimizer.zero_grad()
            loss_task_emb.backward()
            torch.nn.utils.clip_grad_norm_(self.task_embedding.parameters(), max_norm=1.0)
            self.task_emb_optimizer.step()
            # self.task_optim_scheduler.step()

            print('ep:', ep_index, '  loss:', loss_traj_emb)
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
    model.train()
