import random
from utils.normlization import *
import numpy as np
import torch
import os
from dataset.load_dataset import read_file


class Dataset:

    def __init__(self):
        self.data = {}
        self.raw_data = 0

    def sample(self):
        return 0

    def load(self, file):
        return 0


class TaskDataset(Dataset):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data = {
            'obs_data': [],
            'act_data': [],
            'traj_task': [],
            'traj_mask': [],
            'traj_aver': [],
        }
        self.data_max = 0
        self.data_min = 0
        self.data_task = 0
        self.normalization = 0
        self.normalizer = 0
        self.batch_size = config['train_cfgs']['batch_size']
        self.tuple_size = 20
        self.dataset_size = 0

        dataset_file_path = os.path.join('../datasets', config['train_cfgs']['dataset'])
        self.load(file=dataset_file_path)

    def load(self, file):

        raw_data = read_file(file)
        self.raw_data = raw_data

        obs_data = []
        act_data = []
        traj_aver = []
        traj_task = []
        traj_mask = []
        env_num = len(list(raw_data.keys()))

        for env_index, env_name in enumerate(list(raw_data.keys())):
            obs_data.append(raw_data[env_name]['obs'])
            act_data.append(raw_data[env_name]['action'])
            traj_aver.append(raw_data[env_name]['info']['ave_reward'])
            traj_mask.append(raw_data[env_name]['traj_mask'])
            env_traj_num = len(raw_data[env_name]['info']['ave_reward'])
            if 'traj_preference' in raw_data[env_name].keys():
                task_one_hot = raw_data[env_name]['traj_preference']
            else:
                task_one_hot = np.zeros((env_traj_num, env_num))
                task_one_hot[:, env_index] = 1
            traj_task.append(task_one_hot)

        obs_data = np.concatenate(obs_data, 0)
        act_data = np.concatenate(act_data, 0)
        traj_aver = np.concatenate(traj_aver, 0)
        traj_mask = np.concatenate(traj_mask, 0)
        traj_task = np.concatenate(traj_task, 0)

        sort_index = np.argsort(traj_aver)
        obs_data = obs_data[sort_index]
        act_data = act_data[sort_index]
        traj_aver = traj_aver[sort_index]
        traj_mask = traj_mask[sort_index]
        traj_task = traj_task[sort_index]

        obs_data = torch.from_numpy(obs_data).to(torch.float32)
        self.normalizer = GaussianNormalizer(obs_data.view(-1, obs_data.shape[-1]))
        obs_data = self.normalizer.normalize(obs_data)

        self.data['obs_data'] = obs_data
        self.data['act_data'] = torch.from_numpy(act_data).to(torch.float32)
        self.data['traj_aver'] = torch.from_numpy(traj_aver).to(torch.float32)
        self.data['traj_mask'] = torch.from_numpy(traj_mask).to(torch.float32)
        self.data['traj_task'] = torch.from_numpy(traj_task).to(torch.float32)

        self.dataset_size = len(self.data['traj_aver'])

    def sample(self):
        sample_end = torch.randint(low=self.tuple_size, high=self.dataset_size, size=(self.batch_size,))
        sample_index = []
        for end in sample_end:
            sample_index.append(torch.randint(low=0, high=end, size=(self.tuple_size,)))
        sample_index = torch.stack(sample_index, dim=0)
        traj_max = self.data['obs_data'][sample_end]
        traj_min = self.data['obs_data'][sample_index]
        shape = traj_min.shape
        traj_min = traj_min.view(-1, shape[-2], shape[-1])

        traj_max_mask = self.data['traj_mask'][sample_end]
        traj_min_mask = self.data['traj_mask'][sample_index]
        shape = traj_min_mask.shape
        traj_min_mask = traj_min_mask.view(-1, shape[-1])

        traj_task = self.data['traj_task'][sample_end]
        return traj_max, traj_min, traj_max_mask, traj_min_mask, traj_task

    def diffuser_training_sample(self):

        traj_indices = torch.randint(low=0, high=self.dataset_size, size=(self.batch_size,))
        traj_act = self.data['act_data'][traj_indices]
        traj_obs = self.data['obs_data'][traj_indices]
        batch_trajectories = torch.cat((traj_act, traj_obs), dim=-1)
        batch_trajectories = batch_trajectories.to(device=torch.device(self.config['train_cfgs']['device']))
        # batch_returns = batch_returns.to(device=self._device)
        # for key, value in batch_conditions.items():
        #     batch_conditions[key] = batch_conditions[key].to(device=self._device)

        # sample_batch = {}
        # sample_batch['trajectories'] = batch_trajectories

        return batch_trajectories


if __name__ == '__main__':
    td = TaskDataset(0)
    td.sample()
