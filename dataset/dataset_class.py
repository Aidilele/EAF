import random
from utils.normlization import *
import numpy as np
import torch

from dataset.load_dataset import read_pickle


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
            'traj_data': [],
            'traj_task': [],
            'traj_mask': [],
            'traj_aver': [],
        }
        self.data_max = 0
        self.data_min = 0
        self.data_task = 0
        self.normalization = 0
        self.obs_normalization = 0
        self.batch_size = 32
        self.tuple_size = 50
        self.dataset_size = 0

        self.load(file='../datasets/MT10_rn5_sc0.5.pkl')

    def load(self, file):
        raw_data = read_pickle(file)
        self.raw_data = raw_data

        obs_data = []
        for env_name in list(raw_data.keys()):
            obs_data.append(raw_data[env_name]['obs'])
        obs_data = np.concatenate(obs_data, 0)
        self.obs_normalization = GaussianNormalizer(
            torch.from_numpy(obs_data.reshape(-1, obs_data.shape[-1])).to(torch.float32))

        traj_data = []
        traj_aver = []
        traj_task = []
        traj_mask = []
        env_num = len(list(raw_data.keys()))

        for env_index, env_name in enumerate(list(raw_data.keys())):
            traj_data.append(np.concatenate(
                [raw_data[env_name]['obs'],
                 raw_data[env_name]['next_obs'],
                 raw_data[env_name]['action'], ], -1
            ))
            traj_aver.append(raw_data[env_name]['info']['ave_reward'])
            traj_mask.append(raw_data[env_name]['traj_mask'])
            env_traj_num = len(raw_data[env_name]['info']['ave_reward'])
            task_one_hot = np.zeros((env_traj_num, env_num))
            task_one_hot[:, env_index] = 1
            traj_task.append(task_one_hot)

        traj_data = np.concatenate(traj_data, 0)
        traj_aver = np.concatenate(traj_aver, 0)
        traj_mask = np.concatenate(traj_mask, 0)
        traj_task = np.concatenate(traj_task, 0)

        sort_index = np.argsort(traj_aver)
        traj_data = traj_data[sort_index]
        traj_aver = traj_aver[sort_index]
        traj_mask = traj_mask[sort_index]
        traj_task = traj_task[sort_index]

        traj_data = torch.from_numpy(traj_data).to(torch.float32)
        self.normalization = GaussianNormalizer(traj_data)
        traj_data = self.normalization.normalize(traj_data)

        self.data['traj_data'] = traj_data
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
        traj_max = self.data['traj_data'][sample_end]
        traj_min = self.data['traj_data'][sample_index]
        shape = traj_min.shape
        traj_min = traj_min.view(-1, shape[-2], shape[-1])

        traj_max_mask = self.data['traj_mask'][sample_end]
        traj_min_mask = self.data['traj_mask'][sample_index]
        shape = traj_min_mask.shape
        traj_min_mask = traj_min_mask.view(-1, shape[-1])

        traj_task = self.data['traj_task'][sample_end]
        return traj_max, traj_min, traj_max_mask, traj_min_mask, traj_task


if __name__ == '__main__':
    td = TaskDataset(0)
    td.sample()
