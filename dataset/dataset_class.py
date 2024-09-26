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
            'traj_max': [],
            'traj_min': [],
            'traj_max_mask': [],
            'traj_min_mask': [],
            'traj_task': []
        }
        self.data_max = 0
        self.data_min = 0
        self.data_task = 0
        self.normalization = 0
        self.obs_normalization = 0
        self.batch_size = 128
        self.dataset_size = 0
        self.load(file='../datasets/MT10_rn1_sc0.pkl')

    def load(self, file):
        raw_data = read_pickle(file)
        self.raw_data = raw_data

        obs_data = []
        for env_name in list(raw_data.keys()):
            obs_data.append(raw_data[env_name]['obs'])
        obs_data = np.concatenate(obs_data, 0)
        self.obs_normalization = GaussianNormalizer(
            torch.from_numpy(obs_data.reshape(-1, obs_data.shape[-1])).to(torch.float32))

        data = []
        for env_name in list(raw_data.keys()):
            raw_data[env_name]['data'] = np.concatenate(
                [raw_data[env_name]['obs'],
                 raw_data[env_name]['next_obs'],
                 raw_data[env_name]['action'], ], -1
            )
            data.append(raw_data[env_name]['data'])
        data = np.concatenate(data, 0)
        data = torch.from_numpy(data.reshape(-1, data.shape[-1])).to(torch.float32)
        self.normalization = GaussianNormalizer(data)

        data_max = []
        data_min = []
        data_task = []
        data_traj_max_mask = []
        data_traj_min_mask = []

        for env_index, env_name in enumerate(list(raw_data.keys())):
            shape = raw_data[env_name]['obs'].shape
            traj_num = shape[0]
            for i in range(traj_num):
                traj1_index = i
                traj2_index = random.randint(0, traj_num - 1)
                task_one_hot = np.zeros(len(list(raw_data.keys())))
                task_one_hot[env_index] = 1
                data_task.append(task_one_hot)
                traj1_ave_reward = raw_data[env_name]['info']['ave_reward'][traj1_index]
                traj2_ave_reward = raw_data[env_name]['info']['ave_reward'][traj2_index]

                if traj1_ave_reward >= traj2_ave_reward:
                    max_traj_index = traj1_index
                    min_traj_index = traj2_index
                else:
                    max_traj_index = traj2_index
                    min_traj_index = traj1_index

                data_max.append(raw_data[env_name]['data'][max_traj_index])
                data_min.append(raw_data[env_name]['data'][min_traj_index])
                data_traj_max_mask.append(raw_data[env_name]['traj_mask'][max_traj_index])
                data_traj_min_mask.append(raw_data[env_name]['traj_mask'][min_traj_index])

                data_task.append(task_one_hot)
                data_max.append(raw_data[env_name]['data'][traj1_index])
                traj3_env = random.choice(list(raw_data.keys()))
                traj3_index = random.randint(0, traj_num - 1)
                data_min.append(raw_data[traj3_env]['data'][traj3_index])
                data_traj_max_mask.append(raw_data[env_name]['traj_mask'][traj1_index])
                data_traj_min_mask.append(raw_data[traj3_env]['traj_mask'][traj3_index])
        self.data['traj_max'] = self.normalization.normalize(torch.from_numpy(np.stack(data_max, 0)).to(torch.float32))
        self.data['traj_min'] = self.normalization.normalize(torch.from_numpy(np.stack(data_min, 0)).to(torch.float32))
        self.data['traj_max_mask'] = torch.from_numpy(np.stack(data_traj_max_mask, 0)).to(torch.float32)
        self.data['traj_min_mask'] = torch.from_numpy(np.stack(data_traj_min_mask, 0)).to(torch.float32)
        self.data['traj_task'] = torch.from_numpy(np.stack(data_task, 0)).to(torch.float32)
        self.dataset_size = len(self.data['traj_task'])

    def sample(self):
        sample_index = torch.randint(low=0, high=self.dataset_size, size=(self.batch_size,))
        max_traj = self.data['traj_max'][sample_index]
        min_traj = self.data['traj_max'][sample_index]
        max_traj_mask = self.data['traj_max_mask'][sample_index]
        min_traj_mask = self.data['traj_max_mask'][sample_index]
        task = self.data['traj_task'][sample_index]
        return max_traj, min_traj, task, max_traj_mask, min_traj_mask


if __name__ == '__main__':
    td = TaskDataset(0)
    td.sample()
