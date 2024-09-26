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
        self.batch_size = 128
        self.tuple_size = 12
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
        traj_task = []
        traj_mask = []
        env_num = len(list(raw_data.keys()))

        for env_index, env_name in enumerate(list(raw_data.keys())):
            env_traj_data = np.concatenate(
                [raw_data[env_name]['obs'],
                 raw_data[env_name]['next_obs'],
                 raw_data[env_name]['action'], ], -1
            )
            env_traj_aver = raw_data[env_name]['info']['ave_reward']
            env_traj_mask = raw_data[env_name]['traj_mask']
            env_traj_num = len(raw_data[env_name]['info']['ave_reward'])
            env_task_one_hot = np.zeros((env_traj_num, env_num))
            env_task_one_hot[:, env_index] = 1
            env_sort_index = np.argsort(env_traj_aver)
            traj_data.append(env_traj_data[env_sort_index])
            traj_mask.append(env_traj_mask[env_sort_index])
            traj_task.append(env_task_one_hot[env_sort_index])

        traj_data = np.concatenate(traj_data, 0)
        traj_mask = np.concatenate(traj_mask, 0)
        traj_task = np.concatenate(traj_task, 0)
        traj_data = torch.from_numpy(traj_data).to(torch.float32)
        self.normalization = GaussianNormalizer(traj_data)
        traj_data = self.normalization.normalize(traj_data)
        self.data['traj_data'] = traj_data
        self.data['traj_mask'] = torch.from_numpy(traj_mask).to(torch.float32)
        self.data['traj_task'] = torch.from_numpy(traj_task).to(torch.float32)
        self.dataset_size = len(self.data['traj_data'])
        traj_max = []
        traj_min = []
        traj_max_mask = []
        traj_min_mask = []
        traj_task = []
        for env_index in range(10):
            start = env_index * 250
            end = start + 250
            same_env_index = torch.randint(low=start, high=end, size=(500, self.tuple_size)).sort().values

            x = ([x for x in range(start)] + [x for x in range(end, self.dataset_size)]) * 500
            diff_env_index = random.sample(x, 500 * self.tuple_size)
            diff_env_index = torch.tensor(diff_env_index).view(500, -1)
            # if env_index == 0:
            #     diff_env_index = torch.randint(low=end, high=self.dataset_size, size=(500, self.tuple_size))
            # elif env_index == 9:
            #     diff_env_index = torch.randint(low=0, high=start, size=(500, self.tuple_size))
            # else:
            #     diff_env_index1 = torch.randint(low=0, high=start, size=(500, self.tuple_size // 2))
            #     diff_env_index2 = torch.randint(low=end, high=self.dataset_size, size=(500, self.tuple_size // 2))
            #     diff_env_index = torch.cat((diff_env_index1, diff_env_index2), dim=-1)

            traj_max_index = same_env_index[:, -1]
            traj_min_index = torch.cat((diff_env_index, same_env_index[:, :-1]), dim=-1)
            traj_max_ = self.data['traj_data'][traj_max_index]
            traj_min_ = self.data['traj_data'][traj_min_index]
            traj_max_mask_ = self.data['traj_mask'][traj_max_index]
            traj_min_mask_ = self.data['traj_mask'][traj_min_index]
            traj_task_ = self.data['traj_task'][traj_max_index]
            traj_max.append(traj_max_)
            traj_min.append(traj_min_)
            traj_max_mask.append(traj_max_mask_)
            traj_min_mask.append(traj_min_mask_)
            traj_task.append(traj_task_)

        self.data['traj_max'] = torch.stack(traj_max, dim=0).view(500 * 10, 100, 82)
        self.data['traj_min'] = torch.stack(traj_min, dim=0).view(500 * 10, -1, 100, 82)
        self.data['traj_max_mask'] = torch.stack(traj_max_mask, dim=0).view(500 * 10, 100)
        self.data['traj_min_mask'] = torch.stack(traj_min_mask, dim=0).view(500 * 10, -1, 100)
        self.data['traj_task'] = torch.stack(traj_task, dim=0).view(500 * 10, 10)

    def sample(self):
        sample_index = torch.randint(low=0, high=5000, size=(self.batch_size,))
        traj_max = self.data['traj_max'][sample_index]
        traj_min = self.data['traj_min'][sample_index].view(-1, 100, 82)
        traj_max_mask = self.data['traj_max_mask'][sample_index]
        traj_min_mask = self.data['traj_min_mask'][sample_index].view(-1, 100)
        traj_task = self.data['traj_task'][sample_index]
        return traj_max, traj_min, traj_max_mask, traj_min_mask, traj_task


if __name__ == '__main__':
    td = TaskDataset(0)
    td.sample()
