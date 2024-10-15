from utils.normlization import *
import numpy as np
import torch
import os
from dataset.load_dataset import read_file,fun


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
        self.data = raw_data

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

        act_data = torch.from_numpy(act_data).to(torch.float32)
        self.act_normalizer = GaussianNormalizer(act_data.view(-1, act_data.shape[-1]))
        act_data = self.act_normalizer.normalize(act_data)

        self.data['obs_data'] = obs_data
        self.data['act_data'] = act_data
        self.data['traj_aver'] = torch.from_numpy(traj_aver).to(torch.float32)
        self.data['traj_mask'] = torch.from_numpy(traj_mask).to(torch.float32)
        self.data['traj_task'] = torch.from_numpy(traj_task).to(torch.float32)
        self.dataset_size = len(self.data['traj_aver'])
        self.traj_length = self.data['traj_mask'].sum(-1)
        self.traj_max_length = obs_data.shape[1]
        self.horizon = self.config['diffusion_cfgs']['horizon']

    def load2(self, file):
        raw_data = fun(file)
        self.data = raw_data['raw_data']
        self.obs_normalizer = raw_data['obs_normalizer']
        self.act_normalizer = raw_data['act_normalizer']
        self.dataset_size = len(self.data['traj_aver'])
        self.traj_length = self.data['traj_mask'].sum(-1)
        self.traj_max_length = self.data['obs_data'].shape[1]
        self.horizon = self.config['diffusion_cfgs']['horizon']

    def generate_traj_emb(self):
        X_patch = []
        batch_size = 64
        model = self.config['condition_model']
        X = self.data['obs_data']
        X_mask = self.data['traj_mask']
        for i in range(X.shape[0] // batch_size):
            X_patch.append(
                model.trajectory_embedding(
                    X[i * batch_size:(i + 1) * batch_size].to(model.device),
                    X_mask[i * batch_size:(i + 1) * batch_size].to(model.device)
                )[0].detach().cpu())
        X_patch.append(
            model.trajectory_embedding(
                X[(i + 1) * batch_size:].to(model.device),
                X_mask[(i + 1) * batch_size:].to(model.device)
            )[0].detach().cpu())
        self.data['traj_emb'] = torch.concatenate(X_patch, dim=0)

    def generate_comb_obs(self):
        comb_obs = []
        label_act = []
        for i, length in enumerate(self.traj_length):
            comb_obs.append(
                torch.concatenate(
                    (self.data['obs_data'][i, :int(length.item()), :],
                     self.data['obs_data'][i, 1:int(length.item()) + 1, :]), dim=-1))
            label_act.append(self.data['act_data'][i, :int(length.item()), :])
        self.data['comb_obs_data'] = torch.concatenate(comb_obs, dim=0)
        self.data['label_act_data'] = torch.concatenate(label_act, dim=0)

    def condition_model_training_sample(self):
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

    def dynamic_model_training_sample(self):
        traj_indices = torch.randint(low=0, high=self.data['comb_obs_data'].shape[0], size=(1024,))
        comb_obs = self.data['comb_obs_data'][traj_indices].to(device=torch.device(self.config['train_cfgs']['device']))
        label_act = self.data['label_act_data'][traj_indices].to(
            device=torch.device(self.config['train_cfgs']['device']))
        return comb_obs, label_act

    def diffuser_training_sample(self):
        traj_indices = torch.randint(low=0, high=self.dataset_size, size=(self.batch_size,))

        batch_traj = []
        for i, traj_index in enumerate(traj_indices):
            idx_max = min(int(self.traj_length[traj_index].item()), self.traj_max_length - self.horizon)
            start_idx = torch.randint(low=0, high=idx_max, size=(1,))
            end_idx = start_idx + self.horizon
            batch_traj.append(
                torch.concatenate(
                    (self.data['obs_data'][traj_index][start_idx:end_idx],
                     self.data['act_data'][traj_index][start_idx:end_idx]), dim=-1)
            )

        batch_traj = torch.stack(batch_traj, dim=0)
        if not self.config['denoise_action']:
            batch_traj = batch_traj[:, :, :self.config['environment'].observation_space.shape[0]]
        batch_traj = batch_traj.to(device=torch.device(self.config['train_cfgs']['device']))
        traj_emb = self.data['traj_emb'][traj_indices].to(device=torch.device(self.config['train_cfgs']['device']))
        return {
            'traj_data': batch_traj,
            'traj_emb': traj_emb,
            'init_obs': batch_traj[:, 0, :self.config['environment'].observation_space.shape[0]]
        }


if __name__ == '__main__':
    td = TaskDataset(0)
    td.sample()
