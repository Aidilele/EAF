import pickle
from distutils.spawn import find_executable

import numpy as np
import torch
from utils.normlization import *




def read_pickle(file_name):
    with open(file_name, 'rb') as f:
        raw_data = pickle.load(f)

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
        task_one_hot = np.zeros((env_traj_num, env_num))
        task_one_hot[:, env_index] = 1
        traj_task.append(task_one_hot)

    obs_data = np.concatenate(obs_data, 0)
    act_data = np.concatenate(act_data, 0)
    traj_aver = np.concatenate(traj_aver, 0)
    traj_mask = np.concatenate(traj_mask, 0)
    traj_task = np.concatenate(traj_task, 0)

    obs_data = torch.from_numpy(obs_data).to(torch.float32)
    obs_normalizer = GaussianNormalizer(obs_data.view(-1, obs_data.shape[-1]))
    obs_data = obs_normalizer.normalize(obs_data)

    act_data = torch.from_numpy(act_data).to(torch.float32)
    act_normalizer = GaussianNormalizer(act_data.view(-1, act_data.shape[-1]))
    act_data = act_normalizer.normalize(act_data)

    data = {}
    sort_index = np.argsort(traj_aver)
    data['obs_data'] = obs_data[sort_index]
    data['act_data'] = act_data[sort_index]
    data['traj_aver'] = torch.from_numpy(traj_aver[sort_index]).to(torch.float32)
    data['traj_mask'] = torch.from_numpy(traj_mask[sort_index]).to(torch.float32)
    data['traj_task'] = torch.from_numpy(traj_task[sort_index]).to(torch.float32)

    return {
        'raw_data': data,
        'obs_normalizer': obs_normalizer,
        'act_normalizer': act_normalizer}

def read_npz(file_path):
    data = np.load(file_path)
    # env_name = file_path.split('/')[-1].split('.')[0]
    raw_data = {}
    for field in data.files:
        if data[field].ndim == 1:
            raw_data[field] = torch.from_numpy(data[field]).to(torch.float32)
        else:
            raw_data[field] = torch.from_numpy(data[field]).to(torch.float32)

    obs_normalizer = GaussianNormalizer(raw_data['obs'])
    act_normalizer = GaussianNormalizer(raw_data['action'])
    raw_data['obs'] = obs_normalizer.normalize(raw_data['obs'])
    raw_data['action'] = act_normalizer.normalize(raw_data['action'])

    traj_end_point = torch.where(raw_data['done'] == 1)[0]
    horizon = (traj_end_point[1:] - traj_end_point[:-1]).max()

    fixed_data = {}
    for field in data.files:
        if data[field].ndim == 1:
            fixed_data[field] = torch.zeros([traj_end_point.shape[0], horizon])
        else:
            fixed_data[field] = torch.zeros([traj_end_point.shape[0], horizon, data[field].shape[-1]])
    fixed_data['done'][:] = 1

    start_index = 0
    for index, end_index in enumerate(traj_end_point):
        length = end_index - start_index
        true_length = min(length, horizon)
        for field in data.files:
            fixed_data[field][index, :true_length] = raw_data[field][start_index:start_index + true_length]
        start_index = end_index + 1

    traj_mask = 1 - fixed_data['done']
    ave_reward = fixed_data['reward'].mean(-1)
    norm_ave_reward = (ave_reward - ave_reward.min()) / (ave_reward.max() - ave_reward.min())
    sort_index = torch.argsort(norm_ave_reward)

    fixed_data['traj_mask'] = traj_mask[sort_index]
    fixed_data['traj_aver'] = norm_ave_reward[sort_index]
    fixed_data['traj_task'] = norm_ave_reward.reshape(-1, 1)[sort_index]
    fixed_data['obs_data'] = fixed_data['obs'][sort_index]
    fixed_data['act_data'] = fixed_data['action'][sort_index]
    return {
        'raw_data': fixed_data,
        'obs_normalizer': obs_normalizer,
        'act_normalizer': act_normalizer}

def read_file(file_name):
    file_extension = file_name.split('.')[-1]
    if file_extension == 'pkl':
        data = read_pickle(file_name)
    elif file_extension == 'npz':
        data = read_npz(file_name)
    else:
        assert False, 'Unsupported file type'
    return data



if __name__ == "__main__":
    read_pickle()
