import pickle
import numpy as np

def read_npz(flie_path):
    horizon = 100
    data = np.load(flie_path)
    raw_data = {}
    env_name = flie_path.split('/')[-1].split('.')[0]
    raw_data[env_name] = {}
    for field in data.files:
        if data[field].ndim == 1:
            raw_data[env_name][field] = data[field].reshape(-1, horizon)
        else:
            raw_data[env_name][field] = data[field].reshape(-1, horizon, data[field].shape[-1])
    raw_data[env_name]['info'] = {}
    traj_mask = 1 - raw_data[env_name]['done']
    mask_index = np.where(traj_mask == 0)

    for i in range(len(mask_index[0])):
        traj_mask[mask_index[0][i], mask_index[1][i]:] = 1
        traj_mask[mask_index[0][i], mask_index[1][i] + 1:] = 0
    raw_data[env_name]['traj_mask'] = traj_mask
    ave_reward = (raw_data[env_name]['reward'] * traj_mask).mean(-1)
    norm_ave_reward = (ave_reward - ave_reward.min()) / (ave_reward.max() - ave_reward.min())
    raw_data[env_name]['info']['ave_reward'] = norm_ave_reward
    raw_data[env_name]['traj_preference'] = norm_ave_reward.reshape(-1, 1)
    return raw_data


def read_pickle(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


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
