import numpy as np



def npz2raw_data(flie_path):
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
    ave_reward = raw_data[env_name]['reward'].mean(-1)
    norm_ave_reward = (ave_reward-ave_reward.min())/(ave_reward.max()-ave_reward.min())
    raw_data[env_name]['info']['ave_reward'] = norm_ave_reward
    raw_data[env_name]['traj_mask'] = np.ones_like(raw_data[env_name]['done'])
    raw_data[env_name]['traj_preference'] = norm_ave_reward.reshape(-1, 1)
    return raw_data


if __name__ == '__main__':
    file_path = '../datasets/walker2d_full_replay-v2.npz'
    npz2raw_data(file_path)
