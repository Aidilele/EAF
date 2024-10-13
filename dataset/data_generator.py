import random
import numpy as np
import metaworld
from metaworld.policies import *
from metaworld.policies import SawyerPegInsertionSideV2Policy as SawyerPegInsertSideV2Policy
import pickle


def generator_mt_dataset(seed=0, traj_length=100, repeat_num=1, scale=.0, dataset_type="MT10"):
    np.set_printoptions(suppress=True)
    np.random.seed(seed)
    ml = eval("metaworld." + dataset_type)()
    env_list = list(ml.train_classes.keys())
    # env_list = ['peg-insert-side-v2']
    raw_data = {}
    for env_index, env_name in enumerate(env_list):
        env = ml.train_classes[env_name]()
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        tasks = [t for t in ml.train_tasks if t.env_name == env_name]
        policy_name = 'Sawyer'
        for words in env_name.split("-"):
            policy_name += words.capitalize()
        policy_name += 'Policy'
        policy = eval(policy_name)()
        raw_data[env_name] = {}
        raw_data[env_name]['obs'] = np.zeros((len(tasks), repeat_num, traj_length, obs_dim))
        raw_data[env_name]['action'] = np.zeros((len(tasks), repeat_num, traj_length, act_dim))
        raw_data[env_name]['next_obs'] = np.zeros((len(tasks), repeat_num, traj_length, obs_dim))
        raw_data[env_name]['traj_mask'] = np.zeros((len(tasks), repeat_num, traj_length))
        raw_data[env_name]['info'] = {}
        raw_data[env_name]['info']['ave_reward'] = np.zeros((len(tasks), repeat_num))
        raw_data[env_name]['info']['done_count'] = np.zeros((len(tasks), repeat_num))
        raw_data[env_name]['info']['success'] = np.zeros((len(tasks), repeat_num))
        for task_index, task in enumerate(tasks):
            for repeat_index in range(repeat_num):
                env.set_task(task)
                # env.seed(seed)
                # env.action_space.seed(seed)
                # env.observation_space.seed(seed)
                obs, _ = env.reset()
                done = False
                count = 0
                ep_reward = 0
                for step in range(traj_length):
                    action = policy.get_action(obs) * (1 + np.random.normal(loc=0, scale=scale))
                    next_obs, reward, _, _, info = env.step(action)

                    # env.render()
                    if int(info["success"]) == 1:
                        done = True

                    else:
                        ep_reward += reward
                        count += 1

                    raw_data[env_name]['obs'][task_index][repeat_index][step] = obs
                    raw_data[env_name]['next_obs'][task_index][repeat_index][step] = next_obs
                    raw_data[env_name]['action'][task_index][repeat_index][step] = action
                    obs = next_obs

                ave_reward = ep_reward / count
                raw_data[env_name]['traj_mask'][task_index][repeat_index][:count] = 1
                raw_data[env_name]['info']['ave_reward'][task_index][repeat_index] = ave_reward
                raw_data[env_name]['info']['done_count'][task_index][repeat_index] = count
                raw_data[env_name]['info']['success'][task_index][repeat_index] = done

        raw_data[env_name]['obs'] = raw_data[env_name]['obs'].reshape(-1, traj_length, obs_dim)
        raw_data[env_name]['next_obs'] = raw_data[env_name]['next_obs'].reshape(-1, traj_length, obs_dim)
        raw_data[env_name]['action'] = raw_data[env_name]['action'].reshape(-1, traj_length, act_dim)
        raw_data[env_name]['traj_mask'] = raw_data[env_name]['traj_mask'].reshape(-1, traj_length)
        raw_data[env_name]['info']['ave_reward'] = raw_data[env_name]['info']['ave_reward'].reshape(-1)
        raw_data[env_name]['info']['done_count'] = raw_data[env_name]['info']['done_count'].reshape(-1)
        raw_data[env_name]['info']['success'] = raw_data[env_name]['info']['success'].reshape(-1)

    return raw_data


if __name__ == "__main__":
    save_path = '../datasets/'
    seed = 45
    traj_length = 100
    repeat_num = 1
    scale = 0
    dataset_type = 'MT10'
    raw_data = generator_mt_dataset(seed=seed, traj_length=traj_length, repeat_num=repeat_num, scale=scale,
                         dataset_type=dataset_type)
    file_name = save_path + dataset_type + "_rn" + str(repeat_num) + '_sc' + str(scale) + '.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(raw_data, file)
    print('Data generating finished! files save in ' + file_name)
