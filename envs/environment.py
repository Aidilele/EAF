import gym
import numpy as np


class ParallelEnv(gym.Env):

    def __init__(self, env_name, parallel_num):

        self.env_list = []
        self.parallel_num = parallel_num
        for i in range(parallel_num):
            self.env_list.append(gym.make(env_name))
        self.observation_space = self.env_list[0].observation_space
        self.action_space = self.env_list[0].action_space
        self.obs_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        self.terminal_mask = np.zeros(self.parallel_num, dtype=np.bool8)

    def reset(self, **kwargs):
        results = []
        self.terminal_mask = np.zeros(self.parallel_num, dtype=np.bool8)
        for env in self.env_list:
            result = env.reset(**kwargs)
            results.append(result)
        return np.stack(results), np.zeros(self.parallel_num)

    def sample_random_action(self):
        actions = []
        for env in self.env_list:
            actions.append(env.action_space.sample())
        return np.stack(actions)

    def step(self, action):
        observations = np.zeros((self.parallel_num, self.obs_dim))
        rewards = np.zeros(self.parallel_num)
        dones = np.zeros(self.parallel_num)
        infos = []
        for i in range(len(self.env_list)):
            observation, reward, done, info = self.env_list[i].step(action[i])
            if not done:
                observations[i] = observation
                rewards[i] = reward
            dones[i] = done
            infos.append(infos)

        return (
            observations,
            rewards,
            dones,
            infos
        )

    def render(self, mode='human'):
        results = []
        if mode == 'human':
            self.env_list[0].render(mode)
        else:
            for env in self.env_list:
                results.append(env.render(mode=mode))
        return results
