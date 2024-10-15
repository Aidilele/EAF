import gym
import numpy as np
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)

class ParallelEnv(gym.Env):

    def __init__(self, env_name, parallel_num):

        self.env_list = []
        self.parallel_num = parallel_num
        for i in range(parallel_num):
            env = gym.make(env_name)
            self.env_list.append(env)
        self.observation_space = self.env_list[0].observation_space
        self.action_space = self.env_list[0].action_space
        self.obs_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        self.terminal_mask = np.zeros(self.parallel_num, dtype=np.bool8)
        self.frames_cache = []

    def reset(self, **kwargs):
        results = []

        self.terminal_mask = np.zeros(self.parallel_num, dtype=np.bool8)
        for env in self.env_list:
            result = env.reset(**kwargs)
            results.append(result)
            self.frames_cache.append([])
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
        for i, env in enumerate(self.env_list):
            observation, reward, done, info = env.step(action[i])
            if not done:

                rewards[i] = reward
            else:
                self.terminal_mask[i] = True
            observations[i] = observation
            dones[i] = done
            infos.append(infos)

        return (
            observations,
            rewards,
            dones,
            infos
        )

    def render(self,mode='rgb_array'):
        assert mode == 'rgb_array', r'Only support "rgb_array" render mode'
        for i, env in enumerate(self.env_list):
            if not self.terminal_mask[i]:
                self.frames_cache[i].append(env.render(mode=mode))

        # if mode == 'human':
        #     for env in self.env_list:
        #         env.render(mode)
        # elif mode == 'rgb_array':
        #     for i, env in enumerate(self.env_list):
        #         self.frames_cache[i].append(env.render(mode=mode))
        # else:
        #     raise NotImplementedError('Unsupported render mode')