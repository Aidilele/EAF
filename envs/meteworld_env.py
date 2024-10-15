import gym
import numpy as np
import metaworld
import random

class MetaworldEnv(gym.Env):

    def __init__(self, env_name):
        self.env_list = []
        ml = eval("metaworld." + env_name)()
        env_list = list(ml.train_classes.keys())
        self.parallel_num = len(env_list)
        for env_index, env_name in enumerate(env_list):
            env = ml.train_classes[env_name](render_mode='rgb_array')
            tasks = [t for t in ml.train_tasks if t.env_name == env_name]
            env.__setattr__('train_tasks', tasks)
            self.env_list.append(env)

        self.observation_space = self.env_list[0].observation_space
        self.action_space = self.env_list[0].action_space
        self.terminal_mask = np.zeros(self.parallel_num, dtype=np.int8)
        self.frames_cache = []

    def reset(self, **kwargs):
        results = []
        self.terminal_mask = np.zeros(self.parallel_num, dtype=np.int8)
        for env in self.env_list:
            task = random.choice(env.train_tasks)
            env.set_task(task)
            obs, info = env.reset(**kwargs)
            results.append(obs)
            self.frames_cache.append([])
        return np.stack(results), np.zeros(self.parallel_num)

    def sample_random_action(self):
        actions = []
        for env in self.env_list:
            actions.append(env.action_space.sample())
        return np.stack(actions)

    def step(self, action):
        observations = np.zeros((self.parallel_num, self.observation_space.shape[0]))
        rewards = np.zeros(self.parallel_num)
        dones = np.zeros(self.parallel_num)
        infos = []
        for i, env in enumerate(self.env_list):
            if not self.terminal_mask[i]:
                observation, reward, done, truncated, info = env.step(action[i])
                observations[i] = observation
                dones[i] = done
                rewards[i] = reward
                infos.append(infos)
                self.terminal_mask[i] = done

        return (
            observations,
            rewards,
            dones,
            infos
        )

    def render(self, mode='rgb_array'):
        assert mode == 'rgb_array', r'Only support "rgb_array" render mode'
        for i, env in enumerate(self.env_list):
            if not self.terminal_mask[i]:
                self.frames_cache[i].append(env.render())


if __name__ == '__main__':
    env = MetaworldEnv('MT10')
    env.reset()
    for i in range(100):
        a = env.sample_random_action()
        env.step(a)
        env.render()
    print('ok')