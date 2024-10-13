import copy
import torch
import os
import numpy as np


class EMA():
    '''
        empirical moving average
    '''

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class DiffuserTrainer(object):

    def __init__(self,
                 config,
                 env,
                 diffuser_model,
                 dataset,
                 logger,
                 total_steps,
                 ema_decay=0.995,
                 train_lr=2e-5,
                 gradient_accumulate_every=2,
                 step_start_ema=2000,
                 update_ema_every=10,
                 # log_freq=100,
                 sample_freq=1000,
                 save_freq=1000,
                 train_device='cuda',
                 bucket=None,
                 save_checkpoints=True,
                 episode_max_length=1000
                 ):
        super().__init__()
        self.config = config
        self.model = diffuser_model
        self.condition_model = config['condition_model']
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.logger = logger
        self.total_steps = total_steps
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.gradient_accumulate_every = gradient_accumulate_every
        self.dataset = dataset
        self.optimizer = torch.optim.Adam(diffuser_model.parameters(), lr=float(train_lr))
        self.device = torch.device(train_device)
        self.model.to(self.device)
        self.ema_model.to(self.device)
        self.ema_model.eval()
        self.bucket = bucket
        self.save_checkpoints = save_checkpoints
        self.step = 0
        path = os.path.join(self.bucket, f'checkpoint')
        if os.path.exists(path):
            model_files = os.listdir(path)
            model_file_name = path + '/' + model_files[-1]
            data = torch.load(model_file_name)
            self.model.load_state_dict(data['model'])
            self.ema_model.load_state_dict(data['model'])
        self.reset_parameters()
        self.obs_history_length = config['algo_cfgs']['obs_history_length']
        self.multi_step_pred = config['evaluate_cfgs']['multi_step_pred']
        self.episode_max_length = episode_max_length

        self.env = env

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def train(self):

        for step in range(self.total_steps):
            self.optimizer.zero_grad()
            act_dim = self.config['environment'].action_space.shape[0]
            # for _ in range(self.gradient_accumulate_every):
            batch_sample = self.dataset.diffuser_training_sample()
            condition = self.condition_model(batch_sample[:, :, act_dim:]).detach()
            loss, info = self.model.loss(batch_sample, condition)
            # loss = loss / self.gradient_accumulate_every
            loss.backward()
            self.optimizer.step()
            self.logger.write('loss/diffuser', info['loss_diffuser'], step)
            self.logger.write('loss/inv_model', info['loss_inv'], step)

            if self.step % self.update_ema_every == 0:
                self.step_ema()
            if step % self.save_freq == 0:
                self.save()
                ave_reward, std_reward = self.eval()
                self.logger.write('ave_reward', ave_reward, step)
                self.logger.write('std_reward', std_reward, step)
            self.step += 1

    def obs_history_queue(self, obs, obs_queue: list):
        obs_queue.append(obs)
        if len(obs_queue) > self.obs_history_length:
            obs_queue.__delitem__(0)
        return np.stack(obs_queue, axis=-2).astype(np.float32)

    def eval(self):
        self.model.eval()

        condition = torch.ones((self.env.parallel_num, 1), device=self.device)
        condition = self.condition_model.sample(condition)
        obs_history = []
        obs, terminal = self.env.reset()
        obs = self.obs_history_queue(obs, obs_history)
        ep_reward = 0
        obs = torch.from_numpy(obs).to(self.device)
        obs = self.dataset.normalizer.normalize(obs)
        step = 0
        while (step < self.episode_max_length) and (not terminal.all()):
            x = self.model.conditional_sample(obs, condition=condition)
            pred_queue = x[:, self.obs_history_length - 1:]
            for pred_step in range(self.multi_step_pred):
                next_obs = pred_queue[:, pred_step + 1]
                obs_comb = torch.cat([obs[:, -1, :], next_obs], dim=-1)
                pred_action = self.model.inv_model(obs_comb)
                action = pred_action.detach().cpu().numpy()
                next_obs, reward, terminal, _ = self.env.step(action)
                next_obs = self.obs_history_queue(next_obs, obs_history)
                ep_reward += reward
                step += 1
                if terminal.all() or step >= self.episode_max_length:
                    break
                obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
                obs = self.dataset.normalizer.normalize(obs)

        ave_reward = torch.tensor(ep_reward).mean()
        std_reward = torch.tensor(ep_reward).std()
        self.model.train()
        return ave_reward, std_reward

    def save(self):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            # 'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.bucket, 'checkpoint')
        os.makedirs(savepath, exist_ok=True)
        # logger.save_torch(data, savepath)
        if self.save_checkpoints:
            savepath = os.path.join(savepath, f'state_{self.step}.pt')
        else:
            savepath = os.path.join(savepath, 'state.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}')

    def load(self, step=None):
        if self.save_checkpoints and step != None:
            loadpath = os.path.join(self.bucket, f'checkpoint/state_{step}.pt')
        else:
            loadpath = os.path.join(self.bucket, f'checkpoint/state.pt')
        data = torch.load(loadpath)
        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['model'])
