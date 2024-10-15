# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
from distutils.core import setup_keywords

import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from time import time
import logging
import os
import cv2


# from module.models import DiT_models
# from model.diffusion import create_diffusion

#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################
class DiffuserPolicy:
    def __init__(self, config):
        self.config = config
        self.seed = config['seed']
        self.device = torch.device(config['train_cfgs']['device'] if torch.cuda.is_available() else 'cpu')
        self.epochs = config['train_cfgs']['total_episode']
        self.save_freq = config['train_cfgs']['save_model_freq']
        self.bucket = config['logger_cfgs']['log_dir']
        self.condition_model = config['condition_model'].to(self.device)
        self.condition_model.eval()
        self.model = config['denoise_model'].to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0)
        ema = deepcopy(self.model).to(self.device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        self.ema = ema
        self.loader = config['dataset']
        self.update_ema(decay=0)  # Ensure EMA is initialized with synced weights
        self.model.train()  # important! This enables embedding dropout for classifier-free guidance
        self.ema.eval()
        self.diffusion = config['diffusion']
        self.logger = config['logger']
        self.load()
        self.loader.generate_traj_emb()

    @torch.no_grad()
    def update_ema(self, decay=0.9999):
        """
        Step the EMA model towards the current model.
        """
        ema_params = OrderedDict(self.ema.named_parameters())
        model_params = OrderedDict(self.model.named_parameters())

        for name, param in model_params.items():
            # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    def train(self):
        torch.manual_seed(self.seed)
        torch.cuda.set_device(self.device)

        # logger = create_logger()
        # logger.info(f"DiT Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        log_steps = 0
        running_loss = 0
        start_time = time()

        # logger.info(f"Training for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            # logger.info(f"Beginning epoch {train_step}...")
            batch_samples = self.loader.diffuser_training_sample()
            x = batch_samples['traj_data']
            x = x.unsqueeze(1)
            y = batch_samples['traj_emb']
            obs = batch_samples['init_obs']
            t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],), device=self.device)
            model_kwargs = dict(y=y, obs=obs)
            loss_dict = self.diffusion.training_losses(self.model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            self.logger.write('loss/diffuser', loss, epoch)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.update_ema()

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            if epoch % self.save_freq == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=self.device)
                avg_loss = avg_loss.item()
                # logger.info(
                #     f"(step={train_step:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if epoch % self.save_freq == 0 and epoch > 0:
                self.save(train_steps=epoch)
                # checkpoint_path = f"{checkpoint_dir}/{train_step:05d}.pt"
                # torch.save(checkpoint, checkpoint_path)
                # logger.info(f"Saved checkpoint to {checkpoint_path}")
                ep_reward = self.evaluate()
                mean = ep_reward.mean()
                std = ep_reward.std()
                print(f'mean: {mean:.2f}, std: {std:.2f}, max: {ep_reward.max():.2f} ,min: {ep_reward.min():.2f}')
                self.logger.write('ave_reward', ep_reward.mean(), epoch)
                self.logger.write('std_reward', ep_reward.std(), epoch)

        self.model.eval()  # important! This disables randomized embedding dropout
        # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
        # logger.info("Done!")

    def act(self, obs):
        obs = torch.from_numpy(obs).to(torch.float32).to(self.device)
        obs = self.loader.normalizer.normalize(obs)
        obs_dim = self.config['environment'].observation_space.shape[0]
        act_dim = self.config['environment'].action_space.shape[0]
        if self.config['denoise_action']:
            denoise_dim = act_dim + obs_dim
        else:
            denoise_dim = obs_dim
        parallel_num = self.config['environment'].parallel_num
        horizon = self.config['diffusion_cfgs']['horizon']
        target = 0.95 * torch.ones((parallel_num, 1)).to(self.device)
        # for i in range(parallel_num):
        #     target[i] = i + 1 / parallel_num
        y = self.condition_model.sample(target)
        z = torch.randn(parallel_num, horizon, denoise_dim, device=self.device)
        z = z.unsqueeze(1)
        model_kwargs = dict(y=y, obs=obs, cfg_scale=self.config['diffusion_cfgs']['cfg_scale'])
        samples = self.diffusion.ddim_sample_loop(
            self.model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
            device=self.device
        )
        samples = samples.squeeze(1)
        if self.config['denoise_action']:
            actions = samples[:, :self.config['evaluate_cfgs']['multi_step_pred'], obs_dim:]
        else:
            comb_obs = torch.concat((samples[:, :-1, :], samples[:, 1:, :]), dim=-1)
            actions = self.dynamic_model(comb_obs)
        actions = self.loader.act_normalizer.unnormalize(actions)
        actions = torch.einsum('nsa->sna', actions).detach().cpu().numpy()
        return actions

    def evaluate(self, render=False):
        self.model.eval()
        env = self.config['environment']
        parallel_num = env.parallel_num
        ep_reward = np.zeros(parallel_num)
        obs, terminal = env.reset()
        step = 0
        while (step < self.config['evaluate_cfgs']['evaluate_steps']) and (not terminal.all()):
            actions = self.act(obs)
            for action in actions:
                next_obs, reward, terminal, _ = env.step(action)
                ep_reward += reward
                if render:
                    env.render()
                step += 1
                if terminal.all() or step >= self.config['evaluate_cfgs']['evaluate_steps']:
                    break
                obs = next_obs
        if render:
            self.render_frames(0, ep_reward)
        self.model.train()
        return ep_reward

    def render_frames(self, episode, ep_reward):
        frames = self.config['environment'].frames_cache
        video_path = os.path.join(self.bucket, 'video')
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        height = frames[0][0].shape[0]
        width = frames[0][0].shape[1]
        for i, frame in enumerate(frames):
            video_name = video_path + '/E' + str(episode) + '_P' + str(i) + '_R' + str(int(ep_reward[i])) + '.mp4'
            out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 120, (height, width))
            for single_frame in frame:
                out.write(single_frame)
            out.release()
        print('ok')

    def save(self, train_steps=0):
        checkpoint = {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.opt.state_dict(),
            # "args": args
        }
        checkpoint_dir = os.path.join(self.bucket, 'pretrain')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = f"{checkpoint_dir}/{train_steps:05d}.pt"
        torch.save(checkpoint, checkpoint_path)

    def load(self):
        train_steps = self.config['evaluate_cfgs']['denoise_model_index']
        checkpoint_dir = os.path.join(self.bucket, 'pretrain')
        checkpoint_path = f"{checkpoint_dir}/{train_steps:05d}.pt"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model'])
            self.ema.load_state_dict(checkpoint['ema'])
            self.opt.load_state_dict(checkpoint['opt'])
        self.condition_model.load(self.config['evaluate_cfgs']['condition_model_index'])
        if not self.config['denoise_action']:
            self.dynamic_model = self.config['dynamic_model'].to(self.device)
            self.dynamic_model.load(self.config['evaluate_cfgs']['dynamic_model_index'])
            self.dynamic_model.eval()
