# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
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


# import os
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


# def cleanup():
#     """
#     End DDP training.
#     """
#     dist.destroy_process_group()


def create_logger():
    """
    Create a logger that writes to a log file and stdout.
    """
    # if dist.get_rank() == 0:  # real logger
    #     logging.basicConfig(
    #         level=logging.INFO,
    #         format='[\033[34m%(asctime)s\033[0m] %(message)s',
    #         datefmt='%Y-%m-%d %H:%M:%S',
    #         handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    #     )
    #     logger = logging.getLogger(__name__)
    # else:  # dummy logger (does nothing)
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.NullHandler())
    return logger


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
class DiffuserTrainer:
    def __init__(self, config):
        self.config = config
        # Setup DDP:
        # dist.init_process_group("nccl")
        # assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
        # rank = dist.get_rank()
        # device = rank % torch.cuda.device_count()
        # seed = args.global_seed * dist.get_world_size() + rank
        self.seed = config['seed']
        self.device = torch.device(config['train_cfgs']['device'] if torch.cuda.is_available() else 'cpu')
        self.epochs = config['train_cfgs']['total_episode']
        self.save_freq = config['train_cfgs']['save_model_freq']
        self.checkpoint_dir = config['logger_cfgs']['log_dir']
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
        self.logger=config['logger']

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

        logger = create_logger()
        logger.info(f"DiT Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        log_steps = 0
        running_loss = 0
        start_time = time()

        logger.info(f"Training for {self.epochs} epochs...")
        for train_step in range(self.epochs):
            logger.info(f"Beginning epoch {train_step}...")
            x = self.loader.diffuser_training_sample()
            y = self.condition_model(x[:, :, 6:])
            t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],), device=self.device)
            model_kwargs = dict(y=y)
            loss_dict = self.diffusion.training_losses(self.model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.update_ema()

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            if train_step % self.save_freq == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=self.device)
                avg_loss = avg_loss.item()
                logger.info(
                    f"(step={train_step:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_step % self.save_freq == 0 and train_step > 0:
                self.save(train_steps=train_step)
                checkpoint_path = f"{self.checkpoint_dir}/{train_step:05d}.pt"
                # torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                self.evaluate()

        self.model.eval()  # important! This disables randomized embedding dropout
        # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
        logger.info("Done!")

    def evaluate(self):
        self.model.eval()
        env = self.config['environment']
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        parallel_num = env.parallel_num
        horizon = self.config['diffusion_cfgs']['horizon']
        env.reset()
        target = torch.zeros(parallel_num).to(self.device)
        for i in range(parallel_num):
            target[i] = i + 1 / parallel_num
        y = self.condition_model.sample(target)
        model_kwargs = dict(y=y, cfg_scale=self.config['diffusion']['cfg_scale'])
        z = torch.randn(parallel_num, horizon, act_dim + obs_dim, device=self.device)
        samples = self.diffusion.ddim_sample_loop(
            self.model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
            device=self.device
        )
        samples, _ = samples.chunk(2, dim=0)
        actions = samples[:, :, :act_dim]
        actions = torch.einsum('nsa->sna', actions).detach().cpu().numpy()
        ep_reward = np.zeros(parallel_num)
        for step in range(self.config['evaluate_cfgs']['evaluate_steps']):
            next_obs, reward, done, _ = env.step(actions[step])
            ep_reward += reward
        self.model.train()

    def save(self, train_steps=0):
        checkpoint = {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.opt.state_dict(),
            # "args": args
        }
        checkpoint_path = f"{self.checkpoint_dir}/{train_steps:05d}.pt"
        torch.save(checkpoint, checkpoint_path)
