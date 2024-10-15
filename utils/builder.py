import time
import json
import os

from model.dynamic_model import DynamicModel
from utils.read_files import load_yaml
from dataset.dataset_class import TaskDataset
from envs.environment import ParallelEnv
from utils.logger import Logger
from trainer.diffuser_trainer import DiffuserPolicy
from model.condition_model import ConditionModel
from model.diffusion import create_diffusion
from module.dit_module import DiT


def build_config(config_path=None):
    if config_path == None:
        config_path = "../config/config.yaml"
        config = load_yaml(config_path)
        time_info = ''
        for x in list(time.localtime())[:-3]:
            time_info += (str(x) + '-')
        time_info = time_info[:-1]
        bucket = '../runs/' + time_info
        config['logger_cfgs']['log_dir'] = bucket
        config_save = json.dumps(config, indent=4)
        if not os.path.exists(bucket):
            os.makedirs(bucket)
        config_file_name = bucket + '/' + 'config.json'
        with open(config_file_name, "w", encoding='utf-8') as f:  ## 设置'utf-8'编码
            f.write(config_save)
    else:
        config_file_name = config_path + '/' + 'config.json'
        with open(config_file_name, "r", encoding='utf-8') as f:
            config = json.load(f)

    return config


def build_environment(config):
    env_name = config['env_name']
    parallel_num = config['env_parallel_num']
    env = ParallelEnv(env_name=env_name, parallel_num=parallel_num)
    config['environment'] = env
    return env


def build_dataset(config):
    dataset = TaskDataset(config)
    config['dataset'] = dataset
    return dataset


def build_logger(config):
    logger = Logger(config=config)
    config['logger'] = logger
    return logger


def build_condition_model(config):
    build_environment(config)
    build_dataset(config)
    condition_model = ConditionModel(config)
    config['condition_model'] = condition_model

    return condition_model


def build_dynamic_model(config):
    build_environment(config)
    build_dataset(config)
    dynamic_model = DynamicModel(config)
    config['dynamic_model'] = dynamic_model
    return dynamic_model


def build_denoise_module(config):
    env = build_environment(config)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    if config['diffusion_cfgs']['noise_model'] == 'TemporalUnet':
        denoise_model = TemporalUnet(
            horizon=config['algo_cfgs']['horizon'],
            transition_dim=obs_dim,
            dim=config['model_cfgs']['temporalU_model']['dim'],
            dim_mults=config['model_cfgs']['temporalU_model']['dim_mults'],
            returns_condition=config['dataset_cfgs']['include_returns'],
            calc_energy=config['model_cfgs']['temporalU_model']['calc_energy'],
            condition_dropout=config['model_cfgs']['temporalU_model']['condition_dropout'],
        )
    elif config['diffusion_cfgs']['noise_model'] == 'DiT':
        if config['denoise_action']:
            denoise_dim=obs_dim + act_dim
        else:
            denoise_dim=obs_dim
        input_size = (config['diffusion_cfgs']['horizon'], denoise_dim)
        patch_size = (1, denoise_dim)
        denoise_model = DiT(
            input_size=input_size,
            patch_size=patch_size,
            cond_dim=config['model_cfgs']['DiT']['cond_dim'],
            in_channels=1,
            hidden_size=config['model_cfgs']['DiT']['hidden_dim'],
            depth=config['model_cfgs']['DiT']['depth'],
            num_heads=config['model_cfgs']['DiT']['n_heads'],
            mlp_ratio=config['model_cfgs']['DiT']['mlp_ratio'],
            class_dropout_prob=config['model_cfgs']['DiT']['class_dropout'],
            learn_sigma=config['diffusion_cfgs']['learn_sigma'],
        )
    else:
        assert False, 'Unspecified denoise model'
    config['denoise_model'] = denoise_model
    return denoise_model


def build_diffusion(config):
    # env = config['environment']
    # denoise_model = build_denoise_module(config)
    diffusion = create_diffusion(
        timestep_respacing=config['diffusion_cfgs']['timestep_respacing'],
        noise_schedule=config['diffusion_cfgs']['noise_schedule'],
        use_kl=config['diffusion_cfgs']['use_kl'],
        sigma_small=config['diffusion_cfgs']['sigma_small'],
        predict_xstart=config['diffusion_cfgs']['predict_xstart'],
        learn_sigma=config['diffusion_cfgs']['learn_sigma'],
        rescale_learned_sigmas=config['diffusion_cfgs']['rescale_learned_sigmas'],
        diffusion_steps=config['diffusion_cfgs']['diffusion_steps']
    )
    config['diffusion'] = diffusion
    return diffusion


def build_diffuser_trainer(config):
    build_dataset(config)
    build_diffusion(config)
    build_denoise_module(config)
    build_logger(config)
    build_condition_model(config)
    if not config['denoise_action']:
        build_dynamic_model(config)
    trainer = DiffuserPolicy(
        config=config,
    )
    config['diffuser_trainer'] = trainer
    return trainer


if __name__ == '__main__':
    config = build_config()
    model = build_condition_model(config)
    try:
        model.train()
    except:
        model.save(0)
        assert False, 'Exception! Latest model state dict saved!'

    print('ok')
