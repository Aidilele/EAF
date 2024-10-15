from numpy.distutils.command.config import config

from utils.builder import build_diffuser_trainer, build_config, build_condition_model, build_dynamic_model


def train_diffuser(config_path=None):
    config = build_config(config_path)
    trainer = build_diffuser_trainer(config)
    try:
        trainer.train()
    except:
        trainer.save()
        assert False, 'Exception! Latest model state dict saved!'


def train_condition_model():
    config = build_config()
    trainer = build_condition_model(config)
    try:
        trainer.train_model()
    except:
        trainer.save()
        assert False, 'Exception! Latest model state dict saved!'


def train_dynamic_model():
    config = build_config()
    trainer = build_dynamic_model(config)
    config['dataset'].generate_comb_obs()
    # try:
    trainer.train_model()
    # except:
    #     trainer.save()
    #     assert False, 'Exception! Latest model state dict saved!'


if __name__ == '__main__':
    # config_path = r'C:\Project\EmboddieAgentFramework\runs\2024-10-15-0-0-54'
    config_path = None
    train_diffuser(config_path)
    # train_condition_model()
    # train_dynamic_model()
