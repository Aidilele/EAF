from utils.builder import build_diffuser_trainer, build_config, build_condition_model


def train_diffuser():
    config = build_config()
    trainer = build_diffuser_trainer(config)
    # try:
    trainer.train()
    # except:
    #     trainer.save()
    #     assert False, 'Exception! Latest model state dict saved!'

def train_condition_model():
    config = build_config()
    trainer = build_condition_model(config)
    try:
        trainer.train()
    except:
        trainer.save()
        assert False, 'Exception! Latest model state dict saved!'

if __name__ == '__main__':
    train_diffuser()
    # train_condition_model()
