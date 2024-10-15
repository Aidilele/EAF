from utils.builder import build_diffuser_trainer, build_config, build_condition_model


def eval_diffuser(path):
    config = build_config(path)
    trainer = build_diffuser_trainer(config)
    trainer.load()
    trainer.evaluate(True)


if __name__ == '__main__':
    path = r'C:\Project\EmboddieAgentFramework\runs\2024-10-14-15-17-44'
    eval_diffuser(path)
