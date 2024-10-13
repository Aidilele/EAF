from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, config):
        self.bucket = config['logger_cfgs']['log_dir']

        self.writer = SummaryWriter(self.bucket)

    def write(self, label, value, epoch):
        self.writer.add_scalar(label, value, epoch)
