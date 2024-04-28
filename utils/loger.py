from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging
import os


class Loger:
    def __init__(self, log_path=r'./logs'):
        os.makedirs(log_path, exist_ok=True)
        self.log_path = os.path.join(log_path, '{:%Y_%m_%d-%H_%M_%S}.log'.format(datetime.now()))
        # log to file
        logging.basicConfig(level=logging.INFO, format='%(message)s',
                            handlers=[logging.FileHandler(self.log_path), logging.StreamHandler()])
        self.loger = logging.getLogger(name='good_luck')

    def write(self, message):
        self.loger.info(message)


class TensorboardWriter:
    def __init__(self, summary_path=r'./runs'):
        os.makedirs(summary_path, exist_ok=True)
        self.summary_path = os.path.join(summary_path, '{:%Y_%m_%d-%H_%M_%S}'.format(datetime.now()))
        # tensorboard writer
        self.summaryWriter = SummaryWriter(self.summary_path)

    def write(self, tag_name, tag_dict, step):
        self.summaryWriter.add_scalars(tag_name, tag_dict, step)


if __name__ == '__main__':
    loger = Loger()
    loger.write(message='test')
    summary = TensorboardWriter()
    summary.write('Loss', {'train': 1.2}, 0)
