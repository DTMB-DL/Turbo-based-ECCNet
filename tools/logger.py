import time
import os


class Logger:
    def __init__(self, out_name=str(time.time())+'.txt'):
        self.file = out_name

    def save_one_epoch(self, autoencoder_loss, epoch_loss_train, epoch_loss_val):
        if not os.path.exists(self.file):
            with open(self.file, 'w+') as f:
                f.write(str([autoencoder_loss, epoch_loss_train, epoch_loss_val]))
                f.write('\n')
        else:
            with open(self.file, 'a+') as f:
                f.write(str([autoencoder_loss, epoch_loss_train, epoch_loss_val]))
                f.write('\n')


if __name__ == '__main__':
    logger = Logger()
