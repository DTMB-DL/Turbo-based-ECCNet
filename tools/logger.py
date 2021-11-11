import time
import os


class Logger:
    def __init__(self, out_name=str(time.time())+'.txt'):
        self.file = out_name

    def save_one_epoch(self, encoder_loss, decoder_loss, epoch_loss_train, BER_train, epoch_loss_val, BER_val):
        if not os.path.exists(self.file):
            with open(self.file, 'w+') as f:
                f.write(str(
                    [encoder_loss, decoder_loss, epoch_loss_train, BER_train.tolist(), epoch_loss_val,
                     BER_val.tolist()]))
                f.write('\n')
        else:
            with open(self.file, 'a+') as f:
                f.write(str(
                    [encoder_loss, decoder_loss, epoch_loss_train, BER_train.tolist(), epoch_loss_val,
                     BER_val.tolist()]))
                f.write('\n')


if __name__ == '__main__':
    logger = Logger()
