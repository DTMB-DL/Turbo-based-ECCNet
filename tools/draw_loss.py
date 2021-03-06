import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    file_name = 'log.txt'
    p = open('../'+file_name, 'r')
    old_data = p.readlines()
    p.close()
    print(old_data)
    data = [x[1:-2].split(',') for x in old_data]
    data = [[float(y) for y in x] for x in data]
    data = np.array(data)

    autoencoder_loss = data[:, 0]
    epoch_loss_train = data[:, 1]
    epoch_loss_val = data[:, 2]

    epoch = np.arange(0, 100)
    plt.figure(0)
    plt.title('encoder loss')
    plt.plot(epoch, autoencoder_loss)

    plt.figure(1)
    plt.title('epoch loss')
    plt.plot(epoch, epoch_loss_train, label='train')
    plt.plot(epoch, epoch_loss_val, label='val')
    plt.legend(loc='best')

    plt.show()


