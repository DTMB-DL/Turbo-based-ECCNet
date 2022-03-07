import torch
import numpy as np
from torch.utils.data import DataLoader
from models.ECCNet import *
from data.mydataset import *
import os
from setting.setting import device
from tools.logger import Logger
from tools.parse import *


def train(args, trains, vals):
    snr = args.snr
    logger = Logger()

    train_dataloader = DataLoader(dataset=Mydataset(trains), batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=Mydataset(vals), batch_size=args.batch_size, shuffle=True)

    autoencoder = AutoEncoder(G=args.G, N=args.N, qua_bits=1, modem_num=args.modem_num, channel_mode=args.channel_mode)
    autoencoder.to(device)

    optimizer_autoencoder = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
    scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(optimizer_autoencoder, step_size=args.lr_step, gamma=0.1, last_epoch=-1)

    autoencoder.eval()

    best_loss = 99999999

    for epoch in range(args.epoch):
        ac_T = (epoch+1) * args.unit_T
        ''' =============  training the decoder ============='''
        autoencoder.train()
        loss_autoencoder = []
        for idx, train_datas in enumerate(train_dataloader):
            s = train_datas.reshape(args.batch_size, 2, -1).float()

            r = autoencoder(s, snr, ac_T, mode='train')

            optimizer_autoencoder.zero_grad()
            criterion = torch.nn.MSELoss()
            MSE_loss = criterion(r, s)
            MSE_loss.backward()
            optimizer_autoencoder.step()
            loss_autoencoder.append(MSE_loss.item())

        print("epoch [%d/%d]: autoencoder loss is %f " % (epoch + 1, args.epoch, np.mean(np.array(loss_autoencoder))))
        scheduler_autoencoder.step()
        autoencoder.eval()

        ''' =============  testing the whole system (train dataset and val dataset)============='''
        loss_system_val = []
        loss_system_train = []

        with torch.no_grad():
            for idx, train_datas in enumerate(train_dataloader):
                s = train_datas.reshape(args.batch_size, 2, -1).float()
                r = autoencoder(s, snr, ac_T, mode='infer')
                criterion = torch.nn.MSELoss()
                MSE_loss = criterion(r, s)
                loss_system_train.append(MSE_loss.item())

            for idx, val_datas in enumerate(val_dataloader):
                s = val_datas.reshape(args.batch_size, 2, -1).float()
                r = autoencoder(s, snr, ac_T, mode='infer')
                criterion = torch.nn.MSELoss()
                MSE_loss = criterion(r, s)
                loss_system_val.append(MSE_loss.item())

        print("epoch [%d/%d]: system loss of train is %f" % (
            epoch + 1, args.epoch, np.mean(np.array(loss_system_train))))
        print("epoch [%d/%d]: system loss is val %f" % (
            epoch + 1, args.epoch, np.mean(np.array(loss_system_val))))

        logger.save_one_epoch(np.mean(np.array(loss_autoencoder)), np.mean(np.array(loss_system_train)),
                              np.mean(np.array(loss_system_val)))

        if np.mean(np.array(loss_system_val)) < best_loss:
            best_loss = np.mean(np.array(loss_system_val))
            path = 'data/model_cnn_'+args.channel_mode+'_'+str(args.modem_num)
            if not os.path.exists(path):
                os.mkdir(path)
            torch.save(autoencoder.state_dict(), path + '/best_autoencoder.pth')


def test(args, tests):
    test_dataloader = DataLoader(dataset=Mydataset(tests), batch_size=args.batch_size, shuffle=True, num_workers=0)
    path = args.model_path
    autoencoder = AutoEncoder(G=args.G, N=args.N, qua_bits=1, modem_num=args.modem_num, channel_mode=args.channel_mode)
    autoencoder.load_state_dict(torch.load(path + '/best_autoencoder.pth'))
    autoencoder.to(device)
    autoencoder.eval()

    SNR = torch.arange(args.snr_start, args.snr_end, args.snr_step)
    Loss = []
    BER = []

    for idx in range(len(SNR)):
        loss_system_test = []
        snr = SNR[idx]
        with torch.no_grad():
            for idx, test_datas in enumerate(test_dataloader):
                s = test_datas.reshape(args.batch_size, 2, -1).float()
                r = autoencoder(s, snr, 1, mode='infer')
                criterion = torch.nn.MSELoss()
                MSE_loss = criterion(r, s)
                loss_system_test.append(MSE_loss.item())

        loss_system_test = np.mean(np.array(loss_system_test))
        Loss.append(loss_system_test)

    print("loss of awgn channel is:", Loss)
    print("BER of awgn channel is:", BER)

    import matplotlib.pyplot as plt
    plt.figure(0)
    plt.title('loss-SNR')
    plt.plot(SNR, Loss, label='awgn')
    plt.legend(loc='best')
    plt.savefig("loss_SNR.jpg")

    plt.savefig("BER_SNR_CNN_"+args.channel_mode+'_'+str(args.modem_num)+".jpg")
    plt.show()


if __name__ == "__main__":
    args = get_args()
    trains, tests, vals = loaddata(path='./data/gen_data_' + str(args.modem_num) + '.npz')
    if args.mode == 'train':
        train(args, trains, vals)
    elif args.mode == 'test':
        test(args, tests)

