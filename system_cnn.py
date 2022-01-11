import torch
import numpy as np
from torch.utils.data import DataLoader
from models.CNNNet import *
from data.mydataset import *
import os
from setting.setting import device
from tools.logger import Logger
from tools.parse import *
import matlab.engine

def get_modem(num):
    if num == 4:
        return 'QPSK', np.sqrt(2)
    elif num == 16:
        return '16QAM', np.sqrt(10)
    elif num == 64:
        return '64QAM', np.sqrt(42)

def train(args, trains, vals):

    snr = args.snr
    logger = Logger()

    train_dataloader = DataLoader(dataset=Mydataset(trains), batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=Mydataset(vals), batch_size=args.batch_size, shuffle=True)

    encoder = Encoder(G=args.G, K=args.K, modem_num=args.modem_num, channel_mode=args.channel_mode)
    decoder = Decoder(G=args.G, K=args.K, modem_num=args.modem_num)
    encoder.to(device)
    decoder.to(device)

    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=args.lr_encoder)
    scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer_encoder, step_size=args.lr_step, gamma=0.1, last_epoch=-1)
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=args.lr_decoder)
    scheduler_decoder = torch.optim.lr_scheduler.StepLR(optimizer_decoder, step_size=args.lr_step, gamma=0.1, last_epoch=-1)

    encoder.eval()
    decoder.eval()

    best_loss = 99999999

    for epoch in range(args.epoch):
        ac_T = (epoch+1) * args.unit_T
        ''' =============  training the decoder ============='''
        decoder.train()
        loss_decoder = []
        for idx, train_datas in enumerate(train_dataloader):
            s = train_datas.reshape(args.batch_size, 2, -1).float()

            r = encoder(s, snr, ac_T, mode='infer')
            recover_s = decoder(r)

            optimizer_decoder.zero_grad()
            criterion = torch.nn.MSELoss()
            BCE_loss = criterion(recover_s, s)
            BCE_loss.backward()
            optimizer_decoder.step()
            loss_decoder.append(BCE_loss.item())

        print("epoch [%d/%d]: decoder loss is %f " % (epoch + 1, args.epoch, np.mean(np.array(loss_decoder))))
        scheduler_decoder.step()
        decoder.eval()

        ''' =============  training the encoder ============='''
        encoder.train()
        loss_encoder = []
        for idx, train_datas in enumerate(train_dataloader):
            s = train_datas.reshape(args.batch_size, 2, -1).float()

            r = encoder(s, snr, ac_T, mode='train')
            recover_s = decoder(r)

            optimizer_encoder.zero_grad()
            criterion = torch.nn.MSELoss()
            BCE_loss = criterion(recover_s, s)
            BCE_loss.backward()
            optimizer_encoder.step()
            loss_encoder.append(BCE_loss.item())

        print("epoch [%d/%d]: encoder loss is %f " % (epoch + 1, args.epoch, np.mean(np.array(loss_encoder))))
        scheduler_encoder.step()
        encoder.eval()

        ''' =============  testing the whole system (train dataset and val dataset)============='''
        loss_system_val = []
        loss_system_train = []

        with torch.no_grad():
            for idx, train_datas in enumerate(train_dataloader):
                s = train_datas.reshape(args.batch_size, 2, -1).float()

                r = encoder(s, snr, ac_T, mode='infer')
                recover_s = decoder(r)

                criterion = torch.nn.MSELoss()
                BCE_loss = criterion(recover_s, s)
                loss_system_train.append(BCE_loss.item())


            for idx, val_datas in enumerate(val_dataloader):
                s = val_datas.reshape(args.batch_size, 2, -1).float()

                r = encoder(s, snr, ac_T, mode='infer')
                recover_s = decoder(r)

                criterion = torch.nn.MSELoss()
                BCE_loss = criterion(recover_s, s)
                loss_system_val.append(BCE_loss.item())


        print("epoch [%d/%d]: system loss of train is %f || and SER of train is %f " % (
            epoch + 1, args.epoch, np.mean(np.array(loss_system_train)), 0.0))
        print("epoch [%d/%d]: system loss is val %f || and SER of val is %f " % (
            epoch + 1, args.epoch, np.mean(np.array(loss_system_val)), 0.0))

        logger.save_one_epoch(np.mean(np.array(loss_encoder)), np.mean(np.array(loss_decoder)),
                              np.mean(np.array(loss_system_train)), np.array([0]), np.mean(np.array(loss_system_val)),
                              np.array([0]))

        if np.mean(np.array(loss_system_val)) < best_loss:
            best_loss = np.mean(np.array(loss_system_val))
            path = 'data/model_qpsk2_cnn_'+args.channel_mode+'_'+str(args.modem_num)
            if not os.path.exists(path):
                os.mkdir(path)
            torch.save(encoder, path + '/best_encoder.pth')
            torch.save(decoder, path + '/best_decoder.pth')


def test(args, tests):
    test_dataloader = DataLoader(dataset=Mydataset(tests), batch_size=args.batch_size, shuffle=True, num_workers=0)
    path = 'data/model_cnn_'+args.channel_mode+'_'+str(args.modem_num)
    encoder = torch.load(path + '/best_encoder.pth')
    decoder = torch.load(path + '/best_decoder.pth')
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()

    SNR = torch.arange(args.snr_start, args.snr_end, args.snr_step)
    Loss = []
    BER = []

    for idx in range(len(SNR)):
        loss_system_test = []
        snr = SNR[idx]
        with torch.no_grad():
            for idx, test_datas in enumerate(test_dataloader):
                s = test_datas.reshape(args.batch_size, 2, -1).float()

                r = encoder(s, snr, 1, mode='infer')
                recover_s = decoder(r)

                criterion = torch.nn.MSELoss()
                BCE_loss = criterion(recover_s, s)
                loss_system_test.append(BCE_loss.item())

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
    trains, tests, vals, bers = loaddata(path='./data/gen_data.npz')
    if args.mode == 'train':
        train(args, trains, vals)
    elif args.mode == 'test':
        test(args, tests)

