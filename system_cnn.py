import torch
import numpy as np
from torch.utils.data import DataLoader
from models.CNNNet import *
from data.mydataset import *
import os
from setting.setting import device
from tools.logger import Logger
from tools.parse import *


def train(args, trains, vals):

    snr = args.snr + 10 * np.log10(np.log2(args.modem_num) / (5*args.G) / 3)
    logger = Logger()

    train_dataloader = DataLoader(dataset=Mydataset(trains), batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=Mydataset(vals), batch_size=args.batch_size, shuffle=True)

    encoder = Encoder(G=args.G, K=args.K, modem_num=args.modem_num)
    decoder = Decoder(G=args.G, K=args.K, modem_num=args.modem_num)
    encoder.to(device)
    decoder.to(device)

    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=args.lr_encoder)
    scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer_encoder, step_size=args.lr_step, gamma=0.1, last_epoch=-1)
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=args.lr_decoder)
    scheduler_decoder = torch.optim.lr_scheduler.StepLR(optimizer_decoder, step_size=args.lr_step, gamma=0.1, last_epoch=-1)

    encoder.eval()
    decoder.eval()

    best_ser = 99999999

    for epoch in range(args.epoch):
        ac_T = (epoch+1) * args.unit_T
        ''' =============  training the decoder ============='''
        decoder.train()
        loss_decoder = []
        for idx, train_datas in enumerate(train_dataloader):
            s = train_datas.reshape(args.batch_size, 1, -1).float()
            r = encoder(s, snr, ac_T, mode='infer')
            recover_s = decoder(r)

            optimizer_decoder.zero_grad()
            criterion = torch.nn.BCEWithLogitsLoss()
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
            s = train_datas.reshape(args.batch_size, 1, -1).float()
            r = encoder(s, snr, ac_T, mode='train')
            recover_s = decoder(r)

            optimizer_encoder.zero_grad()
            criterion = torch.nn.BCEWithLogitsLoss()
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
        SER_val = 0.0
        SER_train = 0.0

        with torch.no_grad():
            for idx, train_datas in enumerate(train_dataloader):
                s = train_datas.reshape(args.batch_size, 1, -1).float()
                r = encoder(s, snr, ac_T, mode='infer')
                recover_s = decoder(r)

                criterion = torch.nn.BCEWithLogitsLoss()
                BCE_loss = criterion(recover_s, s)
                loss_system_train.append(BCE_loss.item())
                SER_train += torch.sum(recover_s.round().int() != s.int()).float()

            for idx, val_datas in enumerate(val_dataloader):
                s = val_datas.reshape(args.batch_size, 1, -1).float()
                r = encoder(s, snr, ac_T, mode='infer')
                recover_s = decoder(r)

                criterion = torch.nn.BCEWithLogitsLoss()
                BCE_loss = criterion(recover_s, s)
                loss_system_val.append(BCE_loss.item())
                SER_val += torch.sum(recover_s.round().int() != s.int()).float()

        SER_train /= args.train_cut * args.N
        SER_val /= args.val_cut * args.N
        print("epoch [%d/%d]: system loss of train is %f || and SER of train is %f " % (
            epoch + 1, args.epoch, np.mean(np.array(loss_system_train)), SER_train))
        print("epoch [%d/%d]: system loss is val %f || and SER of val is %f " % (
            epoch + 1, args.epoch, np.mean(np.array(loss_system_val)), SER_val))

        logger.save_one_epoch(np.mean(np.array(loss_encoder)), np.mean(np.array(loss_decoder)),
                              np.mean(np.array(loss_system_train)), SER_train, np.mean(np.array(loss_system_val)),
                              SER_val)

        if SER_val < best_ser:
            best_ser = SER_val
            if not os.path.exists('data/model_cnn_'+str(args.modem_num)):
                os.mkdir('data/model_cnn_'+str(args.modem_num))
            torch.save(encoder, 'data/model_cnn_' + str(args.modem_num) + '/best_encoder.pth')
            torch.save(decoder, 'data/model_cnn_' + str(args.modem_num) + '/best_decoder.pth')


def test(args, tests):

    test_dataloader = DataLoader(dataset=Mydataset(tests), batch_size=args.batch_size, shuffle=True, num_workers=0)
    path = 'data/model_cnn_'+str(args.modem_num)
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
        BER_test = 0.0
        snr = SNR[idx] + 10 * np.log10(np.log2(args.modem_num) / (5*args.G) / 3)
        with torch.no_grad():
            for idx, test_datas in enumerate(test_dataloader):

                s = test_datas.reshape(args.batch_size, 1, -1).float()
                r = encoder(s, snr, 1, mode='infer')
                recover_s = decoder(r)

                criterion = torch.nn.BCEWithLogitsLoss()
                BCE_loss = criterion(recover_s, s)
                loss_system_test.append(BCE_loss.item())
                BER_test += torch.sum(recover_s.round().int() != s.int()).float()

        loss_system_test = np.mean(np.array(loss_system_test))
        BER_test /= args.test_cut * args.N
        Loss.append(loss_system_test)
        BER.append(BER_test)

    print("loss of awgn channel is:", Loss)
    print("BER of awgn channel is:", BER)

    import matplotlib.pyplot as plt
    plt.figure(0)
    plt.title('loss-SNR')
    plt.plot(SNR, Loss, label='awgn')
    plt.legend(loc='best')
    plt.savefig("loss_SNR.jpg")

    plt.figure(1)
    plt.title('BER-SNR')
    plt.plot(SNR, BER, label='awgn')
    plt.legend(loc='best')

    plt.savefig("BER_SNR_CNN_"+str(args.modem_num)+".jpg")
    plt.show()


if __name__ == "__main__":
    args = get_args()
    trains, tests, vals, bers = loaddata()
    if args.mode == 'train':
        train(args, trains, vals)
    elif args.mode == 'test':
        test(args, tests)

