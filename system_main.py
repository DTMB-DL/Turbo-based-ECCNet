import matlab.engine
import numpy as np
import matlab
import math
from traditional import *
from setting.setting import *
from tools.parse import *
from tools.utils import *
from models import *
import os
import torch


def baseline(args, bers, mode="unquantized"):
    eng = matlab.engine.start_matlab()
    eng.addpath('./traditional')
    SNR = get_snr(args.snr_start, args.snr_step, args.snr_end)  # range of SNR
    modem, base = get_modem(args.modem_num)
    B = bers.shape[0]
    BER = []
    for snr in SNR:
        snrtmp = snr

        ber = 0.0
        for idx in range(B):
            s = bers[idx]
            ss = eng.lteTurboEncode(matlab.int8(s.tolist()))
            x = eng.lteSymbolModulate(ss, modem)
            x = torch.tensor(x).reshape(-1)
            xx = torch.stack((x.real, x.imag), axis=0).reshape(2, 1, -1).to(device)
            tx = pulse_shaping(xx, ISI=1, rate=5 * args.G)
            [ry, sigma2] = channel(args.channel_mode, tx,
                                   snr + 10 * np.log10(np.log2(args.modem_num) / (tx.shape[2] / xx.shape[2]) / 3),
                                   rate=5 * args.G)
            y = matched_filtering(ry, ISI=1, rate=5 * args.G)

            if mode == 'quantized':
                r = quantize(y, 1)
                r /= torch.tensor(base)
            elif mode == 'unquantized':
                r = y

            z = r[0][0] + 1j * r[1][0]
            s2 = eng.lteSymbolDemodulate(matlab.double(z.tolist(), is_complex=True), modem, 'Soft')
            s_hat = eng.lteTurboDecode(s2)
            s_hat = np.array(s_hat).reshape(-1)
            ber += np.sum(s != s_hat[:args.len])

        ber /= B * args.len
        BER.append(ber)
        print("snr %f, ber is :%f:" % (snrtmp, ber))

    eng.quit()
    if not os.path.exists('./data'):
        os.mkdir('./data')
    np.savez('./data/' + mode + '_' + args.channel_mode + '_' + str(args.modem_num), snr=SNR, ber=np.array(BER))
    print("%s.npz is saved" % (mode + '_' + args.channel_mode + '_' + str(args.modem_num)))


def cnn_test(args, bers):
    path = 'data/model_cnn_'+args.channel_mode+'_'+str(args.modem_num)
    eng = matlab.engine.start_matlab()
    eng.addpath('./traditional')
    SNR = get_snr(args.snr_start, args.snr_step, args.snr_end)  # range of SNR
    modem, base = get_modem(args.modem_num)
    B = bers.shape[0]
    N = args.N
    BER = []
    autoencoder = torch.load(path + '/best_autoencoder.pth')
    autoencoder.to(device)
    autoencoder.eval()

    for snr in SNR:
        snrtmp = snr
        ber = 0.0
        with torch.no_grad():
            for idx in range(B):
                s = bers[idx]
                ss = eng.lteTurboEncode(matlab.int8(s.tolist()))
                x = eng.lteSymbolModulate(ss, modem)
                x = torch.tensor(x).reshape(-1)
                x = torch.stack((x.real, x.imag), axis=0).reshape(2, -1).to(device)
                L = x.shape[1]
                maxcnt = math.ceil(L / N) * N

                s1 = torch.cat((x, x[:, :maxcnt-L]), dim=1)
                s2 = torch.stack((s1[0].reshape(-1, N), s1[1].reshape(-1, N)), dim=1)

                r = autoencoder(s2, snr, 1, mode='infer')

                z = r[:, 0, :].reshape(-1)[:L] + 1j * r[:, 1, :].reshape(-1)[:L]
                yy = eng.lteSymbolDemodulate(matlab.double(z.tolist(), is_complex=True), modem, 'Soft')
                s_hat = eng.lteTurboDecode(yy)
                s_hat = np.array(s_hat).reshape(-1)
                ber += np.sum(s != s_hat[:args.len])

        ber /= B * args.len
        BER.append(ber)
        print("snr %f, ber is :%f:" % (snrtmp, ber))

    eng.quit()
    if not os.path.exists('./data'):
        os.mkdir('./data')
    np.savez('./data/cnn_' + args.channel_mode+'_' + str(args.modem_num), snr=SNR, ber=np.array(BER))
    print("%s.npz is saved" % ('cnn_' + args.channel_mode+'_' + str(args.modem_num)))


if __name__ == '__main__':
    args = get_args()
    bers = np.random.randint(0, 2, [args.ber_len, args.len])
    if args.curve == 'unquantized':
       baseline(args, bers, "unquantized")
    elif args.curve == 'quantized':
       baseline(args, bers, "quantized")
    elif args.curve == 'cnn':
       cnn_test(args, bers)

