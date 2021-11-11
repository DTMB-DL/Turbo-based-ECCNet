import matlab.engine
import numpy as np
import matlab
import math
from traditional import *
from setting.setting import *
from tools.parse import *
from models import *
from data.mydataset import *
import os
import torch


def get_snr(start, step, end):
    total_num = round((end-start)/step+1)
    return np.linspace(start, end, total_num)


def get_modem(num):
    if num == 4:
        return 'QPSK', np.sqrt(2)
    elif num == 16:
        return '16QAM', np.sqrt(10)
    elif num == 64:
        return '64QAM', np.sqrt(42)


def start_matlab():
    eng = matlab.engine.start_matlab()
    return eng


def baseline(args, bers, mode="unquantized"):
    eng = matlab.engine.start_matlab()
    eng.addpath('./traditional')
    SNR = get_snr(args.snr_start, args.snr_step, args.snr_end)  # range of SNR
    modem, base = get_modem(args.modem_num)
    B = bers.shape[0]
    BER = []
    for snr in SNR:
        snrtmp = snr
        snr += 10 * np.log10(np.log2(args.modem_num) / (5*args.G) / 3)
        ber = 0.0
        for idx in range(B):
            s = bers[idx]
            s1 = eng.lteTurboEncode(matlab.int8(s.tolist()))
            s1 = np.array(s1).reshape(-1)

            x = eng.lteSymbolModulate(matlab.int8(s1.tolist()), modem)
            x = x * np.sqrt(1.0)
            xx = torch.tensor([x.real.reshape(-1).tolist(), x.imag.reshape(-1).tolist()]).to(device).reshape(2, 1, -1)
            tx = pulse_shaping(xx, ISI=1, rate=5 * args.G)
            [ry, sigma2] = awgn_channel(tx, snr)
            y = matched_filtering(ry, ISI=1, rate=5 * args.G)

            if mode == 'quantized':
                r = quantize(y, 1, args.modem_num)
                r /= torch.tensor(base)
            elif mode == 'unquantized':
                r = y

            r = np.array(r.cpu())
            z = np.zeros(r.shape[2], dtype='complex')
            z.real = r[0][0]
            z.imag = r[1][0]

            s2 = eng.lteSymbolDemodulate(matlab.double(z.tolist(), is_complex=True), modem, 'Soft')
            s_hat = eng.lteTurboDecode(s2)
            s_hat = np.array(s_hat).reshape(-1)
            ber += np.sum(s != s_hat[:args.len])

        ber /= B*args.len
        BER.append(ber)
        print("snr %f, ber is :%f:" % (snrtmp, ber))

    eng.quit()
    if not os.path.exists('./data'):
        os.mkdir('./data')
    np.savez('./data/'+mode+'_'+str(args.modem_num), snr=SNR, ber=np.array(BER))
    print("%s.npz is saved" % (mode+'_'+str(args.modem_num)))


def cnn_test(args, bers):
    path = 'data/model_cnn_'+str(args.modem_num)
    eng = matlab.engine.start_matlab()
    eng.addpath('./traditional')
    SNR = get_snr(args.snr_start, args.snr_step, args.snr_end)  # range of SNR
    B = bers.shape[0]
    BER = []

    encoder = torch.load(path + '/best_encoder.pth')
    decoder = torch.load(path + '/best_decoder.pth')
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()

    for snr in SNR:
        snrtmp = snr
        snr += 10 * np.log10(np.log2(args.modem_num) / (5 * args.G) / 3)
        ber = 0.0
        with torch.no_grad():
            for idx in range(B):
                s = bers[idx]
                ss = eng.lteTurboEncode(matlab.int8(s.tolist()))
                ss = np.array(ss).reshape(-1)

                maxcnt = math.ceil(args.code_len / args.N)
                yy = torch.zeros(args.code_len, device=device).float()

                for each_block in range(maxcnt):
                    s1 = torch.zeros(1, args.N, device=device).float()
                    s1[0, 0:min(args.N, args.code_len - each_block * args.N)] = torch.from_numpy(
                        ss[each_block * args.N:min(args.N * (each_block + 1), args.code_len)]).to(device)
                    s1 = s1.reshape(1, 1, -1)
                    r = encoder(s1, snr, 1, mode='infer')
                    recover_s = decoder(r)
                    yy[each_block * args.N:min(args.N * (each_block + 1), args.code_len)] = recover_s[0, 0, 0:min(args.N, args.code_len - each_block * args.N)]

                yy = (yy-0.5)*10  # map [0, 1] -> [-x, x] for lteTurboDecode
                s_hat = eng.lteTurboDecode(matlab.double(yy.tolist()))
                s_hat = np.array(s_hat).reshape(-1)
                ber += np.sum(s != s_hat[:args.len])

        ber /= B * args.len
        BER.append(ber)
        print("snr %f, ber is :%f:" % (snrtmp, ber))

    eng.quit()
    if not os.path.exists('./data'):
        os.mkdir('./data')
    np.savez('./data/cnn_' + str(args.modem_num), snr=SNR, ber=np.array(BER))
    print("%s.npz is saved" % ('cnn_' + str(args.modem_num)))


if __name__ == '__main__':
    args = get_args()
    trains, tests, vals, bers = loaddata()
    if args.curve == 'unquantized':
        baseline(args, bers, "unquantized")
    elif args.curve == 'quantized':
        baseline(args, bers, "quantized")
    elif args.curve == 'cnn':
        cnn_test(args, bers)

