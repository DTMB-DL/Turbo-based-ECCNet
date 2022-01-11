import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-ber_dir', type=str, default='../data',
                        help='dir of saved BER results(.npy). no "/" in final position.')
    parser.add_argument('-modem_num', type=int, default=16, help='number of modulation order')
    parser.add_argument('-channel_mode', choices=['awgn', 'rayleigh'], default='awgn', help='for channel')
    args = parser.parse_args()

    unquantized = np.load(args.ber_dir + '/unquantized_' + args.channel_mode + '_' + str(args.modem_num) + '.npz')
    L = (unquantized["ber"] == 0.).argmax(axis=0)
    if L == 0:
        L = 16
    snr = unquantized["snr"][0:L]
    ber = unquantized["ber"][0:L]
    plt.plot(snr, ber, label="Unquantized turbo coding")

    quantized = np.load(args.ber_dir + '/quantized_' + args.channel_mode + '_' + str(args.modem_num) + '.npz')
    L = (quantized["ber"] == 0.).argmax(axis=0)
    if L == 0:
        L = 16
    snr = quantized["snr"][0:L]
    ber = quantized["ber"][0:L]
    plt.plot(snr, ber, '-*', label="One-bit quantized turbo coding")

    cnn = np.load(args.ber_dir + '/cnn_' + args.channel_mode + '_' + str(args.modem_num) + '.npz')
    L = (cnn["ber"] == 0.).argmax(axis=0)
    if L == 0:
        L = 16
    snr = cnn["snr"][0:L]
    ber = cnn["ber"][0:L]
    plt.plot(snr, ber, '-^', label="CNN for one-bit quantization with α=0.5 (proposed)")

    plt.legend(loc='best')
    plt.xlabel('$E_b$/$N_0$(dB)')
    plt.ylabel('BER')
    plt.yscale('log')
    plt.xlim(-1, 3)
    plt.ylim(10 ** (-6), 1)
    plt.show()


