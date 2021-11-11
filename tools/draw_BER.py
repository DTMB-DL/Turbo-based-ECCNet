import matplotlib.pyplot as plt
import numpy as np

plot_sets=['-*','-*','-o','-','--', ]
plot_colors = ['b','r','g','y','k']

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-ber_dir', type=str, default='../data',
                        help='dir of saved BER results(.npy). no "/" in final position.')
    parser.add_argument('-modem_num', type=int, default=4, help='number of modulation order')
    args = parser.parse_args()

    unquantized = np.load(args.ber_dir+'/unquantized_'+str(args.modem_num)+'.npz')
    L = (unquantized["ber"] == 0.).argmax(axis=0)
    snr = unquantized["snr"][0:L]
    ber = unquantized["ber"][0:L]
    print("================unquantized=====================")
    print(snr)
    print(ber)
    plt.plot(snr, ber, label="unquantized")

    quantized = np.load(args.ber_dir + '/quantized_' + str(args.modem_num) + '.npz')
    L = (quantized["ber"] == 0.).argmax(axis=0)
    if(L == 0):
        L = 16
    snr = quantized["snr"][0:L]
    ber = quantized["ber"][0:L]
    print("================quantized=====================")
    print(snr)
    print(ber)
    plt.plot(snr, ber, label="quantized")

    cnn = np.load(args.ber_dir + '/cnn_' + str(args.modem_num) + '.npz')
    L = (cnn["ber"] == 0.).argmax(axis=0)
    snr = cnn["snr"][0:L]
    ber = cnn["ber"][0:L]
    print("================cnn=====================")
    print(snr)
    print(ber)
    plt.plot(snr, ber, label="cnn")

    plt.legend(loc='best')
    plt.xlabel('$E_b$/$N_0$(dB)')
    plt.ylabel('BER')
    plt.yscale('log')
    plt.xlim(-1,3)
    plt.show()


