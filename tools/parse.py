import argparse
__all__ = ['get_args']


def get_args():
    parser = argparse.ArgumentParser()

    '''=============================  generate data opt  ==============================='''
    parser.add_argument('-train_len', type=int, default=300, help='train_len * len is total training length.')
    parser.add_argument('-test_len', type=int, default=200, help='test_len * len is total testing length.')
    parser.add_argument('-val_len', type=int, default=100, help='val_len * len is total validating length.')
    parser.add_argument('-ber_len', type=int, default=100, help='ber_len * len is final testing length.')

    parser.add_argument('-len', type=int, default=6144, help='total length of each piece.')
    parser.add_argument('-code_len', type=int, default=18444, help='total length of each coded piece.')

    parser.add_argument('-train_cut', type=int, default=40000, help='trains: [train_cut, N]')
    parser.add_argument('-test_cut', type=int, default=25000, help='tests: [test_cut, N]')
    parser.add_argument('-val_cut', type=int, default=10000, help='vals: [val_cut, N]')

    '''============================= common opt ==============================='''
    parser.add_argument('-epoch', type=int, default=100, help='total epoch')
    parser.add_argument('-batch_size', type=int, default=200, help='batch size')
    parser.add_argument('-mode', choices=['train', 'test'], default='train', help='train or test')

    parser.add_argument('-G', type=int, default=3, help='G in front layer')
    parser.add_argument('-N', type=int, default=16, help='length of block(bit)')
    parser.add_argument('-modem_num', type=int, default=4, help='number of modulation order. decide ISI.')

    '''=========================== training opt =============================='''
    parser.add_argument('-unit_T', type=int, default=5, help='increasing number of T for each epoch')

    parser.add_argument('-lr', type=float, default=5e-4, help='init learning rate')
    parser.add_argument('-lr_step', type=int, default=25, help='change step of learning rate')

    parser.add_argument('-snr', type=float, default=5.0, help='E_b/n_0 for training.')
    parser.add_argument('-snr_start', type=float, default=-1.0, help='start value of Eb/n0 for testing')
    parser.add_argument('-snr_step', type=float, default=1, help='step value of Eb/n0 for testing')
    parser.add_argument('-snr_end', type=float, default=3.0, help='end value of Eb/n0 for testing')

    '''=========================== choose main opt =============================='''
    parser.add_argument('-curve', choices=['unquantized', 'quantized', 'cnn'], default='cnn', help='for main')
    parser.add_argument('-channel_mode', choices=['awgn', 'fading'], default='awgn', help='for channel')

    args = parser.parse_args()
    return args


