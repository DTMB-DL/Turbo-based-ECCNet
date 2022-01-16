import numpy as np
import matlab
import math
import matlab.engine
from tools.parse import *
from tools.utils import *


def generate_data(args):  # each block is N=16 symbols
    eng = start_matlab()
    train_dataset = np.random.randint(0, 2, [args.train_len, args.len])
    val_dataset = np.random.randint(0, 2, [args.val_len, args.len])
    test_dataset = np.random.randint(0, 2, [args.test_len, args.len])
    modem, base = get_modem(args.modem_num)
    N = args.N
    trains = []
    vals = []
    tests = []
    for k in range(args.train_len):
        data = eng.lteTurboEncode(matlab.int8(train_dataset[k].tolist()))
        x = eng.lteSymbolModulate(data, modem)
        x = np.array(x).reshape(-1)
        x = np.stack((x.real, x.imag), axis=0).reshape(2, -1)

        L = x.shape[1]
        maxcnt = math.ceil(L / N) * N

        s1 = np.concatenate((x, x[:, :maxcnt - L]), axis=1)
        s2 = np.stack((s1[0].reshape(-1, N), s1[1].reshape(-1, N)), axis=1)
        for each in range(len(s2)):
            trains.append(s2[each])
    trains = np.array(trains)

    for k in range(args.val_len):
        data = eng.lteTurboEncode(matlab.int8(val_dataset[k].tolist()))
        x = eng.lteSymbolModulate(data, modem)
        x = np.array(x).reshape(-1)
        x = np.stack((x.real, x.imag), axis=0).reshape(2, -1)

        L = x.shape[1]
        maxcnt = math.ceil(L / N) * N

        s1 = np.concatenate((x, x[:, :maxcnt - L]), axis=1)
        s2 = np.stack((s1[0].reshape(-1, N), s1[1].reshape(-1, N)), axis=1)
        for each in range(len(s2)):
            vals.append(s2[each])
    vals = np.array(vals)

    for k in range(args.test_len):
        data = eng.lteTurboEncode(matlab.int8(test_dataset[k].tolist()))
        x = eng.lteSymbolModulate(data, modem)
        x = np.array(x).reshape(-1)
        x = np.stack((x.real, x.imag), axis=0).reshape(2, -1)

        L = x.shape[1]
        maxcnt = math.ceil(L / N) * N

        s1 = np.concatenate((x, x[:, :maxcnt - L]), axis=1)
        s2 = np.stack((s1[0].reshape(-1, N), s1[1].reshape(-1, N)), axis=1)
        for each in range(len(s2)):
            tests.append(s2[each])
    tests = np.array(tests)

    trains = trains[:args.train_cut]
    tests = tests[:args.test_cut]
    vals = vals[:args.val_cut]

    import os
    if not os.path.exists('./data'):
        os.mkdir('./data')
    np.savez('./data/gen_data_' + str(args.modem_num), trains=trains, tests=tests, vals=vals)
    eng.quit()


if __name__ == "__main__":
    args = get_args()
    generate_data(args)
