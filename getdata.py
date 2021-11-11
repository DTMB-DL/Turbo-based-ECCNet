import numpy as np
import matlab
import math
import matlab.engine
from tools.parse import *


def start_matlab():
    eng = matlab.engine.start_matlab()
    return eng


def generate_data(args):
    eng = start_matlab()
    train_dataset = np.random.randint(0, 2, [args.train_len, args.len])
    val_dataset = np.random.randint(0, 2, [args.val_len, args.len])
    test_dataset = np.random.randint(0, 2, [args.test_len, args.len])
    bers = np.random.randint(0, 2, [args.ber_len, args.len])

    trains = []
    vals = []
    tests = []
    for k in range(args.train_len):
        data = eng.lteTurboEncode(matlab.int8(train_dataset[k].tolist()))
        data = np.array(data).reshape(args.code_len)
        maxcnt = math.ceil(args.code_len / args.N)
        for j in range(maxcnt):
            ones = np.zeros(args.N)
            ones[0:min(args.N, args.code_len - j * args.N)] = data[j * args.N:min((j + 1) * args.N, args.code_len)]
            trains.append(ones)

    for k in range(args.val_len):
        data = eng.lteTurboEncode(matlab.int8(val_dataset[k].tolist()))
        data = np.array(data).reshape(args.code_len)
        maxcnt = math.ceil(args.code_len / args.N)
        for j in range(maxcnt):
            ones = np.zeros(args.N)
            ones[0:min(args.N, args.code_len - j * args.N)] = data[j * args.N:min((j + 1) * args.N, args.code_len)]
            vals.append(ones)

    for k in range(args.test_len):
        data = eng.lteTurboEncode(matlab.int8(test_dataset[k].tolist()))
        data = np.array(data).reshape(args.code_len)
        maxcnt = math.ceil(args.code_len / args.N)
        for j in range(maxcnt):
            ones = np.zeros(args.N)
            ones[0:min(args.N, args.code_len - j * args.N)] = data[j * args.N:min((j + 1) * args.N, args.code_len)]
            tests.append(ones)

    trains = trains[:args.train_cut]
    tests = tests[:args.test_cut]
    vals = vals[:args.val_cut]

    import os
    if not os.path.exists('./data'):
        os.mkdir('./data')
    np.savez('./data/gen_data', trains=trains, tests=tests, vals=vals, bers=bers)
    eng.quit()


if __name__ == "__main__":
    args = get_args()
    generate_data(args)
