import torch.utils.data as data
import torch
import numpy as np
from setting.setting import device
__all__ = ['Mydataset', 'loaddata']


class Mydataset(data.Dataset):
    def __init__(self, datasets):
        super(Mydataset, self).__init__()
        self.datasets = datasets

    def __getitem__(self, index):
        return torch.tensor(self.datasets[index]).to(device).float()

    def __len__(self):
        return len(self.datasets)


def loaddata(path='./data/gen_data_4.npz'):
    data = np.load(path)
    trains = data["trains"]
    vals = data["vals"]
    tests = data["tests"]
    return trains, tests, vals

