import torch.nn as nn
import torch
import math
from models.quantization import *
from setting.setting import device
from traditional import *

__all__ = ['AutoEncoder']


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y


class SE(nn.Module):
    def __init__(self, Cin):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.compress = nn.Conv1d(Cin, Cin, 1)
        self.excitation = nn.Conv1d(Cin, Cin, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = self.relu(out)
        out = self.excitation(out)
        out = self.sigmoid(out)
        out = x * out.expand_as(x)
        return out

    def __call__(self, x):
        return self.forward(x)


class ResSEBlock(nn.Module):
    def __init__(self, Cin, Cout, rate=1):
        super(ResSEBlock, self).__init__()
        if Cin != Cout:
            self.shortcut = nn.Sequential(
                nn.BatchNorm1d(Cin),
                nn.ReLU(),
                nn.Conv1d(Cin, Cout, 1)
            )
        else:
            self.shortcut = nn.Identity()
        self.conv = nn.Sequential(
            nn.BatchNorm1d(Cin),
            nn.ReLU(),
            nn.Conv1d(Cin, Cout, 2 * rate + 1, stride=1, padding=rate),
            nn.BatchNorm1d(Cout),
            nn.ReLU(),
            nn.Conv1d(Cout, Cout, 2 * rate + 1, stride=1, padding=rate),
        )
        self.se = SE(Cout)

    def forward(self, x):
        x_shortcut = self.shortcut(x)
        x_res = self.se(self.conv(x))  # with SE
        # x_res = self.conv(x)  # without SE
        out = x_shortcut + x_res
        return out

    def __call__(self, x):
        return self.forward(x)


class Encoder(nn.Module):
    def __init__(self, G=3, N=16):
        super(Encoder, self).__init__()
        self.G = G
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 128, 11, stride=1, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, 5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm1d(32),
        )
        self.shortcut = nn.Sequential(
            nn.Conv1d(2, 32, 1),
            nn.BatchNorm1d(32),
        )
        self.timedis = nn.Sequential(
            nn.ReLU(),
            TimeDistributed(nn.Linear(32, 2*G), True),
            nn.BatchNorm1d(N),
        )

    def forward(self, x):
        out1 = self.encoder(x)
        out1_ori = self.shortcut(x)
        out = self.timedis((out1 + out1_ori).reshape(x.shape[0], 32, -1).transpose(1, 2).contiguous()).reshape(x.shape[0], 2, -1)
        return out

    def __call__(self, x):
        return self.forward(x)


class Decoder(nn.Module):
    def __init__(self, G=3):
        super(Decoder, self).__init__()
        self.G = G
        self.decoder = nn.Sequential(
            nn.Conv1d(2, 256, 5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            ResSEBlock(256, 128, 10),
            ResSEBlock(128, 128, 5),
            ResSEBlock(128, 64, 3),
            ResSEBlock(64, 64, 2),
            ResSEBlock(64, 32, 2),
            ResSEBlock(32, 32, 1),
            nn.Conv1d(32, 20, 5, stride=1, padding=2),
            nn.BatchNorm1d(20),
            nn.ReLU(),
        )
        self.timedis = nn.Sequential(
            TimeDistributed(nn.Linear(G * 20, 2), batch_first=True),
        )

    def forward(self, x):
        out = self.decoder(x).reshape(x.shape[0], 20 * self.G, -1).transpose(1, 2).contiguous()
        out = self.timedis(out).reshape(x.shape[0], 2, -1)
        return out

    def __call__(self, x):
        return self.forward(x)


class AutoEncoder(nn.Module):
    def __init__(self, G=3, N=16, qua_bits=1, modem_num=4, channel_mode='awgn'):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(G, N)
        self.decoder = Decoder(G)
        self.channel_mode = channel_mode
        self.k = int(math.log2(modem_num))
        self.G = G

        self.quantization = Quantization(qua_bits)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, snr, ac_T, mode='train'):
        out = self.encoder(x)

        xx = torch.cat((out[:, 0, :], out[:, 1, :])).reshape(2, x.shape[0], -1)
        tx = pulse_shaping(xx, ISI=self.G, rate=5 * self.G)
        [ry, sigma2] = channel(self.channel_mode, tx,
                               snr + 10 * math.log10(self.k / (self.G * tx.shape[2] / xx.shape[2]) / 3),
                               rate=5 * self.G)
        y = matched_filtering(ry, ISI=self.G, rate=5 * self.G)
        r = self.quantization(y, mode, ac_T)
        r = torch.cat((r[0], r[1]), 1).reshape(x.shape[0], 2, -1).to(device)

        out = self.decoder(r)

        return out