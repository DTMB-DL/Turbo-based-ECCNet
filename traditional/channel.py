import torch
from setting.setting import device
import torch.nn.functional as F
__all__ = ['channel']


def awgn_channel(input_signals, SNR):
    assert input_signals.dim() == 3   # (2, batch_size, L)
    real = input_signals[0]
    imag = input_signals[1]
    S = torch.mean(real ** 2 + imag ** 2)
    snr_linear = 10 ** (SNR / 10.0)
    noise_variance = S / (2 * snr_linear)
    noise = torch.sqrt(noise_variance) * torch.randn_like(input_signals, device=device)
    output_signal = input_signals + noise

    return [output_signal, noise_variance]


def fading_channel(input_signals, SNR, H_r, H_i, rate=5*3):
    assert input_signals.dim() == 3  # (2, batch_size, L)
    real = input_signals[0]
    imag = input_signals[1]
    S = torch.mean(real ** 2 + imag ** 2)
    snr_linear = 10 ** (SNR / 10.0)
    noise_variance = S / (2 * snr_linear)
    noise = torch.sqrt(noise_variance) * torch.randn_like(input_signals, device=device)
    h_r = torch.zeros((len(H_r) - 1) * rate + 1).to(device)
    h_r[:(len(H_r) - 1) * rate + 1:rate] = H_r
    h_i = torch.zeros((len(H_i) - 1) * rate + 1).to(device)
    h_i[:(len(H_i) - 1) * rate + 1:rate] = H_i

    def conv(x, h):
        y = F.conv1d(x.reshape(input_signals.shape[1], 1, -1), h.flip(dims=[0]).reshape(1, 1, -1), padding=(len(H_i) - 1) * rate)
        y = y.reshape(input_signals.shape[1], -1)[:, :x.shape[1]]
        return y

    out_r = conv(real, h_r) - conv(imag, h_i)
    out_i = conv(imag, h_r) + conv(real, h_i)
    out = torch.stack((out_r, out_i), dim=0) + noise

    return out, noise_variance


def channel(channel_mode, intpu_signals, SNR, H_r=torch.tensor([0.9, -0.05, 0.1]), H_i=torch.tensor([0.05, -0.06, 0.02]), rate=5 * 3):
    if channel_mode == 'awgn':
        return awgn_channel(intpu_signals, SNR)
    elif channel_mode == 'fading':
        return fading_channel(intpu_signals, SNR, H_r, H_i, rate)



