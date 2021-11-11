import torch
from setting.setting import device
__all__ = ['awgn_channel']


def awgn_channel(input_signals, SNR):
    assert input_signals.dim() == 3   # (2, batch_size, L)
    real = input_signals[0]
    imag = input_signals[1]
    S = torch.mean(real**2+imag**2, axis=1)
    B = len(S)
    snr_linear = 10 ** (SNR / 10.0)
    noise_variance = S / (2 * snr_linear)
    noise_real = torch.sqrt(torch.reshape(noise_variance, (B, 1))) * torch.randn(B, len(real[0]), device=device)
    noise_imag = torch.sqrt(torch.reshape(noise_variance, (B, 1))) * torch.randn(B, len(real[0]), device=device)
    noise = torch.stack((noise_real, noise_imag), dim=0)
    output_signal = input_signals + noise

    return [output_signal, noise_variance]


if __name__ == '__main__':
    pass
