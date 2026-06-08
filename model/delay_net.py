import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DelayNet(nn.Module):
    """
    Input:
        x: [B, 2, M, N]
           2 = I/Q channels
           M = receivers
           N = fast-time samples

    Output:
        tau: [B, M]
             predicted delay per receiver
    """

    def __init__(self, M=40, base_ch=32, hidden_ch=128):
        super().__init__()

        self.M = M

        self.encoder = nn.Sequential(
            ConvBlock(3, base_ch),  # 3 channels: FFT real, FFT imag, FFT magnitude

            nn.MaxPool2d(kernel_size=(1, 2)),  # reduce time only

            ConvBlock(base_ch, base_ch * 2),

            nn.MaxPool2d(kernel_size=(1, 2)),  # reduce time only

            ConvBlock(base_ch * 2, base_ch * 4),
        )

        # Keep receiver dimension, collapse time dimension
        self.pool = nn.AdaptiveAvgPool2d((M, 1))

        # Per-receiver delay head
        self.delay_head = nn.Sequential(
            nn.Conv2d(base_ch * 4, hidden_ch, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_ch, 1, kernel_size=1)
        )

    def forward(self, x):
        # x: [B, 2, M, N]  (channel 0 = I, channel 1 = Q)
        # Form complex signal and apply FFT of size 1024 over fast-time dimension.
        z = torch.complex(x[:, 0, :, :].float(), x[:, 1, :, :].float())  # [B, M, N]
        Z = torch.fft.fft(z, n=1024, dim=-1, norm="forward")                              # [B, M, 1024] complex
        Z_shifted = torch.fft.fftshift(Z, dim=-1)
        Z = Z_shifted[:, :, :512]                                               # keep negative-freq half [B, M, 512]
        # Pack real, imaginary, and magnitude as three input channels.
        x = torch.stack([Z.real, Z.imag, Z.abs()], dim=1).to(x.dtype)    # [B, 3, M, 512]

        # single_signal_complex = Z[0, 0, :]
        # single_signal_mag = torch.abs(single_signal_complex)
        # y_data = single_signal_mag.detach().cpu().numpy()
        # x_data = np.linspace(-0.5, 0, len(y_data), endpoint=False)
        # plt.figure()
        # plt.plot(x_data, y_data)
        # plt.grid(True)
        # plt.show(block=False)

        feat = self.encoder(x)          # [B, C, M, N']
        feat = self.pool(feat)          # [B, C, M, 1]

        tau = self.delay_head(feat)     # [B, 1, M, 1]
        tau = tau.squeeze(1).squeeze(-1)  # [B, M]

        return tau