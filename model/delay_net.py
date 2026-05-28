import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
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
            ConvBlock(2, base_ch),

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
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, 1, kernel_size=1),
        )

    def forward(self, x):
        feat = self.encoder(x)          # [B, C, M, N']
        feat = self.pool(feat)          # [B, C, M, 1]

        tau = self.delay_head(feat)     # [B, 1, M, 1]
        tau = tau.squeeze(1).squeeze(-1)  # [B, M]

        return tau