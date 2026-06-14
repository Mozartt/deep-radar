import torch
import torch.nn as nn
import torch.nn.functional as F


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


class DelayPhaseNet(nn.Module):
    """
    Input:
        x: [B, 2, M, N]

    Output:
        tau:     [B, M]
        cos_phi: [B, M]
        sin_phi: [B, M]
    """

    def __init__(self, M=40, base_ch=32, hidden_ch=128, n_fft=1024):
        super().__init__()

        self.M = M
        self.n_fft = n_fft

        self.encoder = nn.Sequential(
            ConvBlock(3, base_ch),

            nn.MaxPool2d(kernel_size=(1, 2)),

            ConvBlock(base_ch, base_ch * 2),

            nn.MaxPool2d(kernel_size=(1, 2)),

            ConvBlock(base_ch * 2, base_ch * 4),
        )

        self.pool = nn.AdaptiveAvgPool2d((M, 1))

        # Keep this EXACTLY like your old DelayNet.
        self.delay_head = nn.Sequential(
            nn.Conv2d(base_ch * 4, hidden_ch, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_ch, 1, kernel_size=1)
        )

        # New parallel phase head.
        self.phase_head = nn.Sequential(
            nn.Conv2d(base_ch * 4, hidden_ch, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_ch, 2, kernel_size=1)
        )

    def forward(self, x):
        # x: [B, 2, M, N]
        z = torch.complex(
            x[:, 0, :, :].float(),
            x[:, 1, :, :].float()
        )  # [B, M, N]

        Z = torch.fft.fft(z, n=self.n_fft, dim=-1, norm="forward")
        Z = torch.fft.fftshift(Z, dim=-1)

        # Same convention as your original model.
        Z = Z[:, :, :self.n_fft // 2]  # [B, M, 512]

        mag = torch.log1p(Z.abs())

        x_fft = torch.stack(
            [Z.real, Z.imag, mag],
            dim=1
        ).to(x.dtype)  # [B, 3, M, 512]

        feat = self.encoder(x_fft)  # [B, C, M, N']
        feat = self.pool(feat)      # [B, C, M, 1]

        tau = self.delay_head(feat)       # [B, 1, M, 1]
        tau = tau.squeeze(1).squeeze(-1)  # [B, M]

        phase = self.phase_head(feat)     # [B, 2, M, 1]
        phase = phase.squeeze(-1)         # [B, 2, M]

        # Enforce cos/sin vector on unit circle.
        phase = F.normalize(phase, p=2, dim=1, eps=1e-8)

        cos_phi = phase[:, 0, :]          # [B, M]
        sin_phi = phase[:, 1, :]          # [B, M]

        return tau, cos_phi, sin_phi