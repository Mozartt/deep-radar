import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaCorrectionHead(nn.Module):
    """
    Small correction network.

    Input:
        z_ri:        [B, 2, M, N]  de-beated I/Q signal
        beta_coh_ri: [B, M, 2]     coherent beta estimate

    Output:
        corr_ri:     [B, M, 2]     unit correction phasor
        confidence:  [B, M]        confidence in beta estimate
    """

    def __init__(self, hidden_ch=64, mlp_dim=128):
        super().__init__()

        # Per-receiver Conv1D over time.
        # Important: no stride, no maxpool. We preserve phase structure.
        self.time_encoder = nn.Sequential(
            nn.Conv1d(2, hidden_ch, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden_ch, hidden_ch, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden_ch, hidden_ch, kernel_size=7, padding=3),
            nn.GELU(),
        )

        # Features:
        # hidden_ch from Conv1D
        # beta_coh real/imag = 2
        # coherence = 1
        # residual energy = 1
        feature_dim = hidden_ch + 2 + 1 + 1

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.GELU(),
        )

        # Predict small correction around identity phasor [1, 0].
        self.corr_head = nn.Linear(mlp_dim, 2)

        # Predict confidence in [0, 1].
        self.conf_head = nn.Linear(mlp_dim, 1)

        # Initialize correction head near zero.
        # Then corr ≈ [1, 0] at the beginning,
        # so beta_hat starts close to beta_coh.
        nn.init.zeros_(self.corr_head.weight)
        nn.init.zeros_(self.corr_head.bias)

    def forward(self, z_ri, beta_coh_ri):
        B, _, M, N = z_ri.shape

        # Convert de-beated signal to complex.
        z = torch.complex(z_ri[:, 0], z_ri[:, 1])  # [B, M, N]

        # Conv1D per receiver.
        x = z_ri.permute(0, 2, 1, 3)     # [B, M, 2, N]
        x = x.reshape(B * M, 2, N)       # [B*M, 2, N]

        h = self.time_encoder(x)         # [B*M, hidden_ch, N]

        # Mean over time is okay here because signal was already de-beated.
        h = h.mean(dim=-1)               # [B*M, hidden_ch]
        h = h.view(B, M, -1)             # [B, M, hidden_ch]

        # Coherence feature:
        # If de-beating was good, z[n] should be almost constant,
        # so |mean(z)| / mean(|z|) should be close to 1.
        z_mean = z.mean(dim=-1)                              # [B, M]
        coherence = z_mean.abs() / (z.abs().mean(dim=-1) + 1e-8)
        coherence = coherence.unsqueeze(-1)                  # [B, M, 1]

        # Residual energy after coherent averaging.
        residual = z - z_mean.unsqueeze(-1)
        residual_energy = torch.sqrt((residual.abs() ** 2).mean(dim=-1) + 1e-8)
        residual_energy = residual_energy.unsqueeze(-1)      # [B, M, 1]

        features = torch.cat(
            [
                h,
                beta_coh_ri,
                coherence,
                residual_energy,
            ],
            dim=-1,
        )  # [B, M, feature_dim]

        q = self.mlp(features)

        # Predict correction as residual around identity phasor.
        delta = self.corr_head(q)  # [B, M, 2]

        identity = torch.zeros_like(delta)
        identity[..., 0] = 1.0

        corr_ri = identity + delta
        corr_ri = F.normalize(corr_ri, dim=-1)

        confidence = torch.sigmoid(self.conf_head(q)).squeeze(-1)  # [B, M]

        return corr_ri, confidence