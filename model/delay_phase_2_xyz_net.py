from math import tau

import torch
import torch.nn as nn

class TauPhase2PredictionNet3D(nn.Module):
    """
    Input:
        tau: [B, M]
            normalized delay estimates per receiver

        delta_phi: [B, M] or [B, M-1]
            phase differences relative to receiver 0.
            If shape is [B, M], delta_phi[:, 0] is assumed to be 0 and is dropped.
            If shape is [B, M-1], it is assumed to contain receivers 1..M-1.

    Output:
        xyz: [B, 3]
            predicted target position.
            I recommend training this in normalized xyz coordinates.
    """

    def __init__(self, M, hidden_dim=256):
        super().__init__()

        self.tau_branch = nn.Sequential(
            nn.Linear(M, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.phi_branch = nn.Sequential(
            nn.Linear(2*M, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            #nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, 3)
        )

    def forward(self, tau, cos_dphi, sin_dphi):
        tau_feat = self.tau_branch(tau)

        phi = torch.cat([cos_dphi, sin_dphi], dim=-1)
        phi_feat = self.phi_branch(phi)

        feat = torch.cat([tau_feat, phi_feat], dim=-1)
        xy = self.fusion(feat)
        return xy