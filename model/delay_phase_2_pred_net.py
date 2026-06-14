import torch
import torch.nn as nn

class TauPhase2PredictionNet(nn.Module):
    """
    Input:
        tau:     [B, M]
                 normalized delay estimate per receiver

        cos_phi: [B, M]
                 cos(delta_phi_m) per receiver

        sin_phi: [B, M]
                 sin(delta_phi_m) per receiver

    Output:
        xy:      [B, 2]
                 predicted target position [x, y]
    """

    def __init__(self, M, hidden_dim=512, dropout=0.0):
        super().__init__()

        self.M = M
        self.in_dim = 3 * M

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),

            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, tau, cos_phi, sin_phi):
        """
        tau, cos_phi, sin_phi: [B, M]
        """

        x = torch.cat([tau, cos_phi, sin_phi], dim=1)  # [B, 3M]

        return self.net(x)