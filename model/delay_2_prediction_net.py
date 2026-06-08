import torch
import torch.nn as nn
import torch.nn.functional as F

class Delay2PredictionNet(nn.Module):
    """
    Input:
        delays: [B, M]
            delay estimate per receiver

    Output:
        xy: [B, 2]
            predicted target position [x, y]
    """

    def __init__(self, M, hidden_dim=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(M, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),

            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, delays):
        return self.net(delays)

