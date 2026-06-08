import torch
import torch.nn as nn
import torch.nn.functional as F

class RadarSignalEmbedding(nn.Module):
    """
    Input:
        y_realimag: [B, 2, M, N]
        channel 0 = real(y)
        channel 1 = imag(y)

    Output:
        x: [B, C, M, N]
    """

    def __init__(self, use_fft=True):
        super().__init__()
        self.use_fft = use_fft

    def forward(self, y_realimag):
        y_real = y_realimag[:, 0]
        y_imag = y_realimag[:, 1]

        y_complex = torch.complex(y_real, y_imag)

        # Normalize per sample
        power = torch.mean(torch.abs(y_complex) ** 2, dim=(1, 2), keepdim=True)
        y_complex = y_complex / (torch.sqrt(power) + 1e-8)

        channels = [
            y_complex.real,
            y_complex.imag,
        ]

        if self.use_fft:
            Yf = torch.fft.fft(y_complex, dim=2)

            channels += [
                Yf.real,
                Yf.imag,
                torch.abs(Yf),
            ]

        x = torch.stack(channels, dim=1)
        return x


# -------------------------
# Basic CNN block
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# -------------------------
# Shared encoder
# -------------------------

class RadarEncoder(nn.Module):
    def __init__(self, in_ch=5, base_ch=32, latent_dim=256):
        super().__init__()
        self.conv1 = ResBlock(in_ch, base_ch)           # 40x80
        self.pool = nn.MaxPool2d(2)                     # 20x40

        self.conv2 = ResBlock(base_ch, base_ch * 2)
        self.conv3 = ResBlock(base_ch * 2, base_ch * 4)
        self.conv4 = ResBlock(base_ch * 4, base_ch * 4)

        self.gap = nn.AdaptiveAvgPool2d(1)          # [B, 128, 20, 40] → [B, 128, 1, 1]

        self.fc = nn.Sequential(
            nn.Linear(base_ch * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)        # [B, 32, 40, 80]
        x = self.pool(x)         # [B, 32, 20, 40]

        x = self.conv2(x)        # [B, 64, 20, 40]
        x = self.conv3(x)        # [B, 128, 20, 40]
        x = self.conv4(x)        # [B, 128, 20, 40]

        x = self.gap(x)          # [B, 128, 1, 1]
        x = torch.flatten(x, start_dim=1)  # [B, 128]
        latent = self.fc(x)

        return latent
    
# -------------------------
# Heatmap decoder head
# -------------------------
class HeatmapHead(nn.Module):
    def __init__(self, latent_dim=256, out_size=64):
        super().__init__()

        self.out_size = out_size

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, latent):
        x = self.fc(latent)                  # [B, 256] -> [B, 4096]
        x = x.view(latent.shape[0], 256, 4, 4)  # [B, 256, 4, 4]
        x = self.decoder(x)                  # [B, 256, 4, 4]
                                             # -> ConvTranspose2d -> [B, 128, 8, 8]
                                             # -> ConvTranspose2d -> [B, 64, 16, 16]
                                             # -> ConvTranspose2d -> [B, 32, 32, 32]
                                             # -> Conv2d          -> [B, 1, 32, 32]

        x = F.interpolate(
            x,
            size=(self.out_size, self.out_size),
            mode="bilinear",
            align_corners=False,
        )                                    # [B, 1, out_size, out_size]

        #heatmap = torch.sigmoid(x)
        heatmap=x
        return heatmap
    
# -------------------------
# Coordinate regression head
# -------------------------

class CoordinateHead(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

    def forward(self, latent):
        return self.mlp(latent)  # [B, 256] -> [B, 128] -> [B, 64] -> [B, 3]

class CoordinateHead2D(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, latent):
        return self.mlp(latent)  # [B, 256] -> [B, 128] -> [B, 64] -> [B, 2]

# -------------------------
# Full model
# -------------------------
class RadarMultiTaskNet(nn.Module):
    def __init__(self, use_fft=True, heatmap_size=64):
        super().__init__()

        in_ch = 5 if use_fft else 2

        self.embedding = RadarSignalEmbedding(use_fft=use_fft)
        self.encoder = RadarEncoder(in_ch=in_ch)
        self.heatmap_head = HeatmapHead(out_size=heatmap_size)
        self.coord_head = CoordinateHead()

    def forward(self, y_realimag):
        x = self.embedding(y_realimag)
        latent = self.encoder(x)

        pred_heatmap = self.heatmap_head(latent)
        pred_coord = self.coord_head(latent)

        return pred_heatmap, pred_coord

class RadarMultiTaskNetPositionOnly3D(nn.Module):
    def __init__(self, use_fft=True, heatmap_size=64):
        super().__init__()

        in_ch = 5 if use_fft else 2

        self.embedding = RadarSignalEmbedding(use_fft=use_fft)
        self.encoder = RadarEncoder(in_ch=in_ch)
        self.coord_head = CoordinateHead()

    def forward(self, y_realimag):
        x = self.embedding(y_realimag)
        latent = self.encoder(x)
        pred_coord = self.coord_head(latent)

        return pred_coord

class RadarMultiTaskNetPositionOnly2D(nn.Module):
    def __init__(self, use_fft=True):
        super().__init__()

        in_ch = 5 if use_fft else 2

        self.embedding = RadarSignalEmbedding(use_fft=use_fft)
        self.encoder = RadarEncoder(in_ch=in_ch)
        self.coord_head = CoordinateHead2D()
        self.test_mlp = RadarMLP(in_ch=in_ch)
        self.memorizer = Memorizer(n_samples=100)

    def forward(self, y_realimag):
        x = self.embedding(y_realimag)
        latent = self.encoder(x)
        pred_coord = self.coord_head(latent)
        return pred_coord

class RadarMultiTaskNetONNX(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = RadarEncoder(in_ch=5)
        self.coord_head = CoordinateHead()

    def forward(self, x_embedded):
        latent = self.encoder(x_embedded)
        pred_coord = self.coord_head(latent)
        return pred_coord
    

class RadarMLP(nn.Module):
    def __init__(self, in_ch=5, M=40, N=80):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * M * N, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.net(x)
    

class Memorizer(nn.Module):
    def __init__(self, n_samples=100):
        super().__init__()

        # one learnable (x,y) per sample
        self.coords = nn.Parameter(
            torch.zeros(n_samples, 2)
        )

    def forward(self, idx):
        return self.coords[idx]


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )

        self.skip = nn.Identity()
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))