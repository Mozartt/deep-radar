import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(3, 7), groups=8):
        super().__init__()
        pad = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad),
            nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
            nn.GELU(),

            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=pad),
            nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)
    
class RadarDETR(nn.Module):
    """
    Radar-DETR v1.

    Input:
        y: complex tensor [B, M, N]
           B = batch
           M = receivers
           N = fast-time samples

    Output:
        dict:
            pos:    [B, num_queries, 3] normalized xy in [-1, 1]
            logits: [B, num_queries, 2] no-target / target logits
            spectrum_logits: [B, 1, M, F]
    """

    def __init__(
        self,
        M: int,
        rx_pos: torch.Tensor,
        n_fft: int = 1024,
        d_model: int = 256,
        num_queries: int = 8,
        top_p: int = 32,
        num_decoder_layers: int = 3,
        nhead: int = 8,
        common_params=None,
    ):
        super().__init__()

        self.M = M
        self.n_fft = n_fft
        self.d_model = d_model
        self.num_queries = num_queries
        self.top_p = top_p
        self.fs = common_params.fs

        # receiver positions [M, 3]
        # stored as buffer so it moves with model.to(device)
        assert rx_pos.shape == (M, 3)
        self.register_buffer("rx_pos", rx_pos.float())

        # FFT/CNN input channels:
        # Re(FFT), Im(FFT), log magnitude
        self.backbone = nn.Sequential(
            ConvBlock2d(3, 64),
            ConvBlock2d(64, 128),
            ConvBlock2d(128, d_model),
        )

        # Interpretable spectral score map
        # This is the learned multi-peak delay/frequency likelihood.
        self.spectrum_head = nn.Conv2d(d_model, 1, kernel_size=1)

        # Token positional embedding from physical metadata:
        # [rx_x, rx_y, rx_z, normalized_frequency]
        self.token_pos_mlp = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Learned object queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        # Coordinate head: normalized xy
        self.pos_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
            nn.Tanh(),  # output in [-1, 1]
        )

        # Classification head: no-target / target
        self.obj_head = nn.Linear(d_model, 2)

    def make_fft_input(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: complex [B, M, N]

        returns:
            x: real [B, 3, M, F]
        """

        assert torch.is_complex(y), "Input y should be complex tensor [B, M, N]"

        # FFT over fast-time
        Y = torch.fft.fft(y, n=self.n_fft, dim=-1, norm="ortho")
        Y = torch.fft.fftshift(Y, dim=-1)

        half_fft_size = self.n_fft // 2
        Y = Y[..., :half_fft_size]  # [B, M, F] use only negative frequencies

        # Per-receiver normalization.
        # Keeps numerical scale stable.
        scale = Y.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        Yn = Y / scale

        re = Yn.real
        im = Yn.imag
        logmag = torch.log1p(Y.abs() / scale)

        x = torch.stack([re, im, logmag], dim=1)  # [B, 3, M, F]
        return x

    def make_frequency_grid(self, device, real_fft_size: int) -> torch.Tensor:
        """
        Returns normalized shifted frequency coordinates in [-1, 1].
        Shape: [F]
        """
        freqs = torch.fft.fftfreq(
            self.n_fft,
            d=1.0 / self.fs,
            device=device,
        )
        freqs = torch.fft.fftshift(freqs)
        freqs = freqs[: self.n_fft // 2]

        return freqs / (self.fs / 2)

    def select_top_tokens(self, feat, spectrum_logits):
        """
        feat: [B, C, M, F]
        spectrum_logits: [B, 1, M, F]

        returns:
            tokens: [B, M * top_p, C]
        """

        B, C, M, Freq = feat.shape
        device = feat.device

        score = spectrum_logits.squeeze(1)  # [B, M, F]

        # Top-P bins per receiver
        _, top_idx = torch.topk(score, k=self.top_p, dim=-1)  # [B, M, P]

        # Convert feature map to [B, M, F, C]
        feat_bmfc = feat.permute(0, 2, 3, 1).contiguous()

        # Gather top-P features
        idx_feat = top_idx.unsqueeze(-1).expand(B, M, self.top_p, C)
        tok_feat = torch.gather(feat_bmfc, dim=2, index=idx_feat)  # [B, M, P, C]

        # Frequency coordinate for selected bins
        f_grid = self.make_frequency_grid(device, self.n_fft)  # [F]
        f_bmf = f_grid.view(1, 1, Freq).expand(B, M, Freq)
        tok_f = torch.gather(f_bmf, dim=2, index=top_idx)  # [B, M, P]
        tok_f = tok_f.unsqueeze(-1)  # [B, M, P, 1]

        # Receiver positions
        rx = self.rx_pos.view(1, M, 1, 3).expand(B, M, self.top_p, 3)

        # Normalize receiver positions for stable learning
        rx_scale = self.rx_pos.abs().max().clamp_min(1.0)
        rx_norm = rx / rx_scale

        # Physical token metadata: [rx_x, rx_y, rx_z, f_norm]
        meta = torch.cat([rx_norm, tok_f], dim=-1)  # [B, M, P, 4]
        tok_pos = self.token_pos_mlp(meta)          # [B, M, P, C]

        tokens = tok_feat + tok_pos
        tokens = tokens.view(B, M * self.top_p, C)

        return tokens

    def forward(self, y: torch.Tensor):
        """
        y: complex [B, M, N]
        """

        B = y.shape[0]

        x = self.make_fft_input(y)        # [B, 3, M, F]
        feat = self.backbone(x)           # [B, d_model, M, F]

        spectrum_logits = self.spectrum_head(feat)  # [B, 1, M, F]

        tokens = self.select_top_tokens(feat, spectrum_logits)
        # tokens: [B, M * top_p, d_model]

        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        # [B, num_queries, d_model]

        hs = self.decoder(
            tgt=queries,
            memory=tokens,
        )
        # [B, num_queries, d_model]

        pos = self.pos_head(hs)       # [B, Q, 2]
        logits = self.obj_head(hs)    # [B, Q, 2]

        return {
            "pos": pos,
            "logits": logits,
            "spectrum_logits": spectrum_logits,
        }