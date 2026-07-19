import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(3, 7), groups=8):
        super().__init__()

        pad = (
            kernel_size[0] // 2,
            kernel_size[1] // 2,
        )

        self.net = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=pad,
            ),
            nn.GroupNorm(
                num_groups=min(groups, out_ch),
                num_channels=out_ch,
            ),
            nn.GELU(),

            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=pad,
            ),
            nn.GroupNorm(
                num_groups=min(groups, out_ch),
                num_channels=out_ch,
            ),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)
    
class RadarDETR(nn.Module):
    """
    Radar-DETR with transformer encoder and decoder.

    Input:
        y: complex tensor [B, M, N]

    Output:
        pos:
            [B, num_queries, 3]
            Normalized xyz coordinates in [-1, 1].

        logits:
            [B, num_queries, 2]
            No-target / target logits.

        spectrum_logits:
            [B, 1, M, F]
            Per-receiver spectral likelihood map.
    """

    def __init__(
        self,
        M: int,
        rx_pos: torch.Tensor,
        n_fft: int = 1024,
        d_model: int = 256,
        num_queries: int = 8,
        top_p: int = 32,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        nhead: int = 8,
        dropout: float = 0.1,
        common_params=None,
    ):
        super().__init__()

        assert common_params is not None
        assert rx_pos.shape == (M, 3)
        assert d_model % nhead == 0
        assert top_p <= n_fft // 2

        self.M = M
        self.n_fft = n_fft
        self.d_model = d_model
        self.num_queries = num_queries
        self.top_p = top_p
        self.fs = common_params.fs

        self.register_buffer(
            "rx_pos",
            rx_pos.float(),
        )

        # ------------------------------------------------------------
        # FFT/CNN backbone
        # ------------------------------------------------------------
        # Channels:
        #   1. Real FFT
        #   2. Imaginary FFT
        #   3. Log magnitude
        self.backbone = nn.Sequential(
            ConvBlock2d(3, 64),
            ConvBlock2d(64, 128),
            ConvBlock2d(128, d_model),
        )

        # Learned spectral likelihood map.
        self.spectrum_head = nn.Conv2d(
            d_model,
            1,
            kernel_size=1,
        )

        # ------------------------------------------------------------
        # Physical token positional embedding
        # ------------------------------------------------------------
        # Metadata:
        # [receiver_x, receiver_y, receiver_z, normalized_frequency]
        self.token_pos_mlp = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # ------------------------------------------------------------
        # Transformer encoder
        # ------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ------------------------------------------------------------
        # Learned DETR object queries
        # ------------------------------------------------------------
        self.query_embed = nn.Embedding(
            num_queries,
            d_model,
        )

        # ------------------------------------------------------------
        # Transformer decoder
        # ------------------------------------------------------------
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ------------------------------------------------------------
        # Prediction heads
        # ------------------------------------------------------------
        self.pos_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3)
        )

        self.obj_head = nn.Linear(
            d_model,
            2,
        )

    def make_fft_input(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y:
                Real input [B, 2, M, N], channel 0 = I, channel 1 = Q.
                (Kept as real channels, not complex, so this module works
                correctly under nn.DataParallel — DataParallel's scatter
                does not reliably support complex tensors.)

        Returns:
            x:
                Real tensor [B, 3, M, F],
                where F = n_fft // 2.
        """

        assert not torch.is_complex(y), "y must be real [B, 2, M, N] (I/Q channels)"
        assert y.dim() == 4 and y.shape[1] == 2, f"expected [B, 2, M, N], got {tuple(y.shape)}"

        y_complex = torch.complex(y[:, 0], y[:, 1])  # [B, M, N]

        # FFT over fast time.
        Y = torch.fft.fft(
            y_complex,
            n=self.n_fft,
            dim=-1,
            norm="ortho",
        )

        Y = torch.fft.fftshift(
            Y,
            dim=-1,
        )

        # Retain negative-frequency half.
        Y = Y[..., : self.n_fft // 2]

        # Independent normalization per receiver.
        scale = Y.abs().amax(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)

        Yn = Y / scale

        re = Yn.real
        im = Yn.imag
        logmag = torch.log1p(Y.abs() / scale)

        x = torch.stack(
            [re, im, logmag],
            dim=1,
        )

        return x

    def make_frequency_grid(
        self,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Returns:
            Normalized negative-frequency grid [F] in approximately [-1, 0).
        """

        freqs = torch.fft.fftfreq(
            self.n_fft,
            d=1.0 / self.fs,
            device=device,
        )

        freqs = torch.fft.fftshift(freqs)

        # Same bins retained in make_fft_input().
        freqs = freqs[: self.n_fft // 2]

        # Normalize by the Nyquist frequency.
        return freqs / (self.fs / 2)

    def select_top_tokens(
        self,
        feat: torch.Tensor,
        spectrum_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Selects the strongest P spectral tokens from every receiver.

        Args:
            feat:
                [B, C, M, F]

            spectrum_logits:
                [B, 1, M, F]

        Returns:
            tokens:
                [B, M * top_p, C]
        """

        B, C, M, Freq = feat.shape
        device = feat.device

        assert M == self.M
        assert self.top_p <= Freq

        score = spectrum_logits.squeeze(1)
        # [B, M, F]

        # Top-P frequency bins independently for every receiver.
        _, top_idx = torch.topk(
            score,
            k=self.top_p,
            dim=-1,
        )
        # [B, M, P]

        # Convert feature map to [B, M, F, C].
        feat_bmfc = feat.permute(
            0,
            2,
            3,
            1,
        ).contiguous()

        idx_feat = top_idx.unsqueeze(-1).expand(
            B,
            M,
            self.top_p,
            C,
        )

        tok_feat = torch.gather(
            feat_bmfc,
            dim=2,
            index=idx_feat,
        )
        # [B, M, P, C]

        # Selected normalized frequencies.
        f_grid = self.make_frequency_grid(device)
        # [F]

        f_bmf = f_grid.view(
            1,
            1,
            Freq,
        ).expand(
            B,
            M,
            Freq,
        )

        tok_f = torch.gather(
            f_bmf,
            dim=2,
            index=top_idx,
        ).unsqueeze(-1)
        # [B, M, P, 1]

        # Receiver geometry.
        rx = self.rx_pos.view(
            1,
            M,
            1,
            3,
        ).expand(
            B,
            M,
            self.top_p,
            3,
        )

        rx_scale = self.rx_pos.abs().amax().clamp_min(1.0)
        rx_norm = rx / rx_scale

        # Physical positional metadata.
        meta = torch.cat(
            [rx_norm, tok_f],
            dim=-1,
        )
        # [B, M, P, 4]

        tok_pos = self.token_pos_mlp(meta)
        # [B, M, P, C]

        # Add physical positional information before the encoder.
        tokens = tok_feat + tok_pos

        tokens = tokens.reshape(
            B,
            M * self.top_p,
            C,
        )

        return tokens

    def forward(self, y: torch.Tensor):
        """
        Args:
            y:
                Real input [B, 2, M, N] (I/Q channels).
        """

        B = y.shape[0]

        # CNN spectral features.
        x = self.make_fft_input(y)
        # [B, 3, M, F]

        feat = self.backbone(x)
        # [B, d_model, M, F]

        spectrum_logits = self.spectrum_head(feat)
        # [B, 1, M, F]

        # Sparse receiver-frequency tokens.
        tokens = self.select_top_tokens(
            feat,
            spectrum_logits,
        )
        # [B, M * top_p, d_model]

        # Encoder self-attention:
        # every selected receiver-frequency token can interact with every
        # other selected token.
        memory = self.encoder(tokens)
        # [B, M * top_p, d_model]

        # Learned object queries.
        queries = self.query_embed.weight.unsqueeze(0).expand(
            B,
            -1,
            -1,
        )
        # [B, num_queries, d_model]

        # Decoder:
        # 1. self-attention between object queries
        # 2. cross-attention from queries to encoded radar tokens
        hs = self.decoder(
            tgt=queries,
            memory=memory,
        )
        # [B, num_queries, d_model]

        pos = self.pos_head(hs)
        # [B, num_queries, 3]

        logits = self.obj_head(hs)
        # [B, num_queries, 2]

        return {
            "pos": pos,
            "logits": logits,
            "spectrum_logits": spectrum_logits,
        }