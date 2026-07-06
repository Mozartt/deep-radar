import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def _num_groups(ch):
    for g in [8, 4, 2, 1]:
        if ch % g == 0:
            return g
    return 1


class SamePadConv2d(nn.Module):
    """
    Same-padding Conv2d.
    Receiver axis padding can be circular, which makes sense if receivers lie on a circle.
    Frequency axis is not circular.
    """
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=(3, 3),
        dilation=(1, 1),
        bias=False,
        circular_rx=True,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.circular_rx = circular_rx

        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=bias,
        )

    def forward(self, x):
        kh, kw = self.kernel_size
        dh, dw = self.dilation

        pad_h = dh * (kh - 1) // 2
        pad_w = dw * (kw - 1) // 2

        # Frequency-axis padding: do not wrap FFT bins.
        if pad_w > 0:
            x = F.pad(x, (pad_w, pad_w, 0, 0), mode="replicate")

        # Receiver-axis padding: circular because receiver 0 and receiver M-1 are neighbors.
        if pad_h > 0:
            mode = "circular" if self.circular_rx else "replicate"
            x = F.pad(x, (0, 0, pad_h, pad_h), mode=mode)

        return self.conv(x)


class ResBlock2D(nn.Module):
    def __init__(
        self,
        ch,
        kernel_size=(3, 5),
        dilation=(1, 1),
        circular_rx=True,
        dropout=0.0,
    ):
        super().__init__()

        self.net = nn.Sequential(
            SamePadConv2d(
                ch,
                ch,
                kernel_size=kernel_size,
                dilation=dilation,
                circular_rx=circular_rx,
                bias=False,
            ),
            nn.GroupNorm(_num_groups(ch), ch),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),

            SamePadConv2d(
                ch,
                ch,
                kernel_size=kernel_size,
                dilation=dilation,
                circular_rx=circular_rx,
                bias=False,
            ),
            nn.GroupNorm(_num_groups(ch), ch),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class TauReadout(nn.Module):
    """
    Dense readout that consumes the *full* per-bin information (logits, sub-bin
    offsets, and the softmax distribution) per receiver, instead of only the
    scalar produced by collapsing prob into a weighted mean.
 
    It outputs:
        tau_delta_sec : [B, M]  a bounded correction (in seconds) to the
                        closed-form soft-argmax estimate.
        fusion_feat   : [B, M, hidden]  a per-receiver dense feature vector
                        meant to be handed to a downstream cross-receiver
                        fusion stage (e.g. a transformer) INSTEAD of the
                        scalar tau. This is the thing that used to get
                        thrown away.
 
    The correction is bounded with tanh * max_correction_sec so training
    starts close to the physically-grounded soft-argmax answer and only
    learns to move away from it where the data actually supports doing so
    (e.g. clutter, a second nearby peak, low SNR).
    """
 
    def __init__(self, K, hidden=256, max_correction_bins=2.0, df=1.0, a=1.0,
                 beat_sign=-1.0, dropout=0.1):
        super().__init__()
 
        in_dim = K * 3  # logits, offsets, prob concatenated along frequency axis
 
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.out = nn.Linear(hidden, 1)
 
        # Bound the learned correction to a few frequency bins' worth of tau,
        # so it acts as a refinement rather than something that can run away
        # during early training.
        max_correction_hz = max_correction_bins * df
        self.max_correction_sec = max_correction_hz / (beat_sign * a) \
            if beat_sign * a != 0 else max_correction_hz
 
    def forward(self, logits, offsets, prob):
        # logits, offsets, prob: each [B, M, K]
        feat_in = torch.cat([logits, offsets, prob], dim=-1)  # [B, M, 3K]
 
        h = self.net(feat_in)                    # [B, M, hidden]  <- fusion_feat
        raw = self.out(h).squeeze(-1)             # [B, M]
        tau_delta_sec = torch.tanh(raw) * abs(self.max_correction_sec)
 
        return tau_delta_sec

class DelayNet(nn.Module):
    """
    Input:
        x: [B, 2, M, N]
           channel 0 = I
           channel 1 = Q

    Output:
        tau: [B, M]
             either normalized tau or tau in seconds, depending on output_mode.

    Important:
        For your signal model y[n] = beta * exp(-j 2*pi*a*tau*Ts*n),
        the beat frequency is negative:

            f_b = -a * tau

        Therefore:
            tau = -f_b / a

        This is controlled by beat_sign = -1.
    """

    def __init__(
        self,
        M=40,
        Fs=5e7,
        a=1e13,
        nfft=1024,
        freq_side="negative",   # "negative", "positive", or "both"
        base_ch=64,
        beat_sign=-1.0,
        output_mode="normalized",  # "normalized" or "seconds"
        tau_mean=None,
        tau_std=None,
        temperature=1.0,
        dropout=0.1,
        readout_hidden=256,
        max_correction_bins=2.0
    ):
        super().__init__()

        assert freq_side in ["negative", "positive", "both"]
        assert output_mode in ["normalized", "seconds"]

        self.M = M
        self.Fs = float(Fs)
        self.a = float(a)
        self.nfft = int(nfft)
        self.freq_side = freq_side
        self.beat_sign = float(beat_sign)
        self.output_mode = output_mode
        self.temperature = float(temperature)

        if output_mode == "normalized":
            if tau_mean is None or tau_std is None:
                raise ValueError(
                    "For output_mode='normalized', pass tau_mean and tau_std."
                )

        # Frequency grid after fftshift.
        freq_grid = torch.fft.fftshift(
            torch.fft.fftfreq(self.nfft, d=1.0 / self.Fs)
        )

        if freq_side == "negative":
            k0, k1 = 0, self.nfft // 2
        elif freq_side == "positive":
            k0, k1 = self.nfft // 2, self.nfft
        else:
            k0, k1 = 0, self.nfft

        self.k0 = k0
        self.k1 = k1

        freq_grid = freq_grid[k0:k1]
        self.K = freq_grid.numel()

        self.register_buffer("freq_grid", freq_grid.float())

        # Normalized frequency coordinate, useful for giving the CNN absolute bin position.
        freq_coord = freq_grid / (self.Fs / 2.0)
        self.register_buffer("freq_coord", freq_coord.float())

        if tau_mean is None:
            tau_mean = torch.tensor(0.0)
        else:
            tau_mean = torch.as_tensor(tau_mean, dtype=torch.float32)

        if tau_std is None:
            tau_std = torch.tensor(1.0)
        else:
            tau_std = torch.as_tensor(tau_std, dtype=torch.float32)

        if tau_mean.ndim == 0:
            tau_mean = tau_mean.view(1, 1)
        else:
            tau_mean = tau_mean.view(1, -1)

        if tau_std.ndim == 0:
            tau_std = tau_std.view(1, 1)
        else:
            tau_std = tau_std.view(1, -1)

        self.register_buffer("tau_mean", tau_mean)
        self.register_buffer("tau_std", tau_std)

        # Input channels:
        # 1. normalized FFT real
        # 2. normalized FFT imag
        # 3. log magnitude
        # 4. frequency coordinate
        in_ch = 4

        self.stem = nn.Sequential(
            SamePadConv2d(
                in_ch,
                base_ch,
                kernel_size=(3, 9),
                circular_rx=True,
                bias=False,
            ),
            nn.GroupNorm(_num_groups(base_ch), base_ch),
            nn.GELU(),
        )

        self.encoder = nn.Sequential(
            # Spectral processing: learn local peak shapes.
            ResBlock2D(base_ch, kernel_size=(1, 9), dilation=(1, 1), dropout=dropout),
            ResBlock2D(base_ch, kernel_size=(1, 9), dilation=(1, 2), dropout=dropout),
            ResBlock2D(base_ch, kernel_size=(1, 9), dilation=(1, 4), dropout=dropout),

            # Wider dilation to extend frequency-axis receptive field toward
            # the full spectrum, so the network can reason about competing
            # peaks/noise far away from the current bin (important at low SNR).
            ResBlock2D(base_ch, kernel_size=(1, 9), dilation=(1, 8), dropout=dropout),
            ResBlock2D(base_ch, kernel_size=(1, 9), dilation=(1, 16), dropout=dropout),

            # Joint receiver-frequency processing.
            ResBlock2D(base_ch, kernel_size=(3, 5), dilation=(1, 2), dropout=dropout),
            ResBlock2D(base_ch, kernel_size=(3, 5), dilation=(1, 4), dropout=dropout),

            # Receiver-context mixing.
            ResBlock2D(base_ch, kernel_size=(5, 1), dilation=(1, 1), dropout=dropout),
            ResBlock2D(base_ch, kernel_size=(5, 1), dilation=(1, 1), dropout=dropout),
        )

        # Per-bin score: [B, 1, M, K]
        self.bin_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(base_ch // 2, 1, kernel_size=1),
        )

        # Sub-bin correction in units of half a frequency bin.
        self.offset_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(base_ch // 2, 1, kernel_size=1),
        )

        # df = self.Fs / self.nfft
        
        # self.tau_readout = TauReadout(
        #     K=self.K,
        #     hidden=readout_hidden,
        #     max_correction_bins=max_correction_bins,
        #     df=df,
        #     a=self.a,
        #     beat_sign=self.beat_sign,
        #     dropout=dropout,
        # )
 

    def make_fft_features(self, x):
        """
        x: [B, 2, M, N]
        returns: [B, 4, M, K]
        """
        B, _, M, N = x.shape

        z = torch.complex(
            x[:, 0, :, :].float(),
            x[:, 1, :, :].float(),
        )  # [B, M, N]

        Z = torch.fft.fft(z, n=self.nfft, dim=-1, norm="forward")
        Z = torch.fft.fftshift(Z, dim=-1)
        Z = Z[:, :, self.k0:self.k1]  # [B, M, K]

        mag = Z.abs()

        # Per receiver normalization.
        # This prevents the network from being dominated by absolute noise scale.
        rms = torch.sqrt(torch.mean(mag ** 2, dim=-1, keepdim=True) + 1e-12)
        Zn = Z / rms

        logmag = torch.log1p(mag / rms)

        freq_coord = self.freq_coord.view(1, 1, self.K).expand(B, M, self.K)

        feat = torch.stack(
            [
                Zn.real,
                Zn.imag,
                logmag,
                freq_coord,
            ],
            dim=1,
        )  # [B, 4, M, K]

        return feat.to(x.dtype)

    def forward(self, x, return_aux=False):
        feat = self.make_fft_features(x)

        h = self.stem(feat)
        h = self.encoder(h)

        logits = self.bin_head(h).squeeze(1)       # [B, M, K]
        offsets = self.offset_head(h).squeeze(1)   # [B, M, K]
        offsets = 0.5 * torch.tanh(offsets)        # [-0.5, 0.5] bin correction

        prob = F.softmax(logits / self.temperature, dim=-1)

        df = self.Fs / self.nfft

        freq_grid = self.freq_grid.view(1, 1, self.K)
        freq_hat = torch.sum(prob * (freq_grid + offsets * df), dim=-1)  # [B, M]

        tau_sec = freq_hat / (self.beat_sign * self.a)

        if self.output_mode == "normalized":
            tau_out = (tau_sec - self.tau_mean) / self.tau_std
        else:
            tau_out = tau_sec

        if return_aux:
            aux = {
                "logits": logits,
                "prob": prob,
                "freq_hat": freq_hat,
                "tau_sec": tau_sec,
            }
            return tau_out, aux 
    
        return tau_out