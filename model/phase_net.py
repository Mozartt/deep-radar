import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseOnlyNet(nn.Module):
    """
    Predicts per-receiver ABSOLUTE carrier phase: (cos φ_m, sin φ_m).

    Architecture
    ------------
    1. FFT of the complex IQ signal; keep the delay half of the spectrum.
    2. Magnitude-weighted soft attention over frequency bins — directly reads
       the unit-phasor (Z_unit) at the dominant (delay-peak) bin without any
       MaxPool that would corrupt the I/Q phase relationship.
    3. Lightweight Conv1d head mixes neighbouring receivers.

    Why absolute phase, not relative?
    -----------------------------------
    The relative target δφ_m = φ_m − φ_0 requires every receiver to "see"
    receiver 0's phase.  With M=40 and a 3-wide conv kernel the receptive
    field is only 13 receivers — receivers 14-39 never receive receiver 0's
    gradient, so the model collapses to predicting (1, 0) for everyone.

    Predicting absolute phase is a *local* task: at the FFT peak bin for
    receiver m, Z_unit already equals exp(j φ_m), so the network learns a
    near-identity readout.  Relative phase is recovered in post-processing:
        cos δφ_m = cos φ_m · cos φ_0 + sin φ_m · sin φ_0
        sin δφ_m = sin φ_m · cos φ_0 − cos φ_m · sin φ_0
    """

    def __init__(
        self,
        M: int = 40,
        hidden_ch: int = 128,
        n_fft: int = 1024,
    ):
        super().__init__()

        self.M = M
        self.n_fft = n_fft

        # Learnable attention temperature; exp(2) ≈ 7.4 sharpens the
        # softmax toward the peak bin from the very first iteration.
        self.log_temp = nn.Parameter(torch.tensor(2.0))

        # Per-receiver refinement head (Conv1d → mixes only local neighbours,
        # which is enough since the task is per-receiver local phase readout).
        self.phase_head = nn.Sequential(
            nn.Conv1d(4, hidden_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(hidden_ch, 2, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, 2, M, N]  — channel 0 = I, channel 1 = Q
        z = torch.complex(x[:, 0].float(), x[:, 1].float())  # [B, M, N]

        Z = torch.fft.fft(z, n=self.n_fft, dim=-1, norm="forward")
        Z = torch.fft.fftshift(Z, dim=-1)
        Z = Z[:, :, : self.n_fft // 2]          # delay half → [B, M, 512]

        mag     = Z.abs()                        # [B, M, 512]
        log_mag = torch.log1p(mag)               # [B, M, 512]
        Z_unit  = Z / (mag + 1e-8)              # [B, M, 512]  unit phasor

        # ── Magnitude-weighted soft attention over frequency ──────────────
        # Higher temperature → sharper peak around the dominant delay bin.
        # No MaxPool: the attention is applied directly to Z_unit so the
        # I/Q phase relationship is never corrupted.
        temp = self.log_temp.exp()
        attn = torch.softmax(log_mag * temp, dim=-1)  # [B, M, 512]

        # Soft readout of the unit phasor at the dominant bin → ≈ exp(jφ_m)
        cos_raw     = (Z_unit.real.to(x.dtype) * attn).sum(-1)  # [B, M]
        sin_raw     = (Z_unit.imag.to(x.dtype) * attn).sum(-1)  # [B, M]

        # Auxiliary pooled features give the head some amplitude context.
        log_mag_p   = (log_mag.to(x.dtype) * attn).sum(-1)      # [B, M]
        z_real_p    = (Z.real.to(x.dtype)   * attn).sum(-1)     # [B, M]

        # ── Receiver-mixing head ─────────────────────────────────────────
        feat  = torch.stack([cos_raw, sin_raw, log_mag_p, z_real_p], dim=1)
        # feat: [B, 4, M]

        phase = self.phase_head(feat)               # [B, 2, M]
        phase = F.normalize(phase, p=2, dim=1, eps=1e-8)

        return phase[:, 0, :], phase[:, 1, :]      # cos φ_m, sin φ_m