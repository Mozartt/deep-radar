import sys
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.delay_2_xyz_net import Delay2XYZNet
from model.delay_net import DelayNet
from data_loaders.my_dataloader import RadarMatDataset

Fs = 5e7
a = 1e13
nfft = 1024
Tc = 20e-6
M_RECV = 40           # number of receivers
_theta = 2 * np.pi * np.arange(M_RECV) / M_RECV
# Q: [3, M]  receiver positions (circle, radius 100 m, z=0)
Q = 100.0 * np.stack([np.cos(_theta), np.sin(_theta), np.zeros(M_RECV)], axis=0)
P_TX = np.zeros(3)    # transmitter at origin

def matched_filter_refinment(y_ell,
    p_hat,
    q,
    p_tx,
    fc,
    a,
    Ts,
    Tc,
    cube_side=2.0,
    resolution=0.5,
    c=3e8,
    normalize=True,):
    """
    Local matched-filter refinement around an initial position estimate.

    Parameters
    ----------
    y_ell : np.ndarray
        Observed signal, shape [M, N], complex.
        If shape is [2, M, N], it is interpreted as [real, imag].
    p_hat : np.ndarray
        Initial position estimate, shape [3].
    q : np.ndarray
        Receiver positions, shape [M, 3].
    p_tx : np.ndarray
        Transmitter position, shape [3].
    fc : float
        Carrier frequency.
    a : float
        Chirp slope.
    Ts : float
        Sampling period.
    Tc : float
        Chirp duration.
    cube_side : float
        Side length of local search cube in meters.
        Default 2.0 means search p_hat +/- 1 meter.
    resolution : float
        Grid resolution in meters.
    c : float
        Speed of light.
    normalize : bool
        If True, use normalized matched-filter score.

    Returns
    -------
    max_val : float
        Maximal matched-filter score.
    p_refined : np.ndarray
        Refined position estimate, shape [3].
    score_cube : np.ndarray
        Matched-filter scores over the local search grid.
    grid_axes : tuple
        The x, y, z searched values.
    """

    y_ell = np.asarray(y_ell)
    y_ell = y_ell[0] + 1j * y_ell[1]
    y_ell = y_ell.astype(np.complex128)

    p_hat = np.asarray(p_hat, dtype=float).reshape(3)
    p_tx = np.asarray(p_tx, dtype=float).reshape(3)
    q = np.asarray(q, dtype=float)

    M, N = y_ell.shape
    n = np.arange(N)
    t = Ts * n  # [N]

    half_side = cube_side / 2.0

    x_vals = np.arange(p_hat[0] - half_side, p_hat[0] + half_side + 1e-9, resolution)
    y_vals = np.arange(p_hat[1] - half_side, p_hat[1] + half_side + 1e-9, resolution)
    z_vals = np.arange(p_hat[2] - half_side, p_hat[2] + half_side + 1e-9, resolution)

    score_cube = np.zeros((len(x_vals), len(y_vals), len(z_vals)), dtype=float)

    max_val = -np.inf
    p_refined = p_hat.copy()

    for ix, x in enumerate(x_vals):
        for iy, y in enumerate(y_vals):
            for iz, z in enumerate(z_vals):

                p = np.array([x, y, z], dtype=float)

                tau_tx = np.linalg.norm(p - p_tx) / c
                tau_rx = np.linalg.norm(q - p[None, :], axis=1) / c
                tau = tau_tx + tau_rx  # [M]

                # beta_m(p)
                beta = (
                    np.exp(-1j * 2 * np.pi * fc * tau)
                    * np.exp(1j * np.pi * a * tau**2)
                )  # [M]

                phase = np.exp(
                    -1j * 2 * np.pi * a * tau[:, None] * t[None, :]
                )  # [M, N]

                win = (t[None, :] >= tau[:, None]) & (t[None, :] <= Tc)  # [M, N]

                s = beta[:, None] * phase * win  # [M, N]

                mf = np.sum(np.conj(s) * y_ell)

                if normalize:
                    energy = np.sum(np.abs(s) ** 2)
                    if energy > 0:
                        score = np.abs(mf) ** 2 / energy
                    else:
                        score = 0.0
                else:
                    score = np.abs(mf) ** 2

                score_cube[ix, iy, iz] = score

                if score > max_val:
                    max_val = score
                    p_refined = p.copy()

    grid_axes = (x_vals, y_vals, z_vals)
    return max_val, p_refined, score_cube, grid_axes

def compute_tau_stats(dataset):
    """Matches train_delay_net_2d (std_floor applied)."""
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)
    all_tau = []
    for _, _, _, tau, _,_ in loader:
        all_tau.append(tau.float())
    all_tau = torch.cat(all_tau, dim=0)   # [N, M]
    tau_mean = all_tau.mean(dim=0)
    tau_std  = all_tau.std(dim=0)
    std_floor = torch.clamp(tau_std.mean() * 0.1, min=1e-6)
    tau_std   = tau_std.clamp(min=std_floor)
    return tau_mean, tau_std


def compute_coord_stats(dataset):
    """Matches train_delay_2_pred (std_floor, same as compute_tau_stats)."""
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)
    all_coord = []
    for _, _, coord, _, _,_ in loader:
        all_coord.append(coord.float()[..., :3])
    all_coord = torch.cat(all_coord, dim=0)  # [N, 3]
    return (
        all_coord.mean(dim=0), all_coord.std(dim=0).clamp(min=1e-8),
    )



@torch.no_grad()
def main():
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    gpu_label = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    print(f"Using {gpu_label}")

    # ── Dataset ──────────────────────────────────────────────
    train_dataset = RadarMatDataset(root_dir="D:\\radar-dataset-clean\\train")
    test_dataset   = RadarMatDataset(root_dir="D:\\radar-dataset-clean\\test")

    print("Computing normalisation stats from train set...")
    tau_mean_1, tau_std_1 = compute_tau_stats(train_dataset)       # delay_net stats (std_floor)
    coord_mean, coord_std = compute_coord_stats(train_dataset)  # coord_net stats

    tau_mean_1 = tau_mean_1.to(device)
    tau_std_1  = tau_std_1.to(device)
    coord_mean = coord_mean.to(device)
    coord_std  = coord_std.to(device)

    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=use_cuda,
    )

    # ── Load models ──────────────────────────────────────────
    ckpt_tau   = torch.load("delay_net_3D_high_noise.pt",  map_location=device, weights_only=True)
    ckpt_coord = torch.load("delay_2_xyz.pt",  map_location=device, weights_only=True)

    delay_net = DelayNet(M=40,
    Fs=5e7,
    a=1e13,
    nfft=1024,
    freq_side="negative",
    base_ch=64,
    beat_sign=-1.0,
    output_mode="normalized",
    tau_mean=tau_mean_1,
    tau_std=tau_std_1).to(device)

    delay_net.load_state_dict(ckpt_tau["model_state_dict"])
    delay_net.eval()

    coord_net = Delay2XYZNet(M=40, hidden_dim=512).to(device)
    coord_net.load_state_dict(ckpt_coord["model_state_dict"])
    coord_net.eval()

    errors_3d = []
    snr_all   = []
    error_rmse = 0.0

    for signal, _, coord_gt, tau_gt, phi_gt, snr in test_loader:
        signal   = signal.to(device, non_blocking=True).float()
        coord_gt = coord_gt.to(device, non_blocking=True).float()[..., :3]

        # Stage 1 — signal → normalised tau
        pred_tau = delay_net(signal)  
        pred_tau_phy = pred_tau * tau_std_1 + tau_mean_1                             # [B, M]

        #stage 2 - initial tau -> normalised coord
        coord_pred_norm = coord_net(pred_tau)                            # [B, 3]
        pred_coord = coord_pred_norm * coord_std + coord_mean

        # for b in range(signal.shape[0]):

        #     p_refined = matched_filter_refinment(
        #         y_ell=signal[b].cpu().numpy(),
        #         p_hat=pred_coord[b].cpu().numpy(),
        #         q=Q.T,
        #         p_tx=P_TX,
        #         fc=Fs / 4,
        #         a=a,
        #         Ts=1 / Fs,
        #         Tc=Tc,
        #         cube_side=2.0,
        #         resolution=0.5,
        #         c=3e8,
        #         normalize=True,
        #     )[1]

        error = torch.linalg.vector_norm(pred_coord - coord_gt, dim=1)
        errors_3d.extend(error.cpu().numpy())
        snr_all.extend(snr.cpu().numpy())
        error_rmse += torch.mean(error ** 2).item()

    errors_3d = np.array(errors_3d)
    snr_all   = np.array(snr_all)
    error_rmse = np.sqrt(error_rmse / len(errors_3d))

    high_snr_mask = snr_all > 5.0
    err_high = errors_3d[high_snr_mask]
    err_low  = errors_3d[~high_snr_mask]

    print(f"\n── 3D position error ({len(errors_3d)} samples) ──")
    print(f"  Mean   : {errors_3d.mean():.3f} m")
    print(f"  Median : {np.median(errors_3d):.3f} m")
    print(f"  90th   : {np.percentile(errors_3d, 90):.3f} m")
    print(f"  Max    : {errors_3d.max():.3f} m")
    print(f"  RMSE   : {error_rmse:.3f} m")

    print(f"\n── SNR > 5 dB ({len(err_high)} samples) ──")
    if len(err_high):
        print(f"  Mean   : {err_high.mean():.3f} m")
        print(f"  Median : {np.median(err_high):.3f} m")
        print(f"  90th   : {np.percentile(err_high, 90):.3f} m")

    print(f"\n── SNR ≤ 5 dB ({len(err_low)} samples) ──")
    if len(err_low):
        print(f"  Mean   : {err_low.mean():.3f} m")
        print(f"  Median : {np.median(err_low):.3f} m")
        print(f"  90th   : {np.percentile(err_low, 90):.3f} m")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"3D position error — all samples ({len(errors_3d)})")

    # Histogram
    axes[0].hist(errors_3d, bins=80, edgecolor="none", alpha=0.85, color="steelblue")
    axes[0].axvline(errors_3d.mean(),              color="red",    linestyle="--",
                    label=f"mean {errors_3d.mean():.2f} m")
    axes[0].axvline(np.median(errors_3d),          color="orange", linestyle="--",
                    label=f"median {np.median(errors_3d):.2f} m")
    # axes[0].axvline(np.percentile(errors_3d, 90),  color="purple", linestyle=":",
    #                 label=f"90th pct {np.percentile(errors_3d, 90):.2f} m")
    axes[0].set_xlabel("3D error (m)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Euclidean error histogram")
    axes[0].legend()

    # CDF
    sorted_err = np.sort(errors_3d)
    axes[1].plot(sorted_err, np.linspace(0, 1, len(sorted_err)), color="steelblue")
    axes[1].axhline(0.5,  color="orange", linestyle="--", label=f"50th: {np.median(errors_3d):.2f} m")
    axes[1].axhline(0.9,  color="purple", linestyle=":",  label=f"90th: {np.percentile(errors_3d, 90):.2f} m")
    axes[1].set_xlabel("3D error (m)")
    axes[1].set_ylabel("CDF")
    axes[1].set_title("Cumulative distribution (all)")
    axes[1].grid(True, alpha=0.4)
    axes[1].legend()

    plt.tight_layout()

    # ── SNR split figure ──────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle("3D position error split by SNR (threshold = 5 dB)")

    def _plot_group(ax, data, label, color):
        if len(data) == 0:
            ax.text(0.5, 0.5, "no samples", ha="center", va="center", transform=ax.transAxes)
            return
        ax.hist(data, bins=60, edgecolor="none", alpha=0.75, color=color, label=label)
        ax.axvline(data.mean(),              color="red",    linestyle="--",
                   label=f"mean {data.mean():.2f} m")
        ax.axvline(np.median(data),          color="black",  linestyle="--",
                   label=f"median {np.median(data):.2f} m")
        ax.axvline(np.percentile(data, 90),  color="purple", linestyle=":",
                   label=f"90th {np.percentile(data, 90):.2f} m")
        ax.set_xlabel("3D error (m)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    _plot_group(axes2[0], err_high, f"SNR > 5 dB  (n={len(err_high)})",  color="steelblue")
    axes2[0].set_title(f"SNR > 5 dB  (n={len(err_high)})")

    _plot_group(axes2[1], err_low,  f"SNR ≤ 5 dB  (n={len(err_low)})",   color="darkorange")
    axes2[1].set_title(f"SNR ≤ 5 dB  (n={len(err_low)})")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
        
        