import os
import sys
from pathlib import Path

from torch.utils import data

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import least_squares
from torch.utils.data import DataLoader

from model.delay_net import DelayNet
from data_loaders.my_dataloader import RadarMatDataset


# ─────────────────────────────────────────────────────────────
# Radar geometry  (must match get_heatmap.m exactly)
# ─────────────────────────────────────────────────────────────

C      = 3e8          # speed of light [m/s]
M_RECV = 40           # number of receivers
_theta = 2 * np.pi * np.arange(M_RECV) / M_RECV
# Q: [3, M]  receiver positions (circle, radius 100 m, z=0)
Q = 100.0 * np.stack([np.cos(_theta), np.sin(_theta), np.zeros(M_RECV)], axis=0)
P_TX = np.zeros(3)    # transmitter at origin


# ─────────────────────────────────────────────────────────────
# Multilateration helpers
# ─────────────────────────────────────────────────────────────

def _tau_model(P: np.ndarray) -> np.ndarray:
    """Predicted tau for target at P given the fixed geometry."""
    d_tx = np.linalg.norm(P - P_TX)            # scalar
    d_rx = np.linalg.norm(Q.T - P, axis=1)     # [M]
    return (d_tx + d_rx) / C                    # [M]


def multilaterate(tau_meas: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
    """Nonlinear least-squares multilateration.  Returns [x, y, z] in metres."""
    if x0 is None:
        x0 = np.array([200.0, 200.0, 200.0])

    def residuals(P):
        return _tau_model(P) - tau_meas

    res = least_squares(residuals, x0, method="lm",
                        ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=2000)
    return res.x


# ─────────────────────────────────────────────────────────────
# Tau stats  (must match train_delay_net_2d.compute_tau_stats)
# ─────────────────────────────────────────────────────────────

def compute_tau_stats(dataset):
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)
    all_tau = []
    for _, _, _, tau, phi, snr in loader:
        all_tau.append(tau.float())
    all_tau  = torch.cat(all_tau, dim=0)
    tau_mean = all_tau.mean(dim=0)
    tau_std  = all_tau.std(dim=0)
    std_floor = torch.clamp(tau_std.mean() * 0.1, min=1e-6)
    tau_std   = tau_std.clamp(min=std_floor)
    return tau_mean, tau_std


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def main():
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    print(f"Using {'GPU: ' + torch.cuda.get_device_name(0) if use_cuda else 'CPU'}")

    # ── Dataset ──────────────────────────────────────────────
    train_dataset = RadarMatDataset(root_dir="D:\\radar-dataset-clean\\train")
    test_dataset   = RadarMatDataset(root_dir="D:\\radar-dataset-clean\\test")

    print("Computing tau normalisation stats from train set...")
    tau_mean, tau_std = compute_tau_stats(train_dataset)
    tau_mean = tau_mean.to(device)
    tau_std  = tau_std.to(device)
    M = tau_mean.numel()
    assert M == M_RECV, f"Model M={M} does not match geometry M={M_RECV}"

    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=use_cuda,
    )

    # ── Load model ───────────────────────────────────────────
    ckpt  = torch.load("delay_net_3D_high_noise.pt",
                       map_location=device, weights_only=True)
    model = DelayNet(M=40,
    Fs=5e7,
    a=1e13,
    nfft=1024,
    freq_side="negative",
    base_ch=64,
    beat_sign=-1.0,
    output_mode="normalized",
    tau_mean=tau_mean,
    tau_std=tau_std).to(device)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded DelayNet — saved at epoch {ckpt.get('epoch', '?')}")

    # ── Inference + multilateration ──────────────────────────
    pred_xyz_all  = []   # [N, 3]
    true_xyz_all  = []   # [N, 3]
    tau_error_all = []

    tau_mean_np = tau_mean.cpu().numpy()
    tau_std_np  = tau_std.cpu().numpy()
    n_samples   = 0
    tau_rmse = 0

    for batch_idx, (signal, _, coord_gt, tau_gt, phi, snr) in enumerate(test_loader):
        signal   = signal.to(device, non_blocking=True).float()
        coord_gt = coord_gt.cpu().numpy()[:, :3]   # [B, 3]
        tau_gt   = tau_gt.to(device, non_blocking=True).float()  # [B, M] physical seconds
        snr      = snr.to(device, non_blocking=True).float()     # [B, M] SNR values
        pred_tau_norm = model(signal).cpu().numpy()                       # [B, M]
        pred_tau_phys = pred_tau_norm * tau_std_np + tau_mean_np          # [B, M] seconds
        tau_rmse += np.sum((np.linalg.norm(pred_tau_phys - tau_gt.cpu().numpy(), axis=1))**2)  # [B] seconds
        tau_error_all.append((pred_tau_phys - tau_gt.cpu().numpy()) * 1e6)  # [B, M] in µs
        B = pred_tau_phys.shape[0]
        pred_xyz_batch = np.zeros((B, 3))

        for i in range(B):
            pred_xyz_batch[i] = multilaterate(pred_tau_phys[i])

        pred_xyz_all.append(pred_xyz_batch)
        true_xyz_all.append(coord_gt)
        n_samples += B

        if (batch_idx + 1) % 5 == 0:
            print(f"  processed {n_samples} samples...")

    pred_xyz = np.concatenate(pred_xyz_all, axis=0)   # [N, 3]
    true_xyz = np.concatenate(true_xyz_all, axis=0)   # [N, 3]
    tau_rmse = np.sqrt(tau_rmse / n_samples) * 1e6  # convert to microseconds
    print(f"\n── DelayNet evaluation on test set ({n_samples} samples) ──")
    print(f"  Tau RMSE: {tau_rmse:.3f} microseconds")

    # ── Metrics ──────────────────────────────────────────────
    err_xyz  = pred_xyz - true_xyz                                      # [N, 3]
    err_3d   = np.linalg.norm(err_xyz, axis=1)                          # [N] 3-D error
    err_2d   = np.linalg.norm(err_xyz[:, :2], axis=1)                   # [N] 2-D (XY)
    ##err_z    = np.abs(err_xyz[:, 2])                                     # [N] Z only

    print(f"\n── Multilateration evaluation on test set ({n_samples} samples) ──")

    # ── Tau error histogram ────────────────────────────────────
    tau_errors = np.concatenate(tau_error_all, axis=0).ravel()  # [N*M] in µs
    abs_errors = np.abs(tau_errors)

    print(f"\n── Tau error statistics (µs) ──")
    print(f"  Mean signed : {tau_errors.mean():.4f}")
    print(f"  Std         : {tau_errors.std():.4f}")
    print(f"  Mean abs    : {abs_errors.mean():.4f}")
    print(f"  Median abs  : {np.median(abs_errors):.4f}")
    print(f"  90th pct    : {np.percentile(abs_errors, 90):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Tau estimation errors — {n_samples} samples, {M_RECV} receivers each")

    # Signed error
    axes[0].hist(tau_errors, bins=100, edgecolor="none", alpha=0.8, color="steelblue")
    axes[0].axvline(tau_errors.mean(),     color="red",    linestyle="--",
                    label=f"mean {tau_errors.mean():.4f} µs")
    axes[0].axvline(np.median(tau_errors), color="orange", linestyle="--",
                    label=f"median {np.median(tau_errors):.4f} µs")
    axes[0].set_xlabel("τ error (µs)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Signed error")
    axes[0].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
