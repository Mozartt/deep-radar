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
        x0 = np.array([500.0, 500.0, 500.0])

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
    for _, _, _, tau, phi in loader:
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
    train_dataset = RadarMatDataset(root_dir="D:\\radar-dataset-noisy\\train")
    test_dataset   = RadarMatDataset(root_dir="D:\\radar-dataset-noisy\\test")

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
    ckpt  = torch.load("delay_net_noisy.pt",
                       map_location=device, weights_only=True)
    model = DelayNet(M=M).to(device)
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

    for batch_idx, (signal, _, coord_gt, tau_gt, phi) in enumerate(test_loader):
        signal   = signal.to(device, non_blocking=True).float()
        coord_gt = coord_gt.cpu().numpy()[:, :3]   # [B, 3]
        tau_gt   = tau_gt.to(device, non_blocking=True).float()  # [B, M] physical seconds
        
        pred_tau_norm = model(signal).cpu().numpy()                       # [B, M]
        pred_tau_phys = pred_tau_norm * tau_std_np + tau_mean_np          # [B, M] seconds
        tau_error = pred_tau_phys - tau_gt.cpu().numpy()                                  # [B, M] seconds
        B = pred_tau_phys.shape[0]
        pred_xyz_batch = np.zeros((B, 3))

        for i in range(B):
            pred_xyz_batch[i] = multilaterate(pred_tau_phys[i])

        pred_xyz_all.append(pred_xyz_batch)
        true_xyz_all.append(coord_gt)
        tau_error_all.append(tau_error)
        n_samples += B

        if (batch_idx + 1) % 5 == 0:
            print(f"  processed {n_samples} samples...")

    pred_xyz = np.concatenate(pred_xyz_all, axis=0)   # [N, 3]
    true_xyz = np.concatenate(true_xyz_all, axis=0)   # [N, 3]
    tau_error = np.concatenate(tau_error_all, axis=0) # [N, M]

    # ── Metrics ──────────────────────────────────────────────
    err_xyz  = pred_xyz - true_xyz                                      # [N, 3]
    err_3d   = np.linalg.norm(err_xyz, axis=1)                          # [N] 3-D error
    err_2d   = np.linalg.norm(err_xyz[:, :2], axis=1)                   # [N] 2-D (XY)
    err_z    = np.abs(err_xyz[:, 2])                                     # [N] Z only

    print(f"\n── Multilateration evaluation on test set ({n_samples} samples) ──")
    print(f"  3-D error  — mean:   {err_3d.mean():.2f} m   "
          f"median: {np.median(err_3d):.2f} m   "
          f"90th: {np.percentile(err_3d, 90):.2f} m   "
          f"max: {err_3d.max():.2f} m")
    print(f"  XY error   — mean:   {err_2d.mean():.2f} m   "
          f"median: {np.median(err_2d):.2f} m")
    print(f"  Z  error   — mean:   {err_z.mean():.2f} m   "
          f"median: {np.median(err_z):.2f} m")
    print(f"\n  Per-axis MAE:")
    print(f"    X: {np.abs(err_xyz[:, 0]).mean():.2f} m")
    print(f"    Y: {np.abs(err_xyz[:, 1]).mean():.2f} m")
    print(f"    Z: {np.abs(err_xyz[:, 2]).mean():.2f} m")
    print(f"\n  Tau error (physical) — mean: {tau_error.mean()*1e6:.4f} µs   "
          f"median: {np.median(tau_error)*1e6:.4f} µs   "
          f"90th: {np.percentile(tau_error, 90)*1e6:.4f} µs   "
          f"max: {tau_error.max()*1e6:.4f} µs       ")

    # ── Plots ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    def _hist(ax, data, label, unit="m"):
        ax.hist(data, bins=60, edgecolor="black")
        ax.axvline(data.mean(),         color="red",    linestyle="--",
                   label=f"mean {data.mean():.2f} {unit}")
        ax.axvline(np.median(data),     color="orange", linestyle="--",
                   label=f"median {np.median(data):.2f} {unit}")
        ax.set_xlabel(f"{label} ({unit})")
        ax.set_ylabel("Count")
        ax.set_title(f"{label} distribution")
        ax.legend(fontsize=8)

    _hist(axes[0, 0], err_3d, "3-D error")
    _hist(axes[0, 1], err_2d, "XY error")
    _hist(axes[0, 2], err_z,  "Z error")

    # CDF of 3-D error
    sorted_3d = np.sort(err_3d)
    axes[1, 0].plot(sorted_3d, np.linspace(0, 1, len(sorted_3d)))
    axes[1, 0].set_xlabel("3-D error (m)")
    axes[1, 0].set_ylabel("CDF")
    axes[1, 0].set_title("Cumulative 3-D error")
    axes[1, 0].grid(True)

    # Predicted vs true scatter (XY plane)
    sc = axes[1, 1].scatter(true_xyz[:, 0], true_xyz[:, 1],
                            c=err_2d, cmap="viridis", s=4, alpha=0.6)
    plt.colorbar(sc, ax=axes[1, 1], label="XY error (m)")
    axes[1, 1].set_xlabel("X true (m)")
    axes[1, 1].set_ylabel("Y true (m)")
    axes[1, 1].set_title("XY error map")
    axes[1, 1].set_aspect("equal")

    # Z prediction vs true
    axes[1, 2].scatter(true_xyz[:, 2], pred_xyz[:, 2], s=4, alpha=0.5)
    z_lo = min(true_xyz[:, 2].min(), pred_xyz[:, 2].min())
    z_hi = max(true_xyz[:, 2].max(), pred_xyz[:, 2].max())
    axes[1, 2].plot([z_lo, z_hi], [z_lo, z_hi], "r--", label="ideal")
    axes[1, 2].set_xlabel("Z true (m)")
    axes[1, 2].set_ylabel("Z predicted (m)")
    axes[1, 2].set_title("Z prediction vs truth")
    axes[1, 2].legend()

    plt.tight_layout()
    out_path = Path(__file__).parent / "eval_delay_net_multilateration.png"
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
