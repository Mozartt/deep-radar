import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from model.delay_net import DelayNet
from data_loaders.my_dataloader import RadarMatDataset


def compute_tau_stats(dataset):
    """Matches train_delay_net_2d (std_floor applied)."""
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)
    all_tau = []
    for _, _, _, tau in loader:
        all_tau.append(tau.float())
    all_tau = torch.cat(all_tau, dim=0)  # [N, M]
    tau_mean = all_tau.mean(dim=0)
    tau_std  = all_tau.std(dim=0)
    std_floor = torch.clamp(tau_std.mean() * 0.1, min=1e-6)
    tau_std   = tau_std.clamp(min=std_floor)
    return tau_mean, tau_std


@torch.no_grad()
def main():
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    print(f"Using {'GPU: ' + torch.cuda.get_device_name(0) if use_cuda else 'CPU'}")

    # ── Dataset ──────────────────────────────────────────────
    train_dataset = RadarMatDataset(root_dir="D:\\radar-dataset-delay-only\\train")
    val_dataset   = RadarMatDataset(root_dir="D:\\radar-dataset-delay-only\\validation")

    print("Computing tau normalisation stats from train set...")
    tau_mean, tau_std = compute_tau_stats(train_dataset)
    tau_mean = tau_mean.to(device)
    tau_std  = tau_std.to(device)
    M = tau_mean.numel()
    print(f"  M={M}  tau_mean(avg)={tau_mean.mean().item()*1e6:.4f} µs  "
          f"tau_std(avg)={tau_std.mean().item()*1e6:.4f} µs")

    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=use_cuda,
    )

    # ── Model ────────────────────────────────────────────────
    ckpt = torch.load("best_radar_model_samples_2_tau.pt",
                      map_location=device, weights_only=True)
    model = DelayNet(M=M).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded DelayNet — saved at epoch {ckpt.get('epoch', '?')}")

    # ── Inference ────────────────────────────────────────────
    # Per-receiver absolute errors in µs, per-sample L2 norm in µs
    per_receiver_abs_us = []   # [N, M]
    per_sample_l2_us    = []   # [N]

    for signal, _, _, tau_gt in val_loader:
        signal = signal.to(device, non_blocking=True).float()
        tau_gt = tau_gt.to(device, non_blocking=True).float()  # [B, M] physical

        pred_tau_norm = model(signal)                          # [B, M] normalised
        pred_tau = pred_tau_norm * tau_std + tau_mean          # [B, M] physical

        err_us = (pred_tau - tau_gt).abs() * 1e6               # [B, M] µs
        l2_us  = torch.linalg.vector_norm(
                     (pred_tau - tau_gt) * 1e6, dim=1)         # [B] µs

        per_receiver_abs_us.append(err_us.cpu())
        per_sample_l2_us.append(l2_us.cpu())

    per_receiver_abs_us = torch.cat(per_receiver_abs_us, dim=0).numpy()  # [N, M]
    per_sample_l2_us    = torch.cat(per_sample_l2_us).numpy()            # [N]

    # ── Metrics ──────────────────────────────────────────────
    mae_per_receiver = per_receiver_abs_us.mean(axis=0)   # [M]
    overall_mae      = per_receiver_abs_us.mean()

    print(f"\n── DelayNet evaluation on validation set ({len(per_sample_l2_us)} samples) ──")
    print(f"  Overall MAE         : {overall_mae:.4f} µs")
    print(f"  Per-sample L2 norm  — mean : {per_sample_l2_us.mean():.4f} µs")
    print(f"  Per-sample L2 norm  — median: {np.median(per_sample_l2_us):.4f} µs")
    print(f"  Per-sample L2 norm  — 90th pct: {np.percentile(per_sample_l2_us, 90):.4f} µs")
    print(f"  Per-sample L2 norm  — max : {per_sample_l2_us.max():.4f} µs")
    print(f"\n  Per-receiver MAE (µs):")
    for i, v in enumerate(mae_per_receiver):
        print(f"    receiver {i:02d}: {v:.4f} µs")

    # ── Plots ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # 1 — Histogram of per-sample L2 norm
    axes[0].hist(per_sample_l2_us, bins=60, edgecolor="black")
    axes[0].axvline(per_sample_l2_us.mean(),        color="red",    linestyle="--",
                    label=f"mean {per_sample_l2_us.mean():.3f} µs")
    axes[0].axvline(np.median(per_sample_l2_us),    color="orange", linestyle="--",
                    label=f"median {np.median(per_sample_l2_us):.3f} µs")
    axes[0].set_xlabel("L2 norm of delay error (µs)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Per-sample L2 error distribution")
    axes[0].legend()

    # 2 — CDF of per-sample L2 norm
    sorted_err = np.sort(per_sample_l2_us)
    axes[1].plot(sorted_err, np.linspace(0, 1, len(sorted_err)))
    axes[1].set_xlabel("L2 norm of delay error (µs)")
    axes[1].set_ylabel("CDF")
    axes[1].set_title("Cumulative L2 error distribution")
    axes[1].grid(True)

    # 3 — MAE per receiver
    axes[2].bar(np.arange(M), mae_per_receiver)
    axes[2].set_xlabel("Receiver index")
    axes[2].set_ylabel("MAE (µs)")
    axes[2].set_title("Mean absolute error per receiver")

    plt.tight_layout()
    out_path = Path(__file__).parent / "eval_delay_net_errors.png"
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
