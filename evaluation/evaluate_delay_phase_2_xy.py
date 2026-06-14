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

from model.delay_phase_net import DelayPhaseNet
from model.delay_net import DelayNet
from model.phase_net import PhaseOnlyNet
from model.delay_phase_2_pred_net import TauPhase2PredictionNet
from data_loaders.my_dataloader import RadarMatDataset


# ─────────────────────────────────────────────────────────────
# Stats helpers — must match each trainer's formula exactly
# ─────────────────────────────────────────────────────────────

def compute_tau_stats(dataset):
    """Matches train_delay_net_2d (std_floor applied)."""
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)
    all_tau = []
    for _, _, _, tau, phi in loader:
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
    all_tau, all_coord = [], []
    for _, _, coord, tau, phi in loader:
        all_tau.append(tau.float())
        all_coord.append(coord.float()[..., :2])
    all_tau   = torch.cat(all_tau,   dim=0)  # [N, M]
    all_coord = torch.cat(all_coord, dim=0)  # [N, 2]
    tau_std = all_tau.std(dim=0)
    std_floor = torch.clamp(tau_std.mean() * 0.1, min=1e-6)
    tau_std = tau_std.clamp(min=std_floor)
    return (
        all_tau.mean(dim=0), tau_std,
        all_coord.mean(dim=0), all_coord.std(dim=0).clamp(min=1e-8),
    )


@torch.no_grad()
def main():
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    gpu_label = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    print(f"Using {gpu_label}")

    # ── Dataset ──────────────────────────────────────────────
    train_dataset = RadarMatDataset(root_dir="D:\\radar-dataset\\train")
    test_dataset   = RadarMatDataset(root_dir="D:\\radar-dataset\\test")

    print("Computing normalisation stats from train set...")
    tau_mean_1, tau_std_1 = compute_tau_stats(train_dataset)       # delay_net stats (std_floor)
    tau_m, tau_sd, coord_mean, coord_std = compute_coord_stats(train_dataset)  # coord_net stats

    tau_mean_1 = tau_mean_1.to(device)
    tau_std_1  = tau_std_1.to(device)
    tau_mean   = tau_m.to(device)
    tau_std    = tau_sd.to(device)
    coord_mean = coord_mean.to(device)
    coord_std  = coord_std.to(device)
    M = 40

    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=use_cuda,
    )

    # ── Load models ──────────────────────────────────────────
    ckpt_tau   = torch.load("delay_net.pt",  map_location=device, weights_only=True)
    ckpt_phi   = torch.load("phase_net.pt",  map_location=device, weights_only=True)
    ckpt_coord = torch.load("delay_phase_2_xy.pt",  map_location=device, weights_only=True)

    delay_net = DelayNet(M=M).to(device)
    delay_net.load_state_dict(ckpt_tau["model_state_dict"])
    delay_net.eval()

    phase_net = PhaseOnlyNet(M=M, hidden_ch=512).to(device)
    phase_net.load_state_dict(ckpt_phi["model_state_dict"])
    phase_net.eval()

    coord_net = TauPhase2PredictionNet(M=M).to(device)
    coord_net.load_state_dict(ckpt_coord["model_state_dict"])
    coord_net.eval()

    print(f"Loaded delay_net  — saved at epoch {ckpt_tau.get('epoch', '?')}")
    print(f"Loaded coord_net  — saved at epoch {ckpt_coord.get('epoch', '?')}")

    # ── Inference ────────────────────────────────────────────
    # Pipeline:
    #   signal  →  delay_net  →  pred_tau_norm  (normalised w.r.t. tau_mean_1/std_1)
    #           →  denorm to physical tau
    #           →  renorm for coord_net  (w.r.t. tau_mean_2/std_2)
    #           →  coord_net  →  pred_coord_norm
    #           →  denorm to metres
    errors_m = []   # per-sample Euclidean error in metres

    for signal, _, coord_gt, tau_gt, phi_gt in test_loader:
        signal   = signal.to(device, non_blocking=True).float()
        coord_gt = coord_gt.to(device, non_blocking=True).float()[..., :2]

        # Stage 1 — signal → normalised tau
        pred_tau_norm = delay_net(signal)                               # [B, M]
        phi = phase_net(signal)                                              # [B, M, 2]
        cos_pred, sin_pred = phi[0], phi[1]                          # [B, M]
        # Relative phase is recovered in post-processing:
        # cos δφ_m = cos φ_m · cos φ_0 + sin φ_m · sin φ_0
        # sin δφ_m = sin φ_m · cos φ_0 − cos φ_m · sin φ_0
        cos_phi_0, sin_phi_0 = cos_pred[:, 0], sin_pred[:, 0]              # [B]
        cos_delta = cos_pred * cos_phi_0[:, None] + sin_pred * sin_phi_0[:, None]  # [B, M]
        sin_delta = sin_pred * cos_phi_0[:, None] - cos_pred * sin_phi_0[:, None]  # [B, M]

        # Stage 2 — normalised tau → normalised coord
        pred_coord_norm = coord_net(pred_tau_norm, cos_delta, sin_delta)                          # [B, 2]
        pred_coord      = pred_coord_norm * coord_std + coord_mean      # [B, 2] metres

        err = torch.linalg.vector_norm(pred_coord - coord_gt, dim=1)   # [B]
        errors_m.append(err.cpu())

    errors_m = torch.cat(errors_m).numpy()

    # ── Metrics ──────────────────────────────────────────────
    print(f"\n── Chained evaluation on test set ({len(errors_m)} samples) ──")
    print(f"  Mean error    : {errors_m.mean():.2f} m")
    print(f"  Median error  : {np.median(errors_m):.2f} m")
    print(f"  Std           : {errors_m.std():.2f} m")
    print(f"  90th pct      : {np.percentile(errors_m, 90):.2f} m")
    print(f"  95th pct      : {np.percentile(errors_m, 95):.2f} m")
    print(f"  Max error     : {errors_m.max():.2f} m")

    # ── Plot ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(errors_m, bins=60, edgecolor="black")
    axes[0].axvline(errors_m.mean(),       color="red",    linestyle="--",
                    label=f"mean {errors_m.mean():.1f} m")
    axes[0].axvline(np.median(errors_m),   color="orange", linestyle="--",
                    label=f"median {np.median(errors_m):.1f} m")
    axes[0].set_xlabel("Euclidean error (m)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Position error distribution")
    axes[0].legend()

    sorted_err = np.sort(errors_m)
    cdf = np.linspace(0, 1, len(sorted_err))
    axes[1].plot(sorted_err, cdf)
    axes[1].set_xlabel("Euclidean error (m)")
    axes[1].set_ylabel("CDF")
    axes[1].set_title("Cumulative error distribution")
    axes[1].grid(True)

    plt.tight_layout()
    out_path = Path(__file__).parent / "eval_2_networks_errors.png"
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
