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
from model.delay_2_xyz_net import Delay2PredictionNet
from data_loaders.my_dataloader import RadarMatDataset


# ─────────────────────────────────────────────────────────────
# Stats helpers — must match each trainer's formula exactly
# ─────────────────────────────────────────────────────────────
def moving_average(x, L):
    """
    x: [N]
    returns moving average with length close to original
    """
    kernel = np.ones(L) / L
    return np.convolve(x, kernel, mode="same")

def estimate_chirp_start_single(y, L=128, guard_edges=True):
    y = y[:400] # limit to first 400 samples for speed
    power = np.abs(y) ** 2
    left_avg = moving_average(power, L)
    right_avg = moving_average(power[::-1], L)[::-1]
    score = right_avg - left_avg
    if guard_edges:
        score[:L] = -np.inf
        score[-L:] = -np.inf
    n0 = int(np.argmax(score))
    return n0, score

def estimate_chirp_starts(Y, L=128):
    """
    Fully vectorised chirp-start estimator — no Python loops.
    Y  : [B, M, N] tensor or numpy array
    Returns n0 : [B, M] int array

    Algorithm: score[i] = forward_avg[i] - backward_avg[i] computed via
    cumulative sums (identical result to the per-row np.convolve approach).
    """
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().numpy()
    B, M, N = Y.shape
    trim  = min(400, N)
    power = np.abs(Y[:, :, :trim]) ** 2            # [B, M, trim]

    pad = np.zeros((B, M, L - 1))

    # Backward-looking average: mean of the L samples ending at each position
    xp_l  = np.concatenate([pad, power], axis=-1)                      # [B, M, trim+L-1]
    cs_l  = np.cumsum(xp_l, axis=-1)
    cs_l0 = np.concatenate([np.zeros((B, M, 1)), cs_l], axis=-1)       # [B, M, trim+L]
    left_avg = (cs_l0[:, :, L:] - cs_l0[:, :, :-L]) / L               # [B, M, trim]

    # Forward-looking average: same on time-reversed signal, then flip back
    xp_r      = np.concatenate([pad, power[:, :, ::-1]], axis=-1)
    cs_r      = np.cumsum(xp_r, axis=-1)
    cs_r0     = np.concatenate([np.zeros((B, M, 1)), cs_r], axis=-1)
    right_avg = ((cs_r0[:, :, L:] - cs_r0[:, :, :-L]) / L)[:, :, ::-1]  # [B, M, trim]

    score = right_avg - left_avg
    score[:, :, :L]  = -np.inf
    score[:, :, -L:] = -np.inf

    return np.argmax(score, axis=-1).astype(int)                        # [B, M]

def refine_tau(pred_tau, signal):
    a = 1e13 # chirp slope 
    Fs = 5e7
    Ts = 1 / Fs
    complex_signal = torch.complex(signal[:,0,:,:], signal[:,1,:,:])  # [B, M, N]
    # pred_tau[:, :, None]: [B, M, 1] → broadcast with arange → [B, M, 1000]
    beta_exp = torch.exp(1j * 2 * torch.pi * a * pred_tau[:, :, None] * Ts * torch.arange(1000, device=pred_tau.device))
    phase_only = complex_signal * beta_exp  # de-beat signal: [B, M, N]

    # Find the latest chirp start across all receivers and batch items
    n0 = estimate_chirp_starts(complex_signal, L=128)  # [B, M]
    max_nz_idx = int(n0.max())  # scalar
    phase_only_nz = phase_only[:, :, max_nz_idx:]  # [B, M, N']
    theta = torch.angle(phase_only_nz)              # [B, M, N']
    theta_np = theta.detach().cpu().numpy()
    theta_unwrapped_np = np.unwrap(theta_np, axis=-1)
    theta_unwrapped = torch.from_numpy(
        theta_np
    ).float()  # [B, M, N']

    # w = torch.ones_like(theta_unwrapped)
    # eps = 1e-12
    # N_actual = theta_unwrapped.shape[-1]
    # w_sum = w.sum(dim=-1, keepdim=True) + eps
    # n = torch.arange(N_actual, device=theta_unwrapped.device)
    # n_view = n.view(1, 1, N_actual)
    # n_bar = (n_view).sum(dim=-1, keepdim=True) / w_sum
    # theta_bar = (theta_unwrapped).sum(dim=-1, keepdim=True) / w_sum
    # n_centered = n_view - n_bar
    # theta_centered = theta_unwrapped - theta_bar
    # numerator = (n_centered * theta_centered).sum(dim=-1)
    # denominator = (n_centered ** 2).sum(dim=-1) + eps
    # omega_hat = numerator / denominator  # [B, M], rad/sample

    #phase_only_nz = phase_only_nz.detach().cpu().numpy()  # [B, M, N']
    #prod = phase_only_nz[:, :, 1:]* np.conj(phase_only_nz[:, :, :-1])
    #omega_hat = np.angle(np.sum(prod))
    # try multi-lag 
    phase_only_nz = phase_only_nz.detach().cpu().numpy()
    eps = 1e-12
    L=32
    lags = np.arange(1, L + 1)
    R_list = []
    for ell in lags:
        R_ell = np.sum(phase_only_nz[:, :, ell:] * np.conj(phase_only_nz[:, :, :-ell]), axis=-1)
        R_list.append(R_ell)
    R = np.stack(R_list, axis=-1)  # [B, M, L]
    theta = np.unwrap(np.angle(R), axis=-1)
    weights = np.abs(R) + eps
    numerator = np.sum(weights * lags[None, None, :] * theta, axis=-1)
    denominator = np.sum(weights * lags[None, None, :] ** 2, axis=-1) + eps

    omega_hat = numerator / denominator

    delta_tau_hat = omega_hat / (2 * torch.pi * a * Ts)
    delta_tau_hat = torch.tensor(delta_tau_hat, dtype=torch.float32, device=pred_tau.device)
    refined_tau = pred_tau - delta_tau_hat
    # refined_tau[:, :, None]: [B, M, 1] → broadcast → [B, M, 1000]
    beta_exp = torch.exp(1j * 2 * torch.pi * a * refined_tau[:, :, None] * Ts * torch.arange(1000, device=refined_tau.device))  # [B, M, N]
    # remove beat frequency from input signal
    phase_only = complex_signal * beta_exp                        # [B, M, N]
    phase_only_nz = phase_only[:, :, max_nz_idx:]                 # [B, M, N']
    phi_hat = torch.angle(phase_only_nz).mean(dim=-1)             # [B, M]
    
    return refined_tau, phi_hat

def compute_tau_stats(dataset):
    """Matches train_delay_net_2d (std_floor applied)."""
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)
    all_tau = []
    for _, _, _, tau, _ in loader:
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
    for _, _, coord, _, _ in loader:
        all_coord.append(coord.float()[..., :2])
    all_coord = torch.cat(all_coord, dim=0)  # [N, 2]
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
    train_dataset = RadarMatDataset(root_dir="D:\\radar-dataset-noisy\\train")
    test_dataset   = RadarMatDataset(root_dir="D:\\radar-dataset-noisy\\test")

    print("Computing normalisation stats from train set...")
    tau_mean_1, tau_std_1 = compute_tau_stats(train_dataset)       # delay_net stats (std_floor)
    coord_mean, coord_std = compute_coord_stats(train_dataset)  # coord_net stats

    tau_mean_1 = tau_mean_1.to(device)
    tau_std_1  = tau_std_1.to(device)
    coord_mean = coord_mean.to(device)
    coord_std  = coord_std.to(device)
    M = 40

    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=use_cuda,
    )

    # ── Load models ──────────────────────────────────────────
    ckpt_tau   = torch.load("delay_net_noisy.pt",  map_location=device, weights_only=True)
    ckpt_coord = torch.load("delay_2_xy_noisy.pt",  map_location=device, weights_only=True)

    delay_net = DelayNet(M=M).to(device)
    delay_net.load_state_dict(ckpt_tau["model_state_dict"])
    delay_net.eval()

    coord_net = Delay2PredictionNet(M=M).to(device)
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

        # Bridge — denorm from delay_net space → renorm for coord_net space
        pred_tau_phys = pred_tau_norm * tau_std_1 + tau_mean_1          # [B, M] (physical)
        refined_tau, phi = refine_tau(pred_tau_phys, signal)  # refine tau and estimate phi_hat

        refined_tau_norm = (refined_tau - tau_mean_1) / tau_std_1  # renorm for coord_net
        # Stage 2 — normalised tau → normalised coord
        pred_coord_norm = coord_net(refined_tau_norm)                          # [B, 2]
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
