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

bin_head_ouput = []
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

def make_soft_bin_targets_from_freq(beat_freq_gt, freq_grid, sigma_bins=1.0):
    """
    beat_freq_gt: [B, M]
        True beat frequency per receiver.

    freq_grid: [K] or [1, 1, K]
        Model frequency grid.

    sigma_bins:
        Width of the soft label in units of FFT bins.

    returns:
        soft_targets: [B, M, K]
    """

    if freq_grid.dim() == 1:
        freq_grid = freq_grid.view(1, 1, -1)

    freq_grid = freq_grid.to(device=beat_freq_gt.device, dtype=beat_freq_gt.dtype)

    # Estimate bin spacing
    df = torch.mean(torch.abs(freq_grid[..., 1:] - freq_grid[..., :-1]))

    sigma_freq = sigma_bins * df

    # Distance between every GT beat frequency and every bin
    dist = freq_grid - beat_freq_gt.unsqueeze(-1)  # [B, M, K]

    soft_targets = torch.exp(-0.5 * (dist / sigma_freq) ** 2)

    # Normalize to probability distribution
    soft_targets = soft_targets / (soft_targets.sum(dim=-1, keepdim=True) + 1e-12)

    return soft_targets

def delay_loss(
    pred_tau_norm,
    aux,
    tau_gt,
    model,
    lambda_bin=0.003
):
    #pred_tau_norm = (pred_tau - model.tau_mean) / model.tau_std
    #tau_gt_norm = (tau_gt - model.tau_mean) / model.tau_std
    pred_tau_phys = pred_tau_norm * model.tau_std + model.tau_mean
    tau_loss = F.smooth_l1_loss(pred_tau_phys*1e6, tau_gt*1e6)

    beat_freq_gt = -1 * model.a * tau_gt
    freq_grid = model.freq_grid.view(1, 1, model.K)

    target_bin = torch.argmin(
        torch.abs(freq_grid - beat_freq_gt.unsqueeze(-1)),
        dim=-1,
    )  # [B, M]

    logits = aux["logits"]

    soft_targets = make_soft_bin_targets_from_freq(
        beat_freq_gt=beat_freq_gt,
        freq_grid=model.freq_grid,
        sigma_bins=0.7,
    )
    
    log_probs = F.log_softmax(logits, dim=-1)
    bin_loss = -(soft_targets * log_probs).sum(dim=-1).mean()
    # bin_loss = F.cross_entropy(
    #     logits.reshape(-1, model.K),
    #     target_bin.reshape(-1),
    # )

    loss = tau_loss + lambda_bin * bin_loss

    return loss, tau_loss, bin_loss

def print_shape_hook(module, input, output):
    bin_head_ouput.append(output)

def _add_noise(signal: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
	"""Add complex Gaussian noise to a [2, M, N] or [B, 2, M, N] signal tensor.

	Matches the MATLAB noise model in get_radar_response_noisy.m:
		signal_power = 1
		noise_power  = N * signal_power / 10^(SNR_dB/10)
		noise        = sqrt(noise_power/2) * (randn + 1j*randn)

	Args:
		signal : [2, M, N] or [B, 2, M, N]
		snr_db : scalar tensor  OR  [B] tensor (one SNR per sample in the batch)
	"""
	N = signal.shape[-1]
	snr_linear = 10.0 ** (snr_db / 10.0)           # scalar or [B]
	noise_power = N / snr_linear                    # scalar or [B]
	std = (noise_power / 2.0) ** 0.5               # scalar or [B]

	if std.ndim > 0:                                # batched: reshape [B] -> [B, 1, 1, 1]
		std = std.view(-1, 1, 1, 1)

	noise = torch.randn_like(signal) * std.to(signal.device)
	return signal + noise


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
   
    tau_mean_1 = tau_mean_1.to(device)
    tau_std_1  = tau_std_1.to(device)
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

    errors_3d = []
    snr_all   = []
    error_rmse = 0.0

    signal, _, coord_gt, tau_gt, phi_gt, snr = train_dataset[0]
    signal   = signal.to(device, non_blocking=True).float()
    coord_gt = coord_gt.to(device, non_blocking=True).float()[..., :3]
    tau_gt   = tau_gt.to(device, non_blocking=True).float()

    delay_net.bin_head.register_forward_hook(print_shape_hook)
    # Stage 1 — signal → normalised tau

    snr = torch.tensor(-5)  # SNR in dB
    noisy_signal = _add_noise(signal, snr)
    pred_tau, aux = delay_net(noisy_signal.unsqueeze(0), return_aux=True)  # [B, M]
    pred_tau_phy = pred_tau * tau_std_1 + tau_mean_1                             # [B, M]

    delay_loss(pred_tau, aux, tau_gt, delay_net)

if __name__ == "__main__":
    main()