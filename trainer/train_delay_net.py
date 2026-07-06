import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.delay_net import DelayNet
from data_loaders.my_dataloader import RadarMatDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import gc

gc.collect()
torch.cuda.empty_cache()



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
    lambda_bin=1
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

def compute_tau_stats(dataset):
    """Compute per-feature mean and std of tau over the full training set."""
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    all_tau = []
    for signal, heatmap, coord, tau, phi, snr in loader:
        all_tau.append(tau.float())
    all_tau = torch.cat(all_tau, dim=0)  # [N, M]
    tau_mean = all_tau.mean(dim=0)
    tau_std = all_tau.std(dim=0)
    # Avoid exploding normalized error for near-constant receivers.
    std_floor = torch.clamp(tau_std.mean() * 0.1, min=1e-6)
    tau_std = tau_std.clamp(min=std_floor)
    return tau_mean, tau_std



def train_one_epoch(model, loader, optimizer, device, scaler, use_amp, tau_mean, tau_std, epoch):
    model.train()

    total_loss = 0.0
    total_tau_loss = 0.0
    total_bin_loss = 0.0

    for signal, heatmap, coord, tau, phi, snr in loader:

        signal = signal.to(device, non_blocking=True).float()
        tau = tau.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        snr = torch.empty(signal.size(0)).uniform_(-5, 20)
        # if epoch <= 5:
        #     snr = torch.empty(signal.size(0)).uniform_(10, 20.0)
        # elif epoch <= 15:
        #     snr = torch.empty(signal.size(0)).uniform_(7, 10)
        # else:
        #     snr = torch.empty(signal.size(0)).uniform_(-5, 7)

        signal = _add_noise(signal, snr)

        with torch.amp.autocast('cuda', enabled=use_amp):
            pred_tau, aux = model(signal, return_aux=True)        # model predicts in normalised space
            #pred_tau = pred_tau_norm * model.tau_std + model.tau_mean        # denorm to physical tau
            loss, tau_loss, bin_loss = delay_loss(pred_tau, aux=aux, tau_gt=tau, model=model)
            #loss = torch.linalg.vector_norm((pred_tau - tau) * 1e6, dim=1).mean()  # loss in µs

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * signal.size(0)
        total_tau_loss += tau_loss.item() * signal.size(0)
        total_bin_loss += bin_loss.item() * signal.size(0)

    print(f"train tau loss: {total_tau_loss / len(loader.dataset):.6f} |"
          f"train bin loss: {total_bin_loss / len(loader.dataset):.6f}")
      
    return total_loss / len(loader.dataset)

# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(model, loader, device, use_amp, tau_mean, tau_std):
    model.eval()

    total_loss = 0.0
    total_tau_loss = 0.0
    total_bin_loss = 0.0

    for signal, heatmap, coord, tau, phi, snr in loader:
        signal = signal.to(device, non_blocking=True).float()
        tau = tau.to(device, non_blocking=True).float()

        with torch.amp.autocast('cuda', enabled=use_amp):
            pred_tau, aux = model(signal, return_aux=True)        # model predicts in normalised space
            loss, tau_loss, bin_loss = delay_loss(pred_tau, aux=aux, tau_gt=tau, model=model)

        total_loss += loss.item() * signal.size(0)
        total_tau_loss += tau_loss.item() * signal.size(0)
        total_bin_loss += bin_loss.item() * signal.size(0)

    print(f"validation tau loss: {total_tau_loss / len(loader.dataset):.6f} |"
          f"validation bin loss: {total_bin_loss / len(loader.dataset):.6f}")
      
    return total_loss / len(loader.dataset)


# ============================================================
# Main
# ============================================================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    use_amp = use_cuda

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name}")
    else:
        print("Using CPU")

    # -------------------------
    # Config
    # -------------------------

    batch_size = 16
    epochs = 40

    lr = 1e-3

    # -------------------------
    # Dataset
    # -------------------------

    train_dataset = RadarMatDataset(
        root_dir="D:\\radar-dataset-clean\\train",
    )

    val_dataset = RadarMatDataset(
        root_dir="D:\\radar-dataset-clean\\validation",
    )

    print("Computing tau normalization statistics from train set...")
    tau_mean, tau_std = compute_tau_stats(train_dataset)
    tau_mean = tau_mean.to(device)
    tau_std = tau_std.to(device)
    print(f"  tau mean={tau_mean.mean():.4f}  std={tau_std.mean():.4f}")

    # Debuggers (debugpy/pydevd) inject into spawned workers and cause hangs.
    import sys
    num_workers = 0 if sys.gettrace() is not None else 4
    print(num_workers, "DataLoader workers")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )

    # -------------------------
    # Model
    # -------------------------

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

    print("Model will predict normalised tau (target ~ N(0,1) per receiver).")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=4,
        threshold=1e-3,
        min_lr=1e-6,
    )

    # -------------------------
    # Training loop
    # -------------------------

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            use_amp,
            tau_mean,
            tau_std,
            epoch
        )

        val_loss = validate(
            model,
            val_loader,
            device,
            use_amp,
            tau_mean,
            tau_std,
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch:03d} | "
            f"train loss: {train_loss:.6f} | "
            f"val loss: {val_loss:.6f} | "
            f"lr: {current_lr:.2e}"
        )

        # -------------------------
        # Save best model
        # -------------------------

        if val_loss < best_val_loss:
            count = 0
            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                "best_radar_model.pt",
            )

            print("Saved best model")
        
        count += 1
        # early stoping is validation hasnt improved for 5 epochs
        if count > 7:
            print("Early stopping")
            break


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

if __name__ == "__main__":
    main()
