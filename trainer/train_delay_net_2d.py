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


def compute_tau_stats(dataset):
    """Compute per-feature mean and std of tau over the full training set."""
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)
    all_tau = []
    for signal, heatmap, coord, tau, phi in loader:
        all_tau.append(tau.float())
    all_tau = torch.cat(all_tau, dim=0)  # [N, M]
    tau_mean = all_tau.mean(dim=0)
    tau_std = all_tau.std(dim=0)
    # Avoid exploding normalized error for near-constant receivers.
    std_floor = torch.clamp(tau_std.mean() * 0.1, min=1e-6)
    tau_std = tau_std.clamp(min=std_floor)
    return tau_mean, tau_std



def train_one_epoch(model, loader, optimizer, device, scaler, use_amp, tau_mean, tau_std):
    model.train()

    total_loss = 0.0

    for signal, heatmap, coord, tau, phi in loader:

        signal = signal.to(device, non_blocking=True).float()
        tau = tau.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            pred_tau_norm = model(signal)                        # model predicts in normalised space
            pred_tau = pred_tau_norm * tau_std + tau_mean        # denorm to physical tau
            loss = torch.linalg.vector_norm((pred_tau - tau) * 1e6, dim=1).mean()  # loss in µs

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * signal.size(0)

    return total_loss / len(loader.dataset)

# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(model, loader, device, use_amp, tau_mean, tau_std):
    model.eval()

    total_loss = 0.0

    for signal, heatmap, coord, tau, phi in loader:
        signal = signal.to(device, non_blocking=True).float()
        tau = tau.to(device, non_blocking=True).float()

        with torch.amp.autocast('cuda', enabled=use_amp):
            pred_tau_norm = model(signal)                        # model predicts in normalised space
            pred_tau = pred_tau_norm * tau_std + tau_mean        # denorm to physical tau
            loss = torch.linalg.vector_norm((pred_tau - tau) * 1e6, dim=1).mean()  # loss in µs

        total_loss += loss.item() * signal.size(0)

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

    batch_size = 64
    epochs = 50

    lr = 1e-3

    # -------------------------
    # Dataset
    # -------------------------

    train_dataset = RadarMatDataset(
        root_dir="D:\\radar-dataset-3d-noisy\\train",
    )

    val_dataset = RadarMatDataset(
        root_dir="D:\\radar-dataset-3d-noisy\\validation",
    )

    print("Computing tau normalization statistics from train set...")
    tau_mean, tau_std = compute_tau_stats(train_dataset)
    tau_mean = tau_mean.to(device)
    tau_std = tau_std.to(device)
    print(f"  tau mean={tau_mean.mean():.4f}  std={tau_std.mean():.4f}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=use_cuda,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=use_cuda,
        persistent_workers=True,
    )

    # -------------------------
    # Model
    # -------------------------

    model = DelayNet(M=tau_mean.numel()).to(device)
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


if __name__ == "__main__":
    main()
