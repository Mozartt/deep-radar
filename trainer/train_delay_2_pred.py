import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.delay_2_prediction_net import Delay2PredictionNet
from model.delay_net import DelayNet
from data_loaders.my_dataloader import RadarMatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR


def compute_dataset_stats(dataset):
    """Compute per-feature mean and std of tau and coord[:2] over the full dataset."""
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)
    all_tau, all_coord = [], []
    for signal, heatmap, coord, tau in loader:
        all_tau.append(tau.float())
        all_coord.append(coord.float()[..., :2])
    all_tau = torch.cat(all_tau, dim=0)      # [N, M]
    all_coord = torch.cat(all_coord, dim=0)  # [N, 2]
    return (
        all_tau.mean(dim=0), all_tau.std(dim=0).clamp(min=1e-8),
        all_coord.mean(dim=0), all_coord.std(dim=0).clamp(min=1e-8),
    )



def train_one_epoch(model,delay_net_model, loader, optimizer, device, scaler, use_amp,
                    tau_mean, tau_std, coord_mean, coord_std):
    model.train()

    total_loss = 0.0

    for signal, heatmap, coord, tau in loader:

        tau = tau.to(device, non_blocking=True).float()

        coord = coord.to(device, non_blocking=True).float()[..., :2]
        #tau_norm = tau * 1e6

        # evaluate delay net to get tau_norm
        #with torch.no_grad():
            #signal = signal.to(device, non_blocking=True).float()
            #pred_tau_norm = delay_net_model(signal)  # [B, M]
        
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            pred_norm  = model(tau)
            # denormalize → compare in meters for loss
            pred_coord = pred_norm * coord_std + coord_mean
            loss = torch.linalg.vector_norm(pred_coord - coord, dim=1).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * tau.size(0)

    return total_loss / len(loader.dataset)

# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(model, delay_net_model, loader, device, use_amp,
             tau_mean, tau_std, coord_mean, coord_std):
    model.eval()

    total_loss = 0.0

    for signal, heatmap, coord, tau in loader:

        coord = coord.to(device, non_blocking=True).float()[..., :2]
        tau = tau.to(device, non_blocking=True).float()
        # evaluate delay net to get tau_norm
        #with torch.no_grad():
        #    signal = signal.to(device, non_blocking=True).float()
        #    pred_tau_norm = delay_net_model(signal)  # [B, M]

        with torch.amp.autocast('cuda', enabled=use_amp):
            pred_norm  = model(tau)
            # denormalize → compare in meters for loss
            pred_coord = pred_norm * coord_std + coord_mean
            loss = torch.linalg.vector_norm(pred_coord - coord, dim=1).mean()

        total_loss += loss.item() * tau.size(0)

    return total_loss / len(loader.dataset)


# ============================================================
# Main
# ============================================================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    use_amp = False  # model is a small MLP — AMP gives no benefit here

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
    epochs = 60

    lr = 1e-3
    M= 40
    # -------------------------
    # Dataset
    # -------------------------

    train_dataset = RadarMatDataset(
        root_dir="D:\\radar-dataset\\train",
    )

    val_dataset = RadarMatDataset(
        root_dir="D:\\radar-dataset\\validation",
    )

    print("Computing normalisation statistics from train set...")
    tau_mean, tau_std, coord_mean, coord_std = compute_dataset_stats(train_dataset)
    tau_mean   = tau_mean.to(device)
    tau_std    = tau_std.to(device)
    coord_mean = coord_mean.to(device)
    coord_std  = coord_std.to(device)
    print(f"  tau   mean={tau_mean.mean():.4f}  std={tau_std.mean():.4f}")
    print(f"  coord mean={coord_mean}  std={coord_std}")

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

    # delay net
    ckpt  = torch.load("best_radar_model_samples_2_tau_clean.pt",
                       map_location=device, weights_only=True)
    delay_net_model = DelayNet(M=M).to(device)
    delay_net_model.load_state_dict(ckpt["model_state_dict"])
    delay_net_model.eval()

    # -------------------------
    # Model
    # -------------------------

    model = Delay2PredictionNet(M).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    # Keep LR from collapsing too early; complete cosine cycle beyond current run.
    scheduler_t_max = epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_t_max, eta_min=1e-5)

    # -------------------------
    # Training loop
    # -------------------------

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):

        train_loss = train_one_epoch(
            model, delay_net_model, train_loader, optimizer, device, scaler, use_amp,
            tau_mean, tau_std, coord_mean, coord_std,
        )

        scheduler.step()  # must come after optimizer.step() (which is inside train_one_epoch)

        val_loss = validate(
            model, delay_net_model, val_loader, device, use_amp,
            tau_mean, tau_std, coord_mean, coord_std,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss: {train_loss:.6f} | "
            f"val loss: {val_loss:.6f} | "
            f"lr: {scheduler.get_last_lr()[0]:.2e}"
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
