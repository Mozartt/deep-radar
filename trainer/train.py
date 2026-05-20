import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.deep_radar import RadarMultiTaskNet
from trainer.my_dataloader import RadarMatDataset

def train_one_epoch(model, loader, optimizer, device, scaler, use_amp, lambda_coord=0.1):
    model.train()

    total_loss = 0.0

    for signal, heatmap, coord in loader:

        signal = signal.to(device, non_blocking=True).float()
        heatmap = heatmap.to(device, non_blocking=True).float()
        coord = coord.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            pred_heatmap, pred_coord = model(signal)

            loss_heatmap = F.mse_loss(pred_heatmap, heatmap)
            loss_coord = F.smooth_l1_loss(pred_coord, coord)

            loss = loss_heatmap + lambda_coord * loss_coord

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * signal.size(0)

    return total_loss / len(loader.dataset)

# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(model, loader, device, use_amp, lambda_coord=0.1):
    model.eval()

    total_loss = 0.0

    for signal, heatmap, coord in loader:

        signal = signal.to(device, non_blocking=True).float()
        heatmap = heatmap.to(device, non_blocking=True).float()
        coord = coord.to(device, non_blocking=True).float()

        with torch.amp.autocast('cuda', enabled=use_amp):
            pred_heatmap, pred_coord = model(signal)

            loss_heatmap = F.mse_loss(pred_heatmap, heatmap)
            loss_coord = F.smooth_l1_loss(pred_coord, coord)

            loss = loss_heatmap + lambda_coord * loss_coord

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

    heatmap_size = 65

    batch_size = 16
    epochs = 50

    lr = 1e-3

    lambda_coord = 0.1

    # -------------------------
    # Dataset
    # -------------------------

    train_dataset = RadarMatDataset(
        root_dir="D:\\radar-dataset3\\train",
    )

    val_dataset = RadarMatDataset(
        root_dir="D:\\radar-dataset3\\validation",
    )

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

    model = RadarMultiTaskNet(
        use_fft=True,
        heatmap_size=heatmap_size,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

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
            lambda_coord=lambda_coord,
        )

        val_loss = validate(
            model,
            val_loader,
            device,
            use_amp,
            lambda_coord=lambda_coord,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss: {train_loss:.6f} | "
            f"val loss: {val_loss:.6f}"
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
