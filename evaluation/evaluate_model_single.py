import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


from model.deep_radar import RadarMultiTaskNet
from trainer.my_dataloader import RadarMatDataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = PROJECT_ROOT / "best_radar_model_401.pt"
    test_root_dir = Path(r"D:\\radar-dataset2\\test")
    sample_index = 20
    heatmap_size = 401

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset = RadarMatDataset(root_dir=str(test_root_dir))
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample_index {sample_index} is out of range for {len(dataset)} test samples")

    signal, heatmap_gt, coord_gt = dataset[sample_index]
    heatmap_gt = heatmap_gt - heatmap_gt.min()
    heatmap_gt = heatmap_gt / (heatmap_gt.max() + 1e-8)

    model = RadarMultiTaskNet(use_fft=True, heatmap_size=heatmap_size).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    signal = signal.unsqueeze(0).to(device).float()
    heatmap_gt = heatmap_gt.unsqueeze(0).to(device).float()
    coord_gt = coord_gt.unsqueeze(0).to(device).float()

    with torch.no_grad():
        pred_heatmap, pred_coord = model(signal)

    heatmap_mse = F.mse_loss(pred_heatmap, heatmap_gt).item()
    coord_l1 = torch.abs(pred_coord - coord_gt).mean().item()
    coord_l2 = torch.linalg.vector_norm(pred_coord - coord_gt, dim=1).mean().item()

    gt_heatmap_np = heatmap_gt.squeeze(0).squeeze(0).cpu().numpy()
    pred_heatmap_np = pred_heatmap.squeeze(0).squeeze(0).cpu().numpy()
    err_heatmap_np = np.abs(pred_heatmap_np - gt_heatmap_np)

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Sample file: {dataset.file_paths[sample_index]}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint val loss: {checkpoint.get('val_loss', 'unknown')}")
    print()
    print("Ground truth coordinate:", coord_gt.squeeze(0).cpu().numpy())
    print("Predicted coordinate:", pred_coord.squeeze(0).cpu().numpy())
    print(f"Coordinate mean absolute error: {coord_l1:.6f}")
    print(f"Coordinate L2 error: {coord_l2:.6f}")
    print(f"Heatmap MSE: {heatmap_mse:.6f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    vmin = min(float(gt_heatmap_np.min()), float(pred_heatmap_np.min()))
    vmax = max(float(gt_heatmap_np.max()), float(pred_heatmap_np.max()))

    im0 = axes[0].imshow(gt_heatmap_np, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    axes[0].set_title("Ground Truth Heatmap")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pred_heatmap_np, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    axes[1].set_title("Predicted Heatmap")
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(err_heatmap_np, cmap="magma", aspect="auto")
    axes[2].set_title("Absolute Error |Pred-GT|")
    axes[2].set_xlabel("Column")
    axes[2].set_ylabel("Row")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.show(block=True)


if __name__ == "__main__":
    main()
