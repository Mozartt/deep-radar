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

from model.delay_2_prediction_net import Delay2PredictionNet
from data_loaders.my_dataloader import RadarMatDataset


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

@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = PROJECT_ROOT / "best_radar_model.pt"
    test_root_dir = Path(r"D:\\radar-dataset-delay-only\\test")
    train_root_dir = Path(r"D:\\radar-dataset-delay-only\\train")

    batch_size = 64
    num_workers = 4

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset = RadarMatDataset(root_dir=str(test_root_dir))
    train_dataset = RadarMatDataset(root_dir=str(train_root_dir))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    tau_mean, tau_std, coord_mean, coord_std = compute_dataset_stats(train_dataset)
    tau_mean   = tau_mean.to(device)
    tau_std    = tau_std.to(device)
    coord_mean = coord_mean.to(device)
    coord_std  = coord_std.to(device)

    model = Delay2PredictionNet(M=40).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    for signal, heatmap, coord, tau in loader:
        
        coord = coord.to(device, non_blocking=True).float()[..., :2]
        tau = tau.to(device, non_blocking=True).float()
        #tau_norm   = tau * 1e6
        tau_norm   = (tau - tau_mean) / tau_std # the mean, sd are from the training set

        pred_coord = model(tau_norm)
        pred_coord = pred_coord * coord_std + coord_mean

        distances = torch.linalg.vector_norm(pred_coord - coord, dim=1)
        print(f"Mean distance error: {distances.mean().item():.4f} m")


if __name__ == "__main__":
    main()