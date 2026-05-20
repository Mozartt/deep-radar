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

from model.deep_radar import RadarMultiTaskNet
from trainer.my_dataloader import RadarMatDataset


@torch.no_grad()
def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	checkpoint_path = PROJECT_ROOT / "best_radar_model_401.pt"
	test_root_dir = Path(r"D:\radar-dataset2\test")

	batch_size = 64
	num_workers = 4
	heatmap_size = 401

	if not checkpoint_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

	dataset = RadarMatDataset(root_dir=str(test_root_dir))
	loader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=device.type == "cuda",
		persistent_workers=num_workers > 0,
	)

	model = RadarMultiTaskNet(use_fft=True, heatmap_size=heatmap_size).to(device)

	checkpoint = torch.load(checkpoint_path, map_location=device)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.eval()

	all_distances = []

	for signal, _, coord_gt in loader:
		signal = signal.to(device, non_blocking=True).float()
		coord_gt = coord_gt.to(device, non_blocking=True).float()

		_, pred_coord = model(signal)

		distances = torch.linalg.vector_norm(pred_coord - coord_gt, dim=1)
		all_distances.append(distances.cpu().numpy())

	distances_np = np.concatenate(all_distances, axis=0)

	mean_dist = float(np.mean(distances_np))
	median_dist = float(np.median(distances_np))
	p90_dist = float(np.percentile(distances_np, 90))
	p95_dist = float(np.percentile(distances_np, 95))

	print(f"Device: {device}")
	print(f"Checkpoint: {checkpoint_path}")
	print(f"Num test samples: {len(dataset)}")
	print(f"Mean Euclidean distance: {mean_dist:.6f}")
	print(f"Median Euclidean distance: {median_dist:.6f}")
	print(f"90th percentile distance: {p90_dist:.6f}")
	print(f"95th percentile distance: {p95_dist:.6f}")

	plt.figure(figsize=(9, 5))
	plt.hist(distances_np, bins=50, color="steelblue", edgecolor="black", alpha=0.85)
	plt.title("Histogram of Euclidean Distance Errors (Test Set)")
	plt.xlabel("Euclidean Distance")
	plt.ylabel("Number of Samples")
	plt.grid(True, alpha=0.25)

	plt.axvline(mean_dist, color="red", linestyle="--", linewidth=2, label=f"Mean = {mean_dist:.3f}")
	plt.axvline(median_dist, color="green", linestyle="--", linewidth=2, label=f"Median = {median_dist:.3f}")
	plt.legend()
	plt.tight_layout()
	plt.show(block=True)


if __name__ == "__main__":
	main()
