import torch
from torch.utils.data import DataLoader
from pathlib import Path

def compute_dataset_stats(dataset):
    """Compute mean/std of individual target positions over the full dataset."""
    # use .pt cache 
    if Path("dataset_stats.pt").exists():
        stats = torch.load("dataset_stats.pt")
        return stats
    
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)
    all_tau = []
    all_coord = []
    for signal, heatmap, coord, tau, phi, snr, numTargets in loader:
        all_tau.append(tau.float())
        all_coord.append(coord)  # list of [K_i, 3] tensors

    all_tau = torch.cat(all_tau, dim=0)   # [N_total_targets, M]
    all_coord = torch.cat(all_coord, dim=0)   # [N_total_targets, 3]

    # tau mean
    tau_mask = (all_tau != 0).float()
    sum_tau = (all_tau * tau_mask).sum(dim=(0, 2))
    count_tau = tau_mask.sum(dim=(0, 2))
    tau_mean = sum_tau / count_tau.clamp(min=1.0)

    # tau std
    tau_var = ((all_tau - tau_mean[None,:,None]) ** 2 * tau_mask).sum(dim=(0, 2)) / count_tau.clamp(min=1.0)
    tau_sd = tau_var.sqrt()
    std_floor = tau_sd.mean() * 0.1
    tau_sd = tau_sd.clamp(min=std_floor)

    # coord mean
    coord_mask = (all_coord != 0).float()
    sum_coord = (all_coord * coord_mask).sum(dim=(0, 1))
    count_coord = coord_mask.sum(dim=(0, 1))
    coord_mean = sum_coord / count_coord.clamp(min=1.0)
    # coord std
    coord_var = ((all_coord - coord_mean) ** 2 * coord_mask).sum(dim=(0, 1)) / count_coord.clamp(min=1.0)
    coord_sd = coord_var.sqrt()

    stats = {}
    stats["tau_mean"] = tau_mean
    stats["tau_sd"] = tau_sd
    stats["coord_mean"] = coord_mean
    stats["coord_sd"] = coord_sd

    torch.save(stats, "dataset_stats.pt")

    return stats