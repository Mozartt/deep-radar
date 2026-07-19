from collections import defaultdict
import os
from pyexpat import model
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

from model.radar_DETR_v1 import RadarDETR
from data_loaders.dataloader_multi_targets import RadarMatDatasetMT
from Utils import dataset_stats
from Utils import common_params as CP
from trainer.train_radar_DETR import radar_full_loss
from scipy.optimize import linear_sum_assignment

def hungarian_matcher(pred, pred_logits, gt_coord, K, lambda_pos=5.0, lambda_obj=1.0):
    """
    pred: [Q, 2] predicted positions (x,y) in metres
    gt_coord: [K, 2] ground truth positions (x,y) in metres
    K: number of targets in this sample

    Returns:
        indices: list of tuples (pred_idx, gt_idx)
    """
    
    if K == 0:
        return []

    # Compute cost matrix based on Euclidean distance, use same cost as in training loss
    cost_pos = torch.cdist(pred, gt_coord[:K,:], p=1)
    prob_target = pred_logits.softmax(dim=-1)[:, 1]  # [Q]

    cost = (
            lambda_pos * cost_pos
            - lambda_obj * prob_target[:, None]
        )

    # Solve the assignment problem using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost.cpu())

    return row_ind, col_ind


@torch.no_grad()
def main():
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    use_cuda = device.type == "cuda"
    gpu_label = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    print(f"Using {gpu_label}")

    # ── Dataset ──────────────────────────────────────────────
    train_dataset = RadarMatDatasetMT(root_dir="D:\\radar-dataset-multi-targets\\train")
    test_dataset   = RadarMatDatasetMT(root_dir="D:\\radar-dataset-multi-targets\\test")

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        num_workers=4, pin_memory=use_cuda,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False,
        num_workers=4, pin_memory=use_cuda,
    )

    ds_stats = dataset_stats.compute_dataset_stats(train_dataset)

    common_params = CP.getCommonParams()
    model = RadarDETR(
        M=common_params.M,
        rx_pos=common_params.rx_pos,
        n_fft=common_params.n_fft,
        d_model=256,
        num_queries=8,   # good for Kmax=2 initially
        top_p=32,
        num_decoder_layers=3,
        nhead=8,
        common_params=common_params
    ).to(device)

    ckpt = torch.load("DETR_v1.pt",  map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_errors = []
    treshhold = 0.5

    for signal, signal_clean, heatmap, coord, tau, phi, snr, numTargets, sample_id in test_loader:
        
        #y_complex = torch.complex(signal[:, 0,:,:].float(), signal[:, 1,:,:].float())  # [B, M, N]
        with torch.no_grad():
            outputs = model(signal_clean.to(device, non_blocking=True).float())

        locations = outputs["pos"]
        prob = outputs["logits"]

        # locations_un_norm = locations * ds_stats["coord_sd"].to(device) + ds_stats["coord_mean"].to(device)  # [B, Q, 2]
        coord_norm = (coord.float() - ds_stats["coord_mean"]) / ds_stats["coord_sd"]
        #coord_norm = coord_norm.to(device, non_blocking=True)
        # batch = {
        #     "y": signal.to(device, non_blocking=True).float(),
        #     "pos_norm": coord_norm,                                   # list of [K_i, 3]
        #     "pos_xyz":  coord.to(device).float(),        # list of [K_i, 3]
        #     "num_targets": numTargets.to(device, non_blocking=True), # B,1
        # }
        # loss = radar_full_loss(outputs, batch, common_params, ds_stats)

        all_dets = []

        for b in range(signal.size(0)):
            
            pred_indecies, gt_indecies = hungarian_matcher(locations[b], prob[b], coord_norm[b].to(device), numTargets[b].to(device), lambda_pos=5.0, lambda_obj=1.0)
            prob_sm = prob[b].softmax(dim=-1)[:, 1]
            pred = locations[b][pred_indecies]
            gt = coord[b][gt_indecies]

            pred_un_norm = pred * ds_stats["coord_sd"].to(device) + ds_stats["coord_mean"].to(device)
                                                                                
            errors = torch.linalg.vector_norm(pred_un_norm - gt.to(device), dim=-1)

            for j in range(len(errors)):
                all_errors.append({
                    "sample_idx": int(sample_id[b].item()),
                    "num_targets": int(numTargets[b].item()),
                    "query_idx": int(pred_indecies[j].item()),
                    "target_idx": int(gt_indecies[j].item()),
                    "error_m": float(errors[j].item()),
                    "prob_target": float(
                        prob_sm[pred_indecies[j]].item()
                    ),
                    "pred": pred_un_norm[j].cpu(),
                    "gt": gt[j].cpu(),
                })


    # all_errors.sort(
    #     key=lambda record: record["error_m"],
    #     reverse=True,
    # )
    errors_tensor = torch.tensor(
        [record["error_m"] for record in all_errors]
    )

    for threshold in [5, 10, 20, 50]:
        count = (errors_tensor > threshold).sum().item()
        fraction = count / len(errors_tensor)

        print(
            f"Error > {threshold:2d} m: "
            f"{count} / {len(errors_tensor)} "
            f"({100 * fraction:.3f}%)"
        )

    from collections import defaultdict

    errors_by_k = defaultdict(list)

    for record in all_errors:
        errors_by_k[record["num_targets"]].append(
            record["error_m"]
        )

    for k in sorted(errors_by_k):
        values = torch.tensor(errors_by_k[k])

        print(
            f"K={k}: "
            f"mean={values.mean():.3f} m, "
            f"median={values.median():.3f} m, "
            f"p90={torch.quantile(values, 0.9):.3f} m, "
            f"max={values.max():.3f} m, "
            f"N={len(values)}"
        )
    # print("Mean error:", errors_tensor.mean().item())
    # print("Median error:", errors_tensor.median().item())
    # print("90th percentile:", torch.quantile(errors_tensor, 0.9).item())
    # print("Maximum error:", errors_tensor.max().item())

if __name__ == "__main__": 
    main()