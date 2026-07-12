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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    ).to(device)

    ckpt = torch.load("DETR_v1.pt",  map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()


    treshhold = 0.5

    for signal, heatmap, coord, tau, phi, snr, numTargets in test_loader:
        
        y_complex = torch.complex(signal[:, 0,:,:].float(), signal[:, 1,:,:].float())  # [B, M, N]
        outputs = model(y_complex.to(device, non_blocking=True))

        locations = outputs["pos"]
        prob = outputs["logits"].softmax(-1)[..., 1]  # [B, Q]

        locations_un_norm = locations * ds_stats["coord_sd"].to(device) + ds_stats["coord_mean"].to(device)  # [B, Q, 2]
        coord_norm = (coord.float() - ds_stats["coord_mean"]) / ds_stats["coord_sd"]
        coord_norm = coord_norm.to(device, non_blocking=True)
        batch = {
            "y": y_complex.to(device, non_blocking=True),
            "pos_norm": coord_norm,                                   # list of [K_i, 3]
            "pos_xyz":  coord.to(device).float(),        # list of [K_i, 3]
            "num_targets": numTargets.to(device, non_blocking=True), # B,1
        }
        loss = radar_full_loss(outputs, batch, common_params)
        B, Q = prob.shape
        all_dets = []

        for b in range(B):
            keep = prob[b] > treshhold

            pos_m = locations_un_norm[b, keep]
            score = prob[b, keep]

            # Sort detections by score
            order = torch.argsort(score, descending=True)

            all_dets.append({
                "pos_m": pos_m[order],
                "score": score[order],
            })
        a=5


if __name__ == "__main__": 
    main()