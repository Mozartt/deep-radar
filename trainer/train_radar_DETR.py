import sys
from pathlib import Path
from xml.parsers.expat import model
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_loaders.dataloader_multi_targets import RadarMatDatasetMT

from scipy.optimize import linear_sum_assignment
import torch
from model.radar_DETR_v1 import RadarDETR
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from Utils.logger import Logger


def radar_detr_loss(
    outputs,
    targets,
    targets_cnt,
    lambda_pos=5.0,
    lambda_obj=1.0,
    no_object_weight=0.1,
):
    """
    outputs:
        dict from RadarDETR:
            pos:    [B, Q, 2]
            logits: [B, Q, 2]

    targets:
        list of length B.
        Each item is a dict:
            {
                "pos": tensor [K, 2] normalized xy positions
            }

        K may be 0.

    Returns:
        loss, loss_dict
    """

    pred_pos = outputs["pos"]        # [B, Q, 2]
    pred_logits = outputs["logits"]  # [B, Q, 2]

    B, Q, _ = pred_pos.shape
    device = pred_pos.device

    ce_weight = torch.tensor(
        [no_object_weight, 1.0],
        device=device,
        dtype=torch.float32,
    )

    total_cls_loss = 0.0
    total_pos_loss = 0.0
    total_matches = 0

    for b in range(B):
        tgt_pos = targets[b]["pos"].to(device)  # [K, 3]
        K = targets_cnt[b]["num_targets"].item()  # [B,1]

        # Default: all queries are no-target
        target_classes = torch.zeros(Q, dtype=torch.long, device=device)

        if K > 0:
            # Pairwise L1 distance: [Q, K]
            cost_pos = torch.cdist(pred_pos[b], tgt_pos[:K,:], p=1)

            # Objectness probability
            prob_target = pred_logits[b].softmax(dim=-1)[:, 1]  # [Q]

            # Matching cost: lower is better
            cost = (
                lambda_pos * cost_pos
                - lambda_obj * prob_target[:, None]
            )

            row_ind, col_ind = linear_sum_assignment(
                cost.detach().cpu().numpy()
            )

            row_ind = torch.as_tensor(row_ind, dtype=torch.long, device=device)
            col_ind = torch.as_tensor(col_ind, dtype=torch.long, device=device)

            # Matched queries are real targets
            target_classes[row_ind] = 1

            # Position loss only for matched queries
            pos_loss = F.l1_loss(
                pred_pos[b, row_ind],
                tgt_pos[col_ind],
                reduction="sum",
            )

            total_pos_loss = total_pos_loss + pos_loss
            total_matches += K

        # Classification loss for all queries
        cls_loss = F.cross_entropy(
            pred_logits[b],
            target_classes,
            weight=ce_weight,
            reduction="mean",
        )

        total_cls_loss = total_cls_loss + cls_loss

    total_cls_loss = total_cls_loss / B

    if total_matches > 0:
        total_pos_loss = total_pos_loss / total_matches
    else:
        total_pos_loss = pred_pos.sum() * 0.0

    loss = lambda_pos * total_pos_loss + lambda_obj * total_cls_loss

    loss_dict = {
        "loss": loss.detach(),
        "loss_pos": total_pos_loss.detach(),
        "loss_cls": total_cls_loss.detach(),
    }

    return loss, loss_dict

def bistatic_tau(points, tx_pos, rx_pos, c=3e8):
    """
    points: [B, K, 3]
    tx_pos: [3]
    rx_pos: [M, 3]

    returns:
        tau: [B, K, M]
    """

    B, K, _ = points.shape
    M = rx_pos.shape[0]

    tx_pos = tx_pos.to(points.device).view(1, 1, 3)
    rx_pos = rx_pos.to(points.device).view(1, 1, M, 3)

    d_tx = torch.linalg.norm(points - tx_pos, dim=-1)  # [B, K]

    p = points.view(B, K, 1, 3)
    d_rx = torch.linalg.norm(p - rx_pos, dim=-1)       # [B, K, M]

    tau = (d_tx.unsqueeze(-1) + d_rx) / c
    return tau

def build_spectrum_target_from_tau(
    tau,
    a,
    fs,
    n_fft,
    sigma_bins=1.5,
):
    """
    tau: [B, K, M]
    a: chirp slope
    fs: sampling frequency
    n_fft: FFT size

    returns:
        target: [B, 1, M, F]
    """

    device = tau.device
    B, K, M = tau.shape
    Freq = n_fft

    # Shifted FFT frequency grid in Hz
    freqs = torch.fft.fftfreq(n_fft, d=1.0 / fs).to(device)
    freqs = torch.fft.fftshift(freqs)  # [F]
    freqs = freqs[:Freq//2]  # use only negative frequencies

    bin_width = fs / n_fft
    sigma_hz = sigma_bins * bin_width

    # Beat frequency according to your sign convention
    fb = -a * tau  # [B, K, M]

    diff = freqs.view(1, 1, 1, Freq//2) - fb.unsqueeze(-1)
    gauss = torch.exp(-0.5 * (diff / sigma_hz) ** 2)  # [B, K, M, F]

    # Multi-target spectrum: max over target peaks
    target = gauss.max(dim=1).values  # [B, M, F]

    return target.unsqueeze(1)  # [B, 1, M, F]

def spectrum_aux_loss(spectrum_logits, spectrum_target, pos_weight=5.0):
    """
    spectrum_logits: [B, 1, M, F]
    spectrum_target: [B, 1, M, F]
    """

    weight = 1.0 + pos_weight * spectrum_target

    loss = F.binary_cross_entropy_with_logits(
        spectrum_logits,
        spectrum_target,
        weight=weight,
        reduction="mean",
    )

    return loss

def radar_full_loss(outputs, batch, common_params, lambda_spectrum=1.0):
    # DETR target format
    targets = [
        {"pos": p} for p in batch["pos_norm"]
    ]

    targets_cnt = [
        {"num_targets": n} for n in batch["num_targets"]
    ]

    loss_detr, loss_dict = radar_detr_loss(outputs, targets, targets_cnt)

    # Optional spectral auxiliary loss
    # This assumes batch["pos_xyz"] is a list of tensors [K, 3].
    pos_xyz_list = batch["pos_xyz"]
    max_k = torch.amax(batch["num_targets"])

    if max_k > 0:
        B = len(pos_xyz_list)
        device = batch["y"].device

        padded = torch.zeros(B, max_k, 3, device=device)
        valid = torch.zeros(B, max_k, dtype=torch.bool, device=device)

        for b, p in enumerate(pos_xyz_list):
            K = targets_cnt[b]["num_targets"].item()

            if K > 0:
                padded[b, :K] = p[:K,:].to(device)
                valid[b, :K] = True

        tau = bistatic_tau(
            padded,
            tx_pos=common_params.tx_pos.to(device),
            rx_pos=common_params.rx_pos,
        )  # [B, max_k, M]

        # For invalid padded targets, move their tau far outside observable range
        tau = torch.where(valid[:, :, None], tau, torch.full_like(tau, 1e9))

        spectrum_target = build_spectrum_target_from_tau(
            tau=tau,
            a=common_params.a,
            fs=common_params.fs,
            n_fft=common_params.n_fft,
            sigma_bins=1.5,
        )

        loss_spec = spectrum_aux_loss(
            outputs["spectrum_logits"],
            spectrum_target,
        )
    else:
        loss_spec = outputs["spectrum_logits"].sum() * 0.0

    total_loss = loss_detr + lambda_spectrum * loss_spec

    loss_dict["loss_spectrum"] = loss_spec
    loss_dict["loss_total"] = total_loss

    return loss_dict

def training_step(
    model,
    batch,
    optimizer,
    common_params,
    lambda_spectrum=1.0,
):
    model.train()
    optimizer.zero_grad()

    y = batch["y"]  # real [B, 2, M, N] (I/Q channels)

    outputs = model(y)

    loss_dict = radar_full_loss(outputs, batch, common_params, lambda_spectrum=lambda_spectrum)

    # before = {
    #     name: p.detach().clone()
    #     for name, p in model.named_parameters()
    #     if "pos_head" in name
    # }
    
    loss_dict["loss_total"].backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    
    # detach loss tensors, for logging only
    loss_dict["loss_total"] = loss_dict["loss_total"].detach()
    loss_dict["loss_spectrum"] = loss_dict["loss_spectrum"].detach()

    return loss_dict

@torch.no_grad()
def detect_targets(model, y, area_radius, threshold=0.5):
    """
    y: complex [B, M, N]
    area_radius: coordinate normalization scale in meters

    returns:
        list of detections per batch item:
            [
                {
                    "pos_m": [num_det, 2],
                    "score": [num_det]
                },
                ...
            ]
    """

    model.eval()
    outputs = model(y)

    pos_norm = outputs["pos"]                 # [B, Q, 2]
    prob = outputs["logits"].softmax(-1)[..., 1]  # [B, Q]

    B, Q, _ = pos_norm.shape

    all_dets = []

    for b in range(B):
        keep = prob[b] > threshold

        pos_m = pos_norm[b, keep] * area_radius
        score = prob[b, keep]

        # Sort detections by score
        order = torch.argsort(score, descending=True)

        all_dets.append({
            "pos_m": pos_m[order],
            "score": score[order],
        })

    return all_dets

def train_one_epoch(model, loader, optimizer, device, ds_stats, common_params):
    model.train()

    total_loss = 0.0

    for signal, signal_clean, heatmap, coord, tau, phi, snr, numTargets, sample_id in loader:

        # coord is a list of [K_i, 3] tensors (K_i may differ between samples)
        coord_norm = (coord.float() - ds_stats["coord_mean"]) / ds_stats["coord_sd"]
        coord_norm = coord_norm.to(device, non_blocking=True)

        batch = {
            #"y": signal.to(device, non_blocking=True).float(),  # real [B, 2, M, N]
            "y": signal_clean.to(device, non_blocking=True).float(),
            "pos_norm": coord_norm,                                   # list of [K_i, 3]
            "pos_xyz":  coord.to(device).float(),        # list of [K_i, 3]
            "num_targets": numTargets.to(device, non_blocking=True), # B,1
        }

        loss_dict = training_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            common_params=common_params,
            lambda_spectrum=1.0,
        )

        total_loss += loss_dict["loss_total"].item() * signal.size(0)

        # print(
        #     f"loss_cls={loss_dict['loss_cls'].item():.4f} | "
        #     f"loss_pos={loss_dict['loss_pos'].item():.4f} | "
        #     f"loss_spectrum={loss_dict['loss_spectrum'].item():.4f} | "
        # )

    return total_loss / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, device, use_amp, ds_stats, common_params):
    model.eval()
    total_loss = 0.0

    for signal, signal_clean, heatmap, coord, tau, phi, snr, numTargets, sample_id in loader:

        coord_norm = (coord.float() - ds_stats["coord_mean"]) / ds_stats["coord_sd"]
        coord_norm = coord_norm.to(device, non_blocking=True)

        batch = {
            #"y": signal.to(device, non_blocking=True).float(),  # real [B, 2, M, N]
            "y": signal_clean.to(device, non_blocking=True).float(),
            "pos_norm": coord_norm,                                   # list of [K_i, 3]
            "pos_xyz":  coord.to(device).float(),        # list of [K_i, 3]
            "num_targets": numTargets.to(device, non_blocking=True), # B,1
        }

        outputs = model(batch["y"])

        loss_dict = radar_full_loss(outputs, batch, common_params, lambda_spectrum=1.0)

        # detach loss tensors, for logging only
        loss_dict["loss_total"] = loss_dict["loss_total"].detach()
        loss_dict["loss_spectrum"] = loss_dict["loss_spectrum"].detach()

        total_loss += loss_dict["loss_total"].item() * signal.size(0)

    return total_loss / len(loader.dataset)

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

def main():

    # -------------------------
    # Logger
    # -------------------------
    log_file_path = PROJECT_ROOT / "logs" / "train_radar_DETR.log"
    logger = Logger(log_file=log_file_path, name="train_radar_DETR")
    logger.info(f"Logging to {log_file_path}")

    # parameters
    # common 
    common_params = type('', (), {})()  # empty object to hold parameters
    common_params.a = 1e13
    common_params.fs = 5e7
    common_params.M = 40
    common_params.n_fft = 1024
    common_params.rx_radius = 100 # m
    
    theta = 2 * torch.pi * torch.arange(common_params.M, dtype=torch.float32) / common_params.M
    rx_pos_tensor = torch.stack([torch.cos(theta), torch.sin(theta), torch.zeros(common_params.M)], dim=1) * common_params.rx_radius

    common_params.tx_pos = torch.zeros(3, dtype=torch.float32)  # transmitter at origin
    common_params.rx_pos = rx_pos_tensor

    # data loader parameters
    batch_size = 32
    num_workers = 4

    logger.info(f"Available GPUs: {torch.cuda.device_count()}")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = RadarDETR(
        M=common_params.M,
        rx_pos=common_params.rx_pos,
        n_fft=common_params.n_fft,
        d_model=256,
        num_queries=8,   # good for Kmax=2 initially
        top_p=32,
        num_decoder_layers=3,
        nhead=8,
        common_params=common_params,
    )
    
    if torch.cuda.device_count() >= 2:
        logger.info("Using GPUs 0 and 1")
        model = DataParallel(
            model,
            device_ids=[0, 1],
            output_device=0,
        )

    model = model.to(device)
    
    train_dataset = RadarMatDatasetMT(root_dir="D:\\radar-dataset-multi-targets\\train")
    validation_dataset = RadarMatDatasetMT(root_dir="D:\\radar-dataset-multi-targets\\validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0
    )

    val_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0
    )

    # compute data stats
    ds_stats = compute_dataset_stats(train_dataset)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        #lr=1e-4,
        lr=3e-4,
        weight_decay=1e-4,
    )

    # scheduler = ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.5,
    #     patience=4,
    #     threshold=1e-3,
    #     min_lr=1e-6,
    # )

    epochs = 10

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):

        train_loss = train_one_epoch(model,
                                    train_loader,
                                    optimizer, 
                                    device, 
                                    ds_stats, 
                                    common_params)

        val_loss = validate(model,
                            val_loader,
                            device,
                            use_amp=False,
                            ds_stats=ds_stats,
                            common_params=common_params)

        #scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(
            f"Epoch {epoch:03d} | "
            f"train loss: {train_loss:.6f} | "
            f"val loss: {val_loss:.6f} | "
            f"lr: {current_lr:.2e}"
        )

        if val_loss < best_val_loss:
        
            best_val_loss = val_loss

            model_to_save = model.module if isinstance(model, DataParallel) else model

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                "best_radar_model.pt",
            )

            logger.info("Saved best model")



if __name__ == "__main__":
    main()