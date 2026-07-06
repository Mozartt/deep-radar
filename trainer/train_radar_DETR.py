from scipy.optimize import linear_sum_assignment
import torch


def radar_detr_loss(
    outputs,
    targets,
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
        tgt_pos = targets[b]["pos"].to(device)  # [K, 2]
        K = tgt_pos.shape[0]

        # Default: all queries are no-target
        target_classes = torch.zeros(Q, dtype=torch.long, device=device)

        if K > 0:
            # Pairwise L1 distance: [Q, K]
            cost_pos = torch.cdist(pred_pos[b], tgt_pos, p=1)

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

    bin_width = fs / n_fft
    sigma_hz = sigma_bins * bin_width

    # Beat frequency according to your sign convention
    fb = -a * tau  # [B, K, M]

    diff = freqs.view(1, 1, 1, Freq) - fb.unsqueeze(-1)
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

def training_step(
    model,
    batch,
    optimizer,
    a,
    fs,
    tx_pos,
    lambda_spectrum=1.0,
):
    model.train()
    optimizer.zero_grad()

    y = batch["y"]  # complex [B, M, N]

    outputs = model(y)

    # DETR target format
    targets = [
        {"pos": p} for p in batch["pos_norm"]
    ]

    loss_detr, loss_dict = radar_detr_loss(outputs, targets)

    # Optional spectral auxiliary loss
    # This assumes batch["pos_xyz"] is a list of tensors [K, 3].
    pos_xyz_list = batch["pos_xyz"]
    max_k = max(p.shape[0] for p in pos_xyz_list)

    if max_k > 0:
        B = len(pos_xyz_list)
        device = y.device

        padded = torch.zeros(B, max_k, 3, device=device)
        valid = torch.zeros(B, max_k, dtype=torch.bool, device=device)

        for b, p in enumerate(pos_xyz_list):
            K = p.shape[0]
            if K > 0:
                padded[b, :K] = p.to(device)
                valid[b, :K] = True

        tau = bistatic_tau(
            padded,
            tx_pos=tx_pos.to(device),
            rx_pos=model.rx_pos,
        )  # [B, max_k, M]

        # For invalid padded targets, move their tau far outside observable range
        tau = torch.where(valid[:, :, None], tau, torch.full_like(tau, 1e9))

        spectrum_target = build_spectrum_target_from_tau(
            tau=tau,
            a=a,
            fs=fs,
            n_fft=model.n_fft,
            sigma_bins=1.5,
        )

        loss_spec = spectrum_aux_loss(
            outputs["spectrum_logits"],
            spectrum_target,
        )
    else:
        loss_spec = outputs["spectrum_logits"].sum() * 0.0

    loss = loss_detr + lambda_spectrum * loss_spec

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    loss_dict["loss_spectrum"] = loss_spec.detach()
    loss_dict["loss_total"] = loss.detach()

    return loss_dict

def main():
    
    # Example
    M = 40
    n_fft = 2048
    area_radius = 1000.0  # meters

    # rx_pos should be [M, 3] in meters
    # Example: torch.tensor([...], dtype=torch.float32)
    rx_pos = rx_pos_tensor

    model = RadarDETR(
        M=M,
        rx_pos=rx_pos,
        n_fft=n_fft,
        d_model=256,
        num_queries=8,   # good for Kmax=2 initially
        top_p=32,
        num_decoder_layers=3,
        nhead=8,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )