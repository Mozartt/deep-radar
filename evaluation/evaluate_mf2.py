import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import math
from typing import Iterator, Sequence

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import torch
from data_loaders.dataloader_multi_targets import RadarMatDatasetMT
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment

def make_cartesian_grid(
    x_range=(-150.0, 150.0),
    y_range=(-150.0, 150.0),
    z_range=(200.0, 300.0),
    resolution=2.0,
    device="cpu",
):
    x = torch.arange(
        x_range[0],
        x_range[1] + resolution / 2,
        resolution,
        device=device,
    )
    y = torch.arange(
        y_range[0],
        y_range[1] + resolution / 2,
        resolution,
        device=device,
    )
    z = torch.arange(
        z_range[0],
        z_range[1] + resolution / 2,
        resolution,
        device=device,
    )

    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
    return torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)

def select_spatial_peaks(
    positions,
    scores,
    num_targets,
    minimum_distance=4.0,
):
    order = torch.argsort(scores, descending=True)
    selected = []

    for index in order:
        candidate = positions[index]

        if not selected:
            selected.append(candidate)
        else:
            selected_tensor = torch.stack(selected)
            distances = torch.linalg.vector_norm(
                selected_tensor - candidate,
                dim=1,
            )

            if torch.all(distances >= minimum_distance):
                selected.append(candidate)

        if len(selected) == num_targets:
            break

    if len(selected) < num_targets:
        for index in order:
            candidate = positions[index]

            if not any(torch.equal(candidate, p) for p in selected):
                selected.append(candidate)

            if len(selected) == num_targets:
                break

    return torch.stack(selected)

@torch.no_grad()
def matched_filter_batch(
    signal,
    grid,
    receiver_positions,
    transmitter_position,
    fc,
    chirp_slope,
    Ts,
    Tc,
    num_targets,
    propagation_speed=299_792_458.0,
    grid_chunk_size=256,
    candidates_per_target=128,
    minimum_peak_distance=4.0,
):
    device = signal.device

    signal = signal.to(torch.complex64)
    grid = grid.to(device)
    receiver_positions = receiver_positions.to(device)
    transmitter_position = transmitter_position.to(device)

    batch_size, num_receivers, num_samples = signal.shape

    time = (
        torch.arange(num_samples, device=device, dtype=torch.float32)
        * Ts
    )

    signal_energy = (
        torch.sum(torch.abs(signal) ** 2, dim=(1, 2))
        .clamp_min(1e-12)
    )

    maximum_targets = int(num_targets.max().item())
    keep_count = max(
        maximum_targets * candidates_per_target,
        maximum_targets,
    )

    best_scores = torch.full(
        (batch_size, keep_count),
        -torch.inf,
        device=device,
    )

    best_indices = torch.full(
        (batch_size, keep_count),
        -1,
        dtype=torch.long,
        device=device,
    )

    for start in range(0, grid.shape[0], grid_chunk_size):
        end = min(start + grid_chunk_size, grid.shape[0])
        points = grid[start:end]

        tx_distance = torch.linalg.vector_norm(
            points - transmitter_position[None],
            dim=1,
        )

        rx_distance = torch.linalg.vector_norm(
            points[:, None, :] - receiver_positions[None, :, :],
            dim=2,
        )

        tau = (
            tx_distance[:, None] + rx_distance
        ) / propagation_speed

        beta_phase = (
            -2.0 * math.pi * fc * tau
            + math.pi * chirp_slope * tau.square()
        )

        beta = torch.exp(1j * beta_phase)

        beat_phase = (
            -2.0
            * math.pi
            * chirp_slope
            * tau[:, :, None]
            * time[None, None, :]
        )

        template = beta[:, :, None] * torch.exp(1j * beat_phase)

        window = (
            (time[None, None, :] >= tau[:, :, None])
            & (time[None, None, :] <= Tc)
        )

        template = template * window

        correlation = torch.einsum(
            "gmn,bmn->bg",
            template.conj(),
            signal,
        )

        template_energy = (
            torch.sum(torch.abs(template) ** 2, dim=(1, 2))
            .clamp_min(1e-12)
        )

        scores = (
            torch.abs(correlation).square()
            / (
                signal_energy[:, None]
                * template_energy[None, :]
            )
        )

        chunk_indices = torch.arange(
            start,
            end,
            device=device,
        )[None].expand(batch_size, -1)

        combined_scores = torch.cat(
            (best_scores, scores),
            dim=1,
        )

        combined_indices = torch.cat(
            (best_indices, chunk_indices),
            dim=1,
        )

        best_scores, selected = torch.topk(
            combined_scores,
            k=min(keep_count, combined_scores.shape[1]),
            dim=1,
        )

        best_indices = torch.gather(
            combined_indices,
            dim=1,
            index=selected,
        )

    predictions = []

    for batch_index in range(batch_size):
        k = int(num_targets[batch_index].item())

        candidate_positions = grid[best_indices[batch_index]]
        candidate_scores = best_scores[batch_index]

        predicted_positions = select_spatial_peaks(
            candidate_positions,
            candidate_scores,
            num_targets=k,
            minimum_distance=minimum_peak_distance,
        )

        predictions.append(predicted_positions.cpu())

    return predictions


def print_matched_filter_statistics(
    errors,
    coordinate_errors,
):
    import numpy as np
    errors = torch.cat(errors).numpy()
    coordinate_errors = torch.cat(coordinate_errors).numpy()

    print(f"Number of targets: {len(errors)}")
    print(f"Mean error:        {errors.mean():.4f} m")
    print(f"Median error:      {np.median(errors):.4f} m")
    print(f"RMSE:              {np.sqrt(np.mean(errors ** 2)):.4f} m")
    print(f"90th percentile:   {np.percentile(errors, 90):.4f} m")
    print(f"95th percentile:   {np.percentile(errors, 95):.4f} m")
    print(f"99th percentile:   {np.percentile(errors, 99):.4f} m")
    print()
    print(f"X RMSE: {np.sqrt(np.mean(coordinate_errors[:, 0] ** 2)):.4f} m")
    print(f"Y RMSE: {np.sqrt(np.mean(coordinate_errors[:, 1] ** 2)):.4f} m")
    print(f"Z RMSE: {np.sqrt(np.mean(coordinate_errors[:, 2] ** 2)):.4f} m")
    print()
    print(f"Error <= 0.5 m: {100 * np.mean(errors <= 0.5):.3f}%")
    print(f"Error <= 1 m:   {100 * np.mean(errors <= 1.0):.3f}%")
    print(f"Error <= 2 m:   {100 * np.mean(errors <= 2.0):.3f}%")
    print(f"Error <= 5 m:   {100 * np.mean(errors <= 5.0):.3f}%")
    print(f"Error <= 10 m:  {100 * np.mean(errors <= 10.0):.3f}%")

    return {
        "errors": errors,
        "coordinate_errors": coordinate_errors,
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "x_rmse": float(
            np.sqrt(np.mean(coordinate_errors[:, 0] ** 2))
        ),
        "y_rmse": float(
            np.sqrt(np.mean(coordinate_errors[:, 1] ** 2))
        ),
        "z_rmse": float(
            np.sqrt(np.mean(coordinate_errors[:, 2] ** 2))
        ),
    }

def match_prediction_to_ground_truth(
    predicted_positions,
    target_positions,
):
    distance_matrix = torch.cdist(
        predicted_positions.float(),
        target_positions.float(),
    )

    predicted_indices, target_indices = linear_sum_assignment(
        distance_matrix.numpy()
    )

    matched_prediction = predicted_positions[predicted_indices]
    matched_target = target_positions[target_indices]

    errors = torch.linalg.vector_norm(
        matched_prediction - matched_target,
        dim=1,
    )

    coordinate_errors = matched_prediction - matched_target

    return errors, coordinate_errors

def make_local_grid(
    center,
    search_radius=2.0,
    resolution=0.5,
    device="cpu",
):
    center = center.to(device)

    x = torch.arange(
        center[0] - search_radius,
        center[0] + search_radius + resolution / 2,
        resolution,
        device=device,
    )

    y = torch.arange(
        center[1] - search_radius,
        center[1] + search_radius + resolution / 2,
        resolution,
        device=device,
    )

    z = torch.arange(
        center[2] - search_radius,
        center[2] + search_radius + resolution / 2,
        resolution,
        device=device,
    )

    x = x.clamp(-150.0, 150.0).unique()
    y = y.clamp(-150.0, 150.0).unique()
    z = z.clamp(200.0, 300.0).unique()

    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")

    return torch.stack(
        (xx, yy, zz),
        dim=-1,
    ).reshape(-1, 3)

@torch.no_grad()
def evaluate_matched_filter_grid(
    signal,
    grid,
    receiver_positions,
    transmitter_position,
    fc,
    chirp_slope,
    Ts,
    Tc,
    propagation_speed=299_792_458.0,
):
    device = signal.device
    signal = signal.to(torch.complex64)
    grid = grid.to(device)

    num_samples = signal.shape[-1]

    time = (
        torch.arange(
            num_samples,
            device=device,
            dtype=torch.float32,
        )
        * Ts
    )

    tx_distance = torch.linalg.vector_norm(
        grid - transmitter_position[None, :],
        dim=1,
    )

    rx_distance = torch.linalg.vector_norm(
        grid[:, None, :] - receiver_positions[None, :, :],
        dim=2,
    )

    tau = (
        tx_distance[:, None] + rx_distance
    ) / propagation_speed

    beta_phase = (
        -2.0 * math.pi * fc * tau
        + math.pi * chirp_slope * tau.square()
    )

    beat_phase = (
        -2.0
        * math.pi
        * chirp_slope
        * tau[:, :, None]
        * time[None, None, :]
    )

    phase = beta_phase[:, :, None] + beat_phase

    template = torch.exp(1j * phase)

    window = (
        (time[None, None, :] >= tau[:, :, None])
        & (time[None, None, :] <= Tc)
    )

    template = template * window

    correlation = torch.einsum(
        "gmn,mn->g",
        template.conj(),
        signal,
    )

    template_energy = (
        torch.sum(torch.abs(template) ** 2, dim=(1, 2))
        .clamp_min(1e-12)
    )

    signal_energy = (
        torch.sum(torch.abs(signal) ** 2)
        .clamp_min(1e-12)
    )

    return (
        torch.abs(correlation).square()
        / (template_energy * signal_energy)
    )

@torch.no_grad()
def refine_predictions(
    signal,
    coarse_predictions,
    receiver_positions,
    transmitter_position,
    fc,
    chirp_slope,
    Ts,
    Tc,
    propagation_speed=299_792_458.0,
    search_radius=2.0,
    resolution=0.5,
):
    device = signal.device
    refined_predictions = []

    for coarse_position in coarse_predictions:
        local_grid = make_local_grid(
            center=coarse_position.to(device),
            search_radius=search_radius,
            resolution=resolution,
            device=device,
        )

        scores = evaluate_matched_filter_grid(
            signal=signal,
            grid=local_grid,
            receiver_positions=receiver_positions,
            transmitter_position=transmitter_position,
            fc=fc,
            chirp_slope=chirp_slope,
            Ts=Ts,
            Tc=Tc,
            propagation_speed=propagation_speed,
        )

        refined_predictions.append(
            local_grid[torch.argmax(scores)]
        )

    return torch.stack(refined_predictions).cpu()


def main():

    from Utils import common_params
    CP = common_params.getCommonParams()
    
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    test_dataset = RadarMatDatasetMT(
        root_dir="D:\\radar-dataset-multi-targets\\test",
        add_noise=True,
    )

    loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=4,
    )

    receiver_positions = torch.tensor(
        CP.rx_pos,
        dtype=torch.float32,
        device=device,
    )

    if receiver_positions.shape[0] == 3:
        receiver_positions = receiver_positions.T

    transmitter_position = torch.tensor(
        CP.tx_pos,
        dtype=torch.float32,
        device=device,
    )

    grid = make_cartesian_grid(
        x_range=(-150.0, 150.0),
        y_range=(-150.0, 150.0),
        z_range=(200.0, 300.0),
        resolution=2.0,
        device=device,
    )

    all_errors = []
    all_coordinate_errors = []

    for (
        signal,
        signal_clean,
        heatmap,
        coord,
        tau,
        phi,
        snr,
        numTargets,
        sample_id,
    ) in loader:

        signal = torch.complex(signal[:, 0, :, :], signal[:, 1, :, :]).to(device)
        numTargets = numTargets.to(device)

        predictions = matched_filter_batch(
            signal=signal,
            grid=grid,
            receiver_positions=receiver_positions,
            transmitter_position=transmitter_position,
            fc=CP.fc,
            chirp_slope=CP.a,
            Ts=1/CP.fs,
            Tc=CP.Tc,
            num_targets=numTargets,
            propagation_speed=CP.c,
            grid_chunk_size=256,
            candidates_per_target=128,
            minimum_peak_distance=4.0,
        )

        refined_predictions = []
        for batch_index, coarse_predictions in enumerate(predictions):
            refined = refine_predictions(
                signal=signal[batch_index],
                coarse_predictions=coarse_predictions,
                receiver_positions=receiver_positions,
                transmitter_position=transmitter_position,
                fc=CP.fc,
                chirp_slope=CP.a,
                Ts=1/CP.fs,
                Tc=CP.Tc,
                propagation_speed=CP.c,
                search_radius=2.0,
                resolution=0.5,
            )

            refined_predictions.append(refined)

        for batch_index, predicted_positions in enumerate(refined_predictions):
            k = int(numTargets[batch_index].item())

            target_positions = coord[batch_index, :k].cpu()

            errors, coordinate_errors = (
                match_prediction_to_ground_truth(
                    predicted_positions,
                    target_positions,
                )
            )

            all_errors.append(errors)
            all_coordinate_errors.append(coordinate_errors)

    statistics = print_matched_filter_statistics(
        all_errors,
        all_coordinate_errors,
    )

    plt.figure()
    plt.hist(statistics["errors"], bins=50)
    plt.xlabel("Error (m)")
    plt.ylabel("Count")
    plt.title("Localization Error Distribution")
    plt.show()


if __name__ == "__main__":
    main()