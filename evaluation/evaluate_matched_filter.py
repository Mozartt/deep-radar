
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import math
from typing import Iterator, Sequence

import torch
from data_loaders.dataloader_multi_targets import RadarMatDatasetMT
from torch.utils.data import DataLoader


@torch.no_grad()
def iter_cylindrical_grid(
    radius_max: float = 150.0,
    z_range: Sequence[float] = (200.0, 300.0),
    dr: float = 0.5,
    dz: float = 0.5,
    arc_resolution: float = 0.5,
    center_xy: Sequence[float] = (0.0, 0.0),
    chunk_size: int = 512,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Iterator[torch.Tensor]:

    z_min, z_max = map(float, z_range)
    if z_max < z_min:
        raise ValueError("z_range must satisfy z_max >= z_min.")

    center_x, center_y = map(float, center_xy)

    radii = torch.arange(
        0.0,
        radius_max + 0.5 * dr,
        dr,
        dtype=torch.float64,
    )

    z_values = torch.arange(
        z_min,
        z_max + 0.5 * dz,
        dz,
        dtype=torch.float64,
    )

    point_buffer = []

    for z in z_values.tolist():
        for radius in radii.tolist():
            if radius < 1e-12:
                ring = torch.tensor(
                    [[center_x, center_y, z]],
                    dtype=dtype,
                )
            else:
                # Number of points needed to keep arc spacing near
                # arc_resolution meters.
                number_of_angles = max(
                    1,
                    math.ceil(2.0 * math.pi * radius / arc_resolution),
                )

                theta = (
                    torch.arange(number_of_angles, dtype=dtype)
                    * (2.0 * math.pi / number_of_angles)
                )

                x = center_x + radius * torch.cos(theta)
                y = center_y + radius * torch.sin(theta)
                z_ring = torch.full_like(x, z)

                ring = torch.stack((x, y, z_ring), dim=-1)

            point_buffer.append(ring)

            buffered_points = sum(points.shape[0] for points in point_buffer)

            if buffered_points >= chunk_size:
                points = torch.cat(point_buffer, dim=0)

                while points.shape[0] >= chunk_size:
                    yield points[:chunk_size].to(device=device)
                    points = points[chunk_size:]

                point_buffer = [points] if points.shape[0] > 0 else []

    if point_buffer:
        yield torch.cat(point_buffer, dim=0).to(device=device)

@torch.no_grad()
def matched_filter_cylindrical_grid(
    y: torch.Tensor,
    receiver_positions: torch.Tensor,
    transmitter_position: torch.Tensor,
    fc: float,
    chirp_slope: float,
    Ts: float,
    Tc: float,
    propagation_speed: float = 299_792_458.0,
    radius_max: float = 150.0,
    z_range: Sequence[float] = (200.0, 300.0),
    dr: float = 0.5,
    dz: float = 0.5,
    arc_resolution: float = 0.5,
    center_xy: Sequence[float] = (0.0, 0.0),
    chunk_size: int = 256,
    device: str | torch.device | None = None,
    return_all_scores: bool = False,
) -> dict:

    if device is None:
        device = y.device if isinstance(y, torch.Tensor) else "cpu"

    device = torch.device(device)

    y = torch.as_tensor(y, device=device)
    receiver_positions = torch.as_tensor(
        receiver_positions,
        dtype=torch.float32,
        device=device,
    )
    transmitter_position = torch.as_tensor(
        transmitter_position,
        dtype=torch.float32,
        device=device,
    )

    if y.ndim != 2:
        raise ValueError(
            f"y must have shape [M, N], but received {tuple(y.shape)}."
        )

    if not torch.is_complex(y):
        raise ValueError("y must be a complex-valued tensor.")

    number_of_receivers, number_of_samples = y.shape

    if receiver_positions.shape != (number_of_receivers, 3):
        raise ValueError(
            "receiver_positions must have shape [M, 3], where M "
            "matches the first dimension of y."
        )

    if transmitter_position.shape != (3,):
        raise ValueError("transmitter_position must have shape [3].")

    # Complex64 normally provides sufficient precision and is considerably
    # faster and smaller than complex128 on a GPU.
    y = y.to(torch.complex64)

    sample_indices = torch.arange(
        number_of_samples,
        dtype=torch.float32,
        device=device,
    )
    sample_times = Ts * sample_indices

    y_energy = torch.sum(torch.abs(y) ** 2).clamp_min(1e-12)

    best_score = torch.tensor(
        -torch.inf,
        dtype=torch.float32,
        device=device,
    )
    best_position = None
    num_candidates = 0

    all_points = []
    all_scores = []

    grid_iterator = iter_cylindrical_grid(
        radius_max=radius_max,
        z_range=z_range,
        dr=dr,
        dz=dz,
        arc_resolution=arc_resolution,
        center_xy=center_xy,
        chunk_size=chunk_size,
        device=device,
        dtype=torch.float32,
    )

    for candidate_positions in grid_iterator:
        # candidate_positions: [G, 3]
        G = candidate_positions.shape[0]
        num_candidates += G

        # Transmitter-to-target distance: [G]
        transmitter_distance = torch.linalg.vector_norm(
            candidate_positions - transmitter_position[None, :],
            dim=-1,
        )

        # Target-to-receiver distance: [G, M]
        receiver_distance = torch.linalg.vector_norm(
            candidate_positions[:, None, :]
            - receiver_positions[None, :, :],
            dim=-1,
        )

        # Propagation delay: [G, M]
        tau = (
            transmitter_distance[:, None] + receiver_distance
        ) / propagation_speed

        # beta from the MATLAB implementation: [G, M]
        beta_phase = (
            -2.0 * math.pi * fc * tau
            + math.pi * chirp_slope * tau.square()
        )
        beta = torch.exp(1j * beta_phase)

        # Beat-frequency phase: [G, M, N]
        beat_phase = (
            -2.0
            * math.pi
            * chirp_slope
            * tau[:, :, None]
            * sample_times[None, None, :]
        )

        template = beta[:, :, None] * torch.exp(1j * beat_phase)

        # Same window as:
        #   win = (t >= tau.') & (t <= Tc)
        active_window = (
            (sample_times[None, None, :] >= tau[:, :, None])
            & (sample_times[None, None, :] <= Tc)
        )

        template = template * active_window

        # Complex matched filter:
        # sum_m,n conj(template[m,n]) * y[m,n]
        correlation = torch.einsum(
            "gmn,mn->g",
            template.conj(),
            y,
        )

        template_energy = torch.sum(
            torch.abs(template) ** 2,
            dim=(1, 2),
        ).clamp_min(1e-12)

        # Squared normalized correlation, typically between 0 and 1.
        scores = (
            torch.abs(correlation).square()
            / (template_energy * y_energy)
        )

        local_best_score, local_best_index = torch.max(scores, dim=0)

        if local_best_score > best_score:
            best_score = local_best_score
            best_position = candidate_positions[local_best_index].clone()

        if return_all_scores:
            all_points.append(candidate_positions.cpu())
            all_scores.append(scores.cpu())

    if best_position is None:
        raise RuntimeError("The generated search grid was empty.")

    center_x, center_y = center_xy
    relative_x = best_position[0] - center_x
    relative_y = best_position[1] - center_y

    best_radius = torch.sqrt(relative_x.square() + relative_y.square())
    best_theta = torch.atan2(relative_y, relative_x)
    best_theta = torch.remainder(best_theta, 2.0 * math.pi)

    result = {
        "best_position": best_position.cpu(),
        "best_cylindrical": torch.stack(
            (best_radius, best_theta, best_position[2])
        ).cpu(),
        "best_score": float(best_score.cpu()),
        "num_candidates": num_candidates,
    }

    if return_all_scores:
        result["grid_points"] = torch.cat(all_points, dim=0)
        result["scores"] = torch.cat(all_scores, dim=0)

    return result

def main():
    device = "cuda"

    test_dataset = RadarMatDatasetMT(root_dir="D:\\radar-dataset-multi-targets\\test", add_noise=True)
    loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)
    
    from Utils import common_params
    CP = common_params.getCommonParams()

    for signal, signal_clean, heatmap, coord, tau, phi, snr, numTargets, sample_id in loader:

        for b in range(signal.shape[0]):

            # y has shape [M, N], complex
            y_torch = torch.complex(signal[b, 0, :, :], signal[b, 1, :, :]).to(
                device=device,
                dtype=torch.complex64,
            ).squeeze()

            # If MATLAB q has shape [3, M], transpose it to [M, 3].
            q_torch = CP.rx_pos.to(
                device=device,
                dtype=torch.float32,
            )

            P_tx_torch = CP.tx_pos.to(
                dtype=torch.float32,
                device=device,
            )

            result = matched_filter_cylindrical_grid(
                y=y_torch,
                receiver_positions=q_torch,
                transmitter_position=P_tx_torch,
                fc=CP.fc,
                chirp_slope=CP.a,
                Ts=1 / CP.fs,
                Tc=CP.Tc,
                propagation_speed=CP.c,

                radius_max=150.0,
                z_range=(200.0, 300.0),

                # Approximately 0.5 m Cartesian-equivalent resolution.
                dr=0.5,
                dz=0.5,
                arc_resolution=0.5,

                chunk_size=256,
                device=device,
                return_all_scores=False,
            )

            print("Best Cartesian position [x, y, z]:")
            print(result["best_position"])

            print("Best cylindrical position [r, theta, z]:")
            print(result["best_cylindrical"])

            print("Normalized matched-filter score:")
            print(result["best_score"])

            print("Number of evaluated points:")
            print(result["num_candidates"])

if __name__ == "__main__":
    main()