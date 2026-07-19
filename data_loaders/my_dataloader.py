from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


loadmat = None

try:
	import h5py
except ImportError:  # pragma: no cover
	h5py = None


class RadarMatDataset(Dataset):
	"""PyTorch dataset for MATLAB samples saved as sample_XXXXXX.mat files."""

	def __init__(self, root_dir: str, pattern: str = "sample_*.mat", add_noise: bool = False):
		self.root_dir = Path(root_dir)
		if not self.root_dir.exists():
			raise FileNotFoundError(f"Dataset folder does not exist: {self.root_dir}")

		self.file_paths: List[Path] = sorted(self.root_dir.glob(pattern))
		if not self.file_paths:
			raise ValueError(f"No .mat files found in {self.root_dir} with pattern '{pattern}'")

		self.add_noise = add_noise

	def __len__(self) -> int:
		return len(self.file_paths)

	def __getitem__(self, idx: int):
		sample = _load_sample_dict(self.file_paths[idx])
		signal = _to_signal_tensor(sample["y_ell"])
		signal_clean = _to_signal_tensor(sample["y_clean"])
		heatmap = _to_heatmap_tensor(sample["heatmap"])
		coord = _to_coord_tensor(sample["target_xyz"])
		tau = _to_tau_tensor(sample["tau"])
		phi = _to_tau_tensor(sample["phi"])
		sample_id = _to_sample_id_tensor(sample["sample_id"])
		snr = sample.get("SNR", sample.get("snr"))
		if snr is None:
			raise KeyError("Field 'SNR' missing in sample")
		if heatmap.numel() > 0:
			heatmap = heatmap - heatmap.min()
			heatmap = heatmap / (heatmap.max() + 1e-8)

		snr_tensor = _to_scalar_tensor(snr)

		if self.add_noise:
			snr_tensor = torch.empty(1).uniform_(-5.0, 20.0).squeeze()
			signal = _add_noise(signal, snr_tensor)

		return signal, signal_clean, heatmap, coord, tau, phi, snr_tensor, sample_id
        

def _add_noise(signal: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
	"""Add complex Gaussian noise to a [2, M, N] real/imag signal tensor.

	Matches the MATLAB noise model in get_radar_response_noisy.m:
		signal_power = 1
		noise_power  = N * signal_power / 10^(SNR_dB/10)
		noise        = sqrt(noise_power/2) * (randn + 1j*randn)

	The two channels of the tensor represent real (0) and imag (1) parts,
	so independent noise is added to each channel with std = sqrt(noise_power/2).
	"""
	N = signal.shape[-1]  # number of time samples
	noise_power = N / (10.0 ** (snr_db.item() / 10.0))  # signal_power = 1
	std = (noise_power / 2.0) ** 0.5
	noise = torch.randn_like(signal) * std
	return signal + noise


def _load_sample_dict(file_path: Path) -> Dict[str, Any]:
	errors = []

	if loadmat is not None:
		try:
			return _load_sample_with_scipy(file_path)
		except Exception as exc:  # pragma: no cover
			errors.append(f"scipy loader failed: {exc}")

	if h5py is not None:
		try:
			return _load_sample_with_h5py(file_path)
		except Exception as exc:  # pragma: no cover
			errors.append(f"h5py loader failed: {exc}")

	error_text = " | ".join(errors) if errors else "No MAT loader is available."
	raise RuntimeError(f"Failed loading {file_path}. {error_text}")


def _load_sample_with_scipy(file_path: Path) -> Dict[str, Any]:
	data = loadmat(file_path, squeeze_me=True, struct_as_record=False)
	if "sample" not in data:
		raise KeyError("MAT file does not contain 'sample'")

	sample_obj = data["sample"]

	if hasattr(sample_obj, "y_ell"):
		return {
			"y_ell": sample_obj.y_ell,
			"y_clean": sample_obj.y_clean,
			"heatmap": sample_obj.heatmap,
			"target_xyz": sample_obj.target_xyz,
			"tau": sample_obj.tau,
			"phi": sample_obj.phi,
			"SNR": getattr(sample_obj, "SNR", None),
			"sample_id": sample_obj.sample_id,
		}

	if isinstance(sample_obj, np.ndarray) and sample_obj.dtype.names:
		elem = sample_obj.reshape(-1)[0]
		has_snr = "SNR" in elem.dtype.names
		return {
			"y_ell": elem["y_ell"],
			"y_clean": elem["y_clean"],
			"heatmap": elem["heatmap"],
			"target_xyz": elem["target_xyz"],
			"tau": elem["tau"],
			"phi": elem["phi"],
			"SNR": elem["SNR"] if has_snr else None,
			"sample_id": elem["sample_id"],
		}
	
def _load_sample_with_h5py(file_path: Path) -> Dict[str, Any]:
	if h5py is None:
		raise RuntimeError("h5py is not installed")

	with h5py.File(file_path, "r") as f:
		if "sample" not in f:
			raise KeyError("MAT file does not contain 'sample'")

		sample_obj = f["sample"]
		sample_group = _resolve_h5_obj(f, sample_obj)

		return {
			"y_ell": _read_h5_field(f, sample_group, "y_ell"),
			"y_clean": _read_h5_field(f, sample_group, "y_clean"),
			"heatmap": _read_h5_field(f, sample_group, "heatmap"),
			"target_xyz": _read_h5_field(f, sample_group, "target_xyz"),
			"tau": _read_h5_field(f, sample_group, "tau"),
			"phi": _read_h5_field(f, sample_group, "phi"),
			"SNR": _read_h5_optional_field(f, sample_group, "SNR"),
			"sample_id": _read_h5_field(f, sample_group, "sample_id"),
		}


def _resolve_h5_obj(f: Any, obj: Any) -> Any:
	if isinstance(obj, h5py.Group):
		return obj

	if isinstance(obj, h5py.Dataset) and obj.dtype == h5py.ref_dtype:
		ref = obj[()].reshape(-1)[0]
		return f[ref]

	return obj


def _read_h5_field(f: Any, sample_group: Any, field_name: str) -> np.ndarray:
	if isinstance(sample_group, h5py.Group):
		field_obj = sample_group[field_name]
	elif isinstance(sample_group, h5py.Dataset):
		if field_name not in sample_group.dtype.names:
			raise KeyError(f"Field '{field_name}' missing in sample struct")
		data = sample_group[field_name][()]
		return _to_numpy_array(data)
	else:
		raise TypeError("Unsupported HDF5 object type for sample")

	if isinstance(field_obj, h5py.Dataset) and field_obj.dtype == h5py.ref_dtype:
		ref = field_obj[()].reshape(-1)[0]
		field_obj = f[ref]

	if isinstance(field_obj, h5py.Dataset):
		data = field_obj[()]
		return _to_numpy_array(data)

	raise TypeError(f"Unsupported field type for '{field_name}'")


def _read_h5_optional_field(f: Any, sample_group: Any, field_name: str) -> np.ndarray | None:
	if isinstance(sample_group, h5py.Group):
		if field_name not in sample_group:
			return None
	elif isinstance(sample_group, h5py.Dataset):
		names = sample_group.dtype.names  # None for non-structured datasets
		if names is None or field_name not in names:
			return None

	return _read_h5_field(f, sample_group, field_name)


def _to_numpy_array(data: Any) -> np.ndarray:
	arr = np.array(data)

	if arr.dtype.names and "real" in arr.dtype.names and "imag" in arr.dtype.names:
		arr = arr["real"] + 1j * arr["imag"]

	# MATLAB/HDF5 stores dimensions in Fortran order; transpose recovers MATLAB layout.
	if arr.ndim >= 2:
		arr = np.transpose(arr)

	return arr


def _to_signal_tensor(signal: Any) -> torch.Tensor:
	arr = np.asarray(signal)

	if np.iscomplexobj(arr):
		arr = np.stack([arr.real, arr.imag], axis=0)
	elif arr.ndim == 3 and arr.shape[0] == 2:
		pass
	elif arr.ndim == 3 and arr.shape[-1] == 2:
		arr = np.transpose(arr, (2, 0, 1))
	else:
		raise ValueError(
			"y_ell must be complex [M,N] or real-imag channels [2,M,N]/[M,N,2]"
		)

	return torch.from_numpy(arr.astype(np.float32, copy=False))


def _to_heatmap_tensor(heatmap: Any) -> torch.Tensor:
	arr = np.asarray(heatmap, dtype=np.float32)
	if arr.ndim == 1:
		return torch.empty(0, dtype=torch.float32)
	if arr.ndim == 2:
		arr = arr[None, ...]
	elif arr.ndim != 3:
		raise ValueError("heatmap must have shape [H,W] or [1,H,W]")

	return torch.from_numpy(arr)


def _to_coord_tensor(coord: Any) -> torch.Tensor:
	arr = np.asarray(coord, dtype=np.float32).reshape(-1)
	if arr.size != 3:
		raise ValueError("target_xyz must contain exactly 3 values")

	return torch.from_numpy(arr)


def _to_tau_tensor(tau: Any) -> torch.Tensor:
	arr = np.asarray(tau, dtype=np.float32).reshape(-1)
	return torch.from_numpy(arr)


def _to_scalar_tensor(value: Any) -> torch.Tensor:
	arr = np.asarray(value, dtype=np.float32).reshape(-1)
	if arr.size == 0:
		raise ValueError("SNR must contain at least one value")
	return torch.tensor(arr[0], dtype=torch.float32)


def _to_sample_id_tensor(value: Any) -> torch.Tensor:
	return torch.tensor(int(np.asarray(value).reshape(-1)[0]), dtype=torch.long)