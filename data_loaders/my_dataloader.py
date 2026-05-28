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

	def __init__(self, root_dir: str, pattern: str = "sample_*.mat"):
		self.root_dir = Path(root_dir)
		if not self.root_dir.exists():
			raise FileNotFoundError(f"Dataset folder does not exist: {self.root_dir}")

		self.file_paths: List[Path] = sorted(self.root_dir.glob(pattern))
		if not self.file_paths:
			raise ValueError(f"No .mat files found in {self.root_dir} with pattern '{pattern}'")

	def __len__(self) -> int:
		return len(self.file_paths)

	def __getitem__(self, idx: int):
		sample = _load_sample_dict(self.file_paths[idx])
		signal = _to_signal_tensor(sample["y_ell"])
		heatmap = _to_heatmap_tensor(sample["heatmap"])
		coord = _to_coord_tensor(sample["target_xyz"])
		tau = _to_tau_tensor(sample["tau"])
		if heatmap.numel() > 0:
			heatmap = heatmap - heatmap.min()
			heatmap = heatmap / (heatmap.max() + 1e-8)

		return signal, heatmap, coord, tau
        

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
			"heatmap": sample_obj.heatmap,
			"target_xyz": sample_obj.target_xyz,
			"tau": sample_obj.tau,
		}

	if isinstance(sample_obj, np.ndarray) and sample_obj.dtype.names:
		elem = sample_obj.reshape(-1)[0]
		return {
			"y_ell": elem["y_ell"],
			"heatmap": elem["heatmap"],
			"target_xyz": elem["target_xyz"],
			"tau": elem["tau"]
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
			"heatmap": _read_h5_field(f, sample_group, "heatmap"),
			"target_xyz": _read_h5_field(f, sample_group, "target_xyz"),
			"tau": _read_h5_field(f, sample_group, "tau"),
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