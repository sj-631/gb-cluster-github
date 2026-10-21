from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class NpyClusteringDataset(Dataset):
    """Dataset exposing observations and stable row indices.

    Reference labels are retained separately for evaluation and are deliberately
    excluded from training batches.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray | None = None):
        if x.ndim != 2:
            raise ValueError(f"X must be a 2D array; received shape {x.shape}.")
        if x.shape[0] < 1 or x.shape[1] < 1:
            raise ValueError("X must contain at least one row and one column.")
        if not np.isfinite(x).all():
            raise ValueError("X contains NaN or infinite values.")

        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y: torch.Tensor | None = None
        if y is not None:
            y = np.asarray(y).reshape(-1)
            if x.shape[0] != y.shape[0]:
                raise ValueError(
                    "X and y disagree on sample size: "
                    f"{x.shape[0]} versus {y.shape[0]}."
                )
            if y.dtype.kind in {"f", "c"} and not np.isfinite(y).all():
                raise ValueError("y contains NaN or infinite values.")
            # Remap labels to consecutive integers for evaluation only.
            _, y_consecutive = np.unique(y, return_inverse=True)
            self.y = torch.as_tensor(y_consecutive, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "image_pt": self.x[index],
            "index": torch.tensor(index, dtype=torch.long),
        }


def load_npy_dataset(
    x_path: Path,
    y_path: Path | None = None,
    standardize: str = "none",
) -> Tuple[NpyClusteringDataset, Dict[str, np.ndarray]]:
    x_path = Path(x_path).expanduser()
    y_path = Path(y_path).expanduser() if y_path is not None else None

    if not x_path.exists():
        raise FileNotFoundError(f"X file does not exist: {x_path}")
    if y_path is not None and not y_path.exists():
        raise FileNotFoundError(f"y file does not exist: {y_path}")

    x = np.load(x_path, allow_pickle=False)
    y = np.load(y_path, allow_pickle=False) if y_path is not None else None

    metadata: Dict[str, np.ndarray] = {}
    if standardize == "zscore":
        mean = x.mean(axis=0, keepdims=True)
        scale = x.std(axis=0, keepdims=True)
        scale = np.where(scale < 1e-8, 1.0, scale)
        x = (x - mean) / scale
        # Preserve the exact statistics used before the final float32 dataset
        # conversion so a validator can reconstruct the encoded input.
        metadata["x_mean"] = mean.copy()
        metadata["x_scale"] = scale.copy()
    elif standardize != "none":
        raise ValueError("standardize must be 'none' or 'zscore'.")

    dataset = NpyClusteringDataset(
        x.astype(np.float32),
        np.asarray(y) if y is not None else None,
    )
    return dataset, metadata
