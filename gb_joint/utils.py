from __future__ import annotations

import json
import hashlib
import random
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(path: Path, values: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2, sort_keys=True, allow_nan=False)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(
    project_root: Path,
    extra_paths: Iterable[Path] = (),
) -> Dict[str, Any]:
    """Hash the executable project sources that determine a fitted run."""
    root = Path(project_root).expanduser().resolve()
    candidates = (
        [root / "train.py"]
        + sorted((root / "gb_joint").glob("*.py"))
        + sorted((root / "scripts").glob("*.R"))
        + [Path(path).expanduser().resolve() for path in extra_paths]
    )
    files: Dict[str, str] = {}
    for path in candidates:
        if path.is_file():
            files[str(path.relative_to(root))] = sha256_file(path)
    combined = hashlib.sha256()
    for relative_path, file_hash in sorted(files.items()):
        combined.update(relative_path.encode("utf-8"))
        combined.update(b"\0")
        combined.update(file_hash.encode("ascii"))
        combined.update(b"\n")
    return {
        "algorithm": "sha256",
        "project_root_at_run": str(root),
        "combined_sha256": combined.hexdigest(),
        "files": files,
    }


def npy_file_manifest(path: Path) -> Dict[str, Any]:
    """Record immutable identity and array metadata for an NPY input."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "exists": False}
    array = np.load(resolved, mmap_mode="r", allow_pickle=False)
    return {
        "path": str(resolved),
        "exists": True,
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def data_manifest(x_path: Path, y_path: Path | None) -> Dict[str, Any]:
    return {
        "format": "npy-input-manifest-v1",
        "x": npy_file_manifest(x_path),
        "y": npy_file_manifest(y_path) if y_path is not None else None,
    }
