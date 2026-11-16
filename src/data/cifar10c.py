"""CIFAR-10-C on disk: one .npy per corruption + labels.npy (Hendrycks & Dietterich; Zenodo 2535967).

Expected layout [to be verified by `load_corruption`'s shape checks and by scripts/check_setup.py]:
    <root>/<corruption>.npy : uint8, shape (5 * 10000, 32, 32, 3), severities 1..5 stacked in order
    <root>/labels.npy       : int,   shape (5 * 10000,), the 10k test labels repeated per severity
Severity s occupies rows [(s-1)*10000, s*10000). We never trust this silently: shapes are asserted, and
check_setup.py confirms that severity-5 error of the source model is higher than severity-1 error.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# The 15 benchmark corruptions, in the order used by the usual continual-TTA benchmarks.
CORRUPTIONS_15: tuple[str, ...] = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur",
    "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression",
)
N_PER_SEVERITY = 10_000
N_SEVERITIES = 5


def load_corruption(
    root: str | Path, name: str, severity: int, n_per_severity: int = N_PER_SEVERITY
) -> tuple[np.ndarray, np.ndarray]:
    """Return (images uint8 (n, 32, 32, 3), labels int64 (n,)) for one corruption at one severity.

    `n_per_severity` exists only so unit tests can use tiny fake files.
    """
    if not 1 <= severity <= N_SEVERITIES:
        raise ValueError(f"severity must be in 1..{N_SEVERITIES}, got {severity}")
    root = Path(root)
    x_path, y_path = root / f"{name}.npy", root / "labels.npy"
    for p in (x_path, y_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run scripts/download_cifar10c.py (or point data.root at the folder).")
    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    total = N_SEVERITIES * n_per_severity
    if x.shape != (total, 32, 32, 3) or x.dtype != np.uint8:
        raise ValueError(f"{x_path}: expected uint8 {(total, 32, 32, 3)}, got {x.dtype} {x.shape}. "
                         "Layout differs from the assumed 5 stacked severities; do not use this data until checked.")
    if y.shape != (total,):
        raise ValueError(f"{y_path}: expected shape {(total,)}, got {y.shape}")
    sl = slice((severity - 1) * n_per_severity, severity * n_per_severity)
    return np.ascontiguousarray(x[sl]), np.asarray(y[sl], dtype=np.int64)
