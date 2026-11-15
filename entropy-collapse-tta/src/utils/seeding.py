"""Seeding utilities.

Why: a research result is only trustworthy if someone else (or future you) can
rerun it and get the same numbers. We seed every random number generator we use.
"""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and (if installed) PyTorch.

    Args:
        seed: the integer seed.
        deterministic: if True, ask PyTorch/cuDNN for deterministic kernels.
            This can be slower, but two runs with the same seed then match.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:  # torch is optional for the numpy-only parts of the project
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Needed for deterministic cuBLAS on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_rng(seed: int) -> np.random.Generator:
    """Return an independent NumPy Generator.

    Prefer passing a Generator to data generators instead of relying on the
    global NumPy state: it keeps experiments independent of each other.
    """
    return np.random.default_rng(seed)
