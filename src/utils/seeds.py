"""Deterministic RNG configuration for NumPy, PyTorch, and Python "random"."""

from __future__ import annotations

import os
import random

import numpy as np

SEEDS = [0, 123, 456, 789, 1011]


def set_seed(seed: int, deterministic_cuda: bool = True) -> None:
    """Set seeds for "random", "numpy", and "torch" (CPU and CUDA).

    Args:
        seed: Integer seed shared across backends.
        deterministic_cuda: If True, prefer deterministic cuDNN algorithms
            when supported (may reduce performance).
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        os.environ.setdefault("PYTHONHASHSEED", str(seed % (2**32)))
    except ImportError:
        pass


def get_seed_from_env(default: int = 0) -> int:
    """Read "AP_SEED" or "SEED" from the environment."""
    for key in ("AP_SEED", "SEED"):
        v = os.environ.get(key)
        if v is not None:
            return int(v)
    return default
