"""Fidelity metrics: per-feature KS and Wasserstein-1 distance (averaged)."""

from __future__ import annotations


import numpy as np
from scipy import stats

from src.utils.tensor import keras_to_numpy as _to_numpy


def compute_ks_wasserstein(
    real: np.ndarray,
    synthetic: np.ndarray,
) -> tuple[float, float]:
    """Compute mean KS statistic and mean Wasserstein distance per feature."""
    real = _to_numpy(real)
    synthetic = _to_numpy(synthetic)
    if real.ndim == 3:
        n_r, l_r, f_r = real.shape
        n_s, l_s, f_s = synthetic.shape
        if f_r != f_s:
            raise ValueError(f"Feature mismatch: {f_r} vs {f_s}")
        real = real.reshape(-1, f_r)
        synthetic = synthetic.reshape(-1, f_s)
    elif real.ndim == 2:
        f_r = real.shape[1]
        if synthetic.shape[1] != f_r:
            raise ValueError("Feature dimensions must match")
    else:
        raise ValueError(f"Expected 2D or 3D real array, got shape {real.shape}")

    ks_list: list[float] = []
    w_list: list[float] = []
    for j in range(real.shape[1]):
        a = real[:, j].astype(np.float64)
        b = synthetic[:, j].astype(np.float64)
        ks_list.append(float(stats.ks_2samp(a, b).statistic))
        w_list.append(float(stats.wasserstein_distance(a, b)))
    return float(np.mean(ks_list)), float(np.mean(w_list))


def fidelity_objective(real: np.ndarray, synthetic: np.ndarray) -> float:
    """Composite "(KS_mean + Wasserstein_mean) / 2" (minimize)."""
    ks, w = compute_ks_wasserstein(real, synthetic)
    if not np.isfinite(ks) or not np.isfinite(w):
        return 1e6
    return (ks + w) / 2.0
