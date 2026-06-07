"""Anomaly preservation metrics: ARD, ARR, and TPS."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def _validate_binary_1d(arr: np.ndarray, name: str) -> np.ndarray:
    """Validate and cast *arr* to a 1-D float64 binary array."""
    arr = np.asarray(arr, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must have at least one element")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    unique = np.unique(arr)
    if not np.all(np.isin(unique, [0.0, 1.0])):
        raise ValueError(
            f"{name} must contain only 0 and 1; got unique values {unique}"
        )
    return arr


def compute_ard_arr(
    y_real: np.ndarray,
    y_syn: np.ndarray,
) -> tuple[float, float]:
    """Compute Anomaly Rate Difference and Anomaly Rate Ratio."""
    y_real = _validate_binary_1d(y_real, "y_real")
    y_syn = _validate_binary_1d(y_syn, "y_syn")

    ar_real = y_real.mean(dtype=np.float64)
    ar_syn = y_syn.mean(dtype=np.float64)

    ard = float(abs(ar_real - ar_syn))

    if ar_real == 0.0 and ar_syn == 0.0:
        arr = 1.0
    elif ar_real == 0.0:
        logger.warning(
            "AR_real == 0 while AR_syn == %.4f; ARR is undefined (returning inf)",
            ar_syn,
        )
        arr = float("inf")
    else:
        arr = float(ar_syn / ar_real)

    return ard, arr


def compute_tps(
    y_real: np.ndarray,
    y_syn: np.ndarray,
) -> float:
    """Compute Temporal Pattern Similarity (TPS)."""
    y_real = _validate_binary_1d(y_real, "y_real")
    y_syn = _validate_binary_1d(y_syn, "y_syn")

    real_anom_count = int(y_real.sum())
    syn_anom_count = int(y_syn.sum())

    if real_anom_count == 0 and syn_anom_count == 0:
        return 0.0
    if real_anom_count == 0 or syn_anom_count == 0:
        logger.warning(
            "TPS: one side has zero anomalies (real=%d, syn=%d); "
            "returning sentinel 1.0",
            real_anom_count,
            syn_anom_count,
        )
        return 1.0

    real_positions = _normalised_positions(y_real)
    syn_positions = _normalised_positions(y_syn)

    return float(stats.wasserstein_distance(real_positions, syn_positions))


def _normalised_positions(y: np.ndarray) -> np.ndarray:
    """Return normalised temporal positions of anomalous windows.

    Positions are in [0, 1]: ``index / (N - 1)`` for N >= 2, else 0.5.
    """
    n = y.shape[0]
    indices = np.where(y == 1.0)[0].astype(np.float64)
    if n >= 2:
        indices /= float(n - 1)
    else:
        indices[:] = 0.5
    return indices


def compute_all_preservation(
    y_real: np.ndarray,
    y_syn: np.ndarray,
) -> dict[str, float]:
    """Convenience wrapper returning all three preservation metrics.

    Args:
        y_real: 1-D binary labels for real ``test_det`` windows.
        y_syn: 1-D binary labels for synthetic windows (same detector
            and threshold).

    Returns:
        ``{"ard": ..., "arr": ..., "tps": ...}``.
    """
    ard, arr = compute_ard_arr(y_real, y_syn)
    tps = compute_tps(y_real, y_syn)
    return {"ard": ard, "arr": arr, "tps": tps}


__all__ = [
    "compute_ard_arr",
    "compute_tps",
    "compute_all_preservation",
]
