"""Shared helpers for rebuilding real / synthetic windows for evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.data.preprocessor import (
    ScalerName,
    load_raw_splits,
    prepare_train_gen_windows,
)
from src.training.utils import (
    load_best_params,
    load_model_preprocessing_cfg,
    repo_root,
)


def prepare_real_windows(
    root: Path,
    dataset: str,
    params: dict[str, Any],
    model_name: str,
) -> np.ndarray:
    """Rebuild the scaled ``train_gen`` windows for ``(dataset, model)``."""
    window_size = int(params["window_size"])
    stride = int(params["stride"])
    scaler_type: ScalerName = str(params["scaler_type"])  # type: ignore[assignment]

    processed = root / "data" / "processed" / dataset
    splits = load_raw_splits(processed)
    profile, max_ar, buf = load_model_preprocessing_cfg(model_name)
    x_scaled, _scaler, _meta = prepare_train_gen_windows(
        splits,
        window_size=window_size,
        stride=stride,
        scaler_name=scaler_type,
        profile=profile,  # type: ignore[arg-type]
        max_anomaly_ratio=max_ar,
        buffer=buf,
        min_windows=1,
    )
    if x_scaled is None:
        raise RuntimeError(
            f"Preprocessing yielded no usable windows for {dataset}/{model_name}"
        )
    return np.asarray(x_scaled, dtype=np.float32)


def load_synthetic(
    root: Path, dataset: str, model: str, seed: int
) -> np.ndarray | None:
    """Load one synthetic seed file or return ``None`` if it does not exist."""
    path = root / "data" / "synthetic" / dataset / model / f"seed_{seed}.npy"
    if not path.is_file():
        return None
    arr = np.load(path)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return np.asarray(arr, dtype=np.float32)


def gaussian_target_window(root: Path, dataset: str) -> int | None:
    """Peek at the first ``GaussianNoise`` seed file to learn its window length."""
    g_dir = root / "data" / "synthetic" / dataset / "GaussianNoise"
    if not g_dir.is_dir():
        return None
    for npy in sorted(g_dir.glob("seed_*.npy")):
        try:
            arr = np.load(npy, mmap_mode="r")
        except (OSError, ValueError):
            continue
        if arr.ndim >= 2:
            return int(arr.shape[1])
    return None


def best_params_or_gaussian(
    dataset: str,
    model_name: str,
    *,
    target_window: int | None = None,
) -> dict[str, Any]:
    """Resolve preprocessing for any model, including the Gaussian baseline."""
    if model_name != "GaussianNoise":
        return load_best_params(dataset, model_name)

    results_dir = repo_root() / "results"
    fallback: dict[str, Any] | None = None
    fallback_name: str | None = None
    for candidate in sorted(results_dir.glob(f"best_params_{dataset}_*.json")):
        with open(candidate, encoding="utf-8") as f:
            data = json.load(f)
        if not all(k in data for k in ("window_size", "stride", "scaler_type")):
            continue
        if fallback is None:
            fallback = data
            fallback_name = candidate.name
        if target_window is not None and int(data["window_size"]) == target_window:
            return data
    if fallback is None:
        raise FileNotFoundError(
            f"No best_params_{dataset}_*.json found to inherit preprocessing for "
            f"GaussianNoise on {dataset}"
        )
    if target_window is not None:
        print(
            f"  WARNING: no best_params_{dataset}_*.json with "
            f"window_size={target_window}; falling back to "
            f"{fallback_name} (window_size={fallback['window_size']})"
        )
    return fallback


__all__ = [
    "prepare_real_windows",
    "load_synthetic",
    "gaussian_target_window",
    "best_params_or_gaussian",
]
