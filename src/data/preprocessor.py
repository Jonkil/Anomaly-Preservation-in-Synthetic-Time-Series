"""Temporal splits, sliding windows, and TSB-AD CSV loading (no leakage)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

ScalerName = Literal["Standard", "MinMax", "Robust"]


class EmptyAfterFilteringError(ValueError):
    """Raised when the contamination filter removes every sliding window.

    Distinct from generic ''ValueError'' so callers can decide between
    "skip this trial" (for tuning) and "abort the pipeline" (for final
    training) without resorting to broad ''except'' clauses (see
    ''robust-code.mdc'' §8).
    """


class InsufficientWindowsError(ValueError):
    """Raised when fewer windows survive than the configured minimum.

    Used by :func:`prepare_train_gen_windows` instead of the previous
    ''(None, None, meta)'' sentinel return.
    """


def temporal_split(
    series: np.ndarray,
    labels: np.ndarray,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, np.ndarray]:
    """Split multivariate series into three contiguous temporal segments.

    Args:
        series: Shape "(T, F)" - time-major, no windowing applied.
        labels: Shape "(T,)" - point-level anomaly labels (0/1).
        ratios: "(train_gen, train_det, test_det)" fractions summing to 1.

    Returns:
        Dictionary with keys "train_gen", "train_det", "test_det" for
        both "values" and "labels" (six arrays total as flat keys).
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {sum(ratios)}")
    n = series.shape[0]
    s1 = int(n * ratios[0])
    s2 = int(n * (ratios[0] + ratios[1]))
    return {
        "train_gen": series[:s1].copy(),
        "train_gen_labels": labels[:s1].copy(),
        "train_det": series[s1:s2].copy(),
        "train_det_labels": labels[s1:s2].copy(),
        "test_det": series[s2:].copy(),
        "test_det_labels": labels[s2:].copy(),
    }


def sliding_window(
    data: np.ndarray,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """Build sliding windows over a single contiguous segment (no boundaries).

    Args:
        data: Shape "(T, F)" or "(T,)" (treated as "(T, 1)").
        window_size: Window length in time steps.
        stride: Step between window starts.

    Returns:
        Array of shape "(n_windows, window_size, F)".
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if window_size < 1 or stride < 1:
        raise ValueError("window_size and stride must be >= 1")
    n = data.shape[0]
    n_windows = (n - window_size) // stride + 1
    if n_windows < 1:
        raise ValueError(
            f"window_size {window_size} too large for series length {n} "
            f"with stride {stride}"
        )
    shape = (n_windows, window_size, data.shape[1])
    strides = (data.strides[0] * stride, data.strides[0], data.strides[1])
    return np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides, writeable=False
    ).copy()


def get_scaler(name: ScalerName) -> BaseEstimator:
    """Return a fresh sklearn scaler instance."""
    if name == "Standard":
        return StandardScaler()
    if name == "MinMax":
        return MinMaxScaler(feature_range=(0.0, 1.0))
    if name == "Robust":
        return RobustScaler()
    raise ValueError(f"Unknown scaler: {name}")


def fit_scaler_on_windows(
    windows: np.ndarray,
    scaler_name: ScalerName,
) -> tuple[np.ndarray, BaseEstimator]:
    """Fit scaler on flattened gen-train windows and return scaled windows.

    Args:
        windows: Shape "(N, L, F)".
        scaler_name: One of "Standard", "MinMax", "Robust".

    Returns:
        Tuple of scaled windows (same shape) and the fitted scaler.
    """
    scaler = get_scaler(scaler_name)
    flat = windows.reshape(-1, windows.shape[-1])
    scaler.fit(flat)
    scaled_flat = scaler.transform(flat)
    np.nan_to_num(scaled_flat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    out = scaled_flat.reshape(windows.shape).astype(np.float32)
    return out, scaler


def transform_windows(
    windows: np.ndarray,
    scaler: BaseEstimator,
) -> np.ndarray:
    """Transform windows with an already-fitted scaler."""
    flat = windows.reshape(-1, windows.shape[-1])
    out = scaler.transform(flat)
    return out.reshape(windows.shape)


def load_tsb_csv(
    csv_path: Path,
    label_column: str = "Label",
) -> tuple[np.ndarray, np.ndarray]:
    """Load a TSB-AD multivariate CSV: all columns except label are features.

    Args:
        csv_path: Path to CSV file.
        label_column: Name of the anomaly label column.

    Returns:
        "values" float64 "(T, F)", "labels" int64 "(T,)".
    """
    df = pd.read_csv(csv_path)
    if label_column not in df.columns:
        raise ValueError(f"Missing label column {label_column!r} in {csv_path}")
    labels = df[label_column].to_numpy(dtype=np.int64)
    feat = df.drop(columns=[label_column]).to_numpy(dtype=np.float64)
    return feat, labels


def save_raw_splits(
    out_dir: Path,
    splits: dict[str, np.ndarray],
    meta: dict[str, Any],
) -> None:
    """Save numpy arrays and a small "metadata.json" for reproducibility."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, arr in splits.items():
        np.save(out_dir / f"{k}.npy", arr)
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_raw_splits(processed_dir: Path) -> dict[str, np.ndarray]:
    """Load arrays saved by :func:'save_raw_splits'."""
    keys = [
        "train_gen",
        "train_gen_labels",
        "train_det",
        "train_det_labels",
        "test_det",
        "test_det_labels",
    ]
    out: dict[str, np.ndarray] = {}
    for k in keys:
        p = processed_dir / f"{k}.npy"
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}")
        out[k] = np.load(p)
    return out


def window_labels_from_point_labels(
    point_labels: np.ndarray,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """Label each window 1 if any contained point is anomalous."""
    w = sliding_window(point_labels.reshape(-1, 1), window_size, stride)
    return (w.max(axis=(1, 2)) >= 1).astype(np.int64)


def compute_window_anomaly_ratio(
    point_labels: np.ndarray,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """Return per-window anomaly fraction aligned with :func:`sliding_window`.

    Args:
        point_labels: Shape "(T,)" binary point-level anomaly labels.
        window_size: Window length in time steps.
        stride: Step between window starts (same as sliding_window).

    Returns:
        Array of shape "(N_windows,)" float64 with anomaly fractions in [0, 1].
    """
    w = sliding_window(point_labels.reshape(-1, 1), window_size, stride)
    return w.reshape(w.shape[0], -1).mean(axis=1).astype(np.float64)


def filter_anomaly_windows(
    windows: np.ndarray,
    anomaly_ratio: np.ndarray,
    max_anomaly_ratio: float = 0.05,
    buffer: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop windows whose anomaly fraction exceeds a threshold, plus neighbours.

    Implements the "contamination buffer" recommended for TimeVAE training:
    windows whose anomaly ratio is above `max_anomaly_ratio` are dropped, and
    an additional `±buffer` windows on each side are dropped too since
    anomalies often have precursor/recovery dynamics.

    Args:
        windows: Shape "(N, L, F)" sliding windows.
        anomaly_ratio: Shape "(N,)" per-window anomaly fractions.
        max_anomaly_ratio: Drop windows with ratio strictly above this value.
        buffer: Additional neighbours (in window index) dropped on each side.

    Returns:
        Tuple ''(kept_windows, keep_mask)'' where ''keep_mask'' is the boolean
        mask applied to the leading axis of ''windows'' (shape ''(N,)'').
    """
    if windows.shape[0] != anomaly_ratio.shape[0]:
        raise ValueError(
            f"windows N={windows.shape[0]} but anomaly_ratio N="
            f"{anomaly_ratio.shape[0]}"
        )
    n = windows.shape[0]
    contaminated = anomaly_ratio > max_anomaly_ratio
    if buffer > 0 and contaminated.any():
        expanded = np.zeros(n, dtype=bool)
        idxs = np.where(contaminated)[0]
        for i in idxs:
            lo = max(0, int(i) - buffer)
            hi = min(n, int(i) + buffer + 1)
            expanded[lo:hi] = True
        keep = ~expanded
    else:
        keep = ~contaminated
    if not keep.any():
        raise EmptyAfterFilteringError(
            "filter_anomaly_windows removed every window; relax "
            f"max_anomaly_ratio={max_anomaly_ratio} or buffer={buffer}"
        )
    return windows[keep], keep


def per_window_znorm(
    windows: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-window, per-feature z-score normalisation.

    Each window's shape is computed independently so the reconstruction target
    is the local pattern rather than the global level.

    Args:
        windows: Shape "(N, L, F)" real-valued windows.
        eps: Added to std to avoid division by zero.

    Returns:
        Tuple ''(z, stats)'' where ''z'' has the same shape as ''windows'' and
        ''stats'' is ''(N, 2, F)'' - channel 0 holds per-window means and
        channel 1 holds per-window stds (with ''eps'' already added).
    """
    if windows.ndim != 3:
        raise ValueError(f"Expected 3D windows (N, L, F); got {windows.shape}")
    mu = windows.mean(axis=1, keepdims=True)  # (N, 1, F)
    sigma = windows.std(axis=1, keepdims=True) + eps
    z = (windows - mu) / sigma
    stats = np.concatenate([mu, sigma], axis=1)  # (N, 2, F)
    return z.astype(np.float32), stats.astype(np.float32)


def sample_inverse_stats(
    stats: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample ''n'' per-window (mean, std) pairs with replacement from ''stats''.

    Used to re-level synthetic windows so downstream detectors see the same
    approximate value range as the real data.

    Args:
        stats: Array of shape "(N, 2, F)" from :func:`per_window_znorm`.
        n: Number of pairs to sample.
        rng: NumPy :class:`~numpy.random.Generator` for reproducibility.

    Returns:
        Array of shape "(n, 2, F)".
    """
    if stats.ndim != 3 or stats.shape[1] != 2:
        raise ValueError(f"Expected stats shape (N, 2, F); got {stats.shape}")
    if n <= 0:
        raise ValueError("n must be positive")
    idx = rng.integers(0, stats.shape[0], size=int(n))
    return stats[idx].astype(np.float32)


def apply_inverse_znorm(
    z_windows: np.ndarray,
    stats: np.ndarray,
) -> np.ndarray:
    """Undo :func:`per_window_znorm` with provided per-window ''stats''.

    Args:
        z_windows: Shape "(N, L, F)" z-normalised windows.
        stats: Shape "(N, 2, F)"; must have the same leading ''N'' as
            ''z_windows''. Channel 0 = mean, channel 1 = std.

    Returns:
        Array of shape "(N, L, F)" on the original level/scale.
    """
    if z_windows.shape[0] != stats.shape[0]:
        raise ValueError(
            f"z_windows N={z_windows.shape[0]} but stats N={stats.shape[0]}"
        )
    mu = stats[:, 0:1, :]
    sigma = stats[:, 1:2, :]
    return (z_windows * sigma + mu).astype(np.float32)


@dataclass
class PerWindowZNormScaler:
    """Lightweight "scaler" that records train-time per-window statistics.

    Unlike sklearn scalers, ''transform'' recomputes mean/std on each new
    window. The stored ''train_stats'' are used only to sample realistic
    levels when inverse-transforming synthetic windows via
    :func:`sample_inverse_stats`.
    """

    train_stats: np.ndarray  # (N_train, 2, F)
    feat_dim: int
    eps: float = 1e-8

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """Per-window z-score new windows; ignores ''train_stats''."""
        z, _ = per_window_znorm(windows, eps=self.eps)
        return z

    def fit_transform(self, windows: np.ndarray) -> np.ndarray:
        return self.transform(windows)

    def inverse_transform_sample(
        self,
        z_windows: np.ndarray,
        seed: int | None = None,
    ) -> np.ndarray:
        """Re-level synthetic windows with stats sampled from the train set."""
        rng = np.random.default_rng(seed)
        stats = sample_inverse_stats(self.train_stats, z_windows.shape[0], rng)
        return apply_inverse_znorm(z_windows, stats)


PreprocessingProfile = Literal["legacy", "improved"]


def prepare_train_gen_windows(
    splits: dict[str, np.ndarray],
    window_size: int,
    stride: int,
    scaler_name: ScalerName,
    *,
    profile: PreprocessingProfile = "legacy",
    max_anomaly_ratio: float = 0.05,
    buffer: int = 0,
    min_windows: int = 32,
) -> tuple[np.ndarray, BaseEstimator | PerWindowZNormScaler, dict[str, Any]]:
    """Build and normalise training windows under a preprocessing profile.

    Two profiles are supported:

    - ''legacy'': slide windows then fit a global sklearn scaler
      (:class:`~sklearn.preprocessing.StandardScaler` / ''MinMaxScaler'' /
      ''RobustScaler'') on the flattened windows. This matches the original
      pipeline and keeps existing results reproducible.
    - ''improved'': slide windows, drop contaminated windows via
      :func:`filter_anomaly_windows` (''max_anomaly_ratio'' with optional
      neighbour ''buffer''), then apply :func:`per_window_znorm`. The
      returned "scaler" is a :class:`PerWindowZNormScaler` that records the
      train-time per-window statistics so synthetic windows can be
      re-levelled at generation time via :func:`sample_inverse_stats`.

    Args:
        splits: Output of :func:`load_raw_splits`; must contain
            ''train_gen'' and, for the improved profile, ''train_gen_labels''.
        window_size: Window length in time steps.
        stride: Step between window starts.
        scaler_name: Sklearn scaler name; unused in the improved profile but
            still stored in the returned metadata for provenance.
        profile: ''"legacy"'' or ''"improved"''.
        max_anomaly_ratio: For the improved profile, drop windows whose
            label fraction exceeds this value.
        buffer: Number of neighbouring windows to drop on each side of
            every contaminated window (''±buffer'').
        min_windows: If fewer windows survive than this, returns
            ''(None, None, meta)''.

    Returns:
        Triple ''(x_scaled, scaler, meta)''. ''x_scaled'' is the scaled /
        normalised training windows. ''scaler'' is the fitted sklearn
        scaler (legacy profile) or :class:`PerWindowZNormScaler`
        (improved profile). ''meta'' always contains ''profile'' plus
        counts and filtering parameters used.

    Raises:
        InsufficientWindowsError: If fewer than ''min_windows'' raw or
            kept windows are available. Distinct exception type lets
            tuning loops decide between "skip this trial" and "abort".
        EmptyAfterFilteringError: If the contamination filter removes
            every window in the improved profile.
        ValueError: For configuration errors (unknown profile, missing
            labels for the improved profile, etc.).
    """
    tg = splits["train_gen"]
    n_raw = _min_windows(tg.shape[0], window_size, stride)
    meta: dict[str, Any] = {
        "profile": profile,
        "window_size": int(window_size),
        "stride": int(stride),
        "scaler_type": str(scaler_name),
        "max_anomaly_ratio": float(max_anomaly_ratio),
        "buffer": int(buffer),
        "n_windows_raw": int(n_raw),
        "n_windows_kept": 0,
    }
    if n_raw < min_windows:
        raise InsufficientWindowsError(
            f"only {n_raw} raw windows of size {window_size} "
            f"with stride {stride} fit in train_gen of length "
            f"{tg.shape[0]}; need >= {min_windows}"
        )

    raw_win = sliding_window(tg, window_size, stride)

    if profile == "legacy":
        scaled, scaler = fit_scaler_on_windows(raw_win, scaler_name)
        meta["n_windows_kept"] = int(scaled.shape[0])
        return scaled, scaler, meta

    if profile == "improved":
        labels = splits.get("train_gen_labels")
        if labels is None:
            raise ValueError(
                "improved profile requires 'train_gen_labels' in splits"
            )
        ratios = compute_window_anomaly_ratio(labels, window_size, stride)
        kept, _keep_mask = filter_anomaly_windows(
            raw_win, ratios, max_anomaly_ratio=max_anomaly_ratio,
            buffer=buffer,
        )
        if kept.shape[0] < min_windows:
            raise InsufficientWindowsError(
                f"only {kept.shape[0]} windows survived contamination "
                f"filter (max_anomaly_ratio={max_anomaly_ratio}, "
                f"buffer={buffer}); need >= {min_windows}"
            )
        z, stats = per_window_znorm(kept)
        scaler = PerWindowZNormScaler(
            train_stats=stats, feat_dim=int(kept.shape[-1]),
        )
        meta["n_windows_kept"] = int(kept.shape[0])
        meta["n_windows_dropped"] = int(n_raw - kept.shape[0])
        return z, scaler, meta

    raise ValueError(f"Unknown preprocessing profile: {profile!r}")


def _min_windows(n_time: int, window: int, stride: int) -> int:
    """Number of full sliding windows (mirror of the tuning helper)."""
    if n_time < window:
        return 0
    return (n_time - window) // stride + 1


def subsample_train_gen(
    splits: dict[str, np.ndarray],
    max_rows: int | None,
) -> dict[str, np.ndarray]:
    """Return a shallow copy dict with "train_gen" truncated to first rows."""
    if max_rows is None:
        return splits
    out = dict(splits)
    tg = splits["train_gen"]
    n = min(max_rows, tg.shape[0])
    out["train_gen"] = tg[:n].copy()
    out["train_gen_labels"] = splits["train_gen_labels"][:n].copy()
    return out
