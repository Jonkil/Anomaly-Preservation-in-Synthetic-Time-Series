"""Abstract base class for window-level anomaly detectors."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score


class AnomalyDetector(ABC):
    """Common fit / score / predict surface for anomaly detectors."""

    def __init__(self) -> None:
        self.threshold: float = float("nan")
        self.threshold_method: str = "unset"

    # ------------------------------------------------------------------
    # Abstract interface - subclasses MUST implement
    # ------------------------------------------------------------------
    @abstractmethod
    def _fit_model(
        self,
        normal_windows: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        lr: float,
        device: torch.device,
        seed: int,
    ) -> dict[str, Any]:
        """Train the detector."""

    @abstractmethod
    def _score_windows(
        self, windows: np.ndarray, *, device: torch.device, batch_size: int
    ) -> np.ndarray:
        """Return per-window anomaly scores."""

    @abstractmethod
    def _state_dict(self) -> dict[str, Any]:
        """Return a serialisable snapshot of all learnable parameters."""

    @abstractmethod
    def _load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore learnable parameters from *state*."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        train_windows: np.ndarray,
        train_labels: np.ndarray,
        *,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 5e-4,
        device: torch.device | str | None = None,
        seed: int = 0,
        threshold_method: str = "best_f1",
        threshold_percentile: float = 95.0,
    ) -> dict[str, Any]:
        """Train the detector and calibrate the threshold on ``train_det``."""
        self._validate_windows(train_windows, "train_windows")
        train_labels = np.asarray(train_labels, dtype=np.float64).ravel()
        if train_labels.shape[0] != train_windows.shape[0]:
            raise ValueError(
                f"label length {train_labels.shape[0]} != "
                f"window count {train_windows.shape[0]}"
            )

        dev = self._resolve_device(device)
        normal_mask = train_labels == 0
        normal_windows = train_windows[normal_mask].astype(np.float32, copy=False)
        if normal_windows.shape[0] == 0:
            raise ValueError("No normal windows (label==0) in train_det")

        diag = self._fit_model(
            normal_windows,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=dev,
            seed=seed,
        )

        scores = self._score_windows(
            train_windows.astype(np.float32, copy=False),
            device=dev,
            batch_size=batch_size,
        )
        thr = self.calibrate_threshold(
            scores,
            train_labels,
            method=threshold_method,
            percentile=threshold_percentile,
        )
        diag["threshold"] = thr
        diag["threshold_method"] = self.threshold_method
        return diag

    def score(
        self,
        windows: np.ndarray,
        *,
        device: torch.device | str | None = None,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Return per-window anomaly scores."""
        self._validate_windows(windows, "windows")
        dev = self._resolve_device(device)
        return self._score_windows(
            windows.astype(np.float32, copy=False), device=dev, batch_size=batch_size
        )

    def predict(
        self,
        windows: np.ndarray,
        *,
        device: torch.device | str | None = None,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Binary predictions using the stored threshold."""
        if not np.isfinite(self.threshold):
            raise RuntimeError(
                "Threshold not calibrated - call fit() or calibrate_threshold() first"
            )
        scores = self.score(windows, device=device, batch_size=batch_size)
        return (scores >= self.threshold).astype(np.int32)

    def calibrate_threshold(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        *,
        method: str = "best_f1",
        percentile: float = 95.0,
    ) -> float:
        """Find the detection threshold from scored ``train_det`` windows."""
        scores = np.asarray(scores, dtype=np.float64).ravel()
        labels = np.asarray(labels, dtype=np.float64).ravel()
        if scores.shape[0] != labels.shape[0]:
            raise ValueError("scores and labels must have the same length")

        has_anomalies = labels.sum() > 0
        if method == "best_f1" and has_anomalies:
            self.threshold = self._best_f1_threshold(scores, labels)
            self.threshold_method = "best_f1"
        else:
            normal_scores = scores[labels == 0]
            if normal_scores.size == 0:
                normal_scores = scores
            self.threshold = float(np.percentile(normal_scores, percentile))
            self.threshold_method = f"percentile_{percentile}"

        return self.threshold

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Persist the detector checkpoint and threshold atomically."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        state = self._state_dict()
        state["__threshold__"] = self.threshold
        state["__threshold_method__"] = self.threshold_method

        pt_path = path / "detector.pt"
        tmp = pt_path.with_suffix(".pt.tmp")
        torch.save(state, tmp)
        os.replace(tmp, pt_path)

        thr_path = path / "threshold.json"
        tmp_thr = thr_path.with_suffix(".json.tmp")
        with open(tmp_thr, "w", encoding="utf-8") as f:
            json.dump(
                {"threshold": self.threshold, "method": self.threshold_method},
                f,
                indent=2,
            )
        os.replace(tmp_thr, thr_path)

    @classmethod
    def load(cls, path: Path, *, device: torch.device | str | None = None) -> "AnomalyDetector":
        """Restore a detector from a saved checkpoint."""
        path = Path(path)
        pt_path = path / "detector.pt"
        if not pt_path.is_file():
            raise FileNotFoundError(f"No checkpoint at {pt_path}")

        dev = cls._resolve_device_static(device)
        state = torch.load(pt_path, map_location=dev, weights_only=False)

        instance = cls.__new__(cls)
        instance.__init__()  # type: ignore[misc]
        instance.threshold = state.pop("__threshold__", float("nan"))
        instance.threshold_method = state.pop("__threshold_method__", "unknown")
        instance._load_state_dict(state)
        return instance

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_windows(x: np.ndarray, name: str) -> None:
        if x.ndim != 3:
            raise ValueError(f"{name} must be 3-D (N, T, F), got shape {x.shape}")
        if x.shape[0] == 0:
            raise ValueError(f"{name} has zero windows")
        if not np.all(np.isfinite(x)):
            raise ValueError(f"{name} contains non-finite values")

    @staticmethod
    def _resolve_device(device: torch.device | str | None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _resolve_device_static(device: torch.device | str | None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
        """Sweep candidate thresholds and return the one maximising F1."""
        candidates = np.unique(scores)
        if len(candidates) <= 1:
            return float(candidates[0]) if len(candidates) == 1 else 0.0

        best_f1 = -1.0
        best_thr = float(candidates[0])
        for thr in candidates:
            preds = (scores >= thr).astype(np.int32)
            f1 = float(f1_score(labels, preds, zero_division=0.0))
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)
        return best_thr
