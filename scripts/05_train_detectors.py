#!/usr/bin/env python3
"""Train anomaly detectors (TadGAN, WGAN) on real train_det data."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.anomaly_detectors import DETECTOR_REGISTRY
from src.data.preprocessor import (
    fit_scaler_on_windows,
    load_raw_splits,
    sliding_window,
    window_labels_from_point_labels,
)
from src.training.utils import get_git_sha, repo_root
from src.utils.seeds import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def _discover_preproc_configs(
    results_dir: Path, dataset: str
) -> list[dict[str, Any]]:
    """Find unique preprocessing configs for *dataset* from best_params files.

    Returns a list of dicts, each with ``window_size``, ``stride``,
    ``scaler_type``, and ``source`` (originating filename).
    """
    seen: set[tuple[int, int, str]] = set()
    configs: list[dict[str, Any]] = []
    for p in sorted(results_dir.glob(f"best_params_{dataset}_*.json")):
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        key = (int(data["window_size"]), int(data["stride"]), str(data["scaler_type"]))
        if key not in seen:
            seen.add(key)
            configs.append({
                "window_size": key[0],
                "stride": key[1],
                "scaler_type": key[2],
                "source": p.name,
            })
    return configs


def _prepare_detector_data(
    processed_dir: Path,
    window_size: int,
    stride: int,
    scaler_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load train_det, scale with scaler fit on train_gen, window, label.

    Returns:
        ``(windows, labels)`` - windows shape ``(N, T, F)`` float32,
        labels shape ``(N,)`` int.
    """
    from src.data.preprocessor import ScalerName, get_scaler

    splits = load_raw_splits(processed_dir)
    train_gen = splits["train_gen"]
    train_det = splits["train_det"]
    train_det_labels = splits["train_det_labels"]

    gen_windows = sliding_window(train_gen, window_size, stride)
    _, scaler = fit_scaler_on_windows(gen_windows, scaler_type)  # type: ignore[arg-type]

    if train_det.ndim == 1:
        train_det = train_det.reshape(-1, 1)
    n_f = train_det.shape[1]
    det_flat = train_det.reshape(-1, n_f)
    det_scaled = scaler.transform(det_flat).reshape(train_det.shape)

    det_windows = sliding_window(det_scaled, window_size, stride)
    det_labels = window_labels_from_point_labels(
        train_det_labels, window_size, stride
    )

    det_windows = det_windows.astype(np.float32, copy=False)
    return det_windows, det_labels


def train_detector_for_dataset(
    dataset: str,
    detector_name: str,
    preproc: dict[str, Any],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int,
    root: Path,
) -> dict[str, Any]:
    """Train one detector on one dataset with one preprocessing config."""
    processed_dir = root / "data" / "processed" / dataset
    ws = preproc["window_size"]
    st = preproc["stride"]
    sc = preproc["scaler_type"]

    logger.info(
        "Preparing %s train_det (window=%d, stride=%d, scaler=%s)",
        dataset, ws, st, sc,
    )
    windows, labels = _prepare_detector_data(processed_dir, ws, st, sc)
    n_normal = int((labels == 0).sum())
    n_anom = int((labels == 1).sum())
    logger.info(
        "%s: %d windows (%d normal, %d anomalous, AR=%.4f)",
        dataset, len(labels), n_normal, n_anom,
        labels.mean(dtype=np.float64),
    )

    det_cls = DETECTOR_REGISTRY[detector_name]
    det = det_cls()

    set_seed(seed)
    import torch
    dev = torch.device(device if device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    ))

    diag = det.fit(
        windows,
        labels,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=dev,
        seed=seed,
    )

    save_dir = root / "models" / dataset / detector_name / f"ws{ws}"
    det.save(save_dir)
    logger.info(
        "Saved %s to %s (threshold=%.6f, method=%s)",
        detector_name, save_dir, det.threshold, det.threshold_method,
    )

    try:
        import mlflow

        mlflow_uri = os.environ.get(
            "MLFLOW_TRACKING_URI",
            f"file:{root / 'logs' / 'mlflow'}",
        )
        mlflow.set_tracking_uri(mlflow_uri)
        exp_name = f"detector_{dataset}_{detector_name}_ws{ws}"
        mlflow.set_experiment(exp_name)
        with mlflow.start_run(run_name=f"{detector_name}_{dataset}_ws{ws}"):
            mlflow.log_params({
                "dataset": dataset,
                "detector": detector_name,
                "window_size": ws,
                "stride": st,
                "scaler_type": sc,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "seed": seed,
                "git_sha": get_git_sha(),
            })
            mlflow.log_metrics({
                "threshold": det.threshold,
                "n_train_windows": len(labels),
                "n_normal_windows": n_normal,
                "n_anomalous_windows": n_anom,
                "anomaly_rate_train_det": float(labels.mean(dtype=np.float64)),
            })
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)

    diag["dataset"] = dataset
    diag["window_size"] = ws
    diag["stride"] = st
    diag["scaler_type"] = sc
    diag["save_dir"] = str(save_dir)
    return diag


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train anomaly detectors on train_det data."
    )
    parser.add_argument("--dataset", type=str, default=None,
                        help="Single dataset name (default: all)")
    parser.add_argument("--detector", type=str, default=None,
                        choices=list(DETECTOR_REGISTRY),
                        help="Single detector (default: all)")
    parser.add_argument("--epochs", type=int,
                        default=int(os.environ.get("DETECTOR_EPOCHS", "50")))
    parser.add_argument("--batch-size", type=int,
                        default=int(os.environ.get("DETECTOR_BATCH", "64")))
    parser.add_argument("--lr", type=float,
                        default=float(os.environ.get("DETECTOR_LR", "5e-4")))
    parser.add_argument("--device", type=str,
                        default=os.environ.get("DETECTOR_DEVICE", "auto"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = repo_root()
    results_dir = root / "results"
    config_dir = root / "config" / "datasets"

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = sorted(
            p.stem for p in config_dir.glob("*.yaml")
            if (root / "data" / "processed" / p.stem).is_dir()
        )

    detectors = [args.detector] if args.detector else list(DETECTOR_REGISTRY)

    logger.info("Datasets: %s", datasets)
    logger.info("Detectors: %s", detectors)

    summary: list[dict[str, Any]] = []
    for ds in datasets:
        preprocs = _discover_preproc_configs(results_dir, ds)
        if not preprocs:
            logger.warning("No best_params found for %s - skipping", ds)
            continue
        logger.info(
            "%s: %d unique preprocessing config(s): %s",
            ds, len(preprocs),
            [(p["window_size"], p["stride"], p["scaler_type"]) for p in preprocs],
        )
        for preproc in preprocs:
            for det_name in detectors:
                try:
                    diag = train_detector_for_dataset(
                        ds, det_name, preproc,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        device=args.device,
                        seed=args.seed,
                        root=root,
                    )
                    summary.append(diag)
                except Exception as e:
                    logger.error(
                        "FAILED %s/%s (ws=%d): %s",
                        ds, det_name, preproc["window_size"], e,
                        exc_info=True,
                    )

    
    print("DETECTOR TRAINING SUMMARY")
    for entry in summary:
        print(
            f"  {entry['dataset']:15s}  {entry.get('detector', '?'):8s}  "
            f"ws={entry['window_size']:4d}  "
            f"threshold={entry['threshold']:.6f}  "
            f"({entry.get('threshold_method', '?')})"
        )
    if not summary:
        print("  No detectors trained.")


if __name__ == "__main__":
    main()
