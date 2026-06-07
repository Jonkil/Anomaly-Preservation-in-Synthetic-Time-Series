#!/usr/bin/env python3
"""Compute anomaly preservation metrics (ARD, ARR, TPS) per detector."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
from src.evaluation.anomaly_preservation import compute_all_preservation
from src.evaluation.fidelity import compute_ks_wasserstein
from src.training.utils import get_git_sha, repo_root
from src.utils.seeds import SEEDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def _discover_models_for_dataset(
    synthetic_dir: Path, dataset: str
) -> list[str]:
    """Return model names that have synthetic data for *dataset*."""
    ds_dir = synthetic_dir / dataset
    if not ds_dir.is_dir():
        return []
    models: list[str] = []
    for d in sorted(ds_dir.iterdir()):
        if d.is_dir() and any(d.glob("seed_*.npy")):
            models.append(d.name)
    return models


def _load_best_params(results_dir: Path, dataset: str, model: str) -> dict[str, Any] | None:
    """Load best_params JSON for a (dataset, model) pair."""
    p = results_dir / f"best_params_{dataset}_{model}.json"
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _find_detector_path(
    models_dir: Path, dataset: str, detector: str, window_size: int
) -> Path | None:
    """Locate a trained detector checkpoint directory."""
    candidate = models_dir / dataset / detector / f"ws{window_size}"
    if (candidate / "detector.pt").is_file():
        return candidate
    return None


def _prepare_test_det_windows(
    processed_dir: Path,
    window_size: int,
    stride: int,
    scaler_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Window and scale test_det data using a scaler fit on train_gen.

    Returns:
        ``(windows, labels)`` - windows ``(N, T, F)`` float32, labels ``(N,)`` int.
    """
    from src.data.preprocessor import get_scaler

    splits = load_raw_splits(processed_dir)
    train_gen = splits["train_gen"]
    test_det = splits["test_det"]
    test_det_labels = splits["test_det_labels"]

    gen_windows = sliding_window(train_gen, window_size, stride)
    _, scaler = fit_scaler_on_windows(gen_windows, scaler_type)  # type: ignore[arg-type]

    if test_det.ndim == 1:
        test_det = test_det.reshape(-1, 1)
    n_f = test_det.shape[1]
    det_flat = test_det.reshape(-1, n_f)
    det_scaled = scaler.transform(det_flat).reshape(test_det.shape)

    det_windows = sliding_window(det_scaled, window_size, stride)
    det_labels = window_labels_from_point_labels(
        test_det_labels, window_size, stride
    )
    return det_windows.astype(np.float32, copy=False), det_labels


def _load_synthetic_windows(
    synthetic_dir: Path, dataset: str, model: str, seed: int
) -> np.ndarray | None:
    """Load synthetic windows for a (dataset, model, seed) triple."""
    path = synthetic_dir / dataset / model / f"seed_{seed}.npy"
    if not path.is_file():
        return None
    arr = np.load(path)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return arr.astype(np.float32, copy=False)


def evaluate_single(
    dataset: str,
    model: str,
    seed: int,
    detector_name: str,
    *,
    root: Path,
    device: str,
    skip_fidelity: bool,
) -> dict[str, Any] | None:
    """Evaluate one (dataset, model, seed, detector) cell.

    Returns a flat dict of results, or None if prerequisites are missing.
    """
    results_dir = root / "results"
    synthetic_dir = root / "data" / "synthetic"
    processed_dir = root / "data" / "processed" / dataset
    models_dir = root / "models"

    params = _load_best_params(results_dir, dataset, model)
    if params is None:
        if model == "GaussianNoise":
            for candidate in sorted(results_dir.glob(f"best_params_{dataset}_*.json")):
                with open(candidate, encoding="utf-8") as f:
                    params = json.load(f)
                break
        if params is None:
            logger.warning("No best_params for %s/%s - skipping", dataset, model)
            return None

    ws = int(params["window_size"])
    stride = int(params["stride"])
    scaler_type = str(params["scaler_type"])

    det_path = _find_detector_path(models_dir, dataset, detector_name, ws)
    if det_path is None:
        logger.warning(
            "No trained %s for %s (ws=%d) - skipping", detector_name, dataset, ws
        )
        return None

    syn = _load_synthetic_windows(synthetic_dir, dataset, model, seed)
    if syn is None:
        logger.warning("No synthetic data for %s/%s/seed_%d", dataset, model, seed)
        return None

    det_cls = DETECTOR_REGISTRY[detector_name]
    import torch
    dev = torch.device(device if device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    ))
    detector = det_cls.load(det_path, device=dev)

    test_windows, test_labels = _prepare_test_det_windows(
        processed_dir, ws, stride, scaler_type
    )

    y_pred_real = detector.predict(test_windows, device=dev)
    y_pred_syn = detector.predict(syn, device=dev)

    aps = compute_all_preservation(y_pred_real, y_pred_syn)

    real_ar = float(y_pred_real.mean(dtype=np.float64))
    syn_ar = float(y_pred_syn.mean(dtype=np.float64))
    gt_ar = float(test_labels.mean(dtype=np.float64))

    result: dict[str, Any] = {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "detector": detector_name,
        "window_size": ws,
        "stride": stride,
        "scaler_type": scaler_type,
        "ARD": aps["ard"],
        "ARR": aps["arr"],
        "TPS": aps["tps"],
        "AR_real_pred": real_ar,
        "AR_syn_pred": syn_ar,
        "AR_real_gt": gt_ar,
        "n_test_windows": len(test_labels),
        "n_syn_windows": len(syn),
        "threshold": detector.threshold,
        "threshold_method": detector.threshold_method,
    }

    if not skip_fidelity:
        try:
            from src.evaluation.window_loading import prepare_real_windows, best_params_or_gaussian
            real_gen = prepare_real_windows(root, dataset, params, model)
            ks, wass = compute_ks_wasserstein(real_gen, syn)
            result["KS"] = ks
            result["Wasserstein"] = wass
        except Exception as e:
            logger.warning("Fidelity computation failed for %s/%s: %s", dataset, model, e)
            result["KS"] = float("nan")
            result["Wasserstein"] = float("nan")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute anomaly preservation metrics across all models."
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--detector", type=str, default=None,
                        choices=list(DETECTOR_REGISTRY))
    parser.add_argument("--skip-fidelity", action="store_true")
    parser.add_argument("--device", type=str,
                        default=os.environ.get("EVAL_DEVICE", "auto"))
    args = parser.parse_args()

    root = repo_root()
    synthetic_dir = root / "data" / "synthetic"
    config_dir = root / "config" / "datasets"

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = sorted(
            p.stem for p in config_dir.glob("*.yaml")
            if (synthetic_dir / p.stem).is_dir()
        )

    detectors = [args.detector] if args.detector else list(DETECTOR_REGISTRY)

    logger.info("Datasets: %s", datasets)
    logger.info("Detectors: %s", detectors)

    all_results: list[dict[str, Any]] = []

    for ds in datasets:
        if args.model:
            models = [args.model]
        else:
            models = _discover_models_for_dataset(synthetic_dir, ds)
        if not models:
            logger.warning("No synthetic data for %s - skipping", ds)
            continue

        logger.info("%s: evaluating models %s", ds, models)

        for model in models:
            for seed in SEEDS:
                for det_name in detectors:
                    try:
                        result = evaluate_single(
                            ds, model, seed, det_name,
                            root=root,
                            device=args.device,
                            skip_fidelity=args.skip_fidelity,
                        )
                        if result is not None:
                            all_results.append(result)
                            logger.info(
                                "  %s/%s/seed_%d/%s: ARD=%.4f ARR=%.4f TPS=%.4f",
                                ds, model, seed, det_name,
                                result["ARD"], result["ARR"], result["TPS"],
                            )
                    except Exception as e:
                        logger.error(
                            "FAILED %s/%s/seed_%d/%s: %s",
                            ds, model, seed, det_name, e,
                            exc_info=True,
                        )

    if not all_results:
        logger.warning("No results computed. Are detectors trained?")
        return

    df = pd.DataFrame(all_results)

    tables_dir = root / "results" / "tables" / f"{args.dataset}_{args.model}_{args.detector}"
    tables_dir.mkdir(parents=True, exist_ok=True)

    per_seed_path = tables_dir / "per_seed_aps_results.csv"
    df.to_csv(per_seed_path, index=False)
    logger.info("Per-seed results: %s (%d rows)", per_seed_path, len(df))

    numeric_cols = ["ARD", "ARR", "TPS"]
    if "KS" in df.columns:
        numeric_cols += ["KS", "Wasserstein"]
    numeric_cols += ["AR_real_pred", "AR_syn_pred"]

    group_cols = ["dataset", "model", "detector"]
    agg_dict: dict[str, list[str]] = {c: ["mean", "std"] for c in numeric_cols}
    agg_dict["seed"] = ["count"]

    agg = df.groupby(group_cols, sort=True).agg(agg_dict)
    agg.columns = ["_".join(col).strip("_") for col in agg.columns]
    agg = agg.rename(columns={"seed_count": "n_seeds"}).reset_index()

    agg_path = tables_dir / "aggregate_aps_results.csv"
    agg.to_csv(agg_path, index=False)
    logger.info("Aggregate results: %s (%d rows)", agg_path, len(agg))

    try:
        import mlflow

        mlflow_uri = os.environ.get(
            "MLFLOW_TRACKING_URI",
            f"file:{root / 'logs' / 'mlflow'}",
        )
        mlflow.set_tracking_uri(mlflow_uri)
        for _, row in df.iterrows():
            exp_name = f"aps_{row['dataset']}_{row['model']}"
            mlflow.set_experiment(exp_name)
            with mlflow.start_run(
                run_name=f"seed{row['seed']}_{row['detector']}"
            ):
                mlflow.log_params({
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "seed": row["seed"],
                    "detector": row["detector"],
                    "git_sha": get_git_sha(),
                })
                metrics_to_log = {
                    "ARD": row["ARD"],
                    "TPS": row["TPS"],
                    "AR_real_pred": row["AR_real_pred"],
                    "AR_syn_pred": row["AR_syn_pred"],
                }
                if np.isfinite(row["ARR"]):
                    metrics_to_log["ARR"] = row["ARR"]
                if "KS" in row and np.isfinite(row.get("KS", float("nan"))):
                    metrics_to_log["KS"] = row["KS"]
                    metrics_to_log["Wasserstein"] = row["Wasserstein"]
                mlflow.log_metrics(metrics_to_log)
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)

    print("ANOMALY PRESERVATION RESULTS SUMMARY")
    for _, row in agg.iterrows():
        arr_str = (
            f"{row['ARR_mean']:.4f}±{row['ARR_std']:.4f}"
            if np.isfinite(row["ARR_mean"])
            else "inf"
        )
        print(
            f"  {row['dataset']:15s}  {row['model']:15s}  {row['detector']:8s}  "
            f"ARD={row['ARD_mean']:.4f}±{row['ARD_std']:.4f}  "
            f"ARR={arr_str}  "
            f"TPS={row['TPS_mean']:.4f}±{row['TPS_std']:.4f}  "
            f"(n={int(row['n_seeds'])})"
        )


if __name__ == "__main__":
    main()
