"""Config loading and small training utilities."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

import yaml


def repo_root() -> Path:
    """Assume project layout: "repo/scripts/foo.py" or "repo/src/..."."""
    return Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(paths: list[Path]) -> dict[str, Any]:
    """Later files override earlier keys."""
    out: dict[str, Any] = {}
    for p in paths:
        d = load_yaml(p)
        out.update(d)
    return out


def get_git_sha() -> str:
    """Return short git SHA or "unknown" if not in a git checkout."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=repo_root(),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _normalize_categorical_choice(value: Any) -> Any:
    """Make categorical choices Optuna-RDB-friendly.

    Optuna warns (and may round-trip poorly in SQLite) when a categorical
    *choice* is a ``list`` — e.g. ``hidden_channels`` candidates like
    ``[64, 128, 256]``. Nested lists should be tuples of scalars.
    """
    if isinstance(value, list):
        return tuple(_normalize_categorical_choice(v) for v in value)
    return value


def suggest_from_dict(trial: Any, name: str, spec: Mapping[str, Any]) -> Any:
    """Map a small YAML search spec to Optuna "suggest_*" calls."""
    st = spec["type"]
    if st == "categorical":
        choices = tuple(
            _normalize_categorical_choice(v) for v in spec["values"]
        )
        return trial.suggest_categorical(name, choices)
    if st == "loguniform":
        return trial.suggest_float(
            name, float(spec["low"]), float(spec["high"]), log=True
        )
    if st == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
    raise ValueError(f"Unsupported search type {st!r} for {name}")


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def env_int(name: str, default: int | None) -> int | None:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return int(v)


def load_model_preprocessing_cfg(model_name: str) -> tuple[str, float, int]:
    """Return ``(profile, max_anomaly_ratio, buffer)`` from the model YAML.

    Defaults to the ``legacy`` profile (used by the original TimeVAE /
    RTSGAN / DDPM / TTSGAN / CSDI configs that do not carry a
    ``preprocessing_profile`` field) so existing runs are unchanged.
    """
    path = repo_root() / "config" / "models" / f"{model_name}.yaml"
    if not path.exists():
        return "legacy", 0.05, 0
    cfg = load_yaml(path)
    profile = str(cfg.get("preprocessing_profile", "legacy"))
    pre = cfg.get("preprocessing", {}) or {}
    max_ar = float(pre.get("max_anomaly_ratio", 0.05))
    buf = int(pre.get("buffer", 0))
    return profile, max_ar, buf


def load_best_params(dataset: str, model: str) -> dict[str, Any]:
    """Load tuning results from ``results/best_params_{dataset}_{model}.json``."""
    path = repo_root() / "results" / f"best_params_{dataset}_{model}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Best params not found: {path}. Run tuning first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)
