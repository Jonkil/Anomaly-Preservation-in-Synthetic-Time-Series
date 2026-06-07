"""Phased Optuna tuning: Phase 1 (preprocessing), Phase 2 (model HPs)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import keras
import mlflow
import numpy as np
import optuna
import yaml
from optuna.pruners import MedianPruner

from src.data.preprocessor import (
    EmptyAfterFilteringError,
    InsufficientWindowsError,
    ScalerName,
    fit_scaler_on_windows,
    load_raw_splits,
    prepare_train_gen_windows,
    sliding_window,
    subsample_train_gen,
)
from src.evaluation.fidelity import fidelity_objective
from src.models._validation import (
    profile_to_scaler_family,
    valid_activations_for,
    validate_scaler_activation,
)
from src.models.timevae_wrapper import build_beta_vae, fit_beta_vae, generate_numpy
from src.training.utils import (
    env_int,
    get_git_sha,
    load_yaml,
    merge_configs,
    repo_root,
    save_json,
    suggest_from_dict,
)
from src.utils.seeds import set_seed


def _min_windows(n_time: int, window: int, stride: int) -> int:
    if n_time < window:
        return 0
    return (n_time - window) // stride + 1


def _optuna_pruner(cfg: dict[str, Any]) -> MedianPruner:
    p = cfg.get("pruner", {})
    return MedianPruner(
        n_startup_trials=int(p.get("n_startup_trials", 2)),
        n_warmup_steps=int(p.get("n_warmup_steps", 2)),
    )


def _prepare_train_gen_windows(
    splits: dict[str, np.ndarray],
    window_size: int,
    stride: int,
    scaler_name: ScalerName,
    min_windows: int = 32,
) -> tuple[np.ndarray, Any] | tuple[None, None]:
    """Window and scale "train_gen"; return "None" if too few windows."""
    tg = splits["train_gen"]
    nw = _min_windows(tg.shape[0], window_size, stride)
    if nw < min_windows:
        return None, None
    raw_win = sliding_window(tg, window_size, stride)
    scaled, scaler = fit_scaler_on_windows(raw_win, scaler_name)
    return scaled, scaler


def _load_preprocessing_cfg(model_name: str) -> tuple[str, float, int]:
    """Return ''(profile, max_anomaly_ratio, buffer)'' from the model YAML."""
    root = repo_root()
    path = root / "config" / "models" / f"{model_name}.yaml"
    if not path.exists():
        return "legacy", 0.05, 0
    cfg = load_yaml(path)
    profile = str(cfg.get("preprocessing_profile", "legacy"))
    pre = cfg.get("preprocessing", {}) or {}
    return (
        profile,
        float(pre.get("max_anomaly_ratio", 0.05)),
        int(pre.get("buffer", 0)),
    )


def _prepare_profile_windows(
    splits: dict[str, np.ndarray],
    window_size: int,
    stride: int,
    scaler_name: ScalerName,
    model_name: str,
    min_windows: int = 32,
) -> tuple[np.ndarray | None, Any]:
    """Profile-aware wrapper used by the new TimeVAE_v2 / TimeVAE_v3 tuners."""
    profile, max_ar, buf = _load_preprocessing_cfg(model_name)
    x_scaled, scaler, _meta = prepare_train_gen_windows(
        splits,
        window_size=window_size,
        stride=stride,
        scaler_name=scaler_name,
        profile=profile,  # type: ignore[arg-type]
        max_anomaly_ratio=max_ar,
        buffer=buf,
        min_windows=min_windows,
    )
    return x_scaled, scaler


def run_phase1_timevae(
    dataset: str,
    n_trials: int | None = None,
    min_windows: int = 32,
) -> dict[str, Any]:
    """Search window size and scaler; train a short TimeVAE; minimize fidelity."""
    root = repo_root()
    base = merge_configs(
        [
            root / "config" / "base.yaml",
            root / "config" / "datasets" / f"{dataset}.yaml",
        ]
    )
    if n_trials is None:
        n_trials = int(base.get("n_trials_phase1", 8))
    n_trials = env_int("TUNE_N_TRIALS_PHASE1", n_trials)
    assert n_trials is not None

    processed = root / base["processed_dir"] / dataset
    splits = load_raw_splits(processed)
    sub_rows = base.get("tuning_subsample_rows")
    if sub_rows is None:
        sub_rows = env_int("TUNE_SUBSAMPLE_ROWS", None)
    splits = subsample_train_gen(splits, sub_rows)

    cand = list(base["window_candidates"])
    fidelity_n = int(
        env_int("TUNE_FIDELITY_N_SAMPLES", int(base.get("fidelity_n_samples", 1000)))
    )
    phase1_epochs = int(env_int("TUNE_PHASE1_EPOCHS", int(base.get("phase1_epochs", 6))))
    phase1_bs = int(base.get("phase1_batch_size", 128))
    seed0 = int(base.get("random_seed_tuning", 0))

    study_name = f"{dataset}_TimeVAE_phase1"
    storage = f"sqlite:///{(root / base['optuna_storage_dir']).resolve()}/optuna_{study_name}.db"
    os.makedirs(root / base["optuna_storage_dir"], exist_ok=True)

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", base["mlflow_tracking_uri_default"])
    )
    mlflow.set_experiment(f"tune_{dataset}_TimeVAE")

    def objective(trial: optuna.Trial) -> float:
        set_seed(seed0 + trial.number)
        window_size = trial.suggest_categorical("window_size", cand)
        stride = max(1, window_size // 2)
        scaler_name: ScalerName = trial.suggest_categorical(
            "scaler_type", ["Standard", "MinMax", "Robust"]
        )

        prepared = _prepare_train_gen_windows(
            splits, window_size, stride, scaler_name, min_windows=min_windows
        )
        x_scaled, _ = prepared
        if x_scaled is None:
            return 1e6

        feat_dim = x_scaled.shape[-1]
        latent_dim = trial.suggest_categorical("phase1_latent_dim", [16, 32])
        beta = trial.suggest_categorical("phase1_beta", [1.0])

        os.environ.setdefault("KERAS_BACKEND", "torch")
        vae = build_beta_vae(window_size, feat_dim, latent_dim, beta=beta)
        fit_beta_vae(vae, x_scaled, phase1_epochs, phase1_bs, seed=seed0 + trial.number)

        syn = generate_numpy(vae, min(fidelity_n, max(32, x_scaled.shape[0] // 2)))
        real_sample = x_scaled[: syn.shape[0]]
        score = fidelity_objective(real_sample, syn)
        if not np.isfinite(score):
            score = 1e6

        with mlflow.start_run(
            run_name=f"phase1_trial_{trial.number}",
            nested=True,
        ):
            mlflow.log_param("git_sha", get_git_sha())
            mlflow.log_param("phase", 1)
            mlflow.log_params(
                {
                    "dataset": dataset,
                    "window_size": window_size,
                    "stride": stride,
                    "scaler_type": scaler_name,
                    "latent_dim": latent_dim,
                    "beta": beta,
                }
            )
            mlflow.log_metric("fidelity_objective", score)

        trial.set_user_attr("stride", stride)
        return score

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        pruner=_optuna_pruner(base),
    )
    study_timeout = env_int("TUNE_STUDY_TIMEOUT", None)
    with mlflow.start_run(run_name=f"{study_name}_study"):
        mlflow.log_param("git_sha", get_git_sha())
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=study_timeout,
            catch=(Exception,),
            show_progress_bar=False,
        )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(
            f"All {len(study.trials)} Phase 1 trials failed for {dataset}. "
            "Check scaler/window combinations and data quality."
        )

    best = study.best_trial
    best_payload = {
        "dataset": dataset,
        "phase": 1,
        "window_size": int(best.params["window_size"]),
        "stride": int(best.user_attrs.get("stride", best.params["window_size"] // 2)),
        "scaler_type": str(best.params["scaler_type"]),
        "value": float(best.value),
    }
    out_path = root / base["results_dir"] / f"best_preproc_{dataset}.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(best_payload, f)
    save_json(
        root / base["results_dir"] / f"best_preproc_{dataset}.json",
        best_payload,
    )
    return best_payload


def run_phase2_timevae(
    dataset: str,
    n_trials: int | None = None,
    min_windows: int = 32,
) -> dict[str, Any]:
    """Fix preprocessing from Phase 1; search model hyperparameters."""
    root = repo_root()
    base = merge_configs(
        [
            root / "config" / "base.yaml",
            root / "config" / "datasets" / f"{dataset}.yaml",
            root / "config" / "models" / "TimeVAE.yaml",
        ]
    )
    if n_trials is None:
        n_trials = int(base.get("n_trials_phase2", 25))
    n_trials = env_int("TUNE_N_TRIALS_PHASE2", n_trials)
    assert n_trials is not None

    pre_path = root / base["results_dir"] / f"best_preproc_{dataset}.yaml"
    if not pre_path.exists():
        raise FileNotFoundError(
            f"Missing {pre_path}; run Phase 1 for {dataset} first."
        )
    pre = load_yaml(pre_path)
    window_size = int(pre["window_size"])
    stride = int(pre["stride"])
    scaler_name: ScalerName = str(pre["scaler_type"])  # type: ignore[assignment]

    processed = root / base["processed_dir"] / dataset
    splits = load_raw_splits(processed)
    sub_rows = base.get("tuning_subsample_rows")
    if sub_rows is None:
        sub_rows = env_int("TUNE_SUBSAMPLE_ROWS", None)
    splits = subsample_train_gen(splits, sub_rows)

    prepared = _prepare_train_gen_windows(
        splits, window_size, stride, scaler_name, min_windows=min_windows
    )
    x_scaled, _ = prepared
    if x_scaled is None:
        raise RuntimeError("Too few windows after Phase 1 preprocessing.")

    phase2_epochs = int(env_int("TUNE_PHASE2_EPOCHS", int(base.get("phase2_epochs", 25))))
    fidelity_n = int(
        env_int("TUNE_FIDELITY_N_SAMPLES", int(base.get("fidelity_n_samples", 1000)))
    )
    seed0 = int(base.get("random_seed_tuning", 0))

    space = base.get("phase2_search_space", {})
    study_name = f"{dataset}_TimeVAE_phase2"
    storage = f"sqlite:///{(root / base['optuna_storage_dir']).resolve()}/optuna_{study_name}.db"
    os.makedirs(root / base["optuna_storage_dir"], exist_ok=True)

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", base["mlflow_tracking_uri_default"])
    )
    mlflow.set_experiment(f"tune_{dataset}_TimeVAE")

    def objective(trial: optuna.Trial) -> float:
        set_seed(seed0 + trial.number)
        latent_dim = suggest_from_dict(trial, "latent_dim", space["latent_dim"])
        beta = suggest_from_dict(trial, "beta", space["beta"])
        lr = suggest_from_dict(trial, "learning_rate", space["learning_rate"])
        batch_size = suggest_from_dict(trial, "batch_size", space["batch_size"])

        feat_dim = x_scaled.shape[-1]
        os.environ.setdefault("KERAS_BACKEND", "torch")
        vae = build_beta_vae(window_size, feat_dim, int(latent_dim), beta=float(beta))
        vae.compile(optimizer=keras.optimizers.Adam(float(lr)))
        fit_beta_vae(
            vae,
            x_scaled,
            phase2_epochs,
            int(batch_size),
            seed=seed0 + trial.number,
        )

        syn = generate_numpy(vae, min(fidelity_n, max(64, x_scaled.shape[0] // 2)))
        real_sample = x_scaled[: syn.shape[0]]
        score = fidelity_objective(real_sample, syn)
        if not np.isfinite(score):
            score = 1e6

        with mlflow.start_run(
            run_name=f"phase2_trial_{trial.number}",
            nested=True,
        ):
            mlflow.log_param("git_sha", get_git_sha())
            mlflow.log_param("phase", 2)
            mlflow.log_params(
                {
                    "dataset": dataset,
                    "window_size": window_size,
                    "stride": stride,
                    "scaler_type": scaler_name,
                    "latent_dim": int(latent_dim),
                    "beta": float(beta),
                    "learning_rate": float(lr),
                    "batch_size": int(batch_size),
                }
            )
            mlflow.log_metric("fidelity_objective", score)

        return score

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        pruner=_optuna_pruner(base),
    )
    study_timeout = env_int("TUNE_STUDY_TIMEOUT", None)
    with mlflow.start_run(run_name=f"{study_name}_study"):
        mlflow.log_param("git_sha", get_git_sha())
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=study_timeout,
            catch=(Exception,),
            show_progress_bar=False,
        )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(
            f"All {len(study.trials)} Phase 2 trials failed for {dataset}. "
            "Check model hyperparameter ranges."
        )

    best = study.best_trial
    merged = {
        "dataset": dataset,
        "model": "TimeVAE",
        "window_size": window_size,
        "stride": stride,
        "scaler_type": scaler_name,
        **best.params,
        "fidelity_objective": float(best.value),
    }
    out_json = root / base["results_dir"] / f"best_params_{dataset}_TimeVAE.json"
    save_json(out_json, merged)
    return merged


def run_phase2_rtsgan(
    dataset: str,
    n_trials: int | None = None,
    min_windows: int = 32,
) -> dict[str, Any]:
    """Fix preprocessing from Phase 1; search RTSGAN hyperparameters."""
    from src.models.rtsgan_wrapper import build_rtsgan, fit_rtsgan, generate_rtsgan

    root = repo_root()
    base = merge_configs(
        [
            root / "config" / "base.yaml",
            root / "config" / "datasets" / f"{dataset}.yaml",
            root / "config" / "models" / "RTSGAN.yaml",
        ]
    )
    if n_trials is None:
        n_trials = int(base.get("n_trials_phase2", 25))
    n_trials = env_int("TUNE_N_TRIALS_PHASE2", n_trials)
    assert n_trials is not None

    pre_path = root / base["results_dir"] / f"best_preproc_{dataset}.yaml"
    if not pre_path.exists():
        raise FileNotFoundError(
            f"Missing {pre_path}; run Phase 1 (TimeVAE) for {dataset} first."
        )
    pre = load_yaml(pre_path)
    window_size = int(pre["window_size"])
    stride = int(pre["stride"])
    scaler_name: ScalerName = str(pre["scaler_type"])  # type: ignore[assignment]

    processed = root / base["processed_dir"] / dataset
    splits = load_raw_splits(processed)
    sub_rows = base.get("tuning_subsample_rows")
    if sub_rows is None:
        sub_rows = env_int("TUNE_SUBSAMPLE_ROWS", None)
    splits = subsample_train_gen(splits, sub_rows)

    prepared = _prepare_train_gen_windows(
        splits, window_size, stride, scaler_name, min_windows=min_windows
    )
    x_scaled, _ = prepared
    if x_scaled is None:
        raise RuntimeError("Too few windows after Phase 1 preprocessing.")

    ae_epochs = int(
        env_int(
            "TUNE_PHASE2_EPOCHS",
            int(base.get("phase2_rtsgan_ae_epochs", 15)),
        )
    )
    gan_iterations = int(
        base.get("phase2_rtsgan_gan_iterations", 2000)
    )
    fidelity_n = int(
        env_int("TUNE_FIDELITY_N_SAMPLES", int(base.get("fidelity_n_samples", 1000)))
    )
    seed0 = int(base.get("random_seed_tuning", 0))
    output_sigmoid = scaler_name == "MinMax"

    space = base.get("phase2_search_space", {})
    study_name = f"{dataset}_RTSGAN_phase2"
    storage = (
        f"sqlite:///{(root / base['optuna_storage_dir']).resolve()}"
        f"/optuna_{study_name}.db"
    )
    os.makedirs(root / base["optuna_storage_dir"], exist_ok=True)

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", base["mlflow_tracking_uri_default"])
    )
    mlflow.set_experiment(f"tune_{dataset}_RTSGAN")

    ae_batch_size = int(base.get("ae_batch_size", 128))
    gan_batch_size = int(base.get("gan_batch_size", 256))
    d_update = int(base.get("d_update", 5))

    def objective(trial: optuna.Trial) -> float:
        set_seed(seed0 + trial.number)
        hidden_dim = int(suggest_from_dict(trial, "hidden_dim", space["hidden_dim"]))
        layers = int(suggest_from_dict(trial, "layers", space["layers"]))
        noise_dim = int(suggest_from_dict(trial, "noise_dim", space["noise_dim"]))
        ae_lr = float(suggest_from_dict(trial, "ae_lr", space["ae_lr"]))
        gan_lr = float(suggest_from_dict(trial, "gan_lr", space["gan_lr"]))

        feat_dim = x_scaled.shape[-1]
        model = build_rtsgan(
            seq_len=window_size,
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            noise_dim=noise_dim,
            layers=layers,
            output_sigmoid=output_sigmoid,
        )
        fit_rtsgan(
            model,
            x_scaled,
            ae_epochs=ae_epochs,
            gan_iterations=gan_iterations,
            ae_batch_size=ae_batch_size,
            gan_batch_size=gan_batch_size,
            ae_lr=ae_lr,
            gan_lr=gan_lr,
            d_update=d_update,
            seed=seed0 + trial.number,
        )
        syn = generate_rtsgan(
            model, min(fidelity_n, max(64, x_scaled.shape[0] // 2))
        )
        real_sample = x_scaled[: syn.shape[0]]
        score = fidelity_objective(real_sample, syn)
        if not np.isfinite(score):
            score = 1e6

        with mlflow.start_run(
            run_name=f"phase2_trial_{trial.number}",
            nested=True,
        ):
            mlflow.log_param("git_sha", get_git_sha())
            mlflow.log_param("phase", 2)
            mlflow.log_params(
                {
                    "dataset": dataset,
                    "model": "RTSGAN",
                    "window_size": window_size,
                    "stride": stride,
                    "scaler_type": scaler_name,
                    "hidden_dim": hidden_dim,
                    "layers": layers,
                    "noise_dim": noise_dim,
                    "ae_lr": ae_lr,
                    "gan_lr": gan_lr,
                }
            )
            mlflow.log_metric("fidelity_objective", score)

        return score

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        pruner=_optuna_pruner(base),
    )
    study_timeout = env_int("TUNE_STUDY_TIMEOUT", None)
    with mlflow.start_run(run_name=f"{study_name}_study"):
        mlflow.log_param("git_sha", get_git_sha())
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=study_timeout,
            catch=(Exception,),
            show_progress_bar=False,
        )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(
            f"All {len(study.trials)} Phase 2 RTSGAN trials failed for {dataset}. "
            "Check model hyperparameter ranges."
        )

    best = study.best_trial
    merged = {
        "dataset": dataset,
        "model": "RTSGAN",
        "window_size": window_size,
        "stride": stride,
        "scaler_type": scaler_name,
        **best.params,
        "fidelity_objective": float(best.value),
    }
    out_json = root / base["results_dir"] / f"best_params_{dataset}_RTSGAN.json"
    save_json(out_json, merged)
    return merged


def run_phase2_ddpm(
    dataset: str,
    n_trials: int | None = None,
    min_windows: int = 32,
) -> dict[str, Any]:
    """Fix preprocessing from Phase 1; search DDPM hyperparameters."""
    from src.models.ddpm_wrapper import build_ddpm, fit_ddpm, generate_ddpm

    root = repo_root()
    base = merge_configs(
        [
            root / "config" / "base.yaml",
            root / "config" / "datasets" / f"{dataset}.yaml",
            root / "config" / "models" / "DDPM.yaml",
        ]
    )
    if n_trials is None:
        n_trials = int(base.get("n_trials_phase2", 25))
    n_trials = env_int("TUNE_N_TRIALS_PHASE2", n_trials)
    assert n_trials is not None

    pre_path = root / base["results_dir"] / f"best_preproc_{dataset}.yaml"
    if not pre_path.exists():
        raise FileNotFoundError(
            f"Missing {pre_path}; run Phase 1 (TimeVAE) for {dataset} first."
        )
    pre = load_yaml(pre_path)
    window_size = int(pre["window_size"])
    stride = int(pre["stride"])
    scaler_name: ScalerName = str(pre["scaler_type"])  # type: ignore[assignment]

    processed = root / base["processed_dir"] / dataset
    splits = load_raw_splits(processed)
    sub_rows = base.get("tuning_subsample_rows")
    if sub_rows is None:
        sub_rows = env_int("TUNE_SUBSAMPLE_ROWS", None)
    splits = subsample_train_gen(splits, sub_rows)

    prepared = _prepare_train_gen_windows(
        splits, window_size, stride, scaler_name, min_windows=min_windows
    )
    x_scaled, _ = prepared
    if x_scaled is None:
        raise RuntimeError("Too few windows after Phase 1 preprocessing.")

    phase2_epochs = int(
        env_int(
            "TUNE_PHASE2_EPOCHS",
            int(base.get("phase2_ddpm_epochs", 15)),
        )
    )
    fidelity_n = int(
        env_int("TUNE_FIDELITY_N_SAMPLES", int(base.get("fidelity_n_samples", 1000)))
    )
    seed0 = int(base.get("random_seed_tuning", 0))

    space = base.get("phase2_search_space", {})
    study_name = f"{dataset}_DDPM_phase2"
    storage = (
        f"sqlite:///{(root / base['optuna_storage_dir']).resolve()}"
        f"/optuna_{study_name}.db"
    )
    os.makedirs(root / base["optuna_storage_dir"], exist_ok=True)

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", base["mlflow_tracking_uri_default"])
    )
    mlflow.set_experiment(f"tune_{dataset}_DDPM")

    def objective(trial: optuna.Trial) -> float:
        set_seed(seed0 + trial.number)
        n_filters = int(suggest_from_dict(trial, "n_filters", space["n_filters"]))
        n_conv_layers = int(suggest_from_dict(trial, "n_conv_layers", space["n_conv_layers"]))
        timesteps = int(suggest_from_dict(trial, "timesteps", space["timesteps"]))
        lr = float(suggest_from_dict(trial, "learning_rate", space["learning_rate"]))
        batch_size = int(suggest_from_dict(trial, "batch_size", space["batch_size"]))

        feat_dim = x_scaled.shape[-1]
        os.environ.setdefault("KERAS_BACKEND", "torch")
        ddpm = build_ddpm(
            seq_len=window_size,
            feat_dim=feat_dim,
            n_filters=n_filters,
            n_conv_layers=n_conv_layers,
            timesteps=timesteps,
        )
        ddpm.compile(optimizer=keras.optimizers.Adam(lr))
        fit_ddpm(
            ddpm,
            x_scaled,
            phase2_epochs,
            batch_size,
            seed=seed0 + trial.number,
        )

        syn = generate_ddpm(ddpm, min(fidelity_n, max(64, x_scaled.shape[0] // 2)))
        real_sample = x_scaled[: syn.shape[0]]
        score = fidelity_objective(real_sample, syn)
        if not np.isfinite(score):
            score = 1e6

        with mlflow.start_run(
            run_name=f"phase2_trial_{trial.number}",
            nested=True,
        ):
            mlflow.log_param("git_sha", get_git_sha())
            mlflow.log_param("phase", 2)
            mlflow.log_params(
                {
                    "dataset": dataset,
                    "model": "DDPM",
                    "window_size": window_size,
                    "stride": stride,
                    "scaler_type": scaler_name,
                    "n_filters": n_filters,
                    "n_conv_layers": n_conv_layers,
                    "timesteps": timesteps,
                    "learning_rate": lr,
                    "batch_size": batch_size,
                }
            )
            mlflow.log_metric("fidelity_objective", score)

        return score

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        pruner=_optuna_pruner(base),
    )
    study_timeout = env_int("TUNE_STUDY_TIMEOUT", None)
    with mlflow.start_run(run_name=f"{study_name}_study"):
        mlflow.log_param("git_sha", get_git_sha())
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=study_timeout,
            catch=(Exception,),
            show_progress_bar=False,
        )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(
            f"All {len(study.trials)} Phase 2 DDPM trials failed for {dataset}. "
            "Check model hyperparameter ranges."
        )

    best = study.best_trial
    merged = {
        "dataset": dataset,
        "model": "DDPM",
        "window_size": window_size,
        "stride": stride,
        "scaler_type": scaler_name,
        **best.params,
        "fidelity_objective": float(best.value),
    }
    out_json = root / base["results_dir"] / f"best_params_{dataset}_DDPM.json"
    save_json(out_json, merged)
    return merged


def run_phase2_ttsgan(
    dataset: str,
    n_trials: int | None = None,
    min_windows: int = 32,
) -> dict[str, Any]:
    """Fix preprocessing from Phase 1; search TTS-GAN hyperparameters."""
    from src.models.ttsgan_wrapper import (
        build_ttsgan,
        fit_ttsgan,
        generate_ttsgan,
    )

    root = repo_root()
    base = merge_configs(
        [
            root / "config" / "base.yaml",
            root / "config" / "datasets" / f"{dataset}.yaml",
            root / "config" / "models" / "TTSGAN.yaml",
        ]
    )
    if n_trials is None:
        n_trials = int(base.get("n_trials_phase2", 25))
    n_trials = env_int("TUNE_N_TRIALS_PHASE2", n_trials)
    assert n_trials is not None

    pre_path = root / base["results_dir"] / f"best_preproc_{dataset}.yaml"
    if not pre_path.exists():
        raise FileNotFoundError(
            f"Missing {pre_path}; run Phase 1 (TimeVAE) for {dataset} first."
        )
    pre = load_yaml(pre_path)
    window_size = int(pre["window_size"])
    stride = int(pre["stride"])
    scaler_name: ScalerName = str(pre["scaler_type"])  # type: ignore[assignment]

    processed = root / base["processed_dir"] / dataset
    splits = load_raw_splits(processed)
    sub_rows = base.get("tuning_subsample_rows")
    if sub_rows is None:
        sub_rows = env_int("TUNE_SUBSAMPLE_ROWS", None)
    splits = subsample_train_gen(splits, sub_rows)

    prepared = _prepare_train_gen_windows(
        splits, window_size, stride, scaler_name, min_windows=min_windows
    )
    x_scaled, _ = prepared
    if x_scaled is None:
        raise RuntimeError("Too few windows after Phase 1 preprocessing.")

    iterations = int(
        env_int(
            "TUNE_PHASE2_TTSGAN_ITERATIONS",
            int(base.get("phase2_ttsgan_iterations", 2000)),
        )
    )
    fidelity_n = int(
        env_int("TUNE_FIDELITY_N_SAMPLES", int(base.get("fidelity_n_samples", 1000)))
    )
    seed0 = int(base.get("random_seed_tuning", 0))
    output_sigmoid = scaler_name == "MinMax"

    space = base.get("phase2_search_space", {})
    batch_size = int(base.get("batch_size", 128))
    d_update = int(base.get("d_update", 3))
    dropout = float(base.get("dropout", 0.1))

    study_name = f"{dataset}_TTSGAN_phase2"
    storage = (
        f"sqlite:///{(root / base['optuna_storage_dir']).resolve()}"
        f"/optuna_{study_name}.db"
    )
    os.makedirs(root / base["optuna_storage_dir"], exist_ok=True)

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", base["mlflow_tracking_uri_default"])
    )
    mlflow.set_experiment(f"tune_{dataset}_TTSGAN")

    def objective(trial: optuna.Trial) -> float:
        set_seed(seed0 + trial.number)
        latent_dim = int(suggest_from_dict(trial, "latent_dim", space["latent_dim"]))
        embed_dim = int(suggest_from_dict(trial, "embed_dim", space["embed_dim"]))
        depth = int(suggest_from_dict(trial, "depth", space["depth"]))
        num_heads = int(suggest_from_dict(trial, "num_heads", space["num_heads"]))
        lr_g = float(suggest_from_dict(trial, "lr_g", space["lr_g"]))
        lr_d = float(suggest_from_dict(trial, "lr_d", space["lr_d"]))

        if embed_dim % num_heads != 0:
            raise optuna.TrialPruned(
                f"embed_dim={embed_dim} not divisible by num_heads={num_heads}"
            )

        feat_dim = x_scaled.shape[-1]
        model = build_ttsgan(
            seq_len=window_size,
            feat_dim=feat_dim,
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
            output_sigmoid=output_sigmoid,
        )
        fit_ttsgan(
            model,
            x_scaled,
            iterations=iterations,
            batch_size=batch_size,
            lr_g=lr_g,
            lr_d=lr_d,
            d_update=d_update,
            seed=seed0 + trial.number,
        )
        syn = generate_ttsgan(
            model, min(fidelity_n, max(64, x_scaled.shape[0] // 2))
        )
        real_sample = x_scaled[: syn.shape[0]]
        score = fidelity_objective(real_sample, syn)
        if not np.isfinite(score):
            score = 1e6

        with mlflow.start_run(
            run_name=f"phase2_trial_{trial.number}",
            nested=True,
        ):
            mlflow.log_param("git_sha", get_git_sha())
            mlflow.log_param("phase", 2)
            mlflow.log_params(
                {
                    "dataset": dataset,
                    "model": "TTSGAN",
                    "window_size": window_size,
                    "stride": stride,
                    "scaler_type": scaler_name,
                    "latent_dim": latent_dim,
                    "embed_dim": embed_dim,
                    "depth": depth,
                    "num_heads": num_heads,
                    "lr_g": lr_g,
                    "lr_d": lr_d,
                }
            )
            mlflow.log_metric("fidelity_objective", score)

        return score

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        pruner=_optuna_pruner(base),
    )
    study_timeout = env_int("TUNE_STUDY_TIMEOUT", None)
    with mlflow.start_run(run_name=f"{study_name}_study"):
        mlflow.log_param("git_sha", get_git_sha())
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=study_timeout,
            catch=(Exception,),
            show_progress_bar=False,
        )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(
            f"All {len(study.trials)} Phase 2 TTSGAN trials failed for {dataset}. "
            "Check model hyperparameter ranges."
        )

    best = study.best_trial
    merged = {
        "dataset": dataset,
        "model": "TTSGAN",
        "window_size": window_size,
        "stride": stride,
        "scaler_type": scaler_name,
        **best.params,
        "fidelity_objective": float(best.value),
    }
    out_json = root / base["results_dir"] / f"best_params_{dataset}_TTSGAN.json"
    save_json(out_json, merged)
    return merged


def run_phase2_csdi(
    dataset: str,
    n_trials: int | None = None,
    min_windows: int = 32,
) -> dict[str, Any]:
    """Fix preprocessing from Phase 1; search CSDI hyperparameters."""
    from src.models.csdi_wrapper import build_csdi, fit_csdi, generate_csdi

    root = repo_root()
    base = merge_configs(
        [
            root / "config" / "base.yaml",
            root / "config" / "datasets" / f"{dataset}.yaml",
            root / "config" / "models" / "CSDI.yaml",
        ]
    )
    if n_trials is None:
        n_trials = int(base.get("n_trials_phase2", 25))
    n_trials = env_int("TUNE_N_TRIALS_PHASE2", n_trials)
    assert n_trials is not None

    pre_path = root / base["results_dir"] / f"best_preproc_{dataset}.yaml"
    if not pre_path.exists():
        raise FileNotFoundError(
            f"Missing {pre_path}; run Phase 1 (TimeVAE) for {dataset} first."
        )
    pre = load_yaml(pre_path)
    window_size = int(pre["window_size"])
    stride = int(pre["stride"])
    scaler_name: ScalerName = str(pre["scaler_type"])  # type: ignore[assignment]

    processed = root / base["processed_dir"] / dataset
    splits = load_raw_splits(processed)
    sub_rows = base.get("tuning_subsample_rows")
    if sub_rows is None:
        sub_rows = env_int("TUNE_SUBSAMPLE_ROWS", None)
    splits = subsample_train_gen(splits, sub_rows)

    prepared = _prepare_train_gen_windows(
        splits, window_size, stride, scaler_name, min_windows=min_windows
    )
    x_scaled, _ = prepared
    if x_scaled is None:
        raise RuntimeError("Too few windows after Phase 1 preprocessing.")

    iterations = int(
        env_int(
            "TUNE_PHASE2_CSDI_ITERATIONS",
            int(base.get("phase2_csdi_iterations", 2000)),
        )
    )
    fidelity_n = int(
        env_int("TUNE_FIDELITY_N_SAMPLES", int(base.get("fidelity_n_samples", 1000)))
    )
    seed0 = int(base.get("random_seed_tuning", 0))

    beta_start = float(base.get("beta_start", 1.0e-4))
    beta_end = float(base.get("beta_end", 0.5))
    diffusion_embedding_dim = int(base.get("diffusion_embedding_dim", 128))
    grad_clip = float(base.get("grad_clip", 1.0))
    is_linear = bool(base.get("is_linear", False))

    space = base.get("phase2_search_space", {})
    study_name = f"{dataset}_CSDI_phase2"
    storage = (
        f"sqlite:///{(root / base['optuna_storage_dir']).resolve()}"
        f"/optuna_{study_name}.db"
    )
    os.makedirs(root / base["optuna_storage_dir"], exist_ok=True)

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", base["mlflow_tracking_uri_default"])
    )
    mlflow.set_experiment(f"tune_{dataset}_CSDI")

    def objective(trial: optuna.Trial) -> float:
        set_seed(seed0 + trial.number)
        channels = int(suggest_from_dict(trial, "channels", space["channels"]))
        layers = int(suggest_from_dict(trial, "layers", space["layers"]))
        nheads = int(suggest_from_dict(trial, "nheads", space["nheads"]))
        num_steps = int(suggest_from_dict(trial, "num_steps", space["num_steps"]))
        timeemb = int(suggest_from_dict(trial, "timeemb", space["timeemb"]))
        featureemb = int(suggest_from_dict(trial, "featureemb", space["featureemb"]))
        schedule = str(suggest_from_dict(trial, "schedule", space["schedule"]))
        lr = float(suggest_from_dict(trial, "learning_rate", space["learning_rate"]))
        batch_size = int(suggest_from_dict(trial, "batch_size", space["batch_size"]))

        if channels % nheads != 0:
            raise optuna.TrialPruned(
                f"channels={channels} not divisible by nheads={nheads}"
            )

        feat_dim = x_scaled.shape[-1]
        model = build_csdi(
            seq_len=window_size,
            feat_dim=feat_dim,
            channels=channels,
            layers=layers,
            nheads=nheads,
            num_steps=num_steps,
            diffusion_embedding_dim=diffusion_embedding_dim,
            timeemb=timeemb,
            featureemb=featureemb,
            schedule=schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            is_linear=is_linear,
        )
        fit_csdi(
            model,
            x_scaled,
            iterations=iterations,
            batch_size=batch_size,
            lr=lr,
            grad_clip=grad_clip,
            seed=seed0 + trial.number,
        )

        syn = generate_csdi(model, min(fidelity_n, max(64, x_scaled.shape[0] // 2)))
        real_sample = x_scaled[: syn.shape[0]]
        score = fidelity_objective(real_sample, syn)
        if not np.isfinite(score):
            score = 1e6

        with mlflow.start_run(
            run_name=f"phase2_trial_{trial.number}",
            nested=True,
        ):
            mlflow.log_param("git_sha", get_git_sha())
            mlflow.log_param("phase", 2)
            mlflow.log_params(
                {
                    "dataset": dataset,
                    "model": "CSDI",
                    "window_size": window_size,
                    "stride": stride,
                    "scaler_type": scaler_name,
                    "channels": channels,
                    "layers": layers,
                    "nheads": nheads,
                    "num_steps": num_steps,
                    "timeemb": timeemb,
                    "featureemb": featureemb,
                    "schedule": schedule,
                    "learning_rate": lr,
                    "batch_size": batch_size,
                }
            )
            mlflow.log_metric("fidelity_objective", score)

        return score

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        pruner=_optuna_pruner(base),
    )
    study_timeout = env_int("TUNE_STUDY_TIMEOUT", None)
    with mlflow.start_run(run_name=f"{study_name}_study"):
        mlflow.log_param("git_sha", get_git_sha())
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=study_timeout,
            catch=(Exception,),
            show_progress_bar=False,
        )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(
            f"All {len(study.trials)} Phase 2 CSDI trials failed for {dataset}. "
            "Check model hyperparameter ranges."
        )

    best = study.best_trial
    merged = {
        "dataset": dataset,
        "model": "CSDI",
        "window_size": window_size,
        "stride": stride,
        "scaler_type": scaler_name,
        **best.params,
        "fidelity_objective": float(best.value),
    }
    out_json = root / base["results_dir"] / f"best_params_{dataset}_CSDI.json"
    save_json(out_json, merged)
    return merged


def _run_phase1_for_model(
    dataset: str,
    model_name: str,
    n_trials: int | None,
    min_windows: int,
) -> dict[str, Any]:
    """Phase 1 search for TimeVAE_v2 / TimeVAE_v3 - uses the improved profile."""
    root = repo_root()
    base = merge_configs(
        [
            root / "config" / "base.yaml",
            root / "config" / "datasets" / f"{dataset}.yaml",
        ]
    )
    if n_trials is None:
        n_trials = int(base.get("n_trials_phase1", 8))
    n_trials = env_int("TUNE_N_TRIALS_PHASE1", n_trials)
    assert n_trials is not None

    processed = root / base["processed_dir"] / dataset
    splits = load_raw_splits(processed)
    sub_rows = base.get("tuning_subsample_rows")
    if sub_rows is None:
        sub_rows = env_int("TUNE_SUBSAMPLE_ROWS", None)
    splits = subsample_train_gen(splits, sub_rows)

    cand = list(base["window_candidates"])
    fidelity_n = int(
        env_int("TUNE_FIDELITY_N_SAMPLES", int(base.get("fidelity_n_samples", 1000)))
    )
    phase1_epochs = int(
        env_int("TUNE_PHASE1_EPOCHS", int(base.get("phase1_epochs", 6)))
    )
    phase1_bs = int(base.get("phase1_batch_size", 128))
    seed0 = int(base.get("random_seed_tuning", 0))

    profile, max_ar, buf = _load_preprocessing_cfg(model_name)

    study_name = f"{dataset}_{model_name}_phase1"
    storage = (
        f"sqlite:///{(root / base['optuna_storage_dir']).resolve()}"
        f"/optuna_{study_name}.db"
    )
    os.makedirs(root / base["optuna_storage_dir"], exist_ok=True)

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", base["mlflow_tracking_uri_default"])
    )
    mlflow.set_experiment(f"tune_{dataset}_{model_name}")

    # Under the improved profile the sklearn scaler name is *unused* by
    # ''prepare_train_gen_windows'' (data is per-window z-normed), so we
    # do not search over it. Storing a fixed sentinel keeps the saved
    # ''best_preproc'' payload self-documenting and future-readable.
    if profile == "improved":
        fixed_scaler_name: ScalerName = "Standard"  # legacy-only no-op slot
        scaler_family = profile_to_scaler_family(profile)
        valid_acts = valid_activations_for(scaler_family)
        # ''valid_acts'' for PerWindowZNorm is exactly ("linear",).
        probe_activation = valid_acts[0]
    else:
        fixed_scaler_name = None  # type: ignore[assignment]
        probe_activation = None  # populated per trial below

    def objective(trial: optuna.Trial) -> float:
        set_seed(seed0 + trial.number)
        window_size = trial.suggest_categorical("window_size", cand)
        stride = max(1, window_size // 2)

        if profile == "improved":
            scaler_name: ScalerName = fixed_scaler_name
            scaler_family = profile_to_scaler_family(profile)
            activation = probe_activation
        else:
            scaler_name = trial.suggest_categorical(
                "scaler_type", ["Standard", "MinMax", "Robust"]
            )
            scaler_family = profile_to_scaler_family(
                profile, sklearn_scaler_name=scaler_name,
            )
            # Pick a probe activation consistent with the scaler family.
            activation = valid_activations_for(scaler_family)[0]

        try:
            x_scaled, _scaler, _meta = prepare_train_gen_windows(
                splits,
                window_size=window_size,
                stride=stride,
                scaler_name=scaler_name,
                profile=profile,  # type: ignore[arg-type]
                max_anomaly_ratio=max_ar,
                buffer=buf,
                min_windows=min_windows,
            )
        except (EmptyAfterFilteringError, InsufficientWindowsError):
            # Document the failure for the trial; Optuna sees a high
            # objective and prunes naturally. We do NOT swallow other
            # exceptions - those propagate and fail the trial loud.
            return 1e6

        feat_dim = x_scaled.shape[-1]
        # Small probe model (TimeVAE_v2 is fastest) to score the preprocessing.
        from src.models.timevae_v2_wrapper import (
            build_timevae_v2,
            fit_timevae_v2,
            generate_numpy as generate_v2,
        )
        latent_dim = trial.suggest_categorical("phase1_latent_dim", [16, 32])

        os.environ.setdefault("KERAS_BACKEND", "torch")
        vae = build_timevae_v2(
            window_size, feat_dim, latent_dim,
            beta=1.0, reconstruction_wt=3.0, kl_anneal_epochs=0,
            output_activation=activation,  # type: ignore[arg-type]
            learning_rate=1e-3,
            scaler_family=scaler_family,  # type: ignore[arg-type]
        )
        fit_timevae_v2(
            vae, x_scaled, epochs=phase1_epochs, batch_size=phase1_bs,
            seed=seed0 + trial.number,
        )
        syn = generate_v2(
            vae,
            min(fidelity_n, max(32, x_scaled.shape[0] // 2)),
            seed=seed0 + trial.number,
        )
        real_sample = x_scaled[: syn.shape[0]]
        score = fidelity_objective(real_sample, syn)
        if not np.isfinite(score):
            score = 1e6

        with mlflow.start_run(
            run_name=f"phase1_trial_{trial.number}", nested=True,
        ):
            mlflow.log_param("git_sha", get_git_sha())
            mlflow.log_param("phase", 1)
            mlflow.log_params(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "window_size": window_size,
                    "stride": stride,
                    "scaler_type": scaler_name,
                    "scaler_family": scaler_family,
                    "probe_activation": activation,
                    "latent_dim": latent_dim,
                    "preprocessing_profile": profile,
                }
            )
            mlflow.log_metric("fidelity_objective", score)

        trial.set_user_attr("stride", stride)
        trial.set_user_attr("scaler_type", scaler_name)
        return score

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        pruner=_optuna_pruner(base),
    )
    study_timeout = env_int("TUNE_STUDY_TIMEOUT", None)
    with mlflow.start_run(run_name=f"{study_name}_study"):
        mlflow.log_param("git_sha", get_git_sha())
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=study_timeout,
            catch=(Exception,),
            show_progress_bar=False,
        )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(
            f"All {len(study.trials)} Phase 1 trials failed for "
            f"{dataset}/{model_name}."
        )

    best = study.best_trial
    best_scaler = best.params.get(
        "scaler_type", best.user_attrs.get("scaler_type", "Standard")
    )
    best_payload = {
        "dataset": dataset,
        "model": model_name,
        "phase": 1,
        "window_size": int(best.params["window_size"]),
        "stride": int(best.user_attrs.get("stride", best.params["window_size"] // 2)),
        "scaler_type": str(best_scaler),
        "preprocessing_profile": profile,
        "scaler_family": profile_to_scaler_family(
            profile,
            sklearn_scaler_name=str(best_scaler) if profile == "legacy" else None,
        ),
        "value": float(best.value),
    }
    out_path = (
        root / base["results_dir"] / f"best_preproc_{dataset}_{model_name}.yaml"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(best_payload, f)
    save_json(
        root / base["results_dir"] / f"best_preproc_{dataset}_{model_name}.json",
        best_payload,
    )
    return best_payload


def run_phase1_timevae_v2(
    dataset: str,
    n_trials: int | None = None,
    min_windows: int = 32,
) -> dict[str, Any]:
    return _run_phase1_for_model(dataset, "TimeVAE_v2", n_trials, min_windows)


def run_phase1_timevae_v3(
    dataset: str,
    n_trials: int | None = None,
    min_windows: int = 32,
) -> dict[str, Any]:
    return _run_phase1_for_model(dataset, "TimeVAE_v3", n_trials, min_windows)


def _load_best_preproc(
    dataset: str, model_name: str,
) -> tuple[int, int, ScalerName]:
    """Find the Phase-1 winner preproc for this dataset/model.

    Prefers the model-specific file, falls back to the shared TimeVAE one.
    """
    root = repo_root()
    cand = [
        root / "results" / f"best_preproc_{dataset}_{model_name}.yaml",
        root / "results" / f"best_preproc_{dataset}.yaml",
    ]
    for p in cand:
        if p.exists():
            pre = load_yaml(p)
            return (
                int(pre["window_size"]),
                int(pre["stride"]),
                str(pre["scaler_type"]),  # type: ignore[return-value]
            )
    raise FileNotFoundError(
        f"No best_preproc YAML found for {dataset}/{model_name}. Tried: "
        + ", ".join(str(p) for p in cand)
    )


def run_phase2_timevae_v2(
    dataset: str,
    n_trials: int | None = None,
    min_windows: int = 32,
) -> dict[str, Any]:
    """Fix preprocessing; search TimeVAE_v2 hyperparameters."""
    from src.models.timevae_v2_wrapper import (
        build_timevae_v2,
        fit_timevae_v2,
        generate_numpy as generate_v2,
    )

    root = repo_root()
    base = merge_configs(
        [
            root / "config" / "base.yaml",
            root / "config" / "datasets" / f"{dataset}.yaml",
            root / "config" / "models" / "TimeVAE_v2.yaml",
        ]
    )
    if n_trials is None:
        n_trials = int(base.get("n_trials_phase2", 25))
    n_trials = env_int("TUNE_N_TRIALS_PHASE2", n_trials)
    assert n_trials is not None

    window_size, stride, scaler_name = _load_best_preproc(dataset, "TimeVAE_v2")

    processed = root / base["processed_dir"] / dataset
    splits = load_raw_splits(processed)
    sub_rows = base.get("tuning_subsample_rows")
    if sub_rows is None:
        sub_rows = env_int("TUNE_SUBSAMPLE_ROWS", None)
    splits = subsample_train_gen(splits, sub_rows)

    x_scaled, _ = _prepare_profile_windows(
        splits, window_size, stride, scaler_name, "TimeVAE_v2", min_windows,
    )
    if x_scaled is None:
        raise RuntimeError("Too few windows after Phase 1 preprocessing.")

    # Effective scaler family seen by the model.
    profile_v2, _, _ = _load_preprocessing_cfg("TimeVAE_v2")
    scaler_family = profile_to_scaler_family(
        profile_v2,
        sklearn_scaler_name=scaler_name if profile_v2 == "legacy" else None,
    )
    valid_acts = valid_activations_for(scaler_family)

    phase2_epochs = int(
        env_int("TUNE_PHASE2_EPOCHS", int(base.get("phase2_epochs", 25)))
    )
    fidelity_n = int(
        env_int("TUNE_FIDELITY_N_SAMPLES", int(base.get("fidelity_n_samples", 1000)))
    )
    seed0 = int(base.get("random_seed_tuning", 0))

    space = base.get("phase2_search_space", {})
    study_name = f"{dataset}_TimeVAE_v2_phase2"
    storage = (
        f"sqlite:///{(root / base['optuna_storage_dir']).resolve()}"
        f"/optuna_{study_name}.db"
    )
    os.makedirs(root / base["optuna_storage_dir"], exist_ok=True)

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", base["mlflow_tracking_uri_default"])
    )
    mlflow.set_experiment(f"tune_{dataset}_TimeVAE_v2")

    def objective(trial: optuna.Trial) -> float:
        set_seed(seed0 + trial.number)
        latent_dim = int(suggest_from_dict(trial, "latent_dim", space["latent_dim"]))
        beta = float(suggest_from_dict(trial, "beta", space["beta"]))
        reconstruction_wt = float(suggest_from_dict(
            trial, "reconstruction_wt", space["reconstruction_wt"]
        ))
        kl_anneal_epochs = int(suggest_from_dict(
            trial, "kl_anneal_epochs", space["kl_anneal_epochs"]
        ))
        output_activation = str(suggest_from_dict(
            trial, "output_activation", space["output_activation"]
        ))
        lr = float(suggest_from_dict(trial, "learning_rate", space["learning_rate"]))
        batch_size = int(suggest_from_dict(trial, "batch_size", space["batch_size"]))

        # Hard fail at the trial boundary if a YAML override accidentally
        # widened the search space to include an activation that does not
        # match the scaler family. Keeping the build-time check is the
        # single source of truth - this trial-level guard just turns the
        # mismatch into a high objective so Optuna can move on.
        if output_activation not in valid_acts:
            return 1e6

        os.environ.setdefault("KERAS_BACKEND", "torch")
        vae = build_timevae_v2(
            window_size, x_scaled.shape[-1], latent_dim,
            beta=beta,
            reconstruction_wt=reconstruction_wt,
            kl_anneal_epochs=kl_anneal_epochs,
            output_activation=output_activation,  # type: ignore[arg-type]
            learning_rate=lr,
            scaler_family=scaler_family,  # type: ignore[arg-type]
        )
        fit_timevae_v2(
            vae, x_scaled, epochs=phase2_epochs, batch_size=batch_size,
            seed=seed0 + trial.number,
        )
        syn = generate_v2(
            vae,
            min(fidelity_n, max(64, x_scaled.shape[0] // 2)),
            seed=seed0 + trial.number,
        )
        real_sample = x_scaled[: syn.shape[0]]
        score = fidelity_objective(real_sample, syn)
        if not np.isfinite(score):
            score = 1e6

        with mlflow.start_run(run_name=f"phase2_trial_{trial.number}", nested=True):
            mlflow.log_param("git_sha", get_git_sha())
            mlflow.log_param("phase", 2)
            mlflow.log_params(
                {
                    "dataset": dataset,
                    "model": "TimeVAE_v2",
                    "window_size": window_size,
                    "stride": stride,
                    "scaler_type": scaler_name,
                    "scaler_family": scaler_family,
                    "latent_dim": latent_dim,
                    "beta": beta,
                    "reconstruction_wt": reconstruction_wt,
                    "kl_anneal_epochs": kl_anneal_epochs,
                    "output_activation": output_activation,
                    "learning_rate": lr,
                    "batch_size": batch_size,
                }
            )
            mlflow.log_metric("fidelity_objective", score)
        return score

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        pruner=_optuna_pruner(base),
    )
    study_timeout = env_int("TUNE_STUDY_TIMEOUT", None)
    with mlflow.start_run(run_name=f"{study_name}_study"):
        mlflow.log_param("git_sha", get_git_sha())
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=study_timeout,
            catch=(Exception,),
            show_progress_bar=False,
        )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(
            f"All {len(study.trials)} Phase 2 TimeVAE_v2 trials failed for {dataset}."
        )

    best = study.best_trial
    merged = {
        "dataset": dataset,
        "model": "TimeVAE_v2",
        "window_size": window_size,
        "stride": stride,
        "scaler_type": scaler_name,
        **best.params,
        "fidelity_objective": float(best.value),
    }
    save_json(
        root / base["results_dir"] / f"best_params_{dataset}_TimeVAE_v2.json",
        merged,
    )
    return merged


def run_phase2_timevae_v3(
    dataset: str,
    n_trials: int | None = None,
    min_windows: int = 32,
) -> dict[str, Any]:
    """Fix preprocessing; search TimeVAE_v3 hyperparameters (native PyTorch)."""
    from src.models.timevae_v3_wrapper import (
        build_timevae_v3,
        fit_timevae_v3,
        generate_numpy as generate_v3,
    )

    root = repo_root()
    base = merge_configs(
        [
            root / "config" / "base.yaml",
            root / "config" / "datasets" / f"{dataset}.yaml",
            root / "config" / "models" / "TimeVAE_v3.yaml",
        ]
    )
    if n_trials is None:
        n_trials = int(base.get("n_trials_phase2", 25))
    n_trials = env_int("TUNE_N_TRIALS_PHASE2", n_trials)
    assert n_trials is not None

    window_size, stride, scaler_name = _load_best_preproc(dataset, "TimeVAE_v3")

    processed = root / base["processed_dir"] / dataset
    splits = load_raw_splits(processed)
    sub_rows = base.get("tuning_subsample_rows")
    if sub_rows is None:
        sub_rows = env_int("TUNE_SUBSAMPLE_ROWS", None)
    splits = subsample_train_gen(splits, sub_rows)

    x_scaled, _ = _prepare_profile_windows(
        splits, window_size, stride, scaler_name, "TimeVAE_v3", min_windows,
    )
    if x_scaled is None:
        raise RuntimeError("Too few windows after Phase 1 preprocessing.")

    profile_v3, _, _ = _load_preprocessing_cfg("TimeVAE_v3")
    scaler_family_v3 = profile_to_scaler_family(
        profile_v3,
        sklearn_scaler_name=scaler_name if profile_v3 == "legacy" else None,
    )
    valid_acts_v3 = valid_activations_for(scaler_family_v3)

    phase2_epochs = int(
        env_int("TUNE_PHASE2_EPOCHS", int(base.get("phase2_epochs", 25)))
    )
    fidelity_n = int(
        env_int("TUNE_FIDELITY_N_SAMPLES", int(base.get("fidelity_n_samples", 1000)))
    )
    seed0 = int(base.get("random_seed_tuning", 0))
    dominant_periods = list(base.get("dominant_periods", []) or [])

    space = base.get("phase2_search_space", {})
    study_name = f"{dataset}_TimeVAE_v3_phase2"
    storage = (
        f"sqlite:///{(root / base['optuna_storage_dir']).resolve()}"
        f"/optuna_{study_name}.db"
    )
    os.makedirs(root / base["optuna_storage_dir"], exist_ok=True)

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", base["mlflow_tracking_uri_default"])
    )
    mlflow.set_experiment(f"tune_{dataset}_TimeVAE_v3")

    def objective(trial: optuna.Trial) -> float:
        set_seed(seed0 + trial.number)
        latent_dim = int(suggest_from_dict(trial, "latent_dim", space["latent_dim"]))
        hidden_channels = list(suggest_from_dict(
            trial, "hidden_channels", space["hidden_channels"]
        ))
        trend_poly = int(suggest_from_dict(trial, "trend_poly", space["trend_poly"]))
        custom_seas_enabled = bool(suggest_from_dict(
            trial, "custom_seas_enabled", space["custom_seas_enabled"]
        ))
        seas_harmonics = int(suggest_from_dict(
            trial, "seas_harmonics", space["seas_harmonics"]
        ))
        use_residual_conn = bool(suggest_from_dict(
            trial, "use_residual_conn", space["use_residual_conn"]
        ))
        reconstruction_wt = float(suggest_from_dict(
            trial, "reconstruction_wt", space["reconstruction_wt"]
        ))
        beta = float(suggest_from_dict(trial, "beta", space["beta"]))
        kl_anneal_epochs = int(suggest_from_dict(
            trial, "kl_anneal_epochs", space["kl_anneal_epochs"]
        ))
        output_activation = str(suggest_from_dict(
            trial, "output_activation", space["output_activation"]
        ))
        lr = float(suggest_from_dict(trial, "learning_rate", space["learning_rate"]))
        batch_size = int(suggest_from_dict(trial, "batch_size", space["batch_size"]))

        if output_activation not in valid_acts_v3:
            return 1e6

        if custom_seas_enabled and dominant_periods:
            custom_seas = [
                (int(p), seas_harmonics)
                for p in dominant_periods
                if int(p) < window_size // 2 and int(p) >= 4
            ]
        else:
            custom_seas = []

        model = build_timevae_v3(
            seq_len=window_size,
            feat_dim=x_scaled.shape[-1],
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            trend_poly=trend_poly,
            custom_seas=custom_seas,
            use_residual_conn=use_residual_conn,
            output_activation=output_activation,  # type: ignore[arg-type]
            reconstruction_wt=reconstruction_wt,
            beta=beta,
            kl_anneal_epochs=kl_anneal_epochs,
            scaler_family=scaler_family_v3,  # type: ignore[arg-type]
        )
        fit_timevae_v3(
            model, x_scaled,
            epochs=phase2_epochs,
            batch_size=batch_size,
            learning_rate=lr,
            seed=seed0 + trial.number,
        )
        syn = generate_v3(
            model,
            min(fidelity_n, max(64, x_scaled.shape[0] // 2)),
            seed=seed0 + trial.number,
        )
        real_sample = x_scaled[: syn.shape[0]]
        score = fidelity_objective(real_sample, syn)
        if not np.isfinite(score):
            score = 1e6

        with mlflow.start_run(run_name=f"phase2_trial_{trial.number}", nested=True):
            mlflow.log_param("git_sha", get_git_sha())
            mlflow.log_param("phase", 2)
            mlflow.log_params(
                {
                    "dataset": dataset,
                    "model": "TimeVAE_v3",
                    "window_size": window_size,
                    "stride": stride,
                    "scaler_type": scaler_name,
                    "scaler_family": scaler_family_v3,
                    "latent_dim": latent_dim,
                    "hidden_channels": str(hidden_channels),
                    "trend_poly": trend_poly,
                    "custom_seas_enabled": custom_seas_enabled,
                    "seas_harmonics": seas_harmonics,
                    "use_residual_conn": use_residual_conn,
                    "reconstruction_wt": reconstruction_wt,
                    "beta": beta,
                    "kl_anneal_epochs": kl_anneal_epochs,
                    "output_activation": output_activation,
                    "learning_rate": lr,
                    "batch_size": batch_size,
                }
            )
            mlflow.log_metric("fidelity_objective", score)
        return score

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        pruner=_optuna_pruner(base),
    )
    study_timeout = env_int("TUNE_STUDY_TIMEOUT", None)
    with mlflow.start_run(run_name=f"{study_name}_study"):
        mlflow.log_param("git_sha", get_git_sha())
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=study_timeout,
            catch=(Exception,),
            show_progress_bar=False,
        )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(
            f"All {len(study.trials)} Phase 2 TimeVAE_v3 trials failed for {dataset}."
        )

    best = study.best_trial
    merged = {
        "dataset": dataset,
        "model": "TimeVAE_v3",
        "window_size": window_size,
        "stride": stride,
        "scaler_type": scaler_name,
        **best.params,
        "fidelity_objective": float(best.value),
    }
    save_json(
        root / base["results_dir"] / f"best_params_{dataset}_TimeVAE_v3.json",
        merged,
    )
    return merged


def run_phased_timevae(
    dataset: str,
    phases: str = "all",
    n_trials_phase1: int | None = None,
    n_trials_phase2: int | None = None,
) -> dict[str, Any]:
    """Run Phase 1 and/or Phase 2 for TimeVAE on "dataset"."""
    out: dict[str, Any] = {}
    if phases in ("1", "all"):
        out["phase1"] = run_phase1_timevae(dataset, n_trials=n_trials_phase1)
    if phases in ("2", "all"):
        out["phase2"] = run_phase2_timevae(dataset, n_trials=n_trials_phase2)
    return out


def run_phased(
    dataset: str,
    model: str = "TimeVAE",
    phases: str = "all",
    n_trials_phase1: int | None = None,
    n_trials_phase2: int | None = None,
) -> dict[str, Any]:
    """Dispatch tuning to the appropriate model.

    Phase 1 (preprocessing search) always uses TimeVAE - it is model-agnostic.
    Phase 2 routes to the model-specific HP search.
    """
    out: dict[str, Any] = {}
    if phases in ("1", "all"):
        if model == "TimeVAE_v2":
            out["phase1"] = run_phase1_timevae_v2(
                dataset, n_trials=n_trials_phase1,
            )
        elif model == "TimeVAE_v3":
            out["phase1"] = run_phase1_timevae_v3(
                dataset, n_trials=n_trials_phase1,
            )
        else:
            out["phase1"] = run_phase1_timevae(
                dataset, n_trials=n_trials_phase1,
            )

    if phases in ("2", "all"):
        if model == "TimeVAE":
            out["phase2"] = run_phase2_timevae(dataset, n_trials=n_trials_phase2)
        elif model == "TimeVAE_v2":
            out["phase2"] = run_phase2_timevae_v2(
                dataset, n_trials=n_trials_phase2,
            )
        elif model == "TimeVAE_v3":
            out["phase2"] = run_phase2_timevae_v3(
                dataset, n_trials=n_trials_phase2,
            )
        elif model == "RTSGAN":
            out["phase2"] = run_phase2_rtsgan(dataset, n_trials=n_trials_phase2)
        elif model == "DDPM":
            out["phase2"] = run_phase2_ddpm(dataset, n_trials=n_trials_phase2)
        elif model == "TTSGAN":
            out["phase2"] = run_phase2_ttsgan(dataset, n_trials=n_trials_phase2)
        elif model == "CSDI":
            out["phase2"] = run_phase2_csdi(dataset, n_trials=n_trials_phase2)
        else:
            raise ValueError(f"Unknown model: {model}")
    return out
