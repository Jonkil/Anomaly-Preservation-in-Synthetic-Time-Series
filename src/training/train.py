"""Final multi-seed training: train generative model with best HPs, generate synthetic data."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np

from src.data.preprocessor import (
    PerWindowZNormScaler,
    ScalerName,
    fit_scaler_on_windows,
    load_raw_splits,
    prepare_train_gen_windows,
    sliding_window,
)
from src.evaluation.fidelity import compute_ks_wasserstein
from src.models._validation import (
    profile_to_scaler_family,
    validate_scaler_activation,
)
from src.training.utils import (
    env_int,
    get_git_sha,
    load_best_params,
    load_model_preprocessing_cfg,
    load_yaml,
    merge_configs,
    repo_root,
)
from src.utils.seeds import SEEDS, set_seed

# Backward-compatible aliases for scripts that import these from train.
_load_model_preprocessing_cfg = load_model_preprocessing_cfg


def _load_model_dataset_cfg(dataset: str) -> dict[str, Any]:
    """Load the per-dataset YAML (e.g. for ''dominant_periods'')."""
    root = repo_root()
    path = root / "config" / "datasets" / f"{dataset}.yaml"
    if not path.exists():
        return {}
    return load_yaml(path)


def _train_timevae(
    x_scaled: np.ndarray,
    window_size: int,
    params: dict[str, Any],
    seed: int,
    epochs: int,
    verbose: int,
) -> tuple[Any, np.ndarray, str]:
    """Train TimeVAE and return (model, synthetic, checkpoint_ext)."""
    import keras
    from src.models.timevae_wrapper import build_beta_vae, fit_beta_vae, generate_numpy

    latent_dim = int(params["latent_dim"])
    beta = float(params["beta"])
    lr = float(params["learning_rate"])
    batch_size = int(params["batch_size"])

    os.environ.setdefault("KERAS_BACKEND", "torch")
    vae = build_beta_vae(window_size, x_scaled.shape[-1], latent_dim, beta=beta)
    vae.compile(optimizer=keras.optimizers.Adam(lr))
    fit_beta_vae(vae, x_scaled, epochs, batch_size, seed=seed, verbose=verbose)

    n_generate = x_scaled.shape[0]
    syn = generate_numpy(vae, n_generate)
    return vae, syn, ".weights.h5"


def _train_timevae_v2(
    x_scaled: np.ndarray,
    window_size: int,
    params: dict[str, Any],
    seed: int,
    epochs: int,
    n_generate: int,
    scaler_family: str,
    verbose: int,
) -> tuple[Any, np.ndarray, str]:
    """Train TimeVAE_v2 (fixed BetaVAE) and return ''(model, synthetic, ext)''.

    Generation uses the trained checkpoint with a Keras-RNG seed equal to
    ''seed'' so that reloading the checkpoint and calling
    :func:`generate_numpy(model, n, seed=seed)` reproduces the same
    synthetic windows.
    """
    from src.models.timevae_v2_wrapper import (
        build_timevae_v2,
        fit_timevae_v2,
        generate_numpy as generate_v2,
    )

    latent_dim = int(params["latent_dim"])
    beta = float(params["beta"])
    reconstruction_wt = float(params.get("reconstruction_wt", 3.0))
    kl_anneal_epochs = int(params.get("kl_anneal_epochs", 0))
    output_activation = str(params.get("output_activation", "linear"))
    lr = float(params["learning_rate"])
    batch_size = int(params["batch_size"])

    os.environ.setdefault("KERAS_BACKEND", "torch")
    vae = build_timevae_v2(
        window_size,
        x_scaled.shape[-1],
        latent_dim,
        beta=beta,
        reconstruction_wt=reconstruction_wt,
        kl_anneal_epochs=kl_anneal_epochs,
        output_activation=output_activation,  # type: ignore[arg-type]
        learning_rate=lr,
        scaler_family=scaler_family,  # type: ignore[arg-type]
    )
    fit_timevae_v2(
        vae, x_scaled, epochs=epochs, batch_size=batch_size, seed=seed,
        verbose=verbose,
    )
    syn = generate_v2(vae, n_generate, seed=seed)
    return vae, syn, ".weights.h5"


def _train_timevae_v3(
    x_scaled: np.ndarray,
    window_size: int,
    params: dict[str, Any],
    seed: int,
    epochs: int,
    n_generate: int,
    scaler_family: str,
    dataset: str,
    verbose: int,
) -> tuple[Any, np.ndarray, str]:
    """Train the native-PyTorch TimeVAE_v3 and return ''(model, synthetic, ext)''.

    Build-time validation enforces the scaler ↔ activation contract.
    Generation uses a local Torch generator seeded with ''seed'' so that
    reloading the checkpoint and calling
    :func:`generate_numpy(model, n, seed=seed)` reproduces the same
    synthetic windows.
    """
    from src.models.timevae_v3_wrapper import (
        build_timevae_v3,
        fit_timevae_v3,
        generate_numpy as generate_v3,
    )

    dataset_cfg = _load_model_dataset_cfg(dataset)
    dominant_periods = list(dataset_cfg.get("dominant_periods", []) or [])

    latent_dim = int(params["latent_dim"])
    hidden_channels = list(params.get("hidden_channels", [64, 128, 256]))
    trend_poly = int(params.get("trend_poly", 1))
    custom_seas_enabled = bool(params.get("custom_seas_enabled", True))
    seas_harmonics = int(params.get("seas_harmonics", 3))
    use_residual_conn = bool(params.get("use_residual_conn", True))
    output_activation = str(params.get("output_activation", "linear"))
    reconstruction_wt = float(params.get("reconstruction_wt", 3.0))
    beta = float(params.get("beta", 1.0))
    kl_anneal_epochs = int(params.get("kl_anneal_epochs", 0))
    lr = float(params["learning_rate"])
    batch_size = int(params["batch_size"])

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
        scaler_family=scaler_family,  # type: ignore[arg-type]
    )
    fit_timevae_v3(
        model, x_scaled,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        seed=seed,
        verbose=verbose,
    )
    syn = generate_v3(model, n_generate, seed=seed)
    return model, syn, ".pt"


def _train_rtsgan(
    x_scaled: np.ndarray,
    window_size: int,
    scaler_type: str,
    params: dict[str, Any],
    seed: int,
    epochs: int,
) -> tuple[Any, np.ndarray, str]:
    """Train RTSGAN and return (model, synthetic, checkpoint_ext)."""
    from src.models.rtsgan_wrapper import build_rtsgan, fit_rtsgan, generate_rtsgan

    root = repo_root()
    model_cfg = merge_configs(
        [root / "config" / "base.yaml", root / "config" / "models" / "RTSGAN.yaml"]
    )

    hidden_dim = int(params["hidden_dim"])
    layers = int(params["layers"])
    noise_dim = int(params["noise_dim"])
    ae_lr = float(params["ae_lr"])
    gan_lr = float(params["gan_lr"])
    ae_epochs = epochs
    gan_iterations = int(
        env_int("FINAL_TRAIN_GAN_ITERATIONS", int(model_cfg.get("gan_iterations", 5000)))
    )
    ae_batch_size = int(model_cfg.get("ae_batch_size", 128))
    gan_batch_size = int(model_cfg.get("gan_batch_size", 256))
    d_update = int(model_cfg.get("d_update", 5))
    output_sigmoid = scaler_type == "MinMax"

    model = build_rtsgan(
        seq_len=window_size,
        feat_dim=x_scaled.shape[-1],
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
        seed=seed,
    )
    syn = generate_rtsgan(model, x_scaled.shape[0])
    return model, syn, ".pt"


def _train_ttsgan(
    x_scaled: np.ndarray,
    window_size: int,
    scaler_type: str,
    params: dict[str, Any],
    seed: int,
) -> tuple[Any, np.ndarray, str]:
    """Train TTS-GAN and return (model, synthetic, checkpoint_ext)."""
    from src.models.ttsgan_wrapper import build_ttsgan, fit_ttsgan, generate_ttsgan

    root = repo_root()
    model_cfg = merge_configs(
        [root / "config" / "base.yaml", root / "config" / "models" / "TTSGAN.yaml"]
    )

    latent_dim = int(params["latent_dim"])
    embed_dim = int(params["embed_dim"])
    depth = int(params["depth"])
    num_heads = int(params["num_heads"])
    lr_g = float(params["lr_g"])
    lr_d = float(params["lr_d"])

    iterations = int(
        env_int("FINAL_TRAIN_TTSGAN_ITERATIONS", int(model_cfg.get("iterations", 5000)))
    )
    batch_size = int(model_cfg.get("batch_size", 128))
    d_update = int(model_cfg.get("d_update", 3))
    dropout = float(model_cfg.get("dropout", 0.1))
    output_sigmoid = scaler_type == "MinMax"

    model = build_ttsgan(
        seq_len=window_size,
        feat_dim=x_scaled.shape[-1],
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
        seed=seed,
    )
    syn = generate_ttsgan(model, x_scaled.shape[0])
    return model, syn, ".pt"


def _train_csdi(
    x_scaled: np.ndarray,
    window_size: int,
    params: dict[str, Any],
    seed: int,
) -> tuple[Any, np.ndarray, str]:
    """Train CSDI and return (model, synthetic, checkpoint_ext)."""
    from src.models.csdi_wrapper import build_csdi, fit_csdi, generate_csdi

    root = repo_root()
    model_cfg = merge_configs(
        [root / "config" / "base.yaml", root / "config" / "models" / "CSDI.yaml"]
    )

    channels = int(params["channels"])
    layers = int(params["layers"])
    nheads = int(params["nheads"])
    num_steps = int(params["num_steps"])
    timeemb = int(params["timeemb"])
    featureemb = int(params["featureemb"])
    schedule = str(params["schedule"])
    lr = float(params["learning_rate"])
    batch_size = int(params["batch_size"])

    iterations = int(
        env_int(
            "FINAL_TRAIN_CSDI_ITERATIONS",
            int(model_cfg.get("iterations", 5000)),
        )
    )
    beta_start = float(model_cfg.get("beta_start", 1.0e-4))
    beta_end = float(model_cfg.get("beta_end", 0.5))
    diffusion_embedding_dim = int(model_cfg.get("diffusion_embedding_dim", 128))
    grad_clip = float(model_cfg.get("grad_clip", 1.0))
    is_linear = bool(model_cfg.get("is_linear", False))

    model = build_csdi(
        seq_len=window_size,
        feat_dim=x_scaled.shape[-1],
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
        seed=seed,
    )
    syn = generate_csdi(model, x_scaled.shape[0])
    return model, syn, ".pt"


def _train_ddpm(
    x_scaled: np.ndarray,
    window_size: int,
    params: dict[str, Any],
    seed: int,
    epochs: int,
    verbose: int,
) -> tuple[Any, np.ndarray, str]:
    """Train DDPM and return (model, synthetic, checkpoint_ext)."""
    import keras
    from src.models.ddpm_wrapper import build_ddpm, fit_ddpm, generate_ddpm

    n_filters = int(params["n_filters"])
    n_conv_layers = int(params["n_conv_layers"])
    timesteps = int(params["timesteps"])
    lr = float(params["learning_rate"])
    batch_size = int(params["batch_size"])

    os.environ.setdefault("KERAS_BACKEND", "torch")
    ddpm = build_ddpm(
        window_size,
        x_scaled.shape[-1],
        n_filters=n_filters,
        n_conv_layers=n_conv_layers,
        timesteps=timesteps,
    )
    ddpm.compile(optimizer=keras.optimizers.Adam(lr))
    fit_ddpm(ddpm, x_scaled, epochs, batch_size, seed=seed, verbose=verbose)

    n_generate = x_scaled.shape[0]
    syn = generate_ddpm(ddpm, n_generate)
    return ddpm, syn, ".weights.h5"


def train_single_seed(
    dataset: str,
    model_name: str,
    seed: int,
    epochs: int | None = None,
    n_generate: int | None = None,
    verbose: int = 0,
) -> dict[str, Any]:
    """Train one model with best HPs, generate synthetic data, save outputs.

    Args:
        dataset: Dataset key (e.g. "SWaT").
        model_name: Model key ("TimeVAE", "RTSGAN", "DDPM", "TTSGAN", or "CSDI").
        seed: Random seed for this run.
        epochs: Training epochs (env "FINAL_TRAIN_EPOCHS", default 200).
        n_generate: Number of synthetic windows to produce. Defaults to
            the number of "train_gen" windows.
        verbose: Keras verbosity (0=silent, 1=progress bar).

    Returns:
        Dict with paths to saved artifacts and fidelity scores.
    """
    root = repo_root()
    params = load_best_params(dataset, model_name)

    window_size = int(params["window_size"])
    stride = int(params["stride"])
    scaler_type: ScalerName = str(params["scaler_type"])  # type: ignore[assignment]

    if epochs is None:
        epochs = int(env_int("FINAL_TRAIN_EPOCHS", 200))

    processed = root / "data" / "processed" / dataset
    splits = load_raw_splits(processed)

    profile, max_anomaly_ratio, buffer = load_model_preprocessing_cfg(model_name)
    x_scaled, scaler, preproc_meta = prepare_train_gen_windows(
        splits,
        window_size=window_size,
        stride=stride,
        scaler_name=scaler_type,
        profile=profile,  # type: ignore[arg-type]
        max_anomaly_ratio=max_anomaly_ratio,
        buffer=buffer,
        min_windows=1,
    )

    if n_generate is None:
        n_generate = x_scaled.shape[0]

    scaler_family = profile_to_scaler_family(
        profile, sklearn_scaler_name=scaler_type if profile == "legacy" else None,
    )

    set_seed(seed)

    if model_name == "TimeVAE":
        model_obj, syn, ckpt_ext = _train_timevae(
            x_scaled, window_size, params, seed, epochs, verbose,
        )
    elif model_name == "TimeVAE_v2":
        model_obj, syn, ckpt_ext = _train_timevae_v2(
            x_scaled, window_size, params, seed, epochs,
            n_generate=n_generate, scaler_family=scaler_family,
            verbose=verbose,
        )
    elif model_name == "TimeVAE_v3":
        model_obj, syn, ckpt_ext = _train_timevae_v3(
            x_scaled, window_size, params, seed, epochs,
            n_generate=n_generate, scaler_family=scaler_family,
            dataset=dataset, verbose=verbose,
        )
    elif model_name == "RTSGAN":
        model_obj, syn, ckpt_ext = _train_rtsgan(
            x_scaled, window_size, scaler_type, params, seed, epochs,
        )
    elif model_name == "DDPM":
        model_obj, syn, ckpt_ext = _train_ddpm(
            x_scaled, window_size, params, seed, epochs, verbose,
        )
    elif model_name == "TTSGAN":
        model_obj, syn, ckpt_ext = _train_ttsgan(
            x_scaled, window_size, scaler_type, params, seed,
        )
    elif model_name == "CSDI":
        model_obj, syn, ckpt_ext = _train_csdi(
            x_scaled, window_size, params, seed,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    np.nan_to_num(syn, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    ckpt_dir = root / "models_checkpoints" / dataset / model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"seed_{seed}{ckpt_ext}"

    if model_name in ("TimeVAE", "TimeVAE_v2", "DDPM"):
        model_obj.save_weights(str(ckpt_path))
    elif model_name == "TimeVAE_v3":
        from src.models.timevae_v3_wrapper import save_timevae_v3
        save_timevae_v3(model_obj, ckpt_path)
    elif model_name == "RTSGAN":
        from src.models.rtsgan_wrapper import save_rtsgan
        save_rtsgan(model_obj, ckpt_path)
    elif model_name == "TTSGAN":
        from src.models.ttsgan_wrapper import save_ttsgan
        save_ttsgan(model_obj, ckpt_path)
    elif model_name == "CSDI":
        from src.models.csdi_wrapper import save_csdi
        save_csdi(model_obj, ckpt_path)

    syn_dir = root / "data" / "synthetic" / dataset / model_name
    syn_dir.mkdir(parents=True, exist_ok=True)
    syn_path = syn_dir / f"seed_{seed}.npy"
    np.save(syn_path, syn)

    scaler_path = ckpt_dir / f"seed_{seed}_scaler.pkl"
    joblib.dump(scaler, str(scaler_path))
    stats_path: Path | None = None
    if isinstance(scaler, PerWindowZNormScaler):
        stats_path = ckpt_dir / f"seed_{seed}_per_window_stats.npy"
        np.save(stats_path, scaler.train_stats)

    real_sample = x_scaled[: min(n_generate, x_scaled.shape[0])]
    syn_sample = syn[: real_sample.shape[0]]
    ks, wass = compute_ks_wasserstein(real_sample, syn_sample)
    if not np.isfinite(ks):
        ks = float("nan")
    if not np.isfinite(wass):
        wass = float("nan")

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "file:./logs/mlflow")
    )
    mlflow.set_experiment(f"final_train_{dataset}_{model_name}")
    with mlflow.start_run(run_name=f"seed_{seed}"):
        mlflow.log_param("git_sha", get_git_sha())
        log_params: dict[str, Any] = {
            "dataset": dataset,
            "model": model_name,
            "seed": seed,
            "window_size": window_size,
            "stride": stride,
            "scaler_type": scaler_type,
            "epochs": epochs,
            "n_generate": n_generate,
        }
        for k in ("latent_dim", "beta", "learning_rate", "batch_size",
                   "hidden_dim", "layers", "noise_dim", "ae_lr", "gan_lr",
                   "n_filters", "n_conv_layers", "timesteps",
                   "embed_dim", "depth", "num_heads", "lr_g", "lr_d",
                   "reconstruction_wt", "kl_anneal_epochs",
                   "output_activation", "hidden_channels", "trend_poly",
                   "custom_seas_enabled", "seas_harmonics",
                   "use_residual_conn",
                   "channels", "nheads", "num_steps",
                   "timeemb", "featureemb", "schedule"):
            if k in params:
                log_params[k] = params[k]
        log_params["preprocessing_profile"] = preproc_meta["profile"]
        log_params["preproc_n_windows_kept"] = preproc_meta.get(
            "n_windows_kept", x_scaled.shape[0]
        )
        log_params["preproc_max_anomaly_ratio"] = preproc_meta.get(
            "max_anomaly_ratio", 0.0
        )
        log_params["preproc_buffer"] = preproc_meta.get("buffer", 0)
        mlflow.log_params(log_params)
        fidelity_composite = (ks + wass) / 2.0
        if not np.isfinite(fidelity_composite):
            fidelity_composite = float("nan")
        mlflow.log_metrics(
            {
                "ks_mean": ks,
                "wasserstein_mean": wass,
                "fidelity_composite": fidelity_composite,
            }
        )

    result = {
        "dataset": dataset,
        "model": model_name,
        "seed": seed,
        "checkpoint": str(ckpt_path),
        "synthetic_data": str(syn_path),
        "scaler": str(scaler_path),
        "per_window_stats": str(stats_path) if stats_path else None,
        "preprocessing_profile": preproc_meta["profile"],
        "preprocessing_meta": preproc_meta,
        "ks_mean": float(ks),
        "wasserstein_mean": float(wass),
        "n_windows_train": int(x_scaled.shape[0]),
        "n_windows_generated": int(syn.shape[0]),
    }
    print(f"  seed {seed}: KS={ks:.4f} W={wass:.4f} -> {syn_path}")
    return result


def train_all_seeds(
    dataset: str,
    model_name: str,
    seeds: list[int] | None = None,
    epochs: int | None = None,
    n_generate: int | None = None,
    verbose: int = 0,
) -> list[dict[str, Any]]:
    """Train across all project seeds and return per-seed results."""
    if seeds is None:
        seeds = list(SEEDS)
    results = []
    for seed in seeds:
        res = train_single_seed(
            dataset, model_name, seed,
            epochs=epochs, n_generate=n_generate, verbose=verbose,
        )
        results.append(res)
    return results
