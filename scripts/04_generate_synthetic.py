#!/usr/bin/env python3
"""Generate synthetic data from saved checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.preprocessor import (
    ScalerName,
    load_raw_splits,
    prepare_train_gen_windows,
)
from src.models._validation import profile_to_scaler_family
from src.training.train import _load_model_preprocessing_cfg, load_best_params
from src.training.utils import repo_root
from src.utils.seeds import SEEDS, set_seed


# ---------------------------------------------------------------------------
# Gaussian baseline (with profile-aware preprocessing)
# ---------------------------------------------------------------------------


def generate_gaussian_baseline(
    dataset: str,
    model_name: str,
    window_size: int,
    stride: int,
    scaler_type: ScalerName,
    seeds: list[int] | None = None,
    n_generate: int | None = None,
) -> Path:
    """Produce a Gaussian baseline that lives in the same data space."""
    if seeds is None:
        seeds = list(SEEDS)

    root = repo_root()
    processed = root / "data" / "processed" / dataset
    splits = load_raw_splits(processed)
    profile, max_ar, buf = _load_model_preprocessing_cfg(model_name)

    x_scaled, _scaler, meta = prepare_train_gen_windows(
        splits,
        window_size=window_size,
        stride=stride,
        scaler_name=scaler_type,
        profile=profile,  # type: ignore[arg-type]
        max_anomaly_ratio=max_ar,
        buffer=buf,
        min_windows=1,
    )

    if n_generate is None:
        n_generate = int(x_scaled.shape[0])

    flat = x_scaled.reshape(-1, x_scaled.shape[-1]).astype(np.float64)
    feat_mean = flat.mean(axis=0)
    feat_std = flat.std(axis=0)
    # Guard against constant features (zero variance): use a small floor so
    # downstream metrics that compare per-feature don't see all-zero columns.
    feat_std = np.where(feat_std > 1e-8, feat_std, 1e-8)

    out_dir = root / "data" / "synthetic" / dataset / f"GaussianNoise_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        noise = rng.standard_normal(
            size=(n_generate, window_size, x_scaled.shape[-1])
        )
        noise = (
            noise * feat_std[None, None, :] + feat_mean[None, None, :]
        ).astype(np.float32)
        out_path = out_dir / f"seed_{seed}.npy"
        np.save(out_path, noise)
        print(
            f"  GaussianNoise_{model_name} seed {seed}: "
            f"{noise.shape} -> {out_path}"
        )

    # Persist the provenance so future evaluation knows the data space.
    sidecar = out_dir / "preprocessing.json"
    with sidecar.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "for_model": model_name,
                "profile": profile,
                "scaler_type": scaler_type,
                "scaler_family": profile_to_scaler_family(
                    profile,
                    sklearn_scaler_name=scaler_type if profile == "legacy" else None,
                ),
                "window_size": int(window_size),
                "stride": int(stride),
                "n_windows_used_for_stats": int(x_scaled.shape[0]),
                "preprocessing_meta": meta,
            },
            f,
            indent=2,
        )
    return out_dir


# ---------------------------------------------------------------------------
# Per-model regeneration
# ---------------------------------------------------------------------------


class _SkippedTracker:
    """Bookkeeping for missing checkpoints across one regenerate call."""

    def __init__(self) -> None:
        self.missing: list[Path] = []

    def add(self, path: Path) -> None:
        self.missing.append(path)
        print(f"  WARNING: checkpoint missing: {path}")


def _regenerate_timevae(
    root: Path,
    dataset: str,
    model_name: str,
    params: dict,
    window_size: int,
    feat_dim: int,
    n_generate: int,
    seeds: list[int],
    skipped: _SkippedTracker,
) -> None:
    """Reload TimeVAE checkpoints and regenerate synthetic data."""
    from src.models.timevae_wrapper import build_beta_vae, generate_numpy

    latent_dim = int(params["latent_dim"])
    beta = float(params["beta"])
    syn_dir = root / "data" / "synthetic" / dataset / model_name
    syn_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        ckpt_path = (
            root / "models_checkpoints" / dataset / model_name
            / f"seed_{seed}.weights.h5"
        )
        if not ckpt_path.exists():
            skipped.add(ckpt_path)
            continue

        set_seed(seed)
        vae = build_beta_vae(window_size, feat_dim, latent_dim, beta=beta)
        vae(np.zeros((1, window_size, feat_dim), dtype=np.float32))
        vae.load_weights(str(ckpt_path))

        syn = generate_numpy(vae, n_generate)
        out_path = syn_dir / f"seed_{seed}.npy"
        np.save(out_path, syn)
        print(f"  {model_name} seed {seed}: {syn.shape} -> {out_path}")


def _regenerate_timevae_v2(
    root: Path,
    dataset: str,
    model_name: str,
    params: dict,
    window_size: int,
    feat_dim: int,
    n_generate: int,
    seeds: list[int],
    scaler_family: str,
    skipped: _SkippedTracker,
) -> None:
    """Reload TimeVAE_v2 checkpoints and regenerate synthetic data.

    Generation is seeded via ''keras.utils.set_random_seed(seed)'' inside
    :func:`generate_numpy` so the regenerated windows match the ones the
    training stage saved alongside the checkpoint.
    """
    from src.models.timevae_v2_wrapper import build_timevae_v2, generate_numpy

    latent_dim = int(params["latent_dim"])
    beta = float(params["beta"])
    reconstruction_wt = float(params.get("reconstruction_wt", 3.0))
    kl_anneal_epochs = int(params.get("kl_anneal_epochs", 0))
    output_activation = str(params.get("output_activation", "linear"))
    lr = float(params.get("learning_rate", 1e-3))

    syn_dir = root / "data" / "synthetic" / dataset / model_name
    syn_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        ckpt_path = (
            root / "models_checkpoints" / dataset / model_name
            / f"seed_{seed}.weights.h5"
        )
        if not ckpt_path.exists():
            skipped.add(ckpt_path)
            continue

        # Build with the same scaler/activation contract as training so a
        # mismatched best_params file fails loud at reload time.
        vae = build_timevae_v2(
            window_size, feat_dim, latent_dim,
            beta=beta,
            reconstruction_wt=reconstruction_wt,
            kl_anneal_epochs=kl_anneal_epochs,
            output_activation=output_activation,  # type: ignore[arg-type]
            learning_rate=lr,
            scaler_family=scaler_family,  # type: ignore[arg-type]
        )
        vae(np.zeros((1, window_size, feat_dim), dtype=np.float32))
        vae.load_weights(str(ckpt_path))

        syn = generate_numpy(vae, n_generate, seed=seed)
        out_path = syn_dir / f"seed_{seed}.npy"
        np.save(out_path, syn)
        print(f"  {model_name} seed {seed}: {syn.shape} -> {out_path}")


def _regenerate_timevae_v3(
    root: Path,
    dataset: str,
    model_name: str,
    n_generate: int,
    seeds: list[int],
    skipped: _SkippedTracker,
) -> None:
    """Reload TimeVAE_v3 ''.pt'' checkpoints and regenerate synthetic data.

    The wrapper restores the architecture from the sidecar ''meta.json''
    and uses a local ''torch.Generator'' seeded with ''seed'' for the
    latent draw, so the output is byte-identical across reruns and
    matches the synthetic data saved at training time.
    """
    from src.models.timevae_v3_wrapper import generate_numpy, load_timevae_v3

    syn_dir = root / "data" / "synthetic" / dataset / model_name
    syn_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        ckpt_path = (
            root / "models_checkpoints" / dataset / model_name
            / f"seed_{seed}.pt"
        )
        if not ckpt_path.exists():
            skipped.add(ckpt_path)
            continue

        model = load_timevae_v3(ckpt_path)
        syn = generate_numpy(model, n_generate, seed=seed)
        out_path = syn_dir / f"seed_{seed}.npy"
        np.save(out_path, syn)
        print(f"  {model_name} seed {seed}: {syn.shape} -> {out_path}")


def _regenerate_ddpm(
    root: Path,
    dataset: str,
    model_name: str,
    params: dict,
    window_size: int,
    feat_dim: int,
    n_generate: int,
    seeds: list[int],
    skipped: _SkippedTracker,
) -> None:
    """Reload DDPM checkpoints and regenerate synthetic data."""
    from src.models.ddpm_wrapper import build_ddpm, generate_ddpm

    n_filters = int(params["n_filters"])
    n_conv_layers = int(params["n_conv_layers"])
    timesteps = int(params["timesteps"])
    syn_dir = root / "data" / "synthetic" / dataset / model_name
    syn_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        ckpt_path = (
            root / "models_checkpoints" / dataset / model_name
            / f"seed_{seed}.weights.h5"
        )
        if not ckpt_path.exists():
            skipped.add(ckpt_path)
            continue

        set_seed(seed)
        ddpm = build_ddpm(
            window_size, feat_dim,
            n_filters=n_filters,
            n_conv_layers=n_conv_layers,
            timesteps=timesteps,
        )
        ddpm(np.zeros((1, window_size, feat_dim), dtype=np.float32))
        ddpm.load_weights(str(ckpt_path))

        syn = generate_ddpm(ddpm, n_generate)
        out_path = syn_dir / f"seed_{seed}.npy"
        np.save(out_path, syn)
        print(f"  {model_name} seed {seed}: {syn.shape} -> {out_path}")


def _regenerate_csdi(
    root: Path,
    dataset: str,
    model_name: str,
    params: dict,
    window_size: int,
    feat_dim: int,
    n_generate: int,
    seeds: list[int],
    skipped: _SkippedTracker,
) -> None:
    """Reload CSDI ''.pt'' checkpoints and regenerate synthetic data."""
    from src.models.csdi_wrapper import generate_csdi, load_csdi

    channels = int(params["channels"])
    layers = int(params["layers"])
    nheads = int(params["nheads"])
    num_steps = int(params["num_steps"])
    timeemb = int(params["timeemb"])
    featureemb = int(params["featureemb"])
    schedule = str(params["schedule"])

    syn_dir = root / "data" / "synthetic" / dataset / model_name
    syn_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        ckpt_path = (
            root / "models_checkpoints" / dataset / model_name / f"seed_{seed}.pt"
        )
        if not ckpt_path.exists():
            skipped.add(ckpt_path)
            continue

        set_seed(seed)
        model = load_csdi(
            ckpt_path,
            seq_len=window_size,
            feat_dim=feat_dim,
            channels=channels,
            layers=layers,
            nheads=nheads,
            num_steps=num_steps,
            timeemb=timeemb,
            featureemb=featureemb,
            schedule=schedule,
        )

        syn = generate_csdi(model, n_generate)
        out_path = syn_dir / f"seed_{seed}.npy"
        np.save(out_path, syn)
        print(f"  {model_name} seed {seed}: {syn.shape} -> {out_path}")


def _regenerate_ttsgan(
    root: Path,
    dataset: str,
    model_name: str,
    params: dict,
    window_size: int,
    feat_dim: int,
    n_generate: int,
    seeds: list[int],
    scaler_type: str,
    skipped: _SkippedTracker,
) -> None:
    """Reload TTS-GAN checkpoints and regenerate synthetic data."""
    from src.models.ttsgan_wrapper import generate_ttsgan, load_ttsgan

    latent_dim = int(params["latent_dim"])
    embed_dim = int(params["embed_dim"])
    depth = int(params["depth"])
    num_heads = int(params["num_heads"])
    output_sigmoid = scaler_type == "MinMax"

    syn_dir = root / "data" / "synthetic" / dataset / model_name
    syn_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        ckpt_path = (
            root / "models_checkpoints" / dataset / model_name / f"seed_{seed}.pt"
        )
        if not ckpt_path.exists():
            skipped.add(ckpt_path)
            continue

        set_seed(seed)
        model = load_ttsgan(
            ckpt_path,
            seq_len=window_size,
            feat_dim=feat_dim,
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            output_sigmoid=output_sigmoid,
        )

        syn = generate_ttsgan(model, n_generate)
        out_path = syn_dir / f"seed_{seed}.npy"
        np.save(out_path, syn)
        print(f"  {model_name} seed {seed}: {syn.shape} -> {out_path}")


def _regenerate_rtsgan(
    root: Path,
    dataset: str,
    model_name: str,
    params: dict,
    window_size: int,
    feat_dim: int,
    n_generate: int,
    seeds: list[int],
    scaler_type: str,
    skipped: _SkippedTracker,
) -> None:
    """Reload RTSGAN checkpoints and regenerate synthetic data."""
    from src.models.rtsgan_wrapper import generate_rtsgan, load_rtsgan

    hidden_dim = int(params["hidden_dim"])
    layers = int(params["layers"])
    noise_dim = int(params["noise_dim"])
    output_sigmoid = scaler_type == "MinMax"

    syn_dir = root / "data" / "synthetic" / dataset / model_name
    syn_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        ckpt_path = (
            root / "models_checkpoints" / dataset / model_name / f"seed_{seed}.pt"
        )
        if not ckpt_path.exists():
            skipped.add(ckpt_path)
            continue

        set_seed(seed)
        model = load_rtsgan(
            ckpt_path,
            seq_len=window_size,
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            noise_dim=noise_dim,
            layers=layers,
            output_sigmoid=output_sigmoid,
        )

        syn = generate_rtsgan(model, n_generate)
        out_path = syn_dir / f"seed_{seed}.npy"
        np.save(out_path, syn)
        print(f"  {model_name} seed {seed}: {syn.shape} -> {out_path}")


def regenerate_from_checkpoints(
    dataset: str,
    model_name: str,
    seeds: list[int] | None = None,
    n_generate: int | None = None,
) -> _SkippedTracker:
    """Load saved model weights and regenerate synthetic data.

    Returns a :class:`_SkippedTracker` recording every checkpoint that
    could not be found, so the caller can decide whether to exit non-zero
    (strict / default) or continue (''--allow-partial'').
    """
    if seeds is None:
        seeds = list(SEEDS)

    os.environ.setdefault("KERAS_BACKEND", "torch")

    root = repo_root()
    params = load_best_params(dataset, model_name)

    window_size = int(params["window_size"])
    stride = int(params["stride"])
    scaler_type: ScalerName = str(params["scaler_type"])  # type: ignore[assignment]

    processed = root / "data" / "processed" / dataset
    splits = load_raw_splits(processed)
    profile, max_ar, buf = _load_model_preprocessing_cfg(model_name)
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
    feat_dim = x_scaled.shape[-1]

    if n_generate is None:
        n_generate = x_scaled.shape[0]

    scaler_family = profile_to_scaler_family(
        profile, sklearn_scaler_name=scaler_type if profile == "legacy" else None,
    )

    skipped = _SkippedTracker()

    if model_name == "TimeVAE":
        _regenerate_timevae(
            root, dataset, model_name, params,
            window_size, feat_dim, n_generate, seeds, skipped,
        )
    elif model_name == "TimeVAE_v2":
        _regenerate_timevae_v2(
            root, dataset, model_name, params,
            window_size, feat_dim, n_generate, seeds,
            scaler_family, skipped,
        )
    elif model_name == "TimeVAE_v3":
        _regenerate_timevae_v3(
            root, dataset, model_name, n_generate, seeds, skipped,
        )
    elif model_name == "DDPM":
        _regenerate_ddpm(
            root, dataset, model_name, params,
            window_size, feat_dim, n_generate, seeds, skipped,
        )
    elif model_name == "TTSGAN":
        _regenerate_ttsgan(
            root, dataset, model_name, params,
            window_size, feat_dim, n_generate, seeds, scaler_type, skipped,
        )
    elif model_name == "CSDI":
        _regenerate_csdi(
            root, dataset, model_name, params,
            window_size, feat_dim, n_generate, seeds, skipped,
        )
    elif model_name == "RTSGAN":
        _regenerate_rtsgan(
            root, dataset, model_name, params,
            window_size, feat_dim, n_generate, seeds, scaler_type, skipped,
        )
    else:
        raise ValueError(f"Unknown model for regeneration: {model_name}")

    return skipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset", help="Dataset key, e.g. SWaT")
    parser.add_argument(
        "model",
        nargs="?",
        default="TimeVAE",
        help="Model key (default: %(default)s)",
    )
    parser.add_argument(
        "--n-generate", type=int, default=None,
        help="Number of synthetic windows (default: match train_gen)",
    )
    parser.add_argument(
        "--gaussian-only", action="store_true",
        help="Only generate the Gaussian noise baseline",
    )
    parser.add_argument(
        "--skip-gaussian", action="store_true",
        help="Skip the Gaussian noise baseline",
    )
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Continue without an error exit even if some seeds are missing",
    )
    args = parser.parse_args()

    try:
        params = load_best_params(args.dataset, args.model)
    except FileNotFoundError:
        if not args.gaussian_only:
            raise
        # Gaussian-only mode: best_params for this model is not required.
        # Fall back to any sibling best_params_{dataset}_*.json for
        # window/stride/scaler values.
        results_dir = Path(__file__).resolve().parents[1] / "results"
        candidates = sorted(results_dir.glob(f"best_params_{args.dataset}_*.json"))
        if not candidates:
            raise
        fallback = candidates[0]
        print(
            f"  (gaussian-only) using fallback window/stride/scaler from "
            f"{fallback.name}"
        )
        with fallback.open(encoding="utf-8") as f:
            params = json.load(f)

    window_size = int(params["window_size"])
    stride = int(params["stride"])
    scaler_type: ScalerName = str(params["scaler_type"])  # type: ignore[assignment]

    if not args.skip_gaussian:
        print(
            f"Generating Gaussian noise baseline for {args.dataset} "
            f"(matched to {args.model} preprocessing)..."
        )
        generate_gaussian_baseline(
            args.dataset, args.model, window_size, stride, scaler_type,
            n_generate=args.n_generate,
        )

    n_skipped = 0
    if not args.gaussian_only:
        print(f"Regenerating {args.model} synthetic data for {args.dataset}...")
        skipped = regenerate_from_checkpoints(
            args.dataset, args.model, n_generate=args.n_generate,
        )
        n_skipped = len(skipped.missing)
        if n_skipped:
            print(
                f"  Summary: {n_skipped} checkpoint(s) missing for "
                f"{args.dataset}/{args.model}:"
            )
            for p in skipped.missing:
                print(f"    - {p}")

    print("Done.")
    if n_skipped > 0 and not args.allow_partial:
        sys.exit(1)


if __name__ == "__main__":
    main()
