"""TSGM BetaVAE (conv VAE) wrapper for TimeVAE-style tuning."""

from __future__ import annotations

import os
from typing import Any

import keras
import numpy as np

from tsgm.models.architectures.zoo import zoo
from tsgm.models.cvae import BetaVAE

from src.utils.tensor import keras_to_numpy


def build_beta_vae(
    seq_len: int,
    feat_dim: int,
    latent_dim: int,
    beta: float = 1.0,
    arch_key: str = "vae_conv5",
) -> BetaVAE:
    """Instantiate a compiled TSGM :class:'BetaVAE' with conv architecture."""
    os.environ.setdefault("KERAS_BACKEND", "torch")
    arch = zoo[arch_key](seq_len=seq_len, feat_dim=feat_dim, latent_dim=latent_dim)
    vae = BetaVAE(arch.encoder, arch.decoder, beta=beta)
    vae.compile(optimizer=keras.optimizers.Adam(1e-3))
    return vae




def fit_beta_vae(
    vae: BetaVAE,
    x: np.ndarray,
    epochs: int,
    batch_size: int,
    seed: int,
    verbose: int = 0,
) -> None:
    """Fit VAE with deterministic shuffle controlled by "seed"."""
    keras.utils.set_random_seed(seed)
    vae.fit(
        x,
        epochs=epochs,
        batch_size=min(batch_size, max(1, x.shape[0])),
        verbose=verbose,
        shuffle=True,
    )


def generate_numpy(vae: BetaVAE, n: int) -> np.ndarray:
    """Generate "n" samples as a float32 "numpy" array "(n, L, F)"."""
    out = vae.generate(n)
    return keras_to_numpy(out).astype(np.float32)


__all__ = [
    "build_beta_vae",
    "fit_beta_vae",
    "generate_numpy",
    "keras_to_numpy",
]
