"""TSGM DDPM (Denoising Diffusion Probabilistic Model) wrapper."""

from __future__ import annotations

import os

import keras
import numpy as np

from tsgm.models.architectures.zoo import zoo
from tsgm.models.ddpm import DDPM

from src.utils.tensor import keras_to_numpy


def build_ddpm(
    seq_len: int,
    feat_dim: int,
    n_filters: int = 64,
    n_conv_layers: int = 3,
    timesteps: int = 200,
    ema: float = 0.999,
) -> DDPM:
    """Instantiate a compiled TSGM :class:`DDPM` with conv denoiser."""
    os.environ.setdefault("KERAS_BACKEND", "torch")
    arch = zoo["ddpm_denoiser"](
        seq_len=seq_len,
        feat_dim=feat_dim,
        n_filters=n_filters,
        n_conv_layers=n_conv_layers,
    )
    network = arch.model
    ema_network = keras.models.clone_model(network)
    ddpm = DDPM(
        network=network,
        ema_network=ema_network,
        timesteps=timesteps,
        ema=ema,
    )
    ddpm.compile(optimizer=keras.optimizers.Adam(1e-3))
    return ddpm




def fit_ddpm(
    ddpm: DDPM,
    x: np.ndarray,
    epochs: int,
    batch_size: int,
    seed: int,
    verbose: int = 0,
) -> None:
    """Train DDPM with deterministic shuffle controlled by *seed*."""
    keras.utils.set_random_seed(seed)
    ddpm.fit(
        x,
        epochs=epochs,
        batch_size=min(batch_size, max(1, x.shape[0])),
        verbose=verbose,
        shuffle=True,
    )


def generate_ddpm(ddpm: DDPM, n: int, batch_size: int = 256) -> np.ndarray:
    """Generate *n* samples as a float32 numpy array ''(n, L, F)''.

    Generates in batches to avoid OOM on large sample counts.
    """
    chunks: list[np.ndarray] = []
    remaining = n
    while remaining > 0:
        bs = min(batch_size, remaining)
        out = ddpm.generate(bs)
        chunks.append(keras_to_numpy(out).astype(np.float32))
        remaining -= bs
    return np.concatenate(chunks, axis=0)[:n]


__all__ = [
    "build_ddpm",
    "fit_ddpm",
    "generate_ddpm",
    "keras_to_numpy",
]
