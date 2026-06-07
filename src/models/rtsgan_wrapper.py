"""High-level wrapper for RTSGAN matching the TimeVAE function interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.models.rtsgan.aegan import AeGAN
from src.utils.seeds import set_seed


def build_rtsgan(
    seq_len: int,
    feat_dim: int,
    hidden_dim: int = 24,
    embed_dim: int = 96,
    noise_dim: int = 96,
    layers: int = 3,
    dropout: float = 0.0,
    output_sigmoid: bool = True,
) -> AeGAN:
    """Instantiate an RTSGAN model on the best available device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return AeGAN(
        feat_dim=feat_dim,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        noise_dim=noise_dim,
        layers=layers,
        dropout=dropout,
        output_sigmoid=output_sigmoid,
        device=device,
    )


def fit_rtsgan(
    model: AeGAN,
    x: np.ndarray,
    ae_epochs: int = 200,
    gan_iterations: int = 5000,
    ae_batch_size: int = 128,
    gan_batch_size: int = 256,
    ae_lr: float = 1e-3,
    gan_lr: float = 1e-4,
    d_update: int = 5,
    seed: int = 0,
    verbose: int = 0,
) -> None:
    """Train autoencoder then WGAN-GP.

    Args:
        model: "AeGAN" instance from :func:'build_rtsgan'.
        x: Training windows "(N, L, F)" as float32 numpy array.
        ae_epochs: Autoencoder training epochs.
        gan_iterations: WGAN-GP training iterations.
        ae_batch_size: Batch size for autoencoder phase.
        gan_batch_size: Batch size for GAN phase.
        ae_lr: Autoencoder Adam learning rate.
        gan_lr: WGAN RMSprop learning rate.
        d_update: Discriminator updates per generator step.
        seed: Random seed.
        verbose: Unused (kept for interface parity).
    """
    set_seed(seed)
    x_tensor = torch.from_numpy(x.astype(np.float32))
    model.train_ae(x_tensor, epochs=ae_epochs, batch_size=ae_batch_size, lr=ae_lr)
    model.train_gan(
        x_tensor,
        iterations=gan_iterations,
        batch_size=gan_batch_size,
        d_update=d_update,
        lr=gan_lr,
    )


def generate_rtsgan(model: AeGAN, n: int) -> np.ndarray:
    """Generate "n" synthetic windows as a float32 numpy array "(n, L, F)"."""
    syn = model.synthesize(n)
    return syn.numpy().astype(np.float32)


def save_rtsgan(model: AeGAN, path: str | Path) -> None:
    """Save RTSGAN checkpoint."""
    model.save(path)


def load_rtsgan(
    path: str | Path,
    seq_len: int,
    feat_dim: int,
    hidden_dim: int = 24,
    embed_dim: int = 96,
    noise_dim: int = 96,
    layers: int = 3,
    dropout: float = 0.0,
    output_sigmoid: bool = True,
) -> AeGAN:
    """Rebuild an RTSGAN model and load weights from checkpoint."""
    model = build_rtsgan(
        seq_len=seq_len,
        feat_dim=feat_dim,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        noise_dim=noise_dim,
        layers=layers,
        dropout=dropout,
        output_sigmoid=output_sigmoid,
    )
    model.load(path)
    return model


__all__ = [
    "build_rtsgan",
    "fit_rtsgan",
    "generate_rtsgan",
    "save_rtsgan",
    "load_rtsgan",
]
