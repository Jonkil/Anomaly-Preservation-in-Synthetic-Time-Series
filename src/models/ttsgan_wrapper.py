"""High-level wrapper for TTS-GAN matching the RTSGAN function interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.models.ttsgan.trainer import TTSGAN
from src.models.ttsgan.transformer import pick_patch_size
from src.utils.seeds import set_seed


def build_ttsgan(
    seq_len: int,
    feat_dim: int,
    latent_dim: int = 100,
    embed_dim: int = 16,
    depth: int = 3,
    num_heads: int = 4,
    patch_size: int | None = None,
    dropout: float = 0.1,
    output_sigmoid: bool = False,
) -> TTSGAN:
    """Instantiate a TTS-GAN model on the best available device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return TTSGAN(
        feat_dim=feat_dim,
        seq_len=seq_len,
        latent_dim=latent_dim,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_size=patch_size,
        dropout=dropout,
        output_sigmoid=output_sigmoid,
        device=device,
    )


def _to_internal(x: np.ndarray) -> torch.Tensor:
    """Convert ''(N, L, F)'' numpy windows to ''(N, F, 1, L)'' torch tensor."""
    if x.ndim != 3:
        raise ValueError(f"Expected (N, L, F) array, got shape {x.shape}")
    arr = np.asarray(x, dtype=np.float32)
    arr = np.transpose(arr, (0, 2, 1))[:, :, None, :]
    return torch.from_numpy(np.ascontiguousarray(arr))


def fit_ttsgan(
    model: TTSGAN,
    x: np.ndarray,
    iterations: int = 5000,
    batch_size: int = 64,
    lr_g: float = 1e-4,
    lr_d: float = 3e-4,
    d_update: int = 3,
    seed: int = 0,
    verbose: int = 0,
) -> None:
    """Train TTS-GAN with an LSGAN objective."""
    del verbose
    set_seed(seed)
    x_tensor = _to_internal(x)
    model.train(
        x_tensor,
        iterations=iterations,
        batch_size=batch_size,
        lr_g=lr_g,
        lr_d=lr_d,
        d_update=d_update,
    )


def generate_ttsgan(model: TTSGAN, n: int) -> np.ndarray:
    """Generate ''n'' synthetic windows as a float32 numpy array ''(n, L, F)''."""
    out = model.generate(n).numpy().astype(np.float32)
    return np.transpose(out[:, :, 0, :], (0, 2, 1))


def save_ttsgan(model: TTSGAN, path: str | Path) -> None:
    """Save a TTS-GAN checkpoint."""
    model.save(path)


def load_ttsgan(
    path: str | Path,
    seq_len: int,
    feat_dim: int,
    latent_dim: int = 100,
    embed_dim: int = 16,
    depth: int = 3,
    num_heads: int = 4,
    patch_size: int | None = None,
    dropout: float = 0.1,
    output_sigmoid: bool = False,
) -> TTSGAN:
    """Rebuild a TTS-GAN model and load weights from checkpoint."""
    model = build_ttsgan(
        seq_len=seq_len,
        feat_dim=feat_dim,
        latent_dim=latent_dim,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_size=patch_size,
        dropout=dropout,
        output_sigmoid=output_sigmoid,
    )
    model.load(path)
    return model


__all__ = [
    "build_ttsgan",
    "fit_ttsgan",
    "generate_ttsgan",
    "save_ttsgan",
    "load_ttsgan",
    "pick_patch_size",
]
