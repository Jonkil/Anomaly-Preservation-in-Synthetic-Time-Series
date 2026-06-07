"""High-level wrapper for CSDI matching the TTS-GAN wrapper interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.models.csdi.trainer import CSDI
from src.utils.seeds import set_seed


def build_csdi(
    seq_len: int,
    feat_dim: int,
    channels: int = 64,
    layers: int = 4,
    nheads: int = 8,
    num_steps: int = 100,
    diffusion_embedding_dim: int = 128,
    timeemb: int = 128,
    featureemb: int = 16,
    schedule: str = "quad",
    beta_start: float = 1e-4,
    beta_end: float = 0.5,
    is_linear: bool = False,
) -> CSDI:
    """Instantiate an unconditional CSDI model on the best available device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return CSDI(
        feat_dim=feat_dim,
        seq_len=seq_len,
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
        device=device,
    )


def _to_internal(x: np.ndarray) -> torch.Tensor:
    """Convert ''(N, L, F)'' numpy windows to ''(N, F, L)'' torch tensor."""
    if x.ndim != 3:
        raise ValueError(f"Expected (N, L, F) array, got shape {x.shape}")
    arr = np.asarray(x, dtype=np.float32)
    arr = np.transpose(arr, (0, 2, 1))
    return torch.from_numpy(np.ascontiguousarray(arr))


def fit_csdi(
    model: CSDI,
    x: np.ndarray,
    iterations: int = 5000,
    batch_size: int = 32,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    seed: int = 0,
    verbose: int = 0,
) -> None:
    """Train CSDI on ''(N, L, F)'' windows with an Adam loop."""
    del verbose
    set_seed(seed)
    x_tensor = _to_internal(x)
    model.train(
        x_tensor,
        iterations=iterations,
        batch_size=batch_size,
        lr=lr,
        grad_clip=grad_clip,
    )


def generate_csdi(model: CSDI, n: int, batch_size: int = 64) -> np.ndarray:
    """Generate ''n'' synthetic windows as a float32 numpy array ''(n, L, F)''."""
    out = model.generate(n, batch_size=batch_size).numpy().astype(np.float32)
    return np.transpose(out, (0, 2, 1))


def save_csdi(model: CSDI, path: str | Path) -> None:
    """Save a CSDI checkpoint."""
    model.save(path)


def load_csdi(
    path: str | Path,
    seq_len: int,
    feat_dim: int,
    channels: int = 64,
    layers: int = 4,
    nheads: int = 8,
    num_steps: int = 100,
    diffusion_embedding_dim: int = 128,
    timeemb: int = 128,
    featureemb: int = 16,
    schedule: str = "quad",
    beta_start: float = 1e-4,
    beta_end: float = 0.5,
    is_linear: bool = False,
) -> CSDI:
    """Rebuild a CSDI model and load weights from checkpoint."""
    model = build_csdi(
        seq_len=seq_len,
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
    model.load(path)
    return model


__all__ = [
    "build_csdi",
    "fit_csdi",
    "generate_csdi",
    "save_csdi",
    "load_csdi",
]
