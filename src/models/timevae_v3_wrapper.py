"""TimeVAE_v3: native PyTorch interpretable TimeVAE."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models._validation import (
    OutputActivation,
    ScalerFamily,
    profile_to_scaler_family,
    validate_scaler_activation,
)

_LOG = logging.getLogger(__name__)


@dataclass
class TimeVAEv3Config:
    """Build-time configuration for :class:`TimeVAEv3`."""

    seq_len: int
    feat_dim: int
    latent_dim: int = 32
    hidden_channels: Sequence[int] = field(default_factory=lambda: [64, 128, 256])
    trend_poly: int = 1
    custom_seas: Sequence[tuple[int, int]] = field(default_factory=list)
    use_residual_conn: bool = True
    output_activation: OutputActivation = "linear"
    reconstruction_wt: float = 3.0
    beta: float = 1.0
    kl_anneal_epochs: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hidden_channels"] = list(self.hidden_channels)
        d["custom_seas"] = [list(p) for p in self.custom_seas]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TimeVAEv3Config":
        seas = [tuple(x) for x in d.get("custom_seas", [])]
        return cls(
            seq_len=int(d["seq_len"]),
            feat_dim=int(d["feat_dim"]),
            latent_dim=int(d.get("latent_dim", 32)),
            hidden_channels=list(d.get("hidden_channels", [64, 128, 256])),
            trend_poly=int(d.get("trend_poly", 1)),
            custom_seas=seas,
            use_residual_conn=bool(d.get("use_residual_conn", True)),
            output_activation=d.get("output_activation", "linear"),
            reconstruction_wt=float(d.get("reconstruction_wt", 3.0)),
            beta=float(d.get("beta", 1.0)),
            kl_anneal_epochs=int(d.get("kl_anneal_epochs", 0)),
        )


def _conv_encoder_seq_len(input_len: int, n_layers: int) -> int:
    """Length after ''n_layers'' of ''Conv1d(kernel=3, stride=2, padding=1)''."""
    if input_len < 1:
        raise ValueError(f"input_len must be >= 1, got {input_len}")
    L = int(input_len)
    for _ in range(int(n_layers)):
        L = (L + 1) // 2
    return L


class _ConvEncoder(nn.Module):
    """Stacked Conv1d -> flatten -> (mu, logvar)."""

    def __init__(self, cfg: TimeVAEv3Config) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = cfg.feat_dim
        for ch in cfg.hidden_channels:
            layers += [
                nn.Conv1d(in_ch, ch, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
            ]
            in_ch = ch
        self.conv = nn.Sequential(*layers)
        self.seq_len = cfg.seq_len
        self.hidden_channels = list(cfg.hidden_channels)
        out_seq = _conv_encoder_seq_len(cfg.seq_len, len(cfg.hidden_channels))
        flat_dim = int(cfg.hidden_channels[-1]) * out_seq
        self.flat_dim = int(flat_dim)
        self.out_seq = int(out_seq)
        self.fc_mu = nn.Linear(self.flat_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, F) -> (B, F, T) for Conv1d.
        h = self.conv(x.transpose(1, 2)).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class _LevelHead(nn.Module):
    """Constant per-sample offset ''(B, 1, F)'' broadcast to ''(B, T, F)''."""

    def __init__(self, cfg: TimeVAEv3Config) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.latent_dim, cfg.feat_dim)
        self.seq_len = cfg.seq_len

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        level = self.fc(z).unsqueeze(1)  # (B, 1, F)
        return level.expand(-1, self.seq_len, -1)


class _TrendHead(nn.Module):
    """Polynomial trend ''sum_{k=1..trend_poly} c_k * t^k''."""

    def __init__(self, cfg: TimeVAEv3Config) -> None:
        super().__init__()
        self.trend_poly = int(cfg.trend_poly)
        self.feat_dim = int(cfg.feat_dim)
        self.seq_len = int(cfg.seq_len)
        if self.trend_poly < 0:
            raise ValueError(f"trend_poly must be >= 0; got {self.trend_poly}")
        if self.trend_poly == 0:
            self.fc: nn.Linear | None = None
            self.register_buffer(
                "basis", torch.zeros(0, cfg.seq_len), persistent=False,
            )
            return
        self.fc = nn.Linear(cfg.latent_dim, self.trend_poly * cfg.feat_dim)
        t = torch.linspace(0.0, 1.0, steps=cfg.seq_len)
        # Powers t^1, t^2, ..., t^trend_poly (NO constant term).
        basis = torch.stack(
            [t ** k for k in range(1, self.trend_poly + 1)], dim=0,
        )  # (trend_poly, T)
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        if self.trend_poly == 0 or self.fc is None:
            return torch.zeros(b, self.seq_len, self.feat_dim, device=z.device)
        coeffs = self.fc(z).view(b, self.trend_poly, self.feat_dim)  # (B, K, F)
        return torch.einsum("bkf,kt->btf", coeffs, self.basis)


class _SeasonalityHead(nn.Module):
    """Per-period Fourier basis with coefficients predicted from ''z''."""

    def __init__(self, cfg: TimeVAEv3Config) -> None:
        super().__init__()
        self.feat_dim = int(cfg.feat_dim)
        self.seq_len = int(cfg.seq_len)
        bases = []
        for raw_period, raw_harmonics in cfg.custom_seas:
            period = int(raw_period)
            harmonics = int(raw_harmonics)
            if period <= 1 or harmonics < 1:
                _LOG.warning(
                    "_SeasonalityHead: dropping invalid (period=%s, "
                    "harmonics=%s) - period must be > 1 and harmonics >= 1",
                    raw_period, raw_harmonics,
                )
                continue
            t = torch.arange(cfg.seq_len, dtype=torch.float32)
            for k in range(1, harmonics + 1):
                bases.append(torch.sin(2.0 * math.pi * k * t / period))
                bases.append(torch.cos(2.0 * math.pi * k * t / period))
        if bases:
            basis = torch.stack(bases, dim=0)  # (n_basis, T)
            self.n_basis = int(basis.shape[0])
            self.register_buffer("basis", basis, persistent=False)
            self.fc: nn.Linear | None = nn.Linear(
                cfg.latent_dim, self.n_basis * cfg.feat_dim,
            )
        else:
            self.n_basis = 0
            self.register_buffer(
                "basis", torch.zeros(0, cfg.seq_len), persistent=False,
            )
            self.fc = None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        if self.n_basis == 0 or self.fc is None:
            return torch.zeros(b, self.seq_len, self.feat_dim, device=z.device)
        coeffs = self.fc(z).view(b, self.n_basis, self.feat_dim)
        return torch.einsum("bkf,kt->btf", coeffs, self.basis)


class _ResidualConvHead(nn.Module):
    """Conv stack mirroring the encoder, acting on a ''z'' upsample."""

    def __init__(self, cfg: TimeVAEv3Config) -> None:
        super().__init__()
        self.feat_dim = cfg.feat_dim
        channels = list(cfg.hidden_channels)
        out_seq = _conv_encoder_seq_len(cfg.seq_len, len(channels))
        self.start_shape = (channels[-1], out_seq)
        self.fc = nn.Linear(cfg.latent_dim, self.start_shape[0] * self.start_shape[1])

        layers: list[nn.Module] = []
        rev = list(reversed(channels))
        for i in range(len(rev)):
            in_ch = rev[i]
            out_ch = rev[i + 1] if i + 1 < len(rev) else channels[0]
            layers += [
                nn.ConvTranspose1d(
                    in_ch, out_ch, kernel_size=3, stride=2, padding=1,
                    output_padding=1,
                ),
                nn.GELU(),
            ]
        layers += [nn.Conv1d(channels[0], cfg.feat_dim, kernel_size=3, padding=1)]
        self.deconv = nn.Sequential(*layers)
        self.seq_len = cfg.seq_len

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        h = self.fc(z).view(b, self.start_shape[0], self.start_shape[1])
        y = self.deconv(h)  # (B, F, T')
        # Trim or pad to match seq_len exactly.
        if y.shape[-1] > self.seq_len:
            y = y[..., : self.seq_len]
        elif y.shape[-1] < self.seq_len:
            pad = self.seq_len - y.shape[-1]
            y = F.pad(y, (0, pad))
        return y.transpose(1, 2)  # (B, T, F)


class TimeVAEv3(nn.Module):
    """Interpretable TimeVAE with a level + trend + seasonality + residual decoder."""

    def __init__(self, cfg: TimeVAEv3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = _ConvEncoder(cfg)
        self.level = _LevelHead(cfg)
        self.trend = _TrendHead(cfg)
        self.seas = _SeasonalityHead(cfg)
        if cfg.use_residual_conn:
            self.residual: nn.Module | None = _ResidualConvHead(cfg)
        else:
            self.residual = None
        self.output_activation = cfg.output_activation

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor,
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.level(z) + self.trend(z) + self.seas(z)
        if self.residual is not None:
            x = x + self.residual(z)
        if self.output_activation == "tanh":
            x = torch.tanh(x)
        elif self.output_activation == "sigmoid":
            x = torch.sigmoid(x)
        return x

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    @torch.no_grad()
    def generate(
        self,
        n: int,
        device: torch.device | None = None,
        *,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Sample ''n'' synthetic windows."""
        if n <= 0:
            raise ValueError(f"n must be positive; got {n}")
        device = device or next(self.parameters()).device
        if seed is None:
            z = torch.randn(n, self.cfg.latent_dim, device=device)
        else:
            gen_device = device if device.type in ("cuda", "cpu") else torch.device("cpu")
            g = torch.Generator(device=gen_device).manual_seed(int(seed))
            z = torch.randn(
                n, self.cfg.latent_dim, generator=g, device=gen_device,
            ).to(device)
        return self.decode(z)


def _kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Standard VAE KL against an isotropic Gaussian prior, mean over batch."""
    return -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )


def build_timevae_v3(
    seq_len: int,
    feat_dim: int,
    *,
    latent_dim: int = 32,
    hidden_channels: Sequence[int] = (64, 128, 256),
    trend_poly: int = 1,
    custom_seas: Sequence[tuple[int, int]] = (),
    use_residual_conn: bool = True,
    output_activation: OutputActivation = "linear",
    reconstruction_wt: float = 3.0,
    beta: float = 1.0,
    kl_anneal_epochs: int = 0,
    scaler_family: ScalerFamily | None = None,
) -> TimeVAEv3:
    """Construct a :class:`TimeVAEv3` from individual hyperparameters."""
    if scaler_family is not None:
        validate_scaler_activation(scaler_family, output_activation)

    cfg = TimeVAEv3Config(
        seq_len=int(seq_len),
        feat_dim=int(feat_dim),
        latent_dim=int(latent_dim),
        hidden_channels=list(hidden_channels),
        trend_poly=int(trend_poly),
        custom_seas=[tuple(p) for p in custom_seas],
        use_residual_conn=bool(use_residual_conn),
        output_activation=output_activation,
        reconstruction_wt=float(reconstruction_wt),
        beta=float(beta),
        kl_anneal_epochs=int(kl_anneal_epochs),
    )
    return TimeVAEv3(cfg)


def fit_timevae_v3(
    model: TimeVAEv3,
    x: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str | torch.device | None = None,
    verbose: int = 0,
) -> dict[str, list[float]]:
    """Train the model with Adam and return per-epoch loss history."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.to(device)
    cfg = model.cfg

    if x.ndim != 3:
        raise ValueError(f"Expected (N, L, F) windows, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("training data contains NaN or Inf - check preprocessing")

    tensor = torch.as_tensor(x, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    # Local generator for the DataLoader's shuffle order so determinism
    # does not depend on the global Torch RNG state at __iter__ time.
    loader_gen = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, max(1, tensor.shape[0])),
        shuffle=True,
        drop_last=False,
        generator=loader_gen,
    )
    opt = torch.optim.Adam(model.parameters(), lr=float(learning_rate))

    history: dict[str, list[float]] = {"loss": [], "recon": [], "kl": []}
    for epoch in range(int(epochs)):
        model.train()
        # Linear KL annealing factor in [0, 1].
        if cfg.kl_anneal_epochs <= 0:
            ramp = 1.0
        else:
            ramp = min(1.0, (epoch + 1) / float(cfg.kl_anneal_epochs))
        kl_weight = float(cfg.beta) * float(ramp)

        ep_loss = ep_recon = ep_kl = 0.0
        n_batches = 0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            x_hat, mu, logvar = model(batch)
            recon = F.mse_loss(x_hat, batch, reduction="mean")
            kl = _kl_divergence(mu, logvar)
            loss = cfg.reconstruction_wt * recon + kl_weight * kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            ep_loss += float(loss.item())
            ep_recon += float(recon.item())
            ep_kl += float(kl.item())
            n_batches += 1

        n_batches = max(1, n_batches)
        history["loss"].append(ep_loss / n_batches)
        history["recon"].append(ep_recon / n_batches)
        history["kl"].append(ep_kl / n_batches)
        if verbose:
            print(
                f"[TimeVAE_v3] epoch {epoch + 1}/{epochs} "
                f"loss={history['loss'][-1]:.4f} "
                f"recon={history['recon'][-1]:.4f} kl={history['kl'][-1]:.4f} "
                f"kl_weight={kl_weight:.3f}"
            )
    return history


def generate_numpy(
    model: TimeVAEv3,
    n: int,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Sample ''n'' windows from the decoder and return them as ''float32''."""
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")
    model.eval()
    device = next(model.parameters()).device
    out = model.generate(n, device=device, seed=seed)
    return out.detach().cpu().numpy().astype(np.float32)


def save_timevae_v3(model: TimeVAEv3, path: Path | str) -> None:
    """Save ''state_dict'' + sidecar ''<path>.meta.json'' with the config."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    meta = model.cfg.to_dict()
    with open(str(path) + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_timevae_v3(path: Path | str, map_location: str = "cpu") -> TimeVAEv3:
    """Reconstruct a model from the sidecar meta file and load weights."""
    path = Path(path)
    meta_path = Path(str(path) + ".meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    cfg = TimeVAEv3Config.from_dict(meta)
    model = TimeVAEv3(cfg)
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    return model


__all__ = [
    "OutputActivation",
    "TimeVAEv3",
    "TimeVAEv3Config",
    "_ConvEncoder",
    "_LevelHead",
    "_ResidualConvHead",
    "_SeasonalityHead",
    "_TrendHead",
    "build_timevae_v3",
    "fit_timevae_v3",
    "generate_numpy",
    "load_timevae_v3",
    "profile_to_scaler_family",
    "save_timevae_v3",
]
