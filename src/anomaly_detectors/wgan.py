"""WGAN-GP anomaly detector."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import nn

from src.anomaly_detectors.base import AnomalyDetector
from src.utils.seeds import set_seed

logger = logging.getLogger(__name__)


# ── Network components ────────────────────────────────────────────────


class _Encoder(nn.Module):
    """1-D CNN encoder - latent vector."""

    def __init__(self, seq_len: int, feat_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )
        self.fc = nn.Linear(128 * seq_len, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.transpose(1, 2))
        return self.fc(h)


class _Decoder(nn.Module):
    """MLP + reshape decoder: latent - reconstructed window."""

    def __init__(self, latent_dim: int, seq_len: int, feat_dim: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, seq_len * feat_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        return out.view(-1, self.seq_len, self.feat_dim)


class _Critic(nn.Module):
    """1-D CNN critic on signal space ``(T, F)``."""

    def __init__(self, seq_len: int, feat_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(feat_dim, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(64 * seq_len, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).squeeze(-1)


# ── Gradient penalty ─────────────────────────────────────────────────


def _gradient_penalty(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    alpha = torch.rand(real.size(0), *([1] * (real.ndim - 1)), device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    scores = critic(interpolated)
    grad = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    grad = grad.reshape(grad.size(0), -1)
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


# ── WGAN-GP detector ─────────────────────────────────────────────────


class WGANDetector(AnomalyDetector):
    """WGAN-GP-based window-level anomaly detector."""

    def __init__(
        self,
        latent_dim: int = 32,
        n_critic: int = 5,
        gp_lambda: float = 10.0,
        score_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.gp_lambda = gp_lambda
        self.score_weight = score_weight

        self._encoder: _Encoder | None = None
        self._decoder: _Decoder | None = None
        self._critic: _Critic | None = None
        self._seq_len: int = 0
        self._feat_dim: int = 0

    # ── ABC implementation ────────────────────────────────────────

    def _fit_model(
        self,
        normal_windows: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        lr: float,
        device: torch.device,
        seed: int,
    ) -> dict[str, Any]:
        set_seed(seed)
        n, self._seq_len, self._feat_dim = normal_windows.shape

        self._encoder = _Encoder(self._seq_len, self._feat_dim, self.latent_dim).to(device)
        self._decoder = _Decoder(self.latent_dim, self._seq_len, self._feat_dim).to(device)
        self._critic = _Critic(self._seq_len, self._feat_dim).to(device)

        opt_ae = torch.optim.Adam(
            list(self._encoder.parameters()) + list(self._decoder.parameters()),
            lr=lr, betas=(0.5, 0.999),
        )
        opt_c = torch.optim.Adam(self._critic.parameters(), lr=lr, betas=(0.5, 0.999))

        data = torch.from_numpy(normal_windows).float()
        dataset = torch.utils.data.TensorDataset(data)
        g = torch.Generator()
        g.manual_seed(seed)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=g,
        )

        recent_losses: list[float] = []
        for epoch in range(epochs):
            epoch_ae_loss = 0.0
            epoch_c_loss = 0.0
            n_batches = 0
            for (real_batch,) in loader:
                real_batch = real_batch.to(device)

                # ── Critic updates ──
                for _ in range(self.n_critic):
                    opt_c.zero_grad()
                    z = self._encoder(real_batch)
                    x_rec = self._decoder(z)

                    c_real = self._critic(real_batch)
                    c_fake = self._critic(x_rec.detach())
                    gp = _gradient_penalty(self._critic, real_batch, x_rec.detach(), device)
                    loss_c = c_fake.mean() - c_real.mean() + self.gp_lambda * gp
                    self._check_finite(loss_c, "Critic", epoch, recent_losses)
                    loss_c.backward()
                    opt_c.step()

                # ── Autoencoder update ──
                opt_ae.zero_grad()
                z = self._encoder(real_batch)
                x_rec = self._decoder(z)
                c_fake = self._critic(x_rec)
                rec_loss = nn.functional.l1_loss(x_rec, real_batch)
                loss_ae = -c_fake.mean() + 10.0 * rec_loss
                self._check_finite(loss_ae, "AE", epoch, recent_losses)
                loss_ae.backward()
                opt_ae.step()

                epoch_ae_loss += loss_ae.item()
                epoch_c_loss += loss_c.item()
                n_batches += 1
                recent_losses.append(loss_ae.item())
                if len(recent_losses) > 20:
                    recent_losses.pop(0)

            if n_batches > 0:
                logger.info(
                    "WGAN epoch %d/%d  AE=%.4f  C=%.4f",
                    epoch + 1, epochs,
                    epoch_ae_loss / n_batches,
                    epoch_c_loss / n_batches,
                )

        self._encoder.eval()
        self._decoder.eval()
        self._critic.eval()

        return {
            "detector": "WGAN",
            "latent_dim": self.latent_dim,
            "epochs": epochs,
            "n_train_windows": normal_windows.shape[0],
        }

    def _score_windows(
        self, windows: np.ndarray, *, device: torch.device, batch_size: int
    ) -> np.ndarray:
        assert self._encoder is not None and self._decoder is not None
        assert self._critic is not None

        self._encoder.to(device).eval()
        self._decoder.to(device).eval()
        self._critic.to(device).eval()

        rec_errors: list[np.ndarray] = []
        critic_scores: list[np.ndarray] = []

        data = torch.from_numpy(windows).float()
        with torch.no_grad():
            for start in range(0, len(data), batch_size):
                batch = data[start : start + batch_size].to(device)
                z = self._encoder(batch)
                x_rec = self._decoder(z)
                rec_err = (batch - x_rec).abs().mean(dim=(1, 2))
                rec_errors.append(rec_err.cpu().numpy())
                c = self._critic(batch)
                critic_scores.append(c.cpu().numpy())

        rec_all = np.concatenate(rec_errors).astype(np.float64)
        c_all = np.concatenate(critic_scores).astype(np.float64)

        rec_norm = self._minmax_norm(rec_all)
        c_norm = self._minmax_norm(-c_all)

        return self.score_weight * rec_norm + (1 - self.score_weight) * c_norm

    def _state_dict(self) -> dict[str, Any]:
        return {
            "encoder": self._encoder.state_dict() if self._encoder else {},
            "decoder": self._decoder.state_dict() if self._decoder else {},
            "critic": self._critic.state_dict() if self._critic else {},
            "latent_dim": self.latent_dim,
            "n_critic": self.n_critic,
            "gp_lambda": self.gp_lambda,
            "score_weight": self.score_weight,
            "seq_len": self._seq_len,
            "feat_dim": self._feat_dim,
        }

    def _load_state_dict(self, state: dict[str, Any]) -> None:
        self.latent_dim = state["latent_dim"]
        self.n_critic = state["n_critic"]
        self.gp_lambda = state["gp_lambda"]
        self.score_weight = state["score_weight"]
        self._seq_len = state["seq_len"]
        self._feat_dim = state["feat_dim"]

        self._encoder = _Encoder(self._seq_len, self._feat_dim, self.latent_dim)
        self._decoder = _Decoder(self.latent_dim, self._seq_len, self._feat_dim)
        self._critic = _Critic(self._seq_len, self._feat_dim)

        self._encoder.load_state_dict(state["encoder"])
        self._decoder.load_state_dict(state["decoder"])
        self._critic.load_state_dict(state["critic"])

        self._encoder.eval()
        self._decoder.eval()
        self._critic.eval()

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _minmax_norm(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-12:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    @staticmethod
    def _check_finite(
        loss: torch.Tensor, name: str, epoch: int, recent: list[float]
    ) -> None:
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite {name} loss at epoch {epoch}: "
                f"loss={loss.item()}, recent={recent[-5:]}"
            )
