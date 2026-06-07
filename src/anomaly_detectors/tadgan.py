"""TadGAN anomaly detector (Geiger et al., 2020)."""

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
    """Bidirectional LSTM encoder - latent vector."""

    def __init__(self, feat_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        return self.fc(h_last)


class _Decoder(nn.Module):
    """LSTM decoder: latent - reconstructed window."""

    def __init__(
        self, latent_dim: int, hidden_dim: int, seq_len: int, feat_dim: int
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(hidden_dim, feat_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc(z))
        h = h.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(h)
        return self.out(out)


class _CriticX(nn.Module):
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


class _CriticZ(nn.Module):
    """MLP critic on latent space."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


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


# ── TadGAN detector ─────────────────────────────────────────────────


class TadGANDetector(AnomalyDetector):
    """TadGAN-based window-level anomaly detector."""

    def __init__(
        self,
        latent_dim: int = 20,
        hidden_dim: int = 100,
        n_critic: int = 5,
        gp_lambda: float = 10.0,
        score_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_critic = n_critic
        self.gp_lambda = gp_lambda
        self.score_weight = score_weight

        self._encoder: _Encoder | None = None
        self._decoder: _Decoder | None = None
        self._critic_x: _CriticX | None = None
        self._critic_z: _CriticZ | None = None
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

        self._encoder = _Encoder(self._feat_dim, self.hidden_dim, self.latent_dim).to(device)
        self._decoder = _Decoder(self.latent_dim, self.hidden_dim, self._seq_len, self._feat_dim).to(device)
        self._critic_x = _CriticX(self._seq_len, self._feat_dim).to(device)
        self._critic_z = _CriticZ(self.latent_dim).to(device)

        opt_gen = torch.optim.Adam(
            list(self._encoder.parameters()) + list(self._decoder.parameters()),
            lr=lr, betas=(0.5, 0.999),
        )
        opt_cx = torch.optim.Adam(self._critic_x.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_cz = torch.optim.Adam(self._critic_z.parameters(), lr=lr, betas=(0.5, 0.999))

        data = torch.from_numpy(normal_windows).float()
        dataset = torch.utils.data.TensorDataset(data)
        g = torch.Generator()
        g.manual_seed(seed)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=g,
        )

        recent_losses: list[float] = []
        for epoch in range(epochs):
            epoch_g_loss = 0.0
            epoch_c_loss = 0.0
            n_batches = 0
            for (real_batch,) in loader:
                real_batch = real_batch.to(device)
                bs = real_batch.size(0)

                # ── Critic updates ──
                for _ in range(self.n_critic):
                    z_enc = self._encoder(real_batch)
                    x_rec = self._decoder(z_enc)
                    z_prior = torch.randn(bs, self.latent_dim, device=device)

                    # Critic_x
                    opt_cx.zero_grad()
                    cx_real = self._critic_x(real_batch)
                    cx_fake = self._critic_x(x_rec.detach())
                    gp_x = _gradient_penalty(self._critic_x, real_batch, x_rec.detach(), device)
                    loss_cx = cx_fake.mean() - cx_real.mean() + self.gp_lambda * gp_x
                    self._check_finite(loss_cx, "Critic_x", epoch, recent_losses)
                    loss_cx.backward()
                    opt_cx.step()

                    # Critic_z
                    opt_cz.zero_grad()
                    cz_real = self._critic_z(z_enc.detach())
                    cz_fake = self._critic_z(z_prior)
                    gp_z = _gradient_penalty(self._critic_z, z_enc.detach(), z_prior, device)
                    loss_cz = cz_real.mean() - cz_fake.mean() + self.gp_lambda * gp_z
                    self._check_finite(loss_cz, "Critic_z", epoch, recent_losses)
                    loss_cz.backward()
                    opt_cz.step()

                # ── Generator (encoder + decoder) update ──
                opt_gen.zero_grad()
                z_enc = self._encoder(real_batch)
                x_rec = self._decoder(z_enc)
                z_prior = torch.randn(bs, self.latent_dim, device=device)

                cx_fake = self._critic_x(x_rec)
                cz_enc = self._critic_z(z_enc)
                cz_prior = self._critic_z(z_prior)

                rec_loss = nn.functional.l1_loss(x_rec, real_batch)
                loss_g = -cx_fake.mean() + (cz_enc.mean() - cz_prior.mean()) + 10.0 * rec_loss
                self._check_finite(loss_g, "Generator", epoch, recent_losses)
                loss_g.backward()
                opt_gen.step()

                epoch_g_loss += loss_g.item()
                epoch_c_loss += loss_cx.item()
                n_batches += 1
                recent_losses.append(loss_g.item())
                if len(recent_losses) > 20:
                    recent_losses.pop(0)

            if n_batches > 0:
                logger.info(
                    "TadGAN epoch %d/%d  G=%.4f  Cx=%.4f",
                    epoch + 1, epochs,
                    epoch_g_loss / n_batches,
                    epoch_c_loss / n_batches,
                )

        self._encoder.eval()
        self._decoder.eval()
        self._critic_x.eval()
        self._critic_z.eval()

        return {
            "detector": "TadGAN",
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "epochs": epochs,
            "n_train_windows": normal_windows.shape[0],
        }

    def _score_windows(
        self, windows: np.ndarray, *, device: torch.device, batch_size: int
    ) -> np.ndarray:
        assert self._encoder is not None and self._decoder is not None
        assert self._critic_x is not None

        self._encoder.to(device).eval()
        self._decoder.to(device).eval()
        self._critic_x.to(device).eval()

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
                cx = self._critic_x(batch)
                critic_scores.append(cx.cpu().numpy())

        rec_all = np.concatenate(rec_errors).astype(np.float64)
        cx_all = np.concatenate(critic_scores).astype(np.float64)

        rec_norm = self._minmax_norm(rec_all)
        cx_norm = self._minmax_norm(-cx_all)

        scores = self.score_weight * rec_norm + (1 - self.score_weight) * cx_norm
        return scores

    def _state_dict(self) -> dict[str, Any]:
        return {
            "encoder": self._encoder.state_dict() if self._encoder else {},
            "decoder": self._decoder.state_dict() if self._decoder else {},
            "critic_x": self._critic_x.state_dict() if self._critic_x else {},
            "critic_z": self._critic_z.state_dict() if self._critic_z else {},
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "n_critic": self.n_critic,
            "gp_lambda": self.gp_lambda,
            "score_weight": self.score_weight,
            "seq_len": self._seq_len,
            "feat_dim": self._feat_dim,
        }

    def _load_state_dict(self, state: dict[str, Any]) -> None:
        self.latent_dim = state["latent_dim"]
        self.hidden_dim = state["hidden_dim"]
        self.n_critic = state["n_critic"]
        self.gp_lambda = state["gp_lambda"]
        self.score_weight = state["score_weight"]
        self._seq_len = state["seq_len"]
        self._feat_dim = state["feat_dim"]

        self._encoder = _Encoder(self._feat_dim, self.hidden_dim, self.latent_dim)
        self._decoder = _Decoder(self.latent_dim, self.hidden_dim, self._seq_len, self._feat_dim)
        self._critic_x = _CriticX(self._seq_len, self._feat_dim)
        self._critic_z = _CriticZ(self.latent_dim)

        self._encoder.load_state_dict(state["encoder"])
        self._decoder.load_state_dict(state["decoder"])
        self._critic_x.load_state_dict(state["critic_x"])
        self._critic_z.load_state_dict(state["critic_z"])

        self._encoder.eval()
        self._decoder.eval()
        self._critic_x.eval()
        self._critic_z.eval()

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
