# Adapted from https://github.com/acphile/RTSGAN (MIT License)
# 
# Simplified for fixed-length, continuous, pre-scaled time-series windows.
# Dependencies removed: fastNLP, Processor.

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import autograd
from torch.utils.data import DataLoader, TensorDataset

from src.models.rtsgan.autoencoder import Autoencoder
from src.models.rtsgan.gan import Discriminator, Generator


def _toggle_grad(model: nn.Module, requires_grad: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def _compute_grad2(d_out: torch.Tensor, x_in: torch.Tensor) -> torch.Tensor:
    batch_size = x_in.size(0)
    (grad_dout,) = autograd.grad(
        outputs=d_out.sum(),
        inputs=x_in,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    return grad_dout.pow(2).view(batch_size, -1).sum(1)


class AeGAN:
    """RTSGAN two-phase model: autoencoder + WGAN-GP in latent space."""

    def __init__(
        self,
        feat_dim: int,
        seq_len: int,
        hidden_dim: int = 24,
        embed_dim: int = 96,
        noise_dim: int = 96,
        layers: int = 3,
        dropout: float = 0.0,
        output_sigmoid: bool = True,
        device: torch.device | None = None,
    ) -> None:
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.noise_dim = noise_dim
        self.layers = layers
        self.dropout = dropout
        self.device = device or torch.device("cpu")

        self.ae = Autoencoder(
            feat_dim, hidden_dim, layers, dropout, output_sigmoid
        ).to(self.device)

        latent_total = hidden_dim + hidden_dim * layers
        self.generator = Generator(noise_dim, hidden_dim, layers).to(self.device)
        self.discriminator = Discriminator(latent_total).to(self.device)

    # ------------------------------------------------------------------
    # Autoencoder training
    # ------------------------------------------------------------------

    def train_ae(
        self,
        x: torch.Tensor,
        epochs: int = 200,
        batch_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> list[float]:
        """Train the GRU autoencoder with MSE reconstruction loss."""
        optimizer = torch.optim.Adam(
            self.ae.parameters(), lr=lr, weight_decay=weight_decay,
        )
        loss_fn = nn.MSELoss()
        loader = DataLoader(
            TensorDataset(x),
            batch_size=min(batch_size, x.size(0)),
            shuffle=True,
            drop_last=False,
        )
        losses: list[float] = []
        self.ae.train()
        for _ in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                recon = self.ae(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            losses.append(epoch_loss / max(n_batches, 1))
        return losses

    # ------------------------------------------------------------------
    # WGAN-GP training
    # ------------------------------------------------------------------

    def _wgan_gp_penalty(
        self, real: torch.Tensor, fake: torch.Tensor, center: float = 1.0,
    ) -> torch.Tensor:
        bs = real.size(0)
        eps = torch.rand(bs, 1, device=self.device)
        interp = ((1 - eps) * real + eps * fake).detach().requires_grad_(True)
        d_out = self.discriminator(interp)
        return (_compute_grad2(d_out, interp).sqrt() - center).pow(2).mean()

    def train_gan(
        self,
        x: torch.Tensor,
        iterations: int = 5000,
        batch_size: int = 256,
        d_update: int = 5,
        lr: float = 1e-4,
        alpha: float = 0.99,
        gp_weight: float = 10.0,
    ) -> list[float]:
        """Train WGAN-GP in the latent space of the frozen encoder.

        Returns:
            Generator loss at each iteration.
        """
        d_optim = torch.optim.RMSprop(
            self.discriminator.parameters(), lr=lr, alpha=alpha,
        )
        g_optim = torch.optim.RMSprop(
            self.generator.parameters(), lr=lr, alpha=alpha,
        )
        loader = DataLoader(
            TensorDataset(x),
            batch_size=min(batch_size, x.size(0)),
            shuffle=True,
            drop_last=False,
        )
        loader_iter = iter(loader)

        self.ae.eval()
        self.generator.train()
        self.discriminator.train()

        g_losses: list[float] = []
        for iteration in range(iterations):
            # --- Discriminator updates ---
            _toggle_grad(self.generator, False)
            _toggle_grad(self.discriminator, True)
            for _ in range(d_update):
                try:
                    (batch,) = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader)
                    (batch,) = next(loader_iter)
                batch = batch.to(self.device)
                d_optim.zero_grad()

                with torch.no_grad():
                    real_rep = self.ae.encoder(batch)
                d_real = self.discriminator(real_rep)
                dloss_real = -d_real.mean()
                dloss_real.backward()

                z = torch.randn(batch.size(0), self.noise_dim, device=self.device)
                with torch.no_grad():
                    fake_rep = self.generator(z)
                fake_rep.requires_grad_(True)
                d_fake = self.discriminator(fake_rep)
                dloss_fake = d_fake.mean()
                dloss_fake.backward()

                gp = gp_weight * self._wgan_gp_penalty(real_rep.detach(), fake_rep.detach())
                gp.backward()
                d_optim.step()

            # --- Generator update ---
            _toggle_grad(self.generator, True)
            _toggle_grad(self.discriminator, False)
            g_optim.zero_grad()
            z = torch.randn(batch_size, self.noise_dim, device=self.device)
            fake = self.generator(z)
            g_loss = -self.discriminator(fake).mean()
            g_loss.backward()
            g_optim.step()

            g_losses.append(g_loss.item())
        return g_losses

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def synthesize(self, n: int, batch_size: int = 512) -> torch.Tensor:
        """Generate "n" synthetic windows "(n, seq_len, feat_dim)"."""
        self.ae.decoder.eval()
        self.generator.eval()
        chunks: list[torch.Tensor] = []
        remaining = n
        while remaining > 0:
            bs = min(batch_size, remaining)
            z = torch.randn(bs, self.noise_dim, device=self.device)
            hidden = self.generator(z)
            dyn = self.ae.decoder.generate_dynamics(hidden, self.seq_len)
            chunks.append(dyn.cpu())
            remaining -= bs
        return torch.cat(chunks, dim=0)[:n]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save all component state dicts to a single ".pt" file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "ae": self.ae.state_dict(),
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "meta": {
                    "feat_dim": self.feat_dim,
                    "seq_len": self.seq_len,
                    "hidden_dim": self.hidden_dim,
                    "noise_dim": self.noise_dim,
                    "layers": self.layers,
                    "dropout": self.dropout,
                },
            },
            str(path),
        )

    def load(self, path: str | Path) -> None:
        """Load component state dicts from a ".pt" file."""
        ckpt: dict[str, Any] = torch.load(str(path), map_location=self.device, weights_only=False)
        self.ae.load_state_dict(ckpt["ae"])
        self.generator.load_state_dict(ckpt["generator"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
