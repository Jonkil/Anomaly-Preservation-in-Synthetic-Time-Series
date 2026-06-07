# Adapted from https://github.com/imics-lab/tts-gan (MIT License).

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.ttsgan.transformer import Discriminator, Generator, pick_patch_size


class TTSGAN:
    """Wrapper around the vendored TTS-GAN generator/discriminator."""

    def __init__(
        self,
        feat_dim: int,
        seq_len: int,
        latent_dim: int = 100,
        embed_dim: int = 16,
        depth: int = 3,
        num_heads: int = 4,
        patch_size: int | None = None,
        dropout: float = 0.1,
        output_sigmoid: bool = False,
        device: torch.device | None = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if patch_size is None:
            patch_size = pick_patch_size(seq_len)

        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.dropout = dropout
        self.output_sigmoid = output_sigmoid
        self.device = device

        self.generator = Generator(
            seq_len=seq_len,
            channels=feat_dim,
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            attn_drop_rate=dropout,
            forward_drop_rate=dropout,
        ).to(device)
        self.discriminator = Discriminator(
            in_channels=feat_dim,
            patch_size=patch_size,
            emb_size=embed_dim,
            seq_length=seq_len,
            depth=depth,
            num_heads=num_heads,
            drop_p=dropout,
            forward_drop_p=dropout,
        ).to(device)

    def train(
        self,
        x: torch.Tensor,
        iterations: int,
        batch_size: int = 64,
        lr_g: float = 1e-4,
        lr_d: float = 3e-4,
        d_update: int = 3,
        beta1: float = 0.0,
        beta2: float = 0.9,
    ) -> None:
        """Train with an LSGAN objective."""
        if x.dim() != 4 or x.shape[1] != self.feat_dim or x.shape[3] != self.seq_len:
            raise ValueError(
                f"x must have shape (N, {self.feat_dim}, 1, {self.seq_len}); got {tuple(x.shape)}"
            )

        bs = max(1, min(batch_size, x.shape[0]))
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)
        if len(loader) == 0:
            loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=False)

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(beta1, beta2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))
        mse = nn.MSELoss()

        self.generator.train()
        self.discriminator.train()
        loader_iter = iter(loader)

        def _next_batch() -> torch.Tensor:
            nonlocal loader_iter
            try:
                (batch,) = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                (batch,) = next(loader_iter)
            return batch.to(self.device, non_blocking=True)

        for _ in range(iterations):
            for _ in range(max(1, d_update)):
                real = _next_batch()
                cur_bs = real.shape[0]
                z = torch.randn(cur_bs, self.latent_dim, device=self.device)
                with torch.no_grad():
                    fake = self.generator(z)
                    if self.output_sigmoid:
                        fake = torch.sigmoid(fake)
                real_score = self.discriminator(real)
                fake_score = self.discriminator(fake)
                real_label = torch.ones_like(real_score)
                fake_label = torch.zeros_like(fake_score)
                d_loss = mse(real_score, real_label) + mse(fake_score, fake_label)
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), 5.0)
                opt_d.step()

            z = torch.randn(bs, self.latent_dim, device=self.device)
            fake = self.generator(z)
            if self.output_sigmoid:
                fake = torch.sigmoid(fake)
            fake_score = self.discriminator(fake)
            real_label = torch.ones_like(fake_score)
            g_loss = mse(fake_score, real_label)
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            nn.utils.clip_grad_norm_(self.generator.parameters(), 5.0)
            opt_g.step()

    @torch.no_grad()
    def generate(self, n: int, batch_size: int = 256) -> torch.Tensor:
        """Sample ''n'' synthetic windows as a CPU tensor of shape ''(n, C, 1, L)''."""
        self.generator.eval()
        chunks: list[torch.Tensor] = []
        remaining = n
        while remaining > 0:
            bs = min(batch_size, remaining)
            z = torch.randn(bs, self.latent_dim, device=self.device)
            out = self.generator(z)
            if self.output_sigmoid:
                out = torch.sigmoid(out)
            chunks.append(out.detach().cpu())
            remaining -= bs
        return torch.cat(chunks, dim=0)[:n]

    def save(self, path: str | Path) -> None:
        """Save generator + discriminator state dicts and architectural metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "config": {
                "feat_dim": self.feat_dim,
                "seq_len": self.seq_len,
                "latent_dim": self.latent_dim,
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "patch_size": self.patch_size,
                "dropout": self.dropout,
                "output_sigmoid": self.output_sigmoid,
            },
        }
        torch.save(payload, str(path))

    def load(self, path: str | Path) -> None:
        """Load weights previously written by :func:`save`."""
        payload = torch.load(str(path), map_location=self.device, weights_only=False)
        self.generator.load_state_dict(payload["generator"])
        self.discriminator.load_state_dict(payload["discriminator"])
