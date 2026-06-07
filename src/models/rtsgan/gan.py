# Adapted from https://github.com/acphile/RTSGAN (MIT License)

from __future__ import annotations

import torch
import torch.nn as nn


class Generator(nn.Module):
    """MLP generator that maps noise to the AE latent space.

    Output dim = "hidden_dim + hidden_dim * layers" to match the
    encoder's concatenated "[glob, h3]" representation.
    """

    def __init__(self, input_dim: int, hidden_dim: int, layers: int) -> None:
        super().__init__()

        def block(inp: int, out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(inp, out),
                nn.LayerNorm(out),
                nn.LeakyReLU(0.2),
            )

        self.block_0 = block(input_dim, input_dim)
        self.block_1 = block(input_dim, input_dim)
        self.block_2 = block(input_dim, hidden_dim)
        self.block_2_1 = block(hidden_dim, hidden_dim)
        self.block_3 = block(input_dim, hidden_dim * layers)
        self.block_3_1 = nn.Linear(hidden_dim * layers, hidden_dim * layers)
        self.final = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_0(x) + x
        x = self.block_1(x) + x
        x1 = self.block_2_1(self.block_2(x))
        x2 = self.block_3_1(self.block_3(x))
        return torch.cat([x1, x2], dim=-1)


class Discriminator(nn.Module):
    """Simple MLP critic for WGAN-GP."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, (2 * input_dim) // 3),
            nn.LeakyReLU(0.2),
            nn.Linear((2 * input_dim) // 3, input_dim // 3),
            nn.LeakyReLU(0.2),
            nn.Linear(input_dim // 3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
