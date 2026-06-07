# Adapted from https://github.com/acphile/RTSGAN (MIT License)

from __future__ import annotations


import torch
import torch.nn as nn


class Encoder(nn.Module):
    """GRU encoder that maps "(batch, seq_len, feat_dim)" to a latent vector."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.rnn = nn.GRU(
            input_dim, hidden_dim, layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim * 3, hidden_dim)
        self.final = nn.LeakyReLU(0.2)

    def forward(self, dynamics: torch.Tensor) -> torch.Tensor:
        """Encode "(B, L, F)" -> "(B, hidden_dim + hidden_dim*layers)"."""
        bs = dynamics.size(0)
        out, h = self.rnn(dynamics)
        # h: (layers, B, hidden_dim)

        h1 = out.max(dim=1).values  # max-pool over time
        h2 = out.mean(dim=1)  # mean-pool over time
        h3_last = h[-1]  # last layer's final hidden state

        glob = self.final(self.fc(torch.cat([h1, h2, h3_last], dim=-1)))

        h3 = h.permute(1, 0, 2).contiguous().view(bs, -1)
        return torch.cat([glob, h3], dim=-1)


class Decoder(nn.Module):
    """GRU decoder that reconstructs "(B, L, F)" from a latent vector."""

    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float = 0.0,
        output_sigmoid: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim
        self.layers = layers
        self.output_sigmoid = output_sigmoid
        self.rnn = nn.GRU(
            hidden_dim + feat_dim, hidden_dim, layers,
            batch_first=True, dropout=dropout,
        )
        self.dynamics_fc = nn.Linear(hidden_dim, feat_dim)

    def _apply_output(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_sigmoid:
            return torch.sigmoid(x)
        return x

    def forward(
        self,
        embed: torch.Tensor,
        dynamics: torch.Tensor,
        forcing: float = 0.5,
    ) -> torch.Tensor:
        """Teacher-forced reconstruction.

        Args:
            embed: Latent vector from encoder "(B, embed_total)".
            dynamics: Ground-truth input "(B, L, F)".
            forcing: Teacher-forcing ratio.
        """
        glob = embed[:, : self.hidden_dim].unsqueeze(1)
        hidden = embed[:, self.hidden_dim :]
        bs, max_len, _ = dynamics.size()
        hidden = hidden.view(bs, self.layers, -1).permute(1, 0, 2).contiguous()

        x = dynamics[:, 0:1, :]
        res = []
        for i in range(max_len):
            x_in = torch.cat([glob, x.detach()], dim=-1)
            out, hidden = self.rnn(x_in, hidden)
            out = self._apply_output(self.dynamics_fc(out.squeeze(1))).unsqueeze(1)
            if torch.rand(1).item() > forcing:
                x = out
            else:
                x = dynamics[:, i + 1 : i + 2, :]
                if x.size(1) == 0:
                    x = out
            res.append(out)
        return torch.cat(res, dim=1)

    def generate_dynamics(self, embed: torch.Tensor, max_len: int) -> torch.Tensor:
        """Autoregressively generate "(B, max_len, feat_dim)"."""
        glob = embed[:, : self.hidden_dim].unsqueeze(1)
        hidden = embed[:, self.hidden_dim :]
        bs = glob.size(0)
        hidden = hidden.view(bs, self.layers, -1).permute(1, 0, 2).contiguous()

        x = torch.zeros((bs, 1, self.feat_dim), device=embed.device)
        res = []
        for _ in range(max_len):
            x_in = torch.cat([glob, x], dim=-1)
            out, hidden = self.rnn(x_in, hidden)
            out = self._apply_output(self.dynamics_fc(out.squeeze(1))).detach()
            x = out.unsqueeze(1)
            res.append(x)
        return torch.cat(res, dim=1)


class Autoencoder(nn.Module):
    """Encoder-Decoder pair for fixed-length time series."""

    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float = 0.0,
        output_sigmoid: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(feat_dim, hidden_dim, layers, dropout)
        self.decoder = Decoder(
            feat_dim, hidden_dim, layers, dropout, output_sigmoid
        )

    def forward(
        self, dynamics: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.encoder(dynamics)
        shifted = torch.zeros_like(dynamics[:, 0:1, :])
        shifted = torch.cat([shifted, dynamics[:, :-1, :]], dim=1)
        return self.decoder(hidden, shifted)
