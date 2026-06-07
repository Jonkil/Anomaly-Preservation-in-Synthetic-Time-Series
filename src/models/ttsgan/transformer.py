# Adapted from https://github.com/imics-lab/tts-gan (MIT License).

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualAdd(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fn(x)


class _MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if emb_size % num_heads != 0:
            raise ValueError(
                f"emb_size ({emb_size}) must be divisible by num_heads ({num_heads})"
            )
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.queries = nn.Linear(emb_size, emb_size)
        self.keys = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.queries(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.keys(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.values(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scaling = self.emb_size ** 0.5
        att = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.matmul(att, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.emb_size)
        return self.projection(out)


class _FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int, drop_p: float) -> None:
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class _TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        emb_size: int,
        num_heads: int,
        drop_p: float,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.0,
    ) -> None:
        super().__init__(
            _ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                _MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p),
            )),
            _ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                _FeedForwardBlock(emb_size, forward_expansion, forward_drop_p),
                nn.Dropout(drop_p),
            )),
        )


class _TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int, **kwargs: object) -> None:
        super().__init__(*[_TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class Generator(nn.Module):
    """Transformer generator producing ''(B, C, 1, L)'' outputs.

    Args:
        seq_len: Output window length ''L''.
        channels: Number of output channels ''C'' (= feat_dim).
        latent_dim: Dimensionality of the input noise vector.
        embed_dim: Transformer embedding size.
        depth: Number of encoder blocks.
        num_heads: Multi-head attention heads (must divide ''embed_dim'').
        attn_drop_rate: Dropout in the attention block.
        forward_drop_rate: Dropout in the feed-forward block.
    """

    def __init__(
        self,
        seq_len: int,
        channels: int,
        latent_dim: int = 100,
        embed_dim: int = 10,
        depth: int = 3,
        num_heads: int = 5,
        attn_drop_rate: float = 0.5,
        forward_drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        self.l1 = nn.Linear(latent_dim, seq_len * embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.blocks = _TransformerEncoder(
            depth=depth,
            emb_size=embed_dim,
            num_heads=num_heads,
            drop_p=attn_drop_rate,
            forward_drop_p=forward_drop_rate,
        )
        self.deconv = nn.Conv2d(embed_dim, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.l1(z).view(-1, self.seq_len, self.embed_dim)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        out = self.deconv(x.permute(0, 3, 1, 2))
        return out.view(-1, self.channels, 1, self.seq_len)


class _PatchEmbedding(nn.Module):
    """1-D patch embedding over the time axis (height stays 1)."""

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        emb_size: int,
        seq_length: int,
    ) -> None:
        super().__init__()
        if seq_length % patch_size != 0:
            raise ValueError(
                f"seq_length ({seq_length}) must be divisible by patch_size ({patch_size})"
            )
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.emb_size = emb_size
        self.num_patches = seq_length // patch_size

        self.projection = nn.Linear(patch_size * in_channels, emb_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn(self.num_patches + 1, emb_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        # x: (B, C, 1, L) -> (B, num_patches, patch_size * C)
        x = x.squeeze(2)                            # (B, C, L)
        x = x.transpose(1, 2)                       # (B, L, C)
        x = x.reshape(batch, self.num_patches, self.patch_size * self.in_channels)
        x = self.projection(x)                      # (B, num_patches, emb_size)
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        return x + self.positions


class _ClassificationHead(nn.Module):
    def __init__(self, emb_size: int, n_classes: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.fc = nn.Linear(emb_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=1)
        return self.fc(self.norm(x))


class Discriminator(nn.Sequential):
    """Transformer discriminator scoring ''(B, C, 1, L)'' inputs.

    Args:
        in_channels: Number of input channels (= feat_dim).
        patch_size: Size of each non-overlapping patch along the time axis.
            ''seq_length'' must be divisible by ''patch_size''.
        emb_size: Patch embedding size used inside the encoder.
        seq_length: Input window length ''L''.
        depth: Number of encoder blocks.
        num_heads: Multi-head attention heads (must divide ''emb_size'').
        n_classes: Output dimension (1 for adversarial scoring).
        drop_p: Dropout in the encoder blocks.
        forward_drop_p: Dropout in the feed-forward sub-layers.
    """

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        emb_size: int,
        seq_length: int,
        depth: int = 3,
        num_heads: int = 5,
        n_classes: int = 1,
        drop_p: float = 0.5,
        forward_drop_p: float = 0.5,
    ) -> None:
        super().__init__(
            _PatchEmbedding(in_channels, patch_size, emb_size, seq_length),
            _TransformerEncoder(
                depth=depth,
                emb_size=emb_size,
                num_heads=num_heads,
                drop_p=drop_p,
                forward_drop_p=forward_drop_p,
            ),
            _ClassificationHead(emb_size, n_classes),
        )


def pick_patch_size(seq_len: int, target_max: int = 16) -> int:
    """Choose a patch size that divides ''seq_len'' and is at most ''target_max''.

    Falls back to 1 if no suitable divisor exists. For the project's standard
    window sizes (powers of two from 64 to 1024) this returns 16.
    """
    for candidate in range(min(seq_len, target_max), 0, -1):
        if seq_len % candidate == 0:
            return candidate
    return 1
