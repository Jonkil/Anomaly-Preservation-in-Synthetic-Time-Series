"""Plotting helpers for the anomaly-preservation paper."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


def window_to_tsne_vector(x: np.ndarray) -> np.ndarray:
    """Reduce a batch of windows to one ``(T,)`` vector per window."""
    arr = np.asarray(x)
    if arr.ndim != 3:
        raise ValueError(
            f"window_to_tsne_vector expected (N, T, F), got shape {arr.shape}"
        )
    arr64 = arr.astype(np.float64, copy=False)
    finite_mask = np.isfinite(arr64).reshape(arr64.shape[0], -1).all(axis=1)
    if not finite_mask.any():
        raise ValueError(
            "window_to_tsne_vector: every window contains non-finite values"
        )
    arr_clean = arr64[finite_mask]
    return arr_clean.mean(axis=2, dtype=np.float64)


def _balanced_perplexity(n_total: int, requested: int) -> int:
    """Clamp the t-SNE perplexity to a value sklearn will accept."""
    if n_total < 2:
        raise ValueError(f"t-SNE needs at least 2 samples, got {n_total}")
    upper = max(2, (n_total - 1) // 3)
    return int(max(5, min(requested, upper)))


def compute_real_vs_synthetic_tsne(
    real: np.ndarray,
    synthetic: np.ndarray,
    *,
    n_per_class: int,
    perplexity: int,
    seed: int,
) -> dict[str, Any] | None:
    """Run t-SNE jointly on real + synthetic windows."""
    real_v = np.asarray(real, dtype=np.float64)
    syn_v = np.asarray(synthetic, dtype=np.float64)
    if real_v.ndim != 2 or syn_v.ndim != 2:
        raise ValueError(
            f"compute_real_vs_synthetic_tsne expected 2-D inputs, got "
            f"real={real_v.shape} syn={syn_v.shape}"
        )
    if real_v.shape[1] != syn_v.shape[1]:
        raise ValueError(
            f"time-axis mismatch: real T={real_v.shape[1]}, "
            f"syn T={syn_v.shape[1]}"
        )

    n_real_available = int(real_v.shape[0])
    n_syn_available = int(syn_v.shape[0])
    if n_real_available < 30 or n_syn_available < 30:
        return None

    rng = np.random.default_rng(int(seed))
    take_real = min(n_per_class, n_real_available)
    take_syn = min(n_per_class, n_syn_available)
    idx_real = rng.choice(n_real_available, size=take_real, replace=False)
    idx_syn = rng.choice(n_syn_available, size=take_syn, replace=False)
    real_sub = real_v[idx_real]
    syn_sub = syn_v[idx_syn]

    x = np.concatenate([real_sub, syn_sub], axis=0)
    n_total = x.shape[0]
    perplexity_used = _balanced_perplexity(n_total, perplexity)

    # Local import: sklearn is a heavy dep and we keep the visualisation
    # module importable in environments where sklearn is missing (e.g.,
    # minimal CI smoke tests for window_to_tsne_vector).
    from sklearn.manifold import TSNE

    import inspect
    _tsne_params = inspect.signature(TSNE).parameters
    iter_kwarg = "max_iter" if "max_iter" in _tsne_params else "n_iter"
    embedder = TSNE(
        n_components=2,
        perplexity=perplexity_used,
        init="pca",
        learning_rate="auto",
        random_state=int(seed),
        **{iter_kwarg: 1000},
    )
    coords = embedder.fit_transform(x)
    real_2d = coords[:take_real]
    syn_2d = coords[take_real:]
    return {
        "real_2d": np.asarray(real_2d, dtype=np.float32),
        "syn_2d": np.asarray(syn_2d, dtype=np.float32),
        "n_real": int(take_real),
        "n_syn": int(take_syn),
        "perplexity_used": int(perplexity_used),
        "seed": int(seed),
    }


def _atomic_savefig(fig, path: Path, *, dpi: int) -> None:
    """Save matplotlib figure atomically (tmp + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = path.suffix.lstrip(".")
    tmp = path.with_suffix(path.suffix + ".tmp")
    fig.savefig(tmp, dpi=dpi, bbox_inches="tight", format=fmt)
    os.replace(tmp, path)


def plot_tsne_grid(
    cells: dict[tuple[str, str], dict[str, Any] | None],
    datasets: list[str],
    models: list[str],
    out_png: Path,
    out_pdf: Path | None = None,
    *,
    title: str | None = None,
    point_size: float = 6.0,
    alpha: float = 0.4,
) -> None:
    """Render a ``(n_datasets, n_models)`` grid of real-vs-synthetic t-SNE scatters."""
    import matplotlib.pyplot as plt

    n_rows = len(datasets)
    n_cols = len(models)
    if n_rows == 0 or n_cols == 0:
        raise ValueError(
            f"plot_tsne_grid needs >=1 row and >=1 column, got "
            f"datasets={n_rows} models={n_cols}"
        )

    fig_w = max(2.0 * n_cols, 6.0)
    fig_h = max(2.0 * n_rows, 4.0)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    real_color = "#1f77b4"
    syn_color = "#ff7f0e"

    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            ax = axes[i][j]
            ax.set_xticks([])
            ax.set_yticks([])
            entry = cells.get((dataset, model))
            if entry is None:
                ax.text(
                    0.5,
                    0.5,
                    "n/a",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="0.6",
                )
                for spine in ax.spines.values():
                    spine.set_edgecolor("0.85")
            else:
                real_2d = np.asarray(entry["real_2d"])
                syn_2d = np.asarray(entry["syn_2d"])
                ax.scatter(
                    real_2d[:, 0],
                    real_2d[:, 1],
                    s=point_size,
                    c=real_color,
                    alpha=alpha,
                    linewidths=0,
                    label="Real",
                )
                ax.scatter(
                    syn_2d[:, 0],
                    syn_2d[:, 1],
                    s=point_size,
                    c=syn_color,
                    alpha=alpha,
                    linewidths=0,
                    label="Synthetic",
                )
                ax.set_aspect("equal", adjustable="datalim")

            if i == 0:
                ax.set_title(model, fontsize=10)
            if j == 0:
                ax.set_ylabel(dataset, fontsize=10, rotation=90, labelpad=8)

    # Single legend at the top-centre of the figure.
    legend_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=real_color, markersize=8, label="Real",
        ),
        plt.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=syn_color, markersize=8, label="Synthetic",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0 if title is None else 0.985),
    )

    if title is not None:
        fig.suptitle(title, y=1.02, fontsize=12)

    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_png = Path(out_png)
    _atomic_savefig(fig, out_png, dpi=300)
    if out_pdf is not None:
        _atomic_savefig(fig, Path(out_pdf), dpi=300)
    plt.close(fig)


__all__ = [
    "window_to_tsne_vector",
    "compute_real_vs_synthetic_tsne",
    "plot_tsne_grid",
]
