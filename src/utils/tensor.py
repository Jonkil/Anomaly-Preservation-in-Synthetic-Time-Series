"""Tensor conversion helpers shared across model wrappers."""

from __future__ import annotations

from typing import Any

import numpy as np


def keras_to_numpy(x: Any) -> np.ndarray:
    """Convert a Keras tensor, PyTorch tensor, or array-like to :class:`numpy.ndarray`."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)
