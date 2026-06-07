"""Abstract base for generative models (minimal interface)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class GenerativeModel(ABC):
    """Common fit/generate surface for tuning scripts."""

    @abstractmethod
    def fit(self, x: np.ndarray, **kwargs: Any) -> None:
        """Train on windowed data "(N, L, F)"."""

    @abstractmethod
    def generate(self, n: int) -> np.ndarray:
        """Sample "n" synthetic windows."""
