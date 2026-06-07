#!/usr/bin/env python3
"""Smoke-check that core imports resolve and GPU is visible (if available)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    print("Python:", sys.executable)
    print("Version:", sys.version)
    print()

    # --- Core scientific stack ---
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn

    print(f"numpy {np.__version__}")
    print(f"pandas {pd.__version__}")
    print(f"scipy {scipy.__version__}")
    print(f"sklearn {sklearn.__version__}")

    # --- Deep learning ---
    import keras

    print(f"keras {keras.__version__} (backend: {keras.backend.backend()})")

    import torch

    print(f"torch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")

    # --- Project-specific ---
    import tsgm

    print(f"tsgm {tsgm.__version__}")

    import optuna

    print(f"optuna {optuna.__version__}")

    import mlflow

    print(f"mlflow {mlflow.__version__}")

    # --- Internal package ---
    from src.data.preprocessor import load_tsb_csv, temporal_split, sliding_window
    from src.evaluation.fidelity import compute_ks_wasserstein
    from src.utils.seeds import set_seed

    print()
    print("All imports OK ✓")


if __name__ == "__main__":
    main()
