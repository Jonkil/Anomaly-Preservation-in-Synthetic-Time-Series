#!/usr/bin/env python3
"""Entry point for phased Optuna tuning (TimeVAE, RTSGAN, DDPM, TTSGAN, CSDI)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.training.tune import run_phased


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Dataset key, e.g. SWaT")
    parser.add_argument(
        "--model",
        default="TimeVAE",
        choices=[
            "TimeVAE",
            "TimeVAE_v2",
            "TimeVAE_v3",
            "RTSGAN",
            "DDPM",
            "TTSGAN",
            "CSDI",
        ],
        help="Generative model to tune",
    )
    parser.add_argument(
        "--phases",
        default="all",
        choices=["1", "2", "all"],
        help="Run preprocessing search, model search, or both",
    )
    parser.add_argument(
        "--n-trials-phase1",
        type=int,
        default=None,
        help="Override config n_trials_phase1",
    )
    parser.add_argument(
        "--n-trials-phase2",
        type=int,
        default=None,
        help="Override config n_trials_phase2",
    )
    args = parser.parse_args()

    os.environ.setdefault("KERAS_BACKEND", "torch")

    out = run_phased(
        args.dataset,
        model=args.model,
        phases=args.phases,
        n_trials_phase1=args.n_trials_phase1,
        n_trials_phase2=args.n_trials_phase2,
    )
    print(out)


if __name__ == "__main__":
    main()
