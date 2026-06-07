#!/usr/bin/env python3
"""Train final models across all seeds with best hyperparameters."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.training.train import load_best_params, train_all_seeds
from src.training.utils import repo_root, save_json
from src.utils.seeds import SEEDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Dataset key, e.g. SWaT")
    parser.add_argument(
        "model",
        nargs="?",
        default="TimeVAE",
        help="Model key, e.g. TimeVAE (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run a single seed instead of all 5",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs (env FINAL_TRAIN_EPOCHS also works)",
    )
    parser.add_argument(
        "--n-generate",
        type=int,
        default=None,
        help="Number of synthetic windows to produce (default: match train_gen)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Keras verbosity: 0=silent, 1=progress bar",
    )
    args = parser.parse_args()

    os.environ.setdefault("KERAS_BACKEND", "torch")

    params = load_best_params(args.dataset, args.model)
    print(f"Best params for {args.dataset}/{args.model}:")
    common = (
        f"  window={params['window_size']} stride={params['stride']} "
        f"scaler={params['scaler_type']}"
    )
    if args.model == "RTSGAN":
        print(
            f"{common} hidden_dim={params['hidden_dim']} "
            f"layers={params['layers']} noise_dim={params['noise_dim']} "
            f"ae_lr={params['ae_lr']:.6f} gan_lr={params['gan_lr']:.6f}"
        )
    elif args.model == "DDPM":
        print(
            f"{common} n_filters={params['n_filters']} "
            f"n_conv_layers={params['n_conv_layers']} "
            f"timesteps={params['timesteps']} "
            f"lr={params['learning_rate']:.6f} batch={params['batch_size']}"
        )
    elif args.model == "TTSGAN":
        print(
            f"{common} latent_dim={params['latent_dim']} "
            f"embed_dim={params['embed_dim']} depth={params['depth']} "
            f"num_heads={params['num_heads']} "
            f"lr_g={params['lr_g']:.6f} lr_d={params['lr_d']:.6f}"
        )
    elif args.model == "CSDI":
        print(
            f"{common} channels={params['channels']} layers={params['layers']} "
            f"nheads={params['nheads']} num_steps={params['num_steps']} "
            f"timeemb={params['timeemb']} featureemb={params['featureemb']} "
            f"schedule={params['schedule']} "
            f"lr={params['learning_rate']:.6f} batch={params['batch_size']}"
        )
    else:
        print(
            f"{common} latent={params['latent_dim']} "
            f"beta={params['beta']} lr={params['learning_rate']:.6f} "
            f"batch={params['batch_size']}"
        )

    seeds = [args.seed] if args.seed is not None else list(SEEDS)
    print(f"Training seeds: {seeds}")

    results = train_all_seeds(
        args.dataset,
        args.model,
        seeds=seeds,
        epochs=args.epochs,
        n_generate=args.n_generate,
        verbose=args.verbose,
    )

    root = repo_root()
    summary_path = root / "results" / f"train_summary_{args.dataset}_{args.model}.json"
    save_json(summary_path, results)
    print(f"\nSummary saved to {summary_path}")

    ks_vals = [r["ks_mean"] for r in results]
    w_vals = [r["wasserstein_mean"] for r in results]
    import numpy as np
    print(f"Fidelity across {len(results)} seeds:")
    print(f"  KS:          {np.mean(ks_vals):.4f} ± {np.std(ks_vals):.4f}")
    print(f"  Wasserstein:  {np.mean(w_vals):.4f} ± {np.std(w_vals):.4f}")


if __name__ == "__main__":
    main()
