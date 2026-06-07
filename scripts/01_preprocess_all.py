#!/usr/bin/env python3
"""Cache raw temporal splits (70/15/15) for configured datasets - no windowing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root on path for "src" imports
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.preprocessor import (
    load_tsb_csv,
    save_raw_splits,
    temporal_split,
)
from src.training.utils import load_yaml, merge_configs, repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["SWaT", "PSM"],
        help="Dataset keys matching config/datasets/<name>.yaml",
    )
    args = parser.parse_args()
    root = repo_root()
    base_path = root / "config" / "base.yaml"
    base = load_yaml(base_path)
    data_root = root / base["data_root"]
    processed_root = root / base["processed_dir"]

    for ds in args.datasets:
        dcfg_path = root / "config" / "datasets" / f"{ds}.yaml"
        dcfg = merge_configs([base_path, dcfg_path])
        csv_name = dcfg["file"]
        label_col = dcfg.get("label_column", "Label")
        csv_path = data_root / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV not found: {csv_path}. Set data_root or place files under {data_root}."
            )
        values, labels = load_tsb_csv(csv_path, label_column=label_col)
        splits = temporal_split(values, labels)
        out_dir = processed_root / ds
        meta = {
            "dataset": ds,
            "source_csv": str(csv_path.relative_to(root)),
            "n_timesteps": int(values.shape[0]),
            "n_features": int(values.shape[1]),
            "label_column": label_col,
            "split_ratios": [0.7, 0.15, 0.15],
        }
        save_raw_splits(out_dir, splits, meta)
        print(f"Wrote raw splits for {ds} -> {out_dir}")


if __name__ == "__main__":
    main()
