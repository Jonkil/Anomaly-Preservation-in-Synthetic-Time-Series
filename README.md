# Beyond Fidelity: Assessing Anomaly Preservation in Synthetic Time Series Generation

Pipeline code for:
- data preprocessing
- hyperparameter tuning with Optuna (two phases: preprocessing, then model)
- training generative models (DDPM, RTSGAN, TTS-GAN, TimeVAE)
- training anomaly detection models (TadGAN, WassersteinGAN)
- anomaly preservation evaluation.

## Datasets and trained models

The input data and trained models did not fit into the GitHub repository. Three zip files worth 5.6 GB were uploded to Zenodo platform (https://zenodo.org/records/20574743).
- anomaly_detectors.zip
- generative_models.zip
- TSB-AD-M.zip

Run the following commands to download the zip-files from Zenodo:

```bash
chmod +x download_zenodo.sh
download_zenodo.sh
```

## Requirements
The computations were run on the ex3 HPC server using Nvidia GPUs and CUDA. The server access was provided by the Simula Research Lab (Oslo, Norway).

- Python 3.10+
- CUDA 12.8 + PyTorch cu128 wheels (see `requirements.txt`)
- Cluster modules (eX3): `slurm/21.08.8`, `cuda12.8/toolkit/12.8.1`

## Environment and when to rebuild `.venv`

Pinned dependencies live in [`requirements.txt`](requirements.txt). **Recreate the virtual environment** if any of the following change:

- Python minor version (e.g. 3.10 -> 3.11)
- CPU architecture or wheel ABI (e.g. x86_64 vs aarch64 - PyTorch wheels differ)
- CUDA driver / toolkit module version on the cluster
- You moved the repo to a host where the old `.venv` path is unavailable or corrupt

If the machine matches the one used for a successful `verify_final` / `scripts/validate_env.py` run, rebuilding is optional; still run validation after copying the tree.

Setup (local or job script):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Smoke-check imports and GPU:

```bash
export KERAS_BACKEND=torch
python scripts/validate_env.py
```

## Data

Place TSB-AD-M CSVs under `TSB-AD-M/` (see `config/base.yaml` -> `data_root`) or adjust `data_root` / per-dataset `file` in `config/datasets/*.yaml`.

## Preprocessing

Writes **raw** temporal splits only (70% / 15% / 15%), no global windowing:

```bash
export KERAS_BACKEND=torch
python scripts/01_preprocess_all.py --datasets SWaT PSM
```

## Tuning (phased Optuna + MLflow)

Phase 1 searches `window_size`, `scaler_type`, and a small TimeVAE; Phase 2 fixes the winning preprocessing and searches model hyperparameters from `config/models/TimeVAE.yaml`.

```bash
export MLFLOW_TRACKING_URI="file:$(pwd)/logs/mlflow"
export KERAS_BACKEND=torch
python scripts/02_tune_model.py SWaT --model TimeVAE --phases all
```

Optional environment overrides for smoke runs:

| Variable | Meaning |
|----------|---------|
| `TUNE_N_TRIALS_PHASE1` | Optuna trials for phase 1 |
| `TUNE_N_TRIALS_PHASE2` | Optuna trials for phase 2 |
| `TUNE_FIDELITY_N_SAMPLES` | Synthetic samples for fidelity during tuning |
| `TUNE_SUBSAMPLE_ROWS` | Use only the first *N* rows of `train_gen` for speed |

## Final training seeds

Canonical seeds for multi-seed final runs: `[0, 123, 456, 789, 1011]` (see `src/utils/seeds.py`).

## Reproducibility

Tuning logs `git_sha` to MLflow when run inside a git checkout (`src/training/tune.py`).
