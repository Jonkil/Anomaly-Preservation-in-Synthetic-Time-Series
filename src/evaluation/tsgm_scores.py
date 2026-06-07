"""TimeGAN-style discriminative and predictive fidelity scores."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _to_numpy(x: np.ndarray) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _validate_3d(arr: np.ndarray, name: str) -> np.ndarray:
    arr = _to_numpy(arr)
    if arr.ndim != 3:
        raise ValueError(
            f"{name} must have shape (N, L, F); got {arr.shape}"
        )
    return arr.astype(np.float32, copy=False)


def _pick_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _default_hidden_dim(feat_dim: int) -> int:
    """TimeGAN default: ''max(4, F // 2)'' for the small benchmark net."""
    return max(4, feat_dim // 2)


def _make_loader(
    tensors: list[torch.Tensor],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    ds = TensorDataset(*tensors)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        drop_last=False,
    )


class _GRUClassifier(nn.Module):
    """2-layer GRU feeding the final hidden state into a linear head."""

    def __init__(self, feat_dim: int, hidden_dim: int, n_layers: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        logit = self.head(h[-1])
        return logit.squeeze(-1)


class _GRUForecaster(nn.Module):
    """2-layer GRU producing one scalar per time step."""

    def __init__(self, feat_dim: int, hidden_dim: int, n_layers: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out).squeeze(-1)


def _train_classifier(
    model: _GRUClassifier,
    loader: DataLoader,
    iterations: int,
    lr: float,
    device: torch.device,
) -> None:
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    step = 0
    while step < iterations:
        for xb, yb in loader:
            if step >= iterations:
                break
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optim.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            step += 1


def _train_forecaster(
    model: _GRUForecaster,
    loader: DataLoader,
    iterations: int,
    lr: float,
    device: torch.device,
) -> None:
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    model.train()
    step = 0
    while step < iterations:
        for xb, yb in loader:
            if step >= iterations:
                break
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
            step += 1


def compute_discriminative_score(
    real: np.ndarray,
    synthetic: np.ndarray,
    *,
    iterations: int = 2000,
    batch_size: int = 128,
    hidden_dim: int | None = None,
    n_layers: int = 2,
    lr: float = 1e-3,
    device: str | torch.device | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Train a 2-layer GRU to classify real vs synthetic windows.

    Args:
        real: Real windows of shape ''(N_r, L, F)''.
        synthetic: Synthetic windows of shape ''(N_s, L, F)''.
        iterations: Total number of gradient steps (TimeGAN default 2000).
        batch_size: Batch size used for both training and evaluation.
        hidden_dim: GRU hidden size. Defaults to ''max(4, F // 2)''.
        n_layers: Number of stacked GRU layers. TimeGAN uses 2.
        lr: Adam learning rate.
        device: Target torch device (auto-detect CUDA when ''None'').
        seed: RNG seed for init, split, and loader order.

    Returns:
        Dict with keys ''score'' (''|acc - 0.5|''), ''test_accuracy'',
        ''n_real'', ''n_syn'', ''hidden_dim'', ''iterations''.
    """
    real = _validate_3d(real, "real")
    synthetic = _validate_3d(synthetic, "synthetic")
    if real.shape[1:] != synthetic.shape[1:]:
        raise ValueError(
            f"Shape mismatch after first axis: {real.shape} vs {synthetic.shape}"
        )
    feat_dim = real.shape[-1]
    if hidden_dim is None:
        hidden_dim = _default_hidden_dim(feat_dim)
    dev = _pick_device(device)

    torch.manual_seed(seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    x = np.concatenate([real, synthetic], axis=0).astype(np.float32, copy=False)
    y = np.concatenate(
        [np.ones(len(real), dtype=np.float32), np.zeros(len(synthetic), dtype=np.float32)]
    )

    idx = rng.permutation(len(x))
    x, y = x[idx], y[idx]

    n_train = int(0.8 * len(x))
    x_tr, x_te = x[:n_train], x[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]

    train_loader = _make_loader(
        [torch.from_numpy(x_tr), torch.from_numpy(y_tr)],
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    model = _GRUClassifier(feat_dim, hidden_dim, n_layers).to(dev)
    _train_classifier(model, train_loader, iterations, lr, dev)

    model.eval()
    preds_chunks: list[np.ndarray] = []
    ys_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_te), batch_size):
            xb = torch.from_numpy(x_te[start : start + batch_size]).to(dev)
            yb = torch.from_numpy(y_te[start : start + batch_size]).to(dev)
            logits = model(xb)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            preds_chunks.append(preds.cpu().numpy())
            ys_chunks.append(yb.cpu().numpy())

    preds_all = (
        np.concatenate(preds_chunks) if preds_chunks else np.zeros(0, dtype=np.float32)
    )
    ys_all = (
        np.concatenate(ys_chunks) if ys_chunks else np.zeros(0, dtype=np.float32)
    )
    total = int(ys_all.shape[0])
    acc = float((preds_all == ys_all).mean()) if total else 0.0

    # Balanced accuracy = mean of per-class recall; equals raw accuracy
    # when classes are balanced. Robust to class imbalance (used for
    # pooled discriminative where syn is concatenated across seeds).
    real_mask = ys_all == 1.0
    syn_mask = ys_all == 0.0
    has_real = bool(real_mask.any())
    has_syn = bool(syn_mask.any())
    if has_real and has_syn:
        tpr = float((preds_all[real_mask] == 1.0).mean())
        tnr = float((preds_all[syn_mask] == 0.0).mean())
        bal_acc = 0.5 * (tpr + tnr)
    else:
        tpr = float("nan")
        tnr = float("nan")
        bal_acc = float("nan")

    score = abs(acc - 0.5)
    score_balanced = (
        abs(bal_acc - 0.5) if not np.isnan(bal_acc) else float("nan")
    )
    return {
        "score": float(score),
        "score_balanced": float(score_balanced),
        "test_accuracy": float(acc),
        "test_balanced_accuracy": float(bal_acc),
        "test_tpr_real": float(tpr),
        "test_tnr_syn": float(tnr),
        "n_real": int(len(real)),
        "n_syn": int(len(synthetic)),
        "hidden_dim": int(hidden_dim),
        "iterations": int(iterations),
        "n_test": int(total),
    }


def _build_predictive_tensors(
    windows: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ''(inputs, targets)'' for the TimeGAN predictive task.

    - ''F >= 2'': inputs = ''X[:, :-1, :-1]'', targets = ''X[:, 1:, -1]''.
    - ''F == 1'': autoregressive next-step prediction on the single feature.
    """
    n, length, feat_dim = windows.shape
    if length < 2:
        raise ValueError(
            f"Windows must have length >= 2 for next-step prediction; got {length}"
        )
    if feat_dim >= 2:
        inputs_np = windows[:, :-1, :-1].astype(np.float32, copy=False)
        targets_np = windows[:, 1:, -1].astype(np.float32, copy=False)
    else:
        inputs_np = windows[:, :-1, :].astype(np.float32, copy=False)
        targets_np = windows[:, 1:, 0].astype(np.float32, copy=False)
    return torch.from_numpy(inputs_np), torch.from_numpy(targets_np)


def compute_predictive_score(
    real: np.ndarray,
    synthetic: np.ndarray,
    *,
    iterations: int = 2000,
    batch_size: int = 128,
    hidden_dim: int | None = None,
    n_layers: int = 2,
    lr: float = 1e-3,
    device: str | torch.device | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Train a GRU forecaster on synthetic data and evaluate MAE on real.

    Args:
        real: Real windows used only for evaluation, shape ''(N_r, L, F)''.
        synthetic: Synthetic windows used only for training, shape
            ''(N_s, L, F)''.
        iterations: Total gradient steps (TimeGAN default 2000).
        batch_size: Batch size for training and evaluation.
        hidden_dim: GRU hidden size. Defaults to ''max(4, F // 2)''.
        n_layers: Stacked GRU layers. TimeGAN uses 2.
        lr: Adam learning rate.
        device: Target torch device.
        seed: RNG seed for init and loader order.

    Returns:
        Dict with keys ''score'' (MAE on real), ''n_real'', ''n_syn'',
        ''hidden_dim'', ''iterations'', ''feat_dim_in''.
    """
    real = _validate_3d(real, "real")
    synthetic = _validate_3d(synthetic, "synthetic")
    if real.shape[1:] != synthetic.shape[1:]:
        raise ValueError(
            f"Shape mismatch after first axis: {real.shape} vs {synthetic.shape}"
        )
    feat_dim = real.shape[-1]
    feat_dim_in = max(1, feat_dim - 1) if feat_dim >= 2 else 1

    if hidden_dim is None:
        hidden_dim = _default_hidden_dim(feat_dim)
    dev = _pick_device(device)

    torch.manual_seed(seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    x_syn, y_syn = _build_predictive_tensors(synthetic)
    x_real, y_real = _build_predictive_tensors(real)

    train_loader = _make_loader(
        [x_syn, y_syn],
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    model = _GRUForecaster(feat_dim_in, hidden_dim, n_layers).to(dev)
    _train_forecaster(model, train_loader, iterations, lr, dev)

    model.eval()
    mae_sum = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(x_real), batch_size):
            xb = x_real[start : start + batch_size].to(dev)
            yb = y_real[start : start + batch_size].to(dev)
            pred = model(xb)
            mae_sum += float(torch.abs(pred - yb).sum().item())
            count += int(yb.numel())

    mae = mae_sum / max(count, 1)
    return {
        "score": float(mae),
        "n_real": int(len(real)),
        "n_syn": int(len(synthetic)),
        "hidden_dim": int(hidden_dim),
        "iterations": int(iterations),
        "feat_dim_in": int(feat_dim_in),
    }
