"""Fidelity and anomaly-preservation metrics."""

from src.evaluation.anomaly_preservation import (
    compute_all_preservation,
    compute_ard_arr,
    compute_tps,
)
from src.evaluation.fidelity import (
    compute_ks_wasserstein,
    fidelity_objective,
)
from src.evaluation.window_loading import (
    best_params_or_gaussian,
    gaussian_target_window,
    load_synthetic,
    prepare_real_windows,
)


def __getattr__(name: str):
    if name in ("compute_discriminative_score", "compute_predictive_score"):
        from src.evaluation.tsgm_scores import (
            compute_discriminative_score,
            compute_predictive_score,
        )
        _lazy = {
            "compute_discriminative_score": compute_discriminative_score,
            "compute_predictive_score": compute_predictive_score,
        }
        globals().update(_lazy)
        return _lazy[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "compute_ks_wasserstein",
    "fidelity_objective",
    "compute_discriminative_score",
    "compute_predictive_score",
    "compute_ard_arr",
    "compute_tps",
    "compute_all_preservation",
    "prepare_real_windows",
    "load_synthetic",
    "gaussian_target_window",
    "best_params_or_gaussian",
]
