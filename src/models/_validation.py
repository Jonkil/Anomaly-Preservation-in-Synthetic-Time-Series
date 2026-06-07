"""Cross-model invariants: scaler family and output activation pairing."""

from __future__ import annotations

from typing import Final, Literal

OutputActivation = Literal["sigmoid", "tanh", "linear"]

SCALER_FAMILIES = ("Standard", "MinMax", "Robust", "PerWindowZNorm")
ScalerFamily = Literal["Standard", "MinMax", "Robust", "PerWindowZNorm"]

# Each scaler family lists the *valid* output activations for a model whose
# reconstruction target lives in that scaled space.
SCALER_ACTIVATION_PAIRS: Final[dict[str, tuple[OutputActivation, ...]]] = {
    "Standard": ("linear",),
    "Robust": ("linear",),
    "MinMax": ("sigmoid",),
    "PerWindowZNorm": ("linear",),
}


def profile_to_scaler_family(
    profile: str, sklearn_scaler_name: str | None = None,
) -> ScalerFamily:
    """Return the effective scaler family for a given preprocessing profile."""
    if profile == "improved":
        return "PerWindowZNorm"
    if profile == "legacy":
        if sklearn_scaler_name not in ("Standard", "MinMax", "Robust"):
            raise ValueError(
                f"legacy profile requires scaler in "
                f"{{'Standard','MinMax','Robust'}}; got {sklearn_scaler_name!r}"
            )
        return sklearn_scaler_name  # type: ignore[return-value]
    raise ValueError(f"unknown preprocessing profile {profile!r}")


def validate_scaler_activation(
    scaler_family: str,
    output_activation: str,
) -> None:
    """Raise :class:`ValueError` if the ''(scaler, activation)'' pair is invalid."""
    if scaler_family not in SCALER_ACTIVATION_PAIRS:
        raise ValueError(
            f"unknown scaler_family {scaler_family!r}; "
            f"valid: {sorted(SCALER_ACTIVATION_PAIRS)}"
        )
    if output_activation not in ("sigmoid", "tanh", "linear"):
        raise ValueError(
            f"unknown output_activation {output_activation!r}; "
            f"valid: ('sigmoid', 'tanh', 'linear')"
        )
    valid = SCALER_ACTIVATION_PAIRS[scaler_family]
    if output_activation not in valid:
        raise ValueError(
            f"output_activation={output_activation!r} is not valid for "
            f"scaler_family={scaler_family!r}; expected one of {valid}. "
            f"This pairing produces clipped or saturated reconstructions "
            f"and corrupts fidelity metrics."
        )


def valid_activations_for(scaler_family: str) -> tuple[OutputActivation, ...]:
    """Return the tuple of activations that are valid for ''scaler_family''."""
    if scaler_family not in SCALER_ACTIVATION_PAIRS:
        raise ValueError(
            f"unknown scaler_family {scaler_family!r}; "
            f"valid: {sorted(SCALER_ACTIVATION_PAIRS)}"
        )
    return SCALER_ACTIVATION_PAIRS[scaler_family]


__all__ = [
    "OutputActivation",
    "SCALER_ACTIVATION_PAIRS",
    "SCALER_FAMILIES",
    "ScalerFamily",
    "profile_to_scaler_family",
    "validate_scaler_activation",
    "valid_activations_for",
]
