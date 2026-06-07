"""Anomaly detector implementations (TadGAN, WGAN-GP).

Both detectors inherit from :class:`AnomalyDetector` and expose the same
``fit`` / ``score`` / ``predict`` / ``save`` / ``load`` contract.
"""

from src.anomaly_detectors.base import AnomalyDetector
from src.anomaly_detectors.tadgan import TadGANDetector
from src.anomaly_detectors.wgan import WGANDetector

DETECTOR_REGISTRY: dict[str, type[AnomalyDetector]] = {
    "TadGAN": TadGANDetector,
    "WGAN": WGANDetector,
}

__all__ = [
    "AnomalyDetector",
    "TadGANDetector",
    "WGANDetector",
    "DETECTOR_REGISTRY",
]
