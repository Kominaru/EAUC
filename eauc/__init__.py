"""EAUC: measure and mitigate eccentricity bias in dyadic regression models."""

from importlib.metadata import PackageNotFoundError, version

from .correction import EccentricityCorrector, make_probe, ml_rus
from .io import load_model_outputs, save_model_outputs
from .metrics import add_eccentricity, dataset_ks, eauc, eccentricity_error_curve, entity_ks

try:
    __version__ = version("eauc")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.0.0"

__all__ = [
    "EccentricityCorrector",
    "add_eccentricity",
    "dataset_ks",
    "eauc",
    "eccentricity_error_curve",
    "entity_ks",
    "load_model_outputs",
    "make_probe",
    "ml_rus",
    "save_model_outputs",
]
