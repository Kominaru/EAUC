"""EAUC: measure and mitigate eccentricity bias in dyadic regression models."""

from importlib.metadata import PackageNotFoundError, version

from .correction import EccentricityCorrector, make_probe
from .metrics import add_eccentricity, dataset_ks, eauc, eccentricity_error_curve

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
    "make_probe",
]
