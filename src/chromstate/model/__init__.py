"""Model layer: emissions, HMM core, fitting, and automatic state annotation."""

from __future__ import annotations

from .annotate import STATE_COLORS, annotate_states, classify_marks, promoter_marks
from .binarize import poisson_binarize, poisson_threshold
from .core import Segmentation, fit, select_n_states
from .emissions import (
    BernoulliEmissions,
    Emissions,
    GaussianEmissions,
    NegativeBinomialEmissions,
    build_emissions,
)
from .hmm import HMM, forward_backward, viterbi

__all__ = [
    "HMM",
    "STATE_COLORS",
    "BernoulliEmissions",
    "Emissions",
    "GaussianEmissions",
    "NegativeBinomialEmissions",
    "Segmentation",
    "annotate_states",
    "build_emissions",
    "classify_marks",
    "fit",
    "forward_backward",
    "poisson_binarize",
    "poisson_threshold",
    "promoter_marks",
    "select_n_states",
    "viterbi",
]
