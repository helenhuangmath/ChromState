"""Benchmarking against competing chromatin state methods."""

from __future__ import annotations

from .baselines import METHODS, MethodResult, export_for_chromhmm
from .metrics import (
    calibration_scores,
    clustering_scores,
    detection_scores,
    match_states,
    quantitative_sensitivity,
    segmentation_agreement,
)
from .run import (
    run_binarization_experiment,
    run_comparison,
    run_full_benchmark,
    run_normalization_experiment,
    run_perturbation_experiment,
)

__all__ = [
    "METHODS",
    "MethodResult",
    "calibration_scores",
    "clustering_scores",
    "detection_scores",
    "export_for_chromhmm",
    "match_states",
    "quantitative_sensitivity",
    "run_binarization_experiment",
    "run_comparison",
    "run_full_benchmark",
    "run_normalization_experiment",
    "run_perturbation_experiment",
    "segmentation_agreement",
]
