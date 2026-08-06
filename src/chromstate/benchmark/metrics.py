"""Evaluation metrics for comparing chromatin state methods against ground truth.

A note on what is *not* here: there is no head-to-head log-likelihood comparison between
the negative-binomial and Bernoulli models. Their likelihoods are computed over different
data -- counts versus binarised calls -- so the numbers are not on the same scale and
comparing them would be meaningless. Held-out likelihood is only used to compare models
fit to the same representation.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    average_precision_score,
    roc_auc_score,
    v_measure_score,
)

EPS = 1e-12


def confusion(true: np.ndarray, pred: np.ndarray, n_true: int, n_pred: int) -> np.ndarray:
    C = np.zeros((n_true, n_pred), dtype=np.int64)
    np.add.at(C, (true.astype(int), pred.astype(int)), 1)
    return C


def match_states(
    true: np.ndarray, pred: np.ndarray, n_true: int, n_pred: int
) -> Tuple[Dict[int, int], float]:
    """Optimally map predicted state indices onto true ones (Hungarian assignment).

    Returns the mapping ``pred_index -> true_index`` and the accuracy achieved under it.
    Necessary because state indices are arbitrary: a method that recovers the truth
    perfectly but numbers the states differently must not be penalised.
    """
    C = confusion(true, pred, n_true, n_pred)
    rows, cols = linear_sum_assignment(-C)
    mapping = {int(c): int(r) for r, c in zip(rows, cols)}
    # Predicted states left unmatched (when n_pred > n_true) go to their modal true state.
    for p in range(n_pred):
        if p not in mapping:
            mapping[p] = int(np.argmax(C[:, p])) if C[:, p].sum() else 0
    mapped = np.array([mapping[int(p)] for p in pred])
    return mapping, float((mapped == true).mean())


def clustering_scores(true: np.ndarray, pred: np.ndarray, n_true: int, n_pred: int) -> Dict[str, float]:
    """Label-agreement metrics that are invariant to how states are numbered."""
    mapping, acc = match_states(true, pred, n_true, n_pred)
    return {
        "ari": float(adjusted_rand_score(true, pred)),
        "ami": float(adjusted_mutual_info_score(true, pred)),
        "v_measure": float(v_measure_score(true, pred)),
        "matched_accuracy": acc,
        "n_states_used": int(len(np.unique(pred))),
    }


def calibration_scores(
    posterior: np.ndarray,
    true: np.ndarray,
    mapping: Dict[int, int],
    n_bins: int = 10,
) -> Dict[str, float]:
    """How honest are a method's posterior probabilities?

    ``overconfidence`` is the headline number: mean posterior confidence minus actual
    accuracy. A method that reports 0.99 certainty and is right 70% of the time scores
    +0.29 and is systematically misleading about how much a given call can be trusted.
    This is precisely the failure mode that makes it hard to tell which regions in a
    segmentation are solid and which are guesses.
    """
    conf = posterior.max(axis=1)
    pred = np.argmax(posterior, axis=1)
    mapped = np.array([mapping.get(int(p), 0) for p in pred])
    correct = (mapped == true).astype(float)

    onehot = np.zeros_like(posterior)
    mapped_all = np.array([mapping.get(k, 0) for k in range(posterior.shape[1])])
    for k_true in np.unique(true):
        cols = np.flatnonzero(mapped_all == k_true)
        if cols.size:
            onehot[true == k_true, cols[0]] = 1.0
    brier = float(((posterior - onehot) ** 2).sum(axis=1).mean())

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        sel = (conf > edges[i]) & (conf <= edges[i + 1])
        if sel.sum() == 0:
            continue
        ece += (sel.mean()) * abs(correct[sel].mean() - conf[sel].mean())

    p = np.clip(posterior, EPS, 1.0)
    entropy = -(p * np.log(p)).sum(axis=1) / np.log(posterior.shape[1])

    return {
        "accuracy": float(correct.mean()),
        "mean_confidence": float(conf.mean()),
        "overconfidence": float(conf.mean() - correct.mean()),
        "brier": brier,
        "ece": float(ece),
        "mean_entropy": float(entropy.mean()),
        "frac_saturated": float((conf >= 0.99).mean()),
    }


def detection_scores(score: np.ndarray, positive: np.ndarray) -> Dict[str, float]:
    """AUROC and average precision for a continuous score against a binary truth."""
    positive = np.asarray(positive).astype(int)
    if positive.sum() == 0 or positive.sum() == len(positive):
        return {"auroc": np.nan, "auprc": np.nan, "n_positive": int(positive.sum())}
    score = np.nan_to_num(np.asarray(score, dtype=float), nan=0.0)
    return {
        "auroc": float(roc_auc_score(positive, score)),
        "auprc": float(average_precision_score(positive, score)),
        "baseline_auprc": float(positive.mean()),
        "n_positive": int(positive.sum()),
    }


def quantitative_sensitivity(score: np.ndarray, true_signal: np.ndarray) -> Dict[str, float]:
    """Does a method's output track the *amount* of signal, not just its presence?

    Restricted to bins the method calls present, because outside those the question is
    trivial. A method that discards magnitude scores near zero here by construction; this
    is the direct measurement of what binarisation costs.
    """
    score = np.asarray(score, dtype=float)
    true_signal = np.asarray(true_signal, dtype=float)
    ok = np.isfinite(score) & np.isfinite(true_signal)
    if ok.sum() < 10 or np.ptp(score[ok]) == 0:
        return {"spearman": np.nan, "pearson": np.nan, "n": int(ok.sum())}
    rho = spearmanr(score[ok], true_signal[ok]).correlation
    with np.errstate(invalid="ignore"):
        r = np.corrcoef(score[ok], true_signal[ok])[0, 1]
    return {"spearman": float(rho), "pearson": float(r), "n": int(ok.sum())}


def segmentation_agreement(pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, float]:
    """Agreement between two segmentations (e.g. across restarts). A stability diagnostic."""
    return {
        "ari": float(adjusted_rand_score(pred_a, pred_b)),
        "ami": float(adjusted_mutual_info_score(pred_a, pred_b)),
    }


def bivalency_truth_from_states(true_states: np.ndarray, bivalent_state: Optional[int]) -> np.ndarray:
    if bivalent_state is None:
        raise ValueError("Simulation has no designated bivalent state")
    return true_states == bivalent_state
