"""Posterior calibration: making a reported probability mean what it says.

The problem this exists to fix is measured, not hypothetical. On the benchmark, 81% of
ChromState's calls sit above 0.99 posterior and mean normalised entropy is 0.05. The
model is the *best calibrated* of the methods compared and still expected-calibration-
error 0.049 with every reliability point below the diagonal. Two consequences:

1. Entropy cannot be read as a plasticity measure while it is pinned near zero.
2. Change scores are compressed to the ends. Total-variation distance between two
   saturated posteriors is nearly binary -- a bin either "completely changed" or "did not
   move" -- so genuine partial shifts, which is what most regulation looks like, are
   invisible.

The fix here is temperature scaling: divide the log-posterior by a single scalar *T* and
renormalise. It is the standard remedy, it cannot change any hard call (the argmax is
invariant), and it has exactly one parameter, so it cannot manufacture structure.

What it is fitted *against* is the part that requires care, and there are two honest
options with different meanings:

``fit_temperature`` (supervised)
    Needs known labels, so it exists for simulation and benchmarking. Calibrates against
    *correctness*.

``fit_temperature_from_replicates`` (unsupervised)
    Uses agreement across replicates of the same condition as the target. Calibrates
    against *reproducibility*, which is what is actually available on real data. A model
    that is confidently wrong in the same way in every replicate will not be caught by
    this -- systematic bias is invisible to a reproducibility criterion, and that limit
    should be stated wherever the calibrated numbers are reported.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .model.core import Segmentation

EPS = 1e-12


def temperature_scale(posterior: np.ndarray, temperature: float) -> np.ndarray:
    """Raise a posterior to the power ``1/T`` and renormalise.

    ``T > 1`` softens (less confident), ``T < 1`` sharpens, ``T == 1`` is a no-op. The
    ranking within each row is untouched, so the hard segmentation never changes.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if temperature == 1.0:
        return np.asarray(posterior, dtype=np.float64)
    logp = np.log(np.clip(np.asarray(posterior, dtype=np.float64), EPS, 1.0)) / temperature
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / np.maximum(p.sum(axis=1, keepdims=True), EPS)


def _nll(posterior: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    scaled = temperature_scale(posterior, temperature)
    picked = scaled[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(picked, EPS, 1.0)).mean())


def _fit_temperature_on_labels(
    posterior: np.ndarray,
    labels: np.ndarray,
    objective: str = "ece",
    bounds: Tuple[float, float] = (0.25, 40.0),
    n_grid: int = 90,
) -> float:
    """Search a temperature over a log-spaced grid, then refine locally.

    ``objective``
        ``ece`` (default) minimises the expected calibration error -- the gap between
        reported confidence and observed accuracy, which is the property the whole module
        exists to fix. ``nll`` minimises log loss, the classic choice for temperature
        scaling.

        The default is not the classic one, deliberately. Log loss is unbounded on
        confidently-wrong predictions, and a genome-scale segmentation has a large
        *absolute* number of those even at high accuracy, so an NLL fit is dragged by the
        tail and overshoots: on the test dataset it turns +0.041 overconfidence into
        -0.053 underconfidence and leaves the calibration error slightly worse than it
        started. A model that understates its confidence is no more usable than one that
        overstates it.

    A grid rather than a gradient method: the search is one-dimensional, the calibration
    error is piecewise constant so derivatives do not exist, and a grid is reproducible
    and immune to a bad starting point.
    """
    if objective not in ("ece", "nll"):
        raise ValueError(f"objective must be 'ece' or 'nll', got {objective!r}")

    def loss(t: float) -> float:
        if objective == "nll":
            return _nll(posterior, labels, t)
        return expected_calibration_error(temperature_scale(posterior, t), labels)

    grid = np.geomspace(bounds[0], bounds[1], n_grid)
    losses = [loss(t) for t in grid]
    best = int(np.argmin(losses))
    lo = grid[max(best - 1, 0)]
    hi = grid[min(best + 1, n_grid - 1)]

    fine = np.linspace(lo, hi, 40)
    fine_losses = [loss(t) for t in fine]
    return float(fine[int(np.argmin(fine_losses))])


@dataclass
class CalibrationResult:
    """A fitted temperature and the evidence for it."""

    temperature: float
    n_used: int
    target: str
    before: Dict[str, float]
    after: Dict[str, float]
    note: str

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (f"CalibrationResult(temperature={self.temperature:.3f}, "
                f"target={self.target!r}, n_used={self.n_used})")


def confidence_summary(posterior: np.ndarray, labels: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Mean confidence, saturation, entropy, and -- with labels -- accuracy and error."""
    conf = posterior.max(axis=1)
    ent = -(posterior * np.log(np.clip(posterior, EPS, 1.0))).sum(axis=1)
    ent /= np.log(posterior.shape[1]) if posterior.shape[1] > 1 else 1.0
    out = {
        "mean_confidence": float(conf.mean()),
        "frac_above_0.99": float((conf > 0.99).mean()),
        "mean_entropy": float(ent.mean()),
    }
    if labels is not None:
        correct = (np.argmax(posterior, axis=1) == labels).astype(float)
        out["accuracy"] = float(correct.mean())
        out["overconfidence"] = float(conf.mean() - correct.mean())
        out["ece"] = float(expected_calibration_error(posterior, labels))
    return out


def expected_calibration_error(posterior: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Weighted mean gap between reported confidence and observed accuracy."""
    conf = posterior.max(axis=1)
    correct = (np.argmax(posterior, axis=1) == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        sel = (conf > edges[i]) & (conf <= edges[i + 1])
        if sel.sum() == 0:
            continue
        total += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(total)


def fit_temperature(
    seg: Segmentation,
    true_states: np.ndarray,
    mapping: Optional[Dict[int, int]] = None,
    holdout_fraction: float = 0.5,
    objective: str = "ece",
    seed: int = 0,
) -> CalibrationResult:
    """Fit a temperature against known labels. Simulation and benchmarking only.

    The temperature is fitted on one half of the bins and reported on the other, because
    a calibration measured on the data that fitted it is not a measurement.
    """
    post = seg.posterior
    true_states = np.asarray(true_states)
    if true_states.shape[0] != post.shape[0]:
        raise ValueError(f"{true_states.shape[0]} labels for {post.shape[0]} bins")

    n_true = int(true_states.max()) + 1
    if mapping is None:
        from .benchmark.metrics import match_states

        mapping, _ = match_states(true_states, seg.states, n_true, seg.n_states)

    # Project the posterior into the truth's index space by summing the columns that map
    # to the same true state, rather than inverting the mapping. The mapping is not
    # required to be injective -- two model states can both be the best match for one true
    # state -- and inverting it would silently discard those bins.
    projection = np.zeros((seg.n_states, n_true))
    for pred_state, true_state in mapping.items():
        projection[int(pred_state), int(true_state)] = 1.0
    unmapped = projection.sum(axis=1) == 0
    if unmapped.any():
        # A model state with no match keeps its own column so its mass is not deleted;
        # it simply can never be counted as correct.
        projection = np.concatenate(
            [projection, np.eye(seg.n_states)[:, unmapped]], axis=1
        )

    labels = true_states.astype(int)
    keep = labels < n_true
    post, labels = post[keep], labels[keep]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labels))
    cut = int(len(idx) * holdout_fraction)
    fit_idx, test_idx = idx[cut:], idx[:cut]

    def project(p: np.ndarray) -> np.ndarray:
        q = p @ projection
        return q / np.maximum(q.sum(axis=1, keepdims=True), EPS)

    temperature = _fit_temperature_on_labels(project(post[fit_idx]), labels[fit_idx],
                                            objective=objective)
    before = confidence_summary(project(post[test_idx]), labels[test_idx])
    after = confidence_summary(project(temperature_scale(post[test_idx], temperature)),
                               labels[test_idx])
    note = (
        f"Temperature {temperature:.3f} fitted on {len(fit_idx)} bins against known labels "
        f"and reported on {len(test_idx)} held-out bins. Calibrated against correctness, "
        f"which is only available in simulation."
    )
    return CalibrationResult(temperature, int(keep.sum()), "known labels", before, after, note)


def fit_temperature_from_replicates(
    seg: Segmentation,
    groups: Dict[str, Sequence[str]],
    min_agreement_bins: int = 200,
    objective: str = "ece",
    seed: int = 0,
) -> CalibrationResult:
    """Fit a temperature against replicate agreement -- the real-data option.

    For every condition with at least two replicates, each bin gets a pseudo-label: the
    state the *other* replicates agree on. A replicate's own posterior is then scored
    against that leave-one-out consensus, which asks the question a probability should
    answer: when this model says 0.9, does the same bin come back the same way in an
    independent library 90% of the time?

    Leave-one-out matters. Scoring a replicate against a consensus it helped define would
    reward confidence for its own sake, and the fitted temperature would come back at 1.0
    no matter how saturated the posteriors were.
    """
    ds = seg.dataset
    usable = {name: list(reps) for name, reps in groups.items() if len(reps) >= 2}
    if not usable:
        raise ValueError(
            "Replicate calibration needs at least one condition with 2+ replicates; "
            f"got { {k: len(v) for k, v in groups.items()} }"
        )
    for reps in usable.values():
        for s in reps:
            if s not in ds.samples:
                raise ValueError(f"Unknown sample {s!r}; dataset has {ds.samples}")

    posts, labels = [], []
    for reps in usable.values():
        stack = np.stack([seg.posterior[ds.sample_mask(s)] for s in reps])  # (R, B, K)
        hard = np.argmax(stack, axis=2)                                     # (R, B)
        n_reps = stack.shape[0]
        for r in range(n_reps):
            others = np.delete(hard, r, axis=0)
            # Leave-one-out consensus: modal state among the other replicates.
            counts = np.zeros((others.shape[1], seg.n_states), dtype=np.int64)
            for row in others:
                counts[np.arange(others.shape[1]), row] += 1
            consensus = np.argmax(counts, axis=1)
            unanimous_or_modal = counts.max(axis=1) > (n_reps - 1) / 2.0
            posts.append(stack[r][unanimous_or_modal])
            labels.append(consensus[unanimous_or_modal])

    post = np.concatenate(posts)
    label = np.concatenate(labels)
    if len(label) < min_agreement_bins:
        raise ValueError(
            f"Only {len(label)} bins had a leave-one-out consensus, below the "
            f"{min_agreement_bins} required for a stable fit"
        )

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(label))
    cut = len(idx) // 2
    temperature = _fit_temperature_on_labels(post[idx[cut:]], label[idx[cut:]],
                                            objective=objective)
    before = confidence_summary(post[idx[:cut]], label[idx[:cut]])
    after = confidence_summary(temperature_scale(post[idx[:cut]], temperature), label[idx[:cut]])
    note = (
        f"Temperature {temperature:.3f} fitted against leave-one-out replicate consensus "
        f"over {len(label)} bin-replicate pairs from "
        f"{', '.join(f'{k} (n={len(v)})' for k, v in usable.items())}. This calibrates "
        f"against reproducibility, not correctness: a model that is confidently wrong the "
        f"same way in every replicate will still look calibrated here."
    )
    return CalibrationResult(temperature, int(len(label)), "replicate consensus",
                             before, after, note)


def apply_temperature(seg: Segmentation, temperature: float) -> Segmentation:
    """A copy of the segmentation with a temperature-scaled posterior.

    Hard states are carried over unchanged -- temperature cannot move an argmax -- so the
    segmentation is identical and only the confidence attached to it moves.
    """
    scaled = temperature_scale(seg.posterior, temperature)
    meta = dict(seg.meta)
    meta["temperature"] = float(temperature)
    return replace(seg, posterior=scaled, meta=meta)


def reliability_table(
    posterior: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Per-confidence-bin reported confidence against observed accuracy."""
    conf = posterior.max(axis=1)
    correct = (np.argmax(posterior, axis=1) == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        sel = (conf > edges[i]) & (conf <= edges[i + 1])
        if sel.sum() == 0:
            continue
        rows.append(
            {
                "bin_low": edges[i],
                "bin_high": edges[i + 1],
                "n": int(sel.sum()),
                "mean_confidence": float(conf[sel].mean()),
                "observed_accuracy": float(correct[sel].mean()),
                "gap": float(conf[sel].mean() - correct[sel].mean()),
            }
        )
    return pd.DataFrame(rows)
