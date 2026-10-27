"""Bivalency and activity scores for tools that only return states (and maybe posteriors).

The in-process methods score bivalency and activity as ``posterior @ f(state profile)``,
where ``f`` is the same harmonic-mean / log-mean formula for every method
(:func:`chromstate.benchmark.baselines._state_bivalency` and ``_state_activity``). Real
tools report their emission parameters in incompatible units -- ChromHMM a presence
probability, EpiCSeg negative-multinomial means, diHMM per-bin-state Bernoulli rates --
so reusing each tool's own parameters would make the score depend on the tool's
parameterisation rather than on its segmentation.

Instead every external tool gets the *same post-hoc profile*: each state's mean
normalised count, weighted by that tool's posterior (or by its hard labels when it has no
posterior). The formula applied to the profile is imported, not copied, so a change to
the baseline definition moves every row together. This means the external tools'
bivalency and magnitude columns measure how well their *partition* separates bivalent
or highly marked bins -- which is what a user of their segmentation would get -- not a
quantity the tool itself reports.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from chromstate.benchmark.baselines import _state_activity, _state_bivalency

from .core import UNASSIGNED

__all__ = ["state_profiles", "posthoc_scores"]


def _weights(states: np.ndarray, posterior: Optional[np.ndarray], n_states: int) -> np.ndarray:
    """Per-bin state weights: the posterior where given, one-hot labels otherwise.

    Unassigned bins get an all-zero row, so they contribute to no profile and score 0.
    """
    if posterior is not None:
        W = np.asarray(posterior, dtype=np.float64).copy()
        W[states == UNASSIGNED] = 0.0
        return W
    W = np.zeros((states.size, max(n_states, int(states.max()) + 1 if states.size else 1)))
    ok = states != UNASSIGNED
    W[np.flatnonzero(ok), states[ok]] = 1.0
    return W


def state_profiles(ds, states: np.ndarray, posterior: Optional[np.ndarray] = None,
                   n_states: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Posterior-weighted mean normalised count per state and mark.

    Returns ``(profiles, weights)`` with ``profiles`` of shape ``(n_states, n_marks)``.
    Normalisation is the dataset's own size factors -- the same scale every in-process
    method sees -- so the profile of a state means the same thing for every tool.
    """
    W = _weights(np.asarray(states), posterior, n_states)
    Y = ds.X / np.maximum(ds.size_factors, 1e-12)
    mass = W.sum(axis=0)
    profiles = (W.T @ Y) / np.maximum(mass, 1e-12)[:, None]
    return profiles, W


def posthoc_scores(ds, states: np.ndarray, posterior: Optional[np.ndarray] = None,
                   n_states: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """``(bivalency_score, activity_score)`` per bin, from the tool's partition."""
    profiles, W = state_profiles(ds, states, posterior, n_states)
    return W @ _state_bivalency(profiles, ds.marks), W @ _state_activity(profiles, ds.marks)
