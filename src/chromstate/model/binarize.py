"""ChromHMM-compatible binarisation, reproduced for benchmarking.

ChromHMM converts each bin to present/absent per mark using a Poisson tail test: with a
background rate estimated from the genome-wide average (optionally from an input control
in a local window), a bin is called present when the probability of seeing at least that
many reads by chance falls below ``p_threshold`` (ChromHMM's default is 1e-4).

Reimplemented here so the benchmark can run ChromHMM's *model* on identical bins without
requiring the Java tool, and so the information lost at this step can be measured
directly (see :mod:`chromstate.benchmark`).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import poisson


def poisson_threshold(rate: float, p_threshold: float = 1e-4, max_count: int = 10_000) -> int:
    """Smallest count ``c`` with ``P(X >= c) < p_threshold`` under ``Poisson(rate)``."""
    if rate <= 0:
        return 1
    c = int(poisson.isf(p_threshold, rate)) + 1
    return int(np.clip(c, 1, max_count))


def poisson_binarize(
    X: np.ndarray,
    control: Optional[np.ndarray] = None,
    p_threshold: float = 1e-4,
    control_window_bins: int = 25,
    pseudocount: float = 1.0,
) -> np.ndarray:
    """Binarise a ``(T, M)`` count matrix the way ChromHMM does.

    Parameters
    ----------
    control
        Optional matching input/IgG counts. When given, the background rate is local:
        the control counts are smoothed over ``control_window_bins`` and scaled to the
        signal library, so that copy-number and mappability structure is absorbed.
    p_threshold
        Poisson tail probability defining "present". ChromHMM's default is 1e-4.

    Returns
    -------
    ``(T, M)`` uint8 array of 0/1 calls.
    """
    X = np.asarray(X, dtype=np.float64)
    T, M = X.shape
    out = np.zeros((T, M), dtype=np.uint8)

    for m in range(M):
        if control is None:
            rate = X[:, m].mean()
            thresh = poisson_threshold(rate, p_threshold)
            out[:, m] = (X[:, m] >= thresh).astype(np.uint8)
        else:
            local = _smooth(control[:, m], control_window_bins) + pseudocount
            scale = X[:, m].sum() / max(local.sum(), 1e-9)
            rates = local * scale
            # Vectorised tail test: present when P(X >= x) < p under the local rate.
            tail = poisson.sf(X[:, m] - 1, np.maximum(rates, 1e-9))
            out[:, m] = ((tail < p_threshold) & (X[:, m] > 0)).astype(np.uint8)
    return out


def _smooth(v: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average with edge padding."""
    if window <= 1:
        return v.astype(np.float64)
    half = window // 2
    padded = np.pad(v.astype(np.float64), half, mode="edge")
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    return np.convolve(padded, kernel, mode="valid")[: len(v)]


def binarization_information_loss(X: np.ndarray, B: np.ndarray) -> dict:
    """Quantify what binarisation discarded.

    Returns the fraction of bins called present, and -- among present bins -- the
    dynamic range of the underlying counts that were collapsed to the single value 1.
    A large ratio here is the direct, measurable statement of ChromHMM's core limitation.
    """
    stats = {}
    for m in range(X.shape[1]):
        present = B[:, m] > 0
        vals = X[present, m]
        stats[m] = {
            "frac_present": float(present.mean()),
            "n_present": int(present.sum()),
            "min_count_present": float(vals.min()) if vals.size else np.nan,
            "max_count_present": float(vals.max()) if vals.size else np.nan,
            "fold_range_collapsed": (
                float(vals.max() / max(vals.min(), 1.0)) if vals.size else np.nan
            ),
            "cv_within_present": float(vals.std() / vals.mean()) if vals.size and vals.mean() else np.nan,
        }
    return stats
