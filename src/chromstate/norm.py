"""Normalisation: turning read counts into quantities comparable across samples.

This is the module the word *quantitative* in ChromState actually rests on.

ChIP-seq and CUT&RUN produce **compositional** data: a library reports the relative
distribution of reads, not the absolute amount of a mark. If an EZH2 inhibitor removes
80% of H3K27me3 genome-wide, depth normalisation will rescale the survivors right back
up and the treated sample will look almost unchanged. Every method that normalises to
total depth -- ChromHMM's binarisation included -- is blind to exactly the global
redistribution that epigenetic drugs cause.

Three estimators are provided, in increasing order of trustworthiness:

``depth``
    Total counts per (sample, mark). Fast, and wrong under global change.
``median_ratio``
    DESeq2's median-of-ratios. Robust to *local* differences, but it assumes most bins
    do not change -- the same blind spot as ``depth``, just better hidden.
``spikein``
    Scale from an exogenous reference (Drosophila S2 chromatin, barcoded nucleosomes).
    This is the only estimator here that survives global mark loss, and it is the one to
    use whenever the experiment provides it.

The chosen factors enter the model as *offsets on the emission mean*, never as a
division of the data, so the negative-binomial likelihood keeps operating on integer
counts and the mean-variance relationship stays intact.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .dataset import ChromStateDataset

METHODS = ("depth", "median_ratio", "spikein", "none")


def compute_size_factors(
    ds: ChromStateDataset,
    method: str = "median_ratio",
    spikein: Optional[Dict[Tuple[str, str], float]] = None,
    reference: Optional[str] = None,
    window_bins: int = 50,
    min_count: float = 5.0,
) -> np.ndarray:
    """Compute a ``(n_samples, n_marks)`` table of size factors.

    Parameters
    ----------
    method
        One of ``depth``, ``median_ratio``, ``spikein``, ``none``.
    spikein
        Required for ``method="spikein"``: maps ``(sample, mark)`` to the number of
        reads assigned to the exogenous reference genome.
    reference
        Sample whose factor is fixed to 1.0. Defaults to the first sample, so factors
        read as "relative to control".
    window_bins
        Median-of-ratios only. Counts are pooled into windows of this many bins before
        the ratio is taken. At 200 bp most bins hold 0-2 reads, so per-bin log ratios
        take a handful of discrete values and their median collapses onto exactly 0 --
        returning a size factor of precisely 1.0 no matter what the data says. Pooling to
        ~10 kb restores a continuous ratio distribution.
    min_count
        Median-of-ratios only. Windows below this count in any sample are excluded, since
        low-count ratios are dominated by sampling noise.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown normalisation method {method!r}; expected one of {METHODS}")

    samples = ds.samples
    n_s, n_m = len(samples), ds.n_marks
    factors = np.ones((n_s, n_m))

    if method == "none":
        return factors

    if method == "spikein":
        if spikein is None:
            raise ValueError("method='spikein' requires the spikein= mapping of reference counts")
        # Why the size factor is *proportional* to spike-in reads:
        #
        # A fixed physical amount of exogenous chromatin S is added to every sample. With
        # endogenous material B_s (the biology we want to measure) and T_s total reads,
        #     endogenous_reads = T_s * B_s / (B_s + S),  spikein_reads = T_s * S / (B_s + S).
        # Sequencing depth enters as T_s = d_s * (B_s + S), so spikein_reads = d_s * S:
        # the spike-in count tracks *technical depth alone*, with the biology cancelled.
        #
        # That is exactly what a size factor must absorb. Setting the offset proportional
        # to spike-in reads removes depth and leaves the genome-wide change in B_s in the
        # data, where the model can see it. Depth and median-of-ratios instead estimate
        # something proportional to d_s * B_s and absorb the biology along with the noise.
        for i, s in enumerate(samples):
            for j, mark in enumerate(ds.marks):
                if (s, mark) not in spikein:
                    raise KeyError(f"No spike-in count for sample {s!r}, mark {mark!r}")
                ref = float(spikein[(s, mark)])
                if ref <= 0:
                    raise ValueError(f"Non-positive spike-in count for ({s}, {mark})")
                factors[i, j] = ref
    else:
        totals = np.zeros((n_s, n_m))
        for i, s in enumerate(samples):
            m = ds.sample_mask(s)
            totals[i] = ds.X[m].sum(axis=0)

        if method == "depth":
            factors = totals.copy()
        else:  # median_ratio
            n_bins_per_sample = [int(ds.sample_mask(s).sum()) for s in samples]
            if len(set(n_bins_per_sample)) != 1:
                raise ValueError(
                    "median_ratio needs the same number of bins in every sample; "
                    f"got {dict(zip(samples, n_bins_per_sample))}. Use method='depth'."
                )
            for j in range(n_m):
                col = np.array([ds.X[ds.sample_mask(s), j] for s in samples])
                pooled = _pool_windows(col, window_bins)
                usable = (pooled >= min_count).all(axis=0)
                if usable.sum() < 20:
                    factors[:, j] = totals[:, j]
                    continue
                log_col = np.log(pooled[:, usable])
                ref_row = log_col.mean(axis=0)  # geometric mean across samples
                for i in range(n_s):
                    factors[i, j] = np.exp(np.median(log_col[i] - ref_row))

    ref_idx = samples.index(reference) if reference is not None else 0
    for j in range(n_m):
        denom = factors[ref_idx, j]
        if denom > 0:
            factors[:, j] /= denom
    factors[factors <= 0] = 1.0
    return factors


def _pool_windows(col: np.ndarray, window_bins: int) -> np.ndarray:
    """Sum ``(n_samples, n_bins)`` counts into non-overlapping windows of ``window_bins``."""
    if window_bins <= 1:
        return col
    n_s, n_b = col.shape
    n_win = n_b // window_bins
    if n_win < 1:
        return col
    trimmed = col[:, : n_win * window_bins]
    return trimmed.reshape(n_s, n_win, window_bins).sum(axis=2)


def apply_size_factors(ds: ChromStateDataset, factors: np.ndarray) -> ChromStateDataset:
    """Broadcast a ``(n_samples, n_marks)`` factor table onto the per-row offsets."""
    samples = ds.samples
    if factors.shape != (len(samples), ds.n_marks):
        raise ValueError(f"factors must have shape ({len(samples)}, {ds.n_marks})")
    offsets = np.ones((ds.n_rows, ds.n_marks))
    for i, s in enumerate(samples):
        offsets[ds.sample_mask(s)] = factors[i]
    ds.size_factors = offsets
    ds.meta["size_factors"] = {
        s: dict(zip(ds.marks, factors[i].tolist())) for i, s in enumerate(samples)
    }
    ds.meta["norm_applied"] = True
    return ds


def normalize(
    ds: ChromStateDataset,
    method: str = "median_ratio",
    spikein: Optional[Dict[Tuple[str, str], float]] = None,
    reference: Optional[str] = None,
    **kwargs,
) -> ChromStateDataset:
    """Compute and apply size factors in one step (modifies and returns ``ds``)."""
    factors = compute_size_factors(
        ds, method=method, spikein=spikein, reference=reference, **kwargs
    )
    ds.meta["norm_method"] = method
    if method in ("depth", "median_ratio") and len(ds.samples) > 1:
        ds.meta["norm_warning"] = (
            f"{method} normalisation assumes no global change in mark abundance between "
            "samples. If this experiment involves a perturbation expected to change a "
            "mark genome-wide (EZH2i, DNMTi, LSD1i), use spike-in normalisation instead "
            "-- otherwise the global effect will be normalised away."
        )
    return apply_size_factors(ds, factors)


def control_ratio(ds: ChromStateDataset, pseudocount: float = 1.0) -> Optional[np.ndarray]:
    """Signal-over-input ratio, or None when the manifest carried no control files.

    Provided as a diagnostic and as an input to Gaussian emissions. The negative-binomial
    path prefers to keep raw counts and use the control as an offset instead, since
    ratios destroy the count scale.
    """
    if ds.control is None:
        return None
    ctrl = ds.control.copy()
    scale = ds.X.sum(axis=0) / np.maximum(ctrl.sum(axis=0), 1.0)
    return (ds.X + pseudocount) / (ctrl * scale + pseudocount)


def log1p_signal(ds: ChromStateDataset) -> np.ndarray:
    """Depth-adjusted ``log1p`` signal, for Gaussian emissions and for embedding."""
    return np.log1p(ds.X / np.maximum(ds.size_factors, 1e-12))
