"""Competing methods, run on identical bins so the comparison is fair.

Each baseline is the *statistical model* of the tool it stands for, reimplemented on
ChromState's binning rather than shelled out to the original software. That buys three
things: the comparison runs anywhere without Java or a conda solve, every method sees
byte-identical input, and the differences that show up are attributable to the model
rather than to preprocessing.

The tradeoff is stated plainly: these are faithful reimplementations of the published
emission models, not the tools themselves, and they do not reproduce tool-specific
heuristics. Before publishing, the ChromHMM row must be re-run against real ChromHMM --
:func:`chromstate.benchmark.baselines.export_for_chromhmm` writes the binarised input in
ChromHMM's own format for exactly that purpose.

Every HMM-based method runs with ``adaptive_restarts=False`` so that all of them get the
*same fixed* number of random initialisations. ChromState's normal default keeps
restarting until an optimum is reached twice, which is the right behaviour for analysis
but would hand it a larger compute budget than the baselines here and inflate the
comparison.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..dataset import ChromStateDataset
from ..model.annotate import canonical_mark
from ..model.binarize import poisson_binarize
from ..model.core import Segmentation, fit

EPS = 1e-12


@dataclass
class MethodResult:
    """Uniform output every method must produce, so metrics can be applied identically."""

    name: str
    states: np.ndarray
    posterior: Optional[np.ndarray]
    bivalency_score: np.ndarray
    activity_score: np.ndarray
    n_states: int
    runtime_s: float
    segmentation: Optional[Segmentation] = None
    notes: str = ""
    meta: Dict[str, object] = field(default_factory=dict)


def _state_bivalency(mean_matrix: np.ndarray, marks: Sequence[str]) -> np.ndarray:
    """Bivalency of each state from its emission profile: harmonic mean of the two marks.

    Identical formula for every method, so no method is advantaged by the definition.
    """
    canon = [canonical_mark(m) for m in marks]
    if "K4me3" not in canon or "K27me3" not in canon:
        return np.zeros(mean_matrix.shape[0])
    a = np.log1p(np.maximum(mean_matrix[:, canon.index("K4me3")], 0))
    r = np.log1p(np.maximum(mean_matrix[:, canon.index("K27me3")], 0))
    return 2.0 * a * r / np.maximum(a + r, EPS)


def _state_activity(mean_matrix: np.ndarray, marks: Sequence[str]) -> np.ndarray:
    canon = [canonical_mark(m) for m in marks]
    active = [j for j, c in enumerate(canon) if c in ("K4me3", "K4me1", "K27ac", "K9ac", "ATAC")]
    if not active:
        return mean_matrix.sum(axis=1)
    return np.log1p(mean_matrix[:, active]).mean(axis=1)


def _scores_from_posterior(
    posterior: np.ndarray, mean_matrix: np.ndarray, marks: Sequence[str]
) -> Dict[str, np.ndarray]:
    return {
        "bivalency": posterior @ _state_bivalency(mean_matrix, marks),
        "activity": posterior @ _state_activity(mean_matrix, marks),
    }


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


def run_chromstate(ds: ChromStateDataset, n_states: int, seed: int = 0, n_restarts: int = 3) -> MethodResult:
    """ChromState: joint negative-binomial HMM on raw counts."""
    t0 = time.time()
    seg = fit(ds, n_states=n_states, emission="nb", seed=seed, n_restarts=n_restarts,
              adaptive_restarts=False)
    scores = _scores_from_posterior(seg.posterior, seg.model.emissions.mean_matrix(), ds.marks)
    return MethodResult(
        name="ChromState (NB-HMM)",
        states=seg.states,
        posterior=seg.posterior,
        bivalency_score=scores["bivalency"],
        activity_score=scores["activity"],
        n_states=n_states,
        runtime_s=time.time() - t0,
        segmentation=seg,
        notes="counts modelled directly; magnitude retained",
    )


def run_chromhmm_like(
    ds: ChromStateDataset, n_states: int, seed: int = 0, n_restarts: int = 3, p_threshold: float = 1e-4
) -> MethodResult:
    """ChromHMM's model: Poisson binarisation followed by a multivariate Bernoulli HMM."""
    t0 = time.time()
    B = poisson_binarize(ds.X, control=ds.control, p_threshold=p_threshold).astype(np.float64)
    ds_bin = ChromStateDataset(
        X=B,
        marks=list(ds.marks),
        bins=ds.bins.copy(),
        chains=list(ds.chains),
        size_factors=np.ones_like(B),
        is_count=True,
        binset=ds.binset,
    )
    seg = fit(ds_bin, n_states=n_states, emission="bernoulli", seed=seed, n_restarts=n_restarts,
              adaptive_restarts=False)

    # Emission means are presence probabilities; map them back onto the count scale using
    # each mark's mean count among present bins, so the shared bivalency formula applies.
    present_mean = np.array(
        [ds.X[B[:, j] > 0, j].mean() if (B[:, j] > 0).any() else 0.0 for j in range(ds.n_marks)]
    )
    pseudo_means = seg.model.emissions.mean_matrix() * present_mean[None, :]
    scores = _scores_from_posterior(seg.posterior, pseudo_means, ds.marks)

    frac_present = float((B > 0).mean())
    return MethodResult(
        name="ChromHMM-like (Bernoulli)",
        states=seg.states,
        posterior=seg.posterior,
        bivalency_score=scores["bivalency"],
        activity_score=scores["activity"],
        n_states=n_states,
        runtime_s=time.time() - t0,
        segmentation=seg,
        notes=f"signal binarised at p<{p_threshold:g}; {frac_present:.1%} of bin-marks called present",
        meta={"frac_present": frac_present, "binarized": B},
    )


def run_segway_like(ds: ChromStateDataset, n_states: int, seed: int = 0, n_restarts: int = 3) -> MethodResult:
    """Segway's flavour: diagonal Gaussian HMM on log-transformed signal."""
    t0 = time.time()
    seg = fit(ds, n_states=n_states, emission="gaussian", seed=seed, n_restarts=n_restarts,
              adaptive_restarts=False)
    scores = _scores_from_posterior(seg.posterior, seg.model.emissions.mean_matrix(), ds.marks)
    return MethodResult(
        name="Segway-like (Gaussian)",
        states=seg.states,
        posterior=seg.posterior,
        bivalency_score=scores["bivalency"],
        activity_score=scores["activity"],
        n_states=n_states,
        runtime_s=time.time() - t0,
        segmentation=seg,
        notes="continuous signal, Gaussian on log scale",
    )


def run_gmm(ds: ChromStateDataset, n_states: int, seed: int = 0, **_) -> MethodResult:
    """Gaussian mixture with no spatial model: isolates what the HMM's chain contributes."""
    from sklearn.mixture import GaussianMixture

    t0 = time.time()
    Z = np.log1p(ds.X / np.maximum(ds.size_factors, EPS))
    gmm = GaussianMixture(
        n_components=n_states, covariance_type="diag", random_state=seed, max_iter=200
    ).fit(Z)
    post = gmm.predict_proba(Z)
    means = np.expm1(gmm.means_).clip(min=0)
    scores = _scores_from_posterior(post, means, ds.marks)
    return MethodResult(
        name="GMM (no spatial model)",
        states=np.argmax(post, axis=1).astype(np.int32),
        posterior=post,
        bivalency_score=scores["bivalency"],
        activity_score=scores["activity"],
        n_states=n_states,
        runtime_s=time.time() - t0,
        notes="bins treated as independent; no domain structure",
    )


def run_kmeans(ds: ChromStateDataset, n_states: int, seed: int = 0, **_) -> MethodResult:
    """k-means on log signal: hard labels only, no probability at all."""
    from sklearn.cluster import KMeans

    t0 = time.time()
    Z = np.log1p(ds.X / np.maximum(ds.size_factors, EPS))
    km = KMeans(n_clusters=n_states, random_state=seed, n_init=10).fit(Z)
    labels = km.labels_.astype(np.int32)
    onehot = np.zeros((len(labels), n_states))
    onehot[np.arange(len(labels)), labels] = 1.0
    means = np.expm1(km.cluster_centers_).clip(min=0)
    scores = _scores_from_posterior(onehot, means, ds.marks)
    return MethodResult(
        name="k-means",
        states=labels,
        posterior=onehot,
        bivalency_score=scores["bivalency"],
        activity_score=scores["activity"],
        n_states=n_states,
        runtime_s=time.time() - t0,
        notes="hard assignment; posterior is one-hot by construction",
    )


def run_naive_signal(ds: ChromStateDataset, n_states: int, **_) -> MethodResult:
    """No model at all: bivalency straight from the two marks.

    The control that matters. If a fitted model cannot beat the harmonic mean of two
    normalised signal tracks at finding bivalent regions, the model is not earning its
    complexity.
    """
    t0 = time.time()
    canon = [canonical_mark(m) for m in ds.marks]
    Z = np.log1p(ds.X / np.maximum(ds.size_factors, EPS))
    if "K4me3" in canon and "K27me3" in canon:
        a, r = Z[:, canon.index("K4me3")], Z[:, canon.index("K27me3")]
        biv = 2.0 * a * r / np.maximum(a + r, EPS)
    else:
        biv = np.zeros(ds.n_rows)
    return MethodResult(
        name="Naive signal (no model)",
        states=np.zeros(ds.n_rows, dtype=np.int32),
        posterior=None,
        bivalency_score=biv,
        activity_score=Z.mean(axis=1),
        n_states=1,
        runtime_s=time.time() - t0,
        notes="reference point: two tracks, one formula, no states",
    )


METHODS = {
    "chromstate": run_chromstate,
    "chromhmm_like": run_chromhmm_like,
    "segway_like": run_segway_like,
    "gmm": run_gmm,
    "kmeans": run_kmeans,
    "naive": run_naive_signal,
}


def export_for_chromhmm(ds: ChromStateDataset, outdir: str, p_threshold: float = 1e-4) -> List[str]:
    """Write ChromHMM-format binarised files so real ChromHMM can be run on the same bins.

    Produces one ``<sample>_<chrom>_binary.txt`` per sample and chromosome, in the format
    ``LearnModel`` consumes. Use this to replace the reimplemented ChromHMM row with the
    genuine article before publication.
    """
    import os

    os.makedirs(outdir, exist_ok=True)
    B = poisson_binarize(ds.X, control=ds.control, p_threshold=p_threshold)
    written: List[str] = []
    for sample in ds.samples:
        smask = ds.sample_mask(sample)
        sub_bins = ds.bins.loc[smask]
        sub_B = B[smask]
        for chrom in sub_bins["chrom"].unique():
            cmask = (sub_bins["chrom"] == chrom).to_numpy()
            path = os.path.join(outdir, f"{sample}_{chrom}_binary.txt")
            with open(path, "w") as fh:
                fh.write(f"{sample}\t{chrom}\n")
                fh.write("\t".join(ds.marks) + "\n")
                for row in sub_B[cmask]:
                    fh.write("\t".join(str(int(v)) for v in row) + "\n")
            written.append(path)
    return written
