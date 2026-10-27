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


def run_epicseg_like(ds: ChromStateDataset, n_states: int, seed: int = 0, n_restarts: int = 3) -> MethodResult:
    """EpiCSeg's model: HMM with a negative multinomial emission (one dispersion per state).

    Counts across marks are coupled through a shared total, so the model captures
    mark-mark correlation that ChromState's independent NB does not, at the price of a
    single overdispersion for all marks (per state here, EpiCSeg's "independent" mode;
    its default shares one across all states) and no per-mark attribution.
    """
    t0 = time.time()
    seg = fit(ds, n_states=n_states, emission="negative_multinomial", seed=seed,
              n_restarts=n_restarts, adaptive_restarts=False)
    scores = _scores_from_posterior(seg.posterior, seg.model.emissions.mean_matrix(), ds.marks)
    return MethodResult(
        name="EpiCSeg-like (negative multinomial)",
        states=seg.states,
        posterior=seg.posterior,
        bivalency_score=scores["bivalency"],
        activity_score=scores["activity"],
        n_states=n_states,
        runtime_s=time.time() - t0,
        segmentation=seg,
        notes="counts, one dispersion per state shared by all marks; marks coupled",
    )


def run_episegmix_like(ds: ChromStateDataset, n_states: int, seed: int = 0, n_restarts: int = 3) -> MethodResult:
    """EpiSegMix's flexible count distributions, approximated by a zero-inflated NB HMM.

    EpiSegMix (Schmitz, Zentgraf & Rahmann, Bioinformatics 2024) chooses a count
    distribution per mark by likelihood (Poisson, negative binomial, Sichel, beta-NB, ...)
    and models segment lengths as negative binomial by expanding each state into a
    left-to-right sub-chain. This baseline keeps a flexible per-mark count distribution
    (zero-inflated NB) and the ordinary geometric-duration chain, so it isolates what a
    richer emission buys; the duration model is not reproduced and is noted as such.
    """
    t0 = time.time()
    seg = fit(ds, n_states=n_states, emission="zinb", seed=seed, n_restarts=n_restarts,
              adaptive_restarts=False)
    scores = _scores_from_posterior(seg.posterior, seg.model.emissions.mean_matrix(), ds.marks)
    return MethodResult(
        name="EpiSegMix-like (ZINB, geometric durations)",
        states=seg.states,
        posterior=seg.posterior,
        bivalency_score=scores["bivalency"],
        activity_score=scores["activity"],
        n_states=n_states,
        runtime_s=time.time() - t0,
        segmentation=seg,
        notes="zero-inflated NB per mark; explicit duration model not reproduced",
    )


def run_dihmm_like(ds: ChromStateDataset, n_states: int, seed: int = 0, n_restarts: int = 3,
                   n_domains: int = 3) -> MethodResult:
    """diHMM's two-level structure, flattened: bin states nested in domain states.

    The model is an HMM over ``n_domains x n_states`` flat states with emissions tied to
    the bin state, so each domain state owns its own transition matrix among bin states
    (the mechanism by which diHMM lets domain context shape bin-level calls). The reported
    bin-level posterior marginalises over domains; the domain posterior is kept in
    ``meta``. diHMM's actual implementation (Marco et al. 2017) uses ChromHMM-style
    binarised input and changes domain state only at the ends of 20-bin blocks; this
    baseline uses ChromState's count emission and lets the domain state change anywhere,
    so the comparison isolates the hierarchy itself.
    """
    from ..model.emissions import NegativeBinomialEmissions, TiedEmissions
    from ..model.hmm import HMM

    t0 = time.time()
    best = None
    for r in range(n_restarts):
        rng = np.random.default_rng(seed + r)
        base = NegativeBinomialEmissions.initialize(ds.X, ds.size_factors, n_states, rng)
        tied = TiedEmissions(base, n_domains)
        hmm = HMM(tied.n_states, tied, self_transition_prior=0.9)
        # Break the symmetry between domains: perturb each domain's block of transitions.
        A = hmm.transitions.copy()
        A *= rng.uniform(0.7, 1.3, size=A.shape)
        hmm.transitions = A / A.sum(axis=1, keepdims=True)
        res = hmm.fit(ds.X, ds.size_factors, ds.chains)
        if best is None or res.loglik > best[0]:
            best = (res.loglik, hmm)
    assert best is not None
    hmm = best[1]
    flat_post = hmm.posterior(ds.X, ds.size_factors, ds.chains)
    post, domain_post = hmm.emissions.fold_posterior(flat_post)
    states = np.argmax(post, axis=1).astype(np.int32)
    scores = _scores_from_posterior(post, hmm.emissions.base.mean_matrix(), ds.marks)
    return MethodResult(
        name=f"diHMM-like (hierarchical, {n_domains} domain states)",
        states=states,
        posterior=post,
        bivalency_score=scores["bivalency"],
        activity_score=scores["activity"],
        n_states=n_states,
        runtime_s=time.time() - t0,
        notes=f"flattened {n_domains}x{n_states}-state HMM with tied NB emissions",
        meta={"domain_posterior": domain_post, "loglik": best[0]},
    )


def run_normr_like(ds: ChromStateDataset, n_states: int, seed: int = 0, **_) -> MethodResult:
    """normR's regime calling, per mark, with no spatial model.

    normR (Helmuth et al.) fits a binomial mixture to ChIP versus control counts and
    calls each bin background or enriched (``enrichR``), or one of several enrichment
    regimes (``regimeR``). It is an enrichment caller, not a segmenter: it has no
    transition model and produces per-mark calls rather than joint states. This baseline
    fits a two-component NB mixture per mark (background / enriched) by EM, then forms
    combinatorial states from the per-mark enrichment probabilities with k-means to reach
    ``n_states``. It is included because joint enrichment of H3K4me3 and H3K27me3 from a
    caller like normR is a natural way to define bivalent regions, which is exactly what
    ChromState's bivalency score quantifies.
    """
    from sklearn.cluster import KMeans

    t0 = time.time()
    rng = np.random.default_rng(seed)
    Z = ds.X / np.maximum(ds.size_factors, EPS)
    P = np.empty_like(Z)
    for m in range(ds.n_marks):
        P[:, m] = _two_component_nb_posterior(ds.X[:, m], ds.size_factors[:, m], rng)
    # Combinatorial enrichment patterns, reduced to n_states by clustering.
    km = KMeans(n_clusters=n_states, n_init=5, random_state=seed).fit(P)
    labels = km.labels_.astype(np.int32)
    onehot = np.zeros((len(labels), n_states))
    onehot[np.arange(len(labels)), labels] = 1.0
    # State profile = mean normalised signal in the cluster, for the shared score formulas.
    means = np.stack([Z[labels == k].mean(axis=0) if (labels == k).any() else Z.mean(axis=0)
                      for k in range(n_states)])
    canon = [canonical_mark(m) for m in ds.marks]
    if "K4me3" in canon and "K27me3" in canon:
        # normR-style bivalency: joint enrichment probability of the two marks.
        biv = P[:, canon.index("K4me3")] * P[:, canon.index("K27me3")]
    else:
        biv = np.zeros(ds.n_rows)
    return MethodResult(
        name="normR-like (per-mark enrichment, no spatial model)",
        states=labels,
        posterior=onehot,
        bivalency_score=biv,
        activity_score=(P * np.log1p(means).mean(axis=0)[None, :]).sum(axis=1),
        n_states=n_states,
        runtime_s=time.time() - t0,
        notes="two-regime NB mixture per mark; combinatorial patterns clustered to K; no chain",
        meta={"enrichment_probability": P},
    )


def _two_component_nb_posterior(x: np.ndarray, offset: np.ndarray, rng: np.random.Generator,
                                n_iter: int = 50) -> np.ndarray:
    """P(enriched) per bin from a two-component NB mixture fitted by EM (shared dispersion)."""
    from scipy.special import gammaln

    scaled = x / np.maximum(offset, EPS)
    hi = scaled > np.quantile(scaled, 0.9)
    mu = np.array([max(scaled[~hi].mean(), 1e-3), max(scaled[hi].mean(), 1e-2)])
    pi = np.array([0.9, 0.1])
    var = scaled.var()
    r = float(np.clip(mu.mean() ** 2 / max(var - mu.mean(), 1e-6), 0.5, 1e3))
    resp = None
    for _ in range(n_iter):
        mean = np.maximum(offset[:, None] * mu[None, :], 1e-6)
        ll = (gammaln(x[:, None] + r) - gammaln(r) - gammaln(x[:, None] + 1.0)
              + r * np.log(r / (r + mean)) + x[:, None] * np.log(mean / (r + mean)))
        ll += np.log(np.maximum(pi, 1e-12))[None, :]
        ll -= ll.max(axis=1, keepdims=True)
        resp = np.exp(ll)
        resp /= resp.sum(axis=1, keepdims=True)
        pi = np.maximum(resp.mean(axis=0), 1e-6)
        mu = np.maximum((resp * x[:, None]).sum(axis=0) / np.maximum((resp * offset[:, None]).sum(axis=0), 1e-12), 1e-6)
    assert resp is not None
    return resp[:, int(np.argmax(mu))]


METHODS = {
    "chromstate": run_chromstate,
    "chromhmm_like": run_chromhmm_like,
    "segway_like": run_segway_like,
    "epicseg_like": run_epicseg_like,
    "episegmix_like": run_episegmix_like,
    "dihmm_like": run_dihmm_like,
    "normr_like": run_normr_like,
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
