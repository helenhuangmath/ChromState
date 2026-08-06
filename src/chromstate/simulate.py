"""Ground-truth simulator.

Real chromatin data has no answer key, so every claim about accuracy has to be made
against simulated data where the true state of every bin is known. This module generates
multi-mark count data from a known HMM with biologically-shaped state profiles, and
optionally applies a perturbation whose responding regions are recorded.

Three properties are deliberately built in, because they are the properties that
distinguish the methods being compared:

1. **Graded signal within a state.** Marks vary continuously, so a method that thresholds
   away magnitude loses real information rather than noise.
2. **True bivalency.** A designated state carries both H3K4me3 and H3K27me3 on the same
   bins, distinct from a mixture of active and repressed bins.
3. **A global mark shift under perturbation.** The treated condition loses H3K27me3
   genome-wide, which is what an EZH2 inhibitor does and what depth normalisation cannot
   see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dataset import ChromStateDataset, dataset_from_arrays

DEFAULT_MARKS = ("H3K4me3", "H3K4me1", "H3K27ac", "H3K36me3", "H3K27me3", "H3K9me3")

# Mean counts per 200 bp bin at unit depth, per state and mark.
# Rows follow DEFAULT_MARKS order.
DEFAULT_STATES: Dict[str, Sequence[float]] = {
    #          K4me3  K4me1  K27ac  K36me3  K27me3  K9me3
    "Quies":  (  0.3,   0.4,   0.3,    0.5,    0.6,    0.8),
    "TssA":   ( 28.0,   8.0,  22.0,    1.0,    0.5,    0.3),
    "EnhA":   (  2.5,  18.0,  16.0,    1.5,    0.8,    0.4),
    "Tx":     (  0.8,   3.0,   2.0,   20.0,    0.7,    0.5),
    "ReprPC": (  0.6,   1.5,   0.6,    0.6,   16.0,    1.5),
    "Het":    (  0.3,   0.5,   0.3,    0.5,    2.0,   14.0),
    "TssBiv": ( 11.0,   6.0,   3.0,    0.7,   12.0,    0.9),
}


@dataclass
class SimulationTruth:
    """Everything the simulator knows that a method is not told."""

    states: np.ndarray
    labels: List[str]
    marks: List[str]
    emission_means: np.ndarray
    transitions: np.ndarray
    dispersion: np.ndarray
    sample_labels: np.ndarray
    chrom_labels: np.ndarray
    treated: List[str] = field(default_factory=list)
    bivalent_state: Optional[int] = None
    bivalent_mask: Optional[np.ndarray] = None
    responder_mask: Optional[np.ndarray] = None
    blurred_mask: Optional[np.ndarray] = None
    depth_factors: Dict[Tuple[str, str], float] = field(default_factory=dict)
    global_mark_scale: Dict[Tuple[str, str], float] = field(default_factory=dict)
    spikein_counts: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def states_for(self, sample: str) -> np.ndarray:
        return self.states[self.sample_labels == sample]

    @property
    def n_states(self) -> int:
        return len(self.labels)


def _sticky_transitions(n_states: int, self_prob: float, rng: np.random.Generator) -> np.ndarray:
    A = rng.dirichlet(np.ones(n_states) * 2.0, size=n_states)
    A *= (1.0 - self_prob)
    np.fill_diagonal(A, 0.0)
    A /= np.maximum(A.sum(axis=1, keepdims=True), 1e-12)
    A *= (1.0 - self_prob)
    np.fill_diagonal(A, self_prob)
    return A


def _sample_path(A: np.ndarray, pi: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    path = np.empty(n, dtype=np.int32)
    path[0] = rng.choice(len(pi), p=pi)
    cum = np.cumsum(A, axis=1)
    u = rng.random(n)
    for t in range(1, n):
        path[t] = int(np.searchsorted(cum[path[t - 1]], u[t]))
    return path


def _nb_sample(mean: np.ndarray, dispersion: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Negative binomial via a gamma-Poisson mixture (keeps the mean exact)."""
    shape = np.broadcast_to(dispersion, mean.shape)
    scale = mean / np.maximum(shape, 1e-12)
    lam = rng.gamma(shape=np.maximum(shape, 1e-6), scale=np.maximum(scale, 1e-12))
    return rng.poisson(lam).astype(np.float64)


def simulate(
    n_bins: int = 6000,
    n_chroms: int = 3,
    samples: Sequence[str] = ("control", "treated"),
    marks: Sequence[str] = DEFAULT_MARKS,
    states: Optional[Dict[str, Sequence[float]]] = None,
    self_transition: float = 0.92,
    dispersion: float = 3.0,
    blur: float = 0.0,
    blur_weight: Tuple[float, float] = (0.25, 0.5),
    depth_factors: Optional[Dict[str, float]] = None,
    perturbation: Optional[dict] = None,
    treated: Optional[Sequence[str]] = None,
    seed: int = 0,
    bin_size: int = 200,
) -> Tuple[ChromStateDataset, SimulationTruth]:
    """Generate a multi-sample, multi-mark dataset with known state labels.

    Parameters
    ----------
    n_bins
        Bins per chromosome per sample.
    blur
        Fraction of bins whose emission profile is a convex mixture of their own state
        and one other state, with mixing weight drawn from ``blur_weight``.

        This matters for the benchmark's credibility. Data generated cleanly from a
        7-state HMM is separable enough that *every* probabilistic method reports ~0.99
        posterior confidence, so calibration metrics cannot distinguish an honest model
        from an overconfident one. Real chromatin is not separable: promoters shade into
        flanking regions, enhancers into weak enhancers, and a bin can genuinely sit
        between states. Blurring recreates that, producing bins where high confidence is
        *wrong* and a well-calibrated method should hedge.
    perturbation
        Controls how the second and later samples differ from the first. Keys:

        ``responder_fraction``
            Fraction of bins in ``from_state`` that switch to ``to_state``.
        ``from_state`` / ``to_state``
            State names for the switch. Default moves bivalent bins to active promoters,
            mimicking a drug releasing poised genes.
        ``global_mark_loss``
            Mapping of mark name to the fraction of signal *retained* genome-wide, e.g.
            ``{"H3K27me3": 0.35}`` for an EZH2 inhibitor. This is the effect that
            depth-normalised methods are structurally unable to detect.

    Returns
    -------
    ``(dataset, truth)``. The dataset carries only what a real analysis would have; the
    truth object carries the answer key.
    """
    rng = np.random.default_rng(seed)
    states = dict(states if states is not None else DEFAULT_STATES)
    labels = list(states.keys())
    marks = list(marks)
    K, M = len(labels), len(marks)

    emis = np.array([list(states[lbl]) for lbl in labels], dtype=np.float64)
    if emis.shape[1] != M:
        raise ValueError(f"State profiles have {emis.shape[1]} values but {M} marks were given")

    A = _sticky_transitions(K, self_transition, rng)
    pi = np.full(K, 1.0 / K)
    disp = np.full(M, float(dispersion))

    perturbation = perturbation or {}
    frac = float(perturbation.get("responder_fraction", 0.5))
    from_state = perturbation.get("from_state", "TssBiv")
    to_state = perturbation.get("to_state", "TssA")
    global_loss: Dict[str, float] = dict(perturbation.get("global_mark_loss", {}))

    samples = list(samples)
    depth_factors = depth_factors or {}

    # Which samples receive the perturbation. The default -- everything after the first --
    # is right for a one-versus-one contrast and wrong for anything with replicates: it
    # silently turns the second and third "control" into treated samples, so a
    # within-condition noise floor measured on them is really a between-condition
    # comparison and comes back far too high.
    if treated is None:
        treated_set = set(samples[1:])
    else:
        treated_set = set(treated)
        unknown = treated_set - set(samples)
        if unknown:
            raise ValueError(f"treated names sample(s) not in the dataset: {sorted(unknown)}")

    # A single shared underlying state path per chromosome, so that bins are matched
    # across samples and the paired comparison is meaningful.
    base_paths = [_sample_path(A, pi, n_bins, rng) for _ in range(n_chroms)]
    base_path = np.concatenate(base_paths)

    from_idx = labels.index(from_state) if from_state in labels else None
    to_idx = labels.index(to_state) if to_state in labels else None
    responder = np.zeros(len(base_path), dtype=bool)
    if from_idx is not None and frac > 0:
        candidates = np.flatnonzero(base_path == from_idx)
        n_resp = int(round(frac * len(candidates)))
        if n_resp:
            responder[rng.choice(candidates, size=n_resp, replace=False)] = True

    # Blur assignment is shared across samples so that bins stay matched between
    # conditions and paired comparisons remain valid.
    n_base = len(base_path)
    blurred = np.zeros(n_base, dtype=bool)
    blur_partner = np.zeros(n_base, dtype=np.int32)
    blur_w = np.zeros(n_base)
    if blur > 0:
        sel = rng.random(n_base) < blur
        blurred = sel
        blur_partner[sel] = rng.integers(0, K, size=int(sel.sum()))
        blur_w[sel] = rng.uniform(blur_weight[0], blur_weight[1], size=int(sel.sum()))

    X_blocks, state_blocks, sample_blocks, chrom_blocks = [], [], [], []
    truth_depth: Dict[Tuple[str, str], float] = {}
    truth_scale: Dict[Tuple[str, str], float] = {}
    spikein: Dict[Tuple[str, str], float] = {}

    for sample in samples:
        path = base_path.copy()
        is_treated = sample in treated_set
        if is_treated and to_idx is not None:
            path[responder] = to_idx

        mean = emis[path].copy()  # (n_rows, M)
        if blur > 0 and blurred.any():
            w = blur_w[blurred][:, None]
            mean[blurred] = (1.0 - w) * mean[blurred] + w * emis[blur_partner[blurred]]

        scale = np.ones(M)
        if is_treated:
            for mk, retained in global_loss.items():
                if mk in marks:
                    scale[marks.index(mk)] = float(retained)
        mean *= scale[None, :]

        depth = np.array(
            [depth_factors.get(f"{sample}:{mk}", rng.uniform(0.7, 1.4)) for mk in marks]
        )
        mean *= depth[None, :]

        X_blocks.append(_nb_sample(mean, disp, rng))
        state_blocks.append(path)
        sample_blocks.append(np.full(len(path), sample, dtype=object))
        chrom_blocks.append(
            np.concatenate([np.full(n_bins, f"chr{c + 1}", dtype=object) for c in range(n_chroms)])
        )

        for j, mk in enumerate(marks):
            truth_depth[(sample, mk)] = float(depth[j])
            truth_scale[(sample, mk)] = float(scale[j])
            # Spike-in reads track technical depth only: the endogenous abundance cancels
            # out of the ratio (see the derivation in chromstate.norm.compute_size_factors).
            # This is why spike-in normalisation leaves a genome-wide mark loss visible
            # while depth-based normalisation absorbs it.
            spikein[(sample, mk)] = float(depth[j]) * 1e4

    X = np.vstack(X_blocks)
    true_states = np.concatenate(state_blocks)
    sample_labels = np.concatenate(sample_blocks)
    chrom_labels = np.concatenate(chrom_blocks)

    ds = dataset_from_arrays(
        X=X,
        marks=marks,
        sample_labels=sample_labels,
        chrom_labels=chrom_labels,
        bin_size=bin_size,
        is_count=True,
    )

    biv_idx = labels.index("TssBiv") if "TssBiv" in labels else None
    truth = SimulationTruth(
        states=true_states,
        labels=labels,
        marks=marks,
        emission_means=emis,
        transitions=A,
        dispersion=disp,
        sample_labels=sample_labels,
        chrom_labels=chrom_labels,
        treated=[s for s in samples if s in treated_set],
        bivalent_state=biv_idx,
        bivalent_mask=None if biv_idx is None else (true_states == biv_idx),
        responder_mask=np.tile(responder, len(samples)),
        blurred_mask=np.tile(blurred, len(samples)),
        depth_factors=truth_depth,
        global_mark_scale=truth_scale,
        spikein_counts=spikein,
    )
    return ds, truth


def write_demo_files(outdir: str, seed: int = 0, n_bins: int = 2000) -> str:
    """Write a small simulated dataset as BED read files plus a ChromHMM-style manifest.

    Exercises the real file-reading path end to end, so the IO layer is covered by the
    demo rather than only by unit tests. Returns the manifest path.
    """
    import os

    os.makedirs(outdir, exist_ok=True)
    ds, truth = simulate(n_bins=n_bins, n_chroms=2, seed=seed)
    bin_size = 200
    rng = np.random.default_rng(seed + 99)

    lines: List[str] = []
    for sample in ds.samples:
        mask = ds.sample_mask(sample)
        bins = ds.bins.loc[mask].reset_index(drop=True)
        counts = ds.X[mask]
        for j, mark in enumerate(ds.marks):
            path = os.path.join(outdir, f"{sample}_{mark}.bed")
            with open(path, "w") as fh:
                for i in range(len(bins)):
                    n = int(counts[i, j])
                    if n == 0:
                        continue
                    chrom = bins.at[i, "chrom"]
                    start0 = int(bins.at[i, "start"])
                    # Place reads uniformly inside the bin; the reader's shift is 0 here
                    # because the simulated reads are already fragment centres.
                    pos = start0 + rng.integers(0, bin_size, size=n)
                    for p in pos:
                        fh.write(f"{chrom}\t{p}\t{p + 1}\tr\t0\t+\n")
            lines.append(f"{sample}\t{mark}\t{os.path.basename(path)}\t\tbed")

    sizes_path = os.path.join(outdir, "chrom.sizes")
    with open(sizes_path, "w") as fh:
        for chrom in sorted(set(ds.bins["chrom"])):
            n = int((ds.bins["chrom"] == chrom).sum() // len(ds.samples))
            fh.write(f"{chrom}\t{n * bin_size}\n")

    manifest = os.path.join(outdir, "cellmarkfiletable.txt")
    with open(manifest, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return manifest
