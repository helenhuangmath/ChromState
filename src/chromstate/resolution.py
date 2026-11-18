"""Multi-resolution quantification: fit coarse, score fine, on a 0-100 scale.

Fitting a hidden Markov model costs time in proportion to the number of bins, and a
10 bp tiling of a mammalian genome has 270 million of them per sample. Nothing about the
*parameters* needs that resolution: an emission mean is a rate per base, and a
transition matrix is a rate per base too. ChromState therefore fits the model once at a
coarse bin width (200 bp by default, the ChromHMM convention) and then **scores every
fine bin with the same model**, rescaled to the fine width, in one forward-backward
pass. The fine pass costs about as much as one EM iteration, not fifty, and the states
mean exactly the same thing at both widths because they are the same states.

How the rescaling works, so the result can be trusted:

**Negative binomial.** If a 200 bp bin's count is ``NB(mu, r)`` and reads land
independently, each of its 20 disjoint 10 bp pieces is ``NB(mu/20, r/20)``: the sum of
independent negative binomials with a common success probability is negative binomial
with the size parameters added. Means and dispersions are therefore divided by the
width ratio, exactly.

**Transitions.** A Markov chain observed every 200 bp with matrix ``A`` and observed
every 10 bp has matrix ``A^(1/20)``, the fractional matrix power. It is computed by
eigendecomposition. A stochastic matrix with near-zero entries need not have an exact
stochastic root (the embedding problem), so small negative entries of the root are
clipped and the rows renormalised; re-powering then reproduces the fitted matrix to
within about 0.005. If the decomposition fails altogether the self-transition is rooted
directly and the off-diagonal mass redistributed in proportion.

**Gaussian (coverage) and Bernoulli (presence) emissions.** Mean coverage has the same
expectation at any width, so Gaussian parameters are kept; a presence probability ``p``
over 200 bp becomes ``1 - (1-p)^(1/20)`` over 10 bp.

Scores are the posterior probability of each state in each fine bin, written on a
0-100 scale as integers (:func:`state_scores`). A bin at 100 is wholly in that state; a
bin at 35 carries 35% of its identity there. They are not thresholds, do not need a
peak caller, and can be averaged over any region.

    coarsen_dataset        fine bins -> wider bins (sum counts, mean coverage)
    rescale_hmm            fitted model at one width -> the same model at another
    decode_at_resolution   coarse Segmentation + fine dataset -> fine Segmentation
    iter_decoded           the same, one chain at a time (genome-scale streaming)
    state_scores           0-100 per-state scores per fine bin
    write_bedgraph         one state's score track for a genome browser
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import replace
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .dataset import ChromStateDataset, _chains_from_runs
from .genome import BinSet
from .model.core import Segmentation
from .model.hmm import HMM, forward_backward, viterbi

DEFAULT_RESOLUTION = 10
DEFAULT_FIT_BIN_SIZE = 200
EPS = 1e-12


# ---------------------------------------------------------------------------
# Datasets at two widths
# ---------------------------------------------------------------------------


def bin_width(ds: ChromStateDataset) -> int:
    """Nominal bin width of a dataset (from its BinSet, else from its first bin)."""
    if ds.binset is not None:
        return int(ds.binset.bin_size)
    return int(ds.bins["end"].iloc[0] - ds.bins["start"].iloc[0])


def coarsen_dataset(
    ds: ChromStateDataset, factor: int, how: Optional[str] = None
) -> ChromStateDataset:
    """Merge every ``factor`` consecutive fine bins of a chain into one coarse bin.

    Counts are summed; coverage or peak scores (``is_count=False``) are averaged, unless
    ``how`` says otherwise. Blocks never cross a chain boundary (chromosome, sample, or
    a blacklist gap), and a chain whose length is not a multiple of ``factor`` ends in a
    shorter block whose ``end`` coordinate records its true extent. Size factors are
    constant within a (sample, mark) so the coarse bin inherits them unchanged.
    """
    factor = int(factor)
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if factor == 1:
        return ds
    how = how or ("sum" if ds.is_count else "mean")
    if how not in ("sum", "mean"):
        raise ValueError(f"how must be 'sum' or 'mean', got {how!r}")

    starts: List[np.ndarray] = []
    lengths: List[np.ndarray] = []
    for sl in ds.chains:
        s = np.arange(sl.start, sl.stop, factor, dtype=np.int64)
        starts.append(s)
        lengths.append(np.minimum(s + factor, sl.stop) - s)
    first = np.concatenate(starts)
    n_in_block = np.concatenate(lengths)
    last = first + n_in_block - 1

    X = np.add.reduceat(ds.X, first, axis=0)
    control = None if ds.control is None else np.add.reduceat(ds.control, first, axis=0)
    if how == "mean":
        X = X / n_in_block[:, None]
        if control is not None:
            control = control / n_in_block[:, None]

    bins = pd.DataFrame(
        {
            "chrom": ds.bins["chrom"].to_numpy()[first],
            "start": ds.bins["start"].to_numpy()[first],
            "end": ds.bins["end"].to_numpy()[last],
            "sample": ds.bins["sample"].to_numpy()[first],
        }
    )
    coarse_width = bin_width(ds) * factor
    chains = _chains_from_runs(bins, coarse_width)
    samples = list(pd.unique(bins["sample"]))
    first_sample = (bins["sample"] == samples[0]).to_numpy()
    binset = BinSet(
        table=bins.loc[first_sample, ["chrom", "start", "end"]].reset_index(drop=True),
        bin_size=coarse_width,
        chain_slices=[sl for sl in _chains_from_runs(bins.loc[first_sample].reset_index(drop=True),
                                                       coarse_width)],
    )
    out = ChromStateDataset(
        X=X,
        marks=list(ds.marks),
        bins=bins,
        chains=chains,
        size_factors=ds.size_factors[first],
        control=control,
        is_count=ds.is_count,
        binset=binset,
        meta={**ds.meta, "coarsened_from": bin_width(ds), "coarsen_factor": factor},
    )
    return out


def copy_size_factors(src: ChromStateDataset, dst: ChromStateDataset) -> ChromStateDataset:
    """Carry per-(sample, mark) size factors from one dataset to another (in place).

    Normalisation is computed on the coarse dataset -- median-of-ratios on 10 bp bins,
    where nearly every bin is 0 or 1, is meaningless -- and copied to the fine one.
    """
    offsets = np.ones((dst.n_rows, dst.n_marks), dtype=dst.size_factors.dtype)
    for s in dst.samples:
        src_mask = src.sample_mask(s)
        if not src_mask.any():
            raise ValueError(f"sample {s!r} is not in the source dataset")
        offsets[dst.sample_mask(s)] = src.size_factors[np.flatnonzero(src_mask)[0]]
    dst.size_factors = offsets
    for key in ("size_factors", "norm_method", "norm_applied", "norm_warning"):
        if key in src.meta:
            dst.meta[key] = src.meta[key]
    return dst


# ---------------------------------------------------------------------------
# Rescaling a fitted model
# ---------------------------------------------------------------------------


def _matrix_root(A: np.ndarray, ratio: float) -> np.ndarray:
    """``A ** ratio`` for a stochastic matrix, kept stochastic."""
    A = np.asarray(A, dtype=np.float64)
    if abs(ratio - 1.0) < 1e-12:
        return A.copy()
    root = None
    try:
        from scipy.linalg import fractional_matrix_power

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cand = fractional_matrix_power(A, ratio)
        if np.all(np.isfinite(cand)) and np.abs(cand.imag).max() < 1e-6:
            cand = cand.real
            if cand.min() > -1e-3:
                root = np.clip(cand, 0.0, None)
    except Exception:  # pragma: no cover - numerical edge cases only
        root = None
    if root is None:
        # Root the self-transition directly and share the rest in the old proportions.
        K = A.shape[0]
        root = np.empty_like(A)
        for i in range(K):
            p = float(np.clip(A[i, i], EPS, 1.0)) ** ratio
            off = A[i].copy()
            off[i] = 0.0
            off_sum = off.sum()
            root[i] = (1.0 - p) * (off / off_sum if off_sum > 0 else np.full(K, 1.0 / max(K - 1, 1)))
            if off_sum <= 0:
                root[i, i] = 0.0
            root[i, i] = p
    return root / np.maximum(root.sum(axis=1, keepdims=True), EPS)


def rescale_hmm(hmm: HMM, ratio: float) -> HMM:
    """The same model, expressed per bin of ``ratio`` times the width it was fitted at.

    ``ratio`` below 1 moves to finer bins (0.05 for 200 bp -> 10 bp), above 1 to wider.
    """
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    em = copy.deepcopy(hmm.emissions)
    if hasattr(em, "_pairs"):
        em._pairs = None  # row cache belongs to the dataset it was built on
    if em.name == "nb":
        em.mu = np.maximum(em.mu * ratio, 1e-6)
        em.dispersion = em.dispersion * ratio
    elif em.name == "bernoulli":
        em.p = np.clip(1.0 - (1.0 - em.p) ** ratio, 1e-4, 1 - 1e-4)
    elif em.name == "gaussian":
        pass  # mean coverage per base is width-invariant
    else:  # pragma: no cover - unknown emission families
        raise ValueError(f"cannot rescale emission model {em.name!r}")
    out = HMM(
        hmm.n_states, em,
        transitions=_matrix_root(hmm.transitions, ratio),
        init_probs=hmm.init_probs.copy(),
        n_jobs=getattr(hmm, "n_jobs", None),
    )
    out.fit_result = hmm.fit_result
    return out


# ---------------------------------------------------------------------------
# Decoding at the fine width
# ---------------------------------------------------------------------------


def _ratio(seg: Segmentation, fine: ChromStateDataset) -> float:
    fit_width = bin_width(seg.dataset)
    fine_width = bin_width(fine)
    if list(fine.marks) != list(seg.dataset.marks):
        raise ValueError(
            f"fine dataset marks {fine.marks} differ from the fitted marks {seg.dataset.marks}"
        )
    return fine_width / fit_width


def iter_decoded(
    seg: Segmentation, fine: ChromStateDataset, hmm: Optional[HMM] = None,
) -> Iterator[Tuple[slice, np.ndarray, np.ndarray]]:
    """Yield ``(chain, posterior, viterbi_path)`` for every chain of ``fine``.

    Emission likelihoods are evaluated one chain at a time, so memory is bounded by the
    longest chromosome rather than the genome, which is what makes a 10 bp genome-wide
    pass feasible on one node.
    """
    hmm = hmm if hmm is not None else rescale_hmm(seg.model, _ratio(seg, fine))
    A, pi = hmm.transitions, hmm.init_probs
    for sl in fine.chains:
        X, off = fine.X[sl], fine.size_factors[sl]
        log_B = hmm.emissions.log_prob(X, off)
        gamma, _, _ = forward_backward(log_B, A, pi)
        yield sl, gamma, viterbi(log_B, A, pi)


def decode_at_resolution(
    seg: Segmentation,
    fine: ChromStateDataset,
    dtype=np.float32,
) -> Segmentation:
    """Score every bin of ``fine`` with the model fitted in ``seg``.

    ``fine`` must share the marks of the fitted dataset and should carry the same
    per-sample size factors (:func:`copy_size_factors`). The returned Segmentation has
    the fitted labels and colours, a rescaled model, and ``meta['fit_bin_size']`` /
    ``meta['bin_size']`` recording both widths. Posteriors are stored as ``dtype``
    (``float32`` by default: at 10 bp a genome of 12 states is 13 GB in float32 and twice
    that in float64).
    """
    ratio = _ratio(seg, fine)
    hmm = rescale_hmm(seg.model, ratio)
    posterior = np.empty((fine.n_rows, seg.n_states), dtype=dtype)
    states = np.empty(fine.n_rows, dtype=np.int32)
    loglik = 0.0
    for sl, gamma, path in iter_decoded(seg, fine, hmm):
        posterior[sl] = gamma
        states[sl] = path
    meta = {
        k: v for k, v in seg.meta.items()
        if k in ("emission", "collapsed", "min_state_separation", "n_states_used", "members")
    }
    meta.update({
        "fit_bin_size": bin_width(seg.dataset),
        "bin_size": bin_width(fine),
        "width_ratio": ratio,
        "decoded": True,
    })
    return Segmentation(
        posterior=posterior,
        states=states,
        labels=list(seg.labels),
        colors=list(seg.colors),
        rules=list(seg.rules),
        model=hmm,
        dataset=fine,
        loglik=loglik,
        meta=meta,
    )


def fit_and_decode(
    fine: ChromStateDataset,
    n_states: int = 10,
    fit_bin_size: int = DEFAULT_FIT_BIN_SIZE,
    norm: Optional[str] = "median_ratio",
    spikein=None,
    collapse_to: Optional[int] = None,
    **fit_kwargs,
) -> Tuple[Segmentation, Segmentation]:
    """Build the coarse dataset, normalise, fit, (optionally collapse), decode fine.

    Returns ``(coarse_segmentation, fine_segmentation)``. ``collapse_to`` fits
    ``n_states`` states and merges them down to that many
    (:func:`chromstate.collapse.collapse_states`) before decoding.
    """
    from .model.core import fit
    from .norm import normalize

    width = bin_width(fine)
    if fit_bin_size % width:
        raise ValueError(f"fit_bin_size {fit_bin_size} must be a multiple of the bin width {width}")
    coarse = coarsen_dataset(fine, fit_bin_size // width)
    if norm and norm != "none":
        normalize(coarse, method=norm, spikein=spikein)
        copy_size_factors(coarse, fine)
    seg = fit(coarse, n_states=n_states, **fit_kwargs)
    if collapse_to is not None and collapse_to < seg.n_states:
        from .collapse import collapse_states

        seg = collapse_states(seg, collapse_to)
    return seg, decode_at_resolution(seg, fine)


# ---------------------------------------------------------------------------
# 0-100 scores
# ---------------------------------------------------------------------------


def state_scores(
    seg: Segmentation,
    scale: int = 100,
    sample: Optional[str] = None,
    integer: bool = True,
) -> pd.DataFrame:
    """Per-bin state scores on a 0-``scale`` scale, one column per state.

    ``score_<state>`` is ``scale * posterior`` (rounded to an integer by default, which
    is what a browser track or a spreadsheet wants); ``state`` is the hard call and
    ``confidence`` its score. Rows sum to ``scale`` up to rounding.
    """
    ds = seg.dataset
    mask = np.ones(ds.n_rows, dtype=bool) if sample is None else ds.sample_mask(sample)
    out = ds.bins.loc[mask, ["chrom", "start", "end", "sample"]].reset_index(drop=True)
    post = seg.posterior[mask]
    vals = post * scale
    if integer:
        vals = np.rint(vals).astype(np.uint8 if scale <= 255 else np.int32)
    for k, lbl in enumerate(seg.labels):
        out[f"score_{lbl}"] = vals[:, k]
    out["state"] = [seg.labels[s] for s in seg.states[mask]]
    conf = post.max(axis=1) * scale
    out["confidence"] = np.rint(conf).astype(np.uint8 if scale <= 255 else np.int32) if integer else conf
    out.attrs["scale"] = scale
    return out


def _runs(values: np.ndarray) -> np.ndarray:
    """Start index of each run of equal consecutive values."""
    if len(values) == 0:
        return np.empty(0, dtype=np.int64)
    change = np.flatnonzero(values[1:] != values[:-1]) + 1
    return np.concatenate([[0], change])


def score_track(
    seg: Segmentation, state: str, sample: str, scale: int = 100, merge: bool = True,
) -> pd.DataFrame:
    """bedGraph-shaped table (``chrom start end value``) of one state's score."""
    if state not in seg.labels:
        raise ValueError(f"{state!r} is not a state; labels are {seg.labels}")
    k = seg.labels.index(state)
    ds = seg.dataset
    mask = ds.sample_mask(sample)
    if not mask.any():
        raise ValueError(f"unknown sample {sample!r}; dataset has {ds.samples}")
    bins = ds.bins.loc[mask, ["chrom", "start", "end"]].reset_index(drop=True)
    val = np.rint(seg.posterior[mask, k] * scale).astype(np.int32)
    if not merge:
        return bins.assign(value=val)
    chrom = bins["chrom"].to_numpy()
    start = bins["start"].to_numpy()
    end = bins["end"].to_numpy()
    key = pd.factorize(chrom)[0].astype(np.int64) * (scale + 1) + val
    contiguous = np.r_[True, start[1:] == end[:-1]]
    boundary = np.r_[0, np.flatnonzero((key[1:] != key[:-1]) | ~contiguous[1:]) + 1]
    last = np.r_[boundary[1:] - 1, len(val) - 1]
    return pd.DataFrame({"chrom": chrom[boundary], "start": start[boundary],
                         "end": end[last], "value": val[boundary]})


def write_bedgraph(
    seg: Segmentation, state: str, sample: str, path: str, scale: int = 100,
    track_line: bool = True,
) -> str:
    """Write one state's 0-``scale`` score for one sample as a bedGraph file."""
    track = score_track(seg, state, sample, scale=scale)
    with open(path, "w") as fh:
        if track_line:
            fh.write(f'track type=bedGraph name="{sample} {state}" '
                     f'description="ChromState {state} score (0-{scale})" visibility=full\n')
        track.to_csv(fh, sep="\t", header=False, index=False)
    return path


def write_bigwig(
    seg: Segmentation, state: str, sample: str, path: str, chrom_sizes: dict, scale: int = 100,
) -> str:  # pragma: no cover - optional dependency
    """Write one state's score as bigWig (needs ``pyBigWig``)."""
    import pyBigWig

    track = score_track(seg, state, sample, scale=scale)
    order = [c for c in chrom_sizes if c in set(track["chrom"])]
    bw = pyBigWig.open(path, "w")
    try:
        bw.addHeader([(c, int(chrom_sizes[c])) for c in order])
        for c in order:
            sub = track.loc[track["chrom"] == c]
            bw.addEntries(sub["chrom"].tolist(), sub["start"].astype(int).tolist(),
                          ends=sub["end"].astype(int).tolist(),
                          values=sub["value"].astype(float).tolist())
    finally:
        bw.close()
    return path


def write_score_tracks(
    seg: Segmentation, outdir: str, samples: Optional[Sequence[str]] = None,
    states: Optional[Sequence[str]] = None, scale: int = 100, fmt: str = "bedgraph",
    chrom_sizes: Optional[dict] = None,
) -> List[str]:
    """One track per (sample, state): ``<outdir>/<sample>.<state>.bedgraph``."""
    import os

    os.makedirs(outdir, exist_ok=True)
    samples = list(samples or seg.dataset.samples)
    states = list(states or seg.labels)
    paths = []
    for s in samples:
        for lbl in states:
            safe = lbl.replace("/", "-")
            if fmt == "bigwig":
                if chrom_sizes is None:
                    raise ValueError("bigWig output needs chrom_sizes")
                p = os.path.join(outdir, f"{s}.{safe}.bw")
                write_bigwig(seg, lbl, s, p, chrom_sizes, scale)
            else:
                p = os.path.join(outdir, f"{s}.{safe}.bedgraph")
                write_bedgraph(seg, lbl, s, p, scale)
            paths.append(p)
    return paths


def subset_segmentation(seg: Segmentation, mask: np.ndarray) -> Segmentation:
    """Row-subset a segmentation (chains rebuilt), e.g. to one chromosome or window."""
    ds = seg.dataset.subset_rows(mask)
    return replace(seg, posterior=seg.posterior[mask], states=seg.states[mask], dataset=ds)


# ---------------------------------------------------------------------------
# Disk-backed score tensor for genome-wide runs
# ---------------------------------------------------------------------------


class ScoreTensor:
    """``(samples x bins x states)`` 0-100 scores stored on disk, one shard per chromosome.

    A genome-wide 10 bp run of a mammalian genome has ~270 million bins per sample. As
    float64 posteriors for 12 states and 4 samples that is 100 GB; as ``uint8`` scores
    (0-100, which is all a score needs) it is 13 GB, and as per-chromosome memory-mapped
    shards none of it has to be in RAM at once. The tensor is written chromosome by
    chromosome while decoding (:func:`decode_to_tensor`) and read back by window or by
    region, so the whole genome is never materialised.

    Layout under ``path``: ``meta.json`` (samples, labels, scale, bin size, chromosome
    lengths in bins) and ``<chrom>.u8`` raw arrays of shape ``(n_samples, n_bins, K)``.
    """

    def __init__(self, path: str, samples: Sequence[str], labels: Sequence[str],
                 bin_size: int, scale: int = 100, chrom_bins: Optional[dict] = None,
                 chrom_starts: Optional[dict] = None):
        import os

        self.path = path
        self.samples = list(samples)
        self.labels = list(labels)
        self.bin_size = int(bin_size)
        self.scale = int(scale)
        self.chrom_bins = dict(chrom_bins or {})
        self.chrom_starts = dict(chrom_starts or {})
        os.makedirs(path, exist_ok=True)

    # -- persistence ---------------------------------------------------------

    def _meta_path(self) -> str:
        import os

        return os.path.join(self.path, "meta.json")

    def save_meta(self) -> None:
        import json

        with open(self._meta_path(), "w") as fh:
            json.dump({"samples": self.samples, "labels": self.labels, "bin_size": self.bin_size,
                       "scale": self.scale, "chrom_bins": self.chrom_bins,
                       "chrom_starts": self.chrom_starts}, fh, indent=1)

    @classmethod
    def open(cls, path: str) -> "ScoreTensor":
        import json
        import os

        with open(os.path.join(path, "meta.json")) as fh:
            m = json.load(fh)
        return cls(path, m["samples"], m["labels"], m["bin_size"], m["scale"], m["chrom_bins"],
                   m.get("chrom_starts", {}))

    def shard(self, chrom: str, mode: str = "r") -> np.ndarray:
        import os

        if chrom not in self.chrom_bins:
            raise KeyError(f"{chrom!r} is not in the tensor; chromosomes: {list(self.chrom_bins)}")
        shape = (len(self.samples), int(self.chrom_bins[chrom]), len(self.labels))
        return np.memmap(os.path.join(self.path, f"{chrom}.u8"), dtype=np.uint8, mode=mode, shape=shape)

    def add_chrom(self, chrom: str, n_bins: int, start: int = 0) -> np.ndarray:
        self.chrom_bins[chrom] = int(n_bins)
        self.chrom_starts[chrom] = int(start)
        self.save_meta()
        return self.shard(chrom, mode="w+")

    @property
    def chroms(self) -> List[str]:
        return list(self.chrom_bins)

    @property
    def n_bins(self) -> int:
        return int(sum(self.chrom_bins.values()))

    @property
    def nbytes(self) -> int:
        return len(self.samples) * self.n_bins * len(self.labels)

    # -- reading ---------------------------------------------------------------

    def window(self, chrom: str, start: int, end: int, sample: Optional[str] = None) -> np.ndarray:
        """Scores for ``[start, end)``: ``(n_samples, n_bins, K)`` or ``(n_bins, K)`` for one sample."""
        sh = self.shard(chrom)
        origin = self.chrom_starts.get(chrom, 0)
        lo = max(0, (int(start) - origin) // self.bin_size)
        hi = min(sh.shape[1], -(-(int(end) - origin) // self.bin_size))
        block = np.asarray(sh[:, lo:hi, :])
        if sample is not None:
            return block[self.samples.index(sample)]
        return block

    def region_means(self, regions: pd.DataFrame, id_col: str = "gene_id") -> pd.DataFrame:
        """Mean score per region, sample and state (one row per region and sample).

        ``regions`` has ``chrom``/``start``/``end`` and an identifier column. Regions on
        chromosomes outside the tensor are skipped; several rows per identifier are
        averaged together, bin-weighted.
        """
        acc: dict = {}
        cnt: dict = {}
        for rec in regions.itertuples(index=False):
            chrom = str(rec.chrom)
            if chrom not in self.chrom_bins:
                continue
            block = self.window(chrom, int(rec.start), int(rec.end))
            if block.shape[1] == 0:
                continue
            key = getattr(rec, id_col)
            s = block.sum(axis=1, dtype=np.float64)
            acc[key] = acc.get(key, 0.0) + s
            cnt[key] = cnt.get(key, 0) + block.shape[1]
        rows = []
        for key, s in acc.items():
            mean = s / cnt[key]
            for i, smp in enumerate(self.samples):
                row = {id_col: key, "sample": smp, "n_bins": cnt[key]}
                row.update({f"score_{lbl}": float(mean[i, k]) for k, lbl in enumerate(self.labels)})
                rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> pd.DataFrame:
        """Per sample and state: base pairs and percent by posterior mass and by hard call."""
        soft = np.zeros((len(self.samples), len(self.labels)))
        hard = np.zeros_like(soft)
        total = 0.0
        for chrom in self.chroms:
            sh = self.shard(chrom)
            for i in range(len(self.samples)):
                block = np.asarray(sh[i], dtype=np.float64)
                soft[i] += block.sum(axis=0) / self.scale
                hard[i] += np.bincount(block.argmax(axis=1), minlength=len(self.labels))
            total += sh.shape[1]
        rows = []
        for i, s in enumerate(self.samples):
            for k, lbl in enumerate(self.labels):
                rows.append({"sample": s, "state": lbl, "state_index": k,
                             "bp_soft": soft[i, k] * self.bin_size, "pct_soft": 100 * soft[i, k] / total,
                             "bp_hard": hard[i, k] * self.bin_size, "pct_hard": 100 * hard[i, k] / total})
        return pd.DataFrame(rows)


def decode_to_tensor(
    seg: Segmentation, fine: ChromStateDataset, path: str, scale: int = 100,
    tensor: Optional[ScoreTensor] = None,
) -> ScoreTensor:
    """Score ``fine`` and write the 0-``scale`` scores straight into a :class:`ScoreTensor`.

    Nothing larger than one chain's posterior is held in memory. ``fine`` may hold any
    subset of chromosomes (one at a time, in a streaming run); the tensor is extended.
    Every sample must share the fine dataset's binning (as :func:`build_dataset` guarantees).
    """
    ds = fine
    samples = ds.samples
    t = tensor if tensor is not None else ScoreTensor(path, samples, seg.labels, bin_width(ds), scale)
    if list(t.samples) != list(samples) or list(t.labels) != list(seg.labels):
        raise ValueError("tensor samples/labels do not match the dataset and model")
    hmm = rescale_hmm(seg.model, _ratio(seg, ds))
    chrom_col = ds.bins["chrom"].to_numpy()
    start_col = ds.bins["start"].to_numpy()
    sample_col = ds.bins["sample"].to_numpy()
    first_sample_mask = sample_col == samples[0]
    shards: dict = {}
    for sl, gamma, _ in iter_decoded(seg, ds, hmm):
        chrom = str(chrom_col[sl.start])
        if chrom not in shards:
            n = int((first_sample_mask & (chrom_col == chrom)).sum())
            origin = int(start_col[first_sample_mask & (chrom_col == chrom)].min())
            shards[chrom] = t.add_chrom(chrom, n, origin)
        i = samples.index(str(sample_col[sl.start]))
        lo = (start_col[sl.start] - t.chrom_starts[chrom]) // t.bin_size
        shards[chrom][i, lo:lo + (sl.stop - sl.start), :] = np.rint(gamma * scale).astype(np.uint8)
    for sh in shards.values():
        sh.flush()
    t.save_meta()
    return t
