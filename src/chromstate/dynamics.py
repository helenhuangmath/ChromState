"""State dynamics: what moved between conditions, by how much, and with what confidence.

Because ChromState fits one joint model across all samples, state *k* means the same
thing in every sample. That is what makes the comparisons here valid: a change in
occupancy is a change in the genome, not a change in how the model was parameterised.

Two views of a transition are provided:

:func:`state_flow`
    The paired view. Bins are matched one-to-one between conditions, so the joint
    distribution of (state before, state after) can be computed exactly from the
    posteriors. Use this whenever the samples share a binning -- which is the normal case.

:func:`transport_flux`
    The unpaired view. When bins cannot be matched, entropic optimal transport finds the
    cheapest reallocation of state mass, with cost given by distance between state
    emission profiles. This also yields a single scalar -- the transport cost -- summarising
    how far the epigenome moved.

Confidence in a change is a separate question from its size, and needs three things that
are easy to skip:

:func:`replicate_noise_floor`
    What a bin's change score looks like between replicates of the *same* condition. Any
    cross-condition effect below that distribution is noise, whatever its p-value.

:func:`differential_regions`
    Testing at region level rather than bin level. The permutation null has a hard
    resolution floor set by the number of distinct group assignments, and bins are
    explicitly correlated by the HMM, so bin-level FDR is both underpowered and
    mis-calibrated.

:mod:`chromstate.calibrate`
    Temperature scaling. Saturated posteriors compress every change score toward 0 or 1,
    so partial shifts -- what most regulation looks like -- disappear before any test runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .model.core import Segmentation

EPS = 1e-12


# ---------------------------------------------------------------------------
# Paired flow
# ---------------------------------------------------------------------------


def _paired_posteriors(
    seg: Segmentation, sample_a: str, sample_b: str
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    ds = seg.dataset
    ma, mb = ds.sample_mask(sample_a), ds.sample_mask(sample_b)
    if ma.sum() != mb.sum():
        raise ValueError(
            f"Samples {sample_a!r} and {sample_b!r} have {ma.sum()} and {mb.sum()} bins. "
            "Paired comparison needs identical binning; use transport_flux() instead."
        )
    bins_a = ds.bins.loc[ma, ["chrom", "start"]].reset_index(drop=True)
    bins_b = ds.bins.loc[mb, ["chrom", "start"]].reset_index(drop=True)
    if not bins_a.equals(bins_b):
        raise ValueError(
            f"Bins of {sample_a!r} and {sample_b!r} are not in the same order. "
            "Rebuild the dataset from a shared BinSet."
        )
    return seg.posterior[ma], seg.posterior[mb], bins_a


def state_flow(seg: Segmentation, sample_a: str, sample_b: str, normalize: bool = True) -> pd.DataFrame:
    """``(K, K)`` joint distribution of state before (rows) and after (columns).

    Computed as ``P_a.T @ P_b / n`` over matched bins, so it uses the full soft posterior
    rather than hardened labels. Row sums give occupancy in ``sample_a``; column sums give
    occupancy in ``sample_b``.

    Parameters
    ----------
    normalize
        Return row-conditional transition probabilities ``P(after | before)`` instead of
        the joint. The joint is the honest quantity for a flow diagram; the conditional
        answers "given a region was in state X, where did it go".
    """
    Pa, Pb, _ = _paired_posteriors(seg, sample_a, sample_b)
    joint = (Pa.T @ Pb) / Pa.shape[0]
    if normalize:
        joint = joint / np.maximum(joint.sum(axis=1, keepdims=True), EPS)
    return pd.DataFrame(joint, index=seg.labels, columns=seg.labels)


def transition_propensity(seg: Segmentation, sample_a: str, sample_b: str) -> pd.DataFrame:
    """Per-bin measures of how much a region's state distribution moved.

    Columns
    -------
    ``tv_distance``
        Total variation distance between the two posteriors, in [0, 1]. Directly
        interpretable as "the fraction of this bin's state identity that changed".
    ``jsd``
        Jensen-Shannon divergence (base 2), in [0, 1]. Less sensitive to a single state
        swap than TV, more sensitive to broad reshuffling.
    ``moved``
        TV distance weighted by how far the mass moved in emission space, so a swap
        between two similar states counts for less than a swap between opposite ones.
    """
    Pa, Pb, bins = _paired_posteriors(seg, sample_a, sample_b)

    tv = 0.5 * np.abs(Pa - Pb).sum(axis=1)

    m = 0.5 * (Pa + Pb)
    jsd = 0.5 * _kl(Pa, m) + 0.5 * _kl(Pb, m)
    jsd = np.clip(jsd / np.log(2), 0.0, 1.0)

    D = state_distance_matrix(seg)
    max_d = max(D.max(), EPS)
    delta = Pb - Pa
    gain = np.clip(delta, 0, None)
    loss = np.clip(-delta, 0, None)
    gain_n = gain / np.maximum(gain.sum(axis=1, keepdims=True), EPS)
    loss_n = loss / np.maximum(loss.sum(axis=1, keepdims=True), EPS)
    travel = np.einsum("ik,kl,il->i", loss_n, D, gain_n) / max_d
    moved = tv * travel

    out = bins.copy()
    out["tv_distance"] = tv
    out["jsd"] = jsd
    out["moved"] = moved
    for k, lbl in enumerate(seg.labels):
        out[f"delta_{lbl}"] = delta[:, k]
    return out


def _kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0)
    q = np.clip(q, EPS, 1.0)
    return (p * np.log(p / q)).sum(axis=1)


def state_distance_matrix(seg: Segmentation, log_transform: bool = True) -> np.ndarray:
    """``(K, K)`` Euclidean distance between state emission profiles.

    Used to weight how "far" a transition is: an enhancer becoming a weak enhancer is a
    smaller move than an enhancer becoming heterochromatin.
    """
    M = seg.model.emissions.mean_matrix()
    if log_transform:
        M = np.log1p(M)
    scale = np.maximum(M.std(axis=0, keepdims=True), EPS)
    Mz = M / scale
    diff = Mz[:, None, :] - Mz[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


# ---------------------------------------------------------------------------
# Replicate-aware differential testing
# ---------------------------------------------------------------------------


@dataclass
class DifferentialResult:
    table: pd.DataFrame
    group_a: List[str]
    group_b: List[str]
    n_permutations: int
    has_pvalues: bool
    note: str


def differential_state(
    seg: Segmentation,
    group_a: Sequence[str],
    group_b: Sequence[str],
    n_permutations: int = 200,
    seed: int = 0,
    fdr: float = 0.05,
) -> DifferentialResult:
    """Test which bins changed state composition between two groups of samples.

    The statistic is the total-variation distance between the group-mean posteriors. The
    null is built by permuting sample labels between groups and recomputing, which
    preserves each bin's own variability and each sample's global behaviour.

    **Replicates are required for p-values.** With one sample per group there is no
    estimate of within-group variability, so any p-value would be measuring biological
    signal against a null of zero -- every bin would look significant. In that case this
    function returns effect sizes with ``has_pvalues=False`` and says so, rather than
    producing numbers that would not survive review.

    Returns
    -------
    A :class:`DifferentialResult` whose ``table`` holds per-bin effect sizes and, when
    replicates allow, ``pvalue`` and BH-adjusted ``qvalue`` plus a ``significant`` flag.
    """
    group_a, group_b = list(group_a), list(group_b)
    ds = seg.dataset
    for s in group_a + group_b:
        if s not in ds.samples:
            raise ValueError(f"Unknown sample {s!r}; dataset has {ds.samples}")

    stacks = {s: seg.posterior[ds.sample_mask(s)] for s in group_a + group_b}
    n_bins = next(iter(stacks.values())).shape[0]
    for s, arr in stacks.items():
        if arr.shape[0] != n_bins:
            raise ValueError(f"Sample {s!r} has {arr.shape[0]} bins, expected {n_bins}")

    mean_a = np.mean([stacks[s] for s in group_a], axis=0)
    mean_b = np.mean([stacks[s] for s in group_b], axis=0)
    stat = 0.5 * np.abs(mean_a - mean_b).sum(axis=1)

    bins = ds.bins.loc[ds.sample_mask(group_a[0]), ["chrom", "start", "end"]].reset_index(drop=True)
    table = bins.copy()
    table["tv_distance"] = stat
    for k, lbl in enumerate(seg.labels):
        table[f"delta_{lbl}"] = mean_b[:, k] - mean_a[:, k]
    table["top_gain"] = [seg.labels[i] for i in np.argmax(mean_b - mean_a, axis=1)]
    table["top_loss"] = [seg.labels[i] for i in np.argmin(mean_b - mean_a, axis=1)]

    can_test = len(group_a) >= 2 and len(group_b) >= 2
    if not can_test:
        note = (
            f"No p-values: group sizes are {len(group_a)} and {len(group_b)}. "
            "At least 2 replicates per group are required to estimate within-group "
            "variability. Effect sizes below are descriptive only."
        )
        return DifferentialResult(table, group_a, group_b, 0, False, note)

    rng = np.random.default_rng(seed)
    all_samples = group_a + group_b
    n_a = len(group_a)
    exceed = np.zeros(n_bins, dtype=np.int64)
    done = 0

    # Deduplicate on the *split*, not on the permutation. Every split of 2n samples is
    # reachable by n_a! * n_b! permutations, and the statistic is symmetric under
    # swapping the two groups, so counting permutations counts each distinct null value
    # many times over. That does not bias the exceedance *ratio*, but it dilutes the +1
    # pseudo-count in the p-value: with 2 versus 2 it turns an honest floor of 1/3 into a
    # reported 1/21, which is the difference between "this design cannot resolve
    # significance" and a bin appearing to pass FDR.
    seen = set()
    for _ in range(max(n_permutations, 1) * 4):
        perm = rng.permutation(len(all_samples))
        pa = [all_samples[i] for i in perm[:n_a]]
        pb = [all_samples[i] for i in perm[n_a:]]
        split = frozenset({frozenset(pa), frozenset(pb)})
        if split in seen:
            continue
        seen.add(split)
        if set(pa) == set(group_a) or set(pa) == set(group_b):
            continue  # the observed labelling, in either orientation
        null = 0.5 * np.abs(
            np.mean([stacks[s] for s in pa], axis=0) - np.mean([stacks[s] for s in pb], axis=0)
        ).sum(axis=1)
        exceed += null >= stat
        done += 1
        if done >= n_permutations:
            break

    if done == 0:
        note = (
            "No valid label permutations were available (group sizes too small for a "
            "distinct relabelling). Effect sizes are descriptive only."
        )
        return DifferentialResult(table, group_a, group_b, 0, False, note)

    pvalue = (exceed + 1.0) / (done + 1.0)
    floor = 1.0 / (done + 1.0)
    table["pvalue"] = pvalue
    table["qvalue"] = benjamini_hochberg(pvalue)
    table["significant"] = table["qvalue"] < fdr
    note = (
        f"Permutation test over {done} distinct group assignments "
        f"({len(group_a)} vs {len(group_b)} samples). Resolution floor on the p-value is "
        f"{floor:.3g}, so no bin can be more significant than that regardless of effect "
        f"size."
    )
    if floor >= fdr:
        note += (
            f" That floor is at or above the requested FDR of {fdr:g}: with this many "
            f"replicates no bin can be called significant, whatever the biology. "
            f"{_replicates_needed(fdr)} Rank by effect size instead, and treat the "
            f"p-values as unusable."
        )
    return DifferentialResult(table, group_a, group_b, done, True, note)


def _replicates_needed(fdr: float) -> str:
    """How many replicates per group the permutation floor alone would require."""
    from math import comb

    for n in range(2, 13):
        # Distinct unordered splits, minus the observed labelling.
        usable = comb(2 * n, n) // 2 - 1
        if 1.0 / (usable + 1.0) < fdr:
            return f"Balanced groups of {n} would be the minimum for the floor alone."
    return "No practical group size reaches that floor."


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH step-up FDR adjustment."""
    p = np.asarray(p, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


# ---------------------------------------------------------------------------
# Optimal transport (unpaired)
# ---------------------------------------------------------------------------


def sinkhorn(
    a: np.ndarray,
    b: np.ndarray,
    cost: np.ndarray,
    reg: float = 0.05,
    n_iter: int = 1000,
    tol: float = 1e-9,
) -> np.ndarray:
    """Entropic optimal transport plan between two distributions.

    A compact Sinkhorn-Knopp implementation so ChromState carries no dependency on POT.
    Stabilised by working with the kernel in log space when ``reg`` is small.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / max(a.sum(), EPS)
    b = b / max(b.sum(), EPS)
    K = np.exp(-cost / max(reg, 1e-9))
    K = np.maximum(K, 1e-300)

    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(n_iter):
        u_prev = u
        u = a / np.maximum(K @ v, EPS)
        v = b / np.maximum(K.T @ u, EPS)
        if np.max(np.abs(u - u_prev)) < tol:
            break
    return u[:, None] * K * v[None, :]


def transport_flux(
    seg: Segmentation,
    sample_a: str,
    sample_b: str,
    reg: float = 0.05,
) -> Tuple[pd.DataFrame, float]:
    """Optimal-transport reallocation of state mass between two samples.

    Use when bins cannot be matched between conditions. Returns the transport plan and
    the total transport cost, a single number summarising how far the epigenome moved in
    state space.

    Note that unlike :func:`state_flow` this is an *inference* about which states fed
    which, not a measurement: OT returns the cheapest explanation consistent with the
    marginals, and cheapest is not necessarily what happened. Prefer paired flow when the
    binning allows it.
    """
    ds = seg.dataset
    occ_a = seg.posterior[ds.sample_mask(sample_a)].mean(axis=0)
    occ_b = seg.posterior[ds.sample_mask(sample_b)].mean(axis=0)
    D = state_distance_matrix(seg)
    D = D / max(D.max(), EPS)
    plan = sinkhorn(occ_a, occ_b, D, reg=reg)
    cost = float((plan * D).sum())
    return pd.DataFrame(plan, index=seg.labels, columns=seg.labels), cost


def occupancy_change(seg: Segmentation, reference: Optional[str] = None) -> pd.DataFrame:
    """Per-sample state occupancy, and its change relative to a reference sample."""
    occ = seg.sample_state_matrix(soft=True)
    ref = reference if reference is not None else occ.index[0]
    delta = occ.subtract(occ.loc[ref], axis=1)
    delta.columns = [f"delta_{c}" for c in delta.columns]
    return pd.concat([occ, delta], axis=1)


# ---------------------------------------------------------------------------
# Confidence: what does "no change" look like, and where can it be tested
# ---------------------------------------------------------------------------


@dataclass
class NoiseFloor:
    """How much a bin moves when nothing happened.

    Two replicates of the same condition differ by library prep, sequencing depth, and
    the model's own instability on ambiguous bins. Every one of those produces a nonzero
    change score. Any cross-condition effect that does not clear this distribution is not
    a finding, whatever its p-value -- and with few replicates the p-value cannot resolve
    it anyway, so in practice this is the operative threshold.
    """

    threshold: float
    quantile: float
    distribution: np.ndarray
    pairs: List[Tuple[str, str]]
    per_condition: pd.DataFrame
    note: str

    def exceeds(self, effect: np.ndarray) -> np.ndarray:
        """Boolean mask of effects above the floor."""
        return np.asarray(effect) > self.threshold


def replicate_noise_floor(
    seg: Segmentation,
    groups: Dict[str, Sequence[str]],
    quantile: float = 0.95,
    statistic: str = "tv",
) -> NoiseFloor:
    """The distribution of change scores between replicates of the *same* condition.

    Parameters
    ----------
    groups
        ``{condition: [sample, ...]}``. Conditions with one sample contribute nothing and
        are reported as such rather than silently ignored.
    quantile
        Quantile of the within-condition distribution used as the threshold. 0.95 means
        "an effect this large occurs in 5% of bins when nothing changed", which is the
        per-bin false-positive rate you are accepting before any testing.
    statistic
        ``tv`` (total variation) or ``jsd`` (Jensen-Shannon), matching
        :func:`transition_propensity`.

    Returns
    -------
    A :class:`NoiseFloor`. Its ``threshold`` is the number to compare effects against;
    ``per_condition`` breaks the distribution down so a single noisy condition cannot
    hide inside a pooled quantile.
    """
    if statistic not in ("tv", "jsd"):
        raise ValueError(f"statistic must be 'tv' or 'jsd', got {statistic!r}")
    ds = seg.dataset

    rows, dists, pairs = [], [], []
    skipped = []
    for condition, reps in groups.items():
        reps = list(reps)
        for s in reps:
            if s not in ds.samples:
                raise ValueError(f"Unknown sample {s!r}; dataset has {ds.samples}")
        if len(reps) < 2:
            skipped.append(condition)
            continue
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                a, b = reps[i], reps[j]
                Pa = seg.posterior[ds.sample_mask(a)]
                Pb = seg.posterior[ds.sample_mask(b)]
                if Pa.shape != Pb.shape:
                    raise ValueError(f"{a} and {b} do not share a binning")
                if statistic == "tv":
                    d = 0.5 * np.abs(Pa - Pb).sum(axis=1)
                else:
                    m = 0.5 * (Pa + Pb)
                    d = np.clip((0.5 * _kl(Pa, m) + 0.5 * _kl(Pb, m)) / np.log(2), 0.0, 1.0)
                dists.append(d)
                pairs.append((a, b))
                rows.append(
                    {
                        "condition": condition,
                        "sample_a": a,
                        "sample_b": b,
                        "median": float(np.median(d)),
                        f"q{quantile:g}": float(np.quantile(d, quantile)),
                        "max": float(d.max()),
                        "frac_above_0.5": float((d > 0.5).mean()),
                    }
                )

    if not dists:
        raise ValueError(
            "No condition has 2+ replicates, so there is no way to measure what an "
            "unchanged bin looks like. Every effect size below would be uninterpretable."
        )

    pooled = np.concatenate(dists)
    threshold = float(np.quantile(pooled, quantile))
    note = (
        f"Floor from {len(pairs)} within-condition replicate pair(s), "
        f"{len(pooled)} bin comparisons: the {quantile:.0%} quantile of {statistic.upper()} "
        f"between replicates is {threshold:.3f}. Cross-condition effects at or below this "
        f"are indistinguishable from replicate noise."
    )
    if skipped:
        note += (
            f" Conditions with a single sample contributed nothing to the floor: "
            f"{', '.join(skipped)}."
        )
    return NoiseFloor(threshold, quantile, pooled, pairs, pd.DataFrame(rows), note)


def _logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    q = np.clip(p, eps, 1.0 - eps)
    return np.log(q / (1.0 - q))


def moderated_t_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    prior_df: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Two-sample t-test with the variance shrunk toward the across-feature typical value.

    With three or four replicates a per-feature variance estimate is worthless on its
    own: a region that happens to have two near-identical replicates gets a tiny standard
    error and a spectacular t-statistic. Borrowing strength across features -- limma's
    idea, in its simplest form -- fixes exactly that failure, and it is why a parametric
    test is usable here where a permutation test is not.

    The trade this makes against the permutation null is explicit: p-values become
    continuous and unbounded (so they can survive multiple-testing correction) at the
    cost of assuming approximate normality across replicates, on the logit scale where a
    bounded occupancy is closest to it.

    Returns ``(t, p, df)``.
    """
    from scipy.stats import t as t_dist

    n_a, n_b = values_a.shape[1], values_b.shape[1]
    if n_a < 2 or n_b < 2:
        raise ValueError(f"need 2+ replicates per group, got {n_a} and {n_b}")
    mean_a = values_a.mean(axis=1)
    mean_b = values_b.mean(axis=1)
    ss = (values_a.var(axis=1, ddof=1) * (n_a - 1)) + (values_b.var(axis=1, ddof=1) * (n_b - 1))
    df = n_a + n_b - 2
    var = ss / df

    prior_var = float(np.median(var[np.isfinite(var) & (var > 0)])) if np.isfinite(var).any() else 1.0
    shrunk = (prior_df * prior_var + df * var) / (prior_df + df)
    se = np.sqrt(np.maximum(shrunk, 1e-12) * (1.0 / n_a + 1.0 / n_b))

    tstat = (mean_b - mean_a) / se
    total_df = df + prior_df
    pvalue = 2.0 * t_dist.sf(np.abs(tstat), total_df)
    return tstat, pvalue, float(total_df)


def _tile_regions(bins: pd.DataFrame, tile_bins: int) -> pd.DataFrame:
    """Fixed tiles of ``tile_bins`` consecutive bins, never spanning a chromosome."""
    frames = []
    for chrom, grp in bins.groupby("chrom", sort=False):
        grp = grp.sort_values("start")
        n = len(grp)
        starts = np.arange(0, n, tile_bins)
        frames.append(
            pd.DataFrame(
                {
                    "chrom": chrom,
                    "start": grp["start"].to_numpy()[starts],
                    "end": grp["end"].to_numpy()[np.minimum(starts + tile_bins, n) - 1],
                    "name": [f"{chrom}:{i}" for i in range(len(starts))],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _region_membership(bins: pd.DataFrame, regions: pd.DataFrame) -> List[np.ndarray]:
    """Row indices of ``bins`` inside each region, matched by coordinate."""
    bins = bins.reset_index(drop=True)
    index = {}
    for chrom, grp in bins.groupby("chrom", sort=False):
        starts = grp["start"].to_numpy()
        rows = grp.index.to_numpy()
        order = np.argsort(starts, kind="stable")
        index[str(chrom)] = (starts[order], rows[order])

    out = []
    for chrom, start, end in zip(regions["chrom"], regions["start"], regions["end"]):
        entry = index.get(str(chrom))
        if entry is None:
            out.append(np.empty(0, dtype=np.int64))
            continue
        starts, rows = entry
        lo = int(np.searchsorted(starts, start, side="left"))
        hi = int(np.searchsorted(starts, end, side="right"))
        out.append(rows[lo:hi])
    return out


def differential_regions(
    seg: Segmentation,
    group_a: Sequence[str],
    group_b: Sequence[str],
    regions: Optional[pd.DataFrame] = None,
    tile_bins: int = 10,
    min_bins: int = 3,
    noise_floor: Optional[NoiseFloor] = None,
    test: str = "moderated_t",
    prior_df: float = 4.0,
    n_permutations: int = 500,
    seed: int = 0,
    fdr: float = 0.05,
) -> DifferentialResult:
    """Test regions rather than bins, which is the only version with usable power.

    Bin-level testing fails twice over. The permutation null has a resolution floor set by
    the number of distinct group assignments -- 1/3 with two replicates a side, 1/35 with
    four -- and then Benjamini-Hochberg divides whatever is left across hundreds of
    thousands of bins that are not independent, because the HMM explicitly correlates
    neighbours. Testing a few thousand regions instead fixes the multiple-testing arm of
    that, and averaging within a region raises the effect-to-noise ratio.

    ``regions`` **must not be derived from the observed effect**. Choosing regions because
    they look changed and then testing those same regions on the same data is circular,
    and the p-values would be meaningless. Pass biologically defined regions -- gene
    promoters are the natural choice, and the ones that link to expression -- or accept
    the default fixed tiles, which are data-independent by construction.

    Parameters
    ----------
    regions
        ``chrom``/``start``/``end`` and optionally ``name``. Defaults to fixed tiles of
        ``tile_bins`` consecutive bins.
    min_bins
        Regions covering fewer bins than this are dropped: a one-bin "region" reintroduces
        exactly the noise the aggregation is there to average out.
    noise_floor
        When given, the effect-size threshold is carried into the result so a region can
        be required to clear replicate noise *and* the test.
    test
        ``moderated_t`` (default) runs a variance-shrunk t-test per state on logit
        occupancy across replicates. ``permutation`` runs the assumption-free label
        permutation.

        The default is parametric for a reason that is arithmetic, not preference. A
        permutation p-value cannot go below 1 / (number of distinct group assignments),
        and Benjamini-Hochberg multiplies the smallest p-value by the number of tests. For
        120 regions the smallest reachable q-value is 1.0 at three replicates a side, 1.0
        at four, 0.95 at five, and only reaches 0.02 at **eight replicates per group**. No
        realistic ChIP-seq design can be tested by permutation at region level, so a test
        that assumes normality on the logit scale and borrows variance across regions is
        the one that can actually reject anything. Both are provided, and the permutation
        route reports its own floor so the limit is never silent.
    """
    group_a, group_b = list(group_a), list(group_b)
    ds = seg.dataset
    for s in group_a + group_b:
        if s not in ds.samples:
            raise ValueError(f"Unknown sample {s!r}; dataset has {ds.samples}")

    bins = ds.bins.loc[ds.sample_mask(group_a[0]), ["chrom", "start", "end"]].reset_index(drop=True)
    if regions is None:
        regions = _tile_regions(bins, tile_bins)
    regions = regions.reset_index(drop=True)
    for col in ("chrom", "start", "end"):
        if col not in regions.columns:
            raise ValueError(f"regions needs a {col!r} column; got {list(regions.columns)}")

    members = _region_membership(bins, regions)
    keep = np.array([len(m) >= min_bins for m in members])
    regions = regions.loc[keep].reset_index(drop=True)
    members = [m for m, k in zip(members, keep) if k]
    if not members:
        raise ValueError(
            f"No region covers at least {min_bins} bins. Check that the region "
            f"coordinates and the dataset's bins are on the same assembly."
        )

    if test not in ("moderated_t", "permutation"):
        raise ValueError(f"test must be 'moderated_t' or 'permutation', got {test!r}")

    stacks = {s: seg.posterior[ds.sample_mask(s)] for s in group_a + group_b}

    def region_stat(samples_a: Sequence[str], samples_b: Sequence[str]) -> np.ndarray:
        mean_a = np.mean([stacks[s] for s in samples_a], axis=0)
        mean_b = np.mean([stacks[s] for s in samples_b], axis=0)
        tv = 0.5 * np.abs(mean_a - mean_b).sum(axis=1)
        return np.array([tv[m].mean() for m in members])

    stat = region_stat(group_a, group_b)
    mean_a = np.mean([stacks[s] for s in group_a], axis=0)
    mean_b = np.mean([stacks[s] for s in group_b], axis=0)
    delta = np.stack([(mean_b[m] - mean_a[m]).mean(axis=0) for m in members])

    table = regions.copy()
    if "name" not in table.columns:
        table["name"] = [f"{c}:{s}-{e}" for c, s, e in
                         zip(table["chrom"], table["start"], table["end"])]
    table["n_bins"] = [len(m) for m in members]
    table["tv_distance"] = stat
    for k, lbl in enumerate(seg.labels):
        table[f"delta_{lbl}"] = delta[:, k]
    table["top_gain"] = [seg.labels[i] for i in np.argmax(delta, axis=1)]
    table["top_loss"] = [seg.labels[i] for i in np.argmin(delta, axis=1)]
    if noise_floor is not None:
        table["above_noise_floor"] = stat > noise_floor.threshold

    if len(group_a) < 2 or len(group_b) < 2:
        note = (
            f"No p-values: group sizes are {len(group_a)} and {len(group_b)}. Region "
            f"effect sizes are descriptive only."
        )
        if noise_floor is not None:
            note += (
                f" {int(table['above_noise_floor'].sum())} of {len(table)} regions exceed "
                f"the replicate noise floor ({noise_floor.threshold:.3f}), which is the "
                f"only evidence available at this replicate count."
            )
        return DifferentialResult(table, group_a, group_b, 0, False, note)

    if test == "moderated_t":
        # (regions, samples, states) occupancy, tested on the logit scale where a bounded
        # proportion is closest to normal and a change from 0.90 to 0.98 is not treated as
        # smaller than one from 0.50 to 0.58.
        def occupancy_of(sample: str) -> np.ndarray:
            post = stacks[sample]
            return np.stack([post[m].mean(axis=0) for m in members])

        occ_a = np.stack([occupancy_of(s) for s in group_a], axis=1)  # (R, na, K)
        occ_b = np.stack([occupancy_of(s) for s in group_b], axis=1)

        n_states = occ_a.shape[2]
        pvals = np.ones((len(stat), n_states))
        for k in range(n_states):
            _, p_k, df = moderated_t_test(_logit(occ_a[:, :, k]), _logit(occ_b[:, :, k]),
                                          prior_df=prior_df)
            pvals[:, k] = p_k
        # One test per region: the most-changed state, Bonferroni-corrected for having
        # looked at every state before picking it.
        best = np.argmin(pvals, axis=1)
        pvalue = np.clip(pvals[np.arange(len(best)), best] * n_states, 0.0, 1.0)
        table["driver_state"] = [seg.labels[i] for i in best]
        table["pvalue"] = pvalue
        table["qvalue"] = benjamini_hochberg(pvalue)
        table["significant"] = table["qvalue"] < fdr
        if noise_floor is not None:
            table["confident_change"] = table["significant"] & table["above_noise_floor"]
        note = (
            f"Moderated t-test on logit state occupancy across replicates, "
            f"{len(group_a)} versus {len(group_b)} samples, {len(table)} regions, "
            f"{df:g} effective degrees of freedom after variance shrinkage. Per region the "
            f"most-changed state is tested and Bonferroni-corrected across the "
            f"{n_states} states, then Benjamini-Hochberg across regions. "
            f"{int(table['significant'].sum())} region(s) pass FDR {fdr:g}"
        )
        if noise_floor is not None:
            note += (
                f"; {int(table['confident_change'].sum())} also clear the replicate noise "
                f"floor of {noise_floor.threshold:.3f}"
            )
        note += (
            ". Parametric: assumes approximate normality across replicates on the logit "
            "scale and borrows variance across regions."
        )
        return DifferentialResult(table, group_a, group_b, 0, True, note)

    rng = np.random.default_rng(seed)
    all_samples = group_a + group_b
    n_a = len(group_a)
    exceed = np.zeros(len(stat), dtype=np.int64)
    done = 0
    seen = set()
    for _ in range(max(n_permutations, 1) * 4):
        perm = rng.permutation(len(all_samples))
        pa = [all_samples[i] for i in perm[:n_a]]
        pb = [all_samples[i] for i in perm[n_a:]]
        split = frozenset({frozenset(pa), frozenset(pb)})
        if split in seen:
            continue
        seen.add(split)
        if set(pa) == set(group_a) or set(pa) == set(group_b):
            continue
        exceed += region_stat(pa, pb) >= stat
        done += 1
        if done >= n_permutations:
            break

    if done == 0:
        return DifferentialResult(
            table, group_a, group_b, 0, False,
            "No valid label permutations were available. Effect sizes are descriptive only.",
        )

    pvalue = (exceed + 1.0) / (done + 1.0)
    floor = 1.0 / (done + 1.0)
    table["pvalue"] = pvalue
    table["qvalue"] = benjamini_hochberg(pvalue)
    table["significant"] = table["qvalue"] < fdr
    if noise_floor is not None:
        table["confident_change"] = table["significant"] & table["above_noise_floor"]

    note = (
        f"Region-level permutation test over {len(table)} regions and {done} distinct "
        f"group assignments. Resolution floor on the p-value is {floor:.3g}. "
        f"{int(table['significant'].sum())} region(s) pass FDR {fdr:g}"
    )
    if noise_floor is not None:
        note += (
            f"; {int(table['confident_change'].sum())} of those also clear the replicate "
            f"noise floor of {noise_floor.threshold:.3f}"
        )
    note += "."
    if floor >= fdr:
        note += (
            f" The floor is at or above the FDR of {fdr:g}: no region can be called "
            f"significant at this replicate count. {_replicates_needed(fdr)}"
        )
    return DifferentialResult(table, group_a, group_b, done, True, note)
