"""Per-region quantitative readouts derived from the posterior.

This is the layer that separates ChromState from a segmenter. A hard segmentation says
"bin 41,203 is state 7". These functions say *how much* state 7, how confident that call
is, how much of the bin's identity is active versus repressive, and which mark is
responsible for the call.

Every function here takes a :class:`~chromstate.model.core.Segmentation` and returns
per-bin values aligned to ``segmentation.dataset.bins``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .model.annotate import canonical_mark, classify_marks
from .model.core import Segmentation

EPS = 1e-12


# ---------------------------------------------------------------------------
# Composition and confidence
# ---------------------------------------------------------------------------


def state_entropy(seg: Segmentation, normalize: bool = True) -> np.ndarray:
    """Shannon entropy of the per-bin state posterior.

    Low entropy means the model is certain which state a bin is in; high entropy means
    the bin sits between states. High-entropy bins are the candidate plastic regions --
    they are also the bins a hard segmentation silently rounds off.

    Parameters
    ----------
    normalize
        Divide by ``log K`` so the value lies in [0, 1] and is comparable between models
        fit with different numbers of states.
    """
    p = np.clip(seg.posterior, EPS, 1.0)
    h = -(p * np.log(p)).sum(axis=1)
    return h / np.log(seg.n_states) if normalize else h


def state_confidence(seg: Segmentation) -> np.ndarray:
    """Posterior mass on the winning state (the max of the posterior)."""
    return seg.posterior.max(axis=1)


def margin(seg: Segmentation) -> np.ndarray:
    """Gap between the best and second-best state posterior.

    A near-zero margin marks a bin whose label is a coin flip. Reporting the margin
    alongside the label is what makes a ChromState call traceable rather than assertive.
    """
    part = np.partition(seg.posterior, -2, axis=1)
    return part[:, -1] - part[:, -2]


def effective_states(seg: Segmentation) -> np.ndarray:
    """Perplexity of the posterior: the effective number of states a bin occupies."""
    p = np.clip(seg.posterior, EPS, 1.0)
    return np.exp(-(p * np.log(p)).sum(axis=1))


# ---------------------------------------------------------------------------
# Bivalency
# ---------------------------------------------------------------------------


def _normalized_signal(seg: Segmentation) -> np.ndarray:
    ds = seg.dataset
    return np.log1p(ds.X / np.maximum(ds.size_factors, EPS))


def bivalency(
    seg: Segmentation,
    active_mark: str = "H3K4me3",
    repressive_mark: str = "H3K27me3",
) -> pd.DataFrame:
    """Continuous bivalency scoring at the signal level.

    Rather than asking whether a bin was assigned to a state someone labelled "bivalent",
    this decomposes co-occurrence into two independently interpretable numbers:

    ``magnitude``
        Geometric mean ``sqrt(a*r)`` of the two normalised signals. Large only when both
        marks are actually present.
    ``balance``
        ``2*sqrt(a*r)/(a+r)``, which is 1 when the two marks are equal and falls towards
        0 as one dominates. Scale-free.
    ``score``
        Their product, which is exactly the harmonic mean ``2*a*r/(a+r)``. Zero if either
        mark is absent, maximal when both are strong and balanced.

    Reporting magnitude and balance separately matters because a weak-but-perfectly-
    balanced promoter and a strong-but-lopsided one get the same harmonic mean, yet mean
    very different things biologically.
    """
    ds = seg.dataset
    canon = [canonical_mark(m) for m in ds.marks]
    a_key, r_key = canonical_mark(active_mark), canonical_mark(repressive_mark)
    if a_key not in canon or r_key not in canon:
        raise ValueError(
            f"Bivalency needs both {active_mark} and {repressive_mark} in the mark panel; "
            f"this dataset has {ds.marks}"
        )
    Z = _normalized_signal(seg)
    a = Z[:, canon.index(a_key)]
    r = Z[:, canon.index(r_key)]

    denom = np.maximum(a + r, EPS)
    magnitude = np.sqrt(np.maximum(a * r, 0.0))
    balance = 2.0 * magnitude / denom
    score = 2.0 * a * r / denom

    return pd.DataFrame(
        {
            "bivalency_score": score,
            "bivalency_magnitude": magnitude,
            "bivalency_balance": balance,
            f"signal_{active_mark}": a,
            f"signal_{repressive_mark}": r,
        }
    )


def state_bivalency_profile(
    seg: Segmentation,
    active_mark: str = "H3K4me3",
    repressive_mark: str = "H3K27me3",
) -> np.ndarray:
    """Bivalency of each *state*, computed from its emission means. Length K."""
    canon = [canonical_mark(m) for m in seg.marks]
    a_idx = canon.index(canonical_mark(active_mark))
    r_idx = canon.index(canonical_mark(repressive_mark))
    M = np.log1p(seg.model.emissions.mean_matrix())
    a, r = M[:, a_idx], M[:, r_idx]
    return 2.0 * a * r / np.maximum(a + r, EPS)


def posterior_bivalency(
    seg: Segmentation,
    active_mark: str = "H3K4me3",
    repressive_mark: str = "H3K27me3",
) -> pd.DataFrame:
    """Bivalency at the posterior level, plus the bulk mixture caveat.

    Two quantities, which are *not* the same thing and are routinely conflated:

    ``within_state``
        Posterior-weighted average of each state's own bivalency. A bin scores high when
        it confidently occupies a state whose emission profile carries both marks --
        i.e. genuine co-occurrence on the same chromatin.
    ``between_state``
        Posterior mass split between a predominantly active state and a predominantly
        repressive state. In *bulk* data this is indistinguishable from a mixture of two
        cell populations, one active and one repressed, at the same locus.

    The distinction is the single biggest confound in bivalency analysis and ChromState
    reports it explicitly rather than folding both into one number. A high
    ``between_state`` score is a flag to go to reChIP or single-cell data, not a result.
    """
    groups = classify_marks(seg.marks)
    if not groups["active"] or not groups["repressive"]:
        raise ValueError("Need at least one activating and one repressive mark")

    biv = state_bivalency_profile(seg, active_mark, repressive_mark)
    within = seg.posterior @ biv

    M = np.log1p(seg.model.emissions.mean_matrix())
    act_strength = M[:, groups["active"]].mean(axis=1)
    rep_strength = M[:, groups["repressive"]].mean(axis=1)
    is_active = act_strength > rep_strength
    active_mass = seg.posterior[:, is_active].sum(axis=1)
    repressive_mass = seg.posterior[:, ~is_active].sum(axis=1)
    between = 2.0 * active_mass * repressive_mass / np.maximum(active_mass + repressive_mass, EPS)

    return pd.DataFrame(
        {
            "biv_within_state": within,
            "biv_between_state": between,
            "active_mass": active_mass,
            "repressive_mass": repressive_mass,
            "mixture_ambiguity": between / np.maximum(within + between, EPS),
        }
    )


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


def attribution(
    seg: Segmentation,
    against: Optional[int] = None,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Per-mark evidence for each bin's state call.

    For every bin, the log-likelihood ratio between the winning state and a comparison
    state is decomposed into per-mark terms. Because marks are conditionally independent
    given the state, these terms sum exactly to the total LLR -- the decomposition is
    exact, not an approximation or a heuristic saliency score.

    Parameters
    ----------
    against
        State index to compare against. Default is the runner-up state for each bin,
        which answers "why this state and not the next most likely one". Passing an
        explicit index (e.g. the Quiescent state) answers "why is this bin not background".

    Returns
    -------
    One column per mark holding its LLR contribution in nats, plus ``top_state``,
    ``against_state``, ``total_llr``, and ``top_mark`` (the mark contributing most).
    Positive values support the winning call; negative values argue against it.
    """
    ds = seg.dataset
    X, offsets = ds.X, ds.size_factors
    n_rows = X.shape[0]
    rows = np.arange(n_rows) if max_rows is None else np.arange(min(n_rows, max_rows))
    X, offsets = X[rows], offsets[rows]
    post = seg.posterior[rows]

    order = np.argsort(-post, axis=1)
    top = order[:, 0]
    runner = order[:, 1] if seg.n_states > 1 else order[:, 0]
    comp = np.full(len(rows), against, dtype=int) if against is not None else runner

    contrib = np.zeros((len(rows), ds.n_marks))
    for k in np.unique(np.concatenate([top, comp])):
        lp_k = seg.model.emissions.log_prob_by_mark(X, offsets, int(k))
        sel_top = top == k
        if sel_top.any():
            contrib[sel_top] += lp_k[sel_top]
        sel_cmp = comp == k
        if sel_cmp.any():
            contrib[sel_cmp] -= lp_k[sel_cmp]

    out = pd.DataFrame(contrib, columns=[f"llr_{m}" for m in ds.marks])
    out.insert(0, "top_state", [seg.labels[i] for i in top])
    out.insert(1, "against_state", [seg.labels[i] for i in comp])
    out["total_llr"] = contrib.sum(axis=1)
    out["top_mark"] = [ds.marks[j] for j in np.argmax(contrib, axis=1)]
    return out


def state_mark_importance(
    seg: Segmentation,
    against: Optional[int] = None,
    n_sample: int = 20_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Average per-mark evidence supporting each state, over bins assigned to it.

    A state-level summary of :func:`attribution`. Rows are states, columns are marks,
    values are mean LLR in nats.

    Parameters
    ----------
    against
        Comparison state, with the same meaning as in :func:`attribution`. The default
        (runner-up) answers *"which mark distinguishes this state from its nearest
        competitor"*, which is often not the mark that names the state: for a Polycomb
        state whose runner-up is a bivalent state, both carry H3K27me3, so the
        discriminating mark is the *absence* of H3K4me3. Pass the quiescent state's index
        to instead ask "which marks define this state relative to background", which is
        the question that recovers the textbook answer.
    """
    rng = np.random.default_rng(seed)
    n_rows = seg.dataset.n_rows
    idx = (
        np.arange(n_rows)
        if n_rows <= n_sample
        else np.sort(rng.choice(n_rows, size=n_sample, replace=False))
    )
    sub = attribution(_subset_segmentation(seg, idx), against=against)
    mark_cols = [c for c in sub.columns if c.startswith("llr_")]
    grouped = sub.groupby("top_state")[mark_cols].mean()
    grouped.columns = [c[4:] for c in grouped.columns]
    return grouped.reindex([lbl for lbl in seg.labels if lbl in grouped.index])


def _subset_segmentation(seg: Segmentation, idx: np.ndarray) -> Segmentation:
    """Row-subset a segmentation for cheap sampling. Chains are not preserved."""
    from dataclasses import replace

    ds = seg.dataset
    sub_ds = type(ds)(
        X=ds.X[idx],
        marks=list(ds.marks),
        bins=ds.bins.iloc[idx].reset_index(drop=True),
        chains=[slice(0, len(idx))],
        size_factors=ds.size_factors[idx],
        control=None if ds.control is None else ds.control[idx],
        is_count=ds.is_count,
        binset=ds.binset,
    )
    return replace(seg, posterior=seg.posterior[idx], states=seg.states[idx], dataset=sub_ds)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def quantify(
    seg: Segmentation,
    active_mark: str = "H3K4me3",
    repressive_mark: str = "H3K27me3",
    include_attribution: bool = True,
) -> pd.DataFrame:
    """Run every per-bin quantification and return one tidy table.

    Bivalency columns are skipped with a recorded reason when the mark panel lacks the
    required pair, rather than failing the whole run.
    """
    ds = seg.dataset
    out = ds.bins[["chrom", "start", "end", "sample"]].copy()
    out["state"] = [seg.labels[s] for s in seg.states]
    out["state_index"] = seg.states
    out["confidence"] = state_confidence(seg)
    out["margin"] = margin(seg)
    out["entropy"] = state_entropy(seg)
    out["effective_states"] = effective_states(seg)

    for k, lbl in enumerate(seg.labels):
        out[f"post_{lbl}"] = seg.posterior[:, k]

    try:
        biv = bivalency(seg, active_mark, repressive_mark)
        post_biv = posterior_bivalency(seg, active_mark, repressive_mark)
        out = pd.concat([out, biv.reset_index(drop=True), post_biv.reset_index(drop=True)], axis=1)
    except ValueError as exc:
        out.attrs["bivalency_skipped"] = str(exc)

    if include_attribution:
        attr = attribution(seg)
        keep = [c for c in attr.columns if c.startswith("llr_")] + ["total_llr", "top_mark"]
        out = pd.concat([out, attr[keep].reset_index(drop=True)], axis=1)

    return out


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Per-sample summary of the quantification table."""
    cols = [
        c
        for c in ("confidence", "margin", "entropy", "effective_states", "bivalency_score",
                  "biv_within_state", "biv_between_state", "mixture_ambiguity")
        if c in table.columns
    ]
    return table.groupby("sample")[cols].mean()


def top_regions(
    table: pd.DataFrame,
    by: str = "bivalency_score",
    n: int = 100,
    sample: Optional[str] = None,
) -> pd.DataFrame:
    """Highest-scoring bins by any quantification column -- the shortlist for follow-up."""
    sub = table if sample is None else table.loc[table["sample"] == sample]
    if by not in sub.columns:
        raise ValueError(f"Column {by!r} not in table; available: {list(sub.columns)}")
    return sub.nlargest(n, by).reset_index(drop=True)


def quantify_regions(
    seg: Segmentation,
    regions: pd.DataFrame,
    samples: Optional[Sequence[str]] = None,
    min_bins: int = 1,
) -> pd.DataFrame:
    """State composition of arbitrary regions (a BED table), per sample.

    One row per (region, sample): ``occ_<state>`` (mean posterior over the region's
    bins; the fraction of the region in each state), ``bp_<state>`` (the same in base
    pairs), ``dominant`` (the state with the largest share), ``dominant_share``,
    ``entropy`` (normalised, of the region's mean composition) and ``n_bins``. Regions
    are matched to bins by coordinate; a region covering fewer than ``min_bins`` bins is
    dropped. This is the "quantify these states in these regions" call: promoters, peaks,
    enhancer sets or any BED file the user supplies.
    """
    from .dynamics import _region_membership

    ds = seg.dataset
    samples = list(samples or ds.samples)
    regions = regions.reset_index(drop=True)
    for col in ("chrom", "start", "end"):
        if col not in regions.columns:
            raise ValueError(f"regions needs a {col!r} column; got {list(regions.columns)}")
    if "name" not in regions.columns:
        regions = regions.assign(name=[f"{c}:{s}-{e}" for c, s, e in
                                       zip(regions["chrom"], regions["start"], regions["end"])])
    frames = []
    K = seg.n_states
    for s in samples:
        m = ds.sample_mask(s)
        bins = ds.bins.loc[m, ["chrom", "start", "end"]].reset_index(drop=True)
        post = seg.posterior[m]
        width = (bins["end"] - bins["start"]).to_numpy(dtype=float)
        members = _region_membership(bins, regions)
        rows = []
        for i, idx in enumerate(members):
            if len(idx) < min_bins:
                continue
            w = width[idx]
            comp = (w[:, None] * post[idx]).sum(axis=0) / max(w.sum(), 1e-12)
            bp = comp * w.sum()
            p = np.clip(comp, 1e-12, 1.0)
            ent = float(-(p * np.log(p)).sum() / np.log(K)) if K > 1 else 0.0
            top = int(np.argmax(comp))
            row = {"name": regions.at[i, "name"], "chrom": regions.at[i, "chrom"],
                   "start": int(regions.at[i, "start"]), "end": int(regions.at[i, "end"]),
                   "sample": s, "n_bins": int(len(idx)),
                   "dominant": seg.labels[top], "dominant_share": float(comp[top]), "entropy": ent}
            for k, lbl in enumerate(seg.labels):
                row[f"occ_{lbl}"] = float(comp[k])
            for k, lbl in enumerate(seg.labels):
                row[f"bp_{lbl}"] = float(bp[k])
            rows.append(row)
        frames.append(pd.DataFrame(rows))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.attrs["states"] = list(seg.labels)
    return out
