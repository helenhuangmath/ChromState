"""Collapse similar states so the final number of states is stable.

Two models fitted with 15 states on two datasets rarely have the same 15 states: a
promoter state splits into "strong" and "weak" in one and an enhancer state splits in
the other. ChromHMM users deal with this by fitting a large model and merging states by
hand into a reference set. ChromState does the merge explicitly and reproducibly:

1. fit with more states than needed (``n_fit``, say 20);
2. cluster the fitted states by the distance between their emission profiles
   (optionally also by how often they sit next to each other along the genome);
3. cut the tree at the number of states wanted (``n_states``, say 10) and **sum the
   posteriors** of the states in each cluster.

Summing posteriors is exact: if a bin is 60% "strong promoter" and 30% "weak promoter",
it is 90% "promoter". The merged model's emission mean is the occupancy-weighted mean of
its members, and the merged transition matrix is the flow between clusters, so the
collapsed segmentation can be rescaled to another resolution and read like any other.
The cluster membership is kept in ``meta['members']`` so the merge can always be undone
on paper.

    collapse_states     merge the states of a fitted Segmentation down to n_states
    fit_collapsed       fit n_fit states, then collapse to n_states
    state_similarity    the distance matrix the merge is based on
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .dynamics import state_distance_matrix
from .model.annotate import annotate_states
from .model.core import Segmentation
from .model.hmm import HMM

EPS = 1e-12


def state_similarity(seg: Segmentation, method: str = "emission", flow_weight: float = 0.5) -> np.ndarray:
    """``(K, K)`` distance between states, in [0, 1] after scaling by the largest.

    ``emission``   Euclidean distance between z-scored log emission profiles.
    ``flow``       1 minus the symmetrised transition probability between the two states,
                   relative to the largest off-diagonal transition (states that hand bins
                   to each other are close).
    ``both``       ``(1 - flow_weight) * emission + flow_weight * flow``.
    """
    D = state_distance_matrix(seg)
    D = D / max(D.max(), EPS)
    if method == "emission":
        return D
    A = np.asarray(seg.model.transitions, dtype=np.float64)
    sym = 0.5 * (A + A.T)
    np.fill_diagonal(sym, 0.0)
    flow = 1.0 - sym / max(sym.max(), EPS)
    np.fill_diagonal(flow, 0.0)
    if method == "flow":
        return flow
    if method == "both":
        return (1.0 - flow_weight) * D + flow_weight * flow
    raise ValueError(f"method must be 'emission', 'flow' or 'both', got {method!r}")


def _cluster(D: np.ndarray, n_states: int, linkage: str) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster
    from scipy.cluster.hierarchy import linkage as _linkage
    from scipy.spatial.distance import squareform

    K = D.shape[0]
    if n_states >= K:
        return np.arange(K)
    Z = _linkage(squareform(D, checks=False), method=linkage)
    labels = fcluster(Z, t=n_states, criterion="maxclust") - 1
    # fcluster can return fewer clusters than asked when ties collapse; split the largest
    # cluster by the tree if that happens (rare, and only with degenerate distances).
    if len(np.unique(labels)) < n_states:
        order = np.argsort(labels, kind="stable")
        labels = np.empty(K, dtype=int)
        labels[order] = np.minimum(np.arange(K), n_states - 1)
    return labels


def collapse_states(
    seg: Segmentation,
    n_states: int,
    method: str = "emission",
    linkage: str = "average",
    flow_weight: float = 0.5,
    members: Optional[Sequence[Sequence[int]]] = None,
) -> Segmentation:
    """Merge the states of ``seg`` into ``n_states`` groups of similar states.

    Parameters
    ----------
    n_states
        Number of states to keep. Must be at most the fitted number.
    method, linkage, flow_weight
        How similarity is measured (:func:`state_similarity`) and which hierarchical
        linkage cuts it (``average`` by default; ``complete`` keeps merged states tight,
        ``single`` chains them).
    members
        Explicit groups of state indices to merge, overriding the clustering. Every
        state must appear exactly once.

    The result is a new :class:`Segmentation` on the same dataset: posteriors summed
    within groups, the Viterbi path mapped through the groups, emissions and transitions
    aggregated by occupancy, and labels re-derived from the merged emission profiles.
    """
    K = seg.n_states
    if n_states < 1 or n_states > K:
        raise ValueError(f"n_states must be in [1, {K}], got {n_states}")
    if members is not None:
        groups = np.full(K, -1, dtype=int)
        for g, idx in enumerate(members):
            for i in idx:
                if groups[i] != -1:
                    raise ValueError(f"state {i} appears in two groups")
                groups[i] = g
        if (groups < 0).any():
            raise ValueError("members must cover every state")
        n_states = len(members)
    else:
        groups = _cluster(state_similarity(seg, method, flow_weight), n_states, linkage)

    # Order new states by the lowest original index they contain, so the quiet-to-active
    # ordering of the fit survives the merge.
    first = pd.Series(np.arange(K)).groupby(groups).min().sort_values()
    remap = {old: new for new, old in enumerate(first.index)}
    groups = np.array([remap[g] for g in groups])
    n_new = len(first)

    M = np.zeros((n_new, K))
    M[groups, np.arange(K)] = 1.0
    posterior = np.asarray(seg.posterior @ M.T, dtype=seg.posterior.dtype)
    states = groups[seg.states].astype(np.int32)

    occ = np.asarray(seg.posterior.mean(axis=0), dtype=np.float64)
    occ_new = M @ occ
    W = M * occ[None, :] / np.maximum(occ_new[:, None], EPS)  # (n_new, K) weights in group

    em = copy.deepcopy(seg.model.emissions)
    if hasattr(em, "_pairs"):
        em._pairs = None
    if em.name == "nb":
        em.mu = W @ em.mu
    elif em.name == "bernoulli":
        em.p = np.clip(W @ em.p, 1e-4, 1 - 1e-4)
    elif em.name == "gaussian":
        mu_old = em.mu
        em.mu = W @ mu_old
        # variance of a mixture: within + between
        within = W @ (em.sigma ** 2)
        between = W @ (mu_old ** 2) - em.mu ** 2
        em.sigma = np.sqrt(np.maximum(within + between, 1e-6))
    em.n_states = n_new

    A = np.asarray(seg.model.transitions, dtype=np.float64)
    flow = (occ[:, None] * A)
    A_new = M @ flow @ M.T
    A_new = A_new / np.maximum(A_new.sum(axis=1, keepdims=True), EPS)
    pi_new = M @ np.asarray(seg.model.init_probs, dtype=np.float64)
    pi_new = pi_new / max(pi_new.sum(), EPS)
    hmm = HMM(n_new, em, transitions=A_new, init_probs=pi_new,
              n_jobs=getattr(seg.model, "n_jobs", None))
    hmm.fit_result = seg.model.fit_result

    labels, colors, rules = annotate_states(em.mean_matrix(), seg.dataset.marks, occupancy=occ_new)
    member_map: Dict[str, List[str]] = {
        labels[g]: [seg.labels[i] for i in np.flatnonzero(groups == g)] for g in range(n_new)
    }
    meta = dict(seg.meta)
    meta.update({
        "collapsed_from": K,
        "collapse_method": "members" if members is not None else f"{method}/{linkage}",
        "members": member_map,
        "state_groups": groups.tolist(),
    })
    return Segmentation(
        posterior=posterior, states=states, labels=labels, colors=colors, rules=rules,
        model=hmm, dataset=seg.dataset, loglik=seg.loglik, meta=meta,
    )


def fit_collapsed(
    ds, n_states: int = 10, n_fit: Optional[int] = None, method: str = "emission",
    linkage: str = "average", **fit_kwargs,
) -> Segmentation:
    """Fit ``n_fit`` states (default ``2 * n_states``) and collapse to ``n_states``.

    Over-fitting then merging is more stable across datasets and seeds than fitting
    ``n_states`` directly: the rare states that a direct fit absorbs into neighbours get
    their own state first and are merged on the evidence of their emission profile rather
    than lost in a local optimum.
    """
    from .model.core import fit

    n_fit = n_fit if n_fit is not None else 2 * n_states
    seg = fit(ds, n_states=max(n_fit, n_states), **fit_kwargs)
    return collapse_states(seg, n_states, method=method, linkage=linkage)


def merge_table(seg: Segmentation) -> pd.DataFrame:
    """Which original states went into each collapsed state (empty if not collapsed)."""
    members = seg.meta.get("members") or {}
    rows = [{"state": k, "members": ", ".join(v), "n_members": len(v)} for k, v in members.items()]
    return pd.DataFrame(rows)


def match_states(seg_a: Segmentation, seg_b: Segmentation) -> pd.DataFrame:
    """Pair the states of two models by emission-profile similarity (Hungarian matching).

    For comparing two independent fits, or a fit against a reference model: the table
    lists each state of ``seg_a`` with its best partner in ``seg_b`` and the distance
    between their z-scored log emission profiles.
    """
    from scipy.optimize import linear_sum_assignment

    marks_a, marks_b = list(seg_a.dataset.marks), list(seg_b.dataset.marks)
    shared = [m for m in marks_a if m in marks_b]
    if not shared:
        raise ValueError("the two models share no marks")
    Ma = np.log1p(seg_a.model.emissions.mean_matrix()[:, [marks_a.index(m) for m in shared]])
    Mb = np.log1p(seg_b.model.emissions.mean_matrix()[:, [marks_b.index(m) for m in shared]])
    scale = np.maximum(np.vstack([Ma, Mb]).std(axis=0, keepdims=True), EPS)
    Ma, Mb = Ma / scale, Mb / scale
    D = np.sqrt(((Ma[:, None, :] - Mb[None, :, :]) ** 2).sum(axis=-1))
    r, c = linear_sum_assignment(D)
    return pd.DataFrame({"state_a": [seg_a.labels[i] for i in r],
                         "state_b": [seg_b.labels[j] for j in c],
                         "distance": D[r, c]}).sort_values("distance").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Spatial smoothing: short segments between like neighbours
# ---------------------------------------------------------------------------


def smooth_segments(seg: Segmentation, min_bp: int = 200, max_passes: int = 5) -> Segmentation:
    """Reassign hard-call segments shorter than ``min_bp`` to a neighbouring state.

    :func:`collapse_states` merges *kinds* of state (two emission profiles that mean
    the same thing). This merges *stretches*: a 30 bp sliver of "weak enhancer" inside
    a 2 kb active promoter at 10 bp resolution is usually resolution noise, and a
    browser track is easier to read without it. A short run takes the state of the
    flanking segment with the higher mean posterior for that state over the run (so the
    posterior, not the neighbour's length, decides), repeated up to ``max_passes`` until
    nothing short remains. Posteriors are left untouched; only ``states`` changes, and
    ``meta['smoothed_min_bp']`` records the setting.
    """
    ds = seg.dataset
    chrom = ds.bins["chrom"].to_numpy()
    sample = ds.bins["sample"].to_numpy()
    start = ds.bins["start"].to_numpy()
    end = ds.bins["end"].to_numpy()
    states = seg.states.copy()
    post = seg.posterior
    n = len(states)
    for _ in range(max_passes):
        brk = np.r_[True, (states[1:] != states[:-1]) | (chrom[1:] != chrom[:-1])
                    | (sample[1:] != sample[:-1]) | (start[1:] != end[:-1])]
        first = np.flatnonzero(brk)
        last = np.r_[first[1:] - 1, n - 1]
        length = end[last] - start[first]
        short = np.flatnonzero(length < min_bp)
        if short.size == 0:
            break
        changed = False
        for s in short:
            lo, hi = first[s], last[s]
            cands = []
            if s > 0 and chrom[first[s - 1]] == chrom[lo] and sample[first[s - 1]] == sample[lo] \
                    and end[last[s - 1]] == start[lo]:
                cands.append(states[first[s - 1]])
            if s + 1 < len(first) and chrom[first[s + 1]] == chrom[lo] and sample[first[s + 1]] == sample[lo] \
                    and start[first[s + 1]] == end[hi]:
                cands.append(states[first[s + 1]])
            cands = [c for c in cands if c != states[lo]]
            if not cands:
                continue
            best = max(cands, key=lambda k: float(post[lo:hi + 1, k].mean()))
            states[lo:hi + 1] = best
            changed = True
        if not changed:
            break
    from dataclasses import replace

    return replace(seg, states=states, meta={**seg.meta, "smoothed_min_bp": int(min_bp)})
