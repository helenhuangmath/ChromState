"""Quantification summaries: how much of each state, in base pairs, and how strong.

Everything is reported per sample and, when samples are grouped into conditions, per
condition with the spread across replicates. Two readouts per state:

**Amount.** Base pairs (and the percentage of the scored genome) in each state, counted
two ways: by the hard call (``bp_hard``, what a segmentation BED file would show) and by
posterior mass (``bp_soft``, which keeps the partial bins a hard call rounds away).

**Strength.** How strongly the regions in a state express it: the posterior-weighted
mean of the state intensity (:func:`chromstate.regulation.state_intensity`), a log fold
of the observed signal over the state's own expected signal on its defining marks. Zero
is a typical instance of the state; +1 is e times stronger. A condition can keep the same
amount of a state and still lose strength (every active promoter is weaker), or keep
the strength and lose amount (fewer promoters are active); the two are distinguishable
only because they are reported separately.

    state_summary         per sample x state: bp, %, strength, segments
    condition_summary     per condition x state: means and standard deviations
    compare_conditions    conditions against a reference: differences, flows, notes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .model.core import Segmentation

EPS = 1e-12


def _widths(seg: Segmentation) -> np.ndarray:
    b = seg.dataset.bins
    return (b["end"].to_numpy() - b["start"].to_numpy()).astype(np.float64)


def _segments(states: np.ndarray, chrom: np.ndarray, start: np.ndarray, end: np.ndarray,
              n_states: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Number of segments per state and the segment lengths per state."""
    n = len(states)
    if n == 0:
        return np.zeros(n_states, dtype=int), [np.empty(0)] * n_states
    brk = np.r_[True, (states[1:] != states[:-1]) | (chrom[1:] != chrom[:-1]) | (start[1:] != end[:-1])]
    first = np.flatnonzero(brk)
    last = np.r_[first[1:] - 1, n - 1]
    length = end[last] - start[first]
    s = states[first]
    counts = np.bincount(s, minlength=n_states)
    per_state = [length[s == k] for k in range(n_states)]
    return counts, per_state


def state_summary(
    seg: Segmentation,
    samples: Optional[Sequence[str]] = None,
    include_strength: bool = True,
) -> pd.DataFrame:
    """Per sample and state: base pairs, percentages, confidence, strength, segments.

    Columns
    -------

    - ``bp_hard``, ``pct_hard``: base pairs and percentage of scored bases by hard call.
    - ``bp_soft``, ``pct_soft``: base pairs and percentage of scored bases by posterior mass.
    - ``mean_score``: mean 0–100 score of the state across all bins.
    - ``confidence``: mean posterior of the state in bins where it is the hard call.
    - ``strength``: posterior-weighted mean intensity, expressed as log fold change.
    - ``strength_sd``: posterior-weighted standard deviation of intensity.
    - ``n_segments``, ``median_segment_bp``: contiguous hard-call segments and their median length.
    """
    ds = seg.dataset
    samples = list(samples or ds.samples)
    width = _widths(seg)
    inten = None
    if include_strength:
        from .regulation import state_intensity

        try:
            inten = state_intensity(seg)
        except Exception:  # pragma: no cover - e.g. bernoulli emissions on odd input
            inten = None

    rows = []
    for s in samples:
        m = ds.sample_mask(s)
        if not m.any():
            raise ValueError(f"unknown sample {s!r}; dataset has {ds.samples}")
        post = np.asarray(seg.posterior[m], dtype=np.float64)
        st = seg.states[m]
        w = width[m]
        total = w.sum()
        chrom = ds.bins["chrom"].to_numpy()[m]
        start = ds.bins["start"].to_numpy()[m]
        end = ds.bins["end"].to_numpy()[m]
        n_seg, seg_len = _segments(st, chrom, start, end, seg.n_states)
        soft = w @ post
        for k, lbl in enumerate(seg.labels):
            called = st == k
            hard_bp = float(w[called].sum())
            row = {
                "sample": s, "state": lbl, "state_index": k,
                "bp_hard": hard_bp, "pct_hard": 100.0 * hard_bp / max(total, EPS),
                "bp_soft": float(soft[k]), "pct_soft": 100.0 * float(soft[k]) / max(total, EPS),
                "mean_score": 100.0 * float((w * post[:, k]).sum() / max(total, EPS)),
                "confidence": float(post[called, k].mean()) if called.any() else np.nan,
                "n_segments": int(n_seg[k]),
                "median_segment_bp": float(np.median(seg_len[k])) if len(seg_len[k]) else np.nan,
            }
            if inten is not None:
                wk = w * post[:, k]
                tot = wk.sum()
                if tot > EPS:
                    mu = float((wk * inten[m][:, k]).sum() / tot)
                    sd = float(np.sqrt(max((wk * (inten[m][:, k] - mu) ** 2).sum() / tot, 0.0)))
                else:
                    mu, sd = np.nan, np.nan
                row["strength"] = mu
                row["strength_sd"] = sd
            rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["total_bp"] = {s: float(width[ds.sample_mask(s)].sum()) for s in samples}
    out.attrs["bin_size"] = int(np.median(width))
    return out


def condition_summary(
    summary: pd.DataFrame, groups: Dict[str, Sequence[str]],
) -> pd.DataFrame:
    """Average a :func:`state_summary` over the replicates of each condition.

    One row per condition and state, with ``<col>`` the mean and ``<col>_sd`` the
    standard deviation across replicates (NaN with one replicate), for every numeric
    column of the per-sample table. The per-sample ``strength_sd`` (spread within a
    sample) is carried as ``strength_within_sd`` so the two spreads cannot be confused.
    """
    known = set(summary["sample"])
    num = [c for c in summary.columns
           if c not in ("sample", "state", "state_index") and pd.api.types.is_numeric_dtype(summary[c])]
    frames = []
    for cond, reps in groups.items():
        reps = list(reps)
        missing = [r for r in reps if r not in known]
        if missing:
            raise ValueError(f"condition {cond!r} names unknown sample(s) {missing}")
        sub = summary.loc[summary["sample"].isin(reps)]
        g = sub.groupby(["state_index", "state"], sort=True)[num]
        mean = g.mean()
        sd = g.std(ddof=1) if len(reps) > 1 else mean * np.nan
        # per-sample spreads (strength_sd) become *_within_sd; *_sd is the replicate sd
        mean.columns = [f"{c[:-3]}_within_sd" if c.endswith("_sd") else c for c in mean.columns]
        sd = sd[[c for c in sd.columns if not c.endswith("_sd")]]
        sd.columns = [f"{c}_sd" for c in sd.columns]
        frame = pd.concat([mean, sd], axis=1).reset_index()
        frame.insert(0, "condition", cond)
        frame.insert(1, "n_replicates", len(reps))
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out.attrs.update(summary.attrs)
    return out


@dataclass
class ConditionComparison:
    """Everything :func:`compare_conditions` computed."""

    per_sample: pd.DataFrame
    per_condition: pd.DataFrame
    differences: pd.DataFrame
    flows: Dict[Tuple[str, str], pd.DataFrame]
    reference: str
    note: str

    def flow(self, a: str, b: str) -> pd.DataFrame:
        return self.flows[(a, b)]


def group_posterior(seg: Segmentation, samples: Sequence[str]) -> Tuple[np.ndarray, pd.DataFrame]:
    """Mean posterior over the replicates of one condition, with the shared bins."""
    ds = seg.dataset
    samples = list(samples)
    ref_bins = None
    acc = None
    for s in samples:
        m = ds.sample_mask(s)
        if not m.any():
            raise ValueError(f"unknown sample {s!r}; dataset has {ds.samples}")
        b = ds.bins.loc[m, ["chrom", "start", "end"]].reset_index(drop=True)
        if ref_bins is None:
            ref_bins, acc = b, np.zeros((m.sum(), seg.n_states))
        elif len(b) != len(ref_bins) or not b.equals(ref_bins):
            raise ValueError(f"sample {s!r} does not share the binning of {samples[0]!r}")
        acc += np.asarray(seg.posterior[m], dtype=np.float64)
    return acc / len(samples), ref_bins


def flow_bp(seg: Segmentation, group_a: Sequence[str], group_b: Sequence[str]) -> pd.DataFrame:
    """``(K, K)`` base pairs that move from each state (rows) to each state (columns).

    The condition-level version of :func:`chromstate.dynamics.state_flow`: the joint of
    the two conditions' mean posteriors on matched bins, weighted by bin width, so the
    table sums to the scored genome size and the diagonal is what stayed put.
    """
    Pa, bins = group_posterior(seg, group_a)
    Pb, _ = group_posterior(seg, group_b)
    w = (bins["end"] - bins["start"]).to_numpy(dtype=np.float64)
    joint = (Pa * w[:, None]).T @ Pb
    return pd.DataFrame(joint, index=seg.labels, columns=seg.labels)


def compare_conditions(
    seg: Segmentation,
    groups: Dict[str, Sequence[str]],
    reference: Optional[str] = None,
    flows: bool = True,
) -> ConditionComparison:
    """Quantitative comparison of chromatin states across conditions.

    For every condition against the ``reference`` (the first by default):
    ``delta_pct_soft`` (percentage points of the genome gained or lost per state),
    ``delta_strength`` (change in log-fold strength), ``log2_ratio_bp`` of soft base
    pairs, and a ``z`` for the percentage difference against the replicate spread when
    both sides have replicates (a screening statistic, not a test:
    :func:`chromstate.dynamics.differential_regions` is the test). ``flows`` holds the
    base-pair flow matrix from the reference to each condition and between consecutive
    conditions.
    """
    conds = list(groups)
    if len(conds) < 2:
        raise ValueError("compare_conditions needs at least two conditions")
    reference = reference or conds[0]
    if reference not in groups:
        raise ValueError(f"reference {reference!r} is not a condition in {conds}")
    per_sample = state_summary(seg, samples=[s for reps in groups.values() for s in reps])
    per_cond = condition_summary(per_sample, groups)

    ref = per_cond.loc[per_cond["condition"] == reference].set_index("state")
    rows = []
    for cond in conds:
        if cond == reference:
            continue
        cur = per_cond.loc[per_cond["condition"] == cond].set_index("state")
        for lbl in seg.labels:
            r, c = ref.loc[lbl], cur.loc[lbl]
            d_pct = float(c["pct_soft"] - r["pct_soft"])
            sd_pool = np.sqrt(np.nansum([
                (r["pct_soft_sd"] ** 2) / max(r["n_replicates"], 1),
                (c["pct_soft_sd"] ** 2) / max(c["n_replicates"], 1)]))
            both_reps = r["n_replicates"] > 1 and c["n_replicates"] > 1
            rows.append({
                "condition": cond, "reference": reference, "state": lbl,
                "pct_soft_ref": float(r["pct_soft"]), "pct_soft": float(c["pct_soft"]),
                "delta_pct_soft": d_pct,
                "log2_ratio_bp": float(np.log2((c["bp_soft"] + 1.0) / (r["bp_soft"] + 1.0))),
                "strength_ref": float(r.get("strength", np.nan)),
                "strength": float(c.get("strength", np.nan)),
                "delta_strength": float(c.get("strength", np.nan) - r.get("strength", np.nan)),
                "z": float(d_pct / sd_pool) if both_reps and sd_pool > 0 else np.nan,
            })
    diff = pd.DataFrame(rows)

    flow_tables: Dict[Tuple[str, str], pd.DataFrame] = {}
    if flows:
        pairs = [(reference, c) for c in conds if c != reference]
        pairs += [(a, b) for a, b in zip(conds[:-1], conds[1:]) if (a, b) not in pairs and a != b]
        for a, b in pairs:
            flow_tables[(a, b)] = flow_bp(seg, groups[a], groups[b])

    n_single = [c for c, reps in groups.items() if len(reps) < 2]
    note = (
        f"{len(conds)} conditions compared to {reference!r} on {len(seg.labels)} states; "
        f"delta_pct_soft is in percentage points of the scored genome, delta_strength in "
        f"log fold of state intensity. z uses the replicate spread and is for screening; "
        f"use differential_regions for a test."
    )
    if n_single:
        note += f" Conditions without replicates (no z): {', '.join(n_single)}."
    return ConditionComparison(per_sample, per_cond, diff, flow_tables, reference, note)


def genome_fraction_table(summary: pd.DataFrame, value: str = "pct_soft") -> pd.DataFrame:
    """Wide ``sample x state`` table of one summary column (for a stacked bar)."""
    key = "condition" if "condition" in summary.columns else "sample"
    return summary.pivot(index=key, columns="state", values=value).fillna(0.0)


# ---------------------------------------------------------------------------
# Statistics for state-level comparisons
# ---------------------------------------------------------------------------


def _block_ids(bins: pd.DataFrame, block_bp: int) -> np.ndarray:
    chrom_code = pd.factorize(bins["chrom"])[0].astype(np.int64)
    return pd.factorize(chrom_code * 10_000_000 + bins["start"].to_numpy() // block_bp)[0]


def state_tests(
    seg: Segmentation,
    groups: Dict[str, Sequence[str]],
    reference: Optional[str] = None,
    block_bp: int = 100_000,
    n_boot: int = 1000,
    seed: int = 0,
    fdr: float = 0.05,
) -> pd.DataFrame:
    """Per state: is the genome share (and strength) different from the reference?

    Two complementary answers, because they rest on different units:

    **Replicates as units** (``t``, ``p_rep``, ``q_rep``): Welch's t-test on the
    per-replicate percentage of the genome in the state (posterior mass), condition
    against reference. This is the test that generalises to new biological samples. It
    needs two or more replicates on both sides, and with two versus two it has one or
    two degrees of freedom, so it is weak by construction; it is NaN otherwise.

    **Genome blocks as units** (``delta_lo``, ``delta_hi``, ``p_block``): the genome is
    cut into blocks of ``block_bp`` and the difference in state share is recomputed
    on ``n_boot`` bootstrap resamples of blocks, with replicates averaged inside each
    condition. This gives a confidence interval even without replicates, but it answers
    a narrower question -- "is the change spread across the genome rather than carried
    by a few loci" -- and says nothing about biological reproducibility. Blocks are
    resampled, not bins, because the HMM makes neighbouring bins dependent.

    ``q_rep`` and ``q_block`` are Benjamini-Hochberg across states within a contrast.
    For *which regions* changed use :func:`chromstate.dynamics.differential_regions`;
    this function is for *how much of the genome* moved per state.
    """
    from scipy import stats

    from .dynamics import benjamini_hochberg

    conds = list(groups)
    reference = reference or conds[0]
    per_sample = state_summary(seg, samples=[s for reps in groups.values() for s in reps],
                               include_strength=False)
    rng = np.random.default_rng(seed)

    # block-level soft bp per state, per condition (mean over replicates)
    cond_block: Dict[str, np.ndarray] = {}
    block_w = None
    for cond in conds:
        P, bins = group_posterior(seg, groups[cond])
        w = (bins["end"] - bins["start"]).to_numpy(dtype=np.float64)
        blocks = _block_ids(bins, block_bp)
        n_blocks = blocks.max() + 1
        M = np.zeros((n_blocks, seg.n_states))
        for k in range(seg.n_states):
            M[:, k] = np.bincount(blocks, weights=w * P[:, k], minlength=n_blocks)
        cond_block[cond] = M
        if block_w is None:
            block_w = np.bincount(blocks, weights=w, minlength=n_blocks)

    rows = []
    for cond in conds:
        if cond == reference:
            continue
        ref_pct = per_sample.loc[per_sample["sample"].isin(groups[reference])]
        cur_pct = per_sample.loc[per_sample["sample"].isin(groups[cond])]
        A, B = cond_block[reference], cond_block[cond]
        n_blocks = len(block_w)
        idx = rng.integers(0, n_blocks, size=(n_boot, n_blocks))
        tot = block_w[idx].sum(axis=1)
        boot_delta = np.empty((n_boot, seg.n_states))
        for k in range(seg.n_states):
            boot_delta[:, k] = 100.0 * (B[idx, k].sum(axis=1) - A[idx, k].sum(axis=1)) / np.maximum(tot, EPS)
        obs_delta = 100.0 * (B.sum(axis=0) - A.sum(axis=0)) / max(block_w.sum(), EPS)
        p_rep, t_stat, p_block = [], [], []
        for k, lbl in enumerate(seg.labels):
            a = ref_pct.loc[ref_pct["state"] == lbl, "pct_soft"].to_numpy(dtype=float)
            b = cur_pct.loc[cur_pct["state"] == lbl, "pct_soft"].to_numpy(dtype=float)
            if len(a) >= 2 and len(b) >= 2 and (a.std() > 0 or b.std() > 0):
                t, p = stats.ttest_ind(b, a, equal_var=False)
                t_stat.append(float(t))
                p_rep.append(float(p))
            else:
                t_stat.append(np.nan)
                p_rep.append(np.nan)
            d = boot_delta[:, k]
            # two-sided bootstrap p: how often the resampled delta crosses zero
            p_block.append(float(min(1.0, 2.0 * min((d <= 0).mean(), (d >= 0).mean()) + 1.0 / n_boot)))
            rows.append({
                "condition": cond, "reference": reference, "state": lbl,
                "pct_ref": float(a.mean()), "pct": float(b.mean()), "delta_pct": float(obs_delta[k]),
                "delta_lo": float(np.quantile(d, 0.025)), "delta_hi": float(np.quantile(d, 0.975)),
                "n_ref": len(a), "n": len(b), "t": t_stat[-1], "p_rep": p_rep[-1], "p_block": p_block[-1],
            })
    out = pd.DataFrame(rows)
    for col in ("p_rep", "p_block"):
        q = np.full(len(out), np.nan)
        for cond in out["condition"].unique():
            sel = (out["condition"] == cond).to_numpy() & out[col].notna().to_numpy()
            if sel.any():
                q[sel] = benjamini_hochberg(out.loc[sel, col].to_numpy())
        out[col.replace("p_", "q_")] = q
    out["significant_rep"] = out["q_rep"] < fdr
    out["significant_block"] = out["q_block"] < fdr
    out.attrs["note"] = (
        f"Welch t-test on per-replicate genome share (p_rep; needs >=2 replicates a side) and a "
        f"{block_bp // 1000} kb block bootstrap of the share difference ({n_boot} resamples; "
        f"delta_lo/hi 95% CI, p_block). q_* are BH across the {seg.n_states} states. "
        f"Region-level testing: differential_regions."
    )
    return out


def flow_test(seg: Segmentation, group_a: Sequence[str], group_b: Sequence[str],
              block_bp: int = 100_000, n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Off-diagonal flows between states with block-bootstrap confidence intervals.

    One row per (from, to) pair with ``bp`` moved, its share of the source state, and a
    95% interval from resampling ``block_bp`` genome blocks. Use it to say whether
    "27 kb of Polycomb became bivalent" is a genome-wide pattern or one locus.
    """
    Pa, bins = group_posterior(seg, group_a)
    Pb, _ = group_posterior(seg, group_b)
    w = (bins["end"] - bins["start"]).to_numpy(dtype=np.float64)
    blocks = _block_ids(bins, block_bp)
    n_blocks = blocks.max() + 1
    K = seg.n_states
    # per-block joint flow tensor (blocks, K, K)
    J = np.zeros((n_blocks, K, K))
    for i in range(K):
        for j in range(K):
            J[:, i, j] = np.bincount(blocks, weights=w * Pa[:, i] * Pb[:, j], minlength=n_blocks)
    obs = J.sum(axis=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_blocks, size=(n_boot, n_blocks))
    boots = np.stack([J[ii].sum(axis=0) for ii in idx])
    rows = []
    for i in range(K):
        src = max(obs[i].sum(), EPS)
        for j in range(K):
            if i == j:
                continue
            b = boots[:, i, j]
            rows.append({"from": seg.labels[i], "to": seg.labels[j], "bp": float(obs[i, j]),
                         "share_of_source": float(obs[i, j] / src),
                         "bp_lo": float(np.quantile(b, 0.025)), "bp_hi": float(np.quantile(b, 0.975)),
                         "n_blocks_contributing": int((J[:, i, j] > 0.5 * np.median(w)).sum())})
    return pd.DataFrame(rows).sort_values("bp", ascending=False).reset_index(drop=True)
