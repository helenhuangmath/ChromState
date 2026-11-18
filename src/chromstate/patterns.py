"""Gene-level chromatin state patterns across conditions, and the tables alluvial plots need.

A gene's promoter in three conditions is three state compositions. Two reductions make
them readable across thousands of genes:

**Dominant state** per condition: the state with the most posterior mass over the
promoter (or any other part), with the fraction it holds. "Polycomb in naive, bivalent
in memory, active in exhausted" is a pattern string, and genes with the same string form
a group.

**Category** of the pattern from the *class* of its states (active, poised, repressed,
background), so that a promoter moving from "weak Polycomb" to "strong Polycomb" is
``stable``, while one moving from any repressed state to any active state is
``activated``. Categories are coarse by design; the pattern string keeps the detail.

The same reduction applied to regions (every bin, weighted by its width) gives the
region-level flow, which is what the regional alluvial plot draws.

    gene_dominant_states     per gene: dominant state per condition, pattern, category
    pattern_counts           how many genes follow each pattern
    region_patterns          per bin: dominant state per condition, weighted by bp
    flow_table               long (stage columns + weight) table for plot_alluvial
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .model.core import Segmentation

STATE_CLASS_ORDER = ("active", "poised", "repressed", "background")


def default_state_classes(seg: Segmentation, promoter: bool = True) -> Dict[str, str]:
    """``label -> class`` for each state, for promoters (H3K4me3-led) or enhancers.

    Uses :func:`chromstate.enhancers.enhancer_state_classes` with H3K4me3 as the active
    mark for promoters, falling back through H3K27ac, H3K4me1 and ATAC when it is absent.
    """
    from .enhancers import enhancer_state_classes
    from .model.annotate import canonical_mark

    canon = {canonical_mark(m): m for m in seg.dataset.marks}
    order = ("K4me3", "K27ac", "K4me1", "ATAC") if promoter else ("K27ac", "K4me1", "ATAC", "K4me3")
    for key in order:
        if key in canon:
            tab = enhancer_state_classes(seg, active_mark=canon[key])
            return dict(zip(tab["label"], tab["class"]))
    return {lbl: "background" for lbl in seg.labels}


def categorise(classes: Sequence[str]) -> str:
    """Category of a sequence of state classes across conditions."""
    cls = list(classes)
    if len(set(cls)) == 1:
        return "stable"
    first, last = cls[0], cls[-1]
    inner = cls[1:-1]
    if first == last:
        return "transient"
    if last == "active":
        return "activated" if first in ("repressed", "background") else "poised_to_active"
    if first == "active":
        return "deactivated" if last in ("repressed", "background") else "active_to_poised"
    if last == "poised":
        return "poised"
    if last == "repressed":
        return "repressed"
    if last == "background":
        return "lost"
    return "switched" if not inner else "other"


def _occ_columns(table: pd.DataFrame, part: str) -> List[str]:
    pre = f"{part}_occ_"
    cols = [c for c in table.columns if c.startswith(pre)]
    if cols:
        return cols
    raise ValueError(f"no {pre}* columns in the table; build it with gene_quantification "
                     f"or gene_regulatory_quantification")


def gene_dominant_states(
    gene_q: pd.DataFrame,
    groups: Dict[str, Sequence[str]],
    part: str = "P",
    state_classes: Optional[Dict[str, str]] = None,
    min_fraction: float = 0.0,
) -> pd.DataFrame:
    """Per gene: the dominant state of ``part`` in every condition, the pattern, the category.

    ``gene_q`` is a per-(gene, sample) table from
    :func:`chromstate.regulation.gene_quantification` (``part='P'`` promoter, ``'E'``
    distal, ``'B'`` body). Occupancies are averaged over each condition's replicates
    before the dominant state is taken; ``frac_<condition>`` is its share, and genes
    whose dominant share is below ``min_fraction`` in any condition are dropped.

    Columns: ``gene_id``, ``name``, ``chrom``, ``dom_<cond>``, ``frac_<cond>``, ``pattern``
    (``A>B>C`` in condition order), ``n_changes``, and with ``state_classes``
    ``class_<cond>`` and ``category``.
    """
    cols = _occ_columns(gene_q, part)
    labels = [c[len(f"{part}_occ_"):] for c in cols]
    conds = list(groups)
    base = gene_q.drop_duplicates("gene_id").set_index("gene_id")
    keep_cols = [c for c in ("name", "chrom") if c in base.columns]
    out = base[keep_cols].copy()
    known = set(gene_q["sample"])
    for cond in conds:
        reps = [s for s in groups[cond]]
        missing = [s for s in reps if s not in known]
        if missing:
            raise ValueError(f"condition {cond!r} names unknown sample(s) {missing}")
        mean = gene_q.loc[gene_q["sample"].isin(reps)].groupby("gene_id")[cols].mean()
        mean = mean.reindex(out.index)
        vals = mean.to_numpy(dtype=float)
        ok = np.isfinite(vals).all(axis=1) & (np.nansum(vals, axis=1) > 0)
        idx = np.where(ok, np.nanargmax(np.where(ok[:, None], vals, 0.0), axis=1), -1)
        out[f"dom_{cond}"] = [labels[i] if i >= 0 else None for i in idx]
        out[f"frac_{cond}"] = np.where(ok, vals[np.arange(len(vals)), np.maximum(idx, 0)]
                                      / np.maximum(np.nansum(vals, axis=1), 1e-12), np.nan)
    dom_cols = [f"dom_{c}" for c in conds]
    out = out.dropna(subset=dom_cols)
    if min_fraction > 0:
        out = out.loc[(out[[f"frac_{c}" for c in conds]] >= min_fraction).all(axis=1)]
    out["pattern"] = out[dom_cols].astype(str).agg(">".join, axis=1)
    doms = out[dom_cols].to_numpy()
    out["n_changes"] = (doms[:, 1:] != doms[:, :-1]).sum(axis=1) if len(conds) > 1 else 0
    if state_classes is not None:
        for cond in conds:
            out[f"class_{cond}"] = out[f"dom_{cond}"].map(state_classes).fillna("background")
        out["category"] = [categorise(r) for r in out[[f"class_{c}" for c in conds]].to_numpy()]
    out = out.reset_index()
    out.attrs.update({"conditions": conds, "part": part})
    return out


def pattern_counts(dom: pd.DataFrame, by: str = "pattern") -> pd.DataFrame:
    """How many genes follow each pattern (or category), most common first."""
    counts = dom[by].value_counts()
    out = counts.rename("n_genes").reset_index().rename(columns={"index": by})
    out["fraction"] = out["n_genes"] / max(len(dom), 1)
    return out


def region_patterns(
    seg: Segmentation,
    groups: Dict[str, Sequence[str]],
    state_classes: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Per bin: dominant state in every condition and the bin's width in bp.

    The regional counterpart of :func:`gene_dominant_states`. Posteriors are averaged
    over each condition's replicates; the result feeds :func:`flow_table`.
    """
    from .summary import group_posterior

    conds = list(groups)
    bins = None
    out = {}
    for cond in conds:
        P, b = group_posterior(seg, groups[cond])
        if bins is None:
            bins = b
        out[f"dom_{cond}"] = np.asarray(seg.labels)[np.argmax(P, axis=1)]
        out[f"frac_{cond}"] = P.max(axis=1)
    frame = bins.copy()
    for k, v in out.items():
        frame[k] = v
    frame["bp"] = frame["end"] - frame["start"]
    if state_classes is not None:
        for cond in conds:
            frame[f"class_{cond}"] = frame[f"dom_{cond}"].map(state_classes).fillna("background")
    frame.attrs["conditions"] = conds
    return frame


def flow_table(
    patterns: pd.DataFrame,
    conditions: Optional[Sequence[str]] = None,
    weight: Optional[str] = None,
    by: str = "dom",
) -> pd.DataFrame:
    """Long table for :func:`chromstate.viz.plot_alluvial`: one row per distinct path.

    Columns are the condition names (each holding the state or class at that stage)
    plus ``weight``: the number of genes, or the base pairs when the table has a ``bp``
    column (regions). ``by='class'`` draws the flow between state classes instead of
    states.
    """
    conds = list(conditions or patterns.attrs.get("conditions") or
                 [c[len("dom_"):] for c in patterns.columns if c.startswith("dom_")])
    cols = [f"{by}_{c}" for c in conds]
    missing = [c for c in cols if c not in patterns.columns]
    if missing:
        raise ValueError(f"pattern table lacks {missing}")
    if weight is None:
        weight = "bp" if "bp" in patterns.columns else None
    sub = patterns[cols].copy()
    sub.columns = conds
    if weight is None:
        out = sub.groupby(conds, sort=False).size().rename("weight").reset_index()
    else:
        out = sub.assign(weight=patterns[weight].to_numpy()).groupby(conds, sort=False)["weight"].sum().reset_index()
    out = out.sort_values("weight", ascending=False).reset_index(drop=True)
    out.attrs["stages"] = conds
    return out
