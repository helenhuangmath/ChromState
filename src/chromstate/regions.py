"""Promoters, enhancers, and the regulatory window of a gene -- with the sizes as settings.

Every number here is a choice someone else will want to change, so none of them is
buried in a function body. A :class:`RegionScheme` holds the three that matter:

``promoter_upstream`` / ``promoter_downstream``
    The promoter is the TSS plus these flanks (1,000 bp each by default, so a 2 kb
    window). ``upstream`` is 5' of the TSS on the gene's strand.
``regulatory_flank``
    A gene's regulatory window is its annotated interval plus this much on each side
    (50 kb by default). Everything in that window that is not the promoter is the
    **distal** space where enhancers are looked for.

Three gene parts come out of it, and they are kept apart on purpose: a promoter state
and an enhancer state do not mean the same thing, and the question "which enhancers
contribute to this gene's expression, beyond its promoter" can only be asked when the
promoter has been taken out of the enhancer space first.::

    RegionScheme.promoters      TSS windows
    RegionScheme.bodies         gene bodies with the promoter trimmed off
    RegionScheme.distal         regulatory window minus the promoter (two pieces per gene)
    RegionScheme.regulatory     the whole window, for tiling in gene-level mode
    call_elements               candidate enhancers from the state track itself
    gene_regulatory_quantification
                                per gene and sample: promoter (P_), distal (E_), body (B_)
    enhancer_contributions      which elements matter to expression, beyond the promoter
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .enhancers import candidate_enhancers, enhancer_state_classes
from .features import FeatureSet
from .model.annotate import canonical_mark, classify_marks
from .model.core import Segmentation

DEFAULT_PROMOTER_FLANK = 1_000
DEFAULT_REGULATORY_FLANK = 50_000

_PART_COLS = ["gene_id", "name", "chrom", "start", "end", "strand", "part"]


@dataclass(frozen=True)
class RegionScheme:
    """How a gene is cut into promoter, body, and distal regulatory space."""

    promoter_upstream: int = DEFAULT_PROMOTER_FLANK
    promoter_downstream: int = DEFAULT_PROMOTER_FLANK
    regulatory_flank: int = DEFAULT_REGULATORY_FLANK
    body_min_length: int = 1_000

    def __post_init__(self) -> None:
        for name in ("promoter_upstream", "promoter_downstream", "regulatory_flank"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be >= 0")

    # -- parts ---------------------------------------------------------------

    def promoters(self, features: FeatureSet) -> pd.DataFrame:
        return features.promoters(upstream=self.promoter_upstream,
                                  downstream=self.promoter_downstream)

    def bodies(self, features: FeatureSet) -> pd.DataFrame:
        return features.bodies(skip_promoter=self.promoter_downstream,
                               min_length=self.body_min_length)

    def regulatory(self, features: FeatureSet) -> pd.DataFrame:
        """Gene interval +/- ``regulatory_flank``: the window a gene is scored over."""
        out = features.table[["gene_id", "name", "chrom", "start", "end", "strand"]].copy()
        out["start"] = np.maximum(out["start"] - self.regulatory_flank, 0)
        out["end"] = out["end"] + self.regulatory_flank
        out["part"] = "regulatory"
        return out[_PART_COLS]

    def distal(self, features: FeatureSet) -> pd.DataFrame:
        """The regulatory window with the promoter cut out: where enhancers live.

        Up to two rows per gene (the piece before the promoter and the piece after it,
        in genome coordinates); either may be empty and is then dropped. The gene body
        stays in, because intronic enhancers are enhancers.
        """
        reg = self.regulatory(features)
        prom = self.promoters(features).set_index("gene_id")
        p_start = prom.loc[reg["gene_id"], "start"].to_numpy()
        p_end = prom.loc[reg["gene_id"], "end"].to_numpy()
        left = reg.copy()
        left["end"] = np.minimum(reg["end"].to_numpy(), p_start)
        right = reg.copy()
        right["start"] = np.maximum(reg["start"].to_numpy(), p_end)
        out = pd.concat([left, right], ignore_index=True)
        out = out.loc[out["end"] > out["start"]].copy()
        out["part"] = "distal"
        order = np.lexsort((out["start"].to_numpy(), out["gene_id"].to_numpy()))
        return out.iloc[order].reset_index(drop=True)[_PART_COLS]

    def regions(self, features: FeatureSet, part: str) -> pd.DataFrame:
        if part == "promoter":
            return self.promoters(features)
        if part == "body":
            return self.bodies(features)
        if part == "distal":
            return self.distal(features)
        if part == "regulatory":
            return self.regulatory(features)
        if part == "gene":
            return features.regions("gene")
        raise ValueError(f"part must be promoter, body, distal, regulatory or gene, got {part!r}")

    def all_parts(self, features: FeatureSet,
                  parts: Sequence[str] = ("promoter", "distal", "body")) -> Dict[str, pd.DataFrame]:
        return {p: self.regions(features, p) for p in parts}

    def describe(self) -> str:
        return (f"promoter = TSS -{self.promoter_upstream:,}/+{self.promoter_downstream:,} bp; "
                f"regulatory window = gene +/- {self.regulatory_flank:,} bp; "
                f"distal = window minus promoter")


# ---------------------------------------------------------------------------
# Candidate elements from the state track
# ---------------------------------------------------------------------------


def _pick_active_mark(marks: Sequence[str]) -> str:
    canon = {canonical_mark(m): m for m in marks}
    for key in ("K27ac", "K4me1", "ATAC", "K9ac", "K4me3"):
        if key in canon:
            return canon[key]
    raise ValueError(f"no activating mark in the panel {list(marks)}; cannot define active states")


def call_elements(
    seg: Segmentation,
    features: Optional[FeatureSet] = None,
    scheme: Optional[RegionScheme] = None,
    classes: Sequence[str] = ("active", "poised"),
    min_score: float = 50.0,
    min_width: int = 200,
    merge_gap: int = 0,
    active_mark: Optional[str] = None,
    state_classes: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Candidate regulatory elements: runs of bins in enhancer-like states.

    A bin belongs to an element when its summed 0-100 score over the chosen state
    ``classes`` (active or poised, from :func:`chromstate.enhancers.enhancer_state_classes`)
    reaches ``min_score`` in **any** sample; runs are merged across samples so that one
    element table serves every condition and can be compared between them. Runs shorter
    than ``min_width`` bp are dropped and runs closer than ``merge_gap`` bp are joined.

    With ``features`` the elements are flagged as promoter or distal using the scheme's
    promoter flanks (:func:`chromstate.enhancers.candidate_enhancers`), which is what
    separates "the promoter" from "the enhancers" downstream. Returns a peaks-style
    table with ``enhancer_id``, ``chrom``, ``start``, ``end``, ``n_samples`` (samples in
    which the element qualifies), ``max_score``, and the promoter flags.
    """
    ds = seg.dataset
    if state_classes is None:
        state_classes = enhancer_state_classes(seg, active_mark or _pick_active_mark(ds.marks))
    cls = state_classes.set_index("state")["class"].reindex(range(seg.n_states)).to_numpy()
    keep_k = np.isin(cls, list(classes))
    if not keep_k.any():
        raise ValueError(f"no state is classified as {classes}; classes are {dict(zip(seg.labels, cls))}")

    bins_one = None
    hit = None
    best = None
    n_hit = None
    for s in ds.samples:
        m = ds.sample_mask(s)
        b = ds.bins.loc[m, ["chrom", "start", "end"]].reset_index(drop=True)
        score = 100.0 * seg.posterior[m][:, keep_k].sum(axis=1)
        if bins_one is None:
            bins_one, hit = b, score >= min_score
            best, n_hit = score.copy(), hit.astype(np.int32)
        else:
            if len(b) != len(bins_one) or not b[["chrom", "start"]].equals(bins_one[["chrom", "start"]]):
                raise ValueError("samples do not share a binning; call_elements needs matched bins")
            h = score >= min_score
            hit |= h
            n_hit += h
            best = np.maximum(best, score)

    chrom = bins_one["chrom"].to_numpy()
    start = bins_one["start"].to_numpy()
    end = bins_one["end"].to_numpy()
    rows = []
    i, n = 0, len(hit)
    while i < n:
        if not hit[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and hit[j + 1] and chrom[j + 1] == chrom[i] and start[j + 1] - end[j] <= merge_gap:
            j += 1
        rows.append((chrom[i], int(start[i]), int(end[j]), int(n_hit[i:j + 1].max()),
                     float(best[i:j + 1].max())))
        i = j + 1
    if not rows:
        raise ValueError(f"no bin reaches a {classes} score of {min_score}")
    out = pd.DataFrame(rows, columns=["chrom", "start", "end", "n_samples", "max_score"])
    out = out.loc[out["end"] - out["start"] >= min_width].reset_index(drop=True)
    out.insert(0, "enhancer_id", [f"{c}:{s}-{e}" for c, s, e in zip(out["chrom"], out["start"], out["end"])])
    out.attrs.update({"classes": list(classes), "min_score": min_score, "min_width": min_width})
    if features is not None:
        scheme = scheme or RegionScheme()
        out = candidate_enhancers(out, features,
                                  tss_window=max(scheme.promoter_upstream, scheme.promoter_downstream))
    return out


# ---------------------------------------------------------------------------
# Gene-level quantification by part
# ---------------------------------------------------------------------------


PART_PREFIX = {"promoter": "P", "body": "B", "gene": "G", "distal": "E", "regulatory": "R"}


def gene_regulatory_quantification(
    seg: Segmentation,
    features: FeatureSet,
    scheme: Optional[RegionScheme] = None,
    parts: Sequence[str] = ("promoter", "distal", "body"),
    **kwargs,
) -> pd.DataFrame:
    """Per (gene, sample) state quantities for the promoter (``P_``), distal space
    (``E_``) and body (``B_``), under one :class:`RegionScheme`.

    Thin wrapper over :func:`chromstate.regulation.gene_quantification` with the regions
    taken from the scheme. Each part has ``occ_``, ``int_`` and ``dose_`` columns per
    state, so :func:`chromstate.regulation.rank_states_for_expression` run on this table
    reports promoter and enhancer states separately.
    """
    from .regulation import gene_quantification

    scheme = scheme or RegionScheme()
    regions = scheme.all_parts(features, parts)
    out = gene_quantification(seg, features, parts=parts, regions=regions, **kwargs)
    out.attrs["scheme"] = scheme.describe()
    return out


def part_importance(importance_table: pd.DataFrame) -> pd.DataFrame:
    """Split a state-importance table's columns by gene part (``P``, ``E``, ``B``).

    Given the ``direction_column`` and ``n_columns`` of
    :class:`chromstate.regulation.StateImportance`, this cannot recover a per-part
    score; use :func:`rank_parts_for_expression` for that.
    """
    tab = importance_table.copy()
    tab["part"] = tab["direction_column"].str.extract(r"^(?:base_)?([A-Z])_")[0]
    return tab


def rank_parts_for_expression(
    gene_q: pd.DataFrame,
    expression: Optional[pd.DataFrame] = None,
    lfc: Optional[pd.Series] = None,
    contrast: Optional[Tuple[Sequence[str], Sequence[str]]] = None,
    samples: Optional[Sequence[str]] = None,
    parts: Sequence[str] = ("P", "E", "B"),
    model="hgb",
    n_splits: int = 5,
    n_boot: int = 300,
    seed: int = 0,
) -> pd.DataFrame:
    """How much held-out expression each gene part explains, alone and in combination.

    Fits the same model on promoter-only, distal-only, body-only columns and on every
    pair and the full set, scored on held-out chromosomes. The ``E`` rows answer
    "do enhancers add anything beyond the promoter": compare ``P+E`` with ``P``.
    """
    from .regulation import (
        compare_representations,
        expression_change_task,
        expression_level_task,
    )

    if lfc is not None:
        if contrast is None:
            raise ValueError("the change task needs contrast=(group_a, group_b)")
        task = expression_change_task(gene_q, lfc, contrast[0], contrast[1])
    elif expression is not None:
        task = expression_level_task(gene_q, expression, samples)
    else:
        raise ValueError("give expression= (level) or lfc= with contrast= (change)")

    cols = list(task.X.columns)
    by_part = {p: [c for c in cols if c.startswith(f"{p}_") and not c.endswith("_n_bins")] for p in parts}
    by_part = {p: v for p, v in by_part.items() if v}
    families: Dict[str, list] = {}
    keys = list(by_part)
    from itertools import combinations

    for r in range(1, len(keys) + 1):
        for combo in combinations(keys, r):
            families["+".join(combo)] = [c for p in combo for c in by_part[p]]
    ref = "+".join(keys)
    res = compare_representations(task, families=families, reference=ref, model=model,
                                  n_splits=n_splits, n_boot=n_boot, seed=seed)
    scores = res.scores.rename(columns={"family": "parts"})
    scores.attrs["contrasts"] = res.contrasts
    return scores


# ---------------------------------------------------------------------------
# Which enhancers matter, beyond the promoter
# ---------------------------------------------------------------------------


def default_rings(flank: int) -> Tuple[int, ...]:
    """Distance rings that split a regulatory flank into near, middle, far thirds-ish."""
    flank = int(flank)
    edges = sorted({0, max(flank // 10, 1), max(int(flank * 0.4), 2), flank})
    return tuple(edges)


def enhancer_contributions(
    seg: Segmentation,
    features: FeatureSet,
    expression: pd.DataFrame,
    elements: Optional[pd.DataFrame] = None,
    scheme: Optional[RegionScheme] = None,
    atac: Optional[pd.DataFrame] = None,
    active_mark: Optional[str] = None,
    rings: Optional[Sequence[int]] = None,
    samples: Optional[Sequence[str]] = None,
    promoter_quantities: Sequence[str] = ("occ", "int"),
    model: str = "auto",
    n_splits: int = 5,
    seed: int = 0,
) -> Dict[str, object]:
    """Rank each gene's enhancers by what they add to predicted expression.

    The model sees, per gene and sample, the promoter's state quantities (``P_occ_*``,
    ``P_int_*``) and the summed activity-by-contact input of distal elements in distance
    rings inside the regulatory window. It is a regression constrained to be monotone
    increasing in the enhancer rings (gradient boosting with ``model="hgb"``, a
    sign-constrained linear model with ``"linear"``; ``"auto"`` uses boosting from 200
    (gene, sample) rows and the linear model below that, see
    :func:`chromstate.enhancers.make_importance_model`), scored on held-out chromosomes,
    and each element's ``importance`` is the drop in predicted expression when that one
    element is deleted in silico (:func:`chromstate.enhancers.enhancer_importance`).

    ``expression`` is genes x samples on a log scale, columns named after the dataset's
    samples. ``elements`` defaults to :func:`call_elements` on the segmentation.

    Returns a dict with ``elements`` (the element table with promoter flags), ``pairs``
    (element-gene links with ABC scores), ``importance`` (one row per element, gene and
    sample, ranked), ``features`` (the model's training table), ``heldout_spearman`` and
    ``model``. Importance is a ranking of candidates, not a measured effect.
    """
    from .enhancers import (
        abc_score,
        enhancer_activity,
        enhancer_importance,
        gene_enhancer_features,
        link_enhancers,
    )
    from .regulation import _spearman_boot, cross_validated_predictions

    scheme = scheme or RegionScheme()
    ds = seg.dataset
    active_mark = active_mark or _pick_active_mark(ds.marks)
    tss_window = max(scheme.promoter_upstream, scheme.promoter_downstream)
    if elements is None:
        elements = call_elements(seg, features, scheme, active_mark=active_mark)
    elif not {"is_promoter", "center"} <= set(elements.columns):
        elements = candidate_enhancers(elements, features, tss_window=tss_window)
    rings = tuple(rings) if rings is not None else default_rings(scheme.regulatory_flank)

    activity = enhancer_activity(seg, elements, atac=atac, active_mark=active_mark)
    pairs = link_enhancers(elements, features, max_distance=scheme.regulatory_flank,
                           tss_window=tss_window)
    scores = abc_score(pairs, activity)
    enh_feat = gene_enhancer_features(scores, rings=rings)

    gq = gene_regulatory_quantification(seg, features, scheme, parts=("promoter",),
                                        include_signal=False)
    p_cols = [c for c in gq.columns
              if any(c.startswith(f"P_{q}_") for q in promoter_quantities)]
    table = enh_feat.merge(gq[["gene_id", "sample", "chrom"] + p_cols], on=["gene_id", "sample"])
    samples = list(samples or [s for s in ds.samples if s in expression.columns])
    if not samples:
        raise ValueError("expression has no column named after a sample of the dataset")
    table = table.loc[table["sample"].isin(samples)].reset_index(drop=True)
    expr = expression.copy()
    if "gene_id" in expr.columns:
        expr = expr.set_index("gene_id")
    y = np.array([expr[s].get(g, np.nan) for g, s in zip(table["gene_id"], table["sample"])],
                 dtype=float)
    ok = np.isfinite(y)
    table, y = table.loc[ok].reset_index(drop=True), y[ok]
    if len(table) < 20:
        raise ValueError(f"only {len(table)} (gene, sample) rows have expression; need more genes")

    from .enhancers import ring_columns

    ring_cols = ring_columns(rings)
    feat_cols = ring_cols + ["prom_input"] + p_cols
    X = table[feat_cols].to_numpy(dtype=float)
    mono = np.array([1 if c in ring_cols else 0 for c in feat_cols])

    from .enhancers import make_importance_model

    model_kind = make_importance_model(model, mono, len(table), seed)[1]

    def make(seed_):
        return make_importance_model(model_kind, mono, len(table), seed_)[0]

    groups = table["chrom"].to_numpy()
    n_groups = len(np.unique(groups))
    heldout = np.nan
    if n_groups >= 2:
        pred = cross_validated_predictions(X, y, groups, make, min(n_splits, n_groups), seed,
                                           keys=table["gene_id"].to_numpy())
        heldout = float(_spearman_boot(y, pred, np.arange(len(y))[None])[0])
    fitted = make(seed).fit(pd.DataFrame(X, columns=feat_cols), y)
    imp = enhancer_importance(fitted, table, scores, feature_columns=feat_cols, rings=rings,
                              samples=samples)
    imp = imp.merge(elements[["enhancer_id", "start", "end"]], on="enhancer_id", how="left")
    return {
        "elements": elements,
        "pairs": scores,
        "importance": imp,
        "features": table.assign(y=y),
        "heldout_spearman": heldout,
        "model": fitted,
        "model_kind": model_kind,
        "rings": rings,
        "note": (
            f"Expression modelled from promoter state quantities plus distal activity x "
            f"contact in rings {rings} bp within a {scheme.regulatory_flank:,} bp window "
            f"({'monotone gradient boosting' if model_kind == 'hgb' else 'sign-constrained linear model'}"
            f" on {len(table)} gene x sample rows); held-out Spearman rho {heldout:.3f}. "
            f"importance = predicted expression lost when the element is deleted in silico; "
            f"a ranking of candidates, not an effect."
        ),
    }


def summarise_contributions(result: Dict[str, object], top_n: int = 3) -> pd.DataFrame:
    """Per gene and sample: promoter input, total distal input, and the top enhancers."""
    imp = result["importance"]
    feat = result["features"]
    rows = []
    for (g, s), sub in imp.groupby(["gene_id", "sample"], sort=False):
        sub = sub.sort_values("importance", ascending=False)
        frow = feat.loc[(feat["gene_id"] == g) & (feat["sample"] == s)]
        rows.append({
            "gene_id": g, "sample": s,
            "name": sub["name"].iloc[0],
            "prom_input": float(frow["prom_input"].iloc[0]) if len(frow) else np.nan,
            "n_distal": int(len(sub)),
            "total_importance": float(sub["importance"].clip(lower=0).sum()),
            "top_enhancers": ";".join(sub["enhancer_id"].head(top_n)),
            "top_importance": ";".join(f"{v:.3f}" for v in sub["importance"].head(top_n)),
            "top_signed_distance": ";".join(str(int(v)) for v in sub["signed_distance"].head(top_n)),
        })
    return pd.DataFrame(rows).sort_values("total_importance", ascending=False).reset_index(drop=True)


__all__ = [
    "RegionScheme", "call_elements", "gene_regulatory_quantification",
    "rank_parts_for_expression", "enhancer_contributions", "summarise_contributions",
    "default_rings", "classify_marks",
]
