"""Enhancers: distal elements, how active they are, and which genes they are linked to.

The gene layer (:mod:`chromstate.features`) reads a gene through its promoter and body.
Most cell-type-specific regulation sits further away, in enhancers 10-500 kb from the
gene they control, and this module is the layer that reaches them. It follows the
Activity-by-Contact logic (Fulco et al., 2019): an element matters to a gene in
proportion to how active it is times how often it touches the gene's promoter.

    candidate_enhancers   peaks -> elements, each flagged promoter or distal
    enhancer_activity     per element and sample: state composition, mark signal, ATAC
    link_enhancers        element-gene pairs within a window, with a contact estimate
    abc_score             activity x contact, as a share of each gene's total input
    abc_change            how each pair's input moved between two groups of samples
    gene_enhancer_features
                          per-gene enhancer input in distance rings, for prediction
    enhancer_importance   in-silico deletion of each enhancer through a trained model

Four decisions worth stating, because each is a place an enhancer analysis quietly goes
wrong:

**Promoter peaks stay in the candidate set.** They are flagged, not dropped. A gene's own
promoter is part of its regulatory input, and leaving it out of the ABC denominator
inflates every distal element's share. Rank distal elements; normalise over all of them.

**Elements are matched to bins by overlap, not by bin start.** An ATAC peak is often
narrower than a 200 bp bin, and the start-inside rule used for promoters would give such
a peak no bins at all.

**State classes come from the emission profile, not from state names.** Automatic names
depend on the mark panel -- the CD8 model, with no H3K4me1, has no state called "EnhA" at
all, and distal acetylation falls in a state named TssFlnk. A state counts as *active*
when the activating mark is enriched over its genome-wide mean and dominates the
repressive marks, *poised* when H3K27me3 is enriched together with any activating mark, and
*repressed* when a repressive mark dominates. :func:`enhancer_state_classes` reports the
call for every state so it can be checked.

**Nothing here correlates one enhancer with one gene across conditions.** With three
conditions that is a correlation over three points. Activity and contact are measured
within a condition, and any learned weighting (:func:`enhancer_importance`) is learned
across genes, where there are thousands of observations.
"""

from __future__ import annotations

import gzip
import io
import warnings
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import sparse

from .features import FeatureSet
from .model.annotate import PRESENT_FOLD, canonical_mark, classify_marks
from .model.core import Segmentation

EPS = 1e-12

#: Distance rings (bp, element centre to TSS) used to summarise a gene's distal input.
DEFAULT_RINGS = (0, 10_000, 50_000, 200_000, 500_000)

_HEADER_ALIASES = {
    "chrom": ("chrom", "chr", "chromosome", "seqnames", "#chrom"),
    "start": ("start", "chromstart"),
    "end": ("end", "chromend"),
    "enhancer_id": ("enhancer_id", "name", "peakn", "peak", "peak_id", "id"),
}


# ---------------------------------------------------------------------------
# Reading and candidate elements
# ---------------------------------------------------------------------------


def _open_text(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def read_peaks(path: str) -> pd.DataFrame:
    """Read peaks from a BED file or from a tab-separated table with a header.

    A headed table (HOMER annotation, a DiffBind export) is recognised when its first
    field names a chromosome column (``Chr``, ``chrom``, ``seqnames``...), and every other
    column is kept -- per-condition ATAC values travel with the peaks that way. A BED
    file takes ``chrom``/``start``/``end`` from the first three columns and the peak name
    from the fourth when there is one.
    """
    with _open_text(path) as fh:
        first = ""
        for line in fh:
            if line.strip() and not line.startswith(("track", "browser")):
                first = line
                break
    if not first:
        raise ValueError(f"No peaks in {path}")

    head = first.rstrip("\n").split("\t")[0].strip().lower()
    if head in _HEADER_ALIASES["chrom"]:
        frame = pd.read_csv(path, sep="\t")
        rename = {}
        for col in frame.columns:
            key = str(col).strip().lower()
            for target, aliases in _HEADER_ALIASES.items():
                if key in aliases and target not in rename.values():
                    rename[col] = target
                    break
        frame = frame.rename(columns=rename)
    else:
        with _open_text(path) as fh:
            body = "".join(line for line in fh if line.strip()
                           and not line.startswith(("#", "track", "browser")))
        frame = pd.read_csv(io.StringIO(body), sep="\t", header=None)
        names = {0: "chrom", 1: "start", 2: "end"}
        if frame.shape[1] > 3:
            names[3] = "enhancer_id"
        frame = frame.rename(columns=names)
        frame.columns = [c if isinstance(c, str) else f"col_{c}" for c in frame.columns]
    return _as_peaks(frame)


def _as_peaks(peaks: pd.DataFrame) -> pd.DataFrame:
    """Validated copy with integer coordinates and a unique ``enhancer_id``."""
    missing = [c for c in ("chrom", "start", "end") if c not in peaks.columns]
    if missing:
        raise ValueError(f"peaks need column(s) {missing}; got {list(peaks.columns)[:10]}")
    out = peaks.copy()
    out["chrom"] = out["chrom"].astype(str)
    out["start"] = out["start"].astype(np.int64)
    out["end"] = out["end"].astype(np.int64)
    if (out["end"] <= out["start"]).any():
        raise ValueError(f"{int((out['end'] <= out['start']).sum())} peak(s) have end <= start")
    if "enhancer_id" not in out.columns:
        out.insert(0, "enhancer_id",
                   out["chrom"] + ":" + out["start"].astype(str) + "-" + out["end"].astype(str))
    out["enhancer_id"] = out["enhancer_id"].astype(str)
    if out["enhancer_id"].duplicated().any():
        dup = out.loc[out["enhancer_id"].duplicated(), "enhancer_id"].iloc[0]
        raise ValueError(f"Duplicate enhancer_id, e.g. {dup!r}")
    return out.reset_index(drop=True)


def _tss_table(features: FeatureSet) -> pd.DataFrame:
    tab = features.table[["gene_id", "name", "chrom", "strand"]].copy()
    tab["chrom"] = tab["chrom"].astype(str)
    tab["tss"] = features.tss().to_numpy().astype(np.int64)
    return tab.reset_index(drop=True)


def candidate_enhancers(
    peaks: pd.DataFrame,
    features: FeatureSet,
    tss_window: int = 2000,
) -> pd.DataFrame:
    """Flag each peak as a promoter or a distal element, and name its nearest gene.

    Parameters
    ----------
    peaks
        ``chrom``/``start``/``end`` and optionally ``enhancer_id`` (see :func:`read_peaks`).
        Extra columns are kept.
    tss_window
        A peak within this distance of *any* annotated TSS is a promoter element.

    Returns
    -------
    The peaks plus ``center``, ``nearest_gene``, ``nearest_tss_distance`` (0 when the TSS
    lies inside the peak), and ``is_promoter``. Promoter elements are kept, not dropped:
    they belong in every gene's ABC denominator (see the module docstring).
    """
    out = _as_peaks(peaks)
    tss = _tss_table(features)
    n = len(out)
    nearest = np.full(n, None, dtype=object)
    dist = np.full(n, np.inf)

    for chrom, idx in out.groupby("chrom").indices.items():
        sub = tss.loc[tss["chrom"] == chrom].sort_values("tss")
        if sub.empty:
            continue
        T = sub["tss"].to_numpy()
        G = sub["gene_id"].to_numpy()
        s = out["start"].to_numpy()[idx]
        e = out["end"].to_numpy()[idx]
        j = np.searchsorted(T, s, side="left")  # first TSS at or after the peak start
        jr = np.minimum(j, len(T) - 1)
        jl = np.maximum(j - 1, 0)
        right = np.where(j < len(T), np.maximum(T[jr] - (e - 1), 0), np.inf)
        left = np.where(j > 0, s - T[jl], np.inf)
        use_right = right <= left
        dist[idx] = np.where(use_right, right, left)
        nearest[idx] = np.where(use_right, G[jr], G[jl])

    out["center"] = (out["start"] + out["end"]) // 2
    out["nearest_gene"] = nearest
    out["nearest_tss_distance"] = dist
    out["is_promoter"] = dist <= tss_window
    out.attrs["tss_window"] = int(tss_window)
    return out


# ---------------------------------------------------------------------------
# Activity
# ---------------------------------------------------------------------------


def _region_bin_matrix(bins: pd.DataFrame, regions: pd.DataFrame):
    """Sparse ``(regions x bins)`` matrix averaging every bin a region overlaps.

    Returns the matrix and the number of bins per region (0 where a region falls off
    the binned genome, e.g. on a chromosome that was not fitted or in a blacklist gap).
    """
    bins = bins.reset_index(drop=True)
    b_start = bins["start"].to_numpy()
    b_end = bins["end"].to_numpy()
    r_chrom = regions["chrom"].astype(str).to_numpy()
    r_start = regions["start"].to_numpy()
    r_end = regions["end"].to_numpy()

    rows, cols = [], []
    for chrom, bidx in bins.groupby(bins["chrom"].astype(str)).indices.items():
        ridx = np.flatnonzero(r_chrom == chrom)
        if not len(ridx):
            continue
        bidx = bidx[np.argsort(b_start[bidx], kind="stable")]
        bs, be = b_start[bidx], b_end[bidx]
        lo = np.searchsorted(be, r_start[ridx], side="right")  # first bin ending after start
        hi = np.searchsorted(bs, r_end[ridx], side="left")     # bins starting before end
        k = np.maximum(hi - lo, 0)
        if not k.sum():
            continue
        offs = np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)
        rows.append(np.repeat(ridx, k))
        cols.append(bidx[np.repeat(lo, k) + offs])

    n_regions = len(regions)
    if not rows:
        return sparse.csr_matrix((n_regions, len(bins))), np.zeros(n_regions, dtype=np.int64)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    counts = np.bincount(rows, minlength=n_regions)
    mat = sparse.csr_matrix((1.0 / counts[rows], (rows, cols)), shape=(n_regions, len(bins)))
    return mat, counts


def enhancer_state_classes(
    seg: Segmentation,
    active_mark: str = "H3K27ac",
    repressive_mark: str = "H3K27me3",
    min_fold: float = PRESENT_FOLD,
    occupancy: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Classify every state as active, poised, repressed, or background from its emissions.

    Fold enrichment is each state's emission mean over the mark's genome-wide mean (the
    occupancy-weighted average of the states), the same scale the automatic names use.
    Pass ``occupancy`` (each state's genome-wide share) when ``seg`` covers only part of
    the genome -- one chromosome at a time, say -- so every part is classified alike.

    ``poised``      ``repressive_mark`` (H3K27me3) at ``min_fold`` or more together with
                    any activating mark (H3K4me1 for a classic poised enhancer, H3K4me3
                    for a bivalent promoter, H3K27ac, ATAC)
    ``active``      ``active_mark`` at ``min_fold`` or more, not poised, and above every
                    repressive mark
    ``repressed``   some repressive mark at ``min_fold`` or more and not active or poised
    ``background``  everything else

    Heterochromatin that carries some acetylation (common at repeats) is therefore
    repressed, not active, because H3K9me3 dominates it. A Polycomb state with a weak
    H3K4me3 shoulder lands in ``poised``; pass an edited table to
    :func:`enhancer_activity` when a call should be overridden.
    """
    marks = list(seg.marks)
    canon = [canonical_mark(m) for m in marks]
    a_key, r_key = canonical_mark(active_mark), canonical_mark(repressive_mark)
    if a_key not in canon:
        raise ValueError(f"{active_mark} is not in the mark panel {marks}")

    M = seg.model.emissions.mean_matrix()
    occ = np.asarray(occupancy, dtype=np.float64) if occupancy is not None \
        else seg.posterior.mean(axis=0)
    occ = occ / max(occ.sum(), EPS)
    F = M / np.maximum((occ[:, None] * M).sum(axis=0), EPS)

    a = F[:, canon.index(a_key)]
    groups = classify_marks(marks)
    any_act = F[:, groups["active"]].max(axis=1)
    r_max = F[:, groups["repressive"]].max(axis=1) if groups["repressive"] else np.zeros(len(a))
    k27 = F[:, canon.index(r_key)] if r_key in canon else np.zeros(len(a))

    poised = (any_act >= min_fold) & (k27 >= min_fold)
    active = (a >= min_fold) & ~poised & (a > r_max)
    repressed = (r_max >= min_fold) & ~poised & ~active
    cls = np.where(poised, "poised", np.where(active, "active",
                   np.where(repressed, "repressed", "background")))

    out = pd.DataFrame({"state": np.arange(seg.n_states), "label": seg.labels, "class": cls})
    for j, m in enumerate(marks):
        out[f"fold_{m}"] = F[:, j]
    return out


def _wide_by_sample(table: pd.DataFrame, samples: Sequence[str], what: str) -> pd.DataFrame:
    df = table.copy()
    if "enhancer_id" in df.columns:
        df = df.set_index("enhancer_id")
    df.index = df.index.astype(str)
    missing = [s for s in samples if s not in df.columns]
    if missing:
        raise ValueError(
            f"{what} has no column for sample(s) {missing}; columns must be named after "
            f"the dataset's samples {list(samples)}"
        )
    return df[list(samples)].astype(float)


def enhancer_activity(
    seg: Segmentation,
    enhancers: pd.DataFrame,
    atac: Optional[pd.DataFrame] = None,
    active_mark: str = "H3K27ac",
    repressive_mark: str = "H3K27me3",
    flank: int = 0,
    state_classes: Optional[pd.DataFrame] = None,
    include_posterior: bool = True,
) -> pd.DataFrame:
    """Per element and sample: state composition, mark signal, ATAC, and activity.

    Parameters
    ----------
    enhancers
        Output of :func:`candidate_enhancers` (or any table :func:`read_peaks` accepts).
    atac
        Optional accessibility per element: indexed by ``enhancer_id`` (or carrying it
        as a column) with one column per sample, named as in the dataset. Use normalised
        values (CPM or DESeq2-scaled), not raw counts, since libraries differ in depth.
    flank
        Extend each element by this many bp on both sides before collecting bins.
    state_classes
        Override the automatic classes from :func:`enhancer_state_classes`.

    Returns
    -------
    Long table, one row per (``enhancer_id``, ``sample``):

    ``n_bins``                       bins the element overlaps (0: off the binned genome)
    ``active_mass``, ``poised_mass``, ``repressed_mass``
                                     posterior share in each class of state
    ``entropy``                      normalised entropy of the element's mean composition
    ``signal_<mark>``                mean depth-adjusted signal (count / size factor)
    ``atac``                         when given
    ``activity``                     ``sqrt(atac x signal_<active_mark>)``, the ABC
                                     activity; ``signal_<active_mark>`` alone without ATAC
    ``activity_state``               ``activity x active_mass``: activity counted only as
                                     far as the model places the element in an active state
    ``post_<state>``                 the full composition, with ``include_posterior``
    """
    ds = seg.dataset
    enh = _as_peaks(enhancers)
    regions = enh[["chrom", "start", "end"]].copy()
    regions["start"] = np.maximum(regions["start"] - flank, 0)
    regions["end"] = regions["end"] + flank

    classes = state_classes if state_classes is not None else enhancer_state_classes(
        seg, active_mark, repressive_mark)
    cls = classes.set_index("state")["class"].reindex(range(seg.n_states)).to_numpy()
    canon = [canonical_mark(m) for m in ds.marks]
    a_col = canon.index(canonical_mark(active_mark))

    atac_wide = None
    if atac is not None:
        atac_wide = _wide_by_sample(atac, ds.samples, "atac").reindex(enh["enhancer_id"])

    signal = ds.X / np.maximum(ds.size_factors, EPS)
    log_k = np.log(seg.n_states) if seg.n_states > 1 else 1.0
    frames = []
    for sample in ds.samples:
        mask = ds.sample_mask(sample)
        mat, n_bins = _region_bin_matrix(ds.bins.loc[mask, ["chrom", "start", "end"]], regions)
        comp = np.asarray(mat @ seg.posterior[mask])
        sig = np.asarray(mat @ signal[mask])
        off = n_bins == 0
        comp[off] = np.nan
        sig[off] = np.nan

        frame = pd.DataFrame({"enhancer_id": enh["enhancer_id"].to_numpy(), "sample": sample,
                              "n_bins": n_bins})
        for name in ("active", "poised", "repressed"):
            frame[f"{name}_mass"] = comp[:, cls == name].sum(axis=1)
        with np.errstate(invalid="ignore"):
            p = np.clip(comp, EPS, 1.0)
            frame["entropy"] = -(p * np.log(p)).sum(axis=1) / log_k
        for j, mark in enumerate(ds.marks):
            frame[f"signal_{mark}"] = sig[:, j]

        acetyl = sig[:, a_col]
        if atac_wide is not None:
            acc = atac_wide[sample].to_numpy()
            frame["atac"] = acc
            frame["activity"] = np.sqrt(np.clip(acc, 0, None) * np.clip(acetyl, 0, None))
        else:
            frame["activity"] = acetyl
        frame["activity_state"] = frame["activity"] * frame["active_mass"]

        if include_posterior:
            for k, lbl in enumerate(seg.labels):
                frame[f"post_{lbl}"] = comp[:, k]
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True)
    out.attrs["state_classes"] = classes
    out.attrs["active_mark"] = active_mark
    out.attrs["n_off_genome"] = int((out["n_bins"] == 0).sum())
    if atac_wide is not None:
        out.attrs["n_missing_atac"] = int(atac_wide.isna().any(axis=1).sum())
    return out


# ---------------------------------------------------------------------------
# Linking and ABC
# ---------------------------------------------------------------------------


def powerlaw_contact(distance: np.ndarray, gamma: float = 1.0, min_distance: int = 5_000) -> np.ndarray:
    """Contact frequency falling as ``distance ** -gamma``, scaled to 1 at ``min_distance``.

    The distance-only stand-in for Hi-C. Exponents near 1 match the average decay of
    contact frequency in mammalian Hi-C over the 10 kb-1 Mb range; below
    ``min_distance`` contact is held at 1, since every element that close is effectively
    in contact with the promoter.
    """
    d = np.maximum(np.asarray(distance, dtype=np.float64), min_distance)
    return (d / min_distance) ** (-gamma)


def link_enhancers(
    enhancers: pd.DataFrame,
    features: FeatureSet,
    max_distance: int = 500_000,
    contact: Union[str, Callable[[pd.DataFrame], np.ndarray]] = "powerlaw",
    gamma: float = 1.0,
    min_distance: int = 5_000,
    tss_window: Optional[int] = None,
) -> pd.DataFrame:
    """Every element-gene pair with the element centre within ``max_distance`` of the TSS.

    Parameters
    ----------
    contact
        ``"powerlaw"`` (see :func:`powerlaw_contact`), or a function that takes the pair
        table -- ``chrom``, ``center``, ``tss`` among its columns -- and returns one
        contact value per row. That is the hook for a Hi-C or HiChIP matrix.
    tss_window
        Distance within which an element is the gene's *own* promoter. Defaults to the
        window :func:`candidate_enhancers` used.

    Returns
    -------
    One row per pair: ``enhancer_id``, ``gene_id``, ``name``, ``chrom``, ``center``,
    ``tss``, ``distance`` (centre to TSS, bp), ``signed_distance`` (negative upstream of
    the TSS on the gene's strand, so Pdcd1's "-23 kb" enhancer reads as -23,000),
    ``contact``, and ``element_type``: ``own_promoter``, ``other_promoter`` (another
    gene's promoter), or ``distal``.
    """
    enh = enhancers if {"is_promoter", "center"} <= set(enhancers.columns) \
        else candidate_enhancers(enhancers, features)
    enh = _as_peaks(enh)
    if tss_window is None:
        tss_window = int(enhancers.attrs.get("tss_window", 2000))
    tss = _tss_table(features)

    frames = []
    for chrom, idx in enh.groupby("chrom").indices.items():
        sub = tss.loc[tss["chrom"] == chrom].sort_values("tss").reset_index(drop=True)
        if sub.empty:
            continue
        T = sub["tss"].to_numpy()
        c = enh["center"].to_numpy()[idx]
        lo = np.searchsorted(T, c - max_distance, side="left")
        hi = np.searchsorted(T, c + max_distance, side="right")
        k = hi - lo
        if not k.sum():
            continue
        offs = np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)
        e_rows = np.repeat(idx, k)
        g_rows = np.repeat(lo, k) + offs
        frames.append(pd.DataFrame({
            "enhancer_id": enh["enhancer_id"].to_numpy()[e_rows],
            "gene_id": sub["gene_id"].to_numpy()[g_rows],
            "name": sub["name"].to_numpy()[g_rows],
            "chrom": chrom,
            "start": enh["start"].to_numpy()[e_rows],
            "end": enh["end"].to_numpy()[e_rows],
            "center": enh["center"].to_numpy()[e_rows],
            "tss": T[g_rows],
            "strand": sub["strand"].to_numpy()[g_rows],
            "is_promoter": enh["is_promoter"].to_numpy()[e_rows],
        }))
    if not frames:
        raise ValueError(
            "No element lies within max_distance of any TSS. The usual cause is a "
            "chromosome naming mismatch ('chr1' versus '1') between peaks and annotation."
        )
    pairs = pd.concat(frames, ignore_index=True)

    offset = pairs["center"] - pairs["tss"]
    pairs["distance"] = offset.abs()
    pairs["signed_distance"] = np.where(pairs["strand"] == "-", -offset, offset)
    to_tss = np.maximum.reduce([pairs["start"] - pairs["tss"],
                                pairs["tss"] - (pairs["end"] - 1),
                                np.zeros(len(pairs), dtype=np.int64)])
    pairs["element_type"] = np.where(
        to_tss <= tss_window, "own_promoter",
        np.where(pairs["is_promoter"], "other_promoter", "distal"))

    if contact == "powerlaw":
        pairs["contact"] = powerlaw_contact(pairs["distance"].to_numpy(), gamma, min_distance)
    elif callable(contact):
        values = np.asarray(contact(pairs), dtype=np.float64)
        if values.shape != (len(pairs),):
            raise ValueError(f"contact function returned shape {values.shape}, "
                             f"expected ({len(pairs)},)")
        pairs["contact"] = values
    else:
        raise ValueError(f"contact must be 'powerlaw' or a function, got {contact!r}")

    cols = ["enhancer_id", "gene_id", "name", "chrom", "center", "tss", "distance",
            "signed_distance", "contact", "element_type"]
    out = pairs[cols]
    out.attrs["max_distance"] = int(max_distance)
    out.attrs["contact"] = contact if isinstance(contact, str) else "custom"
    return out


def abc_score(
    pairs: pd.DataFrame,
    activity: pd.DataFrame,
    activity_col: str = "activity",
) -> pd.DataFrame:
    """Activity x contact for every pair and sample, and its share of the gene's total.

    The table stays one row per pair, with three columns per sample, because pairs times
    samples runs to millions of rows on a real genome:

    ``activity_<sample>``   the element's activity (missing activity counts as 0)
    ``abc_raw_<sample>``    activity x contact, the element's absolute input to the gene
    ``abc_<sample>``        ``abc_raw`` over the sum of ``abc_raw`` across every element
                            linked to that gene, own promoter included: the share of the
                            gene's input this element supplies

    Use ``abc`` to rank elements *within* a gene and ``abc_raw`` to compare a pair
    *between* conditions -- a share can rise only because a neighbour fell.
    """
    if activity_col not in activity.columns:
        raise ValueError(f"activity has no {activity_col!r} column")
    samples = list(pd.unique(activity["sample"]))
    wide = activity.set_index(["enhancer_id", "sample"])[activity_col].unstack("sample")
    wide = wide.reindex(columns=samples)

    out = pairs.copy()
    e_codes = wide.index.get_indexer(out["enhancer_id"])
    g_codes, genes = pd.factorize(out["gene_id"])
    contact = out["contact"].to_numpy()
    n_missing = int((e_codes < 0).sum())

    for s in samples:
        act = np.where(e_codes >= 0, wide[s].to_numpy()[np.maximum(e_codes, 0)], np.nan)
        act = np.nan_to_num(act, nan=0.0)
        raw = act * contact
        total = np.bincount(g_codes, weights=raw, minlength=len(genes))
        out[f"activity_{s}"] = act
        out[f"abc_raw_{s}"] = raw
        out[f"abc_{s}"] = np.where(total[g_codes] > 0, raw / np.maximum(total[g_codes], EPS), 0.0)

    out.attrs.update(pairs.attrs)
    out.attrs["samples"] = samples
    out.attrs["activity_col"] = activity_col
    out.attrs["n_pairs_without_activity"] = n_missing
    return out


def abc_change(
    scores: pd.DataFrame,
    group_a: Sequence[str],
    group_b: Sequence[str],
    pseudocount: Optional[float] = None,
) -> pd.DataFrame:
    """How each pair's input moved between two groups of samples (B minus A).

    Adds ``activity_a/_b``, ``abc_raw_a/_b``, ``abc_a/_b`` (group means), ``delta_abc``,
    ``delta_abc_raw``, and ``log2fc_activity``. The pseudocount for the fold change
    defaults to a tenth of the median positive activity, because activity has no natural
    unit and a fixed pseudocount of 1 means something different on every scale.
    """
    group_a, group_b = list(group_a), list(group_b)
    for s in group_a + group_b:
        if f"abc_raw_{s}" not in scores.columns:
            raise ValueError(f"Sample {s!r} not in the ABC table; it has {scores.attrs.get('samples')}")

    out = scores[[c for c in scores.columns
                  if not c.startswith(("activity_", "abc_raw_", "abc_"))]].copy()
    for tag, group in (("a", group_a), ("b", group_b)):
        for kind in ("activity", "abc_raw", "abc"):
            out[f"{kind}_{tag}"] = scores[[f"{kind}_{s}" for s in group]].mean(axis=1)

    if pseudocount is None:
        pos = np.concatenate([out["activity_a"].to_numpy(), out["activity_b"].to_numpy()])
        pos = pos[pos > 0]
        pseudocount = 0.1 * float(np.median(pos)) if pos.size else 1.0
    out["delta_abc"] = out["abc_b"] - out["abc_a"]
    out["delta_abc_raw"] = out["abc_raw_b"] - out["abc_raw_a"]
    out["log2fc_activity"] = np.log2((out["activity_b"] + pseudocount)
                                     / (out["activity_a"] + pseudocount))
    out.attrs.update(scores.attrs)
    out.attrs["group_a"], out.attrs["group_b"] = group_a, group_b
    out.attrs["pseudocount"] = float(pseudocount)
    return out.sort_values("delta_abc_raw", key=np.abs, ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Gene-level features and importance
# ---------------------------------------------------------------------------


def _ring_index(distance: np.ndarray, rings: Sequence[int]) -> np.ndarray:
    """Ring of each distance (``[lo, hi)``; the outer edge inclusive), -1 outside."""
    edges = np.asarray(rings)
    idx = np.searchsorted(edges, distance, side="right") - 1
    idx = np.where(distance == edges[-1], len(edges) - 2, idx)
    idx[(distance < edges[0]) | (distance > edges[-1])] = -1
    return idx


def _ring_edge_label(bp: int, kb: bool) -> str:
    bp = int(bp)
    if bp == 0:
        return "0"
    return f"{bp // 1000}" if kb else (f"{bp // 1000}kb" if bp % 1000 == 0 else f"{bp}bp")


def ring_columns(rings: Sequence[int] = DEFAULT_RINGS) -> List[str]:
    """Column names :func:`gene_enhancer_features` uses for each distance ring.

    Edges that are all whole kilobases keep the historical ``enh_<lo>_<hi>kb`` form
    (``enh_0_10kb``). Any other edge (a 400 bp ring around a short regulatory window,
    say) is written in base pairs, so two different rings never share a name.
    """
    rings = [int(r) for r in rings]
    if len(set(rings)) != len(rings) or rings != sorted(rings):
        raise ValueError(f"rings must be strictly increasing, got {tuple(rings)}")
    kb = all(r % 1000 == 0 for r in rings)
    if kb:
        return [f"enh_{lo // 1000}_{hi // 1000}kb" for lo, hi in zip(rings[:-1], rings[1:])]
    return [f"enh_{_ring_edge_label(lo, False)}_{_ring_edge_label(hi, False)}"
            for lo, hi in zip(rings[:-1], rings[1:])]


def gene_enhancer_features(
    scores: pd.DataFrame,
    rings: Sequence[int] = DEFAULT_RINGS,
) -> pd.DataFrame:
    """Per (gene, sample): distal input summed in distance rings, plus the own promoter.

    ``enh_<lo>_<hi>kb`` is ``log1p`` of the summed ``abc_raw`` of distal elements whose
    centre lies in that ring; ``prom_input`` is the same for the gene's own promoter
    elements; ``n_distal`` counts linked distal elements. Every feature is a sum over
    elements, which is what lets :func:`enhancer_importance` delete one element exactly.

    ``n_distal`` is a diagnostic, not a model input for importance: a model that uses
    the peak count treats it as an arbitrary signal, and deleting one element then moves
    the prediction through the count rather than through the element's activity.
    Join with promoter state features from :func:`chromstate.features.gene_state_matrix`
    on ``gene_id`` and ``sample`` to build a promoter-plus-enhancer model.
    """
    samples = scores.attrs.get("samples") or [
        c[len("abc_raw_"):] for c in scores.columns if c.startswith("abc_raw_")]
    g_codes, genes = pd.factorize(scores["gene_id"])
    n_genes, cols = len(genes), ring_columns(rings)
    distal = (scores["element_type"] == "distal").to_numpy()
    own = (scores["element_type"] == "own_promoter").to_numpy()
    ring = _ring_index(scores["distance"].to_numpy(), rings)
    in_ring = distal & (ring >= 0)

    frames = []
    for s in samples:
        raw = scores[f"abc_raw_{s}"].to_numpy()
        frame = pd.DataFrame({"gene_id": genes, "sample": s})
        for r, col in enumerate(cols):
            sel = in_ring & (ring == r)
            frame[col] = np.log1p(np.bincount(g_codes[sel], weights=raw[sel], minlength=n_genes))
        frame["prom_input"] = np.log1p(np.bincount(g_codes[own], weights=raw[own], minlength=n_genes))
        frame["n_distal"] = np.bincount(g_codes[in_ring], minlength=n_genes)
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out.attrs["rings"] = tuple(rings)
    return out


def enhancer_importance(
    model,
    features: pd.DataFrame,
    scores: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
    rings: Sequence[int] = DEFAULT_RINGS,
    samples: Optional[Sequence[str]] = None,
    chunk: int = 200_000,
) -> pd.DataFrame:
    """In-silico deletion: how much the predicted expression falls without each enhancer.

    For every distal pair, the gene's feature row is rebuilt with that one element
    removed from its ring, and the trained model predicts again. ``importance`` is the
    full prediction minus the deleted one, in the units the model predicts (for a model
    of log expression, a log change).

    Parameters
    ----------
    model
        Anything with ``predict``, trained on ``features[feature_columns]``. The ring
        columns from :func:`gene_enhancer_features` must be among them; other columns
        (promoter state, mark signal) are held fixed. Constrain the model to be monotone
        increasing in the ring columns (``monotonic_cst`` in scikit-learn's gradient
        boosting); an unconstrained tree model learns non-monotone steps and reports
        negative importance for genuine enhancers.
    features
        The per-(gene, sample) table the model was trained on.

    What this can and cannot say: the model learns how much each *ring* of input is
    worth, across all genes. Within one gene, two elements in the same ring are ordered
    by their ``abc_raw`` alone; the model adds the non-linear weighting between rings and
    against the promoter. The result is a ranking of candidates for perturbation, not a
    measured effect. Deletion also understates redundant enhancers: a gene with many
    strong elements barely moves when any one is removed, so highly induced genes can
    show near-zero importance for every element.
    """
    if feature_columns is None:
        feature_columns = list(getattr(model, "feature_names_in_", []))
        if not feature_columns:
            raise ValueError("feature_columns is required for a model without feature_names_in_")
    feature_columns = list(feature_columns)
    cols = ring_columns(rings)
    missing = [c for c in cols if c not in feature_columns]
    if missing:
        raise ValueError(f"model features lack ring column(s) {missing}; "
                         f"was it trained on gene_enhancer_features with these rings?")
    ring_pos = np.array([feature_columns.index(c) for c in cols])
    count_pos = feature_columns.index("n_distal") if "n_distal" in feature_columns else None
    if count_pos is not None:
        warnings.warn(
            "The model uses n_distal. Deleting an element lowers that count, and the "
            "model's response to a peak count is not an effect of the element; train the "
            "importance model without n_distal.",
            RuntimeWarning, stacklevel=2,
        )

    samples = list(samples) if samples is not None else list(scores.attrs.get("samples") or
                                                              pd.unique(features["sample"]))
    ring = _ring_index(scores["distance"].to_numpy(), rings)
    keep = (scores["element_type"] == "distal").to_numpy() & (ring >= 0)
    pairs = scores.loc[keep].reset_index(drop=True)
    ring = ring[keep]

    frames = []
    for s in samples:
        F = features.loc[features["sample"] == s].set_index("gene_id")
        g_idx = F.index.get_indexer(pairs["gene_id"])
        ok = g_idx >= 0
        if not ok.any():
            continue
        sub = pairs.loc[ok].reset_index(drop=True)
        g_idx, r_idx = g_idx[ok], ring[ok]
        raw = sub[f"abc_raw_{s}"].to_numpy()

        X_genes = F[feature_columns].to_numpy(dtype=np.float64)
        full_gene = _predict(model, X_genes, feature_columns, chunk)

        pred_del = np.empty(len(sub))
        for lo in range(0, len(sub), chunk):
            sl = slice(lo, lo + chunk)
            X = X_genes[g_idx[sl]].copy()
            rows = np.arange(X.shape[0])
            pos = ring_pos[r_idx[sl]]
            X[rows, pos] = np.log1p(np.maximum(np.expm1(X[rows, pos]) - raw[sl], 0.0))
            if count_pos is not None:
                X[:, count_pos] -= 1
            pred_del[sl] = _predict(model, X, feature_columns, chunk)

        frame = sub[["enhancer_id", "gene_id", "name", "chrom", "distance",
                     "signed_distance"]].copy()
        frame["sample"] = s
        frame["abc_raw"] = raw
        frame["abc"] = sub[f"abc_{s}"].to_numpy()
        frame["pred_full"] = full_gene[g_idx]
        frame["pred_deleted"] = pred_del
        frame["importance"] = frame["pred_full"] - frame["pred_deleted"]
        frames.append(frame)

    if not frames:
        raise ValueError("No linked gene appears in the feature table for the requested samples")
    out = pd.concat(frames, ignore_index=True)
    out["rank_in_gene"] = (out.groupby(["gene_id", "sample"])["importance"]
                              .rank(ascending=False, method="first").astype(int))
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


class MonotoneLinear:
    """Linear model with sign constraints per column, for small gene tables.

    ``monotonic_cst`` follows scikit-learn: ``1`` forces a non-negative coefficient,
    ``-1`` a non-positive one, ``0`` leaves it free. Columns are standardised before the
    bounded least-squares fit (a small ridge ``alpha`` keeps it stable), so the
    constraint is on the sign of the effect and not on its scale. Gradient boosting
    needs hundreds of rows before its trees see more than one split per feature; with
    fewer (gene, sample) rows the deletion importance it reports collapses to one
    constant per ring. This model gives an importance proportional to the element's own
    input, which is the right answer when the data cannot support more.
    """

    def __init__(self, monotonic_cst: Optional[Sequence[int]] = None, alpha: float = 1e-3):
        self.monotonic_cst = None if monotonic_cst is None else np.asarray(monotonic_cst, dtype=int)
        self.alpha = float(alpha)

    def fit(self, X, y) -> "MonotoneLinear":
        from scipy.optimize import lsq_linear

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, p = X.shape
        mono = np.zeros(p, dtype=int) if self.monotonic_cst is None else self.monotonic_cst
        if len(mono) != p:
            raise ValueError(f"monotonic_cst has {len(mono)} entries for {p} columns")
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        Z = (X - mu) / sd
        y_mean = float(y.mean())
        A = np.vstack([Z, np.sqrt(self.alpha) * np.eye(p)])
        b = np.concatenate([y - y_mean, np.zeros(p)])
        lb = np.where(mono > 0, 0.0, -np.inf)
        ub = np.where(mono < 0, 0.0, np.inf)
        res = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol="auto")
        self.coef_ = res.x / sd
        self.intercept_ = y_mean - float(mu @ self.coef_)
        self.n_features_in_ = p
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_


def make_importance_model(model: str, monotonic_cst: Sequence[int], n_rows: int, seed: int = 0,
                          min_rows_for_trees: int = 200):
    """The regression behind :func:`enhancer_importance`.

    ``"hgb"`` is monotone gradient boosting, ``"linear"`` :class:`MonotoneLinear`, and
    ``"auto"`` (the default upstream) picks boosting once there are at least
    ``min_rows_for_trees`` (gene, sample) rows and the linear model below that.
    """
    mono = np.asarray(monotonic_cst, dtype=int)
    if model == "auto":
        model = "hgb" if n_rows >= min_rows_for_trees else "linear"
    if model == "linear":
        return MonotoneLinear(mono), "linear"
    if model == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor

        leaf = int(min(20, max(5, n_rows // 10)))
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, monotonic_cst=mono,
                                             min_samples_leaf=leaf, random_state=seed), "hgb"
    raise ValueError(f"model must be 'auto', 'hgb' or 'linear', got {model!r}")


def _predict(model, X: np.ndarray, columns: Sequence[str], chunk: int) -> np.ndarray:
    named = hasattr(model, "feature_names_in_")
    out = np.empty(X.shape[0])
    for lo in range(0, X.shape[0], chunk):
        block = X[lo:lo + chunk]
        if named:
            block = pd.DataFrame(block, columns=list(columns))
        out[lo:lo + chunk] = np.asarray(model.predict(block), dtype=np.float64).ravel()
    return out
