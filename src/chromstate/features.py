"""Genes, promoters, and the bin-to-feature mapping that makes a result traceable.

Coordinates are not an answer. "Bin chr7:1,234,600-1,234,800 gained Active TSS mass" is
not something a bench biologist can act on, and it cannot be joined to expression, to a
knockout, or to a gene list. This module is the layer that turns bins into named features
and back again, so every quantitative claim can be followed to a gene and every gene can
be followed back to the bins, marks, and counts that produced its number.

Two design decisions worth stating, because both are places where packages quietly lose
their traceability:

**Promoter and gene body stay separate.** They carry different states by construction --
an active promoter is H3K4me3 at the TSS while transcription is H3K36me3 across the body
-- so a single "gene-level state" averages two different biological readouts into one
uninterpretable number. Everything here is computed per *part*, and the part is a column.

**A bin overlapping several genes is not silently resolved.** The default maps a bin to
every feature it overlaps, so a shared bidirectional promoter contributes to both genes
and the double-counting is visible in the output. ``assign="nearest_tss"`` picks one, and
which rule was used is recorded on the result rather than left to the reader to guess.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .model.core import Segmentation

REQUIRED = ["gene_id", "chrom", "start", "end", "strand"]


def _open_text(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


@dataclass
class FeatureSet:
    """Genes (or any named intervals) with strand, plus the windows derived from them.

    ``table`` carries ``gene_id``, ``name``, ``chrom``, ``start``, ``end``, ``strand``.
    Coordinates are 0-based half-open, matching the bin table.
    """

    table: pd.DataFrame
    source: str = "custom"

    def __post_init__(self) -> None:
        missing = [c for c in REQUIRED if c not in self.table.columns]
        if missing:
            raise ValueError(f"FeatureSet is missing column(s): {', '.join(missing)}")
        if "name" not in self.table.columns:
            self.table = self.table.assign(name=self.table["gene_id"])
        bad = self.table["end"] <= self.table["start"]
        if bad.any():
            raise ValueError(f"{int(bad.sum())} feature(s) have end <= start")
        if self.table["gene_id"].duplicated().any():
            dup = self.table.loc[self.table["gene_id"].duplicated(), "gene_id"].iloc[0]
            raise ValueError(f"Duplicate gene_id in FeatureSet, e.g. {dup!r}")
        self.table = self.table.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.table)

    @property
    def genes(self) -> List[str]:
        return list(self.table["gene_id"])

    # -- readers ------------------------------------------------------------

    @classmethod
    def from_gtf(
        cls,
        path: str,
        feature: str = "gene",
        id_attribute: str = "gene_id",
        name_attribute: str = "gene_name",
    ) -> "FeatureSet":
        """Read a GTF/GFF, keeping rows whose third column is ``feature``.

        GTF is 1-based inclusive; the start is converted to 0-based half-open here so the
        coordinates line up with the bin table without an off-by-one that would shift
        every promoter window by a base.
        """
        rows = []
        attr_re = re.compile(r'(\S+)\s+"([^"]*)"')
        with _open_text(path) as fh:
            for line in fh:
                if not line.strip() or line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9 or parts[2] != feature:
                    continue
                attrs = dict(attr_re.findall(parts[8]))
                gene_id = attrs.get(id_attribute)
                if gene_id is None:
                    continue
                rows.append(
                    {
                        "gene_id": gene_id,
                        "name": attrs.get(name_attribute, gene_id),
                        "chrom": parts[0],
                        "start": int(parts[3]) - 1,
                        "end": int(parts[4]),
                        "strand": parts[6] if parts[6] in "+-" else "+",
                    }
                )
        if not rows:
            raise ValueError(f"No {feature!r} records with a {id_attribute!r} attribute in {path}")
        frame = pd.DataFrame(rows).drop_duplicates("gene_id")
        return cls(frame, source=path)

    @classmethod
    def from_bed(cls, path: str) -> "FeatureSet":
        """Read BED6 (``chrom start end name score strand``); BED4 defaults to ``+``."""
        rows = []
        with _open_text(path) as fh:
            for line in fh:
                if not line.strip() or line.startswith(("#", "track", "browser")):
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) < 4:
                    continue
                rows.append(
                    {
                        "gene_id": f[3],
                        "name": f[3],
                        "chrom": f[0],
                        "start": int(f[1]),
                        "end": int(f[2]),
                        "strand": f[5] if len(f) > 5 and f[5] in "+-" else "+",
                    }
                )
        if not rows:
            raise ValueError(f"No usable BED records in {path}")
        return cls(pd.DataFrame(rows).drop_duplicates("gene_id"), source=path)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, source: str = "frame") -> "FeatureSet":
        return cls(frame.copy(), source=source)

    # -- derived windows ----------------------------------------------------

    def tss(self) -> pd.Series:
        """Transcription start site, strand-aware."""
        return pd.Series(
            np.where(self.table["strand"] == "-", self.table["end"] - 1, self.table["start"]),
            index=self.table.index,
            name="tss",
        )

    def promoters(self, upstream: int = 2000, downstream: int = 2000) -> pd.DataFrame:
        """Windows around the TSS. ``upstream`` is 5' of the TSS on the gene's strand."""
        tss = self.tss().to_numpy()
        minus = (self.table["strand"] == "-").to_numpy()
        start = np.where(minus, tss - downstream, tss - upstream)
        end = np.where(minus, tss + upstream, tss + downstream)
        out = self.table[["gene_id", "name", "chrom", "strand"]].copy()
        out["start"] = np.maximum(start, 0)
        out["end"] = np.maximum(end, 1)
        out["part"] = "promoter"
        return out[["gene_id", "name", "chrom", "start", "end", "strand", "part"]]

    def bodies(self, skip_promoter: int = 2000, min_length: int = 1000) -> pd.DataFrame:
        """Gene bodies with the promoter window trimmed off the 5' end.

        Trimming matters: leaving the TSS inside the body makes every expressed gene's
        body look partly like an active promoter, and the body's whole purpose here is to
        report transcription-associated states independently of the promoter.
        """
        minus = (self.table["strand"] == "-").to_numpy()
        start = np.where(minus, self.table["start"], self.table["start"] + skip_promoter)
        end = np.where(minus, self.table["end"] - skip_promoter, self.table["end"])
        out = self.table[["gene_id", "name", "chrom", "strand"]].copy()
        out["start"] = np.maximum(start, 0)
        out["end"] = end
        out["part"] = "body"
        out = out.loc[out["end"] - out["start"] >= min_length]
        return out[["gene_id", "name", "chrom", "start", "end", "strand", "part"]].reset_index(drop=True)

    def regions(self, part: str = "promoter", **kwargs) -> pd.DataFrame:
        """``promoter``, ``body``, or ``gene`` (the full annotated interval)."""
        if part == "promoter":
            return self.promoters(**kwargs)
        if part == "body":
            return self.bodies(**kwargs)
        if part == "gene":
            out = self.table[["gene_id", "name", "chrom", "start", "end", "strand"]].copy()
            out["part"] = "gene"
            return out
        raise ValueError(f"part must be 'promoter', 'body', or 'gene', got {part!r}")


# ---------------------------------------------------------------------------
# Bin <-> feature mapping
# ---------------------------------------------------------------------------


def map_bins_to_regions(
    bins: pd.DataFrame,
    regions: pd.DataFrame,
    assign: str = "all",
) -> pd.DataFrame:
    """Long-form ``(bin_row, gene_id)`` overlap table, matched by coordinate.

    A bin belongs to a region when the bin's start falls inside it -- the same rule the
    tool-comparison harness uses to read a segmentation back, so bins are never
    double-counted at a boundary.

    ``assign``
        ``all`` keeps every overlap, so a shared promoter contributes to both genes and
        the duplication is visible downstream. ``nearest_tss`` keeps, for each bin, only
        the region whose TSS is closest, which produces a partition at the cost of
        discarding real shared regulation.
    """
    if assign not in ("all", "nearest_tss"):
        raise ValueError(f"assign must be 'all' or 'nearest_tss', got {assign!r}")

    bins = bins.reset_index(drop=True)
    index: Dict[str, tuple] = {}
    for chrom, grp in bins.groupby("chrom", sort=False):
        starts = grp["start"].to_numpy()
        rows = grp.index.to_numpy()
        order = np.argsort(starts, kind="stable")
        index[str(chrom)] = (starts[order], rows[order])

    frames = []
    for rec in regions.itertuples(index=False):
        entry = index.get(str(rec.chrom))
        if entry is None:
            continue
        starts, rows = entry
        lo = int(np.searchsorted(starts, rec.start, side="left"))
        hi = int(np.searchsorted(starts, rec.end, side="left"))
        if hi <= lo:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "bin_row": rows[lo:hi],
                    "bin_start": starts[lo:hi],
                    "gene_id": rec.gene_id,
                    "part": getattr(rec, "part", "region"),
                    "region_start": rec.start,
                    "region_end": rec.end,
                }
            )
        )
    if not frames:
        raise ValueError(
            "No bin overlaps any region. The usual cause is a chromosome naming mismatch "
            "('chr1' versus '1') or an assembly difference between the annotation and the bins."
        )
    out = pd.concat(frames, ignore_index=True)

    if assign == "nearest_tss":
        centre = (out["region_start"] + out["region_end"]) / 2.0
        out["distance"] = (out["bin_start"] - centre).abs()
        out = (out.sort_values(["bin_row", "distance"])
                  .drop_duplicates("bin_row")
                  .drop(columns="distance")
                  .reset_index(drop=True))
    return out


# ---------------------------------------------------------------------------
# Gene-level state readouts
# ---------------------------------------------------------------------------


def gene_state_matrix(
    seg: Segmentation,
    features: FeatureSet,
    part: str = "promoter",
    assign: str = "all",
    min_bins: int = 1,
    include_marks: bool = True,
    **region_kwargs,
) -> pd.DataFrame:
    """State composition per (gene, sample) -- the table that joins to expression.

    One row per gene per sample, carrying the mean posterior over that gene's bins for
    every state, the normalised entropy, the dominant state, and (by default) the mean
    normalised signal for every mark. ``gene_id`` is the join key for RNA.

    Averaging posteriors across a gene's bins, rather than taking the modal hard call, is
    deliberate: a promoter that is half active and half bivalent across its window is a
    real and common configuration, and a modal call erases it.
    """
    ds = seg.dataset
    regions = features.regions(part, **region_kwargs)
    per_sample = []

    for sample in ds.samples:
        mask = ds.sample_mask(sample)
        bins = ds.bins.loc[mask, ["chrom", "start", "end"]].reset_index(drop=True)
        post = seg.posterior[mask]
        signal = ds.X[mask] / np.maximum(ds.size_factors[mask], 1e-9)

        pairs = map_bins_to_regions(bins, regions, assign=assign)
        grouped = pairs.groupby("gene_id")["bin_row"]

        rows = []
        for gene_id, idx in grouped:
            rows_idx = idx.to_numpy()
            if len(rows_idx) < min_bins:
                continue
            comp = post[rows_idx].mean(axis=0)
            ent = -(comp * np.log(np.clip(comp, 1e-12, 1.0))).sum()
            ent /= np.log(seg.n_states) if seg.n_states > 1 else 1.0
            row = {
                "gene_id": gene_id,
                "sample": sample,
                "part": part,
                "n_bins": len(rows_idx),
                "entropy": float(ent),
                "top_state": seg.labels[int(np.argmax(comp))],
                "top_state_fraction": float(comp.max()),
            }
            row.update({lbl: float(v) for lbl, v in zip(seg.labels, comp)})
            if include_marks:
                row.update({f"signal_{m}": float(v)
                            for m, v in zip(ds.marks, signal[rows_idx].mean(axis=0))})
            rows.append(row)
        per_sample.append(pd.DataFrame(rows))

    out = pd.concat(per_sample, ignore_index=True)
    names = features.table.set_index("gene_id")["name"]
    out.insert(1, "name", out["gene_id"].map(names))
    out.attrs["assign"] = assign
    out.attrs["part"] = part
    return out


def gene_state_change(
    gene_states: pd.DataFrame,
    group_a: Sequence[str],
    group_b: Sequence[str],
    labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Per-gene change in state composition between two groups of samples.

    ``tv_distance`` is the total-variation distance between the group-mean compositions:
    the fraction of the gene's state identity that moved. ``delta_<state>`` columns say
    where it went, and are what gets correlated against expression change.
    """
    group_a, group_b = list(group_a), list(group_b)
    known = set(gene_states["sample"].unique())
    missing = [s for s in group_a + group_b if s not in known]
    if missing:
        raise ValueError(f"Sample(s) not in the gene table: {missing}")

    if labels is None:
        skip = {"gene_id", "name", "sample", "part", "n_bins", "entropy",
                "top_state", "top_state_fraction"}
        labels = [c for c in gene_states.columns
                  if c not in skip and not c.startswith("signal_")]

    mean_a = gene_states.loc[gene_states["sample"].isin(group_a)].groupby("gene_id")[list(labels)].mean()
    mean_b = gene_states.loc[gene_states["sample"].isin(group_b)].groupby("gene_id")[list(labels)].mean()
    shared = mean_a.index.intersection(mean_b.index)
    mean_a, mean_b = mean_a.loc[shared], mean_b.loc[shared]

    delta = mean_b - mean_a
    out = pd.DataFrame(index=shared)
    out["tv_distance"] = 0.5 * delta.abs().sum(axis=1)
    for lbl in labels:
        out[f"delta_{lbl}"] = delta[lbl]
    out["top_gain"] = delta.idxmax(axis=1)
    out["top_loss"] = delta.idxmin(axis=1)

    meta = (gene_states.drop_duplicates("gene_id")
                       .set_index("gene_id")[["name", "part", "n_bins"]])
    out = meta.join(out, how="right").reset_index().rename(columns={"index": "gene_id"})
    return out.sort_values("tv_distance", ascending=False).reset_index(drop=True)


def trace_gene(
    seg: Segmentation,
    features: FeatureSet,
    gene_id: str,
    part: str = "promoter",
    against: Optional[int] = None,
    **region_kwargs,
) -> Dict[str, object]:
    """The full evidence chain for one gene, from its number back to the raw counts.

    Returns a dict with:

    ``bins``
        Every bin of the feature, per sample, with the raw count and normalised signal
        for each mark, the state posterior, the hard call, and the confidence.
    ``attribution``
        Per-bin, per-mark log-likelihood ratio for the called state against a comparison
        state -- the runner-up by default, or ``against=<state index>`` to ask "why is
        this not background". *Which mark is the evidence*, in natural log units.
    ``composition``
        The gene-level state composition per sample -- the number that appears in the
        gene table, so it can be checked against the bins that produced it.

    This is the answer to "which specific region is important, and why". Nothing in it is
    a summary of a summary: every column can be traced to a count in the input matrix.
    """
    from .quantify import attribution as _attribution

    ds = seg.dataset
    regions = features.regions(part, **region_kwargs)
    regions = regions.loc[regions["gene_id"] == gene_id]
    if regions.empty:
        raise ValueError(f"{gene_id!r} has no {part} region; is it in the FeatureSet?")

    attr = _attribution(seg, against=against)
    frames, comps = [], []
    for sample in ds.samples:
        mask = ds.sample_mask(sample)
        bins = ds.bins.loc[mask, ["chrom", "start", "end"]].reset_index(drop=True)
        pairs = map_bins_to_regions(bins, regions, assign="all")
        if pairs.empty:
            continue
        rows_local = pairs["bin_row"].to_numpy()
        rows_global = np.flatnonzero(mask)[rows_local]

        frame = bins.iloc[rows_local].reset_index(drop=True)
        frame.insert(0, "sample", sample)
        frame["gene_id"] = gene_id
        for j, mark in enumerate(ds.marks):
            frame[f"count_{mark}"] = ds.X[rows_global, j]
            frame[f"signal_{mark}"] = ds.X[rows_global, j] / np.maximum(
                ds.size_factors[rows_global, j], 1e-9)
        post = seg.posterior[rows_global]
        for k, lbl in enumerate(seg.labels):
            frame[f"post_{lbl}"] = post[:, k]
        frame["state"] = [seg.labels[s] for s in seg.states[rows_global]]
        frame["confidence"] = post.max(axis=1)
        frames.append(frame)

        comp = post.mean(axis=0)
        comps.append({"sample": sample, "n_bins": len(rows_global),
                      **{lbl: float(v) for lbl, v in zip(seg.labels, comp)}})

    if not frames:
        raise ValueError(f"No bins overlap the {part} of {gene_id!r}")

    bins_out = pd.concat(frames, ignore_index=True)
    all_rows = np.concatenate([
        np.flatnonzero(ds.sample_mask(s))[
            map_bins_to_regions(
                ds.bins.loc[ds.sample_mask(s), ["chrom", "start", "end"]].reset_index(drop=True),
                regions, assign="all")["bin_row"].to_numpy()
        ]
        for s in ds.samples
        if ds.sample_mask(s).any()
    ])
    return {
        "gene_id": gene_id,
        "name": str(features.table.set_index("gene_id").loc[gene_id, "name"]),
        "part": part,
        "regions": regions.reset_index(drop=True),
        "bins": bins_out,
        "attribution": attr.iloc[all_rows].reset_index(drop=True),
        "composition": pd.DataFrame(comps),
    }
