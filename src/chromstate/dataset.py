"""The ChromState data container.

A :class:`ChromStateDataset` stacks every (sample, bin) row into one matrix so that a
*single* model is fit jointly across all samples. This is deliberate: if each sample were
segmented separately, state 4 in the treated sample would not mean the same thing as
state 4 in the control, and no quantitative comparison between them would be valid.
Occupancy is allowed to differ between samples; the definition of a state is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .genome import BinSet, Genome


@dataclass
class ChromStateDataset:
    """Binned multi-mark signal for one or more samples.

    Attributes
    ----------
    X
        ``(n_rows, n_marks)`` signal matrix. Rows are (sample, bin) pairs stacked in
        sample order; columns are marks in ``marks`` order.
    marks
        Mark names, one per column of ``X``.
    bins
        ``(n_rows,)`` row metadata: ``chrom``, ``start``, ``end``, ``sample``.
    chains
        Contiguous row ranges the HMM may treat as single Markov chains. One per
        (sample, chromosome); transitions never span a boundary.
    size_factors
        ``(n_rows, n_marks)`` multiplicative offsets applied to the emission mean. One
        offset per (sample, mark) because every mark is its own sequencing library with
        its own depth. These carry depth and spike-in normalisation, so the *counts*
        stay integers and the likelihood stays a proper count likelihood -- ChromState
        never divides the data through by a scale factor.
    control
        Optional ``(n_rows, n_marks)`` expected background from input/IgG.
    is_count
        Whether ``X`` holds genuine counts. Peak-score input sets this False, which
        blocks the negative-binomial emission model.
    """

    X: np.ndarray
    marks: List[str]
    bins: pd.DataFrame
    chains: List[slice]
    size_factors: np.ndarray
    control: Optional[np.ndarray] = None
    is_count: bool = True
    binset: Optional[BinSet] = None
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_rows, n_marks = self.X.shape
        if len(self.marks) != n_marks:
            raise ValueError(f"{len(self.marks)} mark names for {n_marks} columns of X")
        if len(self.bins) != n_rows:
            raise ValueError(f"bins has {len(self.bins)} rows but X has {n_rows}")
        self.size_factors = np.asarray(self.size_factors, dtype=np.float64)
        if self.size_factors.ndim == 1:
            self.size_factors = np.repeat(self.size_factors[:, None], n_marks, axis=1)
        if self.size_factors.shape != (n_rows, n_marks):
            raise ValueError(f"size_factors must broadcast to ({n_rows}, {n_marks})")
        if self.control is not None and self.control.shape != self.X.shape:
            raise ValueError("control must match the shape of X")
        covered = sum(sl.stop - sl.start for sl in self.chains)
        if covered != n_rows:
            raise ValueError(f"chains cover {covered} rows but X has {n_rows}")

    @property
    def n_rows(self) -> int:
        return self.X.shape[0]

    @property
    def n_marks(self) -> int:
        return self.X.shape[1]

    @property
    def samples(self) -> List[str]:
        return list(pd.unique(self.bins["sample"]))

    def sample_mask(self, sample: str) -> np.ndarray:
        return (self.bins["sample"] == sample).to_numpy()

    def subset_rows(self, mask: np.ndarray) -> "ChromStateDataset":
        """Row-subset the dataset, rebuilding chain boundaries.

        Chains are recomputed from the surviving (sample, chrom) runs, so removing
        blacklisted bins correctly *splits* a chromosome into two chains rather than
        silently joining across the gap.
        """
        mask = np.asarray(mask, dtype=bool)
        bins = self.bins.loc[mask].reset_index(drop=True)
        chains = _chains_from_runs(bins, self.binset.bin_size if self.binset else None)
        return ChromStateDataset(
            X=self.X[mask],
            marks=list(self.marks),
            bins=bins,
            chains=chains,
            size_factors=self.size_factors[mask],
            control=None if self.control is None else self.control[mask],
            is_count=self.is_count,
            binset=self.binset,
            meta=dict(self.meta),
        )

    def to_anndata(self):  # pragma: no cover - optional dependency
        """Export as AnnData for interop with the scanpy/muon ecosystem."""
        import anndata as ad

        return ad.AnnData(
            X=self.X,
            obs=self.bins.copy(),
            var=pd.DataFrame(index=pd.Index(self.marks, name="mark")),
        )

    def summary(self) -> pd.DataFrame:
        """Per-sample, per-mark totals -- a quick sanity check on the input."""
        rows = []
        for sample in self.samples:
            m = self.sample_mask(sample)
            for j, mark in enumerate(self.marks):
                col = self.X[m, j]
                rows.append(
                    {
                        "sample": sample,
                        "mark": mark,
                        "n_bins": int(m.sum()),
                        "total": float(col.sum()),
                        "mean": float(col.mean()),
                        "frac_nonzero": float((col > 0).mean()),
                        "max": float(col.max()),
                    }
                )
        return pd.DataFrame(rows)


def _chains_from_runs(bins: pd.DataFrame, bin_size: Optional[int]) -> List[slice]:
    """Split rows into maximal runs of the same (sample, chrom) that are also contiguous."""
    n = len(bins)
    if n == 0:
        return []
    sample = bins["sample"].to_numpy()
    chrom = bins["chrom"].to_numpy()
    start = bins["start"].to_numpy()

    breaks = (sample[1:] != sample[:-1]) | (chrom[1:] != chrom[:-1])
    if bin_size is not None:
        breaks = breaks | (start[1:] != start[:-1] + bin_size)

    edges = [0, *(np.flatnonzero(breaks) + 1).tolist(), n]
    return [slice(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def build_dataset(
    entries: Sequence,
    binset: BinSet,
    blacklist: Optional[pd.DataFrame] = None,
    reader_kwargs: Optional[Dict[str, dict]] = None,
) -> ChromStateDataset:
    """Load every file in a manifest into a joint dataset.

    Parameters
    ----------
    entries
        ``MarkFile`` records from :func:`chromstate.io.read_manifest`.
    binset
        The genome tiling to project signal onto.
    blacklist
        Optional ``chrom``/``start``/``end`` regions to drop (ENCODE blacklist).
    reader_kwargs
        Per-format keyword arguments forwarded to the readers, e.g.
        ``{"bam": {"min_mapq": 20}}``.
    """
    from .io.counts import is_count_format, load_signal
    from .io.manifest import validate_manifest

    entries = list(entries)
    validate_manifest(entries)
    reader_kwargs = reader_kwargs or {}

    samples = list(dict.fromkeys(e.sample for e in entries))
    marks = list(dict.fromkeys(e.mark for e in entries))
    by_key = {(e.sample, e.mark): e for e in entries}

    n_bins = len(binset)
    blocks = []
    control_blocks = []
    any_control = False
    all_counts = True

    for sample in samples:
        block = np.zeros((n_bins, len(marks)), dtype=np.float64)
        ctrl = np.zeros((n_bins, len(marks)), dtype=np.float64)
        for j, mark in enumerate(marks):
            entry = by_key[(sample, mark)]
            fmt = entry.resolved_format()
            all_counts &= is_count_format(fmt)
            kw = reader_kwargs.get(fmt, {})
            block[:, j] = load_signal(entry.path, binset, fmt=fmt, **kw)
            if entry.control:
                from .io.counts import infer_format

                cfmt = infer_format(entry.control)
                ctrl[:, j] = load_signal(entry.control, binset, fmt=cfmt, **reader_kwargs.get(cfmt, {}))
                any_control = True
        blocks.append(block)
        control_blocks.append(ctrl)

    X = np.vstack(blocks)
    control = np.vstack(control_blocks) if any_control else None

    bed = binset.to_bed()
    bins = pd.concat(
        [bed.assign(sample=s) for s in samples], ignore_index=True
    )[["chrom", "start", "end", "sample"]]

    chains = []
    for i in range(len(samples)):
        for sl in binset.chain_slices:
            chains.append(slice(sl.start + i * n_bins, sl.stop + i * n_bins))

    ds = ChromStateDataset(
        X=X,
        marks=marks,
        bins=bins,
        chains=chains,
        size_factors=np.ones((X.shape[0], len(marks))),
        control=control,
        is_count=all_counts,
        binset=binset,
        meta={"samples": samples},
    )

    if blacklist is not None and len(blacklist):
        keep_one = binset.apply_blacklist(blacklist)
        ds = ds.subset_rows(np.tile(keep_one, len(samples)))
    return ds


def dataset_from_arrays(
    X: np.ndarray,
    marks: Sequence[str],
    sample_labels: Sequence[str],
    chrom_labels: Optional[Sequence[str]] = None,
    bin_size: int = 200,
    is_count: bool = True,
) -> ChromStateDataset:
    """Build a dataset directly from in-memory arrays (used by the simulator and tests)."""
    X = np.asarray(X, dtype=np.float64)
    n_rows = X.shape[0]
    sample_labels = list(sample_labels)
    if chrom_labels is None:
        chrom_labels = ["chrSim"] * n_rows
    chrom_labels = list(chrom_labels)

    starts = np.zeros(n_rows, dtype=np.int64)
    pos = 0
    for i in range(n_rows):
        new_chain = i > 0 and (
            sample_labels[i] != sample_labels[i - 1] or chrom_labels[i] != chrom_labels[i - 1]
        )
        pos = 0 if (i == 0 or new_chain) else pos + bin_size
        starts[i] = pos

    bins = pd.DataFrame(
        {
            "chrom": chrom_labels,
            "start": starts,
            "end": starts + bin_size,
            "sample": sample_labels,
        }
    )
    chains = _chains_from_runs(bins, bin_size)
    return ChromStateDataset(
        X=X,
        marks=list(marks),
        bins=bins,
        chains=chains,
        size_factors=np.ones((n_rows, X.shape[1])),
        is_count=is_count,
        binset=BinSet(table=bins[["chrom", "start", "end"]], bin_size=bin_size, chain_slices=chains),
    )


def make_genome_binset(
    sizes: Dict[str, int], bin_size: int = 200, chroms: Optional[Sequence[str]] = None
) -> BinSet:
    """Convenience wrapper: chromosome-size dict to :class:`BinSet`."""
    return BinSet.from_genome(Genome(sizes), bin_size=bin_size, chroms=chroms)
