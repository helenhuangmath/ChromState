"""Exporting ChromState's bins into other tools' input formats, and reading them back.

The invariant this module exists to protect: *the bin grid never changes*. Every tool is
handed the same intervals, and whatever it returns is mapped back onto those intervals by
genomic coordinate, not by row order -- tools reorder chromosomes, merge adjacent bins
into segments, and drop regions they consider empty, and any of those silently shifts a
row-order alignment by one bin and corrupts every metric downstream.

Reading back is therefore always coordinate-based, and bins no tool output covers stay
:data:`~toolbench.core.UNASSIGNED`.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from chromstate.benchmark.baselines import export_for_chromhmm

from .core import UNASSIGNED

__all__ = [
    "export_chromhmm_binary",
    "export_count_matrix",
    "export_regions_bed",
    "labels_from_segmentation_bed",
    "labels_from_label_file",
    "posterior_from_matrix_file",
]


def export_chromhmm_binary(ds, outdir: str, p_threshold: float = 1e-4) -> List[str]:
    """Write ChromHMM's binarised ``LearnModel`` input for the dataset's bins."""
    return export_for_chromhmm(ds, outdir, p_threshold=p_threshold)


def export_count_matrix(ds, path: str) -> str:
    """Write a plain count matrix: ``chrom start end sample`` then one column per mark.

    This is the generic hand-off for tools that take a matrix (EpiCSeg, epigenome-ssm,
    anything custom). Counts are raw, not normalised: a tool with its own normalisation
    should be allowed to do its own, and one without gets what ChromState's model gets.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    frame = ds.bins[["chrom", "start", "end", "sample"]].copy()
    for j, mark in enumerate(ds.marks):
        frame[mark] = ds.X[:, j]
    frame.to_csv(path, sep="\t", index=False)
    return path


def export_regions_bed(ds, path: str, sample: Optional[str] = None) -> str:
    """Write the bins of one sample as BED3, for tools that take a regions file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    sample = sample or ds.samples[0]
    sub = ds.bins.loc[ds.sample_mask(sample), ["chrom", "start", "end"]]
    sub.to_csv(path, sep="\t", index=False, header=False)
    return path


# ---------------------------------------------------------------------------
# Reading tool output back onto the bin grid
# ---------------------------------------------------------------------------


def _bin_index(ds) -> Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]:
    """(sample, chrom) -> (sorted bin starts, row indices), for coordinate lookup."""
    index: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    # Reset first: the index after a reset is the row position, which is what states[] is
    # keyed by. A dataset carrying a filtered or non-range index would otherwise map every
    # segment onto the wrong rows.
    bins = ds.bins.reset_index(drop=True)
    for (sample, chrom), grp in bins.groupby(["sample", "chrom"], sort=False):
        starts = grp["start"].to_numpy()
        rows = grp.index.to_numpy()
        order = np.argsort(starts, kind="stable")
        index[(str(sample), str(chrom))] = (starts[order], rows[order])
    return index


def labels_from_segmentation_bed(
    ds,
    path: str,
    sample: Optional[str] = None,
    vocabulary: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Map a BED4 segmentation (``chrom start end label``) onto the bin grid.

    Every bin whose start falls inside a segment takes that segment's label. Labels are
    factorised in first-appearance order unless ``vocabulary`` fixes the order; the
    returned vocabulary is what makes a tool's numeric states interpretable afterwards.

    ``sample`` names the dataset sample this file describes. Tools are run per sample --
    ChromHMM writes one ``_segments.bed`` per cell type -- so the caller loops.
    """
    sample = sample or ds.samples[0]
    index = _bin_index(ds)
    states = np.full(ds.n_rows, UNASSIGNED, dtype=np.int64)
    vocab: List[str] = list(vocabulary or [])
    lookup = {name: i for i, name in enumerate(vocab)}

    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            chrom, start, end, label = parts[0], int(parts[1]), int(parts[2]), parts[3]
            key = (sample, chrom)
            if key not in index:
                continue
            if label not in lookup:
                lookup[label] = len(vocab)
                vocab.append(label)
            starts, rows = index[key]
            lo = int(np.searchsorted(starts, start, side="left"))
            hi = int(np.searchsorted(starts, end, side="left"))
            if hi > lo:
                states[rows[lo:hi]] = lookup[label]
    return states, vocab


def labels_from_label_file(
    ds, path: str, sample: Optional[str] = None, vocabulary: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """Read one label per line, in the row order :func:`export_count_matrix` wrote.

    Row-order alignment is the fragile option and is offered only for tools that emit
    nothing else. It is validated by length: a file that does not have exactly one label
    per exported row is an error rather than a silent truncation.
    """
    sample = sample or None
    mask = np.ones(ds.n_rows, dtype=bool) if sample is None else ds.sample_mask(sample)
    with open(path) as fh:
        labels = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    if len(labels) != int(mask.sum()):
        raise ValueError(
            f"{path} has {len(labels)} labels for {int(mask.sum())} bins; "
            "row-order alignment requires exactly one label per exported bin"
        )
    vocab: List[str] = list(vocabulary or [])
    lookup = {name: i for i, name in enumerate(vocab)}
    states = np.full(ds.n_rows, UNASSIGNED, dtype=np.int64)
    out = np.empty(len(labels), dtype=np.int64)
    for i, label in enumerate(labels):
        if label not in lookup:
            lookup[label] = len(vocab)
            vocab.append(label)
        out[i] = lookup[label]
    states[mask] = out
    return states, vocab


def posterior_from_matrix_file(
    ds, path: str, sample: Optional[str] = None, skiprows: int = 0
) -> np.ndarray:
    """Read a per-bin posterior matrix in exported row order into a full-size array.

    Rows that belong to other samples are filled with a uniform distribution, so the
    array is a valid probability matrix everywhere; callers score only the rows they
    supplied. ChromHMM's ``POSTERIOR`` files carry two header lines -- pass ``skiprows=2``.
    """
    P = np.loadtxt(path, skiprows=skiprows, ndmin=2)
    mask = np.ones(ds.n_rows, dtype=bool) if sample is None else ds.sample_mask(sample)
    if P.shape[0] != int(mask.sum()):
        raise ValueError(
            f"{path} has {P.shape[0]} rows for {int(mask.sum())} bins"
        )
    full = np.full((ds.n_rows, P.shape[1]), 1.0 / P.shape[1])
    row_sums = P.sum(axis=1, keepdims=True)
    full[mask] = P / np.maximum(row_sums, 1e-12)
    return full


def frame_from_runs(ds, runs) -> pd.DataFrame:
    """Per-bin table of every tool's call, for writing out or eyeballing in IGV."""
    frame = ds.bins[["chrom", "start", "end", "sample"]].copy()
    for run in runs:
        if run.states is not None:
            frame[run.tool] = run.states
    return frame
