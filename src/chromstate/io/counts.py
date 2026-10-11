"""Binned signal extraction from BAM, BED (aligned reads), and peak BED files.

Three input flavours are supported, because in practice you get whichever one the
upstream pipeline produced:

``bam``
    Aligned reads. Each read is reduced to a single position (5' end shifted by
    ``shift`` bp towards the fragment centre, or the fragment midpoint for properly
    paired reads) and tallied into its bin. This is the highest-fidelity input and the
    only one that gives genuine counts for the negative-binomial emission model.

``bed``
    Aligned reads in BED form (the format ChromHMM's ``BinarizeBed`` consumes). Treated
    identically to BAM.

``peak``
    Called peaks (BED3, BED6, narrowPeak, broadPeak). This is *not* count data — it is
    an already-thresholded summary. ChromState will accept it, but the per-bin value is
    a peak score rather than a count, so the model must use ``gaussian`` or
    ``bernoulli`` emissions. Using ``nb`` emissions on peak input is refused rather than
    silently producing a meaningless dispersion estimate.

``bigwig``
    A coverage track (bamCoverage, bedGraphToBigWig). The per-bin value is the
    base-weighted mean coverage over the bin. A coverage track is scaled and smoothed by
    whatever produced it, so it is not count data either, even when its values look like
    integers. When it is the only input available, rounding a depth-scaled track gives a
    usable count-like approximation for the negative binomial model, but that is a
    choice the caller makes and records -- :func:`bin_bigwig` never makes it silently.
"""

from __future__ import annotations

import gzip
import os
from typing import Optional

import numpy as np
import pandas as pd

from ..genome import BinSet

READ_FORMATS = ("bam", "bed")
PEAK_FORMATS = ("peak", "narrowpeak", "broadpeak")
COVERAGE_FORMATS = ("bigwig", "bw")


def _open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def count_bam(
    path: str,
    binset: BinSet,
    shift: int = 100,
    min_mapq: int = 30,
    use_fragment_midpoint: bool = True,
    exclude_duplicates: bool = True,
) -> np.ndarray:
    """Tally reads from a BAM file into genomic bins.

    Parameters
    ----------
    shift
        Single-end reads are shifted this many bp in the 3' direction to approximate the
        fragment centre. ChromHMM's default is 100. Ignored for properly paired reads
        when ``use_fragment_midpoint`` is set.
    min_mapq
        Reads below this mapping quality are discarded.
    """
    try:
        import pysam
    except ImportError as exc:  # pragma: no cover - exercised only without pysam
        raise ImportError(
            "Reading BAM requires pysam. Install with: pip install 'chromstate[bam]'"
        ) from exc

    counts = np.zeros(len(binset), dtype=np.int32)
    offsets = binset.chrom_offsets()
    bin_size = binset.bin_size

    with pysam.AlignmentFile(path, "rb") as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped or read.mapping_quality < min_mapq:
                continue
            if exclude_duplicates and read.is_duplicate:
                continue
            chrom = read.reference_name
            if chrom not in offsets:
                continue

            if use_fragment_midpoint and read.is_proper_pair and read.template_length != 0:
                if read.template_length < 0:
                    continue  # count each pair once, from the leftmost mate
                pos = read.reference_start + read.template_length // 2
            elif read.is_reverse:
                pos = read.reference_end - shift
            else:
                pos = read.reference_start + shift

            start_row, n_rows = offsets[chrom]
            local = pos // bin_size
            if 0 <= local < n_rows:
                counts[start_row + local] += 1
    return counts


def count_bed_reads(
    path: str,
    binset: BinSet,
    shift: int = 100,
) -> np.ndarray:
    """Tally aligned reads stored as BED intervals into genomic bins.

    Column 6 (strand) is used when present; otherwise the interval midpoint is used.
    """
    counts = np.zeros(len(binset), dtype=np.int32)
    offsets = binset.chrom_offsets()
    bin_size = binset.bin_size

    with _open_text(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.rstrip("\n").split("\t")
            chrom = f[0]
            if chrom not in offsets:
                continue
            start, end = int(f[1]), int(f[2])
            strand = f[5] if len(f) > 5 else "."
            if strand == "+":
                pos = start + shift
            elif strand == "-":
                pos = end - shift
            else:
                pos = (start + end) // 2

            start_row, n_rows = offsets[chrom]
            local = pos // bin_size
            if 0 <= local < n_rows:
                counts[start_row + local] += 1
    return counts


def score_peaks(
    path: str,
    binset: BinSet,
    value_column: Optional[int] = None,
    binary: bool = False,
) -> np.ndarray:
    """Project called peaks onto bins.

    Parameters
    ----------
    value_column
        0-based column holding the per-peak signal. Defaults to column 6
        (``signalValue``) for narrowPeak/broadPeak, column 4 (``score``) for BED6, and
        a constant 1.0 for BED3. A bin overlapping several peaks takes the maximum.
    binary
        If set, emit 1/0 occupancy instead of the peak score.

    Returns
    -------
    Per-bin float array. This is a *score*, not a count -- see the module docstring.
    """
    values = np.zeros(len(binset), dtype=np.float32)
    offsets = binset.chrom_offsets()
    bin_size = binset.bin_size

    with _open_text(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.rstrip("\n").split("\t")
            chrom = f[0]
            if chrom not in offsets:
                continue
            start, end = int(f[1]), int(f[2])

            if binary:
                val = 1.0
            elif value_column is not None:
                val = float(f[value_column])
            elif len(f) >= 7:  # narrowPeak / broadPeak signalValue
                val = float(f[6])
            elif len(f) >= 5:  # BED6 score
                val = float(f[4])
            else:  # BED3
                val = 1.0

            start_row, n_rows = offsets[chrom]
            lo = max(0, start // bin_size)
            hi = min(n_rows, (end - 1) // bin_size + 1)
            if hi > lo:
                sl = slice(start_row + lo, start_row + hi)
                np.maximum(values[sl], val, out=values[sl])
    return values


def _bin_means_from_intervals(
    starts: np.ndarray,
    ends: np.ndarray,
    values: np.ndarray,
    n_bins: int,
    bin_size: int,
    chrom_size: int,
) -> np.ndarray:
    """Base-weighted mean of piecewise-constant coverage over consecutive bins.

    Uses the running integral of coverage, evaluated at every bin edge, so each bin's
    mean is exact however the track's intervals straddle the bin boundaries -- and no
    per-base array the size of the chromosome is ever allocated. Uncovered bases count
    as zero, which is what an absent interval in a coverage track means.
    """
    edges = np.minimum(np.arange(n_bins + 1, dtype=np.int64) * bin_size, chrom_size)
    if len(starts) == 0:
        return np.zeros(n_bins)
    # integral up to the end of interval i
    cum = np.concatenate([[0.0], np.cumsum(values * (ends - starts))])
    # for each edge x: completed intervals are those with end <= x; add the partial one
    i = np.searchsorted(ends, edges, side="right")
    integral = cum[i]
    partial = i < len(starts)
    j = i[partial]
    overlap = np.clip(edges[partial] - starts[j], 0, None)
    integral[partial] += values[j] * overlap
    widths = np.maximum(np.diff(edges), 1)
    return np.diff(integral) / widths


def bin_bigwig(path: str, binset: BinSet) -> np.ndarray:
    """Mean coverage of a bigWig track in every bin of ``binset``.

    Chromosomes in the bin set that the track does not cover are left at zero and
    recorded nowhere else, so check the track's chromosome names against the bin set
    (``chr1`` versus ``1``) when a whole chromosome comes back empty.
    """
    try:
        import pyBigWig
    except ImportError as exc:  # pragma: no cover - exercised only without pyBigWig
        raise ImportError(
            "Reading bigWig requires pyBigWig. Install with: pip install 'chromstate[bigwig]'"
        ) from exc

    out = np.zeros(len(binset), dtype=np.float64)
    offsets = binset.chrom_offsets()
    bw = pyBigWig.open(path)
    try:
        sizes = bw.chroms()
        for chrom, (start_row, n_rows) in offsets.items():
            if chrom not in sizes:
                continue
            ivs = bw.intervals(chrom) or ()
            arr = np.asarray(ivs, dtype=np.float64).reshape(-1, 3)
            order = np.argsort(arr[:, 0], kind="stable")
            arr = arr[order]
            out[start_row:start_row + n_rows] = _bin_means_from_intervals(
                arr[:, 0].astype(np.int64),
                arr[:, 1].astype(np.int64),
                arr[:, 2],
                n_rows,
                binset.bin_size,
                int(sizes[chrom]),
            )
    finally:
        bw.close()
    return out


def load_signal(
    path: str,
    binset: BinSet,
    fmt: Optional[str] = None,
    **kwargs,
) -> np.ndarray:
    """Dispatch to the right reader based on ``fmt`` (or the file extension)."""
    if fmt is None:
        fmt = infer_format(path)
    fmt = fmt.lower()

    if fmt == "bam":
        return count_bam(path, binset, **kwargs).astype(np.float64)
    if fmt == "bed":
        return count_bed_reads(path, binset, **kwargs).astype(np.float64)
    if fmt in PEAK_FORMATS:
        return score_peaks(path, binset, **kwargs).astype(np.float64)
    if fmt in COVERAGE_FORMATS:
        return bin_bigwig(path, binset)
    raise ValueError(f"Unknown input format {fmt!r}; expected one of bam, bed, peak, bigwig")


def infer_format(path: str) -> str:
    """Guess the input format from the file name."""
    base = os.path.basename(path).lower()
    if base.endswith(".gz"):
        base = base[:-3]
    if base.endswith(".bam"):
        return "bam"
    if base.endswith((".bw", ".bigwig")):
        return "bigwig"
    if base.endswith((".narrowpeak", ".broadpeak", ".gappedpeak")):
        return "peak"
    if "peak" in base:
        return "peak"
    if base.endswith((".bed", ".tagalign")):
        return "bed"
    raise ValueError(
        f"Cannot infer format for {path!r}. Pass fmt= explicitly (bam, bed, or peak)."
    )


def is_count_format(fmt: str) -> bool:
    """True when the reader for ``fmt`` yields genuine integer counts."""
    return fmt.lower() in READ_FORMATS


def read_bed_regions(path: str) -> pd.DataFrame:
    """Read a BED file as a ``chrom``/``start``/``end`` DataFrame (for blacklists etc.)."""
    rows = []
    with _open_text(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.rstrip("\n").split("\t")
            rows.append((f[0], int(f[1]), int(f[2])))
    return pd.DataFrame(rows, columns=["chrom", "start", "end"])
