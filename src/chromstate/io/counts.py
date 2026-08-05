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
    raise ValueError(f"Unknown input format {fmt!r}; expected one of bam, bed, peak")


def infer_format(path: str) -> str:
    """Guess the input format from the file name."""
    base = os.path.basename(path).lower()
    if base.endswith(".gz"):
        base = base[:-3]
    if base.endswith(".bam"):
        return "bam"
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
