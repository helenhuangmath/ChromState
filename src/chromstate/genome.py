"""Genome binning: chromosome sizes, fixed-width bins, and blacklist handling.

Bins default to 200 bp to match ChromHMM, so that ChromState output can be compared
bin-for-bin against an existing ChromHMM segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_BIN_SIZE = 200


@dataclass
class Genome:
    """Chromosome sizes for a genome assembly."""

    sizes: Dict[str, int]
    name: str = "custom"

    @classmethod
    def from_file(cls, path: str, name: str = "custom") -> "Genome":
        """Read a UCSC-style two-column ``chrom<TAB>size`` file."""
        sizes: Dict[str, int] = {}
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                sizes[fields[0]] = int(fields[1])
        return cls(sizes=sizes, name=name)

    @classmethod
    def from_bam(cls, path: str, name: str = "from_bam") -> "Genome":
        """Read chromosome sizes from a BAM header."""
        import pysam

        with pysam.AlignmentFile(path, "rb") as bam:
            sizes = dict(zip(bam.references, bam.lengths))
        return cls(sizes=sizes, name=name)

    @property
    def chroms(self) -> List[str]:
        return list(self.sizes.keys())

    def subset(self, chroms: Iterable[str]) -> "Genome":
        keep = [c for c in chroms if c in self.sizes]
        return Genome({c: self.sizes[c] for c in keep}, name=self.name)


@dataclass
class BinSet:
    """A fixed-width tiling of a genome.

    Attributes
    ----------
    table
        ``chrom``/``start``/``end`` DataFrame, one row per bin, sorted by chromosome
        in genome order then by start.
    bin_size
        Width of each bin in bp (the final bin of a chromosome may be shorter).
    chain_slices
        One ``slice`` per chromosome. Each slice is a contiguous run of rows that the
        HMM may treat as a single Markov chain; the chain is *reset* between slices so
        that transitions never span a chromosome boundary.
    """

    table: pd.DataFrame
    bin_size: int
    chain_slices: List[slice]

    def __len__(self) -> int:
        return len(self.table)

    @property
    def n_bins(self) -> int:
        return len(self.table)

    @classmethod
    def from_genome(
        cls,
        genome: Genome,
        bin_size: int = DEFAULT_BIN_SIZE,
        chroms: Optional[Iterable[str]] = None,
    ) -> "BinSet":
        chrom_list = list(chroms) if chroms is not None else genome.chroms
        frames = []
        slices: List[slice] = []
        offset = 0
        for chrom in chrom_list:
            size = genome.sizes[chrom]
            starts = np.arange(0, size, bin_size, dtype=np.int64)
            ends = np.minimum(starts + bin_size, size)
            frames.append(pd.DataFrame({"chrom": chrom, "start": starts, "end": ends}))
            slices.append(slice(offset, offset + len(starts)))
            offset += len(starts)
        table = pd.concat(frames, ignore_index=True)
        return cls(table=table, bin_size=bin_size, chain_slices=slices)

    def index_of(self, chrom: str, pos: int) -> int:
        """Global row index of the bin containing ``pos`` on ``chrom`` (-1 if absent)."""
        offsets = self.chrom_offsets()
        if chrom not in offsets:
            return -1
        start_row, n_rows = offsets[chrom]
        local = pos // self.bin_size
        if local < 0 or local >= n_rows:
            return -1
        return start_row + local

    def chrom_offsets(self) -> Dict[str, Tuple[int, int]]:
        """Map ``chrom -> (first_row_index, n_rows)``."""
        if not hasattr(self, "_offsets_cache"):
            offsets: Dict[str, Tuple[int, int]] = {}
            chrom_col = self.table["chrom"].to_numpy()
            for sl in self.chain_slices:
                chrom = chrom_col[sl.start]
                offsets[chrom] = (sl.start, sl.stop - sl.start)
            object.__setattr__(self, "_offsets_cache", offsets)
        return self._offsets_cache  # type: ignore[attr-defined]

    def apply_blacklist(self, regions: pd.DataFrame) -> np.ndarray:
        """Return a boolean keep-mask excluding bins overlapping blacklist ``regions``."""
        mask = np.ones(len(self), dtype=bool)
        offsets = self.chrom_offsets()
        for row in regions.itertuples(index=False):
            chrom = getattr(row, "chrom")
            if chrom not in offsets:
                continue
            start_row, n_rows = offsets[chrom]
            lo = max(0, int(getattr(row, "start")) // self.bin_size)
            hi = min(n_rows, int(getattr(row, "end")) // self.bin_size + 1)
            if hi > lo:
                mask[start_row + lo : start_row + hi] = False
        return mask

    def to_bed(self) -> pd.DataFrame:
        return self.table[["chrom", "start", "end"]].copy()
