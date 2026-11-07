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

    @classmethod
    def from_regions(
        cls,
        regions: pd.DataFrame,
        bin_size: int = DEFAULT_BIN_SIZE,
        genome: Optional[Genome] = None,
        chrom_order: Optional[Iterable[str]] = None,
    ) -> "BinSet":
        """Tile only the given ``chrom``/``start``/``end`` windows (gene-level mode).

        Overlapping or touching windows on a chromosome are merged first, so a locus
        shared by two genes is tiled once, and each merged window becomes its own Markov
        chain. Windows are clipped to the chromosome size when ``genome`` is given. The
        tiling is anchored at the window start, so bins are ``bin_size`` wide except
        possibly the last of a window.
        """
        req = [c for c in ("chrom", "start", "end") if c not in regions.columns]
        if req:
            raise ValueError(f"regions needs columns {req}")
        reg = regions[["chrom", "start", "end"]].copy()
        reg["chrom"] = reg["chrom"].astype(str)
        reg["start"] = np.maximum(reg["start"].astype(np.int64), 0)
        reg["end"] = reg["end"].astype(np.int64)
        if genome is not None:
            reg = reg.loc[reg["chrom"].isin(genome.sizes)]
            reg["end"] = np.minimum(reg["end"], reg["chrom"].map(genome.sizes).astype(np.int64))
        reg = reg.loc[reg["end"] > reg["start"]]
        if reg.empty:
            raise ValueError("no usable region to tile")
        if chrom_order is None:
            chrom_order = genome.chroms if genome is not None else list(pd.unique(reg["chrom"]))
        chroms = [c for c in chrom_order if c in set(reg["chrom"])]

        frames = []
        slices: List[slice] = []
        offset = 0
        for chrom in chroms:
            sub = reg.loc[reg["chrom"] == chrom].sort_values("start")
            s_arr, e_arr = sub["start"].to_numpy(), sub["end"].to_numpy()
            merged: List[Tuple[int, int]] = []
            for s, e in zip(s_arr, e_arr):
                if merged and s <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], int(e)))
                else:
                    merged.append((int(s), int(e)))
            for s, e in merged:
                starts = np.arange(s, e, bin_size, dtype=np.int64)
                ends = np.minimum(starts + bin_size, e)
                frames.append(pd.DataFrame({"chrom": chrom, "start": starts, "end": ends}))
                slices.append(slice(offset, offset + len(starts)))
                offset += len(starts)
        table = pd.concat(frames, ignore_index=True)
        return cls(table=table, bin_size=bin_size, chain_slices=slices)

    @property
    def is_genome_tiling(self) -> bool:
        """True when every chromosome is tiled from 0 without gaps (``from_genome``)."""
        if not hasattr(self, "_genome_tiling_cache"):
            starts = self.table["start"].to_numpy()
            ok = True
            for sl in self.chain_slices:
                s = starts[sl]
                if s[0] != 0 or (len(s) > 1 and np.any(np.diff(s) != self.bin_size)):
                    ok = False
                    break
            chroms = self.table["chrom"].to_numpy()
            ok = ok and len({chroms[sl.start] for sl in self.chain_slices}) == len(self.chain_slices)
            object.__setattr__(self, "_genome_tiling_cache", ok)
        return self._genome_tiling_cache  # type: ignore[attr-defined]

    def index_of(self, chrom: str, pos: int) -> int:
        """Global row index of the bin containing ``pos`` on ``chrom`` (-1 if absent)."""
        offsets = self.chrom_offsets()
        if chrom not in offsets:
            return -1
        start_row, n_rows = offsets[chrom]
        if self.is_genome_tiling:
            local = pos // self.bin_size
            if local < 0 or local >= n_rows:
                return -1
            return start_row + local
        starts = self.table["start"].to_numpy()[start_row:start_row + n_rows]
        ends = self.table["end"].to_numpy()[start_row:start_row + n_rows]
        j = int(np.searchsorted(starts, pos, side="right")) - 1
        if j < 0 or pos >= ends[j]:
            return -1
        return start_row + j

    def rows_for_positions(self, chrom: str, pos: np.ndarray) -> np.ndarray:
        """Global row index of the bin containing each position (-1 where none).

        Vectorised counterpart of :meth:`index_of`; the read counters use it so that a
        windowed tiling (gene-level mode) is as cheap to fill as a genome tiling.
        """
        pos = np.asarray(pos, dtype=np.int64)
        offsets = self.chrom_offsets()
        if chrom not in offsets:
            return np.full(len(pos), -1, dtype=np.int64)
        start_row, n_rows = offsets[chrom]
        if self.is_genome_tiling:
            local = pos // self.bin_size
            ok = (pos >= 0) & (local < n_rows)
            return np.where(ok, start_row + local, -1)
        starts = self.table["start"].to_numpy()[start_row:start_row + n_rows]
        ends = self.table["end"].to_numpy()[start_row:start_row + n_rows]
        j = np.searchsorted(starts, pos, side="right") - 1
        ok = (j >= 0) & (pos < ends[np.clip(j, 0, n_rows - 1)])
        return np.where(ok, start_row + j, -1)

    def rows_overlapping(self, chrom: str, start: int, end: int) -> slice:
        """Row range of bins overlapping ``[start, end)`` on ``chrom`` (empty if none)."""
        offsets = self.chrom_offsets()
        if chrom not in offsets:
            return slice(0, 0)
        start_row, n_rows = offsets[chrom]
        if self.is_genome_tiling:
            lo = max(0, int(start) // self.bin_size)
            hi = min(n_rows, (max(int(end), int(start) + 1) - 1) // self.bin_size + 1)
            return slice(start_row + lo, start_row + max(hi, lo))
        starts = self.table["start"].to_numpy()[start_row:start_row + n_rows]
        ends = self.table["end"].to_numpy()[start_row:start_row + n_rows]
        lo = int(np.searchsorted(ends, start, side="right"))
        hi = int(np.searchsorted(starts, end, side="left"))
        return slice(start_row + lo, start_row + max(hi, lo))

    def chrom_offsets(self) -> Dict[str, Tuple[int, int]]:
        """Map ``chrom -> (first_row_index, n_rows)``."""
        if not hasattr(self, "_offsets_cache"):
            offsets: Dict[str, Tuple[int, int]] = {}
            chrom_col = self.table["chrom"].to_numpy()
            for sl in self.chain_slices:
                chrom = chrom_col[sl.start]
                if chrom in offsets:  # several windows on one chromosome (from_regions)
                    first, n = offsets[chrom]
                    offsets[chrom] = (first, sl.stop - first)
                else:
                    offsets[chrom] = (sl.start, sl.stop - sl.start)
            object.__setattr__(self, "_offsets_cache", offsets)
        return self._offsets_cache  # type: ignore[attr-defined]

    def apply_blacklist(self, regions: pd.DataFrame) -> np.ndarray:
        """Return a boolean keep-mask excluding bins overlapping blacklist ``regions``."""
        mask = np.ones(len(self), dtype=bool)
        offsets = self.chrom_offsets()
        for row in regions.itertuples(index=False):
            if row.chrom not in offsets:
                continue
            sl = self.rows_overlapping(row.chrom, int(row.start), int(row.end))
            if sl.stop > sl.start:
                mask[sl] = False
        return mask

    def to_bed(self) -> pd.DataFrame:
        return self.table[["chrom", "start", "end"]].copy()
