"""Step 1: bin the pooled CUT&RUN coverage tracks into 200 bp bins, one file per chromosome.

Input is the replicate-pooled, DESeq2-scaled bamCoverage tracks (no BAMs are available
locally). The stored value is mean coverage per bin; step 2 decides how to model it.
"""

from __future__ import annotations

import time

import numpy as np
import pyBigWig
from common import BIN_SIZE, CHROMS, CONDITIONS, MARKS, ensure_out, pooled_track

from chromstate.genome import BinSet, Genome
from chromstate.io.counts import bin_bigwig


def main() -> None:
    out = ensure_out("binned")
    bw = pyBigWig.open(str(pooled_track(MARKS[0], CONDITIONS[0])))
    sizes = {c: int(s) for c, s in bw.chroms().items() if c in CHROMS}
    bw.close()
    binset = BinSet.from_genome(Genome(sizes), bin_size=BIN_SIZE, chroms=CHROMS)
    offsets = binset.chrom_offsets()

    cube = np.zeros((len(CONDITIONS), len(binset), len(MARKS)), dtype=np.float32)
    for ci, cond in enumerate(CONDITIONS):
        for mi, mark in enumerate(MARKS):
            t0 = time.time()
            cube[ci, :, mi] = bin_bigwig(str(pooled_track(mark, cond)), binset)
            print(f"{cond:6s} {mark:9s} {time.time() - t0:6.1f}s", flush=True)

    for chrom in CHROMS:
        start, n = offsets[chrom]
        np.savez_compressed(
            out / f"{chrom}.npz",
            X=cube[:, start:start + n, :],
            conditions=np.array(CONDITIONS),
            marks=np.array(MARKS),
            bin_size=BIN_SIZE,
        )
    print(f"wrote {len(CHROMS)} chromosomes, {len(binset):,} bins each condition -> {out}")


if __name__ == "__main__":
    main()
