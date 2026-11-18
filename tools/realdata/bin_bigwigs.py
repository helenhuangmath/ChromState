"""Bin per-replicate GEO bigWigs (GSE285245) into per-chromosome arrays for ChromState.

    python tools/realdata/bin_bigwigs.py --bw-dir <per_sample_bw> --chroms chr1,...,chr19 -o <out>

Standalone (numpy + pyBigWig only) so it can run in any env that has pyBigWig.
Input files are named ``GSM<id>_<Cond>_P14_Day<dd>_<Mark>_Rep<r>_S<k>.scale.bw``.
Each bin holds the mean depth-scaled coverage over its 200 bp. That is not a read
count: the ChromState driver rounds it to integers, the same convention as the
pooled-track runs, so the two are comparable.

Writes ``<out>/<chrom>.npz`` with X (samples x bins x marks, float32), samples,
marks, bin_size, and ``<out>/samples.tsv``.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import time

from functools import partial
from multiprocessing import Pool

import numpy as np
import pyBigWig


def bin_one(path: str, chrom: str, bin_size: int) -> np.ndarray:
    """Exact mean coverage per bin from the bigWig intervals (uncovered bases count as 0)."""
    bw = pyBigWig.open(path)
    try:
        size = bw.chroms().get(chrom)
        if size is None:
            raise SystemExit(f"{path} has no {chrom}")
        nb = size // bin_size
        ivs = bw.intervals(chrom, 0, nb * bin_size) or ()
    finally:
        bw.close()
    arr = np.asarray(ivs, dtype=np.float64).reshape(-1, 3)
    end = nb * bin_size
    s = np.clip(arr[:, 0].astype(np.int64), 0, end)
    e = np.clip(arr[:, 1].astype(np.int64), 0, end)
    v = arr[:, 2]
    # integral of coverage up to every bin edge, via a cumulative sum over a difference array
    diff = np.zeros(end + 1, dtype=np.float64)
    np.add.at(diff, s, v)
    np.add.at(diff, e, -v)
    cov = np.cumsum(diff[:-1])
    return cov.reshape(nb, bin_size).mean(axis=1).astype(np.float32)

NAME_RE = re.compile(r"GSM\d+_(?P<cond>[A-Za-z0-9]+)_P14_Day(?P<day>\d+)_(?P<mark>[A-Za-z0-9]+)_Rep(?P<rep>\d+)_S\d+\.scale\.bw$")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bw-dir", required=True)
    p.add_argument("--chroms", required=True, help="comma-separated")
    p.add_argument("--marks", default="H3K4me3,H3K27ac,H3K27me3,H3K9me3")
    p.add_argument("--bin-size", type=int, default=200)
    p.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    p.add_argument("-o", "--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    marks = a.marks.split(",")

    files = {}
    for path in sorted(glob.glob(os.path.join(a.bw_dir, "*.scale.bw"))):
        m = NAME_RE.search(os.path.basename(path))
        if not m or m["mark"] not in marks:
            continue
        sample = f"{m['cond']}_d{m['day']}_R{m['rep']}"
        files.setdefault(sample, {})[m["mark"]] = path
    samples = sorted(files, key=lambda s: ({"Naive": 0, "Arm": 1, "Cl13": 2}.get(s.split("_")[0], 9), s))
    for s in samples:
        missing = [mk for mk in marks if mk not in files[s]]
        if missing:
            raise SystemExit(f"sample {s} is missing marks {missing}")
    print(f"{len(samples)} samples x {len(marks)} marks: {samples}", flush=True)
    with open(os.path.join(a.out, "samples.tsv"), "w") as fh:
        fh.write("sample\tcondition\tday\treplicate\t" + "\t".join(marks) + "\n")
        for s in samples:
            cond, day, rep = s.split("_")
            fh.write(f"{s}\t{cond}\t{day[1:]}\t{rep[1:]}\t" + "\t".join(os.path.basename(files[s][mk]) for mk in marks) + "\n")

    jobs = [(files[s][mk], si, mi) for si, s in enumerate(samples) for mi, mk in enumerate(marks)]
    for chrom in a.chroms.split(","):
        t0 = time.time()
        with Pool(a.procs) as pool:
            res = pool.map(partial(bin_one, chrom=chrom, bin_size=a.bin_size), [j[0] for j in jobs])
        X = np.zeros((len(samples), len(res[0]), len(marks)), dtype=np.float32)
        for (path, si, mi), v in zip(jobs, res):
            X[si, :, mi] = v
        np.savez_compressed(os.path.join(a.out, f"{chrom}.npz"), X=X, samples=np.array(samples),
                            marks=np.array(marks), bin_size=a.bin_size)
        tot = X.sum(axis=1)
        print(f"{chrom}: {X.shape[1]:,} bins, {time.time() - t0:.0f}s; per-sample mean bin value by mark:\n"
              + "\n".join(f"  {s:14s} " + " ".join(f"{v / X.shape[1]:7.2f}" for v in tot[i]) for i, s in enumerate(samples)),
              flush=True)

if __name__ == "__main__":
    main()
