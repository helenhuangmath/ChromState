"""Export the Sci Rep 2025 CD8 T-cell bins (analysis/scireports2025/out/binned) as the
count matrix the tool harness reads: chrom, start, end, sample, <marks>.

    python tools/realdata/export_cd8_counts.py --chroms chr19 -o tmp/realbench/cd8_chr19

Values are the replicate-pooled, DESeq2-scaled coverage per 200 bp bin rounded to
integers (the same approximation to counts the Sci Rep analysis used).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BINNED = os.path.join(ROOT, "analysis", "scireports2025", "out", "binned")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--chroms", default="chr19")
    p.add_argument("--conditions", default="Naive,Arm,Cl13")
    p.add_argument("-o", "--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    conds = a.conditions.split(",")
    frames = []
    for chrom in a.chroms.split(","):
        z = np.load(os.path.join(BINNED, f"{chrom}.npz"))
        X, zc, marks, bs = z["X"], list(z["conditions"]), list(z["marks"]), int(z["bin_size"])
        n = X.shape[1]
        for cond in conds:
            ci = zc.index(cond)
            f = pd.DataFrame({"chrom": chrom, "start": np.arange(n) * bs, "end": (np.arange(n) + 1) * bs,
                              "sample": cond})
            for mi, m in enumerate(marks):
                f[m] = np.rint(X[ci, :, mi]).astype(np.int64)
            frames.append(f)
    tab = pd.concat(frames, ignore_index=True)
    path = os.path.join(a.out, "counts.tsv")
    tab.to_csv(path, sep="\t", index=False)
    print(f"wrote {path}: {len(tab):,} bin-rows, marks {marks}, samples {conds}, bin {bs} bp")
    print(tab.groupby("sample")[marks].mean().round(2))


if __name__ == "__main__":
    main()
