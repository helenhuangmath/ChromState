"""Export the ENCODE benchmark bins (bin_bams.py merge output) for the tool harness.

    python tools/realdata/export_encode_counts.py --binned <binned> --rna-dir <raw> --chroms chr19 -o <out>

Writes ``counts.tsv`` (chrom, start, end, sample, <marks>) with the two biological
replicates of each cell line summed into one sample, as the ChromHMM, diHMM and EpiCSeg
papers pooled them, and ``expression.tsv`` (gene_id x cell line, mean log2(TPM + 1) over
the two RNA-seq replicates). ``--per-replicate`` keeps every replicate as its own sample.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

RNA = {"GM12878": ["ENCFF910XWA", "ENCFF413MYB"], "K562": ["ENCFF136UBW", "ENCFF171FQU"],
       "H1": ["ENCFF321HCT", "ENCFF736CCO"]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--binned", required=True)
    p.add_argument("--rna-dir", required=True)
    p.add_argument("--chroms", default="chr19")
    p.add_argument("--per-replicate", action="store_true")
    p.add_argument("-o", "--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    frames = []
    for chrom in a.chroms.split(","):
        z = np.load(os.path.join(a.binned, f"{chrom}.npz"))
        X, samples, marks, bs = z["X"], [str(s) for s in z["samples"]], [str(m) for m in z["marks"]], int(z["bin_size"])
        n = X.shape[1]
        groups = {s: [i] for i, s in enumerate(samples)} if a.per_replicate else {}
        if not a.per_replicate:
            for i, s in enumerate(samples):
                groups.setdefault(s.rsplit("_R", 1)[0], []).append(i)
        for name, idx in groups.items():
            f = pd.DataFrame({"chrom": chrom, "start": np.arange(n) * bs, "end": (np.arange(n) + 1) * bs, "sample": name})
            Y = X[idx].sum(axis=0)
            for mi, m in enumerate(marks):
                f[m] = np.rint(Y[:, mi]).astype(np.int64)
            frames.append(f)
    tab = pd.concat(frames, ignore_index=True)
    tab.to_csv(os.path.join(a.out, "counts.tsv"), sep="\t", index=False)
    print(f"counts.tsv: {len(tab):,} bin-rows, samples {list(groups)}, marks {marks}")

    cols = {}
    for cell, accs in RNA.items():
        reps = []
        for acc in accs:
            t = pd.read_csv(os.path.join(a.rna_dir, f"{acc}.tsv"), sep="\t")
            t = t.loc[t["gene_id"].astype(str).str.startswith("ENSG")]
            t["gene_id"] = t["gene_id"].str.split(".").str[0]
            reps.append(np.log2(t.drop_duplicates("gene_id").set_index("gene_id")["TPM"] + 1.0))
        cols[cell] = pd.concat(reps, axis=1).mean(axis=1)
    expr = pd.DataFrame(cols)
    if a.per_replicate:
        expr = pd.DataFrame({s: expr[s.rsplit("_R", 1)[0]] for s in groups})
    expr.index.name = "gene_id"
    expr.to_csv(os.path.join(a.out, "expression.tsv"), sep="\t")
    print(f"expression.tsv: {expr.shape}")


if __name__ == "__main__":
    main()
