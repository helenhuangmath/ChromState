"""Tutorial 3 driver: predict expression from chromatin states on the CD8 T-cell data.

    python tools/realdata/predict_cd8.py --chroms chr19 -k 12 -o tmp/predict_cd8/chr19_k12
    python tools/realdata/predict_cd8.py --chroms chr19 -k 12 --definitions default -o ...

Input: replicate-pooled, DESeq2-scaled CUT&RUN coverage per 200 bp bin rounded to
integers (analysis/scireports2025/out/binned/<chrom>.npz; Naive, Arm = memory, Cl13 =
exhausted; H3K4me3, H3K27ac, H3K27me3, H3K9me3), RNA-seq means per condition
(out/rna/expression.tsv) and DESeq2 Cl13-vs-Naive (out/rna/de_Cl13_vs_Naive.tsv).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
BINNED = os.path.join(ROOT, "analysis", "scireports2025", "out", "binned")
RNA = os.path.join(ROOT, "analysis", "scireports2025", "out", "rna")
GTF = "/vast/projects/wherry/foundation-models-immuno/hhua/tools/chromstate-bench/annotation/gencode.vM25.annotation.gtf.gz"

import chromstate as cs  # noqa: E402
from chromstate.dataset import ChromStateDataset  # noqa: E402
from toolbench.realdata import read_gtf_genes  # noqa: E402


def load_dataset(chroms, conds=("Naive", "Arm", "Cl13")) -> ChromStateDataset:
    frames = []
    for chrom in chroms:
        z = np.load(os.path.join(BINNED, f"{chrom}.npz"))
        X, zc, marks, bs = z["X"], list(z["conditions"]), list(z["marks"]), int(z["bin_size"])
        n = X.shape[1]
        for cond in conds:
            ci = zc.index(cond)
            f = pd.DataFrame({"chrom": chrom, "start": np.arange(n) * bs, "end": (np.arange(n) + 1) * bs,
                              "sample": cond})
            for mi, m in enumerate(marks):
                f[m] = np.rint(X[ci, :, mi]).astype(np.float64)
            frames.append(f)
    frame = pd.concat(frames, ignore_index=True).sort_values(["sample", "chrom", "start"]).reset_index(drop=True)
    chains, start = [], 0
    for _, grp in frame.groupby(["sample", "chrom"], sort=False):
        chains.append(slice(start, start + len(grp)))
        start += len(grp)
    Xm = frame[marks].to_numpy(dtype=np.float64)
    return ChromStateDataset(X=Xm, marks=list(marks), bins=frame[["chrom", "start", "end", "sample"]],
                             chains=chains, size_factors=np.ones_like(Xm))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--chroms", default="chr19")
    p.add_argument("-k", "--n-states", type=int, default=12)
    p.add_argument("--definitions", default=None, help="'default' or a definitions file")
    p.add_argument("--anchor", default="seed", choices=["seed", "fixed"])
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--promoter", type=int, default=1000)
    p.add_argument("--model", default="hgb")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    chroms = a.chroms.split(",")
    ds = load_dataset(chroms)
    cs.normalize(ds, method="median_ratio")
    print(f"dataset: {ds.n_rows:,} bin-rows, marks {ds.marks}, samples {ds.samples} ({time.time()-t0:.0f}s)")

    defs = None
    if a.definitions:
        defs = cs.StateDefinitions.default(ds.marks) if a.definitions == "default" \
            else cs.StateDefinitions.from_file(a.definitions, ds.marks)
        print("definitions:\n", defs.to_frame().to_string(index=False))
    seg = cs.fit(ds, n_states=a.n_states, n_restarts=a.restarts, seed=a.seed, definitions=defs, anchor=a.anchor)
    print(f"fit: loglik {seg.loglik:,.1f}, {seg.meta['n_restarts']} restarts, hits {seg.meta['restart_hits_at_best']} "
          f"({time.time()-t0:.0f}s)")
    seg.emission_frame().to_csv(os.path.join(a.out, "emissions.tsv"), sep="\t")
    seg.state_table().to_csv(os.path.join(a.out, "states.tsv"), sep="\t", index=False)
    seg.sample_state_matrix().to_csv(os.path.join(a.out, "occupancy.tsv"), sep="\t")
    print(seg.state_table()[["label", "rule", "coverage_soft"]].to_string(index=False))

    features = read_gtf_genes(GTF, chroms=chroms)
    print(f"{len(features):,} protein-coding genes on {chroms}")
    gq = cs.gene_quantification(seg, features, part_kwargs={"promoter": {"upstream": a.promoter, "downstream": a.promoter}})
    gq.to_csv(os.path.join(a.out, "gene_quantification.tsv.gz"), sep="\t", index=False)

    expr = pd.read_csv(os.path.join(RNA, "expression.tsv"), sep="\t").set_index("gene_id")
    expr = np.log2(expr[ds.samples] + 1.0)
    expr = expr.loc[expr.index.intersection(gq["gene_id"].unique())]
    lfc = cs.read_lfc(os.path.join(RNA, "de_Cl13_vs_Naive.tsv"))

    level = cs.predict_expression(gq, expression=expr, model=a.model, seed=a.seed)
    print(level.note)
    level.write(os.path.join(a.out, "level"), seg=seg)
    change = cs.predict_expression(gq, lfc=lfc, contrast=(["Naive"], ["Cl13"]), model=a.model, seed=a.seed)
    print(change.note)
    change.write(os.path.join(a.out, "change"), seg=seg)

    # region-level quantification example: every promoter, both as a table and dominant-state counts
    prom = features.promoters(a.promoter, a.promoter)
    prom["name"] = prom["gene_id"]
    rq = cs.quantify_regions(seg, prom)
    rq.to_csv(os.path.join(a.out, "promoter_quantification.tsv.gz"), sep="\t", index=False)
    dom = rq.groupby(["sample", "dominant"]).size().unstack(fill_value=0)
    dom.to_csv(os.path.join(a.out, "promoter_dominant_counts.tsv"), sep="\t")
    print(dom)

    summary = {"chroms": chroms, "n_states": seg.n_states, "definitions": a.definitions, "anchor": a.anchor,
               "loglik": seg.loglik, "labels": seg.labels, "level": level.summary, "change": change.summary,
               "seconds": time.time() - t0}
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"done in {time.time()-t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
