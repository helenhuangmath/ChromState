"""Step 11: how reproducible is the chromatin signal? Reliability from per-replicate tracks.

The analysis uses replicate-pooled tracks, so chromatin noise could not be measured
directly. The per-replicate CPM tracks that exist locally allow it for part of the
design. For each mark and cell state with >= 2 replicates:

  r1  mean pairwise Spearman correlation of promoter signal (TSS +/- 1 kb) between
      single replicates, over expressed genes -- the reliability of ONE replicate;
  Rn  Spearman-Brown reliability of the mean of n replicates, n r1 / (1 + (n-1) r1),
      i.e. the reliability of the pooled track used in the analysis.

RNA reliability is computed the same way from the TN RNA-seq replicates. Step 9 uses
these numbers to correct the cross-lagged regressions for measurement error.

Coverage of the local per-replicate tracks (others were not available):
  H3K27ac  TN x8, TMEM x4, TEX x4      H3K4me3  TN x7, TEX x3
  H3K27me3 TEX x3                      H3K9me3  TMEM x2
"""

from __future__ import annotations

import itertools
import re

import numpy as np
import pandas as pd
import pyBigWig
from common import CHROMS, DATA, OUT, RNA_COUNTS, ensure_out
from scipy.stats import spearmanr

from chromstate.features import FeatureSet

REP_DIR = DATA / "Wig"
NAME_RE = re.compile(r"^(Naive|Arm|Cl13)_P14_Day\d+_(H3K\w+?)_Rep(\d)_S\d+\.CPM\.bw$")
MIN_BASEMEAN = 10.0


def spearman_brown(r1: float, n: int) -> float:
    return n * r1 / (1 + (n - 1) * r1)


def promoter_signal(path, prom: pd.DataFrame) -> np.ndarray:
    bw = pyBigWig.open(str(path))
    sizes = bw.chroms()
    out = np.full(len(prom), np.nan)
    for i, (c, s, e) in enumerate(zip(prom["chrom"], prom["start"], prom["end"])):
        if c in sizes:
            v = bw.stats(c, int(max(s, 0)), int(min(e, sizes[c])), type="mean", exact=True)[0]
            out[i] = 0.0 if v is None else v
    bw.close()
    return np.log1p(out)


def main() -> None:
    out = ensure_out("reliability")
    genes = pd.read_csv(OUT / "rna" / "genes.tsv", sep="\t")
    genes = genes.loc[genes["chrom"].isin(CHROMS)]
    de = pd.read_csv(OUT / "rna" / "de_Cl13_vs_Naive.tsv", sep="\t", index_col="gene_id")
    expressed = de.index[de["baseMean"] >= MIN_BASEMEAN]
    genes = genes.loc[genes["gene_id"].isin(expressed)].reset_index(drop=True)
    prom = FeatureSet(genes, source="rna").promoters(upstream=1000, downstream=1000)

    files = {}
    for p in sorted(REP_DIR.glob("*.CPM.bw")):
        m = NAME_RE.match(p.name)
        if m:
            files.setdefault((m.group(2), m.group(1)), []).append(p)

    rows = []
    for (mark, cond), paths in sorted(files.items()):
        if len(paths) < 2:
            continue
        sig = np.column_stack([promoter_signal(p, prom) for p in paths])
        ok = np.isfinite(sig).all(axis=1)
        pairs = [spearmanr(sig[ok, i], sig[ok, j]).correlation
                 for i, j in itertools.combinations(range(sig.shape[1]), 2)]
        r1 = float(np.mean(pairs))
        n_pool = 8 if cond == "Naive" else 4   # replicates pooled into the analysis track
        rows.append({"mark": mark, "condition": cond, "n_replicates_here": len(paths),
                     "single_replicate_r": r1, "pooled_n": n_pool,
                     "pooled_reliability": spearman_brown(r1, n_pool), "n_genes": int(ok.sum())})
        print(f"{mark:9s} {cond:6s} reps={len(paths)}  single r={r1:.3f}  "
              f"pooled({n_pool}) R={rows[-1]['pooled_reliability']:.3f}", flush=True)

    # RNA: TN replicates (8), same procedure on log CPM
    raw = pd.read_csv(RNA_COUNTS, sep="\t", comment="#")
    raw["gene_id"] = raw["Geneid"].str.replace(r"\.\d+$", "", regex=True)
    raw = raw.drop_duplicates("gene_id").set_index("gene_id").loc[lambda d: d.index.isin(expressed)]
    cols = [c for c in raw.columns if c.startswith("Naive_") and "_RNA_" in c]
    cpm = np.log2(raw[cols] / raw[cols].sum(axis=0) * 1e6 + 1).to_numpy()
    r1 = float(np.mean([spearmanr(cpm[:, i], cpm[:, j]).correlation
                        for i, j in itertools.combinations(range(len(cols)), 2)]))
    rows.append({"mark": "RNA", "condition": "Naive", "n_replicates_here": len(cols),
                 "single_replicate_r": r1, "pooled_n": len(cols),
                 "pooled_reliability": spearman_brown(r1, len(cols)), "n_genes": len(raw)})
    print(f"RNA       Naive  reps={len(cols)}  single r={r1:.3f}  pooled R={rows[-1]['pooled_reliability']:.3f}")

    pd.DataFrame(rows).to_csv(out / "reliability.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
