"""Step 3: expression levels and differential expression from the raw featureCounts table.

Outputs ``out/rna/genes.tsv`` (gene coordinates for the FeatureSet), ``expression.tsv``
(log2 normalised counts, mean per condition) and ``de_<contrast>.tsv`` from pydeseq2.

TN was sampled twice (day 30 alongside TMEM, day 32 alongside TEX). The TN d32 vs TN d30
contrast measures how much batch/day alone moves expression -- the floor that a real
TN -> TMEM or TN -> TEX change has to clear.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from common import CHROMS, RNA_COUNTS, ensure_out
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

CONTRASTS = {
    "Arm_vs_Naive": ("condition", "Arm", "Naive"),
    "Cl13_vs_Naive": ("condition", "Cl13", "Naive"),
    "Cl13_vs_Arm": ("condition", "Cl13", "Arm"),
}


def sample_table(columns) -> pd.DataFrame:
    rows = []
    for c in columns:
        cond, _, day = c.split("_")[0], None, c.split("_")[2]
        rows.append({"sample": c, "condition": cond, "day": day,
                     "group": f"{cond}_{day}"})
    return pd.DataFrame(rows).set_index("sample")


def run_deseq(counts: pd.DataFrame, meta: pd.DataFrame, factor: str, contrasts) -> dict:
    dds = DeseqDataSet(counts=counts, metadata=meta, design_factors=factor,
                       refit_cooks=True, quiet=True)
    dds.deseq2()
    out = {}
    for name, (fac, a, b) in contrasts.items():
        st = DeseqStats(dds, contrast=[fac, a, b], quiet=True)
        st.summary()
        out[name] = st.results_df
    return out, dds


def main() -> None:
    out = ensure_out("rna")
    raw = pd.read_csv(RNA_COUNTS, sep="\t", comment="#")
    raw = raw.loc[raw["Chr"].isin(CHROMS)].copy()
    raw["gene_id"] = raw["Geneid"].str.replace(r"\.\d+$", "", regex=True)
    raw = raw.drop_duplicates("gene_id")

    genes = pd.DataFrame({
        "gene_id": raw["gene_id"], "name": raw["Gene.Name"], "chrom": raw["Chr"],
        "start": raw["Start"].astype(int) - 1, "end": raw["End"].astype(int),
        "strand": raw["Strand"],
    })
    genes.to_csv(out / "genes.tsv", sep="\t", index=False)

    sample_cols = [c for c in raw.columns if "_RNA_" in c]
    counts = raw.set_index("gene_id")[sample_cols].T.astype(int)
    counts = counts.loc[:, counts.sum(axis=0) >= 10]
    meta = sample_table(sample_cols)
    print(meta.groupby(["condition", "day"]).size())

    res, dds = run_deseq(counts, meta, "condition", CONTRASTS)
    for name, df in res.items():
        df.rename_axis("gene_id").to_csv(out / f"de_{name}.tsv", sep="\t")
        sig = (df["padj"] < 0.05) & (df["log2FoldChange"].abs() > 1)
        print(f"{name:14s} genes tested {df['padj'].notna().sum():6d}   "
              f"|log2FC|>1 & padj<0.05: {int(sig.sum())}")

    # Batch/day floor: TN day 32 vs TN day 30, same cells, different harvest.
    naive = meta.index[meta["condition"] == "Naive"]
    res_n, _ = run_deseq(counts.loc[naive], meta.loc[naive], "day",
                         {"Naive_d32_vs_d30": ("day", "Day32", "Day30")})
    df = res_n["Naive_d32_vs_d30"]
    df.rename_axis("gene_id").to_csv(out / "de_Naive_d32_vs_d30.tsv", sep="\t")
    sig = (df["padj"] < 0.05) & (df["log2FoldChange"].abs() > 1)
    print(f"Naive d32 vs d30 (batch floor): |log2FC|>1 & padj<0.05: {int(sig.sum())}")

    norm = pd.DataFrame(dds.layers["normed_counts"], index=counts.index, columns=counts.columns)
    log_expr = np.log2(norm + 1.0)
    expr = pd.DataFrame({cond: log_expr.loc[meta.index[meta["condition"] == cond]].mean(axis=0)
                         for cond in ["Naive", "Arm", "Cl13"]})
    expr["baseMean"] = norm.mean(axis=0)
    expr.rename_axis("gene_id").to_csv(out / "expression.tsv", sep="\t")
    print(f"wrote {len(expr):,} genes")


if __name__ == "__main__":
    main()
