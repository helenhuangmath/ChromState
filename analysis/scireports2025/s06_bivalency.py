"""Step 6: ChromState bivalency against the published poised-promoter calls.

The paper called naive promoter classes from peak overlap (H3K4me3 and/or H3K27me3
peaks within the 2.5 kb promoter window) and then, among naive-poised promoters,
which resolved towards activation or repression in TMEM/TEX.

Two questions:
1  Concordance -- do ChromState's continuous bivalency scores rank the peak-called
   bivalent promoters first? (Same data underneath, so this is a consistency check,
   not independent validation.)
2  Anticipation -- among naive-poised promoters, does the *naive* quantitative state
   (how strong each mark is, how balanced) predict which way the promoter resolves
   later, and the direction of the expression change? This is the "poised in TN,
   then activated or repressed" finding stated as a prediction made from TN alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from common import OUT, POISED_DIR, ensure_out
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

CLASS_FILES = {
    "Bivalent": "bivalent_promoter.bed",
    "H3K4me3 only": "other_promoter.only_H3K4me3.bed",
    "H3K27me3 only": "other_promoter.only_H3K27me3.bed",
}
RESOLUTION = {
    "Arm": ("PoisedP_change_AvsN_Active_all.bed", "PoisedP_change_AvsN_Repressive_all.bed"),
    "Cl13": ("PoisedP_change_CvsN_Active_all.bed", "PoisedP_change_CvsN_Repressive_all.bed"),
}
SCORES = {
    "ChromState bivalency (within-state)": "P_cs_biv_within",
    "ChromState bivalency score (signal)": "P_cs_biv_score",
    "H3K4me3 x H3K27me3 magnitude": "P_cs_biv_magnitude",
}


def expression_matched_excess(de: pd.DataFrame, naive_expr: pd.Series, n_bins: int = 20) -> pd.Series:
    """log2FC minus the median log2FC of all genes with similar naive expression.

    Fold changes regress to the mean: genes barely expressed in TN have the most room to
    go up. Comparing each gene with genome-wide genes in the same TN-expression bin
    removes that, so what is left is change beyond what expression level alone predicts.
    """
    d = de[["log2FoldChange"]].join(naive_expr.rename("naive"), how="inner").dropna()
    d["bin"] = pd.qcut(d["naive"].rank(method="first"), n_bins, labels=False)
    expected = d.groupby("bin")["log2FoldChange"].transform("median")
    return (d["log2FoldChange"] - expected).rename("excess_log2fc")


def gene_names(path, col: int) -> set:
    return set(pd.read_csv(POISED_DIR / path, sep="\t", header=None)[col].astype(str))


def main() -> None:
    out = ensure_out("bivalency")
    feats = pd.read_csv(OUT / "genes" / "gene_features.tsv.gz", sep="\t")
    genes = pd.read_csv(OUT / "rna" / "genes.tsv", sep="\t")
    naive = feats.loc[feats["condition"] == "Naive"].merge(genes[["gene_id", "name"]], on="gene_id")
    naive = naive.dropna(subset=["P_cs_biv_within"])
    expr = pd.read_csv(OUT / "rna" / "expression.tsv", sep="\t", index_col="gene_id")
    naive["K4_minus_K27"] = naive["P_raw_H3K4me3"] - naive["P_raw_H3K27me3"]

    cls = pd.Series("Neither", index=naive.index)
    for label in ["H3K27me3 only", "H3K4me3 only", "Bivalent"]:
        cls[naive["name"].isin(gene_names(CLASS_FILES[label], 3))] = label
    naive["promoter_class"] = cls
    naive[["gene_id", "name", "promoter_class", *SCORES.values(), "P_cs_biv_balance",
           "P_raw_H3K4me3", "P_raw_H3K27me3"]].to_csv(out / "naive_promoters.tsv", sep="\t", index=False)

    rows = []
    is_biv = (naive["promoter_class"] == "Bivalent").to_numpy()
    for label, col in SCORES.items():
        s = naive[col].to_numpy()
        rows.append({"score": label, "auroc": roc_auc_score(is_biv, s),
                     "auprc": average_precision_score(is_biv, s), "prevalence": is_biv.mean(),
                     "n": len(s)})
    conc = pd.DataFrame(rows)
    conc.to_csv(out / "concordance.tsv", sep="\t", index=False)
    print(conc.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    res_rows, expr_rows = [], []
    for cond, (act_f, rep_f) in RESOLUTION.items():
        act, rep = gene_names(act_f, 3), gene_names(rep_f, 3)
        sub = naive.loc[naive["name"].isin(act | rep)].copy()
        sub["resolved_active"] = sub["name"].isin(act).astype(int)
        y = sub["resolved_active"].to_numpy()
        for label, col in {"Naive H3K4me3 - H3K27me3 (log)": "K4_minus_K27",
                           "Naive bivalency balance": "P_cs_biv_balance",
                           "Naive active-state mass": None}.items():
            if col is None:
                active_cols = [c for c in sub.columns if c.startswith(("P_cs_TssA", "P_cs_TssFlnk"))]
                x = sub[active_cols].sum(axis=1).to_numpy() if active_cols else np.zeros(len(sub))
            else:
                x = sub[col].to_numpy()
            res_rows.append({"condition": cond, "predictor": label, "auroc": roc_auc_score(y, x),
                             "n_active": int(y.sum()), "n_repressive": int((1 - y).sum())})

        name = f"{cond}_vs_Naive"
        de = pd.read_csv(OUT / "rna" / f"de_{name}.tsv", sep="\t", index_col="gene_id")
        de = de.loc[de["baseMean"] >= 10]
        excess = expression_matched_excess(de, expr["Naive"])
        biv = naive.loc[naive["promoter_class"] == "Bivalent"].set_index("gene_id")
        biv = biv.join(de[["log2FoldChange", "baseMean"]], how="inner").dropna(subset=["log2FoldChange"])
        biv = biv.join(excess, how="inner")
        rho_ex = spearmanr(biv["K4_minus_K27"], biv["excess_log2fc"])
        rho = spearmanr(biv["K4_minus_K27"], biv["log2FoldChange"])
        q = pd.qcut(biv["K4_minus_K27"], 4, labels=["Q1 (K27me3-dominant)", "Q2", "Q3",
                                                    "Q4 (K4me3-dominant)"])
        for qi, grp in biv.groupby(q, observed=True):
            expr_rows.append({"condition": cond, "quartile": qi, "n": len(grp),
                              "median_log2fc": grp["log2FoldChange"].median(),
                              "median_excess_log2fc": grp["excess_log2fc"].median(),
                              "frac_up": float((grp["log2FoldChange"] > 0).mean()),
                              "frac_excess_up": float((grp["excess_log2fc"] > 0).mean())})
        res_rows.append({"condition": cond, "predictor": "Spearman(K4-K27, log2FC) bivalent genes",
                         "auroc": rho.correlation, "n_active": len(biv), "n_repressive": np.nan})
        res_rows.append({"condition": cond,
                         "predictor": "Spearman(K4-K27, expression-matched excess log2FC)",
                         "auroc": rho_ex.correlation, "n_active": len(biv), "n_repressive": np.nan})
        lo = biv.loc[q == q.cat.categories[0], "log2FoldChange"]
        hi = biv.loc[q == q.cat.categories[-1], "log2FoldChange"]
        print(f"{cond}: expression-matched rho={rho_ex.correlation:.3f} p={rho_ex.pvalue:.2g}")
        print(f"{cond}: bivalent genes {len(biv)}, rho={rho.correlation:.3f} p={rho.pvalue:.2g}, "
              f"Q4 vs Q1 Mann-Whitney p={mannwhitneyu(hi, lo).pvalue:.2g}")

    res = pd.DataFrame(res_rows)
    res.to_csv(out / "resolution.tsv", sep="\t", index=False)
    pd.DataFrame(expr_rows).to_csv(out / "bivalent_quartiles.tsv", sep="\t", index=False)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(pd.DataFrame(expr_rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
