"""Step 5: does ChromState's quantitative state predict expression better than the alternatives?

Three tasks, each scored by leave-chromosomes-out cross-validation (5 folds grouped by
chromosome, so neighbouring genes never sit on both sides of a split):

A  level        expression in a condition, from that condition's chromatin
B  change       log2FC (TMEM or TEX vs TN; TEX vs TMEM) from the change in chromatin
C  prepattern   log2FC from *naive* chromatin alone -- can the TN state anticipate the
                change that differentiation later makes? Scored against a baseline that
                knows only TN expression, because fold changes regress to the mean and
                a model can look predictive just by knowing how expressed a gene is.

Feature families (identical genes and bins for all):
  raw            mean log1p coverage per mark, promoter and body
  chmm_tss       published ChromHMM 15-state call at the TSS (one-hot)
  chmm_frac      fraction of promoter/body in each ChromHMM state
  cs_tss         ChromState Viterbi state at the TSS (one-hot)
  cs_quant       ChromState posterior composition, entropy, bivalency (promoter+body)
  cs_quant_raw   cs_quant + raw

Confidence intervals: 1,000 bootstrap resamples of genes over the out-of-fold
predictions; differences between families are paired on the same resample.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from common import CONDITIONS, MARKS, MODEL_DIR, OUT, ensure_out
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

FAMILIES = ["raw", "chmm_tss", "chmm_frac", "cs_tss", "cs_quant", "cs_quant_raw"]
MIN_BASEMEAN = 10.0
N_BOOT = 1000
SEED = 0


def load():
    feats = pd.read_csv(OUT / "genes" / "gene_features.tsv.gz", sep="\t")
    expr = pd.read_csv(OUT / "rna" / "expression.tsv", sep="\t", index_col="gene_id")
    de = {name: pd.read_csv(OUT / "rna" / f"de_{name}.tsv", sep="\t", index_col="gene_id")
          for name in ["Arm_vs_Naive", "Cl13_vs_Naive", "Cl13_vs_Arm"]}
    with open(OUT / MODEL_DIR / "model.pkl", "rb") as fh:
        k = pickle.load(fh)["k"]
    return feats, expr, de, k


def family_matrix(df: pd.DataFrame, family: str, k: int) -> pd.DataFrame:
    """Feature columns of one family for a (gene-indexed) single-condition table."""
    cols = df.columns
    if family == "raw":
        return df[[c for c in cols if c.startswith(("P_raw_", "B_raw_"))]]
    if family == "chmm_frac":
        return df[[c for c in cols if c.startswith(("P_chmm_", "B_chmm_"))]]
    if family == "cs_quant":
        return df[[c for c in cols if c.startswith(("P_cs_", "B_cs_"))]]
    if family == "cs_quant_raw":
        return df[[c for c in cols if c.startswith(("P_cs_", "B_cs_", "P_raw_", "B_raw_"))]]
    if family in ("chmm_tss", "cs_tss"):
        col, n = ("chmm_tss_state", 15) if family == "chmm_tss" else ("cs_tss_state", k)
        idx = df[col].fillna(-1).astype(int).to_numpy()
        oh = np.zeros((len(df), n))
        ok = idx >= 0
        oh[np.flatnonzero(ok), idx[ok]] = 1.0
        return pd.DataFrame(oh, index=df.index, columns=[f"{family}_{i}" for i in range(n)])
    raise ValueError(family)


def cv_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                              random_state=SEED)
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
    return pred


def r2(y, p):
    return 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def score_with_ci(y: np.ndarray, preds: dict, rng) -> pd.DataFrame:
    n = len(y)
    boot_idx = rng.integers(0, n, size=(N_BOOT, n))
    rows = []
    boot_rho = {}
    for name, p in preds.items():
        rho = spearmanr(y, p).correlation
        bs = np.array([spearmanr(y[i], p[i]).correlation for i in boot_idx])
        boot_rho[name] = bs
        rows.append({"family": name, "spearman": rho, "r2": r2(y, p),
                     "rho_lo": np.quantile(bs, 0.025), "rho_hi": np.quantile(bs, 0.975),
                     "n_genes": n})
    out = pd.DataFrame(rows)
    return out, boot_rho


def main() -> None:
    res_dir = ensure_out("predict")
    feats, expr, de, k = load()
    by_cond = {c: feats.loc[feats["condition"] == c].set_index("gene_id") for c in CONDITIONS}
    shared = sorted(set.intersection(*(set(d.dropna(subset=["P_raw_H3K4me3"]).index)
                                       for d in by_cond.values())))
    by_cond = {c: d.loc[shared] for c, d in by_cond.items()}
    rng = np.random.default_rng(SEED)

    results, oof, comparisons = [], {}, []

    def run(task: str, y: pd.Series, X_by_family: dict, extra: dict = None):
        genes = y.dropna().index
        yy = y.loc[genes].to_numpy()
        groups = by_cond["Naive"].loc[genes, "chrom"].to_numpy()
        preds = {}
        for fam, X in {**X_by_family, **(extra or {})}.items():
            preds[fam] = cv_predict(X.loc[genes].to_numpy(dtype=float), yy, groups)
        table, boot = score_with_ci(yy, preds, rng)
        table.insert(0, "task", task)
        results.append(table)
        oof[task] = pd.DataFrame(preds, index=genes).assign(y=yy)
        ref = "cs_quant"
        for other in boot:
            if other == ref:
                continue
            d = boot[ref] - boot[other]
            comparisons.append({"task": task, "contrast": f"{ref} - {other}",
                                "delta_rho": float(np.mean(d)),
                                "lo": float(np.quantile(d, 0.025)),
                                "hi": float(np.quantile(d, 0.975)),
                                "p_le_0": float(np.mean(d <= 0))})
        print(table[["task", "family", "spearman", "rho_lo", "rho_hi", "r2"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"), flush=True)

    fam = {c: {f: family_matrix(by_cond[c], f, k) for f in FAMILIES} for c in CONDITIONS}

    # A: level
    for c in CONDITIONS:
        run(f"A_level_{c}", expr[c].reindex(shared), fam[c])

    # B and C: change, on expressed genes only (filter set by RNA, never by chromatin)
    contrasts = {"Arm_vs_Naive": ("Arm", "Naive"), "Cl13_vs_Naive": ("Cl13", "Naive"),
                 "Cl13_vs_Arm": ("Cl13", "Arm")}
    for name, (b, a) in contrasts.items():
        d = de[name]
        keep = d.index[(d["baseMean"] >= MIN_BASEMEAN) & d["log2FoldChange"].notna()]
        y = d.loc[keep.intersection(shared), "log2FoldChange"]
        delta = {f: fam[b][f] - fam[a][f] for f in FAMILIES}
        run(f"B_change_{name}", y, delta)
        if a == "Naive":
            base = expr[["Naive"]].reindex(shared)
            extra = {"naive_rna": base,
                     "naive_rna+cs_quant": pd.concat([base, fam["Naive"]["cs_quant"]], axis=1),
                     "naive_rna+raw": pd.concat([base, fam["Naive"]["raw"]], axis=1)}
            run(f"C_prepattern_{name}", y, fam["Naive"], extra)

    # Which mark carries the change? Single-mark raw models for the change tasks.
    mark_rows = []
    for name, (b, a) in contrasts.items():
        d = de[name]
        keep = d.index[(d["baseMean"] >= MIN_BASEMEAN) & d["log2FoldChange"].notna()]
        genes = keep.intersection(shared)
        yy = d.loc[genes, "log2FoldChange"].to_numpy()
        groups = by_cond["Naive"].loc[genes, "chrom"].to_numpy()
        for m in MARKS:
            cols = [f"P_raw_{m}", f"B_raw_{m}"]
            X = (by_cond[b].loc[genes, cols] - by_cond[a].loc[genes, cols]).to_numpy()
            p = cv_predict(X, yy, groups)
            mark_rows.append({"contrast": name, "mark": m, "spearman": spearmanr(yy, p).correlation,
                              "r2": r2(yy, p)})
    marks = pd.DataFrame(mark_rows)
    print(marks.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    pd.concat(results).to_csv(res_dir / "scores.tsv", sep="\t", index=False)
    pd.DataFrame(comparisons).to_csv(res_dir / "comparisons.tsv", sep="\t", index=False)
    marks.to_csv(res_dir / "single_mark.tsv", sep="\t", index=False)
    with open(res_dir / "oof.pkl", "wb") as fh:
        pickle.dump(oof, fh)


if __name__ == "__main__":
    main()
