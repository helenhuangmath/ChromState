"""Step 9: does chromatin anticipate expression, or follow it? A cross-lagged test.

TN cells are the precursors of TMEM and TEX, so TN -> later is a real (if coarse) time
step. For each mark (promoter and body) and each ChromState state, two regressions on
genes, all variables rank-normalised:

  chromatin -> RNA   RNA_later       ~ RNA_TN + chromatin_TN   coefficient of chromatin_TN
  RNA -> chromatin   chromatin_later ~ chromatin_TN + RNA_TN   coefficient of RNA_TN

If TN chromatin predicts where expression goes, beyond what TN expression already says,
the mark behaves as if it acts first (priming). If TN expression predicts where the mark
goes, beyond its own TN level, the mark behaves as if it follows transcription. The
asymmetry |chromatin -> RNA| - |RNA -> chromatin| summarises which way it leans.
95% CIs: 1,000 bootstrap resamples of genes.

This is NOT a causal or temporal measurement:
* two time points, different cell populations (TN vs its progeny 30 days on);
* a third factor (transcription-factor binding) can drive both, in either order;
* measurement error: noise in a TN variable attenuates its own slope and leaves the
  other predictor to absorb what it failed to control for. With correlated predictors
  (TN expression and TN H3K27me3 are strongly anti-correlated) the bias can go either
  way, so it is not assumed: reliabilities measured from replicates (step 11) are used
  in an errors-in-variables regression, and both raw and corrected results are reported.
  States have no replicate-level calls; their correction uses the reliability of their
  least reliable enriched mark, as a sensitivity analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from common import MARKS, MODEL_DIR, OUT, RNA_COUNTS, ensure_out
from scipy.stats import norm, rankdata, spearmanr

MIN_BASEMEAN = 10.0
N_BOOT = 1000
SEED = 0


def rank_normal(x: np.ndarray) -> np.ndarray:
    return norm.ppf((rankdata(x) - 0.5) / len(x))


def ols_coefs(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(y)), X])
    return np.linalg.lstsq(X1, y, rcond=None)[0][1:]


def eiv_coefs(y: np.ndarray, X: np.ndarray, reliability: np.ndarray) -> np.ndarray:
    """Errors-in-variables regression on standardised predictors.

    Measurement noise in a predictor inflates its observed variance by (1 - reliability)
    and attenuates its slope; it also leaves the *other* predictor to absorb what the
    noisy one failed to control for. Subtracting the error variance from the diagonal of
    the predictor covariance before solving gives slopes for the error-free predictors
    (the classical disattenuation; noise in the outcome biases nothing).
    """
    Xs = (X - X.mean(axis=0)) / X.std(axis=0)
    ys = y - y.mean()
    S = np.cov(Xs, rowvar=False)
    S_true = S - np.diag(1.0 - np.asarray(reliability))
    c = Xs.T @ ys / (len(y) - 1)
    return np.linalg.solve(S_true, c)


def cross_lag(chrom_tn, chrom_late, rna_tn, rna_late, boot_idx, lam_chrom=1.0, lam_rna=1.0):
    """Cross-lagged coefficients; with reliabilities < 1, corrected for measurement error."""
    c0, c1, r0, r1 = (rank_normal(v) for v in (chrom_tn, chrom_late, rna_tn, rna_late))
    corrected = lam_chrom < 1.0 or lam_rna < 1.0

    def both(i):
        if corrected:
            a = eiv_coefs(r1[i], np.column_stack([r0[i], c0[i]]), [lam_rna, lam_chrom])[1]
            b = eiv_coefs(c1[i], np.column_stack([c0[i], r0[i]]), [lam_chrom, lam_rna])[1]
        else:
            a = ols_coefs(r1[i], np.column_stack([r0[i], c0[i]]))[1]   # chromatin -> RNA
            b = ols_coefs(c1[i], np.column_stack([c0[i], r0[i]]))[1]   # RNA -> chromatin
        return a, b

    a, b = both(np.arange(len(c0)))
    bs = np.array([both(i) for i in boot_idx])
    asym = np.abs(bs[:, 0]) - np.abs(bs[:, 1])
    return {
        "chrom_to_rna": a, "chrom_to_rna_lo": np.quantile(bs[:, 0], 0.025),
        "chrom_to_rna_hi": np.quantile(bs[:, 0], 0.975),
        "rna_to_chrom": b, "rna_to_chrom_lo": np.quantile(bs[:, 1], 0.025),
        "rna_to_chrom_hi": np.quantile(bs[:, 1], 0.975),
        "asymmetry": abs(a) - abs(b), "asym_lo": np.quantile(asym, 0.025),
        "asym_hi": np.quantile(asym, 0.975),
    }


def rna_reliability() -> float:
    """Spearman between TN day-30 and day-32 mean log-CPM: how reproducible RNA_TN is."""
    raw = pd.read_csv(RNA_COUNTS, sep="\t", comment="#")
    cols = [c for c in raw.columns if c.startswith("Naive_") and "_RNA_" in c]
    counts = raw[cols].astype(float)
    cpm = np.log2(counts / counts.sum(axis=0) * 1e6 + 1)
    d30 = cpm[[c for c in cols if "Day30" in c]].mean(axis=1)
    d32 = cpm[[c for c in cols if "Day32" in c]].mean(axis=1)
    keep = (counts.mean(axis=1) >= 10).to_numpy()
    return float(spearmanr(d30[keep], d32[keep]).correlation)


def lean(row) -> str:
    if row["asym_lo"] > 0:
        return "chromatin leads"
    if row["asym_hi"] < 0:
        return "RNA leads"
    return "no clear order"


def main() -> None:
    out = ensure_out("cross_lag")
    feats = pd.read_csv(OUT / "genes" / "gene_features.tsv.gz", sep="\t")
    expr = pd.read_csv(OUT / "rna" / "expression.tsv", sep="\t", index_col="gene_id")
    labels = pd.read_csv(OUT / MODEL_DIR / "state_labels.tsv", sep="\t")["label"].tolist()
    by = {c: feats.loc[feats["condition"] == c].set_index("gene_id") for c in ["Naive", "Arm", "Cl13"]}
    rng = np.random.default_rng(SEED)

    # Measured reliabilities (step 11). TN where replicates exist, otherwise the cell
    # state that has them, as a proxy. Gene-body reliability is taken from the promoter.
    rel = pd.read_csv(OUT / "reliability" / "reliability.tsv", sep="\t")
    lam = {}
    for mark in MARKS:
        sub = rel.loc[rel["mark"] == mark]
        tn = sub.loc[sub["condition"] == "Naive"]
        lam[mark] = float((tn if len(tn) else sub)["pooled_reliability"].iloc[0])
    lam_rna = float(rel.loc[rel["mark"] == "RNA", "pooled_reliability"].iloc[0])

    # States have no replicate-level calls, so their reliability cannot be measured.
    # Sensitivity analysis: give each state the reliability of its least reliable
    # enriched mark (fold >= 1.5 over the genome mean); background states stay uncorrected.
    import pickle

    with open(OUT / MODEL_DIR / "model.pkl", "rb") as fh:
        hmm = pickle.load(fh)["hmm"]
    M = hmm.emissions.mean_matrix()
    occ = pd.read_csv(OUT / MODEL_DIR / "occupancy_genome.tsv", sep="\t",
                      index_col=0).loc["Naive", labels].to_numpy()
    fold = M / ((occ[:, None] * M).sum(axis=0) / occ.sum())
    lam_state = {}
    for k, lbl in enumerate(labels):
        enriched = [m for m, f in zip(MARKS, fold[k]) if f >= 1.5]
        lam_state[lbl] = min(lam[m] for m in enriched) if enriched else None

    variables = ([(f"{m} promoter", f"P_raw_{m}", "mark") for m in MARKS]
                 + [(f"{m} gene body", f"B_raw_{m}", "mark") for m in MARKS]
                 + [(f"{lbl} (promoter state)", f"P_cs_{lbl}", "state") for lbl in labels])

    rows = []
    for later, cname in [("Arm", "TMEM"), ("Cl13", "TEX")]:
        de = pd.read_csv(OUT / "rna" / f"de_{later}_vs_Naive.tsv", sep="\t", index_col="gene_id")
        genes = de.index[de["baseMean"] >= MIN_BASEMEAN]
        genes = genes.intersection(by["Naive"].dropna(subset=["P_raw_H3K4me3", "B_raw_H3K4me3"]).index)
        genes = genes.intersection(by[later].index).intersection(expr.index)
        boot_idx = rng.integers(0, len(genes), size=(N_BOOT, len(genes)))
        rna_tn = expr.loc[genes, "Naive"].to_numpy()
        rna_late = expr.loc[genes, later].to_numpy()
        for name, col, kind in variables:
            args = (by["Naive"].loc[genes, col].to_numpy(), by[later].loc[genes, col].to_numpy(),
                    rna_tn, rna_late, boot_idx)
            res = cross_lag(*args)
            row = {"later": cname, "variable": name, "kind": kind, "n_genes": len(genes), **res}
            lam_v = lam[name.split()[0]] if kind == "mark" else lam_state.get(
                name.replace(" (promoter state)", ""))
            if lam_v is not None:
                cor = cross_lag(*args, lam_chrom=lam_v, lam_rna=lam_rna)
                row.update({"reliability": lam_v, **{f"{k}_corrected": v for k, v in cor.items()}})
            rows.append(row)
        print(f"{cname}: {len(genes):,} expressed genes", flush=True)

    table = pd.DataFrame(rows)
    table["lean"] = table.apply(lean, axis=1)
    has = table["asym_lo_corrected"].notna() if "asym_lo_corrected" in table else False
    table.loc[has, "lean_corrected"] = table.loc[has].apply(
        lambda r: lean({"asym_lo": r["asym_lo_corrected"], "asym_hi": r["asym_hi_corrected"]}), axis=1)
    table.to_csv(out / "cross_lag.tsv", sep="\t", index=False)
    rel = rna_reliability()
    pd.DataFrame({"rna_tn_reliability_d30_vs_d32": [rel]}).to_csv(out / "reliability.tsv", sep="\t", index=False)
    show = ["later", "variable", "chrom_to_rna", "rna_to_chrom", "asymmetry", "asym_lo", "asym_hi", "lean"]
    print(table.loc[table["kind"] == "mark", show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    showc = ["later", "variable", "reliability", "asymmetry", "asymmetry_corrected",
             "asym_lo_corrected", "asym_hi_corrected", "lean_corrected"]
    print(table.loc[table["kind"] == "mark", showc].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(table.loc[table["kind"] == "state", showc].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"RNA_TN reliability (TN day 30 vs day 32, Spearman) = {rel:.3f}")


if __name__ == "__main__":
    main()
