"""Step 8: which chromatin states matter for expression, and is it additive or combinatorial?

Two analyses, each on expression level (TN) and on expression change (TMEM vs TN, TEX vs
TN), all scored on held-out chromosomes (5 folds grouped by chromosome).

A  State importance. A gradient-boosted model on ChromState's promoter + body state
   composition. For every state, two numbers:
     marginal  -- held-out Spearman rho of a model that sees only that state;
     unique    -- drop in held-out R^2 when that state's columns are permuted in the full
                  model (averaged over 5 permutations), i.e. what no other state can
                  substitute for.
   The two disagree for redundant states (high marginal, low unique), and that
   disagreement is the point of reporting both.

B  Additive versus combinatorial marks. Models on the four marks (promoter + body):
     single mark          -- each mark alone
     additive             -- all marks, interactions forbidden (sum of per-feature
                             effects; HistGradientBoosting ``interaction_cst``)
     + one mark pair      -- additive, plus interactions allowed within one pair of marks
     full                 -- all interactions allowed
     ChromState states    -- the state composition, which encodes combinations by design
   full - additive is the combinatorial contribution; the pair models say which
   combination carries it. Differences are bootstrapped over genes (paired).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from common import CONDITIONS, MARKS, MODEL_DIR, OUT, ensure_out
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

MIN_BASEMEAN = 10.0
N_BOOT = 500
N_PERM = 5
SEED = 0


def r2(y, p):
    return 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def model(interaction_cst=None):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                         interaction_cst=interaction_cst, random_state=SEED)


def cv_fit_predict(X, y, groups, interaction_cst=None, perm_groups=None, rng=None):
    """Out-of-fold predictions, plus per-group permutation R^2 drops on the held-out fold."""
    pred = np.full(len(y), np.nan)
    drops = {g: [] for g in (perm_groups or {})}
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = model(interaction_cst).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        base = r2(y[te], pred[te])
        for g, cols in (perm_groups or {}).items():
            vals = []
            for _ in range(N_PERM):
                Xp = X[te].copy()
                order = rng.permutation(len(te))
                Xp[:, cols] = Xp[order][:, cols]  # permute the group's columns jointly
                vals.append(base - r2(y[te], m.predict(Xp)))
            drops[g].append(np.mean(vals) * len(te))
    n = len(y)
    return pred, {g: float(np.sum(v) / n) for g, v in drops.items()}


def boot_rho(y, p, idx):
    return np.array([spearmanr(y[i], p[i]).correlation for i in idx])


def tasks(feats, expr, de):
    by = {c: feats.loc[feats["condition"] == c].set_index("gene_id") for c in CONDITIONS}
    shared = sorted(set.intersection(*(set(d.dropna(subset=["P_raw_H3K4me3"]).index)
                                       for d in by.values())))
    by = {c: d.loc[shared] for c, d in by.items()}
    out = {"Level (TN)": (by["Naive"], expr["Naive"].reindex(shared))}
    for name, cond in [("Change (TMEM vs TN)", "Arm"), ("Change (TEX vs TN)", "Cl13")]:
        d = de[f"{cond}_vs_Naive"]
        keep = d.index[(d["baseMean"] >= MIN_BASEMEAN) & d["log2FoldChange"].notna()]
        genes = keep.intersection(shared)
        num = by[cond].select_dtypes("number")
        delta = (num - by["Naive"].select_dtypes("number")).loc[genes]
        delta["chrom"] = by["Naive"].loc[genes, "chrom"]
        out[name] = (delta, d.loc[genes, "log2FoldChange"])
    return out


def main() -> None:
    res = ensure_out("importance")
    feats = pd.read_csv(OUT / "genes" / "gene_features.tsv.gz", sep="\t")
    expr = pd.read_csv(OUT / "rna" / "expression.tsv", sep="\t", index_col="gene_id")
    de = {f"{c}_vs_Naive": pd.read_csv(OUT / "rna" / f"de_{c}_vs_Naive.tsv", sep="\t",
                                      index_col="gene_id") for c in ["Arm", "Cl13"]}
    labels = pd.read_csv(OUT / MODEL_DIR / "state_labels.tsv", sep="\t")["label"].tolist()
    rng = np.random.default_rng(SEED)

    state_rows, combo_rows, combo_diff = [], [], []
    for task, (X_df, y_s) in tasks(feats, expr, de).items():
        y_s = y_s.dropna()
        X_df = X_df.loc[y_s.index]
        y = y_s.to_numpy(dtype=float)
        groups = X_df["chrom"].to_numpy()
        boot_idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))
        print(f"\n== {task}: {len(y):,} genes", flush=True)

        # ---- A: state importance -------------------------------------------------
        state_cols = [c for lbl in labels for c in (f"P_cs_{lbl}", f"B_cs_{lbl}")]
        Xs = X_df[state_cols].to_numpy(dtype=float)
        perm = {lbl: [state_cols.index(f"P_cs_{lbl}"), state_cols.index(f"B_cs_{lbl}")]
                for lbl in labels}
        p_full, drops = cv_fit_predict(Xs, y, groups, perm_groups=perm, rng=rng)
        full_r2 = r2(y, p_full)
        for lbl in labels:
            cols = perm[lbl]
            p1, _ = cv_fit_predict(Xs[:, cols], y, groups)
            state_rows.append({"task": task, "state": lbl,
                               "marginal_rho": spearmanr(y, p1).correlation,
                               "unique_r2_drop": drops[lbl], "full_r2": full_r2})
        top = sorted(((drops[lbl], lbl) for lbl in labels), reverse=True)[:4]
        print("  unique R2 drop, top states:", ", ".join(f"{lbl} {d:.3f}" for d, lbl in top))

        # ---- B: additive vs combinatorial marks -----------------------------------
        mark_cols = [f"{part}_raw_{m}" for m in MARKS for part in ("P", "B")]
        Xm = X_df[mark_cols].to_numpy(dtype=float)
        idx_of = {m: [mark_cols.index(f"P_raw_{m}"), mark_cols.index(f"B_raw_{m}")] for m in MARKS}
        singletons = [[i] for i in range(len(mark_cols))]

        preds = {}
        for m in MARKS:
            preds[f"{m} alone"], _ = cv_fit_predict(Xm[:, idx_of[m]], y, groups)
        preds["Additive (no interactions)"], _ = cv_fit_predict(Xm, y, groups, singletons)
        for a, b in itertools.combinations(MARKS, 2):
            pair = idx_of[a] + idx_of[b]
            cst = [pair] + [[i] for i in range(len(mark_cols)) if i not in pair]
            preds[f"Additive + {a} x {b}"], _ = cv_fit_predict(Xm, y, groups, cst)
        preds["Full interactions"], _ = cv_fit_predict(Xm, y, groups)
        preds["ChromState states"] = p_full

        boots = {}
        for name, p in preds.items():
            bs = boot_rho(y, p, boot_idx)
            boots[name] = bs
            combo_rows.append({"task": task, "model": name,
                               "spearman": spearmanr(y, p).correlation, "r2": r2(y, p),
                               "rho_lo": np.quantile(bs, 0.025), "rho_hi": np.quantile(bs, 0.975)})
        add = boots["Additive (no interactions)"]
        for name in [k for k in preds if k.startswith("Additive +")] + ["Full interactions",
                                                                        "ChromState states"]:
            d = boots[name] - add
            combo_diff.append({"task": task, "model": name, "gain_over_additive": float(d.mean()),
                               "lo": float(np.quantile(d, 0.025)), "hi": float(np.quantile(d, 0.975))})
        cr = pd.DataFrame([r for r in combo_rows if r["task"] == task])
        print(cr[["model", "spearman", "rho_lo", "rho_hi", "r2"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"), flush=True)

    pd.DataFrame(state_rows).to_csv(res / "state_importance.tsv", sep="\t", index=False)
    pd.DataFrame(combo_rows).to_csv(res / "additive_vs_combinatorial.tsv", sep="\t", index=False)
    diffs = pd.DataFrame(combo_diff)
    diffs.to_csv(res / "combinatorial_gain.tsv", sep="\t", index=False)
    print("\n", diffs.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
