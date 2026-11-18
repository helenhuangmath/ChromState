"""Step 16: quantify every state per gene, and rank states by how much expression they explain.

Uses :mod:`chromstate.regulation` on the fitted model from step 2. For every gene x
condition, promoter (TSS +/- 1 kb) and body (TSS+1 kb to TES) as in step 4, each of the
K states gets three numbers -- occupancy, intensity, dose -- plus entropy, bivalency and
raw signal. The published ChromHMM features from step 4 are joined on the same genes.

Tasks (folds grouped by chromosome, as in step 5; change tasks keep genes with DESeq2
baseMean >= 10, a filter set by RNA, never by chromatin):

  level      expression in TN from TN chromatin
  change     log2FC TMEM vs TN, TEX vs TN, TEX vs TMEM from the change in chromatin

Outputs in out/state_quant/:
  gene_quant.tsv.gz   the gene x condition table
  scores.tsv          held-out Spearman/R^2 per feature family and task, with 95% CI
  contrasts.tsv       paired bootstrap differences, quantitative minus each family
  importance.tsv      per-state marginal rho, unique R^2 (95% CI) and direction, per task

The question this answers that step 5 could not: occupancy alone lost to raw signal on
TMEM vs TN. Does adding per-state intensity close that gap without falling back on the
raw marks?
"""

from __future__ import annotations

import pickle
import time

import pandas as pd
from common import CHROMS, CONDITIONS, MODEL_DIR, OUT, ensure_out, load_dataset

from chromstate.features import FeatureSet
from chromstate.model.core import Segmentation
from chromstate.regulation import (
    compare_representations,
    default_families,
    expression_change_task,
    expression_level_task,
    gene_quantification,
    state_importance,
    state_mark_weights,
)

PARTS = {"promoter": dict(upstream=1000, downstream=1000),
         "body": dict(skip_promoter=1000, min_length=1000)}
N_CHROMHMM = 15
MIN_BASEMEAN = 10.0
N_BOOT = 1000
SEED = 0


def build_table() -> pd.DataFrame:
    with open(OUT / MODEL_DIR / "model.pkl", "rb") as fh:
        hmm = pickle.load(fh)["hmm"]
    labels = pd.read_csv(OUT / MODEL_DIR / "state_labels.tsv", sep="\t")["label"].tolist()
    occ = pd.read_csv(OUT / MODEL_DIR / "occupancy_genome.tsv", sep="\t", index_col=0)
    genome_occ = occ.loc["Naive", labels].to_numpy(dtype=float)
    genes = pd.read_csv(OUT / "rna" / "genes.tsv", sep="\t")

    frames, weights = [], None
    for chrom in CHROMS:
        t0 = time.time()
        here = genes.loc[genes["chrom"] == chrom].reset_index(drop=True)
        if here.empty:
            continue
        ds = load_dataset([chrom])
        post, states = hmm.posterior_and_decode(ds.X, ds.size_factors, ds.chains)
        seg = Segmentation(posterior=post, states=states, labels=labels,
                           colors=["#DDDDDD"] * len(labels), rules=[""] * len(labels),
                           model=hmm, dataset=ds, loglik=float("nan"))
        if weights is None:  # genome-wide weights, identical for every chromosome
            weights = state_mark_weights(seg, occupancy=genome_occ)
        gq = gene_quantification(seg, FeatureSet(here, source="featureCounts"),
                                 parts=list(PARTS), part_kwargs=PARTS, mark_weights=weights)
        frames.append(gq)
        print(f"{chrom:6s} {len(here):6,} genes  {time.time() - t0:5.1f}s", flush=True)
    table = pd.concat(frames, ignore_index=True)

    # ChromHMM comparators from step 4, on the same genes and conditions
    s4 = pd.read_csv(OUT / "genes" / "gene_features.tsv.gz", sep="\t")
    chmm_cols = [c for c in s4.columns if c.startswith(("P_chmm_", "B_chmm_"))]
    s4 = s4[["gene_id", "condition", "chmm_tss_state", "P_raw_H3K4me3"] + chmm_cols]
    tss = s4["chmm_tss_state"].fillna(-1).astype(int).to_numpy()
    for k in range(N_CHROMHMM):
        s4[f"chmmtss_E{k + 1}"] = (tss == k).astype(float)
    s4 = s4.drop(columns="chmm_tss_state").rename(columns={"condition": "sample"})
    table = table.merge(s4, on=["gene_id", "sample"], how="left")
    return table


def families(columns) -> dict:
    fam = default_families(columns)
    occ_v1 = [c for c in columns if "_occ_" in c or c.endswith("_entropy") or "_biv_" in c]
    fam["occupancy+entropy+biv (step 5)"] = occ_v1
    fam["chromhmm_fraction"] = [c for c in columns if c.startswith(("P_chmm_", "B_chmm_"))]
    fam["chromhmm_tss"] = [c for c in columns if c.startswith("chmmtss_")]
    return fam


def main() -> None:
    res = ensure_out("state_quant")
    path = res / "gene_quant.tsv.gz"
    if path.exists():
        table = pd.read_csv(path, sep="\t")
    else:
        table = build_table()
        table.to_csv(path, sep="\t", index=False)

    # Genes scored in every condition (the step 5 gene set), so families compare like for like
    ok = table.dropna(subset=["P_raw_H3K4me3"]).groupby("gene_id")["sample"].nunique()
    shared = ok.index[ok == len(CONDITIONS)]
    table = table.loc[table["gene_id"].isin(shared)].drop(columns="P_raw_H3K4me3")
    labels = pd.read_csv(OUT / MODEL_DIR / "state_labels.tsv", sep="\t")["label"].tolist()

    expr = pd.read_csv(OUT / "rna" / "expression.tsv", sep="\t", index_col="gene_id")
    tasks = [expression_level_task(table, expr, ["Naive"])]
    for b, a in [("Arm", "Naive"), ("Cl13", "Naive"), ("Cl13", "Arm")]:
        de = pd.read_csv(OUT / "rna" / f"de_{b}_vs_{a}.tsv", sep="\t", index_col="gene_id")
        keep = de.index[(de["baseMean"] >= MIN_BASEMEAN) & de["log2FoldChange"].notna()]
        tasks.append(expression_change_task(table, de.loc[keep, "log2FoldChange"], [a], [b]))

    scores, contrasts, importance = [], [], []
    for task in tasks:
        t0 = time.time()
        fam = families(task.X.columns)
        rep = compare_representations(task, fam, reference="quantitative", n_boot=N_BOOT,
                                      seed=SEED)
        scores.append(rep.scores)
        contrasts.append(rep.contrasts.assign(task=task.description))
        imp = state_importance(task, states=labels, n_boot=N_BOOT, seed=SEED)
        importance.append(imp.table.assign(task=task.description, full_r2=imp.full_r2,
                                           full_spearman=imp.full_spearman))
        print(f"\n== {task.description} ({len(task.y):,} genes, {time.time() - t0:.0f}s)")
        print(rep.scores[["family", "n_features", "spearman", "rho_lo", "rho_hi", "r2"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(rep.contrasts.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(imp.table[["rank", "state", "marginal_rho", "unique_r2", "unique_lo", "unique_hi",
                         "direction"]].head(8)
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"), flush=True)

    pd.concat(scores).to_csv(res / "scores.tsv", sep="\t", index=False)
    pd.concat(contrasts).to_csv(res / "contrasts.tsv", sep="\t", index=False)
    pd.concat(importance).to_csv(res / "importance.tsv", sep="\t", index=False)
    print(f"\nwrote {res}")


if __name__ == "__main__":
    main()
