"""Step 14: the enhancer layer, genome-wide -- ATAC elements, their activity, and their genes.

Every ATAC union peak becomes an element, flagged promoter (within 2 kb of a TSS) or
distal. Per element and condition: ChromState composition over the element's bins
(collapsed into active / poised / repressed / background state classes), mean mark
signal, ATAC, and ABC activity sqrt(ATAC x H3K27ac). Elements are linked to every TSS
within 500 kb with a distance power-law contact, and scored by Activity-by-Contact.

Outputs (out/enhancers/):
  state_classes.tsv            each state's class and fold enrichment per mark
  elements.tsv.gz              candidate elements with promoter/distal flag
  activity.tsv.gz              element x condition activity and state classes
  abc_scores.tsv.gz            element-gene pairs with ABC per condition
  gene_enhancer_features.tsv.gz  per gene x condition distal input in distance rings
  association_by_distance.tsv  enhancer input change vs RNA change, per distance ring

State classes are fixed from genome-wide occupancy so every chromosome is classified
alike; decoding is per chromosome, as in step 4.
"""

from __future__ import annotations

import pickle

import pandas as pd
from common import ATAC_PEAKS, CHROMS, MODEL_DIR, OUT, ensure_out, load_dataset

import chromstate as cs
from chromstate.model.core import Segmentation

MIN_BASEMEAN = 10.0


def main() -> None:
    res = ensure_out("enhancers")
    with open(OUT / MODEL_DIR / "model.pkl", "rb") as fh:
        hmm = pickle.load(fh)["hmm"]
    labels = pd.read_csv(OUT / MODEL_DIR / "state_labels.tsv", sep="\t")["label"].tolist()
    occ = pd.read_csv(OUT / MODEL_DIR / "occupancy_genome.tsv", sep="\t",
                      index_col=0)[labels].mean(axis=0).to_numpy()
    features = cs.FeatureSet(pd.read_csv(OUT / "rna" / "genes.tsv", sep="\t"), source="featureCounts")

    peaks = cs.read_peaks(str(ATAC_PEAKS))
    peaks = peaks.loc[peaks["chrom"].isin(CHROMS)]
    atac = (peaks.set_index("enhancer_id")[["Mean_Naive", "Mean_Arm", "Mean_Cl13"]]
                 .rename(columns=lambda c: c.replace("Mean_", "")))
    elements = cs.candidate_enhancers(peaks[["enhancer_id", "chrom", "start", "end"]], features)
    print(f"{len(elements):,} ATAC elements, {(~elements['is_promoter']).sum():,} distal", flush=True)

    classes, acts = None, []
    for chrom in CHROMS:
        ds = load_dataset([chrom])
        post = hmm.posterior(ds.X, ds.size_factors, ds.chains)
        seg = Segmentation(posterior=post, states=post.argmax(axis=1), labels=labels,
                           colors=["#DDDDDD"] * len(labels), rules=[""] * len(labels),
                           model=hmm, dataset=ds, loglik=float("nan"))
        if classes is None:
            classes = cs.enhancer_state_classes(seg, occupancy=occ)
        sub = elements.loc[elements["chrom"] == chrom]
        if len(sub):
            acts.append(cs.enhancer_activity(seg, sub, atac=atac, state_classes=classes,
                                             include_posterior=False))
        print(f"{chrom:6s} {len(sub):,} elements", flush=True)
    activity = pd.concat(acts, ignore_index=True)

    pairs = cs.link_enhancers(elements, features)
    scores = cs.abc_score(pairs, activity)
    feats = cs.gene_enhancer_features(scores)
    print(f"{len(pairs):,} element-gene pairs over {pairs['gene_id'].nunique():,} genes", flush=True)

    assoc = []
    for cond in ("Arm", "Cl13"):
        de = pd.read_csv(OUT / "rna" / f"de_{cond}_vs_Naive.tsv", sep="\t")
        de = de.loc[de["baseMean"] >= MIN_BASEMEAN].rename(columns={"log2FoldChange": "log2fc"})
        change = cs.abc_change(scores, ["Naive"], [cond])
        linked = cs.link_enhancer_expression(change, de.dropna(subset=["log2fc"]))
        r = cs.enhancer_expression_association(linked, n_permutations=1000)
        assoc.append(r.table.assign(contrast=f"{cond}_vs_Naive"))
        print(r.table.round(3).to_string(index=False), flush=True)

    classes.to_csv(res / "state_classes.tsv", sep="\t", index=False)
    elements.to_csv(res / "elements.tsv.gz", sep="\t", index=False)
    activity.to_csv(res / "activity.tsv.gz", sep="\t", index=False)
    scores.to_csv(res / "abc_scores.tsv.gz", sep="\t", index=False)
    feats.to_csv(res / "gene_enhancer_features.tsv.gz", sep="\t", index=False)
    pd.concat(assoc, ignore_index=True).to_csv(res / "association_by_distance.tsv", sep="\t", index=False)
    print(f"-> {res}")


if __name__ == "__main__":
    main()
