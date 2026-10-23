"""Step 4: decode every chromosome with the fitted model and summarise each gene.

For every gene x condition, over the promoter (TSS +/- 1 kb) and the gene body (TSS+1 kb
to TES), this records:

* ChromState quantitative features: mean posterior composition over the K states,
  state entropy, posterior-level bivalency (within-state and between-state), and the
  signal-level bivalency score with its magnitude/balance decomposition;
* ChromState hard call: the Viterbi state of the TSS bin;
* raw signal: mean log1p coverage per mark;
* the published ChromHMM run (15 states, binarised peaks): state at the TSS and the
  fraction of the promoter/body in each state.

Every feature family comes from the same bins and the same genes, so the prediction
step compares representations, not preprocessing.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from common import (
    BIN_SIZE,
    CHROMHMM_DIR,
    CHROMHMM_PREFIX,
    CHROMS,
    CONDITIONS,
    MARKS,
    MODEL_DIR,
    OUT,
    ensure_out,
    load_dataset,
)
from scipy import sparse

from chromstate.features import FeatureSet, map_bins_to_regions
from chromstate.model.annotate import annotate_states
from chromstate.model.core import Segmentation
from chromstate.quantify import bivalency, posterior_bivalency, state_entropy

PROMOTER = dict(upstream=1000, downstream=1000)
BODY = dict(skip_promoter=1000, min_length=1000)
N_CHROMHMM = 15


def averaging_matrix(n_bins: int, regions: pd.DataFrame, bins: pd.DataFrame):
    """Sparse (genes x bins) matrix whose rows average a gene's bins."""
    pairs = map_bins_to_regions(bins, regions)
    genes = pd.Index(pairs["gene_id"].unique())
    rows = genes.get_indexer(pairs["gene_id"])
    counts = np.bincount(rows, minlength=len(genes))
    vals = 1.0 / counts[rows]
    G = sparse.csr_matrix((vals, (rows, pairs["bin_row"].to_numpy())), shape=(len(genes), n_bins))
    return G, genes


def chromhmm_bins(cond: str, chrom: str, n_bins: int) -> np.ndarray:
    """ChromHMM state index (0-based) for each 200 bp bin; -1 where not covered."""
    path = CHROMHMM_DIR / f"{CHROMHMM_PREFIX[cond]}_{N_CHROMHMM}_segments.bed"
    seg = pd.read_csv(path, sep="\t", header=None, names=["chrom", "start", "end", "state"])
    seg = seg.loc[seg["chrom"] == chrom]
    out = np.full(n_bins, -1, dtype=np.int16)
    for s, e, st in zip(seg["start"].to_numpy(), seg["end"].to_numpy(), seg["state"].to_numpy()):
        lo, hi = s // BIN_SIZE, min(n_bins, -(-e // BIN_SIZE))
        out[lo:hi] = int(str(st).lstrip("E")) - 1
    return out


def one_hot(idx: np.ndarray, k: int) -> np.ndarray:
    m = np.zeros((len(idx), k), dtype=np.float32)
    ok = idx >= 0
    m[np.flatnonzero(ok), idx[ok]] = 1.0
    return m


def main() -> None:
    gdir = ensure_out("genes")
    sdir = ensure_out("segmentation")
    with open(OUT / MODEL_DIR / "model.pkl", "rb") as fh:
        model = pickle.load(fh)
    hmm = model["hmm"]
    # States are carried as S0..S{K-1} while decoding and named at the end, from their
    # genome-wide occupancy -- the training chromosomes are gene-rich, so naming from
    # them would inflate the background and under-call weak states.
    K = hmm.n_states
    labels = [f"S{k}" for k in range(K)]
    colors = ["#DDDDDD"] * K
    rules = [""] * K
    features = FeatureSet(pd.read_csv(OUT / "rna" / "genes.tsv", sep="\t"), source="featureCounts")

    frames, occupancy, beds = [], [], {c: [] for c in CONDITIONS}
    for chrom in CHROMS:
        ds = load_dataset([chrom])
        post = hmm.posterior(ds.X, ds.size_factors, ds.chains)
        states = hmm.decode(ds.X, ds.size_factors, ds.chains)
        seg = Segmentation(posterior=post, states=states, labels=labels,
                           colors=colors, rules=rules, model=hmm,
                           dataset=ds, loglik=float("nan"))

        per_bin = np.column_stack([
            post,
            state_entropy(seg),
            posterior_bivalency(seg)[["biv_within_state", "biv_between_state"]].to_numpy(),
            bivalency(seg)[["bivalency_score", "bivalency_magnitude", "bivalency_balance"]].to_numpy(),
            np.log1p(ds.X),
        ]).astype(np.float32)
        names = ([f"cs_{lbl}" for lbl in labels] + ["cs_entropy", "cs_biv_within", "cs_biv_between",
                 "cs_biv_score", "cs_biv_magnitude", "cs_biv_balance"]
                 + [f"raw_{m}" for m in MARKS])

        n = len(ds.X) // len(CONDITIONS)
        bins = ds.bins.iloc[:n][["chrom", "start", "end"]].reset_index(drop=True)
        genes_here = features.table.loc[features.table["chrom"] == chrom]
        if genes_here.empty:
            continue
        fs = FeatureSet(genes_here.reset_index(drop=True), source="chrom")
        G_p, genes_p = averaging_matrix(n, fs.promoters(**PROMOTER), bins)
        G_b, genes_b = averaging_matrix(n, fs.bodies(**BODY), bins)
        tss = fs.tss().to_numpy()
        tss_bin = np.clip(tss // BIN_SIZE, 0, n - 1)

        for ci, cond in enumerate(CONDITIONS):
            rows = slice(ci * n, (ci + 1) * n)
            F = per_bin[rows]
            chmm = chromhmm_bins(cond, chrom, n)
            chmm_oh = one_hot(chmm, N_CHROMHMM)
            chmm_names = [f"chmm_E{k + 1}" for k in range(N_CHROMHMM)]

            prom = pd.DataFrame(np.hstack([G_p @ F, G_p @ chmm_oh]), index=genes_p,
                                columns=[f"P_{c}" for c in names + chmm_names])
            body = pd.DataFrame(np.hstack([G_b @ F, G_b @ chmm_oh]), index=genes_b,
                                columns=[f"B_{c}" for c in names + chmm_names])
            tss_df = pd.DataFrame({
                "cs_tss_state": states[rows][tss_bin],
                "chmm_tss_state": chmm[tss_bin],
            }, index=fs.table["gene_id"])
            df = tss_df.join(prom, how="left").join(body, how="left")
            df.insert(0, "condition", cond)
            df.insert(1, "chrom", chrom)
            frames.append(df.reset_index().rename(columns={"index": "gene_id"}))

            occupancy.append(pd.Series(post[rows].sum(axis=0), index=labels, name=(cond, chrom)))
            beds[cond].append(seg.to_bed(sample=cond))
        print(f"{chrom:6s} bins {n:,}  genes {len(fs):,}", flush=True)

    # name from TN occupancy, the same reference the figure labels use
    occ_all = pd.DataFrame([o for o in occupancy if o.name[0] == "Naive"]).sum(axis=0).to_numpy()
    names, colors, rules = annotate_states(hmm.emissions.mean_matrix(), MARKS, occupancy=occ_all)
    rename = dict(zip(labels, names))
    table = pd.concat(frames, ignore_index=True)
    table = table.rename(columns=lambda c: next(
        (c.replace(f"_cs_{k}", f"_cs_{v}") for k, v in rename.items() if c.endswith(f"_cs_{k}")), c))
    labels = names
    table.to_csv(gdir / "gene_features.tsv.gz", sep="\t", index=False)
    occ = pd.DataFrame(occupancy).rename(columns=rename)
    occ.index = pd.MultiIndex.from_tuples(occ.index, names=["condition", "chrom"])
    occ = occ.groupby("condition").sum()
    (occ.div(occ.sum(axis=1), axis=0)).to_csv(OUT / MODEL_DIR / "occupancy_genome.tsv", sep="\t")
    for cond, parts in beds.items():
        bed = pd.concat(parts)
        bed["name"] = bed["name"].map(rename)
        rgb = {n: ",".join(str(int(c.lstrip("#")[i:i + 2], 16)) for i in (0, 2, 4))
               for n, c in zip(names, colors)}
        bed["itemRgb"] = bed["name"].map(rgb)
        bed.to_csv(sdir / f"{cond}_chromstate_K{K}_dense.bed", sep="\t",
                                header=False, index=False)
    pd.DataFrame({"label": labels, "color": colors, "rule": rules}).to_csv(
        OUT / MODEL_DIR / "state_labels.tsv", sep="\t", index=False)
    print(f"wrote {len(table):,} gene x condition rows")


if __name__ == "__main__":
    main()
