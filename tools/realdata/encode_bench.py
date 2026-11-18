"""Real-data test on the ENCODE Broad histone benchmark (GM12878, K562, H1; GRCh38).

    python tools/realdata/encode_bench.py --binned <bin_bams merge out> --annotation <dir> \
        --chroms chr19 -k 18 -o <out> [--control-offset]

These are the cell lines and the Broad/Bernstein ChIP-seq used by the ChromHMM (Ernst
2011), diHMM, EpiCSeg and normR papers. Six marks (H3K4me1, H3K4me3, H3K27ac, H3K27me3,
H3K36me3, H3K9me3), two biological replicates per cell line, each replicate its own
sample. Evaluations, none of which the model sees:

1. agreement with the Roadmap 18-state ChromHMM reference for the same cell line (E116,
   E123, E003; same six marks), next to replicate-vs-replicate agreement as a ceiling;
2. expression level per cell line from promoter states (ENCODE RNA-seq, TPM);
3. expression change between cell lines (K562 vs GM12878, H1 vs GM12878);
4. K562 CRISPRi enhancer-gene benchmark (ENCODE-rE2G): do regulating elements carry
   more enhancer-state mass than non-regulating ones? Raw H3K27ac is the baseline;
5. K562 STARR-seq peaks: state enrichment;
6. replicate noise floor, temperature calibration, and the K562 vs GM12878 differential.

``--control-offset`` multiplies each bin's size factor by the local input-control level
(10 kb mean, relative to the sample's genome mean, clipped to [0.25, 4]) so copy-number
and accessibility bias in the input enters the NB model as an offset.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tools", "realdata"))

import chromstate as cs  # noqa: E402
from chromstate import viz  # noqa: E402
from chromstate.dataset import ChromStateDataset  # noqa: E402
from toolbench.realdata import read_gtf_genes  # noqa: E402
from replicates_cd8 import _keep_mask, n_calls, save_figs  # noqa: E402

T0 = time.time()
ROADMAP = {"GM12878": "E116", "K562": "E123", "H1": "E003"}
MANIFEST_RNA = {"GM12878": ["ENCFF910XWA", "ENCFF413MYB"], "K562": ["ENCFF136UBW", "ENCFF171FQU"],
                "H1": ["ENCFF321HCT", "ENCFF736CCO"]}


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def _auc(y: np.ndarray, s: np.ndarray):
    from sklearn.metrics import average_precision_score, roc_auc_score
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.sum() == 0 or y.sum() == len(y):
        return None, None, int(ok.sum())
    return float(average_precision_score(y, s)), float(roc_auc_score(y, s)), int(ok.sum())


# ---------------------------------------------------------------------------------------
def load_dataset(binned, chroms, min_gap_bins, control_offset):
    Z = {c: np.load(os.path.join(binned, f"{c}.npz")) for c in chroms}
    first = Z[chroms[0]]
    samples, marks, bs = [str(x) for x in first["samples"]], [str(x) for x in first["marks"]], int(first["bin_size"])
    arrays, carrs, starts, runs, dropped, total = {}, {}, {}, {}, 0, 0
    for c in chroms:
        Xc, Cc = Z[c]["X"], Z[c]["C"]
        keep = _keep_mask(Xc, min_gap_bins)
        idx = np.flatnonzero(keep)
        arrays[c], carrs[c], starts[c] = Xc[:, idx, :], Cc[:, idx], idx * bs
        brk = np.flatnonzero(np.diff(idx) > 1) + 1
        bounds = np.concatenate([[0], brk, [len(idx)]])
        runs[c] = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
        dropped += int((~keep).sum()); total += len(keep)
    X = np.concatenate([np.concatenate([arrays[c][si] for c in chroms]) for si in range(len(samples))]).astype(np.float64)
    C = np.concatenate([np.concatenate([carrs[c][si] for c in chroms]) for si in range(len(samples))]).astype(np.float64)
    chrom_col = np.concatenate([np.repeat(c, len(starts[c])) for c in chroms])
    start_col = np.concatenate([starts[c] for c in chroms])
    per = len(chrom_col)
    bins = pd.DataFrame({"chrom": pd.Categorical(np.tile(chrom_col, len(samples)), categories=chroms),
                         "start": np.tile(start_col, len(samples)).astype(np.int64),
                         "sample": pd.Categorical(np.repeat(samples, per), categories=samples)})
    bins.insert(2, "end", bins["start"] + bs)
    chains, off = [], 0
    for _ in samples:
        for c in chroms:
            for a, b in runs[c]:
                chains.append(slice(off + a, off + b))
            off += len(starts[c])
    ds = ChromStateDataset(X=X, marks=marks, bins=bins, chains=chains, size_factors=np.ones_like(X),
                           control=np.repeat(C[:, None], len(marks), axis=1))
    log(f"gap mask: dropped {dropped:,}/{total:,} bins in empty runs of >= {min_gap_bins}; "
        f"{len(chains) // len(samples)} chains per sample")
    cs.normalize(ds, method="median_ratio")
    if control_offset:
        lam = np.ones(len(C))
        for ch in chains:  # 10 kb running mean within each chain, per sample genome mean
            v = C[ch]
            k = min(51, len(v))
            lam[ch] = np.convolve(v, np.ones(k) / k, mode="same")
        for s in samples:
            m = ds.sample_mask(s)
            lam[m] = np.clip(lam[m] / max(lam[m].mean(), 1e-9), 0.25, 4.0)
        ds.size_factors = ds.size_factors * lam[:, None]
        log(f"control offset: input level quantiles {np.quantile(lam, [0.01, 0.5, 0.99]).round(2).tolist()}")
    return ds, samples


def cell_of(s: str) -> str:
    return s.rsplit("_R", 1)[0]


# ---------------------------------------------------------------------------------------
def read_bed_labels(path, bins_one_sample):
    """Label of the reference segment covering each bin midpoint ('' outside)."""
    ref = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3], names=["chrom", "start", "end", "label"])
    out = np.full(len(bins_one_sample), "", dtype=object)
    for c, sub in bins_one_sample.groupby("chrom", observed=True):
        r = ref.loc[ref["chrom"] == c].sort_values("start")
        if r.empty:
            continue
        mid = sub["start"].to_numpy() + 100
        j = np.searchsorted(r["start"].to_numpy(), mid, side="right") - 1
        ok = (j >= 0) & (mid < r["end"].to_numpy()[np.clip(j, 0, None)])
        lab = np.where(ok, r["label"].to_numpy()[np.clip(j, 0, None)], "")
        out[sub.index.to_numpy() - bins_one_sample.index[0]] = lab
    return out


def reference_agreement(seg, samples, annot, outdir):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    rows, ds = [], seg.dataset
    for cell, eid in ROADMAP.items():
        reps = [s for s in samples if cell_of(s) == cell]
        path = os.path.join(annot, f"roadmap18_{eid}.bed.gz")
        if not reps or not os.path.exists(path):
            continue
        m0 = ds.sample_mask(reps[0])
        b0 = ds.bins.loc[m0].reset_index(drop=True)
        ref = read_bed_labels(path, b0)
        has = ref != ""
        calls = {s: np.asarray(seg.states)[ds.sample_mask(s)] for s in reps}
        for s in reps:
            rows.append({"cell": cell, "sample": s, "vs": f"Roadmap {eid}",
                         "ARI": adjusted_rand_score(ref[has], calls[s][has]),
                         "NMI": normalized_mutual_info_score(ref[has], calls[s][has]), "n_bins": int(has.sum())})
            tab = pd.crosstab(pd.Series(ref[has], name="roadmap"),
                              pd.Series(np.asarray(seg.labels)[calls[s][has]], name="chromstate"), normalize="index")
            tab.to_csv(os.path.join(outdir, f"confusion_{s}_vs_{eid}.tsv"), sep="\t")
        if len(reps) >= 2:
            a, b = calls[reps[0]], calls[reps[1]]
            rows.append({"cell": cell, "sample": f"{reps[0]} vs {reps[1]}", "vs": "replicate",
                         "ARI": adjusted_rand_score(a, b), "NMI": normalized_mutual_info_score(a, b), "n_bins": len(a)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
def read_rna(rna_dir, samples, gene_ids):
    cols = {}
    for s in samples:
        acc = MANIFEST_RNA[cell_of(s)][int(s.rsplit("_R", 1)[1]) - 1]
        t = pd.read_csv(os.path.join(rna_dir, f"{acc}.tsv"), sep="\t")
        t = t.loc[t["gene_id"].astype(str).str.startswith("ENSG")]
        t["gene_id"] = t["gene_id"].str.split(".").str[0]
        cols[s] = t.drop_duplicates("gene_id").set_index("gene_id")["TPM"]
    expr = pd.DataFrame(cols)
    return np.log2(expr.loc[expr.index.intersection(gene_ids)] + 1.0)


def element_scores(seg, sample_list, regions, per_row):
    """Mean of each per-bin-row score in ``per_row`` (name -> array) over each region's bins,
    averaged over ``sample_list``."""
    ds = seg.dataset
    out = {k: np.zeros(len(regions)) for k in per_row}
    for s in sample_list:
        m = np.flatnonzero(ds.sample_mask(s))
        b = ds.bins.iloc[m]
        acc = {k: np.full(len(regions), np.nan) for k in per_row}
        for c, sub in regions.groupby("chrom"):
            sel = np.flatnonzero((b["chrom"] == c).to_numpy())
            if not len(sel):
                continue
            starts = b["start"].to_numpy()[sel]
            lo = np.searchsorted(starts, sub["start"].to_numpy() - 199)
            hi = np.searchsorted(starts, sub["end"].to_numpy())
            for i, l, h in zip(sub.index, lo, hi):
                if h > l:
                    rows = m[sel[l:h]]
                    for k, v in per_row.items():
                        acc[k][i] = v[rows].mean()
        for k in per_row:
            out[k] += acc[k] / len(sample_list)
    return out


def crispri_eval(seg, samples, annot, outdir):
    path = os.path.join(annot, "crispri_K562.GRCh38.tsv.gz")
    reps = [s for s in samples if cell_of(s) == "K562"]
    if not os.path.exists(path) or not reps:
        return None
    t = pd.read_csv(path, sep="\t")
    cc = {c.lower(): c for c in t.columns}
    t = t.rename(columns={cc.get("chrom"): "chrom", cc.get("chromstart"): "start", cc.get("chromend"): "end",
                          cc.get("regulated"): "regulated"})
    t["regulated"] = t["regulated"].astype(str).str.upper().isin(["TRUE", "1"])
    if "ValidConnection" in t.columns:  # benchmark convention: drop pairs flagged as invalid
        t = t.loc[t["ValidConnection"].astype(str).str.upper().isin(["TRUE", "1"])]
    el = t.groupby(["chrom", "start", "end"], as_index=False)["regulated"].any()
    el = el.loc[el["chrom"].isin(seg.dataset.bins["chrom"].cat.categories)].reset_index(drop=True)
    ds, post = seg.dataset, np.asarray(seg.posterior)
    k27 = list(ds.marks).index("H3K27ac")
    mu = np.asarray(seg.model.emissions.mean_matrix(), dtype=float)
    occ = post.mean(axis=0)
    fold = mu[:, k27] / max(float(occ @ mu[:, k27]), 1e-12)
    enh = [i for i, l in enumerate(seg.labels) if l.startswith("Enh")]
    active = [i for i in range(len(seg.labels)) if fold[i] >= 2.0]
    per_row = {
        "chromstate_enh_label_mass": post[:, enh].sum(axis=1),          # depends on state names
        "chromstate_active_mass": post[:, active].sum(axis=1),          # label-free: K27ac >= 2x genome
        "chromstate_expected_H3K27ac": post @ mu[:, k27],               # label-free: model-smoothed K27ac
        "raw_H3K27ac": ds.X[:, k27] / ds.size_factors[:, k27],
    }
    sc = element_scores(seg, reps, el, per_row)
    for k, v in sc.items():
        el[k] = v
    el.to_csv(os.path.join(outdir, "crispri_elements.tsv"), sep="\t", index=False)
    y = el["regulated"].to_numpy()
    res = {"n_elements": len(el), "n_positive": int(y.sum()), "enhancer_label_states": [seg.labels[i] for i in enh],
           "active_states": [seg.labels[i] for i in active]}
    for name, s in sc.items():
        ap, roc, n = _auc(y, s)
        res[name] = {"AUPRC": ap, "AUROC": roc, "n_scored": n}
    res["baseline_AUPRC"] = float(y.mean()) if len(y) else None
    return res


def starr_enrichment(seg, samples, annot):
    path = os.path.join(annot, "..", "raw", "ENCFF045TVA.bed.gz")
    reps = [s for s in samples if cell_of(s) == "K562"]
    if not os.path.exists(path) or not reps:
        return None
    pk = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"])
    ds, post = seg.dataset, np.asarray(seg.posterior)
    out = []
    for s in reps:
        m = np.flatnonzero(ds.sample_mask(s))
        b = ds.bins.iloc[m].reset_index(drop=True)
        hit = np.zeros(len(b), dtype=bool)
        for c, sub in pk.groupby("chrom"):
            sel = np.flatnonzero((b["chrom"] == c).to_numpy())
            if not len(sel):
                continue
            starts = b["start"].to_numpy()[sel]
            lo = np.searchsorted(starts, sub["start"].to_numpy() - 199)
            hi = np.searchsorted(starts, sub["end"].to_numpy())
            for l, h in zip(lo, hi):
                hit[sel[l:h]] = True
        p = post[m]
        frac_in = p[hit].mean(axis=0) if hit.any() else np.full(p.shape[1], np.nan)
        out.append(pd.Series(frac_in / np.maximum(p.mean(axis=0), 1e-12), index=seg.labels, name=s))
    return pd.concat(out, axis=1)


# ---------------------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--binned", required=True)
    p.add_argument("--annotation", required=True)
    p.add_argument("--rna-dir", required=True, help="folder with the ENCODE gene quantification TSVs")
    p.add_argument("--chroms", default="chr19")
    p.add_argument("-k", "--n-states", type=int, default=18)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--min-gap-bins", type=int, default=50)
    p.add_argument("--control-offset", action="store_true")
    p.add_argument("--promoter", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    chroms = a.chroms.split(",")
    summary = {"chroms": chroms, "n_states": a.n_states, "min_gap_bins": a.min_gap_bins,
               "control_offset": a.control_offset}

    ds, samples = load_dataset(a.binned, chroms, a.min_gap_bins, a.control_offset)
    cells = list(dict.fromkeys(cell_of(s) for s in samples))
    groups = {c: [s for s in samples if cell_of(s) == c] for c in cells}
    log(f"dataset: {ds.n_rows:,} bin-rows, marks {ds.marks}, groups {groups}")

    defs = cs.StateDefinitions.default(ds.marks)
    t = time.time()
    seg = cs.fit(ds, n_states=a.n_states, n_restarts=a.restarts, seed=a.seed, definitions=defs, anchor="seed")
    summary.update(fit_seconds=time.time() - t, loglik=seg.loglik, labels=list(seg.labels))
    log(f"fit: loglik {seg.loglik:,.1f} ({summary['fit_seconds']:.0f}s)")
    seg.emission_frame().to_csv(os.path.join(a.out, "emissions.tsv"), sep="\t")
    seg.state_table().to_csv(os.path.join(a.out, "states.tsv"), sep="\t", index=False)
    seg.sample_state_matrix().to_csv(os.path.join(a.out, "occupancy.tsv"), sep="\t")
    print(seg.emission_frame().round(2).to_string())
    save_figs({"emissions": viz.plot_emission_heatmap(seg), "transitions": viz.plot_transition_heatmap(seg),
               "coverage": viz.plot_state_coverage(seg)}, os.path.join(a.out, "figures"))

    # 1. reference agreement --------------------------------------------------------------
    agree = reference_agreement(seg, samples, a.annotation, a.out)
    agree.to_csv(os.path.join(a.out, "reference_agreement.tsv"), sep="\t", index=False)
    summary["reference_agreement"] = agree.to_dict(orient="records")
    print(agree.round(3).to_string(index=False))

    # 6a. noise floor + calibration ----------------------------------------------------------
    nf = cs.replicate_noise_floor(seg, groups)
    summary["noise_floor_bin"] = float(nf.threshold)
    cal = cs.fit_temperature_from_replicates(seg, groups, seed=a.seed)
    summary["calibration"] = {"temperature": cal.temperature, "before": cal.before, "after": cal.after}
    log(f"noise floor {nf.threshold:.3f}; calibration {cal.note}")

    # 4-5. enhancer truth sets ----------------------------------------------------------------
    try:
        summary["crispri_K562"] = crispri_eval(seg, samples, a.annotation, a.out)
        log(f"CRISPRi: {summary['crispri_K562']}")
        st = starr_enrichment(seg, samples, a.annotation)
        if st is not None:
            st.to_csv(os.path.join(a.out, "starr_state_enrichment.tsv"), sep="\t")
            summary["starr_enrichment"] = st.mean(axis=1).round(3).to_dict()
            print(st.round(2).to_string())
    except Exception as exc:  # truth-set parsing must not cost the rest of the run
        import traceback; traceback.print_exc()
        summary["enhancer_eval_error"] = repr(exc)

    # 2-3. genes and expression --------------------------------------------------------------
    features = read_gtf_genes(os.path.join(a.annotation, "gencode.v29.annotation.gtf.gz"), chroms=chroms)
    gq = cs.gene_quantification(seg, features, part_kwargs={"promoter": {"upstream": a.promoter, "downstream": a.promoter}})
    gq.to_csv(os.path.join(a.out, "gene_quantification.tsv.gz"), sep="\t", index=False)
    expr = read_rna(a.rna_dir, samples, gq["gene_id"].unique())
    expr.to_csv(os.path.join(a.out, "expression_log2tpm.tsv"), sep="\t")
    summary["predict"] = {}
    level = cs.predict_expression(gq, expression=expr, seed=a.seed)
    level.write(os.path.join(a.out, "predict", "level"), seg=seg)
    summary["predict"]["level"] = level.summary
    log(level.note)
    for cond in [c for c in cells if c != "GM12878"]:
        lfc = expr[groups[cond]].mean(axis=1) - expr[groups["GM12878"]].mean(axis=1)
        ch = cs.predict_expression(gq, lfc=lfc, contrast=(groups["GM12878"], groups[cond]), seed=a.seed)
        ch.write(os.path.join(a.out, "predict", f"change_{cond}_vs_GM12878"), seg=seg)
        summary["predict"][f"change_{cond}_vs_GM12878"] = ch.summary
        log(ch.note)

    # 6b. differential K562 vs GM12878 (2 vs 2) --------------------------------------------
    rep = cs.differential_report(seg, {"GM12878": groups["GM12878"], "K562": groups["K562"]},
                                 reference="GM12878", seed=a.seed, genome_tests=True)
    try:
        rep.write(os.path.join(a.out, "differential", "tiles_K562_vs_GM12878"), seg=seg, png=True)
    except Exception as exc:
        log(f"differential figures failed ({exc!r}); tables only")
        rep.write(os.path.join(a.out, "differential", "tiles_K562_vs_GM12878"), figures=False)
    summary["differential"] = {c: n_calls(rep, c) for c in rep.regions}
    log(f"differential: {summary['differential']}")

    summary["seconds"] = time.time() - T0
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log(f"done -> {a.out}")


if __name__ == "__main__":
    main()
