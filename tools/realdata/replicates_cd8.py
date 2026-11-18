"""Replicate-level real-data test on the CD8 T-cell CUT&RUN (GSE285245 per-replicate bigWigs).

    python tools/realdata/replicates_cd8.py --binned <bin_bigwigs out> --chroms chr19 -k 15 -o <out>

Unlike the pooled-track runs, every replicate is its own sample, so the
differential layer can be tested on real data:

* one joint model over 16 samples (Naive d30 x4, Naive d32 x4, Arm d30 x4, Cl13 d32 x4);
* bin- and region-level replicate noise floors, and a temperature fitted against
  replicate agreement;
* ``differential_report`` for Naive vs Arm and Naive vs Cl13 (8 vs 4) on 2 kb tiles and
  on promoters, plus Arm vs Cl13, each with the moderated t and F tests;
* a **null contrast**, Naive d30 vs Naive d32 (4 vs 4, same cell state, different
  harvest day). It shows how many calls batch alone produces;
* region and gene (promoter) alluvials, marker-gene promoter tables;
* expression level and change prediction, replicates kept as samples.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
RNA = os.path.join(ROOT, "analysis", "scireports2025", "out", "rna")
GTF = "/vast/projects/wherry/foundation-models-immuno/hhua/tools/chromstate-bench/annotation/gencode.vM25.annotation.gtf.gz"
MARKERS = ["Sell", "Ccr7", "Tcf7", "Lef1", "Il7r", "Id3", "Pdcd1", "Tigit", "Havcr2", "Tox", "Lag3", "Ctla4",
           "Cd244a", "Entpd1", "Cd274", "Fas", "Cd5", "Cd6", "Gzmb", "Klrg1", "Cx3cr1", "Prdm1", "Batf", "Id2"]

import chromstate as cs  # noqa: E402
from chromstate import viz  # noqa: E402
from chromstate.dataset import ChromStateDataset  # noqa: E402
from toolbench.realdata import read_gtf_genes  # noqa: E402

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def _keep_mask(Xc: np.ndarray, min_gap_bins: int) -> np.ndarray:
    """False on runs of >= ``min_gap_bins`` bins with zero signal in every sample and mark
    (assembly gaps, unmappable blocks). Shorter empty stretches are real quiescent chromatin."""
    empty = ~(Xc > 0).any(axis=(0, 2))
    keep = np.ones(len(empty), dtype=bool)
    if min_gap_bins <= 0 or not empty.any():
        return keep
    d = np.diff(np.concatenate([[0], empty.astype(np.int8), [0]]))
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if b - a >= min_gap_bins:
            keep[a:b] = False
    return keep


def load_dataset(binned: str, chroms, min_gap_bins: int = 0):
    """Stack per-replicate binned arrays into one dataset; with ``min_gap_bins`` > 0, drop
    long all-empty runs and split chains there so the HMM never walks across a gap."""
    Z = {c: np.load(os.path.join(binned, f"{c}.npz")) for c in chroms}
    first = Z[chroms[0]]
    samples, marks, bs = [str(x) for x in first["samples"]], [str(x) for x in first["marks"]], int(first["bin_size"])
    arrays, starts, runs, dropped, total = {}, {}, {}, 0, 0
    for c in chroms:
        Xc = Z[c]["X"]
        keep = _keep_mask(Xc, min_gap_bins)
        idx = np.flatnonzero(keep)
        arrays[c] = Xc[:, idx, :]
        starts[c] = idx * bs
        brk = np.flatnonzero(np.diff(idx) > 1) + 1
        bounds = np.concatenate([[0], brk, [len(idx)]])
        runs[c] = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
        dropped += int((~keep).sum()); total += len(keep)
    X = np.concatenate([np.concatenate([arrays[c][si] for c in chroms]) for si in range(len(samples))])
    X = np.rint(X).astype(np.float64)
    chrom_col = np.concatenate([np.repeat(c, len(starts[c])) for c in chroms])
    start_col = np.concatenate([starts[c] for c in chroms])
    per = len(chrom_col)
    bins = pd.DataFrame({
        "chrom": pd.Categorical(np.tile(chrom_col, len(samples)), categories=chroms),
        "start": np.tile(start_col, len(samples)).astype(np.int64),
        "sample": pd.Categorical(np.repeat(samples, per), categories=samples),
    })
    bins.insert(2, "end", bins["start"] + bs)
    chains, off = [], 0
    for _ in samples:
        for c in chroms:
            for a, b in runs[c]:
                chains.append(slice(off + a, off + b))
            off += len(starts[c])
    ds = ChromStateDataset(X=X, marks=marks, bins=bins, chains=chains, size_factors=np.ones_like(X))
    if min_gap_bins > 0:
        print(f"gap mask: dropped {dropped:,}/{total:,} bins ({dropped / max(total, 1):.1%}) in empty runs of "
              f">= {min_gap_bins} bins; {len(chains) // len(samples)} chains per sample", flush=True)
    return ds, samples


def groups_from(samples):
    g = {"Naive": [s for s in samples if s.startswith("Naive_")],
         "Arm": [s for s in samples if s.startswith("Arm_")],
         "Cl13": [s for s in samples if s.startswith("Cl13_")]}
    null = {"Naive_d30": [s for s in samples if s.startswith("Naive_d30")],
            "Naive_d32": [s for s in samples if s.startswith("Naive_d32")]}
    return g, null


def n_calls(rep, contrast):
    tab = rep.regions[contrast]
    col = "confident_change" if "confident_change" in tab.columns else "significant"
    sig = int(tab[col].sum()) if col in tab.columns else 0
    raw = int((tab["qvalue"] < 0.05).sum()) if "qvalue" in tab.columns else None
    return {"n_regions": len(tab), "n_confident": sig, "n_q05": raw, "frac_confident": sig / max(len(tab), 1)}


def save_figs(figs, outdir):
    os.makedirs(outdir, exist_ok=True)
    for name, fig in figs.items():
        fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=200, bbox_inches="tight", facecolor=viz.SURFACE)
    viz.save_all(figs, outdir)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--binned", required=True)
    p.add_argument("--chroms", default="chr19")
    p.add_argument("-k", "--n-states", type=int, default=15)
    p.add_argument("--definitions", default="default", help="'default', 'none' or a definitions file")
    p.add_argument("--anchor", default="seed", choices=["seed", "fixed"])
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--promoter", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-predict", action="store_true")
    p.add_argument("--min-gap-bins", type=int, default=0,
                   help="drop runs of this many all-empty bins (assembly gaps); 0 keeps every bin")
    p.add_argument("-o", "--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    chroms = a.chroms.split(",")
    summary = {"chroms": chroms, "n_states": a.n_states, "definitions": a.definitions, "anchor": a.anchor}

    ds, samples = load_dataset(a.binned, chroms, min_gap_bins=a.min_gap_bins)
    summary["min_gap_bins"] = a.min_gap_bins
    cs.normalize(ds, method="median_ratio")
    groups, null_groups = groups_from(samples)
    log(f"dataset: {ds.n_rows:,} bin-rows, marks {ds.marks}, {len(samples)} samples {groups}")
    sf = pd.DataFrame(ds.size_factors[[int(np.argmax(ds.sample_mask(smp))) for smp in samples]], index=samples, columns=ds.marks)
    sf.to_csv(os.path.join(a.out, "size_factors.tsv"), sep="\t")
    print(sf.round(3).to_string())

    defs = None
    if a.definitions != "none":
        defs = cs.StateDefinitions.default(ds.marks) if a.definitions == "default" \
            else cs.StateDefinitions.from_file(a.definitions, ds.marks)
    t = time.time()
    seg = cs.fit(ds, n_states=a.n_states, n_restarts=a.restarts, seed=a.seed, definitions=defs, anchor=a.anchor)
    summary["fit_seconds"] = time.time() - t
    summary["loglik"] = seg.loglik
    summary["labels"] = list(seg.labels)
    log(f"fit: loglik {seg.loglik:,.1f}, restarts hit best {seg.meta.get('restart_hits_at_best')} ({summary['fit_seconds']:.0f}s)")
    seg.emission_frame().to_csv(os.path.join(a.out, "emissions.tsv"), sep="\t")
    seg.state_table().to_csv(os.path.join(a.out, "states.tsv"), sep="\t", index=False)
    occ = seg.sample_state_matrix()
    occ.to_csv(os.path.join(a.out, "occupancy.tsv"), sep="\t")
    print(seg.state_table()[["label", "rule", "coverage_soft"]].to_string(index=False))
    print(occ.round(3).to_string())
    base_figs = {"emissions": viz.plot_emission_heatmap(seg), "transitions": viz.plot_transition_heatmap(seg),
                 "coverage": viz.plot_state_coverage(seg)}
    save_figs(base_figs, os.path.join(a.out, "figures"))

    # ---- replicate noise floor and calibration -------------------------------------------
    rep_groups = {**{k: v for k, v in groups.items() if k != "Naive"}, **null_groups}
    nf = cs.replicate_noise_floor(seg, rep_groups)
    log(nf.note)
    nf.per_condition.to_csv(os.path.join(a.out, "noise_floor_bin.tsv"), sep="\t", index=False)
    summary["noise_floor_bin"] = float(nf.threshold)
    cal = cs.fit_temperature_from_replicates(seg, rep_groups, seed=a.seed)
    log(f"calibration: {cal.note}")
    summary["calibration"] = {"temperature": cal.temperature, "n_used": cal.n_used, "before": cal.before, "after": cal.after}

    # ---- differential reports ------------------------------------------------------------
    features = read_gtf_genes(GTF, chroms=chroms)
    prom = features.promoters(a.promoter, a.promoter)
    prom = prom.assign(name=prom["name"].astype(str))[["chrom", "start", "end", "name", "gene_id"]]
    prom = prom.drop_duplicates("name").reset_index(drop=True)
    log(f"{len(features):,} protein-coding genes, {len(prom):,} promoters")

    specs = [
        ("tiles_vs_naive", groups, "Naive", None),
        ("tiles_arm_vs_cl13", {"Arm": groups["Arm"], "Cl13": groups["Cl13"]}, "Arm", None),
        ("tiles_null_d30_vs_d32", null_groups, "Naive_d30", None),
        ("promoters_vs_naive", groups, "Naive", prom[["chrom", "start", "end", "name"]]),
        ("promoters_null_d30_vs_d32", null_groups, "Naive_d30", prom[["chrom", "start", "end", "name"]]),
    ]
    summary["differential"] = {}
    reports = {}
    for name, g, ref, regions in specs:
        t = time.time()
        rep = cs.differential_report(seg, g, regions=regions, reference=ref, seed=a.seed,
                                     genome_tests=regions is None)
        try:
            rep.write(os.path.join(a.out, "differential", name), seg=seg, png=True)
        except Exception as exc:  # keep the tables when a figure fails; the error is logged
            import traceback
            traceback.print_exc()
            log(f"{name}: figures failed ({exc!r}); writing tables only")
            rep.write(os.path.join(a.out, "differential", name), figures=False)
        reports[name] = rep
        res = {c: n_calls(rep, c) for c in rep.regions}
        res["noise_floor_region"] = None if rep.noise_floor is None else float(rep.noise_floor.threshold)
        if rep.anova is not None and "qvalue" in rep.anova.columns:
            res["anova_q05"] = int((rep.anova["qvalue"] < 0.05).sum())
        if rep.genome_state_tests is not None:
            res["genome_state_tests"] = rep.genome_state_tests.to_dict(orient="records")
        res["seconds"] = time.time() - t
        summary["differential"][name] = res
        log(f"{name}: " + "; ".join(f"{c} {v['n_confident']:,}/{v['n_regions']:,} confident"
                                     for c, v in res.items() if isinstance(v, dict) and "n_confident" in v))

    # marker promoters
    rows = []
    for contrast, tab in reports["promoters_vs_naive"].regions.items():
        sub = tab.loc[tab["name"].isin(MARKERS)].copy()
        sub.insert(0, "contrast", contrast)
        rows.append(sub)
    markers = pd.concat(rows, ignore_index=True)
    markers.to_csv(os.path.join(a.out, "marker_promoters.tsv"), sep="\t", index=False)
    print(markers.to_string(index=False, max_colwidth=24))
    rs = reports["promoters_vs_naive"].region_states
    rs.loc[rs["name"].isin(MARKERS) if "name" in rs.columns else rs.index[:0]].to_csv(
        os.path.join(a.out, "marker_promoter_states.tsv"), sep="\t", index=False)

    # ---- alluvials + gene quantification -------------------------------------------------
    gq = cs.gene_quantification(seg, features, part_kwargs={"promoter": {"upstream": a.promoter, "downstream": a.promoter}})
    gq.to_csv(os.path.join(a.out, "gene_quantification.tsv.gz"), sep="\t", index=False)
    dom = cs.gene_dominant_states(gq, groups, part="P")
    dom.to_csv(os.path.join(a.out, "promoter_dominant_states.tsv"), sep="\t", index=False)
    dom_null = cs.gene_dominant_states(gq, null_groups, part="P")
    summary["promoter_dominant_changes"] = {
        "n_genes": len(dom), "changed_any": int((dom["n_changes"] > 0).sum()),
        "null_d30_d32_changed": int((dom_null["n_changes"] > 0).sum()),
        "top_patterns": dom.loc[dom["n_changes"] > 0, "pattern"].value_counts().head(15).to_dict(),
    }
    print(dom.loc[dom["name"].isin(MARKERS)].to_string(index=False))
    figs = {"region_alluvial": viz.plot_region_alluvial(seg, groups),
            "gene_alluvial": viz.plot_gene_alluvial(dom, seg=seg),
            "gene_alluvial_null_d30_d32": viz.plot_gene_alluvial(dom_null, seg=seg, title="Gene state flow, null contrast")}
    save_figs(figs, os.path.join(a.out, "figures"))
    log("alluvials written")

    # ---- expression prediction -----------------------------------------------------------
    if not a.skip_predict:
        expr = pd.read_csv(os.path.join(RNA, "expression.tsv"), sep="\t").set_index("gene_id")
        cond_of = {s: s.split("_")[0] for s in samples}
        expr = np.log2(pd.DataFrame({s: expr[c] for s, c in cond_of.items()}) + 1.0)
        expr = expr.loc[expr.index.intersection(gq["gene_id"].unique())]
        summary["predict"] = {}
        level = cs.predict_expression(gq, expression=expr, seed=a.seed)
        level.write(os.path.join(a.out, "predict", "level"), seg=seg)
        summary["predict"]["level"] = level.summary
        log(level.note)
        for cond in ("Cl13", "Arm"):
            lfc = cs.read_lfc(os.path.join(RNA, f"de_{cond}_vs_Naive.tsv"))
            ch = cs.predict_expression(gq, lfc=lfc, contrast=(groups["Naive"], groups[cond]), seed=a.seed)
            ch.write(os.path.join(a.out, "predict", f"change_{cond}_vs_Naive"), seg=seg)
            summary["predict"][f"change_{cond}_vs_Naive"] = ch.summary
            log(ch.note)

    summary["seconds"] = time.time() - T0
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log(f"done -> {a.out}")


if __name__ == "__main__":
    main()
