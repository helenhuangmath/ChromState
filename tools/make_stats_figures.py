"""Showcase of the statistical figures on simulated data with replicates, plus the
benchmark precision-recall curves from a saved `tools/run_tools.py run --save-runs`.

    python tools/make_stats_figures.py -o tmp/showcase/stats [--runs tmp/simbench/sim_k7/runs]

Writes PDF and PNG for every figure and a summary of the differential report.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import chromstate as cs  # noqa: E402
from chromstate import viz  # noqa: E402
from chromstate.compare import differential_report  # noqa: E402
from chromstate.simulate import simulate  # noqa: E402

GROUPS = {"ctrl": ["ctrl_1", "ctrl_2", "ctrl_3"], "drug": ["drug_1", "drug_2", "drug_3"],
          "drugB": ["drugB_1", "drugB_2", "drugB_3"]}


def save_png(figs, outdir):
    os.makedirs(outdir, exist_ok=True)
    for name, fig in figs.items():
        viz.enforce_text_style(fig)
        fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=220, bbox_inches="tight", facecolor=viz.SURFACE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--runs", default=None, help="saved benchmark runs for the PR curves")
    p.add_argument("--n-bins", type=int, default=6000)
    p.add_argument("--preview", action="store_true", help="also write PNG previews under <out>/preview (screen only)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    # 1. simulated three-condition experiment with three replicates each
    samples = tuple(s for v in GROUPS.values() for s in v)
    ds, truth = simulate(n_bins=a.n_bins, n_chroms=4, samples=samples,
                         treated=tuple(GROUPS["drug"] + GROUPS["drugB"]),
                         perturbation={"responder_fraction": 0.5, "from_state": "ReprPC", "to_state": "TssA"},
                         seed=7, blur=0.2)
    cs.normalize(ds)
    seg = cs.fit(ds, n_states=truth.n_states, n_restarts=2, seed=0)
    print(f"fit {ds.n_rows:,} rows, loglik {seg.loglik:,.0f} ({time.time()-t0:.0f}s); states {seg.labels}")
    rep = differential_report(seg, GROUPS, tile_bins=10, min_bins=5, fdr=0.05)
    regions = rep.regions["ctrl_vs_drug"][["chrom", "start", "end", "name"]].reset_index(drop=True)
    files = rep.write(os.path.join(a.out, "report"), seg=seg, regions=regions, png=False)
    if a.preview:
        figs = rep.figures(seg, regions=regions)
        save_png(figs, os.path.join(a.out, "preview"))
    print("\n".join(rep.notes))
    print(rep.states.to_string(index=False))
    print(f"wrote {len(files)} report files ({time.time()-t0:.0f}s)")

    # 2. benchmark PR curves
    if a.runs:
        from toolbench.realdata import load_runs
        from toolbench.report import plot_perturbation_curves

        bds, btruth = simulate(n_bins=25000, n_chroms=4, seed=0, blur=0.3,
                               perturbation={"responder_fraction": 0.5, "from_state": "TssBiv", "to_state": "TssA"})
        runs = load_runs(a.runs)
        print("runs:", [r.tool for r in runs])
        fig = plot_perturbation_curves(runs, bds, btruth)
        if a.preview:
            save_png({"perturbation_pr_curves": plot_perturbation_curves(runs, bds, btruth)}, os.path.join(a.out, "preview"))
        viz.save_all({"perturbation_pr_curves": fig}, os.path.join(a.out, "benchmark"))
    print(f"done in {time.time()-t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
