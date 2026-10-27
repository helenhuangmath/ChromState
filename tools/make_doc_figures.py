#!/usr/bin/env python
"""Turn a comparison run and a scaling run into the tables and figures kept in docs/.

    python tools/make_doc_figures.py --compare compare_sim --scaling scaling \
        -o docs/figures/tool_comparison

Writes ``comparison.csv`` and ``scaling.csv`` (copies of the run outputs, trimmed to the
columns the write-up uses), ``accuracy.pdf`` (truth-based metrics, journal double-column
width, short tool names), ``scaling.pdf``, and copies the run's calibration and agreement
PDFs. PDF only, in the package's figure style (Arial, black text).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (os.path.join(ROOT, "src"), HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import pandas as pd  # noqa: E402

from chromstate import viz  # noqa: E402
from toolbench.report import plot_scaling  # noqa: E402

#: Short names for a double-column figure; "real" marks the published software.
SHORT_NAMES = {
    "chromstate": "ChromState",
    "chromhmm_like": "ChromHMM-like (reimplemented)",
    "segway_like": "Segway-like (reimplemented)",
    "gmm": "Gaussian mixture (reimplemented)",
    "kmeans": "k-means (reimplemented)",
    "naive": "Naive signal (no model)",
    "chromhmm": "ChromHMM 1.28 (real)",
    "epicseg": "EpiCSeg 1.0 (real)",
    "normr": "normR 1.36 (real; mark patterns)",
    "dihmm": "diHMM (real; nucleosome level)",
}

KEEP = ["tool", "method", "status", "verified", "n_states", "ari", "ami", "matched_accuracy",
        "biv_auprc", "biv_posthoc_auprc", "biv_auroc", "biv_baseline_auprc", "signal_spearman",
        "accuracy", "mean_confidence", "overconfidence", "ece", "brier", "frac_saturated",
        "bin_coverage", "mean_segment_kb", "states_used", "occupancy_entropy",
        "domain_states_used", "domain_mean_segment_kb", "domain_ari_vs_bin_truth",
        "runtime_s", "peak_rss_mb", "agreement_ari", "reason", "notes"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--compare", required=True, help="output dir of run_tools.py run")
    p.add_argument("--scaling", required=True, help="output dir of scaling.py")
    p.add_argument("-o", "--out", default=os.path.join(ROOT, "docs", "figures",
                                                        "tool_comparison"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    table = pd.read_csv(os.path.join(args.compare, "comparison.csv"))
    table[[c for c in KEEP if c in table.columns]].to_csv(
        os.path.join(args.out, "comparison.csv"), index=False)
    scaling = pd.read_csv(os.path.join(args.scaling, "scaling.csv"))
    scaling.to_csv(os.path.join(args.out, "scaling.csv"), index=False)

    ok = table.loc[table["status"] == "ok"].copy()
    ok["method"] = ok["tool"].map(SHORT_NAMES).fillna(ok["method"])
    figs = {
        "accuracy": viz.plot_method_comparison(
            ok, metrics=("ari", "matched_accuracy", "biv_posthoc_auprc", "signal_spearman"),
            labels={"biv_posthoc_auprc": "Bivalency detection, one scoring rule for all "
                                         "(area under precision-recall curve)"},
            figsize=(viz.WIDTH_DOUBLE, 3.2)),
    }
    scaling = scaling.copy()
    scaling["method"] = scaling["tool"].map(SHORT_NAMES).fillna(scaling["method"])
    figs["scaling"] = plot_scaling(scaling)
    written = viz.save_all(figs, args.out)
    for name in ("tool_calibration.pdf", "tool_agreement.pdf"):
        src = os.path.join(args.compare, "figures", name)
        if os.path.exists(src):
            dst = os.path.join(args.out, name.replace("tool_", ""))
            shutil.copyfile(src, dst)
            written.append(dst)
    print("\n".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
