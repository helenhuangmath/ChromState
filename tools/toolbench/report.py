"""Tables, figures, and a written report for a tool comparison.

Figures reuse :mod:`chromstate.viz` so the type, palette, and spelled-out labelling are
identical to the rest of the package's output -- a comparison figure that looks like it
came from somewhere else invites the reader to treat it as less carefully made.

The written report always states what did *not* run and why. A comparison table listing
four tools when six were requested is the shape of a misleading result, so the skipped
ones are printed with their reason directly under the table rather than in a log file.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chromstate import viz

__all__ = ["write_report", "plot_tool_comparison", "plot_agreement_heatmap",
           "TOOL_METRIC_LABELS"]

#: Spelled-out panel titles for the columns this harness adds on top of
#: :data:`chromstate.viz.METRIC_LABELS`.
TOOL_METRIC_LABELS: Dict[str, str] = {
    "bin_coverage": "Bins the tool assigned\n(fraction of the dataset)",
    "n_segments": "Number of segments",
    "mean_segment_kb": "Mean segment length\n(kilobases)",
    "states_used": "Number of states actually used",
    "occupancy_entropy": "Evenness of state usage\n(normalised entropy)",
    "largest_state_share": "Share of bins in the largest state",
    "mean_max_posterior": "Mean winning posterior probability",
    "frac_above_0.99": "Calls above 0.99 posterior\n(fraction)",
    "mean_posterior_entropy": "Mean posterior entropy",
    "agreement_ari": "Agreement with the reference tool\n(adjusted Rand index)",
    "agreement_ami": "Agreement with the reference tool\n(adjusted mutual information)",
    "runtime_s": "Runtime (seconds)",
    "peak_rss_mb": "Peak memory (megabytes)",
}


def plot_tool_comparison(
    table: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    labels: Optional[Dict[str, str]] = None,
):
    """Small multiples over whichever metrics are present, one panel per metric."""
    default_order = ["ari", "matched_accuracy", "biv_auprc", "signal_spearman",
                     "bin_coverage", "occupancy_entropy"]
    ok = table.loc[table["status"] == "ok"]
    present = [m for m in (metrics or default_order) if m in ok.columns and ok[m].notna().any()]
    if ok.empty or not present:
        fig, ax = viz._new_fig(7.8, 2.4)
        ax.axis("off")
        ax.text(0.0, 0.5, "no tool produced a scoreable segmentation",
                fontsize=viz.FS_LABEL, color=viz.INK2)
        return fig
    width = max(6.0, 3.6 * len(present))
    merged = {**TOOL_METRIC_LABELS, **(labels or {})}
    return viz.plot_method_comparison(ok, metrics=present, labels=merged,
                                      figsize=(width, 1.4 + 0.75 * len(ok)))


def plot_agreement_heatmap(agreement: pd.DataFrame):
    """Pairwise adjusted Rand index between tools, in the package's heatmap style."""
    n = len(agreement)
    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * n, 1.4 + 1.2 * n), facecolor=viz.SURFACE)
    im = ax.imshow(agreement.values, cmap=viz.SEQ_CMAP, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_xticklabels(agreement.columns, rotation=45, ha="right",
                       fontsize=viz.FS_TICK, color=viz.INK2)
    ax.set_yticks(range(n))
    ax.set_yticklabels(agreement.index, fontsize=viz.FS_TICK, color=viz.INK2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=viz.SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)

    for i in range(n):
        for j in range(n):
            v = agreement.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=viz.FS_CELL,
                        color="#ffffff" if v > 0.55 else viz.INK2)

    ax.set_title("Agreement between tools", color=viz.INK, fontsize=viz.FS_TITLE,
                 loc="left", pad=24)
    ax.text(0.0, 1.015,
            "adjusted Rand index on bins both tools called; high agreement means the two "
            "concur, not that either is correct",
            transform=ax.transAxes, color=viz.MUTED, fontsize=viz.FS_SUBTITLE, va="bottom")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(colors=viz.MUTED, labelsize=viz.FS_CELL, length=2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    return fig


def plot_calibration_curves(runs: Sequence, truth) -> Optional[plt.Figure]:
    """Reliability diagram for every method that reports a posterior.

    Needs ground truth, so this is a simulation-only figure. Each method's states are
    matched to the true labels first -- state 3 in one tool is not state 3 in another --
    and the curve then asks the only question that matters for a probability: when a
    method says 0.9, is it right 90% of the time?
    """
    from chromstate.benchmark.metrics import match_states

    posteriors, correct = {}, {}
    true = np.asarray(truth.states)
    for run in runs:
        if not run.ok or run.posterior is None:
            continue
        post = run.posterior
        mapping, _ = match_states(true, np.argmax(post, axis=1),
                                  int(truth.n_states), post.shape[1])
        mapped = np.array([mapping.get(int(s), -1) for s in np.argmax(post, axis=1)])
        name = run.display_name.replace("\n", " ")
        posteriors[name] = post
        correct[name] = mapped == true
    if not posteriors:
        return None
    return viz.plot_calibration(posteriors, correct, figsize=(8.0, 5.4))


def _markdown_table(frame: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def fmt(v):
        if isinstance(v, float):
            return "—" if not np.isfinite(v) else floatfmt.format(v)
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        return str(v)

    head = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def write_report(
    outdir: str,
    table: pd.DataFrame,
    agreement: pd.DataFrame,
    runs: Sequence,
    dataset_note: str,
    metric_notes: Optional[Dict[str, str]] = None,
    truth: Optional[object] = None,
) -> List[str]:
    """Write comparison.csv, agreement.csv, two figures, and REPORT.md. Returns paths."""
    os.makedirs(outdir, exist_ok=True)
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    written: List[str] = []

    csv = os.path.join(outdir, "comparison.csv")
    table.to_csv(csv, index=False)
    written.append(csv)

    if not agreement.empty:
        apath = os.path.join(outdir, "agreement.csv")
        agreement.to_csv(apath)
        written.append(apath)

    figs = {"tool_comparison": plot_tool_comparison(table)}
    if len(agreement) > 1:
        figs["tool_agreement"] = plot_agreement_heatmap(agreement)
    if truth is not None:
        calibration = plot_calibration_curves(runs, truth)
        if calibration is not None:
            figs["tool_calibration"] = calibration
    written.extend(viz.save_all(figs, figdir))

    ok = table.loc[table["status"] == "ok"]
    not_ok = table.loc[table["status"] != "ok"]
    unverified = ok.loc[~ok["verified"].astype(bool), "method"].tolist() if "verified" in ok else []

    show = [c for c in ["method", "ari", "matched_accuracy", "biv_auprc", "signal_spearman",
                        "bin_coverage", "mean_segment_kb", "occupancy_entropy",
                        "frac_saturated", "runtime_s"] if c in ok.columns]

    lines = [
        "# Tool comparison",
        "",
        dataset_note,
        "",
        "## Results",
        "",
        _markdown_table(ok[show]) if not ok.empty else "_No tool produced a segmentation._",
        "",
        "`—` means the metric was not measurable for that tool (no ground truth, no "
        "posterior, or no score of that kind), not that it scored zero.",
        "",
    ]

    if not not_ok.empty:
        lines += [
            "## Not run",
            "",
            _markdown_table(not_ok[["method", "status", "reason"]]),
            "",
        ]
    if unverified:
        lines += [
            "## Unverified adapters",
            "",
            "These ran through a command line that has not been checked against the real "
            "tool in this repository. Confirm the command in `tools/tools.json` and the "
            "log in the tool's work directory before quoting the numbers:",
            "",
            *[f"- {name}" for name in unverified],
            "",
        ]
    if len(agreement) > 1:
        lines += [
            "## Agreement between tools",
            "",
            _markdown_table(agreement.reset_index().rename(columns={"index": "tool"})),
            "",
        ]
    if metric_notes:
        lines += ["## Metrics", ""]
        lines += [f"- **{name}** — {desc}" for name, desc in metric_notes.items()]
        lines += [""]

    path = os.path.join(outdir, "REPORT.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    written.append(path)
    return written
