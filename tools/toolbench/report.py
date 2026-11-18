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

__all__ = ["write_report", "plot_tool_comparison", "plot_agreement_heatmap", "plot_scaling",
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
    "runtime_s": "Run time (s)",
    "peak_rss_mb": "Peak memory (MB)",
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
            if viz.ANNOTATE_VALUES and np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=viz.FS_CELL,
                        color="#ffffff" if v > 0.55 else viz.INK2)

    # adjusted Rand index on bins both tools called; high agreement means the two concur,
    # not that either is correct
    viz._title(ax, "Tool agreement")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Adjusted Rand index", fontsize=viz.FS_TICK)
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


def plot_scaling(table: pd.DataFrame):
    """Runtime and peak memory against total bin-rows, one line per tool, log-log.

    Points that did not run (timeout, failure) are simply absent from a line rather than
    drawn at an invented value; the table beside the figure says why.
    """
    fig, axes = plt.subplots(1, 2, figsize=(viz.WIDTH_DOUBLE, 2.6), facecolor=viz.SURFACE)
    ok = table.loc[table["status"] == "ok"].copy()
    panels = [("runtime_s", "Run time (s)"), ("peak_rss_mb", "Peak memory (MB)")]
    for ax, (col, label) in zip(axes, panels):
        viz._style_axes(ax, grid_axis="both")
        for i, (_tool, grp) in enumerate(ok.groupby("tool", sort=False)):
            grp = grp.dropna(subset=[col]).sort_values("total_bin_rows")
            if grp.empty:
                continue
            name = str(grp["method"].iloc[0]).split(" (")[0]
            color = viz.CVD_PALETTE[i % len(viz.CVD_PALETTE)]
            ax.plot(grp["total_bin_rows"], grp[col], marker="o", markersize=2.5,
                    linewidth=0.9, color=color, label=name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Bins \u00d7 samples", fontsize=viz.FS_LABEL)
        ax.set_ylabel(label, fontsize=viz.FS_LABEL)
    viz._title(axes[0], "Run time",
               "wall clock per tool; external tools timed as subprocesses, in-process "
               "methods timed around fit and decode")
    viz._title(axes[1], "Peak memory",
               "external tools: their process tree; in-process methods: the whole child "
               "interpreter, including the simulated data")
    axes[1].legend(frameon=False, fontsize=viz.FS_LEGEND, loc="upper left",
                   bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    return fig


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

    show = [c for c in ["method", "ari", "matched_accuracy", "biv_auprc", "biv_posthoc_auprc",
                        "signal_spearman",
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



def perturbation_scores(run, ds, truth):
    """Per-bin change score and responder labels for one run (same rule as the metric)."""
    import numpy as np

    from .core import UNASSIGNED

    treated = list(getattr(truth, "treated", []) or [])
    samples = list(ds.samples)
    controls = [s for s in samples if s not in treated]
    masks = {s: ds.sample_mask(s) for s in samples}
    if run.posterior is not None:
        pa = np.mean([run.posterior[masks[s]] for s in controls], axis=0)
        pb = np.mean([run.posterior[masks[s]] for s in treated], axis=0)
        score = 0.5 * np.abs(pa - pb).sum(axis=1)
    else:
        sa = np.stack([run.states[masks[s]] for s in controls])
        sb = np.stack([run.states[masks[s]] for s in treated])
        score = (sa[:, None, :] != sb[None, :, :]).mean(axis=(0, 1))
    ok = np.ones(len(score), dtype=bool)
    for s in samples:
        ok &= run.states[masks[s]] != UNASSIGNED
    y = np.asarray(truth.responder_mask)[masks[controls[0]]]
    return score[ok], y[ok]


def plot_perturbation_curves(runs, ds, truth, figsize=None):
    """Precision-recall (left) and ROC (right) for detecting the simulated responders
    from each tool's control-versus-treated change. Tools without posteriors give a
    single operating point, drawn as a marker."""
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import precision_recall_curve, roc_curve

    from chromstate import viz

    figsize = figsize or (viz.WIDTH_DOUBLE, viz.WIDTH_SINGLE * 0.95)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor=viz.SURFACE)
    cmap = plt.get_cmap("tab10")
    base = None
    for i, run in enumerate([r for r in runs if r.ok and r.states is not None]):
        try:
            score, y = perturbation_scores(run, ds, truth)
        except Exception:
            continue
        if y.sum() == 0:
            continue
        base = float(y.mean())
        label = str(run.tool)
        if run.posterior is not None:
            p, r, _ = precision_recall_curve(y, score)
            ax1.step(r, p, where="post", color=cmap(i % 10), linewidth=0.9, label=label, zorder=3)
            fpr, tpr, _ = roc_curve(y, score)
            ax2.plot(fpr, tpr, color=cmap(i % 10), linewidth=0.9, zorder=3)
        else:
            called = score > 0.5
            tp = float((called & (y > 0)).sum())
            prec = tp / max(called.sum(), 1)
            rec = tp / max(y.sum(), 1)
            ax1.scatter([rec], [prec], color=cmap(i % 10), s=18, marker="D", zorder=4, label=label)
            fpr = float((called & (y == 0)).sum() / max((y == 0).sum(), 1))
            ax2.scatter([fpr], [rec], color=cmap(i % 10), s=18, marker="D", zorder=4)
    if base is not None:
        ax1.axhline(base, color=viz.INK, linewidth=0.5, linestyle="--")
    ax2.plot([0, 1], [0, 1], color=viz.INK, linewidth=0.5, linestyle="--")
    for ax, xl, yl, t in [(ax1, "Recall", "Precision", "Precision-recall"), (ax2, "False positive rate", "True positive rate", "ROC")]:
        viz._style_axes(ax, grid_axis="both")
        ax.set_xlabel(xl, fontsize=viz.FS_LABEL)
        ax.set_ylabel(yl, fontsize=viz.FS_LABEL)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        viz._title(ax, t)
    ax2.legend(*ax1.get_legend_handles_labels(), fontsize=viz.FS_CELL, frameon=False, loc="lower right")
    fig.tight_layout()
    return fig
