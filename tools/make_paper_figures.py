#!/usr/bin/env python
"""Draw the paper figures, one file per panel or row of panels, from saved result tables.

    python tools/make_paper_figures.py [--testing ../testing] [-o docs/figures/paper] [--dpi 600]

Nothing is refitted: every panel reads a table that an earlier run wrote (the simulated
and real-data tool comparisons, the replicate-level CD8 T-cell run and the ENCODE run).
Panels are laid out on a working sheet and each one (or each row of panels that shares
a category axis) is then cropped out with ``viz.save_panels`` as ``Fig<n>_<letters>_<name>``:
Arial 6-7 pt with 8 pt bold panel letters, black text, 0.25-1 pt lines, vector PDF with
editable text plus a PNG at ``--dpi`` (default 600). ``LEGENDS.md`` is written next to them
with the numbers read from the same tables. States are listed most active first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (os.path.join(ROOT, "src"), HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from chromstate import viz  # noqa: E402

MM = viz.MM
ACCENT = "#2a78d6"
BASE = "#b8b6ae"
GAIN = "#e66767"
LOSS = "#3987e5"
NULL = "#8a8a86"
#: Three validated hues, reused for the three conditions and the three cell lines.
TRIO = ["#2a78d6", "#eb6834", "#1baf7a"]
COND = dict(zip(["Naive", "Arm", "Cl13"], TRIO))
COND_NAME = {"Naive": "Naive", "Arm": "Memory (Armstrong)", "Cl13": "Exhausted (clone 13)"}
CELL = dict(zip(["GM12878", "K562", "H1"], TRIO))

SIM_TOOLS = {
    "chromstate": "ChromState", "chromhmm": "ChromHMM", "epicseg": "EpiCSeg",
    "normr": "normR", "dihmm": "diHMM", "segway_like": "Segway-like*",
    "chromhmm_like": "ChromHMM-like*", "gmm": "Gaussian mixture*", "kmeans": "k-means*",
}
REAL_TOOLS = {
    "chromstate": "ChromState", "chromhmm": "ChromHMM", "epicseg": "EpiCSeg",
    "normr": "normR", "dihmm": "diHMM", "segway": "Segway", "raw_signal": "Raw signal",
}
FAMILIES = {
    "quantitative+signal": "Quantitative states + signal",
    "quantitative": "Quantitative states",
    "occupancy+intensity": "State share + intensity",
    "occupancy": "State share",
    "signal": "Raw signal",
    "top_state": "Hard label",
}
PARTS = {"P": "Promoter", "B": "Gene body", "P+B": "Promoter + body"}


# -- layout and small drawing helpers ------------------------------------------


def axes_mm(fig, x, top, w, h, grid="x"):
    """Axes placed in millimetres from the figure's top-left corner."""
    fw, fh = fig.get_figwidth() / MM, fig.get_figheight() / MM
    ax = fig.add_axes([x / fw, 1 - (top + h) / fh, w / fw, h / fh])
    viz._style_axes(ax, grid_axis=grid)
    return ax


def title(ax, text):
    ax.set_title(text, fontsize=viz.FS_TITLE, loc="left", pad=3)


def category_axis(ax, names, show=True):
    n = len(names)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names if show else [""] * n, fontsize=viz.FS_TICK)
    ax.tick_params(axis="y", length=0)
    for label in ax.get_yticklabels():
        if label.get_text() == "ChromState":
            label.set_fontweight("bold")


def zebra(ax, n):
    """Light band behind every other row, so a row can be followed across panels."""
    for i in range(0, n, 2):
        ax.axhspan(i - 0.5, i + 0.5, color="#f3f3f1", linewidth=0, zorder=0)


def bars(ax, values, colors, fmt="{:.2f}", na="n.a.", xmax=None, label_from=None, pad=0.015,
         labels=True):
    """Horizontal bars.  ``labels=False`` prints no numbers (only ``na`` where a value is
    missing): the axis carries the values and the legend quotes the ones that matter."""
    values = np.asarray(values, float)
    y = np.arange(len(values))
    ax.barh(y, np.nan_to_num(values), color=colors, height=0.72, linewidth=0, zorder=2)
    top = xmax if xmax is not None else np.nanmax(values) * 1.22
    ax.set_xlim(0, top)
    for i, v in enumerate(values):
        if not labels and not np.isnan(v):
            continue
        text = na if np.isnan(v) else fmt.format(v)
        start = 0 if np.isnan(v) else v
        if label_from is not None:
            start = max(start, np.nan_to_num(np.asarray(label_from, float)[i]))
        ax.text(start + top * pad, i, text, va="center", ha="left",
                fontsize=viz.FS_CELL)


def dot_ci(ax, y, mid, lo, hi, color, marker="o", size=3.0):
    ax.plot([lo, hi], [y, y], color=color, linewidth=0.75, zorder=2, solid_capstyle="butt")
    ax.plot([mid], [y], marker=marker, markersize=size, color=color, linestyle="None",
            markeredgewidth=0, zorder=3)


def handles(colors, markers=None, size=3.0):
    markers = markers or ["o"] * len(colors)
    return [Line2D([], [], marker=m, linestyle="None", color=c, markersize=size, markeredgewidth=0)
            for c, m in zip(colors, markers)]


def legend(ax, hs, labels, **kw):
    kw.setdefault("handletextpad", 0.3)
    kw.setdefault("borderaxespad", 0.2)
    kw.setdefault("labelspacing", 0.25)
    return ax.legend(hs, labels, fontsize=viz.FS_LEGEND, frameon=False, **kw)


def emission_panel(ax, em):
    z = np.log1p(em.to_numpy())
    z = z / z.max(axis=0, keepdims=True)
    ax.imshow(z, aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    ax.grid(False)
    ax.set_xticks(range(em.shape[1]))
    ax.set_xticklabels(em.columns, rotation=45, ha="right", rotation_mode="anchor",
                       fontsize=viz.FS_TICK)
    ax.set_yticks(range(em.shape[0]))
    ax.set_yticklabels(viz.state_labels(em.index), fontsize=viz.FS_TICK)
    ax.tick_params(length=0)
    frame(ax)


def frame(ax):
    """Thin box around an image panel, so a white cell is not lost against the page."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.25)


def emission_scale(fig, x, top, w):
    """Grey scale bar for an emission panel (0 to the mark's maximum)."""
    cax = axes_mm(fig, x, top, w, 1.5, grid=None)
    cax.imshow(np.linspace(0, 1, 64)[None, :], aspect="auto", cmap="Greys", vmin=0, vmax=1,
               extent=(0, 1, 0, 1), interpolation="nearest")
    cax.grid(False)
    cax.set_yticks([])
    cax.set_xticks([0, 1])
    cax.set_xticklabels(["0", "max"], fontsize=viz.FS_TICK)
    cax.tick_params(length=1.5, pad=1)
    frame(cax)
    cax.text(-0.08, 0.5, "Mark level", transform=cax.transAxes, ha="right", va="center",
             fontsize=viz.FS_TICK)
    return cax


def lettered(axes, letters):
    """Letter the panels and return ``letter -> [axes, letter text]`` for save_panels."""
    return {letter: [ax, viz.panel_letter(ax, letter)] for ax, letter in zip(axes, letters)}


def panel_groups(parts, spec, extra=None):
    """``{file suffix: artists}`` from ``spec`` = {suffix: "abc"}; ``extra`` adds artists
    (a colour bar, a row heading) to the group holding a given letter."""
    extra = extra or {}
    return {suffix: [a for letter in letters for a in parts[letter] + list(extra.get(letter, []))]
            for suffix, letters in spec.items()}


def active_first(states):
    """State order for the figures: active transcription start sites first, quiescent
    states last (the model's own order runs from quiet to active, by signal)."""
    return viz.display_order(states)


def read(path, **kw):
    sep = "\t" if path.endswith((".tsv", ".tsv.gz")) else ","
    return pd.read_csv(path, sep=sep, **kw)


# -- Figure 1: simulation --------------------------------------------------------


def figure1(src):
    table = read(os.path.join(src["sim"], "comparison.csv")).set_index("tool").loc[list(SIM_TOOLS)]
    scaling = read(src["scaling"])
    names = list(SIM_TOOLS.values())
    colors = [ACCENT if t == "chromstate" else BASE for t in table.index]
    fig = viz.figure_mm(200, 92)
    panels = [
        ("ari", "State recovery", "Adjusted Rand index", 1.0),
        ("biv_posthoc_auprc", "Bivalent bins", "AUPRC", 1.0),
        ("signal_spearman", "Signal retained", "Spearman correlation", 1.0),
        ("perturb_auprc", "Perturbed bins", "AUPRC", 1.0),
        ("ece", "Calibration error", "Expected calibration error", None),
    ]
    xs, tops = [25, 76, 127], [8, 54]
    axes = []
    for i, (col, text, xlabel, xmax) in enumerate(panels):
        ax = axes_mm(fig, xs[i % 3], tops[i // 3], 43, 28)
        category_axis(ax, names, show=i % 3 == 0)
        bars(ax, table[col], colors, xmax=xmax, labels=False)
        if col == "perturb_auprc":
            ax.axvline(table["perturb_baseline_auprc"].iloc[0], color="#000000", linewidth=0.5,
                       linestyle=(0, (2, 2)), zorder=4)
        ax.set_xlabel(xlabel)
        title(ax, text)
        axes.append(ax)
    ax = axes_mm(fig, 142, tops[1], 26, 28, grid="both")
    order = ["chromstate", "chromhmm", "epicseg", "normr", "dihmm"]
    line_colors = dict(zip(order, [ACCENT, "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]))
    ends = {}
    for tool in order:
        sub = scaling[scaling["tool"] == tool].sort_values("total_bin_rows")
        ax.plot(sub["total_bin_rows"], sub["runtime_s"], color=line_colors[tool], marker="o",
                markersize=2.5, markeredgewidth=0, linewidth=1.0 if tool == "chromstate" else 0.75)
        ends[tool] = (sub["total_bin_rows"].iloc[-1], sub["runtime_s"].iloc[-1])
    # direct labels, spread so that end points closer than a text line do not collide
    ranked = sorted(ends, key=lambda t: ends[t][1])
    y_text = []
    for tool in ranked:
        y = np.log10(ends[tool][1])
        y_text.append(y if not y_text else max(y, y_text[-1] + 0.24))
    for tool, y in zip(ranked, y_text):
        x_end, y_end = ends[tool]
        # leader in the line's colour ties each label to its curve
        ax.plot([x_end * 1.1, x_end * 1.55], [y_end, 10 ** y], color=line_colors[tool],
                linewidth=0.5, clip_on=False, zorder=3)
        ax.text(x_end * 1.7, 10 ** y, SIM_TOOLS[tool], va="center", ha="left",
                fontsize=viz.FS_ANNOT, fontweight="bold" if tool == "chromstate" else "normal")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Bins \u00d7 samples")
    ax.set_ylabel("Run time (s)")
    title(ax, "Run time")
    axes.append(ax)
    parts = lettered(axes, "abcdef")
    groups = panel_groups(parts, {"a-c_state_recovery": "abc", "d-e_perturbation_calibration": "de",
                                  "f_run_time": "f"})
    t = table
    text = (
        "**Figure 1 | Recovery of known chromatin states on simulated data.** "
        f"Nine methods were run on the same simulated counts (200,000 bins, seven true states); "
        "tools marked * are reimplementations of the published model, the others are the "
        "published software (ChromHMM 1.28, EpiCSeg 1.0, normR 1.36, diHMM). ChromState is "
        "highlighted. "
        f"**a**, Agreement of the called states with the truth (ChromState {t.ari['chromstate']:.2f}, "
        f"ChromHMM {t.ari['chromhmm']:.2f}, EpiCSeg {t.ari['epicseg']:.2f}). "
        "**b**, Detection of truly bivalent bins, scored with one rule for all tools (area under the "
        f"precision-recall curve, AUPRC; ChromState {t.biv_posthoc_auprc['chromstate']:.2f}). "
        "**c**, Spearman correlation between each tool's activity score and the simulated signal "
        f"(ChromState {t.signal_spearman['chromstate']:.2f}). "
        f"**d**, Detection of the bins moved by a simulated perturbation from each tool's "
        f"control-versus-treated change; dashed line, expected value for a random ranking "
        f"({t.perturb_baseline_auprc['chromstate']:.2f}; ChromState {t.perturb_auprc['chromstate']:.2f}). "
        "**e**, Expected calibration error of the state posterior, lower is better "
        f"(ChromState {t.ece['chromstate']:.3f}); n.a., tool returns no posterior. "
        "**f**, Wall-clock time against data size for the published tools on one machine. "
        "The simulation draws counts from a negative-binomial model, the family ChromState fits, "
        "so these panels favour ChromState; Figure 2 gives the comparison on real data."
    )
    return fig, text, groups


# -- Figure 2: real-data benchmark -------------------------------------------------


def _real_block(src_dir, comparison):
    promo = read(os.path.join(src_dir, "promoter_recovery.tsv")).set_index("tool")
    expr = read(os.path.join(src_dir, "expression_prediction.tsv")).set_index("tool")
    agree = read(os.path.join(src_dir, "agreement.tsv")).set_index("tool")
    runtime = read(comparison).set_index("tool")["runtime_s"]
    return promo, expr, agree, runtime


def figure2(src):
    blocks = [
        ("CD8 T cells", _real_block(src["cd8_bench"], src["cd8_bench_cmp"])),
        ("ENCODE cell lines", _real_block(src["enc_bench"], src["enc_bench_cmp"])),
    ]
    tools = list(REAL_TOOLS)
    names = list(REAL_TOOLS.values())
    colors = [ACCENT if t == "chromstate" else BASE for t in tools]
    depth, width = 150, 200
    fig = viz.figure_mm(width, depth)
    xs, w, h = [24, 82, 140], 41, 24
    tops = [15, 62]
    heads = [("active", "Active promoters", "AUPRC"),
             ("level", "Expression level", "Spearman, held-out genes"),
             ("change", "Expression change", "Spearman, held-out genes")]
    axes, heads_text = [], []
    for r, (dataset, (promo, expr, _, _)) in enumerate(blocks):
        top = tops[r]
        heads_text.append(fig.text(xs[0] / width, 1 - (top - 9) / depth, dataset,
                                   fontsize=viz.FS_TITLE, fontweight="bold", ha="left", va="bottom"))
        for k, (col, text, xlabel) in enumerate(heads):
            ax = axes_mm(fig, xs[k], top, w, h)
            category_axis(ax, names, show=k == 0)
            if col == "active":
                vals = promo["active_tss_auprc"].reindex(tools)
                per = promo.filter(regex="^active_tss_auprc_").reindex(tools)
                bars(ax, vals, colors, na="", xmax=max(vals.max(), np.nanmax(per.to_numpy())) * 1.1,
                     labels=False)
                for i in range(len(tools)):
                    ax.plot(per.iloc[i].to_numpy(), [i] * per.shape[1], marker="o", markersize=1.6,
                            color="#000000", linestyle="None", markeredgewidth=0, zorder=4)
                ax.axvline(promo["active_tss_baseline"].iloc[0], color="#000000", linewidth=0.5,
                           linestyle=(0, (2, 2)), zorder=4)
            else:
                for i, tool in enumerate(tools):
                    row = expr.loc[tool]
                    color = ACCENT if tool == "chromstate" else ("#000000" if tool == "raw_signal" else "#77756f")
                    dot_ci(ax, i, row[f"{col}_rho"], row[f"{col}_lo"], row[f"{col}_hi"], color,
                           marker="D" if tool == "raw_signal" else "o")
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.set_xlabel(xlabel)
            title(ax, text)
            axes.append(ax)
    top = 104
    short = [REAL_TOOLS[t] for t in tools[:6]]
    for k, (_, (_, _, agree, _)) in enumerate(blocks):
        ax = axes_mm(fig, 24 + k * 36, top, 30, 30, grid=None)
        m = agree.loc[short, short].to_numpy(float)
        shown = np.where(np.eye(len(short), dtype=bool), np.nan, m)
        im = ax.imshow(shown, cmap=viz.SEQ_CMAP, vmin=0, vmax=0.7, interpolation="nearest")
        ax.set_xticks(range(len(short)))
        ax.set_xticklabels(short, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(short)))
        ax.set_yticklabels(short if k == 0 else [""] * len(short))
        ax.tick_params(length=0)
        frame(ax)
        title(ax, "Tool agreement, " + ("CD8 T cells" if k == 0 else "ENCODE"))
        axes.append(ax)
    cax = axes_mm(fig, 24 + 36 + 30 + 2, top, 1.5, 30, grid=None)
    cbar = fig.colorbar(im, cax=cax, ticks=[0, 0.35, 0.7])
    cbar.outline.set_linewidth(0.25)
    cax.tick_params(length=1.5, pad=1, labelsize=viz.FS_TICK)
    cax.set_ylabel("Adjusted Rand index", fontsize=viz.FS_TICK)
    road = read(src["enc_roadmap"])
    ax = axes_mm(fig, 140, top, w, 30)
    category_axis(ax, short)
    marks = dict(zip(CELL, ["o", "s", "^"]))
    for cell, sub in road.groupby("cell"):
        sub = sub.set_index("tool").reindex(tools[:6])
        ax.plot(sub["ARI"], range(6), marker=marks[cell], color=CELL[cell], linestyle="None",
                markersize=3, markeredgewidth=0, zorder=3)
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("Adjusted Rand index")
    legend(ax, handles(list(CELL.values()), list(marks.values())), list(CELL), loc="upper left",
           bbox_to_anchor=(-0.05, -0.24), ncol=3, columnspacing=0.8)
    title(ax, "Agreement with Roadmap")
    axes.append(ax)
    parts = lettered(axes, "abcdefghi")
    groups = panel_groups(parts, {"a-c_cd8_tcells": "abc", "d-f_encode": "def",
                                  "g-h_tool_agreement": "gh", "i_roadmap_agreement": "i"},
                          extra={"a": [heads_text[0]], "d": [heads_text[1]], "h": [cax]})
    c, e = blocks[0][1], blocks[1][1]
    text = (
        "**Figure 2 | Six tools on real data, scored on genes and RNA-seq they never saw.** "
        "Top row, CD8 T cells (naive, memory, exhausted; CUT&RUN for H3K4me3, H3K27ac, H3K27me3, "
        "H3K9me3; chromosome 19; 15 states). Middle row, GM12878, K562 and H1 (ENCODE ChIP-seq, "
        "six marks; chromosome 19; 18 states). ChromState is highlighted. "
        "**a**,**d**, Area under the precision-recall curve (AUPRC) for finding promoters of expressed "
        "genes from each tool's states (bars, all samples; dots, one sample each; dashed line, random "
        f"ranking). EpiCSeg is highest in both data sets ({c[0].active_tss_auprc['epicseg']:.2f} and "
        f"{e[0].active_tss_auprc['epicseg']:.2f}); ChromState is second "
        f"({c[0].active_tss_auprc['chromstate']:.2f} and {e[0].active_tss_auprc['chromstate']:.2f}). "
        "**b**,**e**, Spearman correlation between measured expression and expression predicted from "
        "promoter states on held-out genes; lines, 95% bootstrap intervals; diamond, the same model "
        "given the raw mark signal instead of states. "
        "**c**,**f**, The same for the change in expression between two conditions (exhausted versus "
        f"naive; K562 versus GM12878). In CD8 T cells ChromState is highest "
        f"({c[1].change_rho['chromstate']:.2f}; raw signal {c[1].change_rho['raw_signal']:.2f}); in the "
        f"ENCODE data EpiCSeg and the raw signal are higher ({e[1].change_rho['epicseg']:.2f} and "
        f"{e[1].change_rho['raw_signal']:.2f} against {e[1].change_rho['chromstate']:.2f}), with "
        "overlapping intervals. "
        "**g**,**h**, Pairwise agreement of the segmentations (adjusted Rand index, ARI). "
        "**i**, Agreement of each tool's ENCODE segmentation with the published Roadmap 18-state "
        "annotation of the same cell type; that annotation was made with ChromHMM. "
        "Wall-clock time (CD8 T cells; ENCODE): "
        + "; ".join(f"{REAL_TOOLS[t]} {c[3][t]:,.0f} s and {e[3][t]:,.0f} s" for t in tools[:6]) + "."
    )
    return fig, text, groups


# -- Figure 3: replicated differential analysis ---------------------------------------


def _condition(sample):
    return sample.split("_")[0]


def figure3(src):
    run = src["reps"]
    em = read(os.path.join(run, "emissions.tsv"), index_col=0)
    occ = read(os.path.join(run, "occupancy.tsv"), index_col=0)
    states = active_first(em.index)
    em = em.loc[states]
    diff = os.path.join(run, "differential")
    summ = read(os.path.join(diff, "tiles_vs_naive", "state_summary.tsv"))
    cl13 = summ[summ["contrast"] == "Naive_vs_Cl13"].set_index("state").loc[states]
    genome = read(os.path.join(diff, "tiles_vs_naive", "genome_state_tests.tsv"))
    null = read(os.path.join(diff, "tiles_null_d30_vs_d32", "genome_state_tests.tsv"))
    meta = json.load(open(os.path.join(run, "summary.json")))["differential"]

    fig = viz.figure_mm(210, 148)
    top, h = 10, 58
    a = axes_mm(fig, 46, top, 20, h, grid=None)
    emission_panel(a, em)
    scale = emission_scale(fig, 46, top + h + 17, 20)
    title(a, "Mark levels")

    b = axes_mm(fig, 72, top, 36, h)
    category_axis(b, states, show=False)
    zebra(b, len(states))
    offsets = {"Naive": -0.25, "Arm": 0.0, "Cl13": 0.25}
    for sample, row in occ.iterrows():
        cond = _condition(sample)
        b.plot(row[states].to_numpy() * 100, np.arange(len(states)) + offsets[cond], marker="o",
               markersize=1.8, color=COND[cond], linestyle="None", markeredgewidth=0, zorder=3)
    b.set_xscale("log")
    b.set_xlabel("Share of genome (%)")
    legend(b, handles(list(COND.values())), [COND_NAME[c] for c in COND], loc="upper left",
           bbox_to_anchor=(-0.05, -0.2))
    title(b, "State share")

    c = axes_mm(fig, 116, top, 36, h)
    category_axis(c, states, show=False)
    zebra(c, len(states))
    gain = 100 * cl13["n_sig_gain"] / cl13["n_regions"]
    loss = 100 * cl13["n_sig_loss"] / cl13["n_regions"]
    y = np.arange(len(states))
    c.barh(y, gain, color=GAIN, height=0.72, linewidth=0, zorder=2)
    c.barh(y, -loss, color=LOSS, height=0.72, linewidth=0, zorder=2)
    c.axvline(0, color="#000000", linewidth=0.5, zorder=4)
    lim = max(gain.max(), loss.max()) * 1.1
    c.set_xlim(-lim, lim)
    c.set_xticks(c.get_xticks())
    c.set_xticklabels([f"{abs(v):g}" for v in c.get_xticks()])
    c.set_xlim(-lim, lim)
    c.set_xlabel("Tiles losing | gaining (%)")
    title(c, "Exhausted versus naive")

    d = axes_mm(fig, 160, top, 36, h)
    category_axis(d, states, show=False)
    zebra(d, len(states))
    series = [("Arm", genome[genome["condition"] == "Arm"], COND["Arm"], -0.25),
              ("Cl13", genome[genome["condition"] == "Cl13"], COND["Cl13"], 0.0),
              ("null", null, NULL, 0.25)]
    for _, frame, color, off in series:
        frame = frame.set_index("state").loc[states]
        for i, (_, row) in enumerate(frame.iterrows()):
            dot_ci(d, i + off, row["delta_pct"], row["delta_lo"], row["delta_hi"], color, size=2.2)
    d.axvline(0, color="#000000", linewidth=0.5, zorder=1)
    d.set_xlabel("Change in share (points)")
    legend(d, handles([COND["Arm"], COND["Cl13"], NULL]),
           ["Memory vs naive", "Exhausted vs naive", "Naive vs naive"], loc="upper left",
           bbox_to_anchor=(-0.05, -0.2))
    title(d, "Genome-wide change")

    e = axes_mm(fig, 16, 102, 52, 34, grid="both")
    curves = [
        ("tiles_null_d30_vs_d32", "regions_Naive_d30_vs_Naive_d32.tsv", "Naive vs naive", NULL),
        ("tiles_arm_vs_cl13", "regions_Arm_vs_Cl13.tsv", "Exhausted vs memory", "#eda100"),
        ("tiles_vs_naive", "regions_Naive_vs_Arm.tsv", "Memory vs naive", COND["Arm"]),
        ("tiles_vs_naive", "regions_Naive_vs_Cl13.tsv", "Exhausted vs naive", COND["Cl13"]),
    ]
    q = np.linspace(0, 1, 400)
    for folder, name, label, color in curves:
        tv = read(os.path.join(diff, folder, name), usecols=["tv_distance"])["tv_distance"].to_numpy()
        e.plot(np.quantile(tv, q), q, color=color, linewidth=1.0, label=label)
    floor = meta["tiles_vs_naive"]["noise_floor_region"]
    e.axvline(floor, color="#000000", linewidth=0.5, linestyle=(0, (2, 2)))
    e.text(floor + 0.015, 0.05, "noise floor", fontsize=viz.FS_ANNOT, va="bottom", ha="left")
    e.set_xlim(0, 1)
    e.set_ylim(0, 1)
    e.set_xlabel("Change in state composition (total variation)")
    e.set_ylabel("Cumulative fraction of tiles")
    e.legend(fontsize=viz.FS_LEGEND, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5),
             handlelength=1.4, labelspacing=0.25)
    title(e, "Change per 2-kb tile")

    f = axes_mm(fig, 130, 102, 61, 34, grid="y")
    contrasts = [("Naive vs\nnaive", ("tiles_null_d30_vs_d32", "Naive_d30_vs_Naive_d32"),
                  ("promoters_null_d30_vs_d32", "Naive_d30_vs_Naive_d32")),
                 ("Exhausted vs\nmemory", ("tiles_arm_vs_cl13", "Arm_vs_Cl13"), None),
                 ("Memory vs\nnaive", ("tiles_vs_naive", "Naive_vs_Arm"), ("promoters_vs_naive", "Naive_vs_Arm")),
                 ("Exhausted vs\nnaive", ("tiles_vs_naive", "Naive_vs_Cl13"), ("promoters_vs_naive", "Naive_vs_Cl13"))]
    fracs = {}
    for i, (_, tiles, promoters) in enumerate(contrasts):
        for k, (key, color) in enumerate([(tiles, "#77756f"), (promoters, "#cfcdc6")]):
            if key is None:
                f.text(i + (k - 0.5) * 0.36, 0.006, "n.a.", ha="center", va="bottom", fontsize=viz.FS_CELL)
                continue
            v = 100 * meta[key[0]][key[1]]["frac_confident"]
            fracs[(i, k)] = v
            x = i + (k - 0.5) * 0.36
            f.bar(x, v, width=0.34, color=color, linewidth=0, zorder=2)
            f.text(x, v * 1.15, f"{v:.2g}" if v < 1 else f"{v:.1f}", ha="center", va="bottom",
                   fontsize=viz.FS_CELL)
    f.set_yscale("log")
    f.set_ylim(0.005, 200)
    f.set_xticks(range(len(contrasts)))
    f.set_xticklabels([c[0] for c in contrasts])
    f.tick_params(axis="x", length=0)
    f.set_ylabel("Regions changed (%)")
    hs = [Line2D([], [], marker="s", linestyle="None", color=col, markersize=4, markeredgewidth=0)
          for col in ("#77756f", "#cfcdc6")]
    legend(f, hs, ["2-kb tiles", "Promoters"], loc="upper left")
    title(f, "Confident changes")

    parts = lettered([a, b, c, d, e, f], "abcdef")
    groups = panel_groups(parts, {"a-d_states_and_changes": "abcd", "e_change_per_tile": "e",
                                  "f_confident_changes": "f"}, extra={"a": [scale]})
    n_tiles = meta["tiles_vs_naive"]["Naive_vs_Cl13"]["n_regions"]
    big_null = null.loc[null["delta_pct"].abs().idxmax()]
    text = (
        "**Figure 3 | Replicated comparison of chromatin states between CD8 T-cell conditions.** "
        "One 15-state model was fitted to 16 CUT&RUN samples (eight naive from two batches, four "
        "memory, four exhausted; four marks; five chromosomes). "
        "**a**, Mark level in each state (log(1 + mean count), scaled to each mark's maximum). "
        "**b**, Share of the genome in each state, one dot per replicate. "
        f"**c**, Percentage of {n_tiles:,} 2-kb tiles in which a state's share rose (right) or fell "
        "(left) in exhausted relative to naive cells (moderated t-test per tile and state, "
        "Benjamini-Hochberg FDR 5%). "
        "**d**, Change in each state's genome share relative to naive cells, with 95% block-bootstrap "
        "intervals in percentage points; grey (naive vs naive), the same statistic between the two "
        "batches of naive cells. "
        f"Batch alone moves a state by up to {abs(big_null.delta_pct):.1f} percentage points "
        f"({viz.state_labels([big_null.state])[0]}), so genome-wide shifts of that size are not "
        "evidence of a condition effect. "
        "**e**, Cumulative distribution of the change in state composition per 2-kb tile (total "
        "variation distance); dashed line, the noise floor estimated from replicate pairs. "
        "**f**, Percentage of regions called changed (FDR 5% and above the replicate noise floor); "
        "n.a., promoters were not tested for that contrast. The contrast "
        f"between the two naive batches calls {fracs[(0, 0)]:.2g}% of tiles and {fracs[(0, 1)]:.2g}% of "
        f"promoters; exhausted versus naive calls {fracs[(3, 0)]:.1f}% and {fracs[(3, 1)]:.1f}%."
    )
    return fig, text, groups


# -- Figure 4: expression ---------------------------------------------------------------


def figure4(src):
    tasks = {
        ("CD8", "level"): os.path.join(src["reps"], "predict", "level"),
        ("CD8", "change"): os.path.join(src["reps"], "predict", "change_Cl13_vs_Naive"),
        ("ENCODE", "level"): os.path.join(src["encode"], "predict", "level"),
        ("ENCODE", "change"): os.path.join(src["encode"], "predict", "change_K562_vs_GM12878"),
    }
    data_color = {"CD8": TRIO[1], "ENCODE": TRIO[2]}
    data_name = {"CD8": "CD8 T cells (CUT&RUN, 4 marks)", "ENCODE": "ENCODE cell lines (ChIP-seq, 6 marks)"}
    fig = viz.figure_mm(183, 126)

    pred = read(os.path.join(tasks[("CD8", "level")], "gene_predictions.tsv"), usecols=["observed", "predicted"])
    a = axes_mm(fig, 13, 10, 40, 40, grid=None)
    lo, hi = np.quantile(pred["observed"], [0.001, 0.999])
    a.hexbin(pred["observed"], pred["predicted"], gridsize=42, cmap=viz.SEQ_CMAP, norm=LogNorm(),
             linewidths=0, mincnt=1, extent=(lo, hi, lo, hi))
    a.plot([lo, hi], [lo, hi], color="#000000", linewidth=0.5, linestyle=(0, (2, 2)))
    a.set_xlim(lo, hi)
    a.set_ylim(lo, hi)
    rep = read(os.path.join(tasks[("CD8", "level")], "representation_scores.tsv")).set_index("family")
    rho = rep.loc["quantitative", "spearman"]
    a.text(0.04, 0.96, f"Spearman {rho:.2f}", transform=a.transAxes, va="top", fontsize=viz.FS_ANNOT)
    a.set_xlabel("Measured expression (log)")
    a.set_ylabel("Predicted expression (log)")
    title(a, "CD8 T cells")

    reps = {}
    fam = list(FAMILIES)
    for k, (kind, text) in enumerate([("level", "Expression level"), ("change", "Expression change")]):
        ax = axes_mm(fig, 101 + k * 41, 10, 36, 40)
        category_axis(ax, list(FAMILIES.values()), show=k == 0)
        for j, data in enumerate(["CD8", "ENCODE"]):
            t = read(os.path.join(tasks[(data, kind)], "representation_scores.tsv")).set_index("family").loc[fam]
            reps[(data, kind)] = t
            for i, (_, row) in enumerate(t.iterrows()):
                dot_ci(ax, i + (j - 0.5) * 0.3, row["spearman"], row["rho_lo"], row["rho_hi"], data_color[data])
        ax.set_xlabel("Spearman, held-out genes")
        title(ax, text)
        if k == 0:
            b = ax
            legend(ax, handles(list(data_color.values())), ["CD8 T cells", "ENCODE"], loc="lower right")
        else:
            c = ax

    imp_axes = []
    for k, data in enumerate(["CD8", "ENCODE"]):
        imp = read(os.path.join(tasks[(data, "level")], "state_importance.tsv")).sort_values(
            "unique_r2", ascending=False).head(8)
        top_state = imp.iloc[0]
        # the eight most important states, listed active first like every other state panel
        order = active_first(read(os.path.join(src["reps" if data == "CD8" else "encode"],
                                               "emissions.tsv"), index_col=0).index)
        imp = imp.set_index("state", drop=False).loc[[st for st in order if st in set(imp["state"])]]
        ax = axes_mm(fig, 40 + k * 66, 74, 26, 38)
        category_axis(ax, viz.state_labels(imp["state"]))
        colors = viz._label_colors(list(imp["state"]), palette="nature")
        ax.barh(range(len(imp)), imp["unique_r2"], color=[colors[s] for s in imp["state"]], height=0.72,
                linewidth=0.25, edgecolor="#000000", zorder=2)
        ax.errorbar(imp["unique_r2"], range(len(imp)),
                    xerr=[imp["unique_r2"] - imp["unique_lo"], imp["unique_hi"] - imp["unique_r2"]],
                    fmt="none", ecolor="#000000", elinewidth=0.5, zorder=3)
        ax.set_xlim(0, None)
        ax.set_xlabel("Held-out R² lost on permutation")
        title(ax, f"Importance, {'CD8 T cells' if data == 'CD8' else 'ENCODE'}")
        imp_axes.append((ax, top_state))

    f = axes_mm(fig, 153, 74, 24, 38)
    category_axis(f, list(PARTS.values()))
    marks = {"level": "o", "change": "^"}
    for j, ((data, kind), folder) in enumerate(tasks.items()):
        t = read(os.path.join(folder, "part_ranking.tsv")).set_index("parts").loc[list(PARTS)]
        for i, (_, row) in enumerate(t.iterrows()):
            dot_ci(f, i + (j - 1.5) * 0.18, row["spearman"], row["rho_lo"], row["rho_hi"], data_color[data],
                   marker=marks[kind])
    f.set_xlabel("Spearman, held-out genes")
    legend(f, handles(["#000000", "#000000"], ["o", "^"]), ["Level", "Change"], loc="upper left",
           bbox_to_anchor=(0.0, -0.22), ncol=2, columnspacing=0.8)
    title(f, "Gene part")

    parts = lettered([a, b, c, imp_axes[0][0], imp_axes[1][0], f], "abcdef")
    groups = panel_groups(parts, {"a_predicted_vs_measured": "a", "b-c_representations": "bc",
                                  "d_importance_cd8": "d", "e_importance_encode": "e",
                                  "f_gene_part": "f"})
    r = reps
    top_cd8, top_enc = imp_axes[0][1], imp_axes[1][1]
    text = (
        "**Figure 4 | Chromatin states predict gene expression, and which states do.** "
        "Per-gene state tables (promoter and gene body) were used to predict RNA-seq with "
        "gradient-boosted trees; every prediction is for genes on chromosomes held out of training. "
        f"Orange, {data_name['CD8']}; green, {data_name['ENCODE']}. "
        "**a**, Predicted against measured expression in CD8 T cells, on held-out chromosomes "
        f"({len(pred):,} gene x sample pairs; colour, number of points, log scale). "
        "**b**,**c**, Accuracy for expression level (**b**) and for the change between two conditions "
        "(**c**; exhausted versus naive, K562 versus GM12878) when the gene is described in six ways; "
        "lines, 95% bootstrap intervals. For level, quantitative states add little over the raw signal "
        f"({r[('CD8', 'level')].spearman['quantitative']:.2f} against {r[('CD8', 'level')].spearman['signal']:.2f} "
        f"in CD8 T cells). For change, a hard state label loses most of the information "
        f"({r[('CD8', 'change')].spearman['top_state']:.2f} against "
        f"{r[('CD8', 'change')].spearman['quantitative']:.2f} in CD8 T cells; "
        f"{r[('ENCODE', 'change')].spearman['top_state']:.2f} against "
        f"{r[('ENCODE', 'change')].spearman['quantitative']:.2f} in ENCODE). "
        "**d**,**e**, Drop in held-out R² when one state's columns are permuted (95% intervals); the eight "
        f"largest are shown, most active state first. {viz.state_labels([top_cd8.state])[0]} is the largest in both "
        f"({top_cd8.unique_r2:.2f} and {top_enc.unique_r2:.2f}). "
        "**f**, Accuracy from promoter columns, gene-body columns, or both (circles, level; triangles, change)."
    )
    return fig, text, groups


# -- Figure 5: ENCODE validation ------------------------------------------------------------


def figure5(src):
    run = src["encode"]
    em = read(os.path.join(run, "emissions.tsv"), index_col=0)
    occ = read(os.path.join(run, "occupancy.tsv"), index_col=0)
    starr = read(os.path.join(run, "starr_state_enrichment.tsv"), index_col=0)
    agree = read(os.path.join(run, "reference_agreement.tsv"))
    meta = json.load(open(os.path.join(run, "summary.json")))
    states = active_first(em.index)
    em = em.loc[states]
    fig = viz.figure_mm(200, 146)
    top, h = 10, 62

    a = axes_mm(fig, 46, top, 28, h, grid=None)
    emission_panel(a, em)
    scale = emission_scale(fig, 46, top + h + 17, 28)
    title(a, "Mark levels")

    b = axes_mm(fig, 80, top, 44, h)
    category_axis(b, states, show=False)
    zebra(b, len(states))
    for sample, row in occ.iterrows():
        cell = sample.split("_")[0]
        off = (list(CELL).index(cell) - 1) * 0.25
        b.plot(row[states].to_numpy() * 100, np.arange(len(states)) + off, marker="o", markersize=1.8,
               color=CELL[cell], linestyle="None", markeredgewidth=0, zorder=3)
    b.set_xscale("log")
    b.set_xlabel("Share of genome (%)")
    legend(b, handles(list(CELL.values())), list(CELL), loc="upper left", bbox_to_anchor=(-0.03, -0.14),
           ncol=3, columnspacing=0.8)
    title(b, "State share")

    c = axes_mm(fig, 131, top, 54, h)
    category_axis(c, states, show=False)
    zebra(c, len(states))
    colors = viz._label_colors(states, palette="nature")
    mean = starr.loc[states].mean(axis=1)
    c.barh(range(len(states)), np.log2(mean), color=[colors[s] for s in states], height=0.72,
           linewidth=0.25, edgecolor="#000000", zorder=2)
    for col in starr.columns:
        c.plot(np.log2(starr.loc[states, col]), range(len(states)), marker="o", markersize=1.6,
               color="#000000", linestyle="None", markeredgewidth=0, zorder=4)
    c.axvline(0, color="#000000", linewidth=0.5, zorder=3)
    c.set_xlabel("STARR-seq enrichment in K562 (log2)")
    title(c, "Reporter activity")

    d = axes_mm(fig, 24, 104, 40, 28)
    cells = list(CELL)
    category_axis(d, cells)
    for i, cell in enumerate(cells):
        sub = agree[agree["cell"] == cell]
        ref = sub[sub["vs"] != "replicate"]["ARI"]
        d.plot(ref, [i] * len(ref), marker="o", markersize=3, color="#77756f", linestyle="None",
               markeredgewidth=0, zorder=3)
        d.plot(sub[sub["vs"] == "replicate"]["ARI"], [i], marker="D", markersize=3, color=ACCENT,
               linestyle="None", markeredgewidth=0, zorder=3)
    d.set_xlim(0, 0.6)
    d.set_xlabel("Adjusted Rand index")
    legend(d, handles(["#77756f", ACCENT], ["o", "D"]), ["Roadmap annotation", "Replicate"],
           loc="upper left", bbox_to_anchor=(0.0, -0.3), ncol=2, columnspacing=0.8)
    title(d, "Segmentation agreement")

    cr = meta["crispri_K562"]
    scores = [("chromstate_active_mass", "Active-state share"),
              ("chromstate_expected_H3K27ac", "Expected H3K27ac"),
              ("chromstate_enh_label_mass", "Enhancer-state share"),
              ("raw_H3K27ac", "Raw H3K27ac")]
    e = axes_mm(fig, 110, 104, 28, 28)
    category_axis(e, [s[1] for s in scores])
    vals = [cr[k]["AUPRC"] for k, _ in scores]
    bars(e, vals, [ACCENT, ACCENT, ACCENT, BASE], xmax=0.4, labels=False)
    e.axvline(cr["baseline_AUPRC"], color="#000000", linewidth=0.5, linestyle=(0, (2, 2)), zorder=4)
    e.set_xlabel("AUPRC")
    title(e, "CRISPRi enhancers, K562")

    f = axes_mm(fig, 152, 104, 25, 28, grid="y")
    runs = [("CD8 T cells", json.load(open(os.path.join(src["reps"], "summary.json")))["calibration"]),
            ("ENCODE", meta["calibration"])]
    for i, (_, cal) in enumerate(runs):
        f.bar(i - 0.18, cal["before"]["ece"], width=0.34, color=BASE, linewidth=0, zorder=2)
        f.bar(i + 0.18, cal["after"]["ece"], width=0.34, color=ACCENT, linewidth=0, zorder=2)
        f.text(i - 0.18, cal["before"]["ece"] + 0.005, f"{cal['before']['ece']:.2f}", ha="center",
               fontsize=viz.FS_CELL)
        f.text(i + 0.18, cal["after"]["ece"] + 0.005, f"{cal['after']['ece']:.3f}", ha="center",
               fontsize=viz.FS_CELL)
    f.set_xticks(range(len(runs)))
    f.set_xticklabels([r[0] for r in runs])
    f.tick_params(axis="x", length=0)
    f.set_ylim(0, 0.27)
    f.set_ylabel("Expected calibration error")
    hs = [Line2D([], [], marker="s", linestyle="None", color=col, markersize=4, markeredgewidth=0)
          for col in (BASE, ACCENT)]
    legend(f, hs, ["Raw", "Temperature-scaled"], loc="upper left",
           bbox_to_anchor=(-0.25, -0.3))
    title(f, "Posterior calibration")

    parts = lettered([a, b, c, d, e, f], "abcdef")
    groups = panel_groups(parts, {"a-c_states_and_reporter_activity": "abc", "d_segmentation_agreement": "d",
                                  "e_crispri_enhancers": "e", "f_posterior_calibration": "f"},
                          extra={"a": [scale]})
    best = mean.idxmax()
    road = agree[agree["vs"] != "replicate"]["ARI"]
    repl = agree[agree["vs"] == "replicate"]["ARI"]
    text = (
        "**Figure 5 | External checks of an 18-state ChromState model on ENCODE data.** "
        "One model was fitted to six histone marks in GM12878, K562 and H1 (two replicates each; "
        "chromosomes 1, 8, 11, 19 and X). "
        "**a**, Mark level in each state. **b**, Share of the genome in each state, one dot per replicate. "
        "**c**, Enrichment of STARR-seq enhancer activity in K562 in each state (bars, mean; dots, the two "
        f"replicates); the most enriched state is {viz.state_labels([best])[0]} ({mean[best]:.1f}-fold). "
        f"**d**, Agreement with the Roadmap 18-state annotation of the same cell type "
        f"(ARI {road.min():.2f}-{road.max():.2f}) and between the two replicates of a cell type "
        f"({repl.min():.2f}-{repl.max():.2f}). Replicates of one cell type agree only moderately, which "
        "bounds how far any single-replicate segmentation should be trusted. "
        f"**e**, Ranking of {cr['n_elements']:,} CRISPRi-tested K562 elements ({cr['n_positive']} regulate "
        "a gene) by three ChromState scores (blue: share of active states, expected H3K27ac, share of "
        "enhancer-named states) and by the raw H3K27ac signal (grey); dashed line, random ranking. "
        f"The raw signal ({cr['raw_H3K27ac']['AUPRC']:.2f}) is not improved on by the state scores "
        f"({cr['chromstate_active_mass']['AUPRC']:.2f} for the share of active states), and the states "
        f"the model names as enhancers rank below chance ({cr['chromstate_enh_label_mass']['AUPRC']:.2f}): "
        "the validated elements sit in promoter-like states. "
        "**f**, Calibration error of the state posterior, measured as agreement between replicates, "
        "before and after temperature scaling (the temperature was fitted on the same bins)."
    )
    return fig, text, groups


FIGURES = [
    ("Fig1_simulation_benchmark", figure1),
    ("Fig2_realdata_benchmark", figure2),
    ("Fig3_condition_comparison", figure3),
    ("Fig4_expression", figure4),
    ("Fig5_encode_validation", figure5),
]


def sources(testing):
    t = os.path.abspath(testing)
    enc_bench = os.path.join(t, "realdata_encode", "toolbench", "chr19", "k18_run")
    cd8_bench = os.path.join(t, "tmp", "realbench", "cd8_chr19", "k15_real")
    return {
        "sim": os.path.join(t, "tmp", "simbench", "sim_k7"),
        "scaling": os.path.join(ROOT, "docs", "figures", "tool_comparison", "scaling.csv"),
        "cd8_bench": os.path.join(cd8_bench, "realdata"),
        "cd8_bench_cmp": os.path.join(cd8_bench, "comparison.csv"),
        "enc_bench": os.path.join(enc_bench, "realdata"),
        "enc_bench_cmp": os.path.join(enc_bench, "comparison.csv"),
        "enc_roadmap": os.path.join(enc_bench, "roadmap_agreement.tsv"),
        "reps": os.path.join(t, "realdata_reps", "immune5_k15_final"),
        "encode": os.path.join(t, "realdata_encode", "runs", "crispr5_k18"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--testing", default=os.path.join(ROOT, "..", "testing"),
                   help="folder holding the saved runs (see sources())")
    p.add_argument("-o", "--out", default=os.path.join(ROOT, "docs", "figures", "paper"))
    p.add_argument("--dpi", type=int, default=600,
                   help="resolution of the PNG copy written next to each PDF (0: PDF only)")
    p.add_argument("--only", default=None, help="comma-separated figure numbers, e.g. 1,3")
    args = p.parse_args()
    src = sources(args.testing)
    keep = {int(v) for v in args.only.split(",")} if args.only else None
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    face = next((f for f in viz.FONT_STACK if f in installed), "none of the Arial-metric faces")
    print(f"typeface: {face}" + ("" if face == "Arial" else " (Arial is not installed here)"))
    # Each panel (or row of panels sharing an axis) is written as its own figure, so the
    # type runs at the top of the journal's 5-7 pt range.
    viz.FS_TITLE, viz.FS_LABEL = 7.0, 7.0
    viz.FS_TICK = viz.FS_LEGEND = viz.FS_ANNOT = 6.5
    viz.FS_CELL = 6.0
    viz.use_arial()
    os.makedirs(args.out, exist_ok=True)
    legends = []
    for number, (name, draw) in enumerate(FIGURES, start=1):
        if keep and number not in keep:
            continue
        fig, text, groups = draw(src)
        short = name.split("_")[0]
        # drop this figure's earlier files (composites and panels), so stale ones do not linger
        for old in os.listdir(args.out):
            if old.startswith(short + "_") and old.endswith((".pdf", ".png")):
                os.remove(os.path.join(args.out, old))
        paths = viz.save_panels(fig, groups, args.out, short, png_dpi=args.dpi)
        files = ", ".join(f"`{os.path.basename(p)}`" for p in paths)
        legends.append(f"{text}\n\n*Files (PDF" + (f" and {args.dpi} dpi PNG" if args.dpi else "")
                       + f"): {files}.*")
        print("\n".join(paths))
    if not keep:
        with open(os.path.join(args.out, "LEGENDS.md"), "w") as fh:
            fh.write("# Paper figures and legends (version 1.2)\n\n"
                     "Versions 1.0 and 1.1 (composite pages) are kept unchanged in `v1.0/` and "
                     "`v1.1/`. In version 1.2 every panel, or row of panels that shares an axis, is "
                     "its own file with 6-7 pt type, panel heads are one line, the numbers are in "
                     "the legends instead of on the bars, and states are listed from the most "
                     "active (active transcription start site) to quiescent.\n\n"
                     "Drawn by `tools/make_paper_figures.py` from saved run tables; every number in a "
                     "legend is read from the table its panel draws. Vector PDF with editable text, "
                     "Arial-metric type.\n\n" + "\n\n".join(legends) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
