"""Step 7: Nature-style figures from the step 2-6 outputs.

Fig. 1  The fitted model: emissions, genome occupancy per cell state, K selection,
        expression by promoter state.
Fig. 2  Does chromatin predict expression? Level, change, and naive prepattern tasks,
        and which single mark carries the change.
Fig. 3  Bivalency: concordance with peak-called poised promoters, and whether naive
        bivalent balance anticipates later expression change beyond expression level.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import nature_style as ns
import numpy as np
import pandas as pd
from common import CONDITIONS, MARKS, MODEL_DIR, OUT, ensure_out
from matplotlib.lines import Line2D

from chromstate.model.annotate import full_state_name

FIG = ensure_out("figures")

FAMILY_STYLE = {
    # name: (display, colour, filled)
    "raw": ("Raw signal", ns.BASELINE, True),
    "chmm_tss": ("ChromHMM label at TSS", ns.BASELINE_DARK, True),
    "chmm_frac": ("ChromHMM state composition", ns.BASELINE_DARK, False),
    "cs_tss": ("ChromState label at TSS", "#8fb6ea", True),
    "cs_quant": ("ChromState quantitative", ns.ACCENT, True),
    "cs_quant_raw": ("ChromState quantitative + raw", "#1b4f8f", True),
    "naive_rna": ("TN expression only", ns.BASELINE, False),
    "naive_rna+raw": ("TN expression + raw signal", ns.BASELINE_DARK, False),
    "naive_rna+cs_quant": ("TN expression + ChromState", ns.ACCENT, False),
}
CONTRAST_NAMES = {"Arm_vs_Naive": "TMEM\nvs TN", "Cl13_vs_Naive": "TEX\nvs TN",
                  "Cl13_vs_Arm": "TEX\nvs TMEM"}


def state_display(labels):
    return [full_state_name(lbl) for lbl in labels]


# ---------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------


def figure1():
    mdir = OUT / MODEL_DIR
    import pickle
    with open(mdir / "model.pkl", "rb") as fh:
        model = pickle.load(fh)
    lab = pd.read_csv(mdir / "state_labels.tsv", sep="\t")
    labels = lab["label"].tolist()
    em = pd.DataFrame(model["hmm"].emissions.mean_matrix(), index=labels, columns=MARKS)
    occ = pd.read_csv(mdir / "occupancy_genome.tsv", sep="\t", index_col=0).loc[CONDITIONS]
    feats = pd.read_csv(OUT / "genes" / "gene_features.tsv.gz", sep="\t")
    expr = pd.read_csv(OUT / "rna" / "expression.tsv", sep="\t", index_col="gene_id")
    ksel_path = mdir / "k_selection.tsv"

    fig = plt.figure(figsize=(ns.DOUBLE, 118 * ns.MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0],
                          wspace=0.95, hspace=0.55, left=0.17, right=0.98, bottom=0.08, top=0.95)

    # a  emissions on a log scale, each mark scaled to its maximum state: counts span
    # two orders of magnitude, and a linear scale hides every state but the strongest
    ax = fig.add_subplot(gs[0, 0])
    Z = (np.log1p(em) / np.log1p(em.max(axis=0))).to_numpy()
    im = ax.imshow(Z, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(MARKS)))
    ax.set_xticklabels(MARKS, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(state_display(labels))
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.04, ticks=[0, 1])
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=1.5, width=0.4)
    cb.set_label("log(1 + mean) / mark max.", labelpad=1)
    ax.set_title("State emissions")
    ns.panel_label(ax, "a", x=-0.95)

    # b  genome occupancy per cell state
    ax = fig.add_subplot(gs[0, 1])
    y = np.arange(len(labels))
    h = 0.26
    for i, c in enumerate(CONDITIONS):
        ax.barh(y + (i - 1) * h, occ.loc[c, labels] * 100, height=h,
                color=ns.CONDITION_COLORS[c], label=ns.CONDITION_NAMES[c], linewidth=0)
    ax.set_yticks(y)
    ax.set_yticklabels(state_display(labels))
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Genome covered (%, log scale)")
    ax.legend(loc="lower right", handlelength=0.8, handletextpad=0.4, borderaxespad=0)
    ax.set_title("Occupancy by cell state")
    ns.panel_label(ax, "b", x=-0.95)

    # c  K selection
    ax = fig.add_subplot(gs[1, 0])
    if ksel_path.exists():
        ks = pd.read_csv(ksel_path, sep="\t")
        ax.plot(ks["n_states"], ks["heldout_loglik_per_bin"], "-o", color=ns.INK,
                markersize=2.5, markerfacecolor="white", markeredgewidth=0.6)
        ax.axvline(model["k"], color=ns.ACCENT, linewidth=0.6, linestyle="--")
        ax.text(model["k"], ax.get_ylim()[0], f" K = {model['k']}", color=ns.INK,
                va="bottom", ha="left", fontsize=5.5)
        ax.set_xticks(ks["n_states"])
    ax.set_xlabel("Number of states (K)")
    ax.set_ylabel("Held-out log-likelihood per bin")
    ax.set_title("Model selection")
    ns.panel_label(ax, "c", x=-0.95)

    # d  TN expression by dominant promoter state
    ax = fig.add_subplot(gs[1, 1])
    naive = feats.loc[feats["condition"] == "Naive"].set_index("gene_id")
    comp = naive[[f"P_cs_{lbl}" for lbl in labels]].dropna()
    top = comp.idxmax(axis=1).str.replace("P_cs_", "", regex=False)
    e = expr["Naive"].reindex(top.index)
    order = [lbl for lbl in labels if (top == lbl).sum() >= 50]
    order = sorted(order, key=lambda lbl: e[top == lbl].median())
    data = [e[top == lbl].dropna().to_numpy() for lbl in order]
    bp = ax.boxplot(data, vert=False, widths=0.6, showfliers=False, patch_artist=True,
                    medianprops=dict(color=ns.INK, linewidth=0.8),
                    whiskerprops=dict(linewidth=0.5), capprops=dict(linewidth=0.5),
                    boxprops=dict(linewidth=0.5))
    for patch, lbl in zip(bp["boxes"], order):
        patch.set_facecolor(lab.set_index("label").loc[lbl, "color"])
        patch.set_alpha(0.9)
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels([f"{full_state_name(lbl)} (n={(top == lbl).sum():,})" for lbl in order])
    ax.set_xlabel("TN expression, log2(normalized count + 1)")
    ax.set_title("Expression by promoter state (TN)")
    ns.panel_label(ax, "d", x=-0.95)

    ns.save(fig, FIG, "Fig1_model")


# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------


def dot_panel(ax, sub: pd.DataFrame, families, groups, group_names, title):
    n_f = len(families)
    span = 0.8
    offsets = np.linspace(-span / 2, span / 2, n_f) if n_f > 1 else [0.0]
    for gi, g in enumerate(groups):
        for fi, fam in enumerate(families):
            row = sub.loc[(sub["task"] == g) & (sub["family"] == fam)]
            if row.empty:
                continue
            name, color, filled = FAMILY_STYLE[fam]
            x = gi + offsets[fi]
            r = row.iloc[0]
            ax.plot([x, x], [r["rho_lo"], r["rho_hi"]], color=color, linewidth=0.7)
            ax.plot(x, r["spearman"], "o", markersize=3.2, color=color,
                    markerfacecolor=color if filled else "white", markeredgewidth=0.7)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(group_names)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel("Spearman ρ, held-out chromosomes")
    ax.set_title(title)
    ax.grid(axis="y", linewidth=0.3, color="#e4e3df")
    ax.set_axisbelow(True)


def legend_handles(families):
    hs = []
    for fam in families:
        name, color, filled = FAMILY_STYLE[fam]
        hs.append(Line2D([0], [0], marker="o", linestyle="", color=color, markersize=3.2,
                         markerfacecolor=color if filled else "white", markeredgewidth=0.7,
                         label=name))
    return hs


def figure2():
    sc = pd.read_csv(OUT / "predict" / "scores.tsv", sep="\t")
    marks = pd.read_csv(OUT / "predict" / "single_mark.tsv", sep="\t")
    fams = ["raw", "chmm_tss", "chmm_frac", "cs_tss", "cs_quant", "cs_quant_raw"]

    fig = plt.figure(figsize=(ns.DOUBLE, 68 * ns.MM))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 0.85, 0.9], wspace=0.5,
                          left=0.07, right=0.97, bottom=0.36, top=0.88)

    ax = fig.add_subplot(gs[0])
    dot_panel(ax, sc, fams, [f"A_level_{c}" for c in CONDITIONS],
              [ns.CONDITION_NAMES[c] for c in CONDITIONS], "Expression level")
    ns.panel_label(ax, "a", x=-0.3)

    ax = fig.add_subplot(gs[1])
    contrasts = list(CONTRAST_NAMES)
    dot_panel(ax, sc, fams, [f"B_change_{c}" for c in contrasts],
              [CONTRAST_NAMES[c] for c in contrasts], "Expression change\nfrom chromatin change")
    ax.set_ylabel("")
    ns.panel_label(ax, "b", x=-0.2)

    ax = fig.add_subplot(gs[2])
    pre_fams = ["naive_rna", "naive_rna+raw", "naive_rna+cs_quant", "raw", "cs_quant"]
    pre = [c for c in contrasts if c.endswith("_vs_Naive")]
    dot_panel(ax, sc, pre_fams, [f"C_prepattern_{c}" for c in pre],
              [CONTRAST_NAMES[c] for c in pre], "Later change from\nTN chromatin only")
    ax.set_ylabel("")
    ns.panel_label(ax, "c", x=-0.2)

    ax = fig.add_subplot(gs[3])
    piv = marks.pivot(index="mark", columns="contrast", values="spearman").loc[MARKS, contrasts]
    y = np.arange(len(MARKS))
    h = 0.26
    tones = ["#9cc0ec", ns.ACCENT, "#1b4f8f"]
    for i, c in enumerate(contrasts):
        ax.barh(y + (i - 1) * h, piv[c], height=h, color=tones[i], linewidth=0,
                label=CONTRAST_NAMES[c].replace("\n", " "))
    ax.set_yticks(y)
    ax.set_yticklabels(MARKS)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman ρ,\nsingle-mark model")
    ax.legend(loc="lower right", handlelength=0.8, borderaxespad=0)
    ax.set_title("Which mark tracks\nexpression change")
    ns.panel_label(ax, "d", x=-0.45)

    handles = legend_handles(fams) + legend_handles(["naive_rna", "naive_rna+raw",
                                                     "naive_rna+cs_quant"])
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0),
               columnspacing=1.2, handletextpad=0.2)
    ns.save(fig, FIG, "Fig2_prediction")


# ---------------------------------------------------------------------------
# Figure 3
# ---------------------------------------------------------------------------


def figure3():
    bdir = OUT / "bivalency"
    prom = pd.read_csv(bdir / "naive_promoters.tsv", sep="\t")
    q = pd.read_csv(bdir / "bivalent_quartiles.tsv", sep="\t")

    fig = plt.figure(figsize=(ns.DOUBLE * 0.78, 58 * ns.MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.45,
                          left=0.1, right=0.98, bottom=0.27, top=0.88)

    ax = fig.add_subplot(gs[0])
    classes = ["Neither", "H3K27me3 only", "H3K4me3 only", "Bivalent"]
    data = [prom.loc[prom["promoter_class"] == c, "P_cs_biv_score"].dropna() for c in classes]
    bp = ax.boxplot(data, widths=0.55, showfliers=False, patch_artist=True,
                    medianprops=dict(color=ns.INK, linewidth=0.8),
                    whiskerprops=dict(linewidth=0.5), capprops=dict(linewidth=0.5),
                    boxprops=dict(linewidth=0.5))
    for patch, c in zip(bp["boxes"], classes):
        patch.set_facecolor(ns.ACCENT if c == "Bivalent" else "#d9d8d4")
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels([f"{c}\n(n={len(d):,})" for c, d in zip(classes, data)],
                       rotation=30, ha="right")
    ax.set_ylabel("ChromState bivalency score (TN)")
    ax.set_title("Peak-called TN promoter class")
    conc = pd.read_csv(bdir / "concordance.tsv", sep="\t").set_index("score")
    auc = conc.loc["ChromState bivalency score (signal)", "auroc"]
    ax.text(0.03, 0.97, f"AUROC = {auc:.2f}", transform=ax.transAxes, va="top", fontsize=5.5)
    ns.panel_label(ax, "a", x=-0.32)

    ax = fig.add_subplot(gs[1])
    order = ["Q1 (K27me3-dominant)", "Q2", "Q3", "Q4 (K4me3-dominant)"]
    x = np.arange(len(order))
    for cond, dx in [("Arm", -0.12), ("Cl13", 0.12)]:
        sub = q.loc[q["condition"] == cond].set_index("quartile").loc[order]
        col = ns.CONDITION_COLORS[cond]
        ax.plot(x + dx, sub["median_log2fc"], "o", color=col, markerfacecolor="white",
                markeredgewidth=0.7, markersize=3.2)
        ax.plot(x + dx, sub["median_excess_log2fc"], "o", color=col, markersize=3.2)
        for xi, a, b in zip(x + dx, sub["median_log2fc"], sub["median_excess_log2fc"]):
            ax.plot([xi, xi], [a, b], color=col, linewidth=0.5)
    ax.axhline(0, color=ns.INK, linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(["Q1\nH3K27me3-\ndominant", "Q2", "Q3", "Q4\nH3K4me3-\ndominant"])
    ax.set_xlabel("TN bivalent promoters, quartile of H3K4me3 − H3K27me3")
    ax.set_ylabel("Median log2 fold change vs TN")
    ax.set_title("Naive bivalent balance anticipates later expression")
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=ns.CONDITION_COLORS["Arm"],
               markersize=3.2, label="TMEM"),
        Line2D([0], [0], marker="o", linestyle="", color=ns.CONDITION_COLORS["Cl13"],
               markersize=3.2, label="TEX"),
        Line2D([0], [0], marker="o", linestyle="", color=ns.INK_SECONDARY,
               markerfacecolor="white", markersize=3.2, label="Raw log2FC"),
        Line2D([0], [0], marker="o", linestyle="", color=ns.INK_SECONDARY, markersize=3.2,
               label="Beyond expression-matched genes"),
    ]
    ax.legend(handles=handles, loc="upper right", borderaxespad=0)
    ns.panel_label(ax, "b", x=-0.22)
    ns.save(fig, FIG, "Fig3_bivalency")


def main() -> None:
    ns.apply()
    figure1()
    figure2()
    figure3()
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
