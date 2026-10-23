"""Step 10: Nature-style figures for ordering (step 9) and importance (step 8).

Fig. 4  Cross-lagged asymmetry: does each mark / state in TN anticipate later expression
        change (chromatin leads) or follow TN expression (RNA leads)?
Fig. 5  Which states matter (unique vs marginal), and how much of the mark-to-expression
        relationship is additive versus combinatorial.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import nature_style as ns
import numpy as np
import pandas as pd
from common import MARKS, MODEL_DIR, OUT, ensure_out

from chromstate.model.annotate import full_state_name

FIG = ensure_out("figures")


def state_descriptions() -> dict:
    """State label -> display name with the marks it is actually enriched for.

    The automatic names under-call weak states on real data, so every figure label adds
    each mark whose state mean is at least 1.5x its genome-wide (occupancy-weighted TN)
    mean, with the fold enrichment; states with none are marked as near background.
    """
    import pickle

    with open(OUT / MODEL_DIR / "model.pkl", "rb") as fh:
        hmm = pickle.load(fh)["hmm"]
    labels = pd.read_csv(OUT / MODEL_DIR / "state_labels.tsv", sep="\t")["label"].tolist()
    occ = pd.read_csv(OUT / MODEL_DIR / "occupancy_genome.tsv", sep="\t",
                      index_col=0).loc["Naive", labels].to_numpy()
    M = hmm.emissions.mean_matrix()
    fold = M / ((occ[:, None] * M).sum(axis=0) / occ.sum())
    out = {}
    for k, lbl in enumerate(labels):
        parts = [f"{m.replace('H3', '')} {f:.0f}×" if f >= 10 else f"{m.replace('H3', '')} {f:.1f}×"
                 for m, f in zip(MARKS, fold[k]) if f >= 1.5]
        out[lbl] = f"{full_state_name(lbl)} [{', '.join(parts) if parts else 'near background'}]"
    return out
LATER_COLORS = {"TMEM": ns.CONDITION_COLORS["Arm"], "TEX": ns.CONDITION_COLORS["Cl13"]}
TASK_TONES = {"Level (TN)": "#9cc0ec", "Change (TMEM vs TN)": ns.ACCENT,
              "Change (TEX vs TN)": "#1b4f8f"}


def asym_panel(ax, sub: pd.DataFrame, order, names, title):
    """Measurement-corrected asymmetry (filled) where a reliability exists, else raw (open)."""
    y = np.arange(len(order))
    for later, dy in [("TMEM", -0.17), ("TEX", 0.17)]:
        d = sub.loc[sub["later"] == later].set_index("variable").loc[order]
        col = LATER_COLORS[later]
        has = d["asymmetry_corrected"].notna() if "asymmetry_corrected" in d else pd.Series(False, d.index)
        val = np.where(has, d.get("asymmetry_corrected"), d["asymmetry"])
        lo = np.where(has, d.get("asym_lo_corrected"), d["asym_lo"])
        hi = np.where(has, d.get("asym_hi_corrected"), d["asym_hi"])
        ax.hlines(y + dy, lo, hi, color=col, linewidth=0.7)
        ax.plot(val[has.to_numpy()], (y + dy)[has.to_numpy()], "o", color=col, markersize=3,
                label=later)
        ax.plot(val[~has.to_numpy()], (y + dy)[~has.to_numpy()], "o", color=col, markersize=3,
                markerfacecolor="white", markeredgewidth=0.7)
    ax.axvline(0, color=ns.INK, linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("|chromatin → RNA| − |RNA → chromatin|\n(standardised coefficients)")
    ax.set_title(title)
    lo_, hi_ = ax.get_xlim()
    lim = max(abs(lo_), abs(hi_))
    ax.set_xlim(-lim, lim)
    ax.text(0.02, -0.16, "← RNA leads", transform=ax.transAxes, ha="left", va="top",
            fontsize=5.5)
    ax.text(0.98, -0.16, "chromatin leads →", transform=ax.transAxes, ha="right", va="top",
            fontsize=5.5)
    ax.grid(axis="x", linewidth=0.3, color="#e6e6e6")
    ax.set_axisbelow(True)


def figure4():
    t = pd.read_csv(OUT / "cross_lag" / "cross_lag.tsv", sep="\t")
    rel = pd.read_csv(OUT / "reliability" / "reliability.tsv", sep="\t").set_index("mark")
    rel_txt = ", ".join(f"{m} {rel.loc[m, 'pooled_reliability'].min():.2f}"
                        for m in ["H3K4me3", "H3K27ac", "H3K27me3", "H3K9me3", "RNA"])
    marks = t.loc[t["kind"] == "mark"]
    states = t.loc[t["kind"] == "state"]

    fig = plt.figure(figsize=(ns.DOUBLE, 92 * ns.MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1], wspace=1.25, left=0.12, right=0.98,
                          bottom=0.24, top=0.9)
    ax = fig.add_subplot(gs[0])
    order = list(dict.fromkeys(marks["variable"]))
    asym_panel(ax, marks, order, order, "Histone marks in TN")
    ax.legend(loc="lower right", handletextpad=0.2, borderaxespad=0.2)
    ns.panel_label(ax, "a", x=-0.45)

    ax = fig.add_subplot(gs[1])
    order = list(dict.fromkeys(states["variable"]))
    desc = state_descriptions()
    names = [desc[v.replace(" (promoter state)", "")] for v in order]
    asym_panel(ax, states, order, names, "ChromState promoter states in TN")
    ns.panel_label(ax, "b", x=-0.75)

    fig.text(0.01, 0.015,
             "Filled: corrected for measurement error using replicate reliabilities "
             f"({rel_txt}); open: uncorrected (background states). Bars: 95% bootstrap CI over genes.",
             fontsize=5.5, ha="left", va="bottom")
    ns.save(fig, FIG, "Fig4_cross_lag")


def figure5():
    st = pd.read_csv(OUT / "importance" / "state_importance.tsv", sep="\t")
    combo = pd.read_csv(OUT / "importance" / "additive_vs_combinatorial.tsv", sep="\t")
    gain = pd.read_csv(OUT / "importance" / "combinatorial_gain.tsv", sep="\t")
    tasks = list(TASK_TONES)

    fig = plt.figure(figsize=(ns.DOUBLE, 100 * ns.MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.8, 0.8], wspace=1.15, left=0.25,
                          right=0.97, bottom=0.2, top=0.9)

    # a  unique importance of each state
    ax = fig.add_subplot(gs[0])
    lvl = st.loc[st["task"] == tasks[0]].set_index("state")["unique_r2_drop"]
    order = lvl.sort_values(ascending=False).index.tolist()
    y = np.arange(len(order))
    h = 0.27
    for i, task in enumerate(tasks):
        d = st.loc[st["task"] == task].set_index("state").loc[order]
        ax.barh(y + (i - 1) * h, d["unique_r2_drop"], height=h, color=TASK_TONES[task],
                linewidth=0, label=task)
    ax.set_yticks(y)
    desc = state_descriptions()
    ax.set_yticklabels([desc[s] for s in order])
    ax.invert_yaxis()
    ax.axvline(0, color=ns.INK, linewidth=0.5)
    ax.set_xlabel("Drop in held-out R² when the state is permuted\n(unique contribution)")
    ax.set_title("Which states matter")
    ax.legend(loc="lower right", handlelength=0.8, borderaxespad=0.2)
    ns.panel_label(ax, "a", x=-1.25)

    # b  single marks vs additive vs full vs states
    ax = fig.add_subplot(gs[1])
    models = ["H3K4me3 alone", "H3K27ac alone", "H3K27me3 alone", "H3K9me3 alone",
              "Additive (no interactions)", "Full interactions", "ChromState states"]
    names = ["H3K4me3 alone", "H3K27ac alone", "H3K27me3 alone", "H3K9me3 alone",
             "All marks, additive", "All marks, interacting", "ChromState states"]
    y = np.arange(len(models))
    for i, task in enumerate(tasks):
        d = combo.loc[combo["task"] == task].set_index("model").loc[models]
        yy = y + (i - 1) * 0.25
        ax.hlines(yy, d["rho_lo"], d["rho_hi"], color=TASK_TONES[task], linewidth=0.7)
        ax.plot(d["spearman"], yy, "o", color=TASK_TONES[task], markersize=2.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman ρ, held-out chromosomes")
    ax.set_title("Single, additive\nor combined")
    ax.grid(axis="x", linewidth=0.3, color="#e6e6e6")
    ax.set_axisbelow(True)
    ns.panel_label(ax, "b", x=-0.75)

    # c  gain over the additive model from each interaction
    ax = fig.add_subplot(gs[2])
    rows = [m for m in gain["model"].unique()]
    names = [m.replace("Additive + ", "").replace(" x ", " × ") for m in rows]
    names = [n if n not in ("Full interactions", "ChromState states") else
             {"Full interactions": "All interactions", "ChromState states": "ChromState states"}[n]
             for n in names]
    y = np.arange(len(rows))
    for i, task in enumerate(tasks):
        d = gain.loc[gain["task"] == task].set_index("model").loc[rows]
        yy = y + (i - 1) * 0.25
        ax.hlines(yy, d["lo"], d["hi"], color=TASK_TONES[task], linewidth=0.7)
        ax.plot(d["gain_over_additive"], yy, "o", color=TASK_TONES[task], markersize=2.8)
    ax.axvline(0, color=ns.INK, linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Gain in ρ over\nthe additive model")
    ax.set_title("Combinatorial\ncontribution")
    ax.grid(axis="x", linewidth=0.3, color="#e6e6e6")
    ax.set_axisbelow(True)
    ns.panel_label(ax, "c", x=-0.6)

    ns.save(fig, FIG, "Fig5_importance")


def main() -> None:
    ns.apply()
    figure4()
    figure5()
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
