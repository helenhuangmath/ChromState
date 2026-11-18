"""Step 12 (Fig. 6): promoter chromatin state and expression of CD8 T-cell marker genes.

For canonical naive/memory, effector, exhaustion and naive-bivalent genes:
  a  promoter (TSS +/- 1 kb) state composition in TN, TMEM and TEX: the mean of the ten
     200 bp bins' posteriors over all 15 states, summed into five classes for display;
  b  the promoter signal of each mark (mean reads per bin), so that the strength of
     marking -- which the five-class grouping deliberately hides -- stays visible;
  c  expression in the same three states.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import nature_style as ns
import numpy as np
import pandas as pd
from common import CONDITIONS, MARKS, MODEL_DIR, OUT, ensure_out
from matplotlib.patches import Patch

FIG = ensure_out("figures")

GENE_GROUPS = {
    "Naive / memory": ["Sell", "Ccr7", "Tcf7", "Lef1", "Il7r", "Bach2"],
    "Effector": ["Klrg1", "Cx3cr1", "Gzmb", "Zeb2", "Id2"],
    "Exhaustion": ["Tox", "Pdcd1", "Havcr2", "Entpd1", "Lag3", "Tigit", "Cd160", "Nr4a2"],
    "Bivalent in TN": ["Eomes", "Il18r1", "Itgb8", "Yes1", "Il2ra", "Epas1"],
}
# State classes, keyed by the mnemonic prefix of the fitted state labels.
CLASSES = [
    ("Active promoter", ("TssA", "TssFlnk"), "#d62728"),
    ("Bivalent", ("TssBiv", "BivFlnk"), "#e377c2"),
    ("Polycomb", ("ReprPC", "ReprPCWk"), "#7f7f7f"),
    ("H3K9me3", ("Het", "HetWk"), "#8c8fd0"),
    ("Background", ("Quies", "Unknown", "EnhA", "EnhWk", "Tx", "TxWk"), "#e8e8e8"),
]


def class_of(label: str) -> str:
    base = label.split("_")[0]
    for name, prefixes, _ in CLASSES:
        if base in prefixes:
            return name
    return "Background"


def main() -> None:
    ns.apply()
    feats = pd.read_csv(OUT / "genes" / "gene_features.tsv.gz", sep="\t")
    genes = pd.read_csv(OUT / "rna" / "genes.tsv", sep="\t")
    expr = pd.read_csv(OUT / "rna" / "expression.tsv", sep="\t", index_col="gene_id")
    labels = pd.read_csv(OUT / MODEL_DIR / "state_labels.tsv", sep="\t")["label"].tolist()

    # one Ensembl id per symbol: the most expressed
    g = genes.merge(expr[["baseMean"]], left_on="gene_id", right_index=True, how="left")
    g = g.sort_values("baseMean", ascending=False).drop_duplicates("name").set_index("name")

    rows, comp, rna, sig, missing = [], [], [], [], []
    for group, names in GENE_GROUPS.items():
        for name in names:
            if name not in g.index:
                missing.append(name)
                continue
            gid = g.loc[name, "gene_id"]
            per_cond = []
            for cond in CONDITIONS:
                r = feats.loc[(feats["gene_id"] == gid) & (feats["condition"] == cond)]
                if r.empty:
                    break
                v = r.iloc[0]
                per_cond.append({c[0]: sum(float(v[f"P_cs_{lbl}"]) for lbl in labels
                                           if class_of(lbl) == c[0]) for c in CLASSES})
            if len(per_cond) < 3:
                missing.append(name)
                continue
            rows.append((group, name))
            comp.append(per_cond)
            sig.append([[float(np.expm1(feats.loc[(feats["gene_id"] == gid) & (feats["condition"] == c),
                                                   f"P_raw_{m}"].iloc[0])) for c in CONDITIONS]
                        for m in MARKS])
            rna.append([expr.loc[gid, c] for c in CONDITIONS])
    if missing:
        print("not found:", missing)

    table = []
    for (group, name), pc, ex in zip(rows, comp, rna):
        for cond, d, e in zip(CONDITIONS, pc, ex):
            table.append({"group": group, "gene": name, "condition": ns.CONDITION_NAMES[cond],
                          **{k: round(val, 3) for k, val in d.items()}, "log2_expression": round(e, 2)})
    for i, s_ in enumerate(sig):
        for ci in range(3):
            for mi, m in enumerate(MARKS):
                table[i * 3 + ci][f"promoter_{m}_reads"] = round(s_[mi][ci], 1)
    pd.DataFrame(table).to_csv(OUT / "importance" / "marker_genes.tsv", sep="\t", index=False)

    n = len(rows)
    fig = plt.figure(figsize=(ns.DOUBLE, (26 + 4.6 * n) * ns.MM))
    gs = fig.add_gridspec(1, 5, width_ratios=[0.75, 0.75, 0.75, 1.5, 0.95], wspace=0.14,
                          left=0.08, right=0.965, bottom=0.12, top=0.92)
    y = np.arange(n)
    for ci, cond in enumerate(CONDITIONS):
        ax = fig.add_subplot(gs[ci])
        left = np.zeros(n)
        for cname, _, color in CLASSES:
            vals = np.array([pc[ci][cname] for pc in comp])
            ax.barh(y, vals, left=left, height=0.72, color=color, linewidth=0)
            left += vals
        ax.set_xlim(0, 1)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_xticks([0, 0.5, 1])
        ax.set_title(ns.CONDITION_NAMES[cond])
        if ci == 0:
            ax.set_yticks(y)
            ax.set_yticklabels([name for _, name in rows], fontstyle="italic")
            ns.panel_label(ax, "a", x=-0.62, y=1.03)
        else:
            ax.set_yticks([])
        if ci == 1:
            ax.set_xlabel("Promoter state share")
        for spine in ("left",):
            ax.spines[spine].set_visible(ci == 0)
        # group separators
        bounds = np.cumsum([sum(1 for gname, _ in rows if gname == grp) for grp in GENE_GROUPS])
        for b in bounds[:-1]:
            ax.axhline(b - 0.5, color=ns.INK, linewidth=0.5)

    # b  promoter signal per mark, log-coloured within each mark, value printed
    ax = fig.add_subplot(gs[3])
    S = np.array(sig)                      # genes x marks x conditions
    grid = S.reshape(n, len(MARKS) * 3)    # columns: mark-major, condition-minor
    # log colour scaled to each mark's maximum, floored at 20 reads so that a mark with
    # almost no signal (H3K9me3 here) stays pale instead of saturating at ~1 read
    norm = np.log1p(grid) / np.log1p(np.repeat(np.maximum(S.max(axis=(0, 2)), 20.0), 3))[None, :]
    ax.imshow(norm, cmap=ns.mpl.colors.LinearSegmentedColormap.from_list(
        "sig", ["#ffffff", "#9ec5f4", "#3987e5"]), vmin=0, vmax=1, aspect="auto")
    # the read counts behind the colours are in out/importance/marker_genes.tsv
    for x in (2.5, 5.5, 8.5):
        ax.axvline(x, color=ns.INK, linewidth=0.5)
    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels(["N", "M", "E"] * len(MARKS), fontsize=6)
    for mi, m in enumerate(MARKS):
        ax.text(mi * 3 + 1, -0.9, m, ha="center", va="bottom", fontsize=ns.FS)
    ax.set_yticks([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Promoter signal (N, TN; M, TMEM; E, TEX)")
    for b in bounds[:-1]:
        ax.axhline(b - 0.5, color=ns.INK, linewidth=0.5)
    ns.panel_label(ax, "b", x=-0.03, y=1.03)

    ax = fig.add_subplot(gs[4])
    for ci, cond in enumerate(CONDITIONS):
        ax.plot([r[ci] for r in rna], y + (ci - 1) * 0.22, "o", markersize=2.8,
                color=ns.CONDITION_COLORS[cond], label=ns.CONDITION_NAMES[cond])
    for b in bounds[:-1]:
        ax.axhline(b - 0.5, color=ns.INK, linewidth=0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks([])
    ax.set_xlabel("Expression (log2)")
    ax.set_title("RNA")
    ax.grid(axis="x", linewidth=0.3, color="#e6e6e6")
    ax.set_axisbelow(True)
    ns.panel_label(ax, "c", x=-0.08, y=1.03)
    # group names on the right
    starts = np.concatenate([[0], bounds[:-1]])
    for grp, s0, s1 in zip(GENE_GROUPS, starts, bounds):
        ax.text(1.02, (s0 + s1 - 1) / 2, grp, transform=ax.get_yaxis_transform(),
                rotation=270, va="center", ha="left", fontsize=ns.FS)
    ax.legend(loc="lower right", handletextpad=0.2, borderaxespad=0.2)

    handles = [Patch(facecolor=c, label=name) for name, _, c in CLASSES]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.45, 0.0),
               handlelength=1.0, columnspacing=1.2)
    ns.save(fig, FIG, "Fig6_marker_genes", groups=False)     # one figure: the panels share the gene rows
    print(f"{n} genes -> {FIG / 'Fig6_marker_genes.pdf'}")


if __name__ == "__main__":
    main()
