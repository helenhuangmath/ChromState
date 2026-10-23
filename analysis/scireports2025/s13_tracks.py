"""Step 13 (Fig. 7): genome-browser tracks for selected genes.

For each gene, the window spans the gene plus 10 kb on each side. For each cell state
(TN, TMEM, TEX) it shows:
- the four marks, as mean reads per 200 bp bin (the exact model input; one shared y-axis
  per mark, so heights compare across cell states);
- RNA-seq coverage (pooled, scaled);
- the ChromState state of every bin (the most probable state), as a coloured ribbon.
The promoter window used in the gene-level analyses (TSS +/- 1 kb) is shaded.

One PDF per gene: out/figures/Fig7_tracks_<gene>.pdf. Text is set at 10 pt (12 pt for the gene name), larger than Nature's minimum, for legibility.

Usage: python s13_tracks.py [GENE ...]      (default: Pdcd1 Tigit Tox Tcf7)
"""

from __future__ import annotations

import pickle
import sys
import textwrap

import matplotlib.pyplot as plt
import nature_style as ns
import numpy as np
import pandas as pd
import pyBigWig
from common import BIN_SIZE, CONDITIONS, DATA, MARKS, MODEL_DIR, OUT, ensure_out, load_dataset
from matplotlib.patches import Patch, Rectangle

from chromstate.model.annotate import full_state_name

FIG = ensure_out("figures")
FLANK = 10000   # gene body +/- 10 kb
FS = 10.0       # all track text
FS_SMALL = 9.0  # y-axis range labels
FS_GENE = 12.0  # gene name
TRACK_MARKS = ["H3K27ac", "H3K4me3", "H3K27me3", "H3K9me3"]  # display order (top to bottom)
# Cell-state colours for this figure only: TN green, TMEM blue, TEX orange.
COND_COLORS = {"Naive": "#2ca25f", "Arm": "#2a78d6", "Cl13": "#f28e2b"}
# Chromatin-state colours: active = reds, repressive (H3K27me3) = blues, heterochromatin
# (H3K9me3) = purples, background = light grey. Keyed by full label, then by mnemonic.
STATE_PALETTE = {
    "Prom_TssA": "#e41a1c", "Prom_TssFlnk": "#fb7a5c", "Prom_TssBiv": "#f4a6c0",
    "Dist_TssA": "#8b0000", "Dist_TssFlnk": "#fdbf6f", "Dist_TssBiv": "#d9a0a0",
    "ReprPC_1": "#5aa9e6", "ReprPC_2": "#0b3d91", "ReprPCWk": "#b3d4ee",
    "Het": "#6a1b9a", "HetWk": "#c9a8e0", "Quies": "#ececec",
}
TSS_WIN = 2000  # an active state is promoter-proximal within this distance of an annotated TSS
ACTIVE = ("TssA", "TssFlnk", "TssBiv")
# The model has only four marks, so "TSS" in a state name means "H3K27ac/H3K4me3 high", not
# "near a TSS". Names below say what the marks show; position decides promoter vs distal.
MARK_NAMES = {
    "TssA": "active (H3K27ac+H3K4me3)",
    "TssFlnk": "weak active (H3K27ac+H3K4me3, lower)",
    "TssBiv": "bivalent (H3K4me3+H3K27me3)",
}
RNA_TRACK = DATA / "RNA" / "Wig" / "pool_RNA_{}.bw"


def tss_positions() -> dict:
    """Annotated TSS coordinates per chromosome (sorted), from the gene table."""
    genes = pd.read_csv(OUT / "rna" / "genes.tsv", sep="\t")
    tss = np.where(genes["strand"] == "-", genes["end"] - 1, genes["start"])
    return {c: np.sort(tss[(genes["chrom"] == c).to_numpy()]) for c in genes["chrom"].unique()}


def near_tss(chrom: str, bins: np.ndarray, tss: dict) -> np.ndarray:
    """True for bins whose centre lies within TSS_WIN of an annotated TSS."""
    t = tss.get(chrom)
    if t is None or len(t) == 0:
        return np.zeros(len(bins), bool)
    pos = bins * BIN_SIZE + BIN_SIZE // 2
    i = np.searchsorted(t, pos)
    left = np.abs(pos - t[np.clip(i - 1, 0, len(t) - 1)])
    right = np.abs(pos - t[np.clip(i, 0, len(t) - 1)])
    return np.minimum(left, right) <= TSS_WIN


def display_key(label: str, near: bool) -> str:
    """State key for display: active states are split into promoter-proximal and distal."""
    base = label.split("_")[0]
    return f"{'Prom' if near else 'Dist'}_{base}" if base in ACTIVE else label


def key_name(key: str) -> str:
    """Legend / table name for a display key."""
    if key.startswith(("Prom_", "Dist_")):
        where = "promoter-proximal" if key.startswith("Prom_") else "distal, enhancer-like"
        return f"{where[0].upper()}{where[1:]}: {MARK_NAMES[key.split('_', 1)[1]]}"
    return "Quiescent, low signal (background)" if key.startswith("Quies") else full_state_name(key)


def gene_window(name: str):
    genes = pd.read_csv(OUT / "rna" / "genes.tsv", sep="\t")
    expr = pd.read_csv(OUT / "rna" / "expression.tsv", sep="\t", index_col="gene_id")
    g = genes.loc[genes["name"] == name].merge(expr[["baseMean"]], left_on="gene_id",
                                                 right_index=True)
    g = g.sort_values("baseMean").iloc[-1]
    lo = (int(g["start"]) - FLANK) // BIN_SIZE
    hi = (int(g["end"]) + FLANK) // BIN_SIZE + 1
    tss = int(g["end"]) - 1 if g["strand"] == "-" else int(g["start"])
    return g, lo, hi, tss, expr.loc[g["gene_id"], CONDITIONS].to_numpy()


def gene_tracks(name: str, hmm, labels, colors):
    g, lo, hi, tss, rna_level = gene_window(name)
    ds = load_dataset([g["chrom"]])
    states = hmm.decode(ds.X, ds.size_factors, ds.chains)
    n = len(ds.X) // len(CONDITIONS)
    z = np.load(OUT / "binned" / f"{g['chrom']}.npz")
    sig = {c: z["X"][list(z["conditions"]).index(c)][lo:hi] for c in CONDITIONS}  # bins x marks
    near = near_tss(g["chrom"], np.arange(lo, hi), tss_positions())
    st = {c: np.array([display_key(labels[k], nr) for k, nr in
                       zip(states[ci * n + lo: ci * n + hi], near)])
          for ci, c in enumerate(CONDITIONS)}
    rna = {}
    for c in CONDITIONS:
        bw = pyBigWig.open(str(RNA_TRACK).format(c))
        vals = bw.stats(g["chrom"], lo * BIN_SIZE, hi * BIN_SIZE, nBins=hi - lo, type="mean",
                        exact=True)
        bw.close()
        rna[c] = np.nan_to_num(np.array(vals, dtype=float))
    x = (np.arange(lo, hi) * BIN_SIZE) / 1e3
    return dict(gene=g, tss=tss, x=x, sig=sig, states=st, rna=rna, rna_level=rna_level)


def draw_gene(fig, gs_col, d, labels, colors, panel):
    g, x = d["gene"], d["x"]
    rows = (["gene"] + [(m, c) for m in TRACK_MARKS for c in CONDITIONS]
            + [("RNA", c) for c in CONDITIONS] + [("state", c) for c in CONDITIONS])
    heights = [0.9] + [1.0] * (len(TRACK_MARKS) * 3) + [1.0] * 3 + [0.45] * 3
    sub = gs_col.subgridspec(len(rows), 1, height_ratios=heights, hspace=0.18)
    axes = [fig.add_subplot(sub[i]) for i in range(len(rows))]
    tss_kb = d["tss"] / 1e3
    width = BIN_SIZE / 1e3

    for ax, row in zip(axes, rows):
        ax.set_xlim(x[0], x[-1] + width)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.axvspan(tss_kb - 1, tss_kb + 1, color="#f2f2f2", zorder=0, linewidth=0)
        if row == "gene":
            ax.set_ylim(0, 1)
            ax.plot([g["start"] / 1e3, g["end"] / 1e3], [0.45, 0.45], color=ns.INK,
                    linewidth=1.6, solid_capstyle="butt")
            ax.annotate("", xy=(tss_kb - 1.6 if g["strand"] == "-" else tss_kb + 1.6, 0.45),
                        xytext=(tss_kb, 0.45),
                        arrowprops=dict(arrowstyle="->", linewidth=0.6, color=ns.INK))
            ax.text((g["start"] + g["end"]) / 2e3, 0.9, g["name"], ha="center", va="center",
                    fontsize=FS_GENE, fontstyle="italic")
            ax.axis("off")
            continue
        kind, cond = row
        ax.set_yticks([])
        ax.tick_params(axis="x", labelbottom=False, length=0)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_linewidth(0.4)
        if kind == "state":
            for xi, s in zip(x, d["states"][cond]):
                ax.add_patch(Rectangle((xi, 0), width, 1, color=colors[s], linewidth=0))
            ax.set_ylim(0, 1)
            ax.spines["left"].set_visible(False)
            ax.text(-0.01, 0.5, f"States {ns.CONDITION_NAMES[cond]}", transform=ax.transAxes,
                    ha="right", va="center", fontsize=FS)
            continue
        if kind == "RNA":
            vals = d["rna"][cond]
            ymax = max(max(d["rna"][c].max() for c in CONDITIONS), 1e-9)
            color = "#4d4d4d"
            label = f"RNA {ns.CONDITION_NAMES[cond]}"
        else:
            j = MARKS.index(kind)
            vals = d["sig"][cond][:, j]
            ymax = max(max(d["sig"][c][:, j].max() for c in CONDITIONS), 1.0)
            color = COND_COLORS[cond]
            label = f"{kind} {ns.CONDITION_NAMES[cond]}"
        ax.bar(x, vals, width=width, align="edge", color=color, linewidth=0)
        ax.set_ylim(0, ymax * 1.05)
        ax.text(-0.01, 0.5, label, transform=ax.transAxes, ha="right", va="center",
                fontsize=FS)
        if cond == CONDITIONS[0]:
            ax.text(1.0, 0.98, f"0–{ymax:.0f}", transform=ax.transAxes, ha="right",
                    va="top", fontsize=FS_SMALL)

    ax = axes[-1]
    ax.tick_params(axis="x", labelbottom=True, length=2, labelsize=FS)
    ax.spines["bottom"].set_visible(True)
    ax.set_xlabel(f"{g['chrom']} position (kb)", fontsize=FS)
    return axes


def state_color(key: str, label_colors: dict) -> str:
    return STATE_PALETTE.get(key, label_colors.get(key, "#bdbdbd"))


def state_summary() -> None:
    """Genome-wide state summary per condition, with active states split by position.

    Reads the K15 segmentation BEDs and expands them to 200 bp bins. Numbered states are
    pooled by mnemonic (ReprPC_1/2 -> ReprPC); active states are split into
    promoter-proximal (within TSS_WIN of an annotated TSS) and distal. Writes
    out/tracks/state_summary.tsv (segments, Mb, % genome) and
    out/tracks/differential_regions.tsv (runs of bins where the promoter-proximal or
    distal active class differs between conditions, with RNA-independent coordinates).
    """
    tdir = ensure_out("tracks")
    tss = tss_positions()
    per = {}
    for c in CONDITIONS:
        b = pd.read_csv(OUT / "segmentation" / f"{c}_chromstate_K15_dense.bed", sep="\t",
                        header=None, usecols=[0, 1, 2, 3], names=["chrom", "start", "end", "state"])
        n = ((b["end"] - b["start"]) // BIN_SIZE).to_numpy()
        df = pd.DataFrame({"chrom": np.repeat(b["chrom"].to_numpy(), n),
                           "bin": np.concatenate([np.arange(s // BIN_SIZE, e // BIN_SIZE)
                                                  for s, e in zip(b["start"], b["end"])]),
                           "state": np.repeat(b["state"].to_numpy(), n)})
        key = np.empty(len(df), dtype=object)
        for ch, idx in df.groupby("chrom").indices.items():
            near = near_tss(ch, df["bin"].to_numpy()[idx], tss)
            key[idx] = [display_key(st, nr) for st, nr in zip(df["state"].to_numpy()[idx], near)]
        df["cls"] = pd.Series(key, index=df.index).str.replace(r"_(\d+)$", "", regex=True)
        per[c] = df

    rows = []
    for c, df in per.items():
        run = (df["cls"].ne(df["cls"].shift()) | df["chrom"].ne(df["chrom"].shift())
               | (df["bin"] != df["bin"].shift() + 1))
        df["run"] = run.cumsum()
        for k, g in df.groupby("cls"):
            rows.append(dict(condition=ns.CONDITION_NAMES[c], state=key_name(k),
                             segments=g["run"].nunique(), mb=len(g) * BIN_SIZE / 1e6,
                             pct_genome=100 * len(g) / len(df)))
    summ = pd.DataFrame(rows)
    summ.to_csv(tdir / "state_summary.tsv", sep="\t", index=False, float_format="%.3f")
    print(summ.pivot(index="state", columns="condition", values="segments")
          [["TN", "TMEM", "TEX"]].to_string())

    m = per[CONDITIONS[0]][["chrom", "bin", "cls"]].rename(columns={"cls": CONDITIONS[0]})
    for c in CONDITIONS[1:]:
        m = m.merge(per[c][["chrom", "bin", "cls"]].rename(columns={"cls": c}),
                    on=["chrom", "bin"])
    is_act = m[CONDITIONS].apply(lambda col: col.str.startswith(("Prom_", "Dist_")))
    diff = m[is_act.any(axis=1) & (m[CONDITIONS].nunique(axis=1) > 1)].sort_values(["chrom", "bin"])
    key = diff[CONDITIONS].astype(str).agg("|".join, axis=1)
    run = ((diff["chrom"] != diff["chrom"].shift()) | (diff["bin"] != diff["bin"].shift() + 1)
           | (key != key.shift())).cumsum()
    reg = diff.groupby(run).agg(chrom=("chrom", "first"), start=("bin", "min"), end=("bin", "max"),
                                TN=("Naive", "first"), TMEM=("Arm", "first"), TEX=("Cl13", "first"))
    reg["bins"] = reg["end"] - reg["start"] + 1
    reg["start"] *= BIN_SIZE
    reg["end"] = (reg["end"] + 1) * BIN_SIZE
    reg.to_csv(tdir / "differential_regions.tsv", sep="\t", index=False)
    big = reg[reg["bins"] >= 5]  # >= 1 kb
    print(f"\n{len(reg)} differential active regions; {len(big)} span >= 1 kb")
    print(big.groupby(["TN", "TMEM", "TEX"]).size().sort_values(ascending=False).head(12)
          .to_string())


def main() -> None:
    names = sys.argv[1:] or ["Pdcd1", "Tigit", "Tox", "Tcf7"]
    state_summary()
    ns.apply()
    with open(OUT / MODEL_DIR / "model.pkl", "rb") as fh:
        hmm = pickle.load(fh)["hmm"]
    lab = pd.read_csv(OUT / MODEL_DIR / "state_labels.tsv", sep="\t")
    labels = lab["label"].tolist()
    label_colors = dict(zip(labels, lab["color"]))

    for nm in names:
        d = gene_tracks(nm, hmm, labels, None)
        colors = {k: state_color(k, label_colors) for k in set(np.concatenate(list(d["states"].values())))
                  } | {k: state_color(k.split("_")[0], label_colors) for k in ()}
        # non-active labels fall back to their mnemonic colour
        for k in colors:
            if k not in STATE_PALETTE:
                colors[k] = STATE_PALETTE.get(k.split("_")[0], label_colors.get(k, "#bdbdbd"))
        # Tracks on the left, legend in its own column on the right, so it never covers a
        # track or the coordinate axis.
        fig = plt.figure(figsize=(ns.DOUBLE, 235 * ns.MM))
        outer = fig.add_gridspec(1, 1, left=0.2, right=0.64, bottom=0.06, top=0.98)
        draw_gene(fig, outer[0, 0], d, labels, colors, "")

        order = ["Prom_TssA", "Prom_TssFlnk", "Prom_TssBiv", "Dist_TssA", "Dist_TssFlnk",
                 "Dist_TssBiv"]
        used = set(np.concatenate(list(d["states"].values())))
        used = [k for k in order if k in used] + sorted(k for k in used if k not in order)
        handles, seen = [], set()
        for k in used:
            name = key_name(k)
            if name in seen:
                continue
            seen.add(name)
            handles.append(Patch(facecolor=colors[k], edgecolor="#bdbdbd", linewidth=0.3,
                                 label=textwrap.fill(name, 30)))
        handles.append(Patch(facecolor="#f2f2f2", label="Promoter window (TSS ± 1 kb)"))
        # Anchored at the bottom, level with the state tracks the colours describe.
        fig.legend(handles=handles, loc="lower left", ncol=1, bbox_to_anchor=(0.655, 0.045),
                   handlelength=1.2, fontsize=FS, frameon=False, labelspacing=0.6)
        out = f"Fig7_tracks_{nm}"
        ns.save(fig, FIG, out)
        g = d["gene"]
        print(f"{nm}: {g['chrom']}:{g['start']}-{g['end']} ({g['strand']}), "
              f"RNA log2 TN/TMEM/TEX = {np.round(d['rna_level'], 1)} -> {out}.pdf")


if __name__ == "__main__":
    main()
