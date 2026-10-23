"""Step 15 (Figs. 8 and 9): chromatin programmes that RNA alone cannot show.

a  Poised versus silent. Genes silent in TN are split by their TN promoter state class
   (active / poised / repressed / background, from step 14). For each class: the share
   of genes that are later up in TMEM and in TEX, with Wilson 95% intervals and a Fisher
   odds ratio against unmarked (background) promoters. RNA cannot tell these classes
   apart -- every gene in them is silent.
b  Chromatin ahead of RNA. Per gene and contrast, the promoter shift
   (gain in active share minus gain in poised + repressed share) against the RNA log2
   fold change. Genes whose promoter moved by more than half its composition while the
   RNA did not follow are listed, in both directions. With three cell states and pooled
   chromatin this marks discordance, not order: "ahead" needs a time course.
c  Genes grouped by promoter trajectory TN > TMEM > TEX, two ways:
   (i)  exact pattern of the dominant class per condition (A active, P poised,
        R repressed, - background), which keeps rare patterns such as A > P > A (Tox);
   (ii) k-means on three equally weighted blocks: promoter state composition in each
        condition, the change from TN in promoter mark strength, and the change from
        TN in distal enhancer input within 200 kb -- so genes that stay "active" but
        gain or lose strength (Pdcd1, Tcf7) separate from genes that do not move.

Outputs: out/programs/*.tsv, out/figures/Fig8_poised_genes.pdf,
out/figures/Fig9_chromatin_programs.pdf.
"""

from __future__ import annotations

import sys
import textwrap

import matplotlib.pyplot as plt
import nature_style as ns
import numpy as np
import pandas as pd
from common import CONDITIONS, MARKS, OUT, ensure_out
from matplotlib.colors import LinearSegmentedColormap
from s12_marker_genes import GENE_GROUPS
from scipy.stats import fisher_exact, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from chromstate.enhancers import ring_columns

FIG = ensure_out("figures")
RES = ensure_out("programs")
SHORT = ns.CONDITION_NAMES                      # Naive -> TN, Arm -> TMEM, Cl13 -> TEX
CLASSES = ["active", "poised", "repressed", "background"]
LETTER = {"active": "A", "poised": "P", "repressed": "R", "background": "-"}
CLASS_COLORS = {"active": "#d62728", "poised": "#e377c2", "repressed": "#7f7f7f",
                "background": "#d9d9d9"}

SILENT_LOG2 = 5.0          # TN log2 expression below this counts as silent (~48% of genes)
UP_LFC, UP_PADJ = 1.0, 0.05
SHIFT_MIN = 0.5            # promoter composition moved by at least half
WARN_EXPR = 10.0           # "RNA still high": log2 expression in the later state
N_CLUSTERS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
MIN_PATTERN = 30
KEY_GENES = [g for genes in GENE_GROUPS.values() for g in genes] + ["Tox2", "S1pr1", "Id3", "Prdm1"]
SEED = 0


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def load():
    classes = pd.read_csv(OUT / "enhancers" / "state_classes.tsv", sep="\t").set_index("label")["class"]
    gf = pd.read_csv(OUT / "genes" / "gene_features.tsv.gz", sep="\t")
    expr = pd.read_csv(OUT / "rna" / "expression.tsv", sep="\t", index_col="gene_id")
    de = {c: pd.read_csv(OUT / "rna" / f"de_{c}_vs_Naive.tsv", sep="\t", index_col="gene_id")
          for c in ("Arm", "Cl13")}
    genes = pd.read_csv(OUT / "rna" / "genes.tsv", sep="\t").drop_duplicates("gene_id")
    names = genes.set_index("gene_id")["name"]

    comp, marks = {}, {}
    for cond in CONDITIONS:
        sub = gf.loc[gf["condition"] == cond].set_index("gene_id")
        comp[cond] = pd.DataFrame({c: sub[[f"P_cs_{s}" for s in classes.index[classes == c]]].sum(axis=1)
                                   for c in CLASSES})
        marks[cond] = sub[[f"P_raw_{m}" for m in MARKS]].rename(columns=lambda c: c[len("P_raw_"):])

    ef = pd.read_csv(OUT / "enhancers" / "gene_enhancer_features.tsv.gz", sep="\t")
    near = [c for c in ring_columns() if not c.startswith("enh_200")]
    ef["enh_input"] = np.log1p(np.expm1(ef[near]).sum(axis=1))      # distal input within 200 kb
    enh = ef.pivot(index="gene_id", columns="sample", values="enh_input")

    universe = comp["Naive"].index.intersection(expr.index)
    return classes, comp, marks, enh, expr, de, names, universe


def is_up(de, cond, genes):
    d = de[cond].reindex(genes)
    return ((d["log2FoldChange"] > UP_LFC) & (d["padj"] < UP_PADJ)).fillna(False)


def named(names, genes):
    n = names.reindex(genes).fillna("")
    return pd.Index(genes)[~n.str.contains(r"^Gm\d|Rik$|^$").to_numpy()]


# ---------------------------------------------------------------------------
# a. poised versus silent
# ---------------------------------------------------------------------------


def poised_vs_silent(comp, expr, de, names, universe):
    silent = universe[expr.loc[universe, "Naive"] < SILENT_LOG2]
    dom = comp["Naive"].loc[silent].idxmax(axis=1)
    rows = []
    for cond in ("Arm", "Cl13"):
        up = is_up(de, cond, silent)
        bg = silent[dom == "background"]
        for c in CLASSES:
            g = silent[dom == c]
            k, n = int(up.loc[g].sum()), len(g)
            lo, hi = wilson(k, n)
            kb, nb = int(up.loc[bg].sum()), len(bg)
            odds, p = fisher_exact([[k, n - k], [kb, nb - kb]]) if c != "background" else (1.0, 1.0)
            rows.append({"later_state": SHORT[cond], "tn_promoter_class": c, "n_silent_genes": n,
                         "n_up": k, "frac_up": k / max(n, 1), "ci_lo": lo, "ci_hi": hi,
                         "odds_ratio_vs_background": odds, "fisher_p": p})
    table = pd.DataFrame(rows)
    table.to_csv(RES / "poised_vs_silent.tsv", sep="\t", index=False)

    poised = silent[dom == "poised"]
    genes = pd.DataFrame({"name": names.reindex(poised),
                          "tn_poised_share": comp["Naive"].loc[poised, "poised"],
                          **{f"log2_expr_{SHORT[c]}": expr.loc[poised, c] for c in CONDITIONS},
                          "up_in_TMEM": is_up(de, "Arm", poised), "up_in_TEX": is_up(de, "Cl13", poised)})
    genes.sort_values("log2_expr_TEX", ascending=False).to_csv(RES / "poised_in_naive_genes.tsv", sep="\t")
    print(table.round(3).to_string(index=False))
    return table


# ---------------------------------------------------------------------------
# b. chromatin ahead of RNA
# ---------------------------------------------------------------------------


def chromatin_vs_rna(comp, expr, de, names, universe):
    keep = named(names, universe)
    frames, lists = [], {}
    for cond in ("Arm", "Cl13"):
        d = comp[cond].loc[keep] - comp["Naive"].loc[keep]
        shift = d["active"] - (d["poised"] + d["repressed"])          # + toward active
        dd = de[cond].reindex(keep)
        lfc = dd["log2FoldChange"]
        frame = pd.DataFrame({"name": names.reindex(keep), "contrast": f"{SHORT[cond]}_vs_TN",
                              "promoter_shift": shift, "rna_log2fc": lfc,
                              "log2_expr_TN": expr.loc[keep, "Naive"], "log2_expr_later": expr.loc[keep, cond],
                              "baseMean": dd["baseMean"]})
        frames.append(frame)
        to_silence = frame.loc[(shift < -SHIFT_MIN) & (frame["log2_expr_later"] >= WARN_EXPR)
                               & lfc.between(-1.5, 0.5)]
        to_active = frame.loc[(shift > SHIFT_MIN) & lfc.between(-0.5, 1.0) & (frame["baseMean"] >= 50)]
        lists[(cond, "silencing")] = to_silence.sort_values("promoter_shift")
        lists[(cond, "activation")] = to_active.sort_values("promoter_shift", ascending=False)
        ok = frame["rna_log2fc"].notna() & (frame["baseMean"] >= 10)
        rho = spearmanr(frame.loc[ok, "promoter_shift"], frame.loc[ok, "rna_log2fc"]).correlation
        print(f"{SHORT[cond]} vs TN: promoter shift vs RNA log2FC rho {rho:.3f}; "
              f"chromatin toward silencing with RNA still high: {len(to_silence)}; "
              f"toward activation with RNA not up: {len(to_active)}")
    allf = pd.concat(frames)
    allf.to_csv(RES / "promoter_shift_vs_rna.tsv.gz", sep="\t")
    for (cond, kind), t in lists.items():
        t.to_csv(RES / f"chromatin_ahead_{SHORT[cond]}_{kind}.tsv", sep="\t")
    return allf, lists


# ---------------------------------------------------------------------------
# c. clusters
# ---------------------------------------------------------------------------


def pattern_of(row_by_cond) -> str:
    return " > ".join(LETTER[r.idxmax()] for r in row_by_cond)


def exact_patterns(comp, expr, de, names, universe):
    dom = pd.DataFrame({c: comp[c].loc[universe].idxmax(axis=1).map(LETTER) for c in CONDITIONS})
    pat = dom["Naive"] + " > " + dom["Arm"] + " > " + dom["Cl13"]
    up = is_up(de, "Cl13", universe)
    d = de["Cl13"].reindex(universe)
    down = ((d["log2FoldChange"] < -UP_LFC) & (d["padj"] < UP_PADJ)).fillna(False)
    rows = []
    for p, g in pat.groupby(pat):
        if len(g) < MIN_PATTERN or p == "- > - > -":
            continue
        idx = g.index
        rows.append({"pattern": p, "n_genes": len(idx),
                     **{f"median_log2_expr_{SHORT[c]}": expr.loc[idx, c].median() for c in CONDITIONS},
                     "pct_up_TEX": 100 * up.loc[idx].mean(), "pct_down_TEX": 100 * down.loc[idx].mean(),
                     "key_genes": ", ".join(sorted(set(names.reindex(idx).dropna()) & set(KEY_GENES)))})
    table = pd.DataFrame(rows).sort_values("n_genes", ascending=False)
    table.to_csv(RES / "promoter_patterns.tsv", sep="\t", index=False)
    pd.DataFrame({"name": names.reindex(universe), "pattern": pat}).to_csv(
        RES / "promoter_pattern_genes.tsv", sep="\t")
    print(table.round(1).to_string(index=False))
    return pat


def cluster_features(comp, marks, enh, universe):
    bg_all = pd.concat([comp[c].loc[universe, "background"] for c in CONDITIONS], axis=1).min(axis=1)
    genes = universe[(bg_all < 0.95).to_numpy()]                     # not background in all three
    genes = genes.intersection(enh.index)
    S = pd.concat([comp[c].loc[genes, ["active", "poised", "repressed"]].add_prefix(f"{SHORT[c]}_")
                   for c in CONDITIONS], axis=1)
    # Strength and enhancer input enter as *changes* from TN. As levels they dominate the
    # clustering with how strongly a gene is marked overall, and genes whose promoter
    # stays "active" but gains or loses strength (Pdcd1, Tcf7) fall into one big
    # always-active cluster. The state block already carries the level.
    later = [c for c in CONDITIONS if c != "Naive"]
    M = pd.concat([(marks[c].loc[genes] - marks["Naive"].loc[genes]).add_prefix(f"d{SHORT[c]}_")
                   for c in later], axis=1)
    E = pd.concat([(enh.loc[genes, c] - enh.loc[genes, "Naive"]).rename(f"d{SHORT[c]}_enhancer")
                   for c in later], axis=1)
    blocks = {"state": S, "marks": M / M.std(), "enhancer": E / E.std()}
    # Each block contributes equal total variance, so 12 mark columns do not outvote
    # 9 state columns or 3 enhancer columns.
    X = pd.concat([b / np.sqrt(b.var().sum()) for b in blocks.values()], axis=1)
    return genes, X, blocks


def kmeans_clusters(comp, marks, enh, expr, de, names, universe):
    genes, X, blocks = cluster_features(comp, marks, enh, universe)
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(X), size=min(6000, len(X)), replace=False)
    sel = []
    for k in range(6, 15):
        lab = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X.to_numpy())
        sel.append({"k": k, "silhouette": silhouette_score(X.to_numpy()[sample], lab[sample])})
    pd.DataFrame(sel).to_csv(RES / "kmeans_k_selection.tsv", sep="\t", index=False)

    km = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=SEED).fit(X.to_numpy())
    lab = pd.Series(km.labels_, index=genes)
    lfc = de["Cl13"]["log2FoldChange"].reindex(genes)
    # Display order: from clusters going down in TEX to clusters going up.
    order = lab.groupby(lab).apply(lambda g: lfc.loc[g.index].median()).sort_values().index
    remap = {old: new for new, old in enumerate(order)}
    lab = lab.map(remap)

    up = is_up(de, "Cl13", genes)
    rows = []
    for k in range(N_CLUSTERS):
        idx = lab.index[lab == k]
        S = blocks["state"].loc[idx].mean()
        centre = [pd.Series({c: S[f"{SHORT[cond]}_{c}"] if c != "background" else
                             1 - sum(S[f"{SHORT[cond]}_{x}"] for x in ("active", "poised", "repressed"))
                             for c in CLASSES}) for cond in CONDITIONS]
        key = sorted(set(names.reindex(idx).dropna()) & set(KEY_GENES))
        movers = names.reindex(named(names, idx))
        movers = movers.loc[~movers.isin(key)]
        movers = movers.reindex(lfc.loc[movers.index].abs().sort_values(ascending=False).index)
        rows.append({
            "cluster": k + 1, "n_genes": len(idx), "pattern": pattern_of(centre),
            **{f"{SHORT[c]}_active": S[f"{SHORT[c]}_active"] for c in CONDITIONS},
            **{f"{SHORT[c]}_poised": S[f"{SHORT[c]}_poised"] for c in CONDITIONS},
            **{f"{SHORT[c]}_repressed": S[f"{SHORT[c]}_repressed"] for c in CONDITIONS},
            **{f"median_log2_expr_{SHORT[c]}": expr.loc[idx, c].median() for c in CONDITIONS},
            "median_log2fc_TEX": lfc.loc[idx].median(), "pct_up_TEX": 100 * up.loc[idx].mean(),
            "key_genes": ", ".join(key), "other_examples": ", ".join(movers.head(4)),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(RES / "cluster_summary.tsv", sep="\t", index=False)
    out = pd.DataFrame({"name": names.reindex(genes), "cluster": lab + 1})
    out.join(X).to_csv(RES / "gene_clusters.tsv.gz", sep="\t")
    print(pd.DataFrame(sel).round(3).to_string(index=False))
    print(summary[["cluster", "n_genes", "pattern", "median_log2_expr_TN", "median_log2_expr_TMEM",
                   "median_log2_expr_TEX", "pct_up_TEX", "key_genes", "other_examples"]]
          .round(1).to_string(index=False))
    return lab, blocks, summary


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def fig_poised(table, shift, lists):
    fig, axes = plt.subplots(1, 2, figsize=(ns.DOUBLE, 62 * ns.MM),
                             gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.35})
    ax = axes[0]
    w = 0.38
    for i, (cond, colour) in enumerate((("TMEM", ns.CONDITION_COLORS["Arm"]),
                                        ("TEX", ns.CONDITION_COLORS["Cl13"]))):
        t = table.loc[table["later_state"] == cond].set_index("tn_promoter_class").loc[CLASSES]
        x = np.arange(len(CLASSES)) + (i - 0.5) * w
        ax.bar(x, 100 * t["frac_up"], w, color=colour, label=f"Up in {cond}")
        ax.errorbar(x, 100 * t["frac_up"], yerr=[100 * (t["frac_up"] - t["ci_lo"]),
                                                 100 * (t["ci_hi"] - t["frac_up"])],
                    fmt="none", ecolor="black", elinewidth=0.5, capsize=1.2)
    n = table.loc[table["later_state"] == "TEX"].set_index("tn_promoter_class").loc[CLASSES, "n_silent_genes"]
    ax.set_xticks(range(len(CLASSES)), [f"{c.capitalize()}\n(n = {n[c]:,})" for c in CLASSES])
    ax.set_xlabel("TN promoter state of genes silent in TN")
    ax.set_ylabel("Genes switched on later (%)")
    ax.legend(loc="upper right")
    ns.panel_label(ax, "a")

    ax = axes[1]
    s = shift.loc[(shift["contrast"] == "TEX_vs_TN") & shift["rna_log2fc"].notna() & (shift["baseMean"] >= 10)]
    ax.scatter(s["promoter_shift"], s["rna_log2fc"].clip(-10, 10), s=1, color="#c8c8c8",
               linewidths=0, rasterized=True)
    sil, act = lists[("Cl13", "silencing")], lists[("Cl13", "activation")]
    ax.scatter(sil["promoter_shift"], sil["rna_log2fc"], s=2.5, color=CLASS_COLORS["poised"],
               linewidths=0, rasterized=True, label="Promoter silenced, RNA still high")
    ax.scatter(act["promoter_shift"], act["rna_log2fc"], s=2.5, color=CLASS_COLORS["active"],
               linewidths=0, rasterized=True, label="Promoter activated, RNA not up")
    offsets = {"Tcf7": (6, -6), "S1pr1": (6, 4), "Klrg1": (-14, 6), "Tox2": (-22, 6), "Pdcd1": (6, 6)}
    for g, off in offsets.items():
        r = s.loc[s["name"] == g]
        if len(r):
            r = r.iloc[0]
            ax.annotate(g, (r["promoter_shift"], np.clip(r["rna_log2fc"], -10, 10)), xytext=off,
                        textcoords="offset points", fontsize=5, fontstyle="italic",
                        arrowprops={"arrowstyle": "-", "lw": 0.3, "color": "black"})
    ax.axhline(0, color="0.6", lw=0.4)
    ax.axvline(0, color="0.6", lw=0.4)
    ax.set_xlabel("Promoter shift, TEX vs TN\n(gain in active minus gain in poised + repressed share)")
    ax.set_ylabel("RNA log2 fold change, TEX vs TN")
    ax.legend(loc="lower right", markerscale=3, handletextpad=0.2)
    ns.panel_label(ax, "b")
    ns.save(fig, FIG, "Fig8_poised_genes")


def fig_programs(lab, blocks, summary, expr):
    order = lab.sort_values(kind="stable").index
    S = blocks["state"].loc[order]
    M = blocks["marks"].loc[order].clip(-2.5, 2.5)
    E = blocks["enhancer"].loc[order].clip(-2.5, 2.5)
    R = expr.loc[order, CONDITIONS]
    R = R.sub(R.mean(axis=1), axis=0).clip(-3, 3)                    # change, not level

    panels = [(f"{c.capitalize()}\nshare", S[[f"{SHORT[k]}_{c}" for k in CONDITIONS]].to_numpy(),
               LinearSegmentedColormap.from_list(c, ["white", CLASS_COLORS[c]]), 0, 1)
              for c in ("active", "poised", "repressed")]
    panels += [(f"\u0394 {m}", M[[f"d{k}_{m}" for k in ("TMEM", "TEX")]].to_numpy(),
                ns.DIVERGING, -2.5, 2.5) for m in MARKS]
    panels += [("\u0394 Enhancer\ninput", E.to_numpy(), ns.DIVERGING, -2.5, 2.5),
               ("RNA\n(gene-centred)", R.to_numpy(), "PuOr_r", -3, 3)]

    fig = plt.figure(figsize=(ns.DOUBLE, 150 * ns.MM))
    gs = fig.add_gridspec(2, len(panels) + 2, height_ratios=[1, 0.025], hspace=0.08, wspace=0.12,
                          width_ratios=[0.25] + [1] * len(panels) + [2.4], left=0.03, right=0.99,
                          top=0.90, bottom=0.05)
    bar = fig.add_subplot(gs[0, 0])
    cluster_cmap = plt.get_cmap("tab10" if N_CLUSTERS <= 10 else "tab20")
    bar.imshow(lab.loc[order].to_numpy()[:, None], aspect="auto", cmap=cluster_cmap,
               vmin=-0.5, vmax=cluster_cmap.N - 0.5, interpolation="nearest")
    bar.set_xticks([])
    bar.set_yticks([])
    for s in bar.spines.values():
        s.set_visible(False)
    bounds = np.r_[0, np.cumsum(summary["n_genes"].to_numpy())]

    images = {}
    for j, (title, data, cmap, lo, hi) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j + 1])
        images[title] = ax.imshow(data, aspect="auto", cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(title, fontsize=5.5, rotation=0)
        ticks = ["M", "E"] if data.shape[1] == 2 else ["N", "M", "E"]
        ax.set_xticks(range(len(ticks)), ticks, fontsize=5)
        ax.xaxis.tick_top()
        ax.set_yticks([])
        for b in bounds[1:-1]:
            ax.axhline(b - 0.5, color="black", lw=0.3)
        for s in ax.spines.values():
            s.set_visible(False)

    txt = fig.add_subplot(gs[0, -1])
    txt.set_xlim(0, 1)
    txt.set_ylim(bounds[-1], 0)
    txt.axis("off")
    # Labels sit at their cluster's midpoint unless that would overlap a neighbour;
    # then they are pushed apart and joined to the cluster by a thin leader line.
    mids = (bounds[:-1] + bounds[1:]) / 2
    gap = 0.075 * bounds[-1]
    pos = mids.copy()
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    pos[-1] = min(pos[-1], bounds[-1] - gap / 2)
    for i in range(len(pos) - 2, -1, -1):
        pos[i] = min(pos[i], pos[i + 1] - gap)
    for (_, r), mid, y in zip(summary.iterrows(), mids, pos):
        genes = textwrap.fill(r["key_genes"] or r["other_examples"], 48)
        label = (f"{r['cluster']}  {r['pattern']}   n = {r['n_genes']:,}\n"
                 f"RNA {r['median_log2_expr_TN']:.1f} > {r['median_log2_expr_TMEM']:.1f} > "
                 f"{r['median_log2_expr_TEX']:.1f}; {r['pct_up_TEX']:.0f}% up in TEX\n{genes}")
        txt.plot([0, 0.05], [mid, y], color="black", lw=0.3, clip_on=False)
        txt.text(0.07, y, label, va="center", ha="left", fontsize=4.8, linespacing=1.15)

    for j, (key, label) in enumerate([("Active\nshare", "State share"),
                                      ("\u0394 H3K4me3", "Change from TN (SD units)"),
                                      ("RNA\n(gene-centred)", "RNA, log2 minus gene mean")]):
        cax = fig.add_subplot(gs[1, 1 + 3 * j: 3 + 3 * j])
        cb = fig.colorbar(images[key], cax=cax, orientation="horizontal")
        cb.set_label(label, fontsize=5)
        cb.ax.tick_params(labelsize=5, width=0.4, length=1.5)
        cb.outline.set_linewidth(0.4)
    fig.text(0.03, 0.985, f"{len(order):,} genes clustered on promoter state, promoter mark "
             "strength change and distal enhancer input change (k-means, k = "
             f"{N_CLUSTERS}); N = TN, M = TMEM, E = TEX; \u0394 = change from TN", fontsize=5.5)
    ns.save(fig, FIG, "Fig9_chromatin_programs")


def main() -> None:
    ns.apply()
    classes, comp, marks, enh, expr, de, names, universe = load()
    print(f"{len(universe):,} genes with promoter state and RNA")
    print("\n== a. poised versus silent")
    table = poised_vs_silent(comp, expr, de, names, universe)
    print("\n== b. chromatin ahead of RNA")
    shift, lists = chromatin_vs_rna(comp, expr, de, names, universe)
    print("\n== c(i). exact promoter patterns")
    exact_patterns(comp, expr, de, names, universe)
    print("\n== c(ii). k-means on state + strength + enhancer input")
    lab, blocks, summary = kmeans_clusters(comp, marks, enh, expr, de, names, universe)
    fig_poised(table, shift, lists)
    fig_programs(lab, blocks, summary, expr)
    print(f"-> {RES}, {FIG}/Fig8_poised_genes.pdf, {FIG}/Fig9_chromatin_programs.pdf")


if __name__ == "__main__":
    main()
