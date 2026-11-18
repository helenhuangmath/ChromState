"""Evaluating segmentation tools on real data, where no state truth exists.

Simulated truth answers "did the tool recover the states that generated the data";
nothing like it exists for a real genome. What *does* exist is biology the states are
supposed to capture and that no tool was told about: where transcription start sites
are, which genes are expressed and how much, and how expression changes between cell
states. Every figure here asks each tool the same question about those external facts,
on the same bins, with the same cross-validation, so the rows are comparable even though
none of them is a score against truth.

    save_runs / load_runs     persist every tool's per-bin calls so figures can be redrawn
    mark_blend_colours        one colour per state from its mark profile, so states of
                              different tools are drawn in a common language without
                              matching them to each other
    plot_state_profiles       what each state of each tool is, in marks
    plot_state_fractions      how much of the genome each tool puts in active, bivalent,
                              Polycomb, heterochromatin and quiescent states, per sample
    plot_segment_lengths      segment length distributions
    plot_state_overlap        how each tool's states partition the reference tool's
    promoter_recovery         cross-validated AUPRC for finding TSS bins, and the TSS bins
                              of expressed genes, from the tool's states alone
    expression_prediction     held-out Spearman for expression level and change from the
                              promoter's state composition
    plot_locus                a browser-style view: marks, every tool's states, genes
    write_realdata_report     all of the above, figures plus REPORT.md

Caveats that apply to every number here: a segmentation can be biologically useful
without separating promoters (a domain caller, say), and expression prediction rewards
tools that keep promoter states fine-grained. Read a row as "how well this tool's states,
used naively, answer this question", not as the tool's quality.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chromstate import viz
from chromstate.features import FeatureSet
from chromstate.model.annotate import canonical_mark

from .core import UNASSIGNED, ToolRun
from .scores import _weights, state_profiles

__all__ = [
    "save_runs", "load_runs", "read_gtf_genes", "read_expression", "mark_blend_colours",
    "state_order", "plot_state_profiles", "plot_state_fractions", "plot_segment_lengths",
    "plot_state_overlap", "promoter_composition", "promoter_recovery",
    "expression_prediction", "plot_locus", "write_realdata_report",
]

EPS = 1e-12


# ---------------------------------------------------------------------------
# Persisting runs
# ---------------------------------------------------------------------------


def save_runs(runs: Sequence[ToolRun], outdir: str) -> List[str]:
    """Write each run's states (and posterior) as ``<tool>.npz`` plus ``runs.json``."""
    os.makedirs(outdir, exist_ok=True)
    written, meta = [], []
    for run in runs:
        row = {k: v for k, v in asdict(run).items()
               if k not in ("states", "posterior", "bivalency_score", "activity_score")}
        row["meta"] = {k: (v if isinstance(v, (str, int, float, bool, list, type(None))) else str(v))
                       for k, v in (run.meta or {}).items()}
        meta.append(row)
        if run.states is not None:
            arrays = {"states": run.states.astype(np.int32)}
            if run.posterior is not None:
                arrays["posterior"] = run.posterior.astype(np.float32)
            path = os.path.join(outdir, f"{run.tool}.npz")
            np.savez_compressed(path, **arrays)
            written.append(path)
    path = os.path.join(outdir, "runs.json")
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=1, default=str)
    written.append(path)
    return written


def load_runs(outdir: str) -> List[ToolRun]:
    with open(os.path.join(outdir, "runs.json")) as fh:
        meta = json.load(fh)
    runs = []
    for row in meta:
        states = posterior = None
        path = os.path.join(outdir, f"{row['tool']}.npz")
        if os.path.exists(path):
            z = np.load(path)
            states = z["states"]
            posterior = z["posterior"] if "posterior" in z.files else None
        runs.append(ToolRun(
            tool=row["tool"], display_name=row["display_name"], status=row["status"],
            reason=row.get("reason", ""), states=states, posterior=posterior,
            n_states=int(row.get("n_states", 0)), runtime_s=float(row.get("runtime_s", np.nan)),
            peak_rss_mb=row.get("peak_rss_mb"), version=row.get("version", ""),
            command=row.get("command", ""), workdir=row.get("workdir"),
            verified=bool(row.get("verified", True)), notes=row.get("notes", ""),
            meta=row.get("meta", {}),
        ))
    return runs


# ---------------------------------------------------------------------------
# External facts: genes and expression
# ---------------------------------------------------------------------------


def read_gtf_genes(path: str, chroms: Optional[Sequence[str]] = None,
                   gene_types: Optional[Sequence[str]] = ("protein_coding",)) -> FeatureSet:
    """Genes from a GENCODE/Ensembl GTF with the version stripped from ``gene_id``."""
    import gzip
    import re

    attr_re = re.compile(r'(\S+)\s+"([^"]*)"')
    opener = gzip.open if path.endswith(".gz") else open
    rows = []
    keep_types = set(gene_types) if gene_types else None
    keep_chroms = set(chroms) if chroms else None
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            if keep_chroms and parts[0] not in keep_chroms:
                continue
            attrs = dict(attr_re.findall(parts[8]))
            if keep_types and attrs.get("gene_type", attrs.get("gene_biotype")) not in keep_types:
                continue
            rows.append({"gene_id": attrs["gene_id"].split(".")[0],
                         "name": attrs.get("gene_name", attrs["gene_id"]),
                         "chrom": parts[0], "start": int(parts[3]) - 1, "end": int(parts[4]),
                         "strand": parts[6] if parts[6] in "+-" else "+"})
    if not rows:
        raise ValueError(f"no gene records in {path} for chroms {chroms} and types {gene_types}")
    return FeatureSet(pd.DataFrame(rows).drop_duplicates("gene_id"), source=path)


def read_expression(path: str, samples: Sequence[str]) -> pd.DataFrame:
    """Genes x samples (columns named after the dataset's samples), ``gene_id`` index."""
    tab = pd.read_csv(path, sep="\t")
    if "gene_id" not in tab.columns:
        tab = tab.rename(columns={tab.columns[0]: "gene_id"})
    tab["gene_id"] = tab["gene_id"].astype(str).str.split(".").str[0]
    tab = tab.drop_duplicates("gene_id").set_index("gene_id")
    missing = [s for s in samples if s not in tab.columns]
    if missing:
        raise ValueError(f"expression table lacks column(s) {missing}; has {list(tab.columns)[:8]}")
    return tab[list(samples)].astype(float)


# ---------------------------------------------------------------------------
# A common language for states: colour from the mark profile
# ---------------------------------------------------------------------------


def _channels(ds) -> Dict[str, List[int]]:
    canon = [canonical_mark(m) for m in ds.marks]
    chan = {"active": [], "polycomb": [], "hetero": []}
    for j, c in enumerate(canon):
        if c in ("K27ac", "K4me3", "K4me1", "K9ac", "ATAC", "K36me3"):
            chan["active"].append(j)
        elif c == "K27me3":
            chan["polycomb"].append(j)
        elif c == "K9me3":
            chan["hetero"].append(j)
    return chan


def state_fold(ds, run: ToolRun) -> Tuple[np.ndarray, np.ndarray]:
    """``(log2 fold of each state's mean signal over the genome mean, occupancy)``."""
    profiles, W = state_profiles(ds, run.states, run.posterior, run.n_states)
    Y = ds.X / np.maximum(ds.size_factors, EPS)
    genome = Y.mean(axis=0)
    fold = np.log2((profiles + 0.05 * genome) / (genome + 0.05 * genome + EPS))
    occ = W.sum(axis=0) / max(W.sum(), EPS)
    return fold, occ


def _saturate(fold: np.ndarray, full: float = 3.0) -> np.ndarray:
    """0 at or below the genome mean, 1 at ``2**full``-fold enrichment."""
    return np.clip(fold / full, 0.0, 1.0)


def state_strengths(ds, run: ToolRun) -> pd.DataFrame:
    """Per state: saturated active / Polycomb / heterochromatin strength, class, occupancy."""
    fold, occ = state_fold(ds, run)
    chan = _channels(ds)
    rows = []
    for k in range(fold.shape[0]):
        a = _saturate(fold[k, chan["active"]].max()) if chan["active"] else 0.0
        p = _saturate(fold[k, chan["polycomb"]].max()) if chan["polycomb"] else 0.0
        h = _saturate(fold[k, chan["hetero"]].max()) if chan["hetero"] else 0.0
        if a >= 0.4 and p >= 0.4:
            cls = "bivalent"
        elif a >= 0.4:
            cls = "active"
        elif p >= 0.4:
            cls = "polycomb"
        elif h >= 0.4:
            cls = "heterochromatin"
        elif max(a, p, h) >= 0.15:
            cls = "weak"
        else:
            cls = "quiescent"
        rows.append({"state": k, "active": a, "polycomb": p, "hetero": h, "class": cls,
                     "occupancy": float(occ[k])})
    return pd.DataFrame(rows)


def mark_blend_colours(strengths: pd.DataFrame) -> List[str]:
    """Subtractive blend: active -> red, Polycomb -> blue, heterochromatin -> violet,
    nothing -> near-white. Bivalent states come out magenta, as in the ChromHMM
    convention, without any state matching."""
    out = []
    for r in strengths.itertuples():
        absorb = (r.active * np.array([0.0, 0.85, 0.85])
                  + r.polycomb * np.array([0.85, 0.55, 0.0])
                  + r.hetero * np.array([0.35, 0.75, 0.25]))
        rgb = np.clip(1.0 - absorb, 0.0, 1.0)
        rgb = 0.93 * rgb + 0.0  # keep quiescent states just off-white
        out.append("#%02x%02x%02x" % tuple(int(round(255 * v)) for v in rgb))
    return out


def state_order(strengths: pd.DataFrame) -> np.ndarray:
    """Active first, then bivalent, Polycomb, heterochromatin, weak, quiescent."""
    rank = {"active": 0, "bivalent": 1, "polycomb": 2, "heterochromatin": 3, "weak": 4,
            "quiescent": 5}
    key = strengths.assign(r=strengths["class"].map(rank),
                           s=-(strengths["active"] + strengths["polycomb"] + strengths["hetero"]))
    return key.sort_values(["r", "s"]).index.to_numpy()


CLASS_COLOURS = {"active": "#d7301f", "bivalent": "#b03a8c", "polycomb": "#3b6fd1",
                 "heterochromatin": "#7b5ea7", "weak": "#c9c9c9", "quiescent": "#efefef"}


def _ok_runs(runs: Sequence[ToolRun]) -> List[ToolRun]:
    return [r for r in runs if r.ok]


def _short(run: ToolRun) -> str:
    return run.display_name.split(" (")[0]


# ---------------------------------------------------------------------------
# Figures: what the states are
# ---------------------------------------------------------------------------


def plot_state_profiles(ds, runs: Sequence[ToolRun], vmax: float = 4.0):
    """One heatmap per tool: log2 fold of each mark in each state, states ordered by class."""
    ok = _ok_runs(runs)
    if not ok:
        return None
    n = len(ok)
    heights = [max(r.n_states, 3) for r in ok]
    fig, axes = plt.subplots(1, n, figsize=(min(viz.WIDTH_DOUBLE, 1.45 * n + 0.6), 2.9),
                             facecolor=viz.SURFACE, gridspec_kw={"width_ratios": [1] * n, "wspace": 0.75})
    axes = np.atleast_1d(axes)
    for ax, run in zip(axes, ok):
        fold, occ = state_fold(ds, run)
        st = state_strengths(ds, run)
        order = state_order(st)
        cols = mark_blend_colours(st)
        im = ax.imshow(fold[order], cmap=viz.DIV_CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(ds.n_marks))
        ax.set_xticklabels(ds.marks, rotation=90, fontsize=viz.FS_TICK)
        ax.set_yticks(range(len(order)))
        # the share of bins per state is in state_classes.tsv; printing it beside every row
        # puts ~15 numbers on each panel
        ax.set_yticklabels([f"{occ[k] * 100:4.1f}%" for k in order] if viz.ANNOTATE_VALUES
                           else [""] * len(order), fontsize=viz.FS_TICK)
        for yi, k in enumerate(order):
            ax.add_patch(plt.Rectangle((ds.n_marks - 0.42, yi - 0.5), 0.35, 1.0, color=cols[k],
                                       clip_on=False, transform=ax.transData))
        ax.set_xlim(-0.5, ds.n_marks - 0.5)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        viz._title(ax, _short(run), f"{run.n_states} states; left: share of bins,\nright: state colour")
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.03, pad=0.02)
    cb.set_label("Mark level (log2 fold over genome mean)", fontsize=viz.FS_LABEL)
    cb.ax.tick_params(labelsize=viz.FS_TICK)
    return fig


def class_fractions(ds, runs: Sequence[ToolRun]) -> pd.DataFrame:
    """Share of each sample's bins in each state class, per tool."""
    rows = []
    bins = ds.bins.reset_index(drop=True)
    for run in _ok_runs(runs):
        st = state_strengths(ds, run)
        cls = st["class"].to_numpy()
        W = _weights(run.states, run.posterior, run.n_states)
        for sample in ds.samples:
            m = (bins["sample"] == sample).to_numpy()
            mass = W[m].sum(axis=0)
            tot = max(mass.sum(), EPS)
            for c in CLASS_COLOURS:
                rows.append({"tool": run.tool, "method": _short(run), "sample": sample, "class": c,
                             "fraction": float(mass[cls == c].sum() / tot)})
    return pd.DataFrame(rows)


def plot_state_fractions(ds, runs: Sequence[ToolRun]):
    """Stacked bars: per tool and sample, the share of the genome in each state class."""
    frac = class_fractions(ds, runs)
    if frac.empty:
        return None
    tools = list(dict.fromkeys(frac["method"]))
    samples = list(ds.samples)
    fig, ax = viz._new_fig(viz.WIDTH_DOUBLE, 0.5 + 0.28 * len(tools) * len(samples))
    viz._style_axes(ax, grid_axis="x")
    y = 0
    ticks, labels = [], []
    for t in tools:
        for s in samples:
            left = 0.0
            sub = frac[(frac["method"] == t) & (frac["sample"] == s)].set_index("class")["fraction"]
            for c, col in CLASS_COLOURS.items():
                v = float(sub.get(c, 0.0))
                ax.barh(y, v, left=left, color=col, edgecolor=viz.SURFACE, linewidth=0.3, height=0.8)
                if viz.ANNOTATE_VALUES and v > 0.06:
                    ax.text(left + v / 2, y, f"{v * 100:.0f}", ha="center", va="center",
                            fontsize=viz.FS_ANNOT, color="white" if c in ("active", "polycomb", "heterochromatin", "bivalent") else viz.INK)
                left += v
            ticks.append(y)
            # the tool is named once, on the first of its rows
            labels.append(f"{t}  {s}" if s == samples[0] else str(s))
            y += 1
        y += 0.5
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=viz.FS_TICK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of bins", fontsize=viz.FS_LABEL)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLASS_COLOURS.values()]
    ax.legend(handles, [c.capitalize() for c in CLASS_COLOURS], fontsize=viz.FS_LEGEND, ncol=6, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.12))
    viz._title(ax, "State classes",
               "classes from each state's own mark profile (active >= 2.3x, Polycomb, H3K9me3)")
    return fig


def segment_lengths(ds, run: ToolRun) -> np.ndarray:
    bins = ds.bins.reset_index(drop=True)
    bin_size = int(bins["end"].iloc[0] - bins["start"].iloc[0])
    out = []
    for _, grp in bins.groupby(["sample", "chrom"], sort=False):
        seq = run.states[grp.index.to_numpy()]
        if seq.size == 0:
            continue
        change = np.flatnonzero(seq[1:] != seq[:-1]) + 1
        edges = np.concatenate([[0], change, [seq.size]])
        lens = np.diff(edges)
        keep = seq[edges[:-1]] != UNASSIGNED
        out.append(lens[keep] * bin_size)
    return np.concatenate(out) if out else np.array([])


def plot_segment_lengths(ds, runs: Sequence[ToolRun]):
    ok = _ok_runs(runs)
    if not ok:
        return None
    fig, ax = viz._new_fig(viz.WIDTH_SINGLE, 2.3)
    viz._style_axes(ax)
    cmap = plt.get_cmap("tab10")
    for i, run in enumerate(ok):
        L = np.sort(segment_lengths(ds, run))
        if L.size == 0:
            continue
        y = np.arange(1, L.size + 1) / L.size
        ax.step(L, y, where="post", lw=1.1, color=cmap(i % 10),
                label=_short(run))
    ax.set_xscale("log")
    ax.set_xlabel("Segment length (bp)", fontsize=viz.FS_LABEL)
    ax.set_ylabel("Cumulative fraction", fontsize=viz.FS_LABEL)
    ax.legend(fontsize=viz.FS_LEGEND, frameon=False, loc="lower right")
    viz._title(ax, "Segment lengths", "hard calls, all samples")
    return fig


def plot_state_overlap(ds, runs: Sequence[ToolRun], reference: str = "chromstate"):
    """P(tool state | reference state), one panel per tool, both axes ordered by class."""
    ok = _ok_runs(runs)
    ref = next((r for r in ok if r.tool == reference), None)
    others = [r for r in ok if r.tool != reference]
    if ref is None or not others:
        return None
    ref_st = state_strengths(ds, ref)
    ref_order = state_order(ref_st)
    ref_cols = mark_blend_colours(ref_st)
    n = len(others)
    fig, axes = plt.subplots(1, n, figsize=(min(viz.WIDTH_DOUBLE, 1.6 * n + 0.8), 3.1),
                             facecolor=viz.SURFACE, gridspec_kw={"wspace": 0.35})
    axes = np.atleast_1d(axes)
    for ax, run in zip(axes, others):
        st = state_strengths(ds, run)
        order = state_order(st)
        cols = mark_blend_colours(st)
        keep = (ref.states != UNASSIGNED) & (run.states != UNASSIGNED)
        M = np.zeros((ref.n_states, run.n_states))
        np.add.at(M, (ref.states[keep], run.states[keep]), 1.0)
        P = M / np.maximum(M.sum(axis=1, keepdims=True), 1.0)
        im = ax.imshow(P[np.ix_(ref_order, order)], cmap=viz.SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([""] * len(order))
        ax.set_yticks(range(len(ref_order)))
        ax.set_yticklabels([""] * len(ref_order))
        for xi, k in enumerate(order):
            ax.add_patch(plt.Rectangle((xi - 0.5, len(ref_order) - 0.5), 1.0, 0.45, color=cols[k],
                                       clip_on=False))
        for yi, k in enumerate(ref_order):
            ax.add_patch(plt.Rectangle((-0.95, yi - 0.5), 0.45, 1.0, color=ref_cols[k], clip_on=False))
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        viz._title(ax, _short(run), f"rows: {_short(ref)} states\ncolumns: {_short(run)} states")
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.03, pad=0.02)
    cb.set_label(f"Fraction of the {_short(ref)} state", fontsize=viz.FS_LABEL)
    cb.ax.tick_params(labelsize=viz.FS_TICK)
    return fig


# ---------------------------------------------------------------------------
# Promoters and expression
# ---------------------------------------------------------------------------


def _sample_chrom_index(ds) -> Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]:
    """``(sample, chrom) -> (row indices sorted by start, starts)``."""
    bins = ds.bins.reset_index(drop=True)
    out = {}
    for (s, c), grp in bins.groupby(["sample", "chrom"], sort=False):
        idx = grp.index.to_numpy()
        starts = grp["start"].to_numpy()
        o = np.argsort(starts, kind="stable")
        out[(s, c)] = (idx[o], starts[o])
    return out


def promoter_composition(ds, run: ToolRun, features: FeatureSet, flank: int = 1000,
                         index=None) -> pd.DataFrame:
    """Per (gene, sample): mean state weight over the bins whose start lies within
    ``flank`` of the TSS. Columns ``s0..s{K-1}``; genes with no bins are dropped."""
    index = index or _sample_chrom_index(ds)
    W = _weights(run.states, run.posterior, run.n_states)
    K = W.shape[1]
    tss = features.tss().to_numpy()
    genes = features.table
    frames = []
    for sample in ds.samples:
        for chrom, sub in genes.groupby("chrom"):
            key = (sample, chrom)
            if key not in index:
                continue
            rows, starts = index[key]
            cs = np.vstack([np.zeros((1, K)), np.cumsum(W[rows], axis=0)])
            t = tss[sub.index.to_numpy()]
            lo = np.searchsorted(starts, t - flank, side="left")
            hi = np.searchsorted(starts, t + flank, side="left")
            n = hi - lo
            ok = n > 0
            comp = (cs[hi[ok]] - cs[lo[ok]]) / n[ok][:, None]
            f = pd.DataFrame(comp, columns=[f"s{k}" for k in range(K)])
            f.insert(0, "sample", sample)
            f.insert(0, "gene_id", sub["gene_id"].to_numpy()[ok])
            frames.append(f)
    if not frames:
        raise ValueError("no gene overlaps the dataset's bins; check chromosome names")
    return pd.concat(frames, ignore_index=True)


def promoter_signal(ds, features: FeatureSet, flank: int = 1000, index=None) -> pd.DataFrame:
    """Mean normalised signal per mark in the promoter window (the no-model baseline)."""
    index = index or _sample_chrom_index(ds)
    Y = ds.X / np.maximum(ds.size_factors, EPS)
    tss = features.tss().to_numpy()
    frames = []
    for sample in ds.samples:
        for chrom, sub in features.table.groupby("chrom"):
            key = (sample, chrom)
            if key not in index:
                continue
            rows, starts = index[key]
            cs = np.vstack([np.zeros((1, Y.shape[1])), np.cumsum(Y[rows], axis=0)])
            t = tss[sub.index.to_numpy()]
            lo = np.searchsorted(starts, t - flank, side="left")
            hi = np.searchsorted(starts, t + flank, side="left")
            n = hi - lo
            ok = n > 0
            sig = (cs[hi[ok]] - cs[lo[ok]]) / n[ok][:, None]
            f = pd.DataFrame(np.log1p(sig), columns=[f"sig_{m}" for m in ds.marks])
            f.insert(0, "sample", sample)
            f.insert(0, "gene_id", sub["gene_id"].to_numpy()[ok])
            frames.append(f)
    return pd.concat(frames, ignore_index=True)


def _blocks(positions: np.ndarray, block_bp: int = 1_000_000) -> np.ndarray:
    return (positions // block_bp).astype(int)


def _auprc(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(average_precision_score(y, score))


def promoter_recovery(ds, runs: Sequence[ToolRun], features: FeatureSet,
                      expression: Optional[pd.DataFrame] = None, flank: int = 1000,
                      top_fraction: float = 0.3, block_bp: int = 1_000_000) -> pd.DataFrame:
    """How well each tool's states, used as a score, find promoter bins.

    Per bin the score is its state's promoter rate, *learned on other 1 Mb blocks*
    (two-fold by alternating blocks) so a tool is not rewarded for simply having many
    states. ``tss_auprc``: positives are bins within ``flank`` of any TSS.
    ``active_tss_auprc`` (needs expression): positives are TSS bins of genes in the top
    ``top_fraction`` of expression in that sample; bins of other TSSs are excluded, so the
    task is "which promoters are on", not "where are promoters". Baselines are the
    positive rates.
    """
    bins = ds.bins.reset_index(drop=True)
    index = _sample_chrom_index(ds)
    tss = features.tss().to_numpy()
    genes = features.table
    n = len(bins)
    is_tss = np.zeros(n, dtype=bool)
    gene_of_bin = np.full(n, -1)
    for sample in ds.samples:
        for chrom, sub in genes.groupby("chrom"):
            key = (sample, chrom)
            if key not in index:
                continue
            rows, starts = index[key]
            t = tss[sub.index.to_numpy()]
            lo = np.searchsorted(starts, t - flank, side="left")
            hi = np.searchsorted(starts, t + flank, side="left")
            for g, a, b in zip(sub.index.to_numpy(), lo, hi):
                r = rows[a:b]
                is_tss[r] = True
                gene_of_bin[r] = g
    block = _blocks(bins["start"].to_numpy(), block_bp)
    fold = block % 2
    sample_arr = bins["sample"].to_numpy()

    def cv_score(states: np.ndarray, K: int, y: np.ndarray, use: np.ndarray) -> np.ndarray:
        score = np.full(n, np.nan)
        overall = float(y[use].mean()) if use.any() else 0.0
        for f in (0, 1):
            train = use & (fold != f)
            test = use & (fold == f)
            base = float(y[train].mean()) if train.any() else overall
            rate = np.full(K, base)
            for k in range(K):
                m = train & (states == k)
                if m.any():
                    rate[k] = y[m].mean()
            score[test] = rate[np.clip(states[test], 0, K - 1)]
        return np.nan_to_num(score, nan=overall)

    rows_out = []
    for run in _ok_runs(runs):
        states = run.states
        assigned = states != UNASSIGNED
        y = is_tss.astype(float)
        sc = cv_score(states, run.n_states, y, assigned)
        row = {"tool": run.tool, "method": _short(run), "tss_auprc": _auprc(y[assigned], sc[assigned]),
               "tss_baseline": float(y[assigned].mean())}
        if expression is not None:
            vals, labels, used = [], [], []
            for sample in ds.samples:
                expr = expression[sample]
                thr = expr.quantile(1 - top_fraction)
                hi_genes = set(np.flatnonzero(genes["gene_id"].map(lambda g: expr.get(g, np.nan) >= thr).fillna(False)))
                known = genes["gene_id"].isin(expr.index).to_numpy()
                in_sample = sample_arr == sample
                pos = in_sample & is_tss & np.isin(gene_of_bin, list(hi_genes))
                other_tss = in_sample & is_tss & ~pos
                use = in_sample & assigned & ~other_tss
                # TSS bins of genes without expression are excluded via other_tss
                yy = pos.astype(float)
                s = cv_score(states, run.n_states, yy, use)
                vals.append(_auprc(yy[use], s[use]))
                used.append(float(yy[use].mean()))
                labels.append(sample)
            row["active_tss_auprc"] = float(np.nanmean(vals))
            row["active_tss_baseline"] = float(np.nanmean(used))
            for lab, v in zip(labels, vals):
                row[f"active_tss_auprc_{lab}"] = v
        rows_out.append(row)
    return pd.DataFrame(rows_out)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).correlation)


def _cv_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0,
                n_splits: int = 5) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    pred = np.full(len(y), np.nan)
    n_groups = len(np.unique(groups))
    if n_groups < 2:  # one block only: fall back to plain K-fold over rows
        groups = np.arange(len(y)) % max(2, min(n_splits, len(y)))
        n_groups = len(np.unique(groups))
    n_splits = max(2, min(n_splits, n_groups))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=seed)
        pred[te] = m.fit(X[tr], y[tr]).predict(X[te])
    return pred


def _boot_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 300, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        vals.append(_spearman(y[i], p[i]))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def expression_prediction(ds, runs: Sequence[ToolRun], features: FeatureSet,
                          expression: pd.DataFrame, flank: int = 1000,
                          contrast: Optional[Tuple[str, str]] = None, seed: int = 0,
                          block_bp: int = 1_000_000) -> pd.DataFrame:
    """Held-out Spearman between expression and a model of the promoter's state
    composition, per tool, plus a raw-signal baseline (no segmentation at all).

    Level: all (gene, sample) pairs, folds grouped by 1 Mb block of the TSS. Change:
    per gene, composition in ``contrast[1]`` minus ``contrast[0]`` against the expression
    difference (defaults to last sample minus first).
    """
    index = _sample_chrom_index(ds)
    genes = features.table.set_index("gene_id")
    tss = pd.Series(features.tss().to_numpy(), index=features.table["gene_id"])
    a, b = contrast or (ds.samples[0], ds.samples[-1])

    def tasks(comp: pd.DataFrame, cols: List[str]):
        comp = comp.loc[comp["gene_id"].isin(expression.index)].reset_index(drop=True)
        y = np.array([expression.at[g, s] for g, s in zip(comp["gene_id"], comp["sample"])], dtype=float)
        grp = _blocks(tss.loc[comp["gene_id"]].to_numpy(), block_bp)
        X = comp[cols].to_numpy(dtype=float)
        ok = np.isfinite(y)
        p = _cv_predict(X[ok], y[ok], grp[ok], seed)
        out = {"level_rho": _spearman(y[ok], p), "level_n": int(ok.sum())}
        out["level_lo"], out["level_hi"] = _boot_ci(y[ok], p, seed=seed)
        wa = comp.loc[comp["sample"] == a].set_index("gene_id")[cols]
        wb = comp.loc[comp["sample"] == b].set_index("gene_id")[cols]
        common = wa.index.intersection(wb.index)
        dX = (wb.loc[common] - wa.loc[common]).to_numpy(dtype=float)
        dy = (expression.loc[common, b] - expression.loc[common, a]).to_numpy(dtype=float)
        okc = np.isfinite(dy)
        gc = _blocks(tss.loc[common].to_numpy(), block_bp)
        pc = _cv_predict(dX[okc], dy[okc], gc[okc], seed)
        out["change_rho"] = _spearman(dy[okc], pc)
        out["change_lo"], out["change_hi"] = _boot_ci(dy[okc], pc, seed=seed)
        out["change_n"] = int(okc.sum())
        return out

    rows = []
    for run in _ok_runs(runs):
        comp = promoter_composition(ds, run, features, flank, index)
        cols = [c for c in comp.columns if c not in ("gene_id", "sample")]
        rows.append({"tool": run.tool, "method": _short(run), "n_features": len(cols),
                     **tasks(comp, cols)})
    sig = promoter_signal(ds, features, flank, index)
    scols = [c for c in sig.columns if c not in ("gene_id", "sample")]
    rows.append({"tool": "raw_signal", "method": "Raw mark signal (no states)", "n_features": len(scols),
                 **tasks(sig, scols)})
    out = pd.DataFrame(rows)
    out.attrs["contrast"] = (a, b)
    return out


def plot_expression_prediction(table: pd.DataFrame, contrast: Tuple[str, str]):
    fig, axes = plt.subplots(1, 2, figsize=(viz.WIDTH_DOUBLE, 0.7 + 0.32 * len(table)),
                             facecolor=viz.SURFACE)
    names = table["method"].tolist()
    y = np.arange(len(names))
    for ax, key, title in zip(axes, ("level", "change"),
                              ("Expression level",
                               f"Expression change, {contrast[1]} vs {contrast[0]}")):
        viz._style_axes(ax, grid_axis="x")
        v = table[f"{key}_rho"].to_numpy(float)
        lo = table[f"{key}_lo"].to_numpy(float)
        hi = table[f"{key}_hi"].to_numpy(float)
        cols = ["#2a78d6" if n.startswith("ChromState") else ("#6b6b6b" if n.startswith("Raw") else "#b8b6ae")
                for n in names]
        ax.barh(y, np.nan_to_num(v), color=cols, edgecolor=viz.SURFACE, linewidth=0.5, zorder=3)
        ax.errorbar(v, y, xerr=[v - lo, hi - v], fmt="none", ecolor=viz.INK, elinewidth=0.6, capsize=1.5, zorder=4)
        for yi, vv, hh in zip(y, v, hi):
            if viz.ANNOTATE_VALUES and np.isfinite(vv):
                ax.text(max(hh, vv, 0) + 0.015, yi, f"{vv:.2f}", va="center", fontsize=viz.FS_ANNOT)
        ax.tick_params(axis="y", length=0)
        ax.set_yticks(y)
        ax.set_yticklabels(names if key == "level" else [""] * len(names), fontsize=viz.FS_TICK)
        ax.invert_yaxis()
        ax.set_xlim(min(0, np.nanmin(lo) - 0.05), max(0.3, np.nanmax(hi) + 0.05))
        ax.set_xlabel("Spearman, held-out genes", fontsize=viz.FS_LABEL)
        viz._title(ax, title, f"n = {int(table[f'{key}_n'].max()):,} gene x sample" if key == "level"
                   else f"n = {int(table[f'{key}_n'].max()):,} genes")
    fig.tight_layout()
    return fig


def plot_promoter_recovery(table: pd.DataFrame):
    keys = [k for k in ("tss_auprc", "active_tss_auprc") if k in table.columns]
    fig, axes = plt.subplots(1, len(keys), figsize=(viz.WIDTH_SINGLE * len(keys), 0.7 + 0.3 * len(table)),
                             facecolor=viz.SURFACE)
    axes = np.atleast_1d(axes)
    names = table["method"].tolist()
    y = np.arange(len(names))
    titles = {"tss_auprc": ("All promoters", "positives: bins within the TSS flank of any gene"),
              "active_tss_auprc": ("Active promoters", "positives: TSS bins of top-30% expressed genes")}
    for ax, key in zip(axes, keys):
        viz._style_axes(ax, grid_axis="x")
        v = table[key].to_numpy(float)
        cols = ["#2a78d6" if n.startswith("ChromState") else "#b8b6ae" for n in names]
        ax.barh(y, np.nan_to_num(v), color=cols, edgecolor=viz.SURFACE, linewidth=0.5, zorder=3)
        base = float(table[key.replace("auprc", "baseline")].mean())
        ax.axvline(base, color=viz.INK, lw=0.6, ls="--")
        # dashed line: the value a random ranking gives
        for yi, vv in zip(y, v):
            if viz.ANNOTATE_VALUES and np.isfinite(vv):
                ax.text(vv + 0.01, yi, f"{vv:.3f}", va="center", fontsize=viz.FS_ANNOT)
        ax.tick_params(axis="y", length=0)
        ax.set_yticks(y)
        ax.set_yticklabels(names if key == keys[0] else [""] * len(names), fontsize=viz.FS_TICK)
        ax.invert_yaxis()
        ax.set_xlim(0, max(0.2, np.nanmax(v) * 1.1))
        ax.set_xlabel("AUPRC", fontsize=viz.FS_LABEL)
        viz._title(ax, *titles[key])
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Locus view
# ---------------------------------------------------------------------------


def plot_locus(ds, runs: Sequence[ToolRun], chrom: str, start: int, end: int,
               features: Optional[FeatureSet] = None, samples: Optional[Sequence[str]] = None,
               title: Optional[str] = None):
    """Marks on top, one strip per tool and sample below, genes at the bottom."""
    ok = _ok_runs(runs)
    samples = list(samples or ds.samples)
    index = _sample_chrom_index(ds)
    Y = ds.X / np.maximum(ds.size_factors, EPS)
    n_marks = ds.n_marks
    n_rows = n_marks + len(ok) + (1 if features is not None else 0)
    heights = [1.0] * n_marks + [0.45 * len(samples)] * len(ok) + ([0.6] if features is not None else [])
    fig, axes = plt.subplots(n_rows, len(samples), figsize=(viz.WIDTH_DOUBLE, 0.32 * sum(heights) + 0.6),
                             facecolor=viz.SURFACE, sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": heights, "hspace": 0.15, "wspace": 0.08})
    colours = {}
    for run in ok:
        st = state_strengths(ds, run)
        colours[run.tool] = mark_blend_colours(st)
    mark_cols = ["#d7301f", "#f16913", "#3b6fd1", "#7b5ea7", "#2ca25f", "#636363"]
    ymax = {}
    for j, sample in enumerate(samples):
        rows, starts = index[(sample, chrom)]
        sel = (starts >= start) & (starts < end)
        r, x = rows[sel], starts[sel]
        for mi in range(n_marks):
            ymax[mi] = max(ymax.get(mi, 0.0), float(np.percentile(Y[r, mi], 99.5)) if r.size else 1.0)
    for j, sample in enumerate(samples):
        rows, starts = index[(sample, chrom)]
        sel = (starts >= start) & (starts < end)
        r, x = rows[sel], starts[sel]
        for mi in range(n_marks):
            ax = axes[mi, j]
            ax.fill_between(x / 1e6, 0, Y[r, mi], color=mark_cols[mi % len(mark_cols)], lw=0, step="post")
            ax.set_ylim(0, max(ymax[mi], 1e-3) * 1.05)
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if j == 0:
                ax.set_ylabel(ds.marks[mi], fontsize=viz.FS_TICK, rotation=0, ha="right", va="center")
            if mi == 0:
                ax.set_title(sample, fontsize=viz.FS_TITLE, loc="left")
        for ti, run in enumerate(ok):
            ax = axes[n_marks + ti, j]
            cols = colours[run.tool]
            states = run.states[r]
            if states.size:
                change = np.flatnonzero(states[1:] != states[:-1]) + 1
                edges = np.concatenate([[0], change, [states.size]])
                bin_size = int(ds.bins["end"].iloc[0] - ds.bins["start"].iloc[0])
                for a_, b_ in zip(edges[:-1], edges[1:]):
                    k = states[a_]
                    if k == UNASSIGNED:
                        continue
                    ax.axvspan(x[a_] / 1e6, (x[b_ - 1] + bin_size) / 1e6, color=cols[k], lw=0)
            ax.set_yticks([])
            ax.set_ylim(0, 1)
            for s in ax.spines.values():
                s.set_visible(False)
            if j == 0:
                ax.set_ylabel(_short(run), fontsize=viz.FS_TICK, rotation=0, ha="right", va="center")
        if features is not None:
            ax = axes[-1, j]
            g = features.table
            sub = g[(g["chrom"] == chrom) & (g["end"] > start) & (g["start"] < end)]
            for gi, row in enumerate(sub.itertuples()):
                yv = 0.75 - 0.5 * (gi % 2)
                ax.plot([max(row.start, start) / 1e6, min(row.end, end) / 1e6], [yv, yv], color=viz.INK, lw=1.4,
                        solid_capstyle="butt")
                tss = row.start if row.strand == "+" else row.end
                ax.annotate("", xy=((tss + (2000 if row.strand == "+" else -2000)) / 1e6, yv + 0.18),
                            xytext=(tss / 1e6, yv), arrowprops=dict(arrowstyle="->", lw=0.6, color=viz.INK))
                ax.text((max(row.start, start) + min(row.end, end)) / 2e6, yv - 0.22, row.name,
                        fontsize=viz.FS_ANNOT, ha="center", va="top", style="italic")
            ax.set_ylim(-0.3, 1.1)
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_xlabel(f"{chrom} (Mb)", fontsize=viz.FS_LABEL)
            ax.tick_params(labelsize=viz.FS_TICK)
            from matplotlib.ticker import MaxNLocator

            ax.xaxis.set_major_locator(MaxNLocator(4, prune="both"))
    axes[-1, 0].set_xlim(start / 1e6, end / 1e6)
    fig.suptitle(title or f"{chrom}:{start:,}-{end:,}", fontsize=viz.FS_TITLE, x=0.01, ha="left")
    return fig


# ---------------------------------------------------------------------------
# Everything
# ---------------------------------------------------------------------------


def _md(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in frame.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append("—" if not np.isfinite(v) else f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_realdata_report(
    outdir: str,
    ds,
    runs: Sequence[ToolRun],
    features: Optional[FeatureSet] = None,
    expression: Optional[pd.DataFrame] = None,
    flank: int = 1000,
    loci: Sequence[Tuple[str, int, int, str]] = (),
    reference: str = "chromstate",
    contrast: Optional[Tuple[str, str]] = None,
    comparison: Optional[pd.DataFrame] = None,
    dataset_note: str = "",
    seed: int = 0,
) -> List[str]:
    os.makedirs(outdir, exist_ok=True)
    figdir = os.path.join(outdir, "figures")
    written: List[str] = []
    figs: Dict[str, plt.Figure] = {}
    tables: Dict[str, pd.DataFrame] = {}

    f = plot_state_profiles(ds, runs)
    if f is not None:
        figs["state_profiles"] = f
    f = plot_state_fractions(ds, runs)
    if f is not None:
        figs["state_class_fractions"] = f
        tables["class_fractions"] = class_fractions(ds, runs)
    f = plot_segment_lengths(ds, runs)
    if f is not None:
        figs["segment_lengths"] = f
    f = plot_state_overlap(ds, runs, reference)
    if f is not None:
        figs["state_overlap"] = f
    if len(_ok_runs(runs)) > 1:
        from .evaluate import agreement_matrix
        from .report import plot_agreement_heatmap

        agreement = agreement_matrix(list(runs))
        if len(agreement) > 1:
            short = {r.display_name: _short(r) for r in runs}
            agreement = agreement.rename(index=short, columns=short)
            tables["agreement"] = agreement.reset_index().rename(columns={"index": "tool"})
            figs["tool_agreement"] = plot_agreement_heatmap(agreement)
    strengths = []
    for run in _ok_runs(runs):
        st = state_strengths(ds, run)
        st.insert(0, "tool", run.tool)
        strengths.append(st)
    if strengths:
        tables["state_classes"] = pd.concat(strengths, ignore_index=True)

    if features is not None:
        rec = promoter_recovery(ds, runs, features, expression, flank)
        tables["promoter_recovery"] = rec
        figs["promoter_recovery"] = plot_promoter_recovery(rec)
        if expression is not None:
            pred = expression_prediction(ds, runs, features, expression, flank, contrast, seed)
            tables["expression_prediction"] = pred
            figs["expression_prediction"] = plot_expression_prediction(pred, pred.attrs["contrast"])
    for chrom, start, end, name in loci:
        try:
            figs[f"locus_{name}"] = plot_locus(ds, runs, chrom, start, end, features, title=name)
        except KeyError as exc:
            print(f"[realdata] locus {name} skipped: {exc}")

    # PNG copies for quick viewing (the PDFs are the publication copies).
    os.makedirs(os.path.join(figdir, "png"), exist_ok=True)
    for name, fig in figs.items():
        viz.enforce_text_style(fig)
        fig.savefig(os.path.join(figdir, "png", f"{name}.png"), dpi=160, bbox_inches="tight",
                    facecolor=viz.SURFACE)
    written += viz.save_all(figs, figdir)
    for name, tab in tables.items():
        path = os.path.join(outdir, f"{name}.tsv")
        tab.to_csv(path, sep="\t", index=False, float_format="%.5g")
        written.append(path)

    ok = _ok_runs(runs)
    lines = ["# Tool comparison on real data", "", dataset_note, "",
             f"{len(ok)} tool(s) produced a segmentation: " + ", ".join(_short(r) for r in ok) + ".", ""]
    not_ok = [r for r in runs if not r.ok]
    if not_ok:
        lines += ["Not run: " + "; ".join(f"{_short(r)} ({r.status}: {r.reason})" for r in not_ok), ""]
    if comparison is not None:
        show = [c for c in ["method", "n_states", "bin_coverage", "n_segments", "mean_segment_kb",
                            "states_used", "occupancy_entropy", "mean_max_posterior", "agreement_ari",
                            "runtime_s", "peak_rss_mb"] if c in comparison.columns]
        lines += ["## Truth-free metrics", "", _md(comparison.loc[comparison["status"] == "ok", show]), ""]
    if "promoter_recovery" in tables:
        lines += ["## Promoter recovery (cross-validated AUPRC)", "",
                  "Per bin, the score is its state's promoter rate learned on the other half of the "
                  "1 Mb blocks. Chance level is the positive rate.", "",
                  _md(tables["promoter_recovery"]), ""]
    if "expression_prediction" in tables:
        a, b = tables["expression_prediction"].attrs["contrast"]
        lines += ["## Expression from promoter states (held-out Spearman)", "",
                  f"Gradient boosting on the promoter's (TSS +/- {flank:,} bp) state composition, "
                  f"5-fold by 1 Mb block. Change is {b} minus {a}. 'Raw mark signal' is the "
                  "no-segmentation baseline: a segmentation that scores below it has thrown away "
                  "information the marks carried.", "",
                  _md(tables["expression_prediction"][["method", "n_features", "level_rho", "level_lo",
                                                        "level_hi", "change_rho", "change_lo", "change_hi"]]),
                  ""]
    if "agreement" in tables:
        lines += ["## Agreement between tools (adjusted Rand index, bins both called)", "",
                  _md(tables["agreement"]), ""]
    if "class_fractions" in tables:
        wide = tables["class_fractions"].pivot_table(index=["method", "sample"], columns="class",
                                                     values="fraction").reset_index()
        lines += ["## Genome share by state class", "", _md(wide), ""]
    lines += ["## Figures", ""] + [f"- `figures/{n}.pdf`" for n in figs] + [""]
    lines += ["## How to read these", "",
              "- State colours blend each state's own mark enrichment (red: active marks, blue: "
              "H3K27me3, violet: H3K9me3, magenta: both active and H3K27me3), so tools are drawn in "
              "one language without matching their states.",
              "- No row is a score against truth. Each asks how well the tool's states, used "
              "naively, answer a question about genes the tool never saw.",
              "- Agreement between tools is consistency, not correctness.", ""]
    path = os.path.join(outdir, "REPORT.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    written.append(path)
    return written
