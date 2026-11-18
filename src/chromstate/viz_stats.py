"""Statistical figures for the condition comparison (``chromstate.compare``).

Each figure answers one question about the differential report:

``plot_region_state_heatmap``     which states carry the change, region by region
``plot_state_gain_loss``          how many regions gain or lose each state at the FDR
``plot_pvalue_histograms``        is the test calibrated: observed against label-permuted p-values
``plot_replicate_concordance``    replicate-versus-replicate occupancy against condition-versus-condition
``plot_change_ecdf``              the change statistic for replicate pairs against condition pairs
``plot_region_state_volcano``     which state moved and by how much, per region (volcano by state)
``plot_anova_volcano``            three or more conditions: largest difference against q per region and state

The omnibus per-region volcano with the noise floor is :func:`chromstate.viz.plot_change_volcano`.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import viz
from .viz import (
    FS_CELL,
    FS_LABEL,
    FS_TICK,
    INK,
    WIDTH_DOUBLE,
    WIDTH_MID,
    WIDTH_SINGLE,
    _new_fig,
    _title,
)

GAIN = "#AE231B"
LOSS = "#395493"
NEUTRAL = "#9E9E9E"


def plot_region_state_heatmap(region_states: pd.DataFrame, contrast: Optional[str] = None,
                              top: int = 40, seg=None, palette: str = viz.DEFAULT_PALETTE,
                              figsize=None):
    """Change in share (percentage points) per region (rows) and state (columns): red =
    gain, blue = loss; a dot marks q < 0.05.

    Rows are the ``top`` regions by evidence (smallest q over states, ties broken by the
    size of the change), ordered by hierarchical clustering so regions that move the same
    way sit together. Colour is the effect size and not the q-value: q saturates for
    every clear change, which would paint the whole column one colour.
    """
    rs = region_states
    if contrast is not None:
        rs = rs.loc[rs["contrast"] == contrast]
    if "qvalue" not in rs.columns:
        raise ValueError("region_states has no qvalue column (no replicates?)")
    states = list(pd.unique(rs["state"]))
    mat = rs.assign(v=100 * rs["delta"].astype(float)).pivot_table(
        index="name", columns="state", values="v", dropna=False)
    mat = mat.reindex(columns=states)  # a state never tested (all-NaN q) would otherwise be dropped
    qmat = rs.pivot_table(index="name", columns="state", values="qvalue", dropna=False).reindex(columns=states)
    rank = pd.DataFrame({"q": rs.groupby("name")["qvalue"].min(),
                         "d": rs.groupby("name")["delta"].apply(lambda v: -float(np.abs(v).max()))})
    keep = rank.sort_values(["q", "d"]).index[:top]
    mat = mat.reindex(keep)
    if len(mat) > 2:
        from scipy.cluster.hierarchy import leaves_list, linkage

        order = leaves_list(linkage(np.nan_to_num(mat.to_numpy()), method="average"))
        mat = mat.iloc[order]
    figsize = figsize or (WIDTH_MID, min(0.09 * len(mat) + 1.2, 7.0))
    fig, ax = _new_fig(*figsize)
    ax.grid(False)
    vmax = max(float(np.nanmax(np.abs(mat.to_numpy()))), 5.0)
    data = mat.to_numpy()
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    sig = qmat.reindex(mat.index).to_numpy(dtype=float) < 0.05
    yy, xx = np.nonzero(sig)
    ax.scatter(xx, yy, s=3, color=INK, linewidths=0, zorder=3)
    ax.set_xticks(range(len(states)))
    colors = viz._label_colors(states, seg, palette)
    ax.set_xticklabels(viz.state_labels(states), rotation=90, fontsize=FS_TICK)
    for tick, s in zip(ax.get_xticklabels(), states):
        tick.set_color(colors[s])
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels([str(n)[:18] for n in mat.index], fontsize=FS_CELL)
    ax.tick_params(axis="y", length=1.5, pad=1.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Change in share (points)", fontsize=FS_LABEL)
    cb.ax.tick_params(labelsize=FS_TICK)
    _title(ax, (contrast or "").replace("_vs_", " → "))
    fig.tight_layout()
    return fig


def plot_state_gain_loss(states: pd.DataFrame, seg=None, palette: str = viz.DEFAULT_PALETTE, figsize=None):
    """Regions gaining (right, red) and losing (left, blue) each state at the FDR, per
    contrast, with the mean change over all regions as a dot (share units x 100)."""
    if "n_sig_gain" not in states.columns:
        raise ValueError("states table has no significance columns (no replicates?)")
    contrasts = list(pd.unique(states["contrast"]))
    figsize = figsize or (WIDTH_SINGLE * len(contrasts), 0.22 * states["state"].nunique() + 1.3)
    fig, axes = plt.subplots(1, len(contrasts), figsize=figsize, facecolor=viz.SURFACE, squeeze=False)
    for ax, c in zip(axes[0], contrasts):
        viz._style_axes(ax, grid_axis="x")
        sub = states.loc[states["contrast"] == c].reset_index(drop=True)
        y = np.arange(len(sub))
        ax.barh(y, sub["n_sig_gain"], color=GAIN, linewidth=0, zorder=3, label="Gain")
        ax.barh(y, -sub["n_sig_loss"], color=LOSS, linewidth=0, zorder=3, label="Loss")
        ax.axvline(0, color=INK, linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(viz.state_labels(sub["state"]), fontsize=FS_TICK)
        colors = viz._label_colors(list(sub["state"]), seg, palette)
        for tick, s in zip(ax.get_yticklabels(), sub["state"]):
            tick.set_color(colors[s])
        m = max(1, int(max(sub["n_sig_gain"].max(), sub["n_sig_loss"].max())))
        ax.set_xlim(-1.15 * m, 1.15 * m)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels([f"{abs(v):g}" for v in ax.get_xticks()])
        ax.set_xlim(-1.15 * m, 1.15 * m)
        ax.set_xlabel("Significant regions", fontsize=FS_LABEL)
        ax.set_title(c.replace("_vs_", " → "), fontsize=viz.FS_TITLE, loc="left")
        if ax is not axes[0][0]:
            ax.set_yticklabels([])
        if ax is axes[0][0]:
            ax.legend(fontsize=FS_CELL, loc="lower right", frameon=False)
    fig.tight_layout()
    return fig


def plot_pvalue_histograms(observed: pd.DataFrame, permuted: Optional[pd.DataFrame] = None,
                           states: Optional[Sequence[str]] = None, bins: int = 20, figsize=None):
    """Per state, the distribution of region p-values for the real contrast (filled) and
    for a label-permuted contrast (outline). A calibrated test is flat under permutation;
    real signal shows as a spike near zero. Regions where the state is absent in both
    groups are not tested and are counted in the panel title instead."""
    states = list(states or pd.unique(observed["state"]))
    n = len(states)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    figsize = figsize or (WIDTH_DOUBLE, 1.4 * nrow + 0.4)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, facecolor=viz.SURFACE, squeeze=False, sharex=True)
    edges = np.linspace(0, 1, bins + 1)
    for i, s in enumerate(states):
        ax = axes[i // ncol][i % ncol]
        viz._style_axes(ax, grid_axis="y")
        p = observed.loc[observed["state"] == s, "pvalue"].dropna().to_numpy()
        ax.hist(p, bins=edges, color=GAIN, alpha=0.75, density=True, zorder=3, label="Observed")
        if permuted is not None:
            pp = permuted.loc[permuted["state"] == s, "pvalue"].dropna().to_numpy()
            ax.hist(pp, bins=edges, histtype="step", color=INK, linewidth=0.8, density=True, zorder=4,
                    label="Permuted")
        ax.axhline(1.0, color=NEUTRAL, linewidth=0.5, linestyle="--")
        ax.set_title(viz.state_labels([s])[0], fontsize=FS_LABEL, loc="left")
        ax.tick_params(labelsize=FS_TICK)
        if i // ncol == nrow - 1:
            ax.set_xlabel("P value", fontsize=FS_LABEL)
        if i % ncol == 0:
            ax.set_ylabel("Density", fontsize=FS_LABEL)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    axes[0][0].legend(fontsize=FS_CELL, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    fig.tight_layout()
    return fig


def region_occupancy(seg, regions: pd.DataFrame, samples: Sequence[str], min_bins: int = 1
                     ) -> Dict[str, np.ndarray]:
    """``sample -> (regions, states)`` mean posterior per region; the regions table order."""
    from .dynamics import _region_membership

    ds = seg.dataset
    out = {}
    for s in samples:
        m = ds.sample_mask(s)
        bins = ds.bins.loc[m, ["chrom", "start", "end"]].reset_index(drop=True)
        members = _region_membership(bins, regions)
        post = seg.posterior[m]
        out[s] = np.stack([post[idx].mean(axis=0) if len(idx) >= min_bins else np.full(post.shape[1], np.nan)
                           for idx in members])
    return out


def plot_replicate_concordance(seg, regions: pd.DataFrame, state: str, replicate_pair: Tuple[str, str],
                               condition_pair: Tuple[str, str], noise_floor=None, figsize=None):
    """Occupancy of one state per region: replicate against replicate (grey) and condition
    against condition (red). The shaded band is the replicate noise floor on the
    difference, so red points outside it are changes replicates do not produce."""
    k = list(seg.labels).index(state)
    occ = region_occupancy(seg, regions, list(replicate_pair) + list(condition_pair))
    a, b = occ[replicate_pair[0]][:, k], occ[replicate_pair[1]][:, k]
    c, d = occ[condition_pair[0]][:, k], occ[condition_pair[1]][:, k]
    figsize = figsize or (WIDTH_DOUBLE, WIDTH_SINGLE * 0.9)
    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor=viz.SURFACE)
    for ax, (x, y, col, lab, names) in zip(axes, [(a, b, NEUTRAL, "Replicates", replicate_pair),
                                                  (c, d, GAIN, "Conditions", condition_pair)]):
        viz._style_axes(ax, grid_axis=None)
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=5, color=col, alpha=0.6, linewidths=0, zorder=3)
        ax.plot([0, 1], [0, 1], color=INK, linewidth=0.5)
        if noise_floor is not None:
            t = float(noise_floor.threshold)
            xx = np.linspace(0, 1, 50)
            ax.fill_between(xx, np.clip(xx - t, 0, 1), np.clip(xx + t, 0, 1), color=LOSS, alpha=0.12,
                            linewidth=0, zorder=2, label="Noise floor")
        rho = pd.Series(x[ok]).corr(pd.Series(y[ok]), method="spearman") if ok.sum() > 2 else np.nan
        ax.set_xlabel(f"{state} share, {names[0]}", fontsize=FS_LABEL)
        ax.set_ylabel(f"{state} share, {names[1]}", fontsize=FS_LABEL)
        if np.isfinite(rho):
            ax.text(0.04, 0.93, f"ρ = {rho:.2f}", transform=ax.transAxes, fontsize=FS_CELL, color=INK)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=FS_TICK)
        _title(ax, lab)
        if noise_floor is not None:
            ax.legend(fontsize=FS_CELL, frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def plot_change_ecdf(seg, groups: Dict[str, Sequence[str]], regions: pd.DataFrame, reference: Optional[str] = None,
                     noise_floor=None, figsize=None):
    """Per-region total-variation distance between two samples: every replicate pair
    (grey) and every reference-versus-condition pair (coloured), as ECDFs. The
    horizontal gap at a given distance is the fraction of regions whose change replicates
    never show."""
    groups = {c: list(v) for c, v in groups.items()}
    reference = reference or next(iter(groups))
    samples = sorted({s for v in groups.values() for s in v})
    occ = region_occupancy(seg, regions, samples)

    def tv(s1, s2):
        d = 0.5 * np.abs(occ[s1] - occ[s2]).sum(axis=1)
        return np.sort(d[np.isfinite(d)])

    figsize = figsize or (WIDTH_SINGLE, 2.6)
    fig, ax = _new_fig(*figsize)
    viz._style_axes(ax, grid_axis="both")
    first = True
    for reps in groups.values():
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                d = tv(reps[i], reps[j])
                ax.step(d, np.arange(1, len(d) + 1) / len(d), color=NEUTRAL, linewidth=0.7,
                        label="Replicate pairs" if first else None, zorder=3)
                first = False
    palette = plt.get_cmap("tab10")
    for n, (c, reps) in enumerate([(c, r) for c, r in groups.items() if c != reference]):
        first_c = True
        for s1 in groups[reference]:
            for s2 in reps:
                d = tv(s1, s2)
                ax.step(d, np.arange(1, len(d) + 1) / len(d), color=palette(n), linewidth=0.9,
                        label=f"{reference} vs {c}" if first_c else None, zorder=4)
                first_c = False
    if noise_floor is not None:
        ax.axvline(float(noise_floor.threshold), color=LOSS, linewidth=0.7, linestyle="--",
                   label="Noise floor")
    ax.set_xlabel("Change in state composition", fontsize=FS_LABEL)
    ax.set_ylabel("Fraction of regions", fontsize=FS_LABEL)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=FS_CELL, frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def plot_region_state_volcano(region_states: pd.DataFrame, contrast: Optional[str] = None, fdr: float = 0.05,
                              seg=None, palette: str = viz.DEFAULT_PALETTE, label_top: int = 3,
                              min_delta: float = 0.0, figsize=None):
    """Volcano per state: change in a state's share (percentage points) against -log10 q,
    one point per region and state, coloured by state. The omnibus volcano
    (:func:`chromstate.viz.plot_change_volcano`) asks *whether* a region changed; this
    one asks *which state* moved and *by how much*."""
    rs = region_states
    if contrast is not None:
        rs = rs.loc[rs["contrast"] == contrast]
    rs = rs.loc[rs["qvalue"].notna()] if "qvalue" in rs.columns else rs
    if rs.empty:
        raise ValueError("no tested region x state rows")
    states = list(pd.unique(rs["state"]))
    colors = viz._label_colors(states, seg, palette)
    figsize = figsize or (WIDTH_SINGLE, 3.4)
    fig, ax = _new_fig(*figsize)
    viz._style_axes(ax, grid_axis="both")
    x = 100 * rs["delta"].to_numpy(dtype=float)
    y = -np.log10(np.clip(rs["qvalue"].to_numpy(dtype=float), 1e-300, 1))
    sig = rs["significant"].to_numpy(dtype=bool) if "significant" in rs.columns else (rs["qvalue"] < fdr).to_numpy()
    for s_ in states:
        m = (rs["state"] == s_).to_numpy()
        ax.scatter(x[m & ~sig], y[m & ~sig], s=3, color=colors[s_], alpha=0.25, linewidths=0, zorder=2)
        ax.scatter(x[m & sig], y[m & sig], s=6, color=colors[s_], alpha=0.9, linewidths=0, zorder=3,
                   label=viz.state_labels([s_])[0])
    ax.axhline(-np.log10(fdr), color=INK, linewidth=0.5, linestyle="--")
    if min_delta > 0:
        for v in (-100 * min_delta, 100 * min_delta):
            ax.axvline(v, color=INK, linewidth=0.5, linestyle=":")
    ax.axvline(0, color=NEUTRAL, linewidth=0.4)
    top = rs.loc[sig].assign(_y=y[sig], _x=x[sig]).sort_values(["_y", "_x"], ascending=False).head(label_top)
    viz._stagger_labels(ax, [(r["_x"], r["_y"], str(r["name"])) for _, r in top.iterrows()], fontsize=FS_CELL)
    ax.set_xlabel("Change in state share (points)", fontsize=FS_LABEL)
    ax.set_ylabel("−log10 q", fontsize=FS_LABEL)
    lim = max(5.0, float(np.abs(x).max()) * 1.05)
    ax.set_xlim(-lim, lim)
    ax.legend(fontsize=FS_CELL, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    _title(ax, (contrast or "").replace("_vs_", " → "))
    fig.tight_layout()
    return fig


def plot_anova_volcano(anova: pd.DataFrame, fdr: float = 0.05, seg=None, palette: str = viz.DEFAULT_PALETTE,
                       label_top: int = 3, figsize=None):
    """Three or more conditions: the largest difference in a state's share between any
    two conditions (percentage points) against -log10 q of the moderated F-test, one
    point per region and state, coloured by state.

    The x-axis is an effect size and not F itself: q is a monotone function of F, so F
    against q is a single curve per state and shows nothing.
    """
    share_cols = [c for c in anova.columns if c.startswith("share_")]
    anova = anova.loc[anova["qvalue"].notna()]
    states = list(pd.unique(anova["state"]))
    colors = viz._label_colors(states, seg, palette)
    figsize = figsize or (WIDTH_SINGLE, 3.3)
    fig, ax = _new_fig(*figsize)
    viz._style_axes(ax, grid_axis="both")
    y = -np.log10(np.clip(anova["qvalue"].to_numpy(dtype=float), 1e-300, 1))
    shares = anova[share_cols].to_numpy(dtype=float)
    x = 100 * (shares.max(axis=1) - shares.min(axis=1))
    sig = y > -np.log10(fdr)
    for s in states:
        m = (anova["state"] == s).to_numpy()
        ax.scatter(x[m & ~sig], y[m & ~sig], s=3, color=colors[s], alpha=0.25, linewidths=0, zorder=2)
        ax.scatter(x[m & sig], y[m & sig], s=6, color=colors[s], alpha=0.9, linewidths=0, zorder=3,
                   label=viz.state_labels([s])[0])
    ax.axhline(-np.log10(fdr), color=INK, linewidth=0.5, linestyle="--")
    order = np.lexsort((-x, -y))[:label_top]
    viz._stagger_labels(ax, [(x[i], y[i], str(anova["name"].iloc[i])) for i in order], fontsize=FS_CELL)
    ax.set_xlabel("Largest difference (points)", fontsize=FS_LABEL)
    ax.set_ylabel("−log10 q", fontsize=FS_LABEL)
    ax.set_xlim(0, None)
    ax.legend(fontsize=FS_CELL, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    return fig
