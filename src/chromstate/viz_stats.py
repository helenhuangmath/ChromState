"""Statistical figures for the condition comparison (``chromstate.compare``).

Each figure answers one question about the differential report:

``plot_region_state_heatmap``     which states carry the change, region by region
``plot_state_gain_loss``          how many regions gain or lose each state at the FDR
``plot_pvalue_histograms``        is the test calibrated: observed against label-permuted p-values
``plot_replicate_concordance``    replicate-versus-replicate occupancy against condition-versus-condition
``plot_change_ecdf``              the change statistic for replicate pairs against condition pairs
``plot_region_state_volcano``     which state moved and by how much, per region (volcano by state)
``plot_anova_volcano``            three or more conditions: F against q per region and state

The omnibus per-region volcano with the noise floor is :func:`chromstate.viz.plot_change_volcano`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import viz
from .viz import FS_CELL, FS_LABEL, FS_TICK, GRID, INK, WIDTH_DOUBLE, WIDTH_MID, WIDTH_SINGLE, _new_fig, _title

GAIN = "#AE231B"
LOSS = "#395493"
NEUTRAL = "#9E9E9E"


def _signed_logq(frame: pd.DataFrame, cap: float = 6.0) -> np.ndarray:
    q = np.clip(frame["qvalue"].to_numpy(dtype=float), 10 ** (-cap), 1.0)
    return -np.log10(q) * np.sign(frame["delta"].to_numpy(dtype=float))


def plot_region_state_heatmap(region_states: pd.DataFrame, contrast: Optional[str] = None,
                              top: int = 40, seg=None, palette: str = viz.DEFAULT_PALETTE,
                              figsize=None):
    """Signed -log10 q per region (rows) and state (columns): red = gain, blue = loss.

    Rows are the ``top`` regions by omnibus evidence (smallest q over states), ordered by
    hierarchical clustering of the signed profile, so regions that move the same way sit
    together. Cells with q above 0.5 are left white; the dot marks q < 0.05.
    """
    rs = region_states
    if contrast is not None:
        rs = rs.loc[rs["contrast"] == contrast]
    if "qvalue" not in rs.columns:
        raise ValueError("region_states has no qvalue column (no replicates?)")
    states = list(pd.unique(rs["state"]))
    mat = rs.assign(v=_signed_logq(rs)).pivot_table(index="name", columns="state", values="v", dropna=False)
    mat = mat.reindex(columns=states)  # a state never tested (all-NaN q) would otherwise be dropped
    qmin = rs.groupby("name")["qvalue"].min()
    keep = qmin.sort_values().index[:top]
    mat = mat.reindex(keep)
    if len(mat) > 2:
        from scipy.cluster.hierarchy import leaves_list, linkage

        order = leaves_list(linkage(np.nan_to_num(mat.to_numpy()), method="average"))
        mat = mat.iloc[order]
    figsize = figsize or (WIDTH_MID, min(0.09 * len(mat) + 1.2, 7.0))
    fig, ax = _new_fig(*figsize)
    ax.grid(False)
    vmax = max(float(np.nanmax(np.abs(mat.to_numpy()))), 1.3)
    data = mat.to_numpy()
    shown = np.where(np.abs(data) < 0.3, np.nan, data)  # q > 0.5 stays white
    im = ax.imshow(shown, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    sig = np.abs(data) > -np.log10(0.05)
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
    cb.set_label("signed −log10 q  (red gain, blue loss)", fontsize=FS_LABEL)
    cb.ax.tick_params(labelsize=FS_TICK)
    _title(ax, f"Which states change, region by region{' · ' + contrast if contrast else ''}",
           f"top {len(mat)} regions by evidence; dot = q < 0.05")
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
        ax.barh(y, sub["n_sig_gain"], color=GAIN, edgecolor=INK, linewidth=0.3, zorder=3, label="regions gaining")
        ax.barh(y, -sub["n_sig_loss"], color=LOSS, edgecolor=INK, linewidth=0.3, zorder=3, label="regions losing")
        ax2 = ax.twiny()
        ax2.scatter(100 * sub["mean_delta"], y, color=INK, s=8, zorder=4, label="mean change (pp)")
        lim = max(1.0, float(np.abs(100 * sub["mean_delta"]).max()) * 1.3)
        ax2.set_xlim(-lim, lim)
        ax2.tick_params(labelsize=FS_TICK)
        ax2.set_xlabel("mean change over regions (percentage points)", fontsize=FS_LABEL)
        ax.axvline(0, color=INK, linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(viz.state_labels(sub["state"]), fontsize=FS_TICK)
        colors = viz._label_colors(list(sub["state"]), seg, palette)
        for tick, s in zip(ax.get_yticklabels(), sub["state"]):
            tick.set_color(colors[s])
        m = max(1, int(max(sub["n_sig_gain"].max(), sub["n_sig_loss"].max())))
        ax.set_xlim(-1.15 * m, 1.15 * m)
        ax.set_xlabel("regions significant at FDR (loss ← → gain)", fontsize=FS_LABEL)
        ax.set_title(c.replace("_vs_", " → "), fontsize=viz.FS_TITLE, loc="left", pad=22)
        if ax is axes[0][0]:
            ax.legend(fontsize=FS_CELL, loc="lower right", frameon=False)
    fig.suptitle("Regions significant at the FDR (bars) and mean change (dots)", fontsize=viz.FS_SUBTITLE, y=1.02)
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
        n_untested = int((observed["state"] == s).sum() - len(p))
        ax.hist(p, bins=edges, color=GAIN, alpha=0.75, density=True, zorder=3, label="observed")
        if permuted is not None:
            pp = permuted.loc[permuted["state"] == s, "pvalue"].dropna().to_numpy()
            ax.hist(pp, bins=edges, histtype="step", color=INK, linewidth=0.8, density=True, zorder=4,
                    label="permuted labels")
        ax.axhline(1.0, color=NEUTRAL, linewidth=0.5, linestyle="--")
        ax.set_title(f"{s}   n = {len(p):,}" + (f" ({n_untested:,} absent)" if n_untested else ""),
                     fontsize=FS_LABEL, loc="left")
        ax.tick_params(labelsize=FS_TICK)
        if i // ncol == nrow - 1:
            ax.set_xlabel("p-value", fontsize=FS_LABEL)
        if i % ncol == 0:
            ax.set_ylabel("density", fontsize=FS_LABEL)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    axes[0][0].legend(fontsize=FS_CELL, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    fig.suptitle("Test calibration: observed versus label-permuted p-values", fontsize=viz.FS_TITLE, y=1.0)
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
    for ax, (x, y, col, lab) in zip(axes, [(a, b, NEUTRAL, f"{replicate_pair[0]} vs {replicate_pair[1]} (replicates)"),
                                           (c, d, GAIN, f"{condition_pair[0]} vs {condition_pair[1]} (conditions)")]):
        viz._style_axes(ax, grid_axis=None)
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=5, color=col, alpha=0.6, linewidths=0, zorder=3)
        ax.plot([0, 1], [0, 1], color=INK, linewidth=0.5)
        if noise_floor is not None:
            t = float(noise_floor.threshold)
            xx = np.linspace(0, 1, 50)
            ax.fill_between(xx, np.clip(xx - t, 0, 1), np.clip(xx + t, 0, 1), color=LOSS, alpha=0.12,
                            linewidth=0, zorder=2, label=f"noise floor ±{t:.2f}")
            out = np.abs(x[ok] - y[ok]) > t
            ax.text(0.03, 0.93, f"{out.mean() * 100:.1f}% of regions outside the floor", transform=ax.transAxes,
                    fontsize=FS_CELL, color=INK)
        rho = pd.Series(x[ok]).corr(pd.Series(y[ok]), method="spearman") if ok.sum() > 2 else np.nan
        ax.set_xlabel(f"{state} share, first sample", fontsize=FS_LABEL)
        ax.set_ylabel(f"{state} share, second sample", fontsize=FS_LABEL)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=FS_TICK)
        _title(ax, lab, f"Spearman ρ = {rho:.3f}, n = {int(ok.sum()):,} regions")
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
    for c, reps in groups.items():
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                d = tv(reps[i], reps[j])
                ax.step(d, np.arange(1, len(d) + 1) / len(d), color=NEUTRAL, linewidth=0.7,
                        label="replicate pairs" if first else None, zorder=3)
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
                   label=f"noise floor {noise_floor.threshold:.2f}")
    ax.set_xlabel("total-variation distance of region composition", fontsize=FS_LABEL)
    ax.set_ylabel("fraction of regions", fontsize=FS_LABEL)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=FS_CELL, frameon=False, loc="lower right")
    _title(ax, "How much change replicates produce, against conditions",
           f"{len(regions):,} regions; one curve per sample pair")
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
    y = -np.log10(np.clip(rs["qvalue"].to_numpy(dtype=float), 1e-12, 1))
    sig = rs["significant"].to_numpy(dtype=bool) if "significant" in rs.columns else (rs["qvalue"] < fdr).to_numpy()
    for s_ in states:
        m = (rs["state"] == s_).to_numpy()
        ax.scatter(x[m & ~sig], y[m & ~sig], s=3, color=colors[s_], alpha=0.25, linewidths=0, zorder=2)
        ax.scatter(x[m & sig], y[m & sig], s=6, color=colors[s_], alpha=0.9, linewidths=0, zorder=3,
                   label=f"{viz.state_labels([s_])[0]} ({int((m & sig).sum())})")
    ax.axhline(-np.log10(fdr), color=INK, linewidth=0.5, linestyle="--")
    if min_delta > 0:
        for v in (-100 * min_delta, 100 * min_delta):
            ax.axvline(v, color=INK, linewidth=0.5, linestyle=":")
    ax.axvline(0, color=NEUTRAL, linewidth=0.4)
    top = rs.loc[sig].assign(_y=y[sig], _x=x[sig]).sort_values(["_y", "_x"], ascending=False).head(label_top)
    viz._stagger_labels(ax, [(r["_x"], r["_y"], str(r["name"])) for _, r in top.iterrows()], fontsize=FS_CELL)
    ax.set_xlabel("change in state share (percentage points)", fontsize=FS_LABEL)
    ax.set_ylabel("−log10 q (BH within state)", fontsize=FS_LABEL)
    lim = max(5.0, float(np.abs(x).max()) * 1.05)
    ax.set_xlim(-lim, lim)
    ax.legend(fontsize=FS_CELL, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              title="significant regions per state", title_fontsize=FS_CELL)
    _title(ax, f"Which state moved, and by how much{' · ' + contrast if contrast else ''}",
           f"{int(sig.sum()):,} region x state pairs pass FDR {fdr:g}; dashed = FDR line")
    fig.tight_layout()
    return fig


def plot_anova_volcano(anova: pd.DataFrame, fdr: float = 0.05, seg=None, palette: str = viz.DEFAULT_PALETTE,
                       label_top: int = 3, figsize=None):
    """Moderated F against -log10 q for every region x state, coloured by state; points
    above the FDR line are regions where the state's share differs among conditions, and
    the top ones are labelled with the condition that deviates most."""
    share_cols = [c for c in anova.columns if c.startswith("share_")]
    states = list(pd.unique(anova["state"]))
    colors = viz._label_colors(states, seg, palette)
    figsize = figsize or (WIDTH_SINGLE, 3.3)
    fig, ax = _new_fig(*figsize)
    viz._style_axes(ax, grid_axis="both")
    q = np.clip(anova["qvalue"].to_numpy(dtype=float), 1e-12, 1)
    F = anova["F"].to_numpy(dtype=float)
    for s in states:
        m = (anova["state"] == s).to_numpy()
        ax.scatter(np.log10(F[m] + 1e-9), -np.log10(q[m]), s=5, color=colors[s], alpha=0.7, linewidths=0,
                   label=viz.state_labels([s])[0], zorder=3)
    ax.axhline(-np.log10(fdr), color=INK, linewidth=0.5, linestyle="--")
    top = anova.sort_values("qvalue").head(label_top)
    pts = []
    for _, r in top.iterrows():
        shares = r[share_cols].astype(float)
        dev = (shares - shares.median()).abs().idxmax().replace("share_", "")
        pts.append((np.log10(r["F"] + 1e-9), -np.log10(max(r["qvalue"], 1e-12)), f"{r['name']} · {r['state']} · {dev}"))
    viz._stagger_labels(ax, pts, fontsize=FS_CELL)
    ax.set_xlabel("log10 moderated F", fontsize=FS_LABEL)
    ax.set_ylabel("−log10 q", fontsize=FS_LABEL)
    ax.set_xlim(max(-2.0, float(np.nanmin(np.log10(F + 1e-9)))), None)
    n_sig = int((anova["qvalue"] < fdr).sum())
    ax.legend(fontsize=FS_CELL, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    _title(ax, "Which region x state pairs differ among conditions",
           f"{n_sig:,} of {len(anova):,} pass FDR {fdr:g}; label = most deviant condition")
    fig.tight_layout()
    return fig
