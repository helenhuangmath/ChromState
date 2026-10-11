"""Figures: the ChromHMM-standard plots plus the ones ChromState adds.

Colour policy
-------------
Two categorical palettes ship, and the choice is a real tradeoff rather than a preference:

``roadmap`` (default)
    The Roadmap Epigenomics scheme every ChromHMM user already reads at a glance -- red
    promoters, yellow enhancers, green transcription, grey Polycomb. It is *not*
    colourblind-safe, and not marginally so: TssA (``#FF0000``) and TssFlnk (``#FF4500``)
    sit at ΔE 3.0 for normal vision and 3.0 under deuteranopia, far below the ΔE 15 floor
    at which two colours are reliably distinguishable at all. Several states also fall
    below 3:1 contrast on white. It is the default only because field recognisability has
    real value, and every plot here carries direct labels so colour never carries meaning
    alone.

``cvd``
    Eight hues in a fixed, validated order (worst adjacent pair ΔE 9.1 under protanopia,
    19.6 normal vision). Use for anything going into a talk, a poster, or a paper figure.
    Capped at 8 states; beyond that, states fold into "Other" rather than cycling hues.

Continuous scales follow the same discipline: magnitude uses a single blue hue
light-to-dark, and signed quantities (attribution log-ratios, occupancy changes) use a
blue-grey-red diverging scale with a neutral midpoint, so zero reads as nothing.

Type, size and wording
----------------------
Figures follow Nature's artwork guidelines: drawn at final print size (89 mm single
column, 120 mm one-and-a-half, 183 mm double), Arial at 5-7 pt, every piece of text
black, lines 0.25-1 pt, white background, and saved as vector PDF with the text kept as
editable TrueType. Colour is used for data marks only, never for text. Labels spell things out -- states print as "Bivalent transcription start site"
rather than ``TssBiv``, axes as "held-out log-likelihood per bin" rather than
"log-lik/bin" -- because a figure is read by people who do not have the mnemonic table
in front of them, and often out of the context of the paper that defines it.
"""

from __future__ import annotations

import textwrap
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

from .model.annotate import STATE_COLORS, full_state_name
from .model.core import Segmentation

# -- type --------------------------------------------------------------------

#: Arial first, then faces with the same metrics, so a missing Arial changes the glyphs
#: but never the line breaks or the figure size.
FONT_STACK = [
    "Arial",
    "Helvetica",
    "Helvetica Neue",
    "Liberation Sans",
    "Nimbus Sans",
    "Arimo",
    "DejaVu Sans",
]


# -- type scale --------------------------------------------------------------
#
# One scale, used by every figure: Nature's 5-7 pt range at final print size. Heatmap
# cell numbers sit at the 5 pt floor -- they must fit inside a cell and are a redundant
# readout of the colour rather than the message.
FS_TITLE = 7.0
FS_SUBTITLE = 6.0
FS_LABEL = 6.0
FS_TICK = 5.5
FS_LEGEND = 5.5
FS_ANNOT = 5.5
FS_CELL = 5.0

MM = 1 / 25.4
#: Nature figure widths in inches: single column, one-and-a-half column, double column.
WIDTH_SINGLE = 89 * MM
WIDTH_MID = 120 * MM
WIDTH_DOUBLE = 183 * MM


def use_arial() -> None:
    """Set Arial (with fallbacks) and the type scale for every figure this module draws.

    Called once on import. Text is also embedded as real fonts rather than paths in PDF
    and EPS output (``fonttype`` 42), so journal production systems can restyle it.
    """
    existing = [f for f in mpl.rcParams["font.sans-serif"] if f not in FONT_STACK]
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = FONT_STACK + existing
    # Keep maths (rho, Delta, superscripts) in a sans face too, so it does not fall back
    # to a serif italic in the middle of an Arial label.
    mpl.rcParams["mathtext.fontset"] = "dejavusans"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"
    # Defaults, so anything drawn without an explicit size still lands on the scale.
    mpl.rcParams["font.size"] = FS_ANNOT
    mpl.rcParams["axes.titlesize"] = FS_TITLE
    mpl.rcParams["axes.labelsize"] = FS_LABEL
    mpl.rcParams["xtick.labelsize"] = FS_TICK
    mpl.rcParams["ytick.labelsize"] = FS_TICK
    mpl.rcParams["legend.fontsize"] = FS_LEGEND
    mpl.rcParams["figure.titlesize"] = FS_TITLE
    # Nature house style: black text and axes, thin lines, no top/right spines.
    for key in ("text.color", "axes.labelcolor", "axes.edgecolor", "xtick.color",
                "ytick.color", "legend.labelcolor"):
        mpl.rcParams[key] = "#000000"
    mpl.rcParams["axes.linewidth"] = 0.5
    mpl.rcParams["lines.linewidth"] = 0.75
    mpl.rcParams["patch.linewidth"] = 0.5
    mpl.rcParams["xtick.major.width"] = 0.5
    mpl.rcParams["ytick.major.width"] = 0.5
    mpl.rcParams["xtick.major.size"] = 2.0
    mpl.rcParams["ytick.major.size"] = 2.0
    mpl.rcParams["axes.titlepad"] = 3.0
    mpl.rcParams["axes.labelpad"] = 2.0
    mpl.rcParams["legend.frameon"] = False


use_arial()


def _wrap_lines(text: str, width: int) -> List[str]:
    """Wrap to ``width`` characters, honouring newlines the caller already chose."""
    out: List[str] = []
    for part in text.split("\n"):
        out.extend(textwrap.wrap(part, width=width) or [""])
    return out


def state_labels(labels: Sequence[str]) -> List[str]:
    """Spell out state mnemonics for display (``TssBiv`` -> bivalent promoter, etc.)."""
    return [full_state_name(str(lbl)) for lbl in labels]


# -- chrome ------------------------------------------------------------------

# All text is black (Nature style); INK2/MUTED remain as names so reference lines and
# swatches keep their roles, but they render black as well.
INK = "#000000"
INK2 = "#000000"
MUTED = "#000000"
GRID = "#e6e6e6"
AXIS = "#000000"
SURFACE = "#ffffff"

# Validated colourblind-safe categorical order (see module docstring).
CVD_PALETTE = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]

# Light-to-mid blue only: cell values are printed in black, so the darkest step must
# still give black text at least 4.5:1 contrast.
_BLUE_RAMP = [
    "#f2f7fe", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5",
]
SEQ_CMAP = LinearSegmentedColormap.from_list("chromstate_seq", _BLUE_RAMP)
DIV_CMAP = LinearSegmentedColormap.from_list(
    "chromstate_div",
    ["#3987e5", "#9ec5f4", "#f0efec", "#f2a09f", "#e66767"],
)


def _style_axes(ax: plt.Axes, grid_axis: Optional[str] = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.5)
    ax.tick_params(colors=INK, labelsize=FS_TICK, length=2, width=0.5)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.3, zorder=0)
        ax.set_axisbelow(True)


def _new_fig(w: float, h: float) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE)
    _style_axes(ax)
    return fig, ax


def _title(ax: plt.Axes, text: str, subtitle: Optional[str] = None) -> None:
    """Left-aligned title with an optional one-line explanation wrapped to the panel width.

    Wrapping matters at print size: an unwrapped explanation runs past the panel, and a
    tight bounding box would then silently widen the figure beyond the journal column.
    """
    lines: List[str] = []
    if subtitle:
        fig = ax.figure
        width_in = ax.get_position().width * fig.get_figwidth()
        chars = max(20, int(width_in * 72 / (FS_SUBTITLE * 0.5)))
        lines = _wrap_lines(subtitle, chars)
    pad = 3 + (FS_SUBTITLE * 1.25) * len(lines)
    ax.set_title(text, color=INK, fontsize=FS_TITLE, loc="left", pad=pad)
    if lines:
        ax.text(
            0.0, 1.0, "\n".join(lines), transform=ax.transAxes,
            color=INK, fontsize=FS_SUBTITLE, va="bottom", ha="left",
            linespacing=1.15,
        )


def state_colors(seg: Segmentation, palette: str = "roadmap") -> List[str]:
    """Resolve per-state colours under the requested palette."""
    if palette == "roadmap":
        return list(seg.colors)
    if palette == "cvd":
        if seg.n_states > len(CVD_PALETTE):
            # Never cycle hues: states past the cap share the "Other" grey and rely on
            # their direct labels.
            return [
                CVD_PALETTE[i] if i < len(CVD_PALETTE) else "#b8b6ae"
                for i in range(seg.n_states)
            ]
        return CVD_PALETTE[: seg.n_states]
    raise ValueError(f"Unknown palette {palette!r}; expected 'roadmap' or 'cvd'")


# ---------------------------------------------------------------------------
# ChromHMM-standard figures
# ---------------------------------------------------------------------------


def plot_emission_heatmap(seg: Segmentation, normalize: bool = True, figsize=(WIDTH_SINGLE, 3.0)):
    """States x marks emission heat map -- ChromHMM's signature figure.

    With ``normalize``, each mark column is scaled to its own maximum, so marks with very
    different antibody efficiency and depth are comparable. The printed cell values stay
    on the original count scale, because that magnitude is the thing ChromState keeps and
    a binarised model throws away.
    """
    E = seg.emission_frame()
    Z = E / E.max(axis=0).replace(0, 1) if normalize else E

    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    im = ax.imshow(Z.values, cmap=SEQ_CMAP, aspect="auto", vmin=0, vmax=1 if normalize else None)

    ax.set_xticks(range(len(E.columns)))
    ax.set_xticklabels(E.columns, rotation=45, ha="right", fontsize=FS_TICK, color=INK2)
    ax.set_yticks(range(len(E.index)))
    ax.set_yticklabels(state_labels(E.index), fontsize=FS_TICK, color=INK2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 2px surface gap between cells.
    ax.set_xticks(np.arange(-0.5, len(E.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(E.index), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            ax.text(
                j, i, f"{E.values[i, j]:.0f}",
                ha="center", va="center", fontsize=FS_CELL,
                color=INK,
            )

    ax.set_xlabel("histone mark", color=INK2, fontsize=FS_LABEL)
    ax.set_ylabel("chromatin state", color=INK2, fontsize=FS_LABEL)
    _title(ax, "Emission parameters",
           "cell colour = fraction of that mark's maximum across states; "
           "number = mean read count per 200 base-pair bin")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(colors=INK, labelsize=FS_TICK, length=1.5, width=0.4)
    cb.outline.set_visible(False)
    fig.tight_layout()
    return fig


def plot_transition_heatmap(seg: Segmentation, figsize=(WIDTH_MID, 3.9)):
    """State-to-state transition probabilities along the genome."""
    T = seg.transition_frame()
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    im = ax.imshow(T.values, cmap=SEQ_CMAP, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(T.columns)))
    ax.set_xticklabels(state_labels(T.columns), rotation=45, ha="right", fontsize=FS_TICK, color=INK2)
    ax.set_yticks(range(len(T.index)))
    ax.set_yticklabels(state_labels(T.index), fontsize=FS_TICK, color=INK2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(T.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(T.index), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            v = T.values[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=FS_CELL,
                        color=INK)

    ax.set_xlabel("state in the next bin", color=INK2, fontsize=FS_LABEL)
    ax.set_ylabel("state in the current bin", color=INK2, fontsize=FS_LABEL)
    _title(ax, "Transition probabilities",
           "strong diagonal = states form extended domains rather than alternating bin to bin")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(colors=INK, labelsize=FS_TICK, length=1.5, width=0.4)
    cb.outline.set_visible(False)
    fig.tight_layout()
    return fig


def plot_state_coverage(seg: Segmentation, palette: str = "roadmap", figsize=(WIDTH_MID, 2.6)):
    """Genome coverage per state, hard assignment versus soft posterior mass.

    The gap between the two bars is informative: a state whose soft mass exceeds its hard
    coverage is one the model keeps considering but rarely commits to -- typically a
    transitional or ambiguous state that a Viterbi segmentation makes invisible.
    """
    tab = seg.state_table()
    colors = state_colors(seg, palette)
    x = np.arange(len(tab))
    w = 0.38

    fig, ax = _new_fig(*figsize)
    ax.bar(x - w / 2 - 0.01, tab["coverage_hard"], w, color=colors, edgecolor=SURFACE,
           linewidth=0.5, label="hard assignment (Viterbi path)", zorder=3)
    ax.bar(x + w / 2 + 0.01, tab["coverage_soft"], w, color=colors, edgecolor=SURFACE,
           linewidth=0.5, alpha=0.45, label="soft assignment (mean posterior probability)",
           zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(state_labels(tab["label"]), rotation=45, ha="right",
                       fontsize=FS_TICK, color=INK2)
    ax.set_ylabel("fraction of genome bins", fontsize=FS_LABEL)
    _title(ax, "State coverage",
           "solid = hard assignment, translucent = mean posterior probability mass")
    # Neutral swatches: taking them from the bars would key the legend to whichever state
    # happens to be first, which for a near-white quiescent state is invisible.
    handles = [
        Patch(facecolor=INK2, label="hard assignment (Viterbi path)"),
        Patch(facecolor=INK2, alpha=0.45, label="soft assignment (mean posterior probability)"),
    ]
    leg = ax.legend(handles=handles, frameon=False, fontsize=FS_LEGEND, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    return fig


def plot_segmentation_track(
    seg: Segmentation,
    sample: Optional[str] = None,
    chrom: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    max_bins: int = 600,
    palette: str = "roadmap",
    figsize=(WIDTH_DOUBLE, 4.2),
):
    """Genome-browser view: signal, hard segmentation, and posterior composition.

    The three panels are the argument for ChromState in one picture. The middle panel is
    what ChromHMM gives you -- one colour per bin, no indication of confidence. The bottom
    panel is the same inference before it was rounded off: where the stack is a single
    solid colour the call is unambiguous, and where it splits the region is genuinely
    between states.
    """
    ds = seg.dataset
    sample = sample or ds.samples[0]
    mask = ds.sample_mask(sample)
    bins = ds.bins.loc[mask].reset_index(drop=True)
    post = seg.posterior[mask]
    states = seg.states[mask]
    X = ds.X[mask] / np.maximum(ds.size_factors[mask], 1e-9)

    sel = np.ones(len(bins), dtype=bool)
    if chrom is not None:
        sel &= (bins["chrom"] == chrom).to_numpy()
    if start is not None:
        sel &= (bins["start"] >= start).to_numpy()
    if end is not None:
        sel &= (bins["end"] <= end).to_numpy()
    idx = np.flatnonzero(sel)[:max_bins]
    if idx.size == 0:
        raise ValueError("No bins in the requested window")

    bins, post, states, X = bins.iloc[idx], post[idx], states[idx], X[idx]
    pos = bins["start"].to_numpy() / 1e3
    colors = state_colors(seg, palette)

    fig, axes = plt.subplots(
        3, 1, figsize=figsize, facecolor=SURFACE, sharex=True,
        gridspec_kw={"height_ratios": [2.6, 0.42, 1.9], "hspace": 0.18},
    )

    ax = axes[0]
    _style_axes(ax)
    for j, mark in enumerate(ds.marks):
        ax.plot(pos, X[:, j], linewidth=0.75, color=CVD_PALETTE[j % len(CVD_PALETTE)], label=mark)
    ax.set_ylabel("normalised read count", fontsize=FS_LABEL)
    leg = ax.legend(frameon=False, fontsize=FS_LEGEND, ncol=len(ds.marks), loc="lower left",
                    bbox_to_anchor=(0, 1.01), handlelength=1.4, columnspacing=1.2)
    for t in leg.get_texts():
        t.set_color(INK2)
    # Title goes on the figure, above the mark legend, so the two cannot collide.
    fig.suptitle(f"{sample} — {bins['chrom'].iloc[0]}", color=INK, fontsize=FS_TITLE,
                 x=0.08, ha="left", y=0.985)

    ax = axes[1]
    ax.set_facecolor(SURFACE)
    width = pos[1] - pos[0] if len(pos) > 1 else 0.2
    ax.bar(pos, np.ones(len(pos)), width=width, color=[colors[s] for s in states], linewidth=0)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("hard state\ncall", fontsize=FS_TICK, color=INK2, rotation=0,
                  ha="right", va="center")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    ax = axes[2]
    _style_axes(ax, grid_axis=None)
    full = state_labels(seg.labels)
    ax.stackplot(pos, post.T, colors=colors, labels=full, linewidth=0.4, edgecolor=SURFACE)
    ax.set_ylim(0, 1)
    ax.set_ylabel("posterior probability", fontsize=FS_LABEL)
    ax.set_xlabel(f"position on {bins['chrom'].iloc[0]} (kilobases)", fontsize=FS_LABEL)
    handles = [Patch(facecolor=c, label=lbl) for c, lbl in zip(colors, full)]
    # Spelled-out state names are wide, so the legend runs at most three to a row.
    leg = ax.legend(handles=handles, frameon=False, fontsize=FS_LEGEND, ncol=min(seg.n_states, 3),
                    loc="upper left", bbox_to_anchor=(0, -0.28), handlelength=1.2)
    for t in leg.get_texts():
        t.set_color(INK2)

    # tight_layout cannot handle the borderless state ribbon or the out-of-axes legends;
    # the margins are set explicitly instead.
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.26)
    return fig


# ---------------------------------------------------------------------------
# ChromState-specific figures
# ---------------------------------------------------------------------------


def plot_confidence(table: pd.DataFrame, figsize=(WIDTH_DOUBLE, 2.0)):
    """How much of the genome is confidently called, and how much is being rounded off."""
    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor=SURFACE)
    for ax in axes:
        _style_axes(ax)

    axes[0].hist(table["confidence"], bins=40, color="#2a78d6", edgecolor=SURFACE, linewidth=0.5)
    axes[0].set_xlabel("posterior probability of the winning state", fontsize=FS_LABEL)
    axes[0].set_ylabel("number of genome bins", fontsize=FS_LABEL)
    _title(axes[0], "Call confidence")

    axes[1].hist(table["entropy"], bins=40, color="#eb6834", edgecolor=SURFACE, linewidth=0.5)
    axes[1].set_xlabel("normalised entropy across states", fontsize=FS_LABEL)
    _title(axes[1], "State ambiguity")

    amb = float((table["confidence"] < 0.8).mean())
    axes[2].hist(table["margin"], bins=40, color="#1baf7a", edgecolor=SURFACE, linewidth=0.5)
    axes[2].set_xlabel("winning minus runner-up posterior probability", fontsize=FS_LABEL)
    _title(axes[2], "Decision margin",
           f"{amb:.1%} of bins fall below 0.8 posterior probability")

    fig.tight_layout()
    return fig


def plot_bivalency(table: pd.DataFrame, sample: Optional[str] = None, figsize=(WIDTH_DOUBLE, 2.4)):
    """Bivalency decomposed: signal plane, magnitude-balance plane, and the mixture flag."""
    sub = table if sample is None else table.loc[table["sample"] == sample]
    a_col = next(c for c in sub.columns if c.startswith("signal_H3K4me3"))
    r_col = next(c for c in sub.columns if c.startswith("signal_H3K27me3"))

    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor=SURFACE)
    for ax in axes:
        _style_axes(ax, grid_axis="both")

    sc = axes[0].scatter(sub[a_col], sub[r_col], c=sub["bivalency_score"], cmap=SEQ_CMAP,
                         s=1.5, linewidths=0, alpha=0.75)
    axes[0].set_xlabel("H3K4me3 signal (log-transformed, normalised)", fontsize=FS_LABEL)
    axes[0].set_ylabel("H3K27me3 signal (log-transformed, normalised)", fontsize=FS_LABEL)
    _title(axes[0], "Signal plane", "colour = bivalency score")
    cb = fig.colorbar(sc, ax=axes[0], fraction=0.045, pad=0.02)
    cb.ax.tick_params(colors=INK, labelsize=FS_TICK, length=1.5, width=0.4)
    cb.outline.set_visible(False)

    axes[1].scatter(sub["bivalency_magnitude"], sub["bivalency_balance"],
                    c="#4a3aa7", s=1.5, linewidths=0, alpha=0.5)
    axes[1].set_xlabel("magnitude: geometric mean of active and repressive signal", fontsize=FS_LABEL)
    axes[1].set_ylabel("balance: active against repressive signal", fontsize=FS_LABEL)
    _title(axes[1], "Decomposition", "top right = strong and balanced")

    axes[2].scatter(sub["biv_within_state"], sub["biv_between_state"],
                    c="#eb6834", s=1.5, linewidths=0, alpha=0.5)
    axes[2].set_xlabel("bivalency within a single state", fontsize=FS_LABEL)
    axes[2].set_ylabel("bivalency between states (mixture-confounded)", fontsize=FS_LABEL)
    _title(axes[2], "Mixture caveat",
           "high on the vertical axis = possibly two cell populations, not one bivalent region")

    fig.tight_layout()
    return fig


def plot_attribution(importance: pd.DataFrame, figsize=(WIDTH_SINGLE, 3.0)):
    """Which marks provide the evidence for each state call (log-likelihood ratio, nats)."""
    V = importance.values
    lim = float(np.nanmax(np.abs(V))) or 1.0

    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    im = ax.imshow(V, cmap=DIV_CMAP, aspect="auto", norm=TwoSlopeNorm(0.0, -lim, lim))
    ax.set_xticks(range(importance.shape[1]))
    ax.set_xticklabels(importance.columns, rotation=45, ha="right", fontsize=FS_TICK, color=INK2)
    ax.set_yticks(range(importance.shape[0]))
    ax.set_yticklabels(state_labels(importance.index), fontsize=FS_TICK, color=INK2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, importance.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, importance.shape[0], 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            ax.text(i * 0 + j, i, f"{V[i, j]:.1f}", ha="center", va="center", fontsize=FS_CELL,
                    color=INK)

    ax.set_xlabel("histone mark", color=INK2, fontsize=FS_LABEL)
    ax.set_ylabel("chromatin state", color=INK2, fontsize=FS_LABEL)
    # Warm = positive, cool = negative, neutral grey at zero: the standard diverging
    # convention. The caption must state the sign mapping explicitly, since "supports"
    # and "argues against" are not inferable from the colours alone.
    _title(ax, "Mark attribution",
           "evidence for the called state over the runner-up state, in natural log units; "
           "red supports the call, blue argues against it")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(colors=INK, labelsize=FS_TICK, length=1.5, width=0.4)
    cb.outline.set_visible(False)
    fig.tight_layout()
    return fig


def plot_state_flow(flow: pd.DataFrame, sample_a: str, sample_b: str, figsize=(WIDTH_MID, 3.9)):
    """Where each state's regions went between two conditions."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    im = ax.imshow(flow.values, cmap=SEQ_CMAP, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(flow.shape[1]))
    ax.set_xticklabels(state_labels(flow.columns), rotation=45, ha="right",
                       fontsize=FS_TICK, color=INK2)
    ax.set_yticks(range(flow.shape[0]))
    ax.set_yticklabels(state_labels(flow.index), fontsize=FS_TICK, color=INK2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, flow.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, flow.shape[0], 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    for i in range(flow.shape[0]):
        for j in range(flow.shape[1]):
            v = flow.values[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=FS_CELL,
                        color=INK)

    ax.set_xlabel(f"state in sample {sample_b}", color=INK2, fontsize=FS_LABEL)
    ax.set_ylabel(f"state in sample {sample_a}", color=INK2, fontsize=FS_LABEL)
    _title(ax, "State flow",
           "probability of the state afterwards given the state before, "
           "from matched bins and full posterior probabilities")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(colors=INK, labelsize=FS_TICK, length=1.5, width=0.4)
    cb.outline.set_visible(False)
    fig.tight_layout()
    return fig


def plot_occupancy_change(occ: pd.DataFrame, figsize=(WIDTH_MID, 2.6)):
    """Change in state occupancy per sample relative to the reference."""
    delta_cols = [c for c in occ.columns if c.startswith("delta_")]
    D = occ[delta_cols]
    D.columns = [c[6:] for c in delta_cols]
    D = D.drop(index=D.index[0])

    fig, ax = _new_fig(*figsize)
    x = np.arange(D.shape[1])
    n = D.shape[0]
    w = 0.8 / max(n, 1)
    for i, (name, row) in enumerate(D.iterrows()):
        ax.bar(x + (i - (n - 1) / 2) * w, row.values, w * 0.92,
               color=CVD_PALETTE[i % len(CVD_PALETTE)], edgecolor=SURFACE, linewidth=0.5,
               label=str(name), zorder=3)
    ax.axhline(0, color=AXIS, linewidth=0.5, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(state_labels(D.columns), rotation=45, ha="right", fontsize=FS_TICK, color=INK2)
    ax.set_ylabel("change in fraction of genome bins", fontsize=FS_LABEL)
    _title(ax, "State occupancy change", f"relative to sample {occ.index[0]}")
    if n > 1:
        leg = ax.legend(frameon=False, fontsize=FS_LEGEND)
        for t in leg.get_texts():
            t.set_color(INK2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Benchmark figures
# ---------------------------------------------------------------------------


# Spelled-out panel titles for benchmark metric columns. Anything not listed falls back
# to its column name, which is the signal that a new metric still needs a label.
METRIC_LABELS: Dict[str, str] = {
    "ari": "Adjusted Rand index",
    "ami": "Adjusted mutual information",
    "v_measure": "V-measure",
    "matched_accuracy": "Accuracy against known truth",
    "biv_auprc": "Bivalency detection\n(area under precision-recall curve)",
    "biv_auroc": "Bivalency detection\n(area under ROC curve)",
    "signal_spearman": "Tracks signal magnitude\n(Spearman correlation)",
    "signal_pearson": "Tracks signal magnitude\n(Pearson correlation)",
    "overconfidence": "Overconfidence (lower is better)",
    "frac_saturated": "Saturated calls (lower is better)",
    "ece": "Expected calibration error\n(lower is better)",
    "mean_entropy": "Mean posterior entropy",
}

# Display names for the benchmark rows. The table keys stay short because they are also
# joined against and written to disk; only the figure spells them out.
METHOD_DISPLAY_NAMES: Dict[str, str] = {
    "ChromState (NB-HMM)": "ChromState\n(negative-binomial hidden Markov model)",
    "ChromHMM-like (Bernoulli)": "ChromHMM-like\n(binarised Bernoulli hidden Markov model)",
    "Segway-like (Gaussian)": "Segway-like\n(Gaussian hidden Markov model)",
    "GMM (no spatial model)": "Gaussian mixture model\n(no spatial model)",
    "k-means": "k-means clustering\n(hard assignment)",
    "Naive signal (no model)": "Naive signal combination\n(no model)",
}


def plot_method_comparison(
    table: pd.DataFrame,
    metrics: Sequence[str] = ("ari", "matched_accuracy", "biv_auprc", "signal_spearman"),
    labels: Optional[Dict[str, str]] = None,
    figsize=(WIDTH_DOUBLE, 2.6),
):
    """Small multiples, one panel per metric -- never a dual axis.

    Each metric gets its own panel because they are on different scales and mean different
    things; forcing them onto shared axes would invite comparisons that are not valid.
    """
    labels = {**METRIC_LABELS, **(labels or {})}
    metrics = [m for m in metrics if m in table.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, facecolor=SURFACE)
    axes = np.atleast_1d(axes)

    # Titles are left-aligned inside narrow panels, so a long spelled-out metric name runs
    # into the next panel's title. Wrap to the panel's width, then pad every title to the
    # same number of lines so all of them start at the same height.
    panel_chars = max(16, int(figsize[0] / max(len(metrics), 1) / FS_TITLE * 108))
    wrapped = [_wrap_lines(labels.get(m, m), panel_chars) for m in metrics]
    n_lines = max(len(w) for w in wrapped) if wrapped else 1
    titles = ["\n".join(w + [""] * (n_lines - len(w))) for w in wrapped]

    names = table["method"].tolist()
    display_names = [METHOD_DISPLAY_NAMES.get(n, n) for n in names]
    is_chromstate = [n.startswith("ChromState") for n in names]
    bar_colors = ["#2a78d6" if c else "#b8b6ae" for c in is_chromstate]

    for ax, metric, title in zip(axes, metrics, titles):
        _style_axes(ax, grid_axis="x")
        vals = table[metric].to_numpy(dtype=float)
        y = np.arange(len(names))
        ax.barh(y, np.nan_to_num(vals), color=bar_colors, edgecolor=SURFACE, linewidth=0.5, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(display_names if metric == metrics[0] else [""] * len(names),
                           fontsize=FS_TICK, color=INK2)
        ax.invert_yaxis()
        for yi, v in zip(y, vals):
            if np.isfinite(v):
                ax.text(v + 0.015, yi, f"{v:.2f}", va="center", fontsize=FS_ANNOT, color=INK2)
            else:
                ax.text(0.02, yi, "not applicable", va="center", fontsize=FS_ANNOT, color=MUTED,
                        style="italic")
        ax.set_xlim(0, max(1.0, np.nanmax(vals) * 1.18) if np.isfinite(vals).any() else 1.0)
        _title(ax, title)

    fig.tight_layout()
    return fig


def plot_calibration(
    posteriors: Dict[str, np.ndarray],
    correct: Dict[str, np.ndarray],
    n_bins: int = 10,
    figsize=(WIDTH_SINGLE, 2.8),
):
    """Reliability diagram: is a reported probability of 0.9 right 90% of the time?

    Points below the diagonal are overconfident -- the model claims more certainty than it
    earns, which is what makes it hard to know which regions of a segmentation to trust.
    """
    fig, ax = _new_fig(*figsize)
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=0.6, linestyle=(0, (3, 2)), zorder=2)
    ax.text(0.62, 0.66, "perfectly calibrated model", fontsize=FS_ANNOT, color=MUTED, rotation=38)

    edges = np.linspace(0, 1, n_bins + 1)
    for i, (name, post) in enumerate(posteriors.items()):
        conf = post.max(axis=1)
        corr = correct[name].astype(float)
        xs, ys = [], []
        for b in range(n_bins):
            sel = (conf > edges[b]) & (conf <= edges[b + 1])
            if sel.sum() < 20:
                continue
            xs.append(conf[sel].mean())
            ys.append(corr[sel].mean())
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=0.5,
                color=CVD_PALETTE[i % len(CVD_PALETTE)],
                label=METHOD_DISPLAY_NAMES.get(name, name).replace("\n", " "),
                markeredgecolor=SURFACE, markeredgewidth=0.5, zorder=3)

    ax.set_xlabel("reported confidence (posterior probability)", fontsize=FS_LABEL)
    ax.set_ylabel("observed accuracy", fontsize=FS_LABEL)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    _title(ax, "Calibration",
           "below the diagonal = the model is overconfident")
    leg = ax.legend(frameon=False, fontsize=FS_LEGEND, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    return fig


NORMALIZATION_DISPLAY_NAMES: Dict[str, str] = {
    "depth": "sequencing depth",
    "median_ratio": "median of ratios",
    "spikein": "spike-in scaling",
    "none": "no normalisation",
    "quantile": "quantile normalisation",
}


def plot_normalization_experiment(table: pd.DataFrame, figsize=(WIDTH_SINGLE, 2.6)):
    """Which normalisation preserves a genome-wide mark loss, and which erases it."""
    fig, ax = _new_fig(*figsize)
    x = np.arange(len(table))
    vals = table["biology_absorbed"].to_numpy()
    ok = table["preserves_global_loss"].to_numpy()
    colors = ["#0ca30c" if o else "#d03b3b" for o in ok]

    ax.bar(x, vals, 0.58, color=colors, edgecolor=SURFACE, linewidth=0.5, zorder=3)
    ax.axhline(1.0, color=INK2, linewidth=0.6, linestyle=(0, (3, 2)), zorder=4)
    retention = float(table["true_retention"].iloc[0])
    ax.axhline(retention, color=MUTED, linewidth=0.5, linestyle=(0, (2, 3)), zorder=4)
    # Spelled out, the two reference-line labels are too long to sit on their own lines
    # without running through a bar, so they are stacked in the headroom above the bars
    # and name their line style instead of relying on position.
    ax.text(-0.35, 1.26, "dashed line: correct, the biological effect is untouched",
            fontsize=FS_ANNOT, color=INK2, ha="left", va="center")
    ax.text(-0.35, 1.14,
            f"dotted line: the biological effect is fully absorbed ({retention:g})",
            fontsize=FS_ANNOT, color=MUTED, ha="left", va="center")

    for xi, v, o in zip(x, vals, ok):
        ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=FS_TICK, color=INK2)
        ax.text(xi, 0.04, "preserved" if o else "erased", ha="center", fontsize=FS_ANNOT,
                color=INK, weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([NORMALIZATION_DISPLAY_NAMES.get(m, m) for m in table["method"]],
                       fontsize=FS_TICK, color=INK2)
    ax.set_xlabel("normalisation method", fontsize=FS_LABEL)
    ax.set_ylabel("fraction of the biological effect surviving", fontsize=FS_LABEL)
    ax.set_ylim(0, 1.36)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    _title(ax, "Normalisation under genome-wide H3K27me3 loss",
           "the size factor should absorb sequencing depth only, never the drug effect")
    fig.tight_layout()
    return fig


def plot_binarization_loss(table: pd.DataFrame, figsize=(WIDTH_SINGLE, 2.6)):
    """What ChromHMM's binarisation step collapses: the count range mapped onto '1'."""
    fig, ax = _new_fig(*figsize)
    x = np.arange(len(table))
    fold = table["fold_range_collapsed"].to_numpy(dtype=float)
    ax.bar(x, fold, 0.58, color="#eb6834", edgecolor=SURFACE, linewidth=0.5, zorder=3)
    # Both annotations go above the bar: spelled out, "% of bins called present" is wider
    # than a bar and would spill over its edges if set inside one.
    top = float(fold.max())
    for xi, v, f in zip(x, fold, table["frac_present"]):
        ax.annotate(f"{f:.0%} of bins\ncalled present", xy=(xi, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=FS_CELL, color=MUTED, linespacing=1.3)
        ax.annotate(f"{v:.1f}-fold", xy=(xi, v), xytext=(0, 15),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=FS_TICK, color=INK2)
    ax.set_ylim(0, top * 1.34)
    ax.set_xticks(x)
    ax.set_xticklabels(table["mark"], rotation=45, ha="right", fontsize=FS_TICK, color=INK2)
    ax.set_xlabel("histone mark", fontsize=FS_LABEL)
    ax.set_ylabel("fold range collapsed onto a single \"present\" call", fontsize=FS_LABEL)
    _title(ax, "Information discarded by binarisation",
           "among bins called present, the ratio of the largest to the smallest read count")
    fig.tight_layout()
    return fig


def plot_model_selection(table: pd.DataFrame, figsize=(WIDTH_MID, 2.2)):
    """Held-out likelihood per bin against K, with BIC shown as a separate panel."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor=SURFACE)
    for ax in axes:
        _style_axes(ax)

    axes[0].plot(table["n_states"], table["heldout_loglik_per_bin"], marker="o", markersize=3,
                 linewidth=0.5, color="#2a78d6", markeredgecolor=SURFACE, markeredgewidth=0.5)
    best = table.loc[table["heldout_loglik_per_bin"].idxmax(), "n_states"]
    axes[0].axvline(best, color=MUTED, linewidth=0.5, linestyle=(0, (3, 3)))
    axes[0].set_xlabel("number of chromatin states", fontsize=FS_LABEL)
    axes[0].set_ylabel("held-out log-likelihood per bin", fontsize=FS_LABEL)
    _title(axes[0], "Held-out likelihood",
           f"best number of states = {best} (recommended criterion)")

    axes[1].plot(table["n_states"], table["bic"], marker="o", markersize=3, linewidth=0.5,
                 color="#eb6834", markeredgecolor=SURFACE, markeredgewidth=0.5)
    axes[1].set_xlabel("number of chromatin states", fontsize=FS_LABEL)
    axes[1].set_ylabel("Bayesian information criterion\n(lower is better)", fontsize=FS_LABEL)
    _title(axes[1], "Bayesian information criterion",
           "over-penalises: adjacent bins are not independent")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Change confidence and expression linkage
# ---------------------------------------------------------------------------


def plot_change_volcano(
    table: pd.DataFrame,
    noise_floor: Optional[float] = None,
    fdr: float = 0.05,
    label_top: int = 8,
    figsize=(WIDTH_SINGLE, 2.8),
):
    """Effect size against significance for a differential test, with the noise floor.

    The two thresholds do different jobs and both belong on the picture: the vertical
    line is what replicates of the *same* condition already produce, and the horizontal
    line is the FDR. A point can clear one without the other, and only the top-right
    quadrant is a confident change.
    """
    fig, ax = _new_fig(*figsize)
    effect = table["tv_distance"].to_numpy(dtype=float)

    if "qvalue" in table.columns:
        q = table["qvalue"].to_numpy(dtype=float)
        y = -np.log10(np.clip(q, 1e-12, 1.0))
        ylabel = "statistical significance, -log10(q)"
    else:
        y = np.zeros_like(effect)
        ylabel = "no p-values available (too few replicates)"

    above = effect > noise_floor if noise_floor is not None else np.ones_like(effect, bool)
    sig = (table["qvalue"].to_numpy() < fdr) if "qvalue" in table.columns else np.zeros_like(above)
    confident = above & sig

    ax.scatter(effect[~confident], y[~confident], s=4, linewidths=0, alpha=0.55,
               color="#b8b6ae", zorder=3, label="not confident")
    if confident.any():
        ax.scatter(effect[confident], y[confident], s=6, linewidths=0, alpha=0.9,
                   color="#2a78d6", zorder=4, label="clears both thresholds")

    if noise_floor is not None:
        ax.axvline(noise_floor, color="#e34948", linewidth=0.6, linestyle=(0, (3, 2)), zorder=5)
        ax.text(noise_floor, ax.get_ylim()[1], "  replicate noise floor", color=INK,
                fontsize=FS_ANNOT, va="top", ha="left")
    if "qvalue" in table.columns:
        ax.axhline(-np.log10(fdr), color=MUTED, linewidth=0.6, linestyle=(0, (2, 3)), zorder=5)
        ax.text(ax.get_xlim()[1], -np.log10(fdr), f"false discovery rate {fdr:g}  ",
                color=MUTED, fontsize=FS_ANNOT, va="bottom", ha="right")

    if label_top and "name" in table.columns and confident.any():
        order = np.argsort(-(effect * (y + 1)) * confident)[:label_top]
        for i in order:
            if confident[i]:
                ax.annotate(str(table["name"].iloc[i]), (effect[i], y[i]),
                            xytext=(2, 2), textcoords="offset points",
                            fontsize=FS_ANNOT, color=INK2)

    ax.set_xlabel("effect size: fraction of state identity that moved", fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    n_conf = int(confident.sum())
    _title(ax, "Confident state changes",
           f"{n_conf} of {len(table)} regions clear both the replicate noise floor "
           f"and the false-discovery threshold")
    leg = ax.legend(frameon=False, fontsize=FS_LEGEND, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    return fig


def plot_noise_floor(noise_floor, contrast_effect: Optional[np.ndarray] = None,
                     figsize=(WIDTH_SINGLE, 2.4)):
    """What an unchanged bin looks like, against what the contrast produced.

    Overlapping histograms, because the question is not "is the contrast distribution
    shifted" but "how much of it lies outside the range replicates already cover".
    """
    fig, ax = _new_fig(*figsize)
    bins = np.linspace(0, 1, 60)
    ax.hist(noise_floor.distribution, bins=bins, color="#b8b6ae", edgecolor=SURFACE,
            linewidth=0.4, label="between replicates of one condition", zorder=3)
    if contrast_effect is not None:
        ax.hist(np.asarray(contrast_effect), bins=bins, color="#2a78d6", alpha=0.68,
                edgecolor=SURFACE, linewidth=0.4, label="between conditions", zorder=4)
    ax.axvline(noise_floor.threshold, color="#e34948", linewidth=0.75,
               linestyle=(0, (3, 2)), zorder=5)
    ax.text(noise_floor.threshold, ax.get_ylim()[1] * 0.96,
            f"  {noise_floor.quantile:.0%} of replicate noise below {noise_floor.threshold:.2f}",
            color=INK, fontsize=FS_ANNOT, va="top")
    ax.set_xlabel("fraction of state identity that moved", fontsize=FS_LABEL)
    ax.set_ylabel("number of bins", fontsize=FS_LABEL)
    _title(ax, "Replicate noise floor",
           "anything the two distributions share is not evidence of a condition effect")
    leg = ax.legend(frameon=False, fontsize=FS_LEGEND, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    return fig


def plot_expression_link(
    linked: pd.DataFrame,
    association: Optional[pd.DataFrame] = None,
    state: Optional[str] = None,
    figsize=(WIDTH_DOUBLE, 2.8),
):
    """Chromatin change against expression change: the scatter and the per-state summary.

    Left: one gene per point, change in the chosen state's promoter occupancy against
    change in expression. Right: the Spearman correlation for every state, so the picture
    cannot be read off one flattering state alone.
    """
    delta_cols = [c for c in linked.columns if c.startswith("delta_")]
    if not delta_cols:
        raise ValueError("linked table has no delta_<state> columns")
    if state is None:
        if association is not None and len(association):
            state = "delta_" + str(association.iloc[0]["state"])
        else:
            state = delta_cols[0]
    if state not in linked.columns:
        raise ValueError(f"{state!r} not in the linked table")

    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.05, 1]})
    for ax in axes:
        _style_axes(ax, grid_axis="both")

    x = linked[state].to_numpy(dtype=float)
    y = linked["log2fc"].to_numpy(dtype=float)
    axes[0].axhline(0, color=AXIS, linewidth=0.5, zorder=2)
    axes[0].axvline(0, color=AXIS, linewidth=0.5, zorder=2)
    axes[0].scatter(x, y, s=4, linewidths=0, alpha=0.6, color="#2a78d6", zorder=3)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() > 2:
        coef = np.polyfit(x[ok], y[ok], 1)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
        axes[0].plot(xs, np.polyval(coef, xs), color=INK2, linewidth=0.75,
                     linestyle=(0, (3, 2)), zorder=4)
    label = full_state_name(state.replace("delta_", ""))
    axes[0].set_xlabel(f"change in {label} occupancy at the promoter", fontsize=FS_LABEL)
    axes[0].set_ylabel("change in expression, log2 fold change", fontsize=FS_LABEL)
    _title(axes[0], "Chromatin change against expression change",
           f"{int(ok.sum())} genes; association only, direction of causality not identified")

    if association is not None and len(association):
        assoc = association.dropna(subset=["spearman"]).iloc[::-1]
        names = [full_state_name(s) for s in assoc["state"]]
        vals = assoc["spearman"].to_numpy(dtype=float)
        colors = ["#2a78d6" if v >= 0 else "#e34948" for v in vals]
        pos = np.arange(len(vals))
        axes[1].barh(pos, vals, color=colors, edgecolor=SURFACE, linewidth=0.5, zorder=3)
        axes[1].axvline(0, color=AXIS, linewidth=0.5, zorder=4)
        # Headroom on both sides: a negative bar's value label would otherwise run into
        # the spelled-out state name on the left-hand axis.
        span = max(abs(np.nanmin(vals)), abs(np.nanmax(vals)), 0.05)
        axes[1].set_xlim(-span * 1.4, span * 1.4)
        axes[1].set_yticks(pos)
        axes[1].set_yticklabels(names, fontsize=FS_TICK, color=INK2)
        if "significant" in assoc.columns:
            for p, v, sig in zip(pos, vals, assoc["significant"]):
                axes[1].text(v + (0.01 if v >= 0 else -0.01), p,
                             f"{v:.2f}" + ("*" if bool(sig) else ""),
                             va="center", ha="left" if v >= 0 else "right",
                             fontsize=FS_ANNOT, color=INK2)
        axes[1].set_xlabel("Spearman correlation with expression change", fontsize=FS_LABEL)
        _title(axes[1], "Every state, not just the best one",
               "* marks associations passing the false-discovery threshold")
    else:
        axes[1].axis("off")

    fig.tight_layout()
    return fig


def plot_gene_trace(trace: Dict[str, object], palette: str = "roadmap", figsize=(WIDTH_DOUBLE, 4.2)):
    """One gene, end to end: signal, state composition per sample, and mark attribution.

    The traceability figure. Top panel is the normalised signal for every mark across the
    feature; middle is each sample's state composition over the same bins; bottom is the
    per-mark evidence for the call, so the question "why is this gene called that" has a
    visible answer rather than a citation to a method.
    """
    bins = trace["bins"]
    attr = trace["attribution"]
    comp = trace["composition"]
    marks = [c[len("signal_"):] for c in bins.columns if c.startswith("signal_")]
    samples = list(comp["sample"])

    fig, axes = plt.subplots(3, 1, figsize=figsize, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [1.5, 1.0, 1.2], "hspace": 0.55})

    ax = axes[0]
    _style_axes(ax)
    for si, sample in enumerate(samples):
        sub = bins.loc[bins["sample"] == sample]
        pos = sub["start"].to_numpy() / 1e3
        for j, mark in enumerate(marks):
            ax.plot(pos, sub[f"signal_{mark}"].to_numpy(),
                    linewidth=0.75 if si == 0 else 0.4,
                    alpha=1.0 if si == 0 else 0.4,
                    linestyle="-" if si == 0 else (0, (3, 2)),
                    color=CVD_PALETTE[j % len(CVD_PALETTE)],
                    label=mark if si == 0 else None)
    ax.set_ylabel("normalised\nread count", fontsize=FS_LABEL)
    ax.set_xlabel(f"position on {bins['chrom'].iloc[0]} (kilobases)", fontsize=FS_LABEL)
    leg = ax.legend(frameon=False, fontsize=FS_LEGEND, ncol=min(len(marks), 6),
                    loc="lower left", bbox_to_anchor=(0, 1.02))
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.text(1.0, 1.02, "solid = first sample, dashed = the others", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=FS_SUBTITLE, color=MUTED)

    ax = axes[1]
    _style_axes(ax, grid_axis="x")
    left = np.zeros(len(samples))
    order = list(comp.columns[2:])
    for lbl in order:
        vals = comp[lbl].to_numpy(dtype=float)
        ax.barh(np.arange(len(samples)), vals, left=left, height=0.62,
                color=STATE_COLORS.get(str(lbl).split("_")[0], "#DDDDDD"),
                edgecolor=SURFACE, linewidth=0.5, zorder=3,
                label=full_state_name(str(lbl)))
        left += vals
    ax.set_yticks(np.arange(len(samples)))
    ax.set_yticklabels(samples, fontsize=FS_TICK, color=INK2)
    ax.set_xlim(0, 1)
    ax.set_xlabel("state composition over the feature", fontsize=FS_LABEL)
    # Outside to the right: the spelled-out state names are far too wide to sit under the
    # panel without colliding with the attribution axes below it.
    leg = ax.legend(frameon=False, fontsize=FS_LEGEND, loc="upper left",
                    bbox_to_anchor=(1.01, 1.06), handlelength=1.2)
    for t in leg.get_texts():
        t.set_color(INK2)

    ax = axes[2]
    _style_axes(ax, grid_axis="x")
    llr_cols = [c for c in attr.columns if c.startswith("llr_")]
    means = [float(attr[c].mean()) for c in llr_cols]
    names = [c[len("llr_"):] for c in llr_cols]
    colors = ["#e34948" if v >= 0 else "#2a78d6" for v in means]
    ax.barh(np.arange(len(names)), means, color=colors, edgecolor=SURFACE,
            linewidth=0.5, zorder=3)
    ax.axvline(0, color=AXIS, linewidth=0.5, zorder=4)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=FS_TICK, color=INK2)
    ax.set_xlabel("evidence for the called state over the runner-up, natural log units",
                  fontsize=FS_LABEL)

    fig.suptitle(f"{trace['name']} — {trace['part']}", color=INK, fontsize=FS_TITLE,
                 x=0.02, ha="left", y=0.995)
    # Margins are set explicitly: two of the three panels carry legends outside their
    # axes, which tight_layout cannot account for.
    fig.subplots_adjust(left=0.13, right=0.98, top=0.86, bottom=0.09)
    return fig


def enforce_text_style(fig: plt.Figure) -> None:
    """Make every piece of text in ``fig`` black Arial (Nature house style).

    The plotting functions already draw text this way; this is the backstop, so a label
    added later in a caller's own code cannot leave a grey or coloured word in a figure.
    """
    from matplotlib import font_manager
    from matplotlib.text import Text

    # Only request installed faces: asking for a missing family logs a warning per text
    # object, which floods the console on machines without the full fallback stack.
    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = [f for f in FONT_STACK if f in installed] or ["sans-serif"]
    for t in fig.findobj(Text):
        if t.get_text():
            t.set_color(INK)
            t.set_fontfamily(family)


def save_all(figs: Dict[str, plt.Figure], outdir: str) -> List[str]:
    """Write a dict of ``name -> figure`` to ``outdir`` as vector PDFs.

    PDF only, with fonts embedded as editable TrueType (``pdf.fonttype`` 42), which is
    what journal production expects; figures are drawn at their print size, so no
    rescaling is needed when they are placed.
    """
    import os

    os.makedirs(outdir, exist_ok=True)
    paths = []
    for name, fig in figs.items():
        enforce_text_style(fig)
        path = os.path.join(outdir, f"{name}.pdf")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02, facecolor=SURFACE)
        plt.close(fig)
        paths.append(path)
    return paths
