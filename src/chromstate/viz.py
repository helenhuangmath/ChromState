"""Figures: the ChromHMM-standard plots plus the ones ChromState adds.

Colour policy
-------------
Two categorical palettes ship, and the choice is a real tradeoff rather than a preference:

``scirep2025`` (default)
    The scheme of Huang, Baxter et al., Sci Rep 2025 (Supplementary Fig. S4a): active
    states in reds, poised in magenta, repressed in blues, repetitive/heterochromatin in
    violet, low signal in grey. Hex values sampled from the published figure.

``nature``
    The Okabe-Ito scheme (Wong, Nature Methods 2011), the colourblind-safe palette
    Nature's figure guidelines recommend: one hue per state family, lighter tints for
    weak variants, light grey for quiescent. Use for papers, posters and talks.

``roadmap``
    The Roadmap Epigenomics scheme every ChromHMM user already reads at a glance -- red
    promoters, yellow enhancers, green transcription, grey Polycomb. It is *not*
    colourblind-safe, and not marginally so: TssA (``#FF0000``) and TssFlnk (``#FF4500``)
    sit at ΔE 3.0 for normal vision and 3.0 under deuteranopia, far below the ΔE 15 floor
    at which two colours are reliably distinguishable at all. Several states also fall
    below 3:1 contrast on white. Field recognisability has
    real value, and every plot here carries direct labels so colour never carries meaning
    alone. Segment BED files always carry Roadmap colours so browsers show the familiar
    track.

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

from .model.annotate import (  # noqa: F401  (display_order is re-exported for figure scripts)
    OKABE_ITO,
    SCIREP2025_GROUP_COLORS,
    STATE_COLORS,
    STATE_COLORS_NATURE,
    STATE_COLORS_SCIREP2025,
    display_order,
    full_state_name,
)
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


#: Explanatory subtitles under panel titles. Off: a figure carries its title, axes and
#: legend, and the explanation belongs in the figure legend. Set True for self-explaining
#: exploratory output.
SHOW_SUBTITLES = False

#: Numbers printed on the figure itself: a value beside every bar, in every heat-map cell
#: and on every stacked segment. Off: the axis or the colour bar carries the values and the
#: exact numbers belong in the legend or a table. Set True for exploratory output.
ANNOTATE_VALUES = False


def _title(ax: plt.Axes, text: str, subtitle: Optional[str] = None) -> None:
    """Left-aligned title with an optional one-line explanation wrapped to the panel width.

    Wrapping matters at print size: an unwrapped explanation runs past the panel, and a
    tight bounding box would then silently widen the figure beyond the journal column.
    """
    lines: List[str] = []
    if subtitle and SHOW_SUBTITLES:
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


DEFAULT_PALETTE = "scirep2025"


def _shade(hex_color: str, amount: float) -> str:
    """Lighten a colour towards white by ``amount`` (0-1), for duplicate mnemonics."""
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    rgb = [int(round(v + (255 - v) * min(max(amount, 0.0), 0.8))) for v in rgb]
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def state_colors(seg: Segmentation, palette: str = DEFAULT_PALETTE) -> List[str]:
    """Resolve per-state colours under the requested palette.

    ``scirep2025`` (default): the scheme of Huang et al., Sci Rep 2025 (Supplementary
    Fig. S4a) -- active states in reds, poised in magenta, repressed in blues,
    heterochromatin in violet, low signal in grey (``STATE_COLORS_SCIREP2025``).
    ``nature``: the Okabe-Ito colourblind-safe scheme, one hue per state family
    (``STATE_COLORS_NATURE``). In both, a second state of the same family takes a
    lighter shade and unnamed families take spare Okabe-Ito hues. ``roadmap``: the
    Roadmap Epigenomics colours ChromHMM users know (not colourblind-safe). ``cvd``:
    eight validated hues in a fixed order, ignoring state identity.
    """
    if palette == "roadmap":
        return list(seg.colors)
    if palette in ("nature", "scirep2025"):
        table = STATE_COLORS_NATURE if palette == "nature" else STATE_COLORS_SCIREP2025
        out, used = [], set()
        for lbl in seg.labels:
            base = str(lbl).split("_")[0]
            c = table.get(base)
            if c is None:
                free = [h for h in OKABE_ITO if h not in used] or OKABE_ITO
                c = free[0]
            elif c in used and base != "Quies":
                c = _shade(c, 0.22 * (1 + sum(1 for u in out if u.startswith(c[:4]))))
            used.add(c)
            out.append(c)
        return out
    if palette == "cvd":
        if seg.n_states > len(CVD_PALETTE):
            # Never cycle hues: states past the cap share the "Other" grey and rely on
            # their direct labels.
            return [
                CVD_PALETTE[i] if i < len(CVD_PALETTE) else "#b8b6ae"
                for i in range(seg.n_states)
            ]
        return CVD_PALETTE[: seg.n_states]
    raise ValueError(f"Unknown palette {palette!r}; expected 'scirep2025', 'nature', 'roadmap' or 'cvd'")


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

    if ANNOTATE_VALUES:
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                ax.text(
                    j, i, f"{E.values[i, j]:.0f}",
                    ha="center", va="center", fontsize=FS_CELL,
                    color=INK,
                )

    _title(ax, "Mark levels",
           "cell colour = fraction of that mark's maximum across states; "
           "number = mean read count per 200 base-pair bin")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Fraction of mark maximum" if normalize else "Mean count", fontsize=FS_TICK)
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
            if ANNOTATE_VALUES and v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=FS_CELL,
                        color=INK)

    ax.set_xlabel("Next bin", color=INK2, fontsize=FS_LABEL)
    ax.set_ylabel("Current bin", color=INK2, fontsize=FS_LABEL)
    _title(ax, "Transition probabilities",
           "strong diagonal = states form extended domains rather than alternating bin to bin")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Probability", fontsize=FS_TICK)
    cb.ax.tick_params(colors=INK, labelsize=FS_TICK, length=1.5, width=0.4)
    cb.outline.set_visible(False)
    fig.tight_layout()
    return fig


def plot_state_coverage(seg: Segmentation, palette: str = DEFAULT_PALETTE, figsize=(WIDTH_MID, 2.6)):
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
           linewidth=0.5, label="Hard calls", zorder=3)
    ax.bar(x + w / 2 + 0.01, tab["coverage_soft"], w, color=colors, edgecolor=SURFACE,
           linewidth=0.5, alpha=0.45, label="Posterior mass",
           zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(state_labels(tab["label"]), rotation=45, ha="right",
                       fontsize=FS_TICK, color=INK2)
    ax.set_ylabel("Fraction of bins", fontsize=FS_LABEL)
    _title(ax, "State coverage",
           "solid = hard assignment, translucent = mean posterior probability mass")
    # Neutral swatches: taking them from the bars would key the legend to whichever state
    # happens to be first, which for a near-white quiescent state is invisible.
    handles = [
        Patch(facecolor=INK2, label="Hard calls"),
        Patch(facecolor=INK2, alpha=0.45, label="Posterior mass"),
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
    palette: str = DEFAULT_PALETTE,
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
    ax.set_ylabel("Normalised signal", fontsize=FS_LABEL)
    leg = ax.legend(frameon=False, fontsize=FS_LEGEND, ncol=len(ds.marks), loc="lower left",
                    bbox_to_anchor=(0, 1.01), handlelength=1.4, columnspacing=1.2)
    for t in leg.get_texts():
        t.set_color(INK2)
    # Title goes on the figure, above the mark legend, so the two cannot collide.
    fig.suptitle(str(sample), color=INK, fontsize=FS_TITLE,
                 x=0.08, ha="left", y=0.985)

    ax = axes[1]
    ax.set_facecolor(SURFACE)
    width = pos[1] - pos[0] if len(pos) > 1 else 0.2
    ax.bar(pos, np.ones(len(pos)), width=width, color=[colors[s] for s in states], linewidth=0)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("State", fontsize=FS_TICK, color=INK2, rotation=0,
                  ha="right", va="center")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    ax = axes[2]
    _style_axes(ax, grid_axis=None)
    full = state_labels(seg.labels)
    ax.stackplot(pos, post.T, colors=colors, labels=full, linewidth=0.4, edgecolor=SURFACE)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Posterior", fontsize=FS_LABEL)
    ax.set_xlabel(f"Position on {bins['chrom'].iloc[0]} (kb)", fontsize=FS_LABEL)
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
    axes[0].set_xlabel("Posterior of the called state", fontsize=FS_LABEL)
    axes[0].set_ylabel("Bins", fontsize=FS_LABEL)
    _title(axes[0], "Call confidence")

    axes[1].hist(table["entropy"], bins=40, color="#eb6834", edgecolor=SURFACE, linewidth=0.5)
    axes[1].set_xlabel("Normalised entropy", fontsize=FS_LABEL)
    _title(axes[1], "State ambiguity")

    amb = float((table["confidence"] < 0.8).mean())
    axes[2].hist(table["margin"], bins=40, color="#1baf7a", edgecolor=SURFACE, linewidth=0.5)
    axes[2].set_xlabel("Posterior margin over runner-up", fontsize=FS_LABEL)
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
    axes[0].set_xlabel("H3K4me3 signal (log)", fontsize=FS_LABEL)
    axes[0].set_ylabel("H3K27me3 signal (log)", fontsize=FS_LABEL)
    _title(axes[0], "Signal plane", "colour = bivalency score")
    cb = fig.colorbar(sc, ax=axes[0], fraction=0.045, pad=0.02)
    cb.set_label("Bivalency score", fontsize=FS_TICK)
    cb.ax.tick_params(colors=INK, labelsize=FS_TICK, length=1.5, width=0.4)
    cb.outline.set_visible(False)

    axes[1].scatter(sub["bivalency_magnitude"], sub["bivalency_balance"],
                    c="#4a3aa7", s=1.5, linewidths=0, alpha=0.5)
    axes[1].set_xlabel("Magnitude", fontsize=FS_LABEL)
    axes[1].set_ylabel("Balance", fontsize=FS_LABEL)
    _title(axes[1], "Decomposition", "top right = strong and balanced")

    axes[2].scatter(sub["biv_within_state"], sub["biv_between_state"],
                    c="#eb6834", s=1.5, linewidths=0, alpha=0.5)
    axes[2].set_xlabel("Within one state", fontsize=FS_LABEL)
    axes[2].set_ylabel("Between states", fontsize=FS_LABEL)
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

    if ANNOTATE_VALUES:
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                ax.text(i * 0 + j, i, f"{V[i, j]:.1f}", ha="center", va="center", fontsize=FS_CELL,
                        color=INK)

    # Warm = positive, cool = negative, neutral grey at zero: the standard diverging
    # convention. The caption must state the sign mapping explicitly, since "supports"
    # and "argues against" are not inferable from the colours alone.
    _title(ax, "Mark attribution",
           "evidence for the called state over the runner-up state, in natural log units; "
           "red supports the call, blue argues against it")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Evidence for the call (nats)", fontsize=FS_TICK)
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
            if ANNOTATE_VALUES and v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=FS_CELL,
                        color=INK)

    ax.set_xlabel(str(sample_b), color=INK2, fontsize=FS_LABEL)
    ax.set_ylabel(str(sample_a), color=INK2, fontsize=FS_LABEL)
    _title(ax, "State flow",
           "probability of the state afterwards given the state before, "
           "from matched bins and full posterior probabilities")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Probability", fontsize=FS_TICK)
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
    ax.set_ylabel("Change in fraction of bins", fontsize=FS_LABEL)
    _title(ax, f"Occupancy change vs {occ.index[0]}")
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
    "ari": "State recovery (ARI)",
    "ami": "State recovery (AMI)",
    "v_measure": "V-measure",
    "matched_accuracy": "Accuracy",
    "biv_auprc": "Bivalent bins (AUPRC)",
    "biv_posthoc_auprc": "Bivalent bins (AUPRC)",
    "biv_auroc": "Bivalent bins (AUROC)",
    "signal_spearman": "Signal retained (Spearman)",
    "signal_pearson": "Signal retained (Pearson)",
    "overconfidence": "Overconfidence",
    "frac_saturated": "Saturated calls",
    "ece": "Calibration error",
    "mean_entropy": "Posterior entropy",
}

# Display names for the benchmark rows. The table keys carry the model family because they
# are also joined against and written to disk; the figure shows the short name only and
# the legend says what each method is.
METHOD_DISPLAY_NAMES: Dict[str, str] = {
    "ChromState (NB-HMM)": "ChromState",
    "ChromHMM-like (Bernoulli)": "ChromHMM-like",
    "Segway-like (Gaussian)": "Segway-like",
    "GMM (no spatial model)": "Gaussian mixture",
    "k-means": "k-means",
    "Naive signal (no model)": "Naive signal",
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
            if not np.isfinite(v):
                ax.text(0.02, yi, "n.a.", va="center", fontsize=FS_ANNOT, color=MUTED)
            elif ANNOTATE_VALUES:
                ax.text(v + 0.015, yi, f"{v:.2f}", va="center", fontsize=FS_ANNOT, color=INK2)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, max(1.0, np.nanmax(vals) * 1.05) if np.isfinite(vals).any() else 1.0)
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

    ax.set_xlabel("Reported confidence", fontsize=FS_LABEL)
    ax.set_ylabel("Observed accuracy", fontsize=FS_LABEL)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    _title(ax, "Calibration",
           "below the diagonal = the model is overconfident")
    leg = ax.legend(frameon=False, fontsize=FS_LEGEND, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    return fig


NORMALIZATION_DISPLAY_NAMES: Dict[str, str] = {
    "depth": "Depth",
    "median_ratio": "Median ratio",
    "spikein": "Spike-in",
    "none": "None",
    "quantile": "Quantile",
}


def plot_normalization_experiment(table: pd.DataFrame, figsize=(WIDTH_SINGLE, 2.6)):
    """Which normalisation preserves a genome-wide mark loss, and which erases it.

    Green bars preserve the loss, red bars erase it. Dashed line: the biological effect is
    untouched (1.0). Dotted line: the effect is fully absorbed by the size factor.
    """
    fig, ax = _new_fig(*figsize)
    x = np.arange(len(table))
    vals = table["biology_absorbed"].to_numpy()
    ok = table["preserves_global_loss"].to_numpy()
    colors = ["#0ca30c" if o else "#d03b3b" for o in ok]

    ax.bar(x, vals, 0.58, color=colors, edgecolor=SURFACE, linewidth=0.5, zorder=3)
    ax.axhline(1.0, color=INK2, linewidth=0.6, linestyle=(0, (3, 2)), zorder=4)
    retention = float(table["true_retention"].iloc[0])
    ax.axhline(retention, color=MUTED, linewidth=0.5, linestyle=(0, (2, 3)), zorder=4)
    if ANNOTATE_VALUES:
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=FS_TICK, color=INK2)

    ax.set_xticks(x)
    ax.set_xticklabels([NORMALIZATION_DISPLAY_NAMES.get(m, m) for m in table["method"]],
                       fontsize=FS_TICK, color=INK2)
    ax.set_ylabel("Effect surviving (fraction)", fontsize=FS_LABEL)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    _title(ax, "Normalisation",
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
    if ANNOTATE_VALUES:
        for xi, v in zip(x, fold):
            ax.annotate(f"{v:.1f}", xy=(xi, v), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=FS_TICK, color=INK2)
    ax.set_ylim(0, top * 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(table["mark"], rotation=45, ha="right", fontsize=FS_TICK, color=INK2)
    ax.set_ylabel("Fold range called \"present\"", fontsize=FS_LABEL)
    _title(ax, "Binarisation loss",
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
    axes[0].set_xlabel("States (K)", fontsize=FS_LABEL)
    axes[0].set_ylabel("Log-likelihood per bin", fontsize=FS_LABEL)
    _title(axes[0], "Held-out likelihood",
           f"best number of states = {best} (recommended criterion)")

    axes[1].plot(table["n_states"], table["bic"], marker="o", markersize=3, linewidth=0.5,
                 color="#eb6834", markeredgecolor=SURFACE, markeredgewidth=0.5)
    axes[1].set_xlabel("States (K)", fontsize=FS_LABEL)
    axes[1].set_ylabel("BIC", fontsize=FS_LABEL)
    _title(axes[1], "BIC",
           "over-penalises: adjacent bins are not independent")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Change confidence and expression linkage
# ---------------------------------------------------------------------------


def _stagger_labels(ax: plt.Axes, points, fontsize: float = FS_ANNOT, step: float = 7.0) -> None:
    """Annotate ``(x, y, text)`` points without overlap or clipping.

    Labels go to the right of points in the left half of the axis and to the left of
    points in the right half; among labels that share (nearly) the same height, each is
    dropped one step below the previous, with a thin leader back to the point.
    """
    if not points:
        return
    x0, x1 = ax.get_xlim()
    mid = 0.5 * (x0 + x1)
    pts = sorted(points, key=lambda t: (-t[1], t[0]))
    placed = []  # (y, k) of labels already placed
    for x, y, text in pts:
        k = sum(1 for (py, _) in placed if abs(py - y) < 0.08 * (ax.get_ylim()[1] - ax.get_ylim()[0]))
        placed.append((y, k))
        right = x <= mid
        ax.annotate(text, (x, y), xytext=(4 if right else -4, -1 - step * k), textcoords="offset points",
                    fontsize=fontsize, color=INK, ha="left" if right else "right", va="top",
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.4, shrinkA=0, shrinkB=1))


def plot_change_volcano(
    table: pd.DataFrame,
    noise_floor: Optional[float] = None,
    fdr: float = 0.05,
    label_top: int = 3,
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
        y = -np.log10(np.clip(q, 1e-300, 1.0))
        ylabel = "−log10 q"
    else:
        y = np.zeros_like(effect)
        ylabel = "No p-values (too few replicates)"

    above = effect > noise_floor if noise_floor is not None else np.ones_like(effect, bool)
    sig = (table["qvalue"].to_numpy() < fdr) if "qvalue" in table.columns else np.zeros_like(above)
    confident = above & sig

    ax.scatter(effect[~confident], y[~confident], s=4, linewidths=0, alpha=0.55,
               color="#b8b6ae", zorder=3)
    if confident.any():
        ax.scatter(effect[confident], y[confident], s=6, linewidths=0, alpha=0.9,
                   color="#2a78d6", zorder=4, label="Confident")

    if noise_floor is not None:
        ax.axvline(noise_floor, color="#e34948", linewidth=0.6, linestyle=(0, (3, 2)), zorder=5)
        ax.text(noise_floor, ax.get_ylim()[1], "  Noise floor", color=INK,
                fontsize=FS_ANNOT, va="top", ha="left")
    if "qvalue" in table.columns:
        ax.axhline(-np.log10(fdr), color=MUTED, linewidth=0.6, linestyle=(0, (2, 3)), zorder=5)
        ax.text(ax.get_xlim()[1], -np.log10(fdr), f"FDR {fdr:g}  ",
                color=MUTED, fontsize=FS_ANNOT, va="bottom", ha="right")

    if label_top and "name" in table.columns and confident.any():
        order = [i for i in np.argsort(-(effect * (y + 1)) * confident)[:label_top] if confident[i]]
        _stagger_labels(ax, [(effect[i], y[i], str(table["name"].iloc[i])) for i in order])

    ax.set_xlabel("Change in state composition", fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    n_conf = int(confident.sum())
    _title(ax, "Change per region",
           f"{n_conf} of {len(table)} regions clear both the replicate noise floor "
           f"and the false-discovery threshold")
    if confident.any():
        ax.legend(frameon=False, fontsize=FS_LEGEND, loc="lower right")
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
            linewidth=0.4, label="Replicates", zorder=3)
    if contrast_effect is not None:
        ax.hist(np.asarray(contrast_effect), bins=bins, color="#2a78d6", alpha=0.68,
                edgecolor=SURFACE, linewidth=0.4, label="Conditions", zorder=4)
    ax.axvline(noise_floor.threshold, color="#e34948", linewidth=0.75,
               linestyle=(0, (3, 2)), zorder=5)
    ax.set_xlabel("Change in state composition", fontsize=FS_LABEL)
    ax.set_ylabel("Bins", fontsize=FS_LABEL)
    _title(ax, "Noise floor",
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
    axes[0].set_xlabel("Change in promoter occupancy", fontsize=FS_LABEL)
    axes[0].set_ylabel("Expression change (log2)", fontsize=FS_LABEL)
    _title(axes[0], label,
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
                if not ANNOTATE_VALUES and not bool(sig):
                    continue
                axes[1].text(v + (0.01 if v >= 0 else -0.01), p,
                             (f"{v:.2f}" if ANNOTATE_VALUES else "") + ("*" if bool(sig) else ""),
                             va="center", ha="left" if v >= 0 else "right",
                             fontsize=FS_ANNOT, color=INK2)
        axes[1].set_xlabel("Spearman with expression change", fontsize=FS_LABEL)
        _title(axes[1], "All states",
               "* marks associations passing the false-discovery threshold")
    else:
        axes[1].axis("off")

    fig.tight_layout()
    return fig


def plot_gene_trace(trace: Dict[str, object], palette: str = DEFAULT_PALETTE, figsize=(WIDTH_DOUBLE, 4.2)):
    """One gene, end to end: signal, state composition per sample, and mark attribution.

    In the signal panel the first sample is drawn solid and the others dashed.

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
    ax.set_ylabel("Normalised\nsignal", fontsize=FS_LABEL)
    ax.set_xlabel(f"Position on {bins['chrom'].iloc[0]} (kb)", fontsize=FS_LABEL)
    leg = ax.legend(frameon=False, fontsize=FS_LEGEND, ncol=min(len(marks), 6),
                    loc="lower left", bbox_to_anchor=(0, 1.02))
    for t in leg.get_texts():
        t.set_color(INK2)

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
    ax.set_xlabel("State composition", fontsize=FS_LABEL)
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
    ax.set_xlabel("Evidence for the call (nats)", fontsize=FS_LABEL)

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


# ---------------------------------------------------------------------------
# Quantification summaries and condition comparison
# ---------------------------------------------------------------------------


def _label_colors(labels: Sequence[str], seg: Optional[Segmentation] = None,
                  palette: str = DEFAULT_PALETTE, colors: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Colour per label: explicit dict > segmentation colours > Roadmap mnemonic > CVD cycle."""
    out: Dict[str, str] = dict(colors or {})
    if seg is not None:
        seg_colors = state_colors(seg, palette)
        out.update({lbl: c for lbl, c in zip(seg.labels, seg_colors) if lbl not in out})
    class_colors = (dict(SCIREP2025_GROUP_COLORS) if palette == "scirep2025" else
                    {"active": "#D55E00", "poised": "#CC79A7", "repressed": "#0072B2",
                     "background": "#E8E8E8"})
    i = 0
    for lbl in labels:
        if lbl in out:
            continue
        base = str(lbl).split("_")[0]
        table = {"roadmap": STATE_COLORS, "nature": STATE_COLORS_NATURE}.get(palette, STATE_COLORS_SCIREP2025)
        if base in table:
            out[lbl] = table[base]
        elif str(lbl) in class_colors:
            out[lbl] = class_colors[str(lbl)]
        else:
            out[lbl] = CVD_PALETTE[i % len(CVD_PALETTE)]
            i += 1
    return out


def plot_state_percentages(
    summary: pd.DataFrame,
    seg: Optional[Segmentation] = None,
    value: str = "pct_soft",
    palette: str = DEFAULT_PALETTE,
    figsize=(WIDTH_MID, 2.8),
):
    """Stacked bar of the percentage of the scored genome in each state, per sample or condition.

    ``summary`` is :func:`chromstate.summary.state_summary` (per sample) or
    :func:`chromstate.summary.condition_summary` (per condition). ``value`` picks the
    column: ``pct_soft`` (posterior mass, the default) or ``pct_hard``.
    """
    key = "condition" if "condition" in summary.columns else "sample"
    wide = summary.pivot(index=key, columns="state", values=value).fillna(0.0)
    order = (summary.drop_duplicates("state").sort_values("state_index")["state"].tolist()
             if "state_index" in summary.columns else list(wide.columns))
    wide = wide[order]
    colors = _label_colors(order, seg, palette)

    fig, ax = _new_fig(*figsize)
    bottom = np.zeros(len(wide))
    x = np.arange(len(wide))
    for lbl in order:
        v = wide[lbl].to_numpy()
        ax.bar(x, v, bottom=bottom, color=colors[lbl], edgecolor=SURFACE, linewidth=0.4,
               width=0.7, label=full_state_name(lbl), zorder=3)
        for xi, (b, h) in enumerate(zip(bottom, v)):
            if ANNOTATE_VALUES and h >= 6:
                ax.text(xi, b + h / 2, f"{h:.0f}%", ha="center", va="center", fontsize=FS_CELL,
                        color=INK)
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels(wide.index, rotation=0, fontsize=FS_TICK)
    ax.set_ylabel("Share of genome (%)" if value.startswith("pct") else value, fontsize=FS_LABEL)
    ax.set_ylim(0, max(bottom.max() * 1.02, 1))
    _title(ax, "State composition",
           "share of the scored genome in each state"
           + (", from posterior mass" if value == "pct_soft" else ", from hard calls"))
    leg = ax.legend(frameon=False, fontsize=FS_LEGEND, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                    handlelength=1.0)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    return fig


def plot_state_strength(
    summary: pd.DataFrame,
    seg: Optional[Segmentation] = None,
    palette: str = DEFAULT_PALETTE,
    figsize=(WIDTH_MID, 2.6),
):
    """Strength of each state per sample or condition: mean log-fold intensity, with spread.

    Zero is a typical instance of the state in the joint model; above zero the regions
    in that state carry more of its defining marks than usual, below zero less. Error
    bars are the replicate standard deviation when ``summary`` is per condition, else
    the within-sample posterior-weighted spread.
    """
    if "strength" not in summary.columns:
        raise ValueError("summary has no 'strength' column; run state_summary(include_strength=True)")
    key = "condition" if "condition" in summary.columns else "sample"
    groups = list(pd.unique(summary[key]))
    order = (summary.drop_duplicates("state").sort_values("state_index")["state"].tolist()
             if "state_index" in summary.columns else list(pd.unique(summary["state"])))
    err_col = "strength_sd"
    colors = _label_colors(order, seg, palette)

    fig, ax = _new_fig(*figsize)
    n = len(groups)
    w = 0.8 / max(n, 1)
    x = np.arange(len(order))
    hatches = [None, "////", "....", "xxxx", "\\\\\\\\", "++++"]
    for i, g in enumerate(groups):
        sub = summary.loc[summary[key] == g].set_index("state").reindex(order)
        vals = sub["strength"].to_numpy(dtype=float)
        err = sub[err_col].to_numpy(dtype=float) if err_col in sub.columns else None
        ax.bar(x - 0.4 + w * (i + 0.5), vals, w, color=[colors[s] for s in order],
               edgecolor=INK, linewidth=0.4, hatch=hatches[i % len(hatches)], label=str(g),
               yerr=None if err is None else np.nan_to_num(err), error_kw={"elinewidth": 0.5,
               "capsize": 1.2, "ecolor": INK}, zorder=3)
    ax.axhline(0, color=INK, linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(state_labels(order), rotation=45, ha="right", fontsize=FS_TICK)
    ax.set_ylabel("Strength (log fold)", fontsize=FS_LABEL)
    _title(ax, "State strength",
           "posterior-weighted mean intensity of each state; hatching distinguishes "
           + ("conditions" if key == "condition" else "samples"))
    handles = [Patch(facecolor=SURFACE, edgecolor=INK, hatch=hatches[i % len(hatches)], label=str(g))
               for i, g in enumerate(groups)]
    leg = ax.legend(handles=handles, frameon=False, fontsize=FS_LEGEND, loc="upper left",
                    bbox_to_anchor=(1.01, 1.0))
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    return fig


def plot_condition_differences(
    differences: pd.DataFrame,
    seg: Optional[Segmentation] = None,
    palette: str = DEFAULT_PALETTE,
    figsize=(WIDTH_DOUBLE, 2.6),
):
    """Per state: change in genome share and in strength for every condition vs the reference.

    ``differences`` is ``ConditionComparison.differences`` from
    :func:`chromstate.summary.compare_conditions`.
    """
    conds = list(pd.unique(differences["condition"]))
    ref = str(differences["reference"].iloc[0])
    order = list(pd.unique(differences["state"]))
    colors = _label_colors(order, seg, palette)
    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor=SURFACE)
    for ax, col, ylabel, title in (
        (axes[0], "delta_pct_soft", "Change in genome share (points)", "Amount"),
        (axes[1], "delta_strength", "Change in strength (log fold)", "Strength"),
    ):
        _style_axes(ax)
        n = len(conds)
        w = 0.8 / max(n, 1)
        x = np.arange(len(order))
        for i, c in enumerate(conds):
            sub = differences.loc[differences["condition"] == c].set_index("state").reindex(order)
            ax.bar(x - 0.4 + w * (i + 0.5), sub[col].to_numpy(dtype=float), w,
                   color=[colors[s] for s in order], edgecolor=INK, linewidth=0.4,
                   alpha=1.0 - 0.5 * i / max(n - 1, 1), zorder=3)
        if n > 1 and ax is axes[1]:
            # one key for the shading, in place of a condition name on every bar
            handles = [Patch(facecolor="#77756f", edgecolor=INK, linewidth=0.4,
                             alpha=1.0 - 0.5 * i / max(n - 1, 1), label=str(c))
                       for i, c in enumerate(conds)]
            ax.legend(handles=handles, frameon=False, fontsize=FS_LEGEND, loc="best")
        ax.axhline(0, color=INK, linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(state_labels(order), rotation=45, ha="right", fontsize=FS_TICK)
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        _title(ax, f"{title} vs {ref}", f"conditions: {', '.join(conds)}")
    fig.tight_layout()
    return fig


def plot_alluvial(
    flows: pd.DataFrame,
    stages: Optional[Sequence[str]] = None,
    weight: str = "weight",
    colors: Optional[Dict[str, str]] = None,
    seg: Optional[Segmentation] = None,
    palette: str = DEFAULT_PALETTE,
    order: Optional[Sequence[str]] = None,
    min_fraction: float = 0.002,
    gap: float = 0.015,
    node_width: float = 0.08,
    label_nodes: bool = True,
    title: str = "State flow",
    subtitle: Optional[str] = None,
    unit: str = "",
    figsize=(WIDTH_MID, 3.2),
):
    """Alluvial (multi-stage flow) plot from a long table of paths and weights.

    ``flows`` has one column per stage (``stages``, in order) holding a categorical
    label, plus a ``weight`` column (genes, or base pairs). Build it with
    :func:`chromstate.patterns.flow_table`. Ribbons are coloured by their label at the
    first stage; nodes are stacked in ``order`` (default: the order labels appear, which
    for a segmentation follows the quiet-to-active state order). Paths below
    ``min_fraction`` of the total are pooled into the node heights but not drawn, which
    keeps a genome-wide plot from becoming a thousand hairlines.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    stages = list(stages or flows.attrs.get("stages") or
                  [c for c in flows.columns if c != weight])
    if len(stages) < 2:
        raise ValueError("an alluvial plot needs at least two stages")
    tab = flows[stages + [weight]].copy()
    tab[stages] = tab[stages].astype(str)
    total = float(tab[weight].sum())
    if total <= 0:
        raise ValueError("flows carry no weight")

    labels_seen: List[str] = []
    if order is not None:
        labels_seen = [str(o) for o in order]
    if seg is not None:
        labels_seen += [lbl for lbl in seg.labels if lbl not in labels_seen]
    for s in stages:
        for lbl in tab[s]:
            if lbl not in labels_seen:
                labels_seen.append(lbl)
    colors = _label_colors(labels_seen, seg, palette, colors)

    # node geometry: per stage, cumulative start of each label's block
    n_stages = len(stages)
    usable = 1.0 - gap * (len(labels_seen) - 1)
    node_pos: List[Dict[str, Tuple[float, float]]] = []
    for s in stages:
        sizes = tab.groupby(s)[weight].sum()
        present = [lbl for lbl in labels_seen if lbl in sizes.index and sizes[lbl] > 0]
        y = 0.0
        pos: Dict[str, Tuple[float, float]] = {}
        for lbl in present:
            h = usable * float(sizes[lbl]) / total
            pos[lbl] = (y, h)
            y += h + gap
        node_pos.append(pos)

    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    xs = np.linspace(0.0, 1.0, n_stages)

    # ribbons between consecutive stages, stacked within each node in path order
    tab = tab.sort_values([stages[0]] + stages[1:], key=lambda c: c.map({lbl: i for i, lbl in enumerate(labels_seen)}))
    for i in range(n_stages - 1):
        a, b = stages[i], stages[i + 1]
        offs_a = {lbl: y for lbl, (y, _) in node_pos[i].items()}
        offs_b = {lbl: y for lbl, (y, _) in node_pos[i + 1].items()}
        pair = tab.groupby([a, b], sort=False)[weight].sum().reset_index()
        pair["first"] = tab.groupby([a, b], sort=False)[stages[0]].first().to_numpy()
        pair = pair.sort_values(
            [a, b], key=lambda c: c.map({lbl: k for k, lbl in enumerate(labels_seen)}))
        # left-side offsets follow the source order, right-side offsets the target order
        left_cursor = dict(offs_a)
        right_cursor = dict(offs_b)
        ribbons = []
        for _, r in pair.iterrows():
            h = usable * float(r[weight]) / total
            y0 = left_cursor[r[a]]
            left_cursor[r[a]] += h
            ribbons.append((r[a], r[b], h, y0, r["first"], float(r[weight])))
        for src, dst, h, y0, first, wgt in sorted(ribbons, key=lambda t: (labels_seen.index(t[1]), labels_seen.index(t[0]))):
            y1 = right_cursor[dst]
            right_cursor[dst] += h
            if wgt / total < min_fraction:
                continue
            x0, x1 = xs[i] + node_width / 2, xs[i + 1] - node_width / 2
            xm = 0.5 * (x0 + x1)
            verts = [(x0, y0), (xm, y0), (xm, y1), (x1, y1), (x1, y1 + h), (xm, y1 + h),
                     (xm, y0 + h), (x0, y0 + h), (x0, y0)]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
                     Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
            ax.add_patch(PathPatch(Path(verts, codes), facecolor=colors[first],
                                   edgecolor="none", alpha=0.55 if src == dst else 0.75, zorder=2))

    for i, pos in enumerate(node_pos):
        for lbl, (y, h) in pos.items():
            ax.add_patch(plt.Rectangle((xs[i] - node_width / 2, y), node_width, h,
                                       facecolor=colors[lbl], edgecolor=INK, linewidth=0.4, zorder=3))
            if label_nodes and h >= 0.03:
                share = 100.0 * h / usable
                txt = full_state_name(lbl) + (f"\n{share:.0f}%{unit}" if ANNOTATE_VALUES else "")
                ha = "right" if i == 0 else ("left" if i == n_stages - 1 else "center")
                x = xs[i] - node_width / 2 - 0.01 if i == 0 else (
                    xs[i] + node_width / 2 + 0.01 if i == n_stages - 1 else xs[i])
                ax.text(x, y + h / 2, txt, ha=ha, va="center", fontsize=FS_CELL, color=INK,
                        zorder=4, bbox=None if i in (0, n_stages - 1) else
                        dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none", alpha=0.8))
    for i, s in enumerate(stages):
        ax.text(xs[i], 1.0 + gap, s, ha="center", va="bottom", fontsize=FS_LABEL, color=INK)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.01, 1.0 + 3 * gap + 0.02)
    ax.axis("off")
    _title(ax, title, subtitle)
    fig.tight_layout()
    return fig


def plot_region_alluvial(seg: Segmentation, groups: Dict[str, Sequence[str]],
                         by: str = "dom", palette: str = DEFAULT_PALETTE, **kwargs):
    """Alluvial of base pairs moving between states across conditions."""
    from .patterns import default_state_classes, flow_table, region_patterns

    classes = default_state_classes(seg) if by == "class" else None
    pat = region_patterns(seg, groups, state_classes=classes)
    flows = flow_table(pat, by=by)
    kwargs.setdefault("title", "Regional state flow")
    kwargs.setdefault("subtitle", "each ribbon is the base pairs whose dominant state follows that path")
    return plot_alluvial(flows, seg=seg, palette=palette, **kwargs)


def plot_gene_alluvial(dom: pd.DataFrame, seg: Optional[Segmentation] = None,
                       by: str = "dom", palette: str = DEFAULT_PALETTE, **kwargs):
    """Alluvial of genes moving between dominant promoter (or enhancer) states."""
    from .patterns import flow_table

    flows = flow_table(dom, by=by, weight=None)
    kwargs.setdefault("title", "Gene state flow")
    kwargs.setdefault("subtitle", f"{len(dom):,} genes by dominant state of the "
                      f"{ {'P': 'promoter', 'E': 'distal space', 'B': 'gene body'}.get(dom.attrs.get('part'), 'region')}")
    kwargs.setdefault("min_fraction", 0.005)
    return plot_alluvial(flows, seg=seg, palette=palette, **kwargs)


def plot_pattern_counts(counts: pd.DataFrame, by: str = "pattern", top: int = 20,
                        figsize=(WIDTH_SINGLE, 2.8)):
    """Horizontal bars of the most common gene state patterns or categories."""
    sub = counts.head(top).iloc[::-1]
    fig, ax = _new_fig(*figsize)
    ax.barh(np.arange(len(sub)), sub["n_genes"], color="#2a78d6", edgecolor=SURFACE, zorder=3)
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels([" > ".join(full_state_name(p) for p in str(v).split(">")) if by == "pattern" else str(v)
                        for v in sub[by]], fontsize=FS_TICK)
    ax.set_xlabel("Genes", fontsize=FS_LABEL)
    _title(ax, f"Gene {by}s", f"top {len(sub)} of {len(counts)}")
    fig.tight_layout()
    return fig


def plot_enhancer_contributions(
    importance: pd.DataFrame, gene: str, sample: Optional[str] = None,
    figsize=(WIDTH_SINGLE, 2.2),
):
    """One gene's enhancers along the regulatory window, bar height = importance."""
    sub = importance.loc[(importance["gene_id"] == gene) | (importance["name"] == gene)]
    if sample is not None:
        sub = sub.loc[sub["sample"] == sample]
    if sub.empty:
        raise ValueError(f"no enhancer rows for {gene!r}")
    fig, ax = _new_fig(*figsize)
    x = sub["signed_distance"].to_numpy(dtype=float) / 1e3
    y = sub["importance"].to_numpy(dtype=float)
    ax.bar(x, y, width=max(1.0, (x.max() - x.min()) / 60) if len(x) > 1 else 1.0,
           color=np.where(y >= 0, "#e34948", "#2a78d6"), edgecolor=INK, linewidth=0.3, zorder=3)
    ax.axvline(0, color=INK, linewidth=0.5, linestyle=":")
    ax.axhline(0, color=INK, linewidth=0.5)
    ax.set_xlabel("Distance from TSS (kb)", fontsize=FS_LABEL)
    ax.set_ylabel("Importance", fontsize=FS_LABEL)
    name = str(sub["name"].iloc[0])
    _title(ax, f"Enhancers of {name}", "in-silico deletion through the promoter + enhancer model"
           + (f", {sample}" if sample else ""))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Statistical tests and expression ranking
# ---------------------------------------------------------------------------


def _stars(q: float) -> str:
    if not np.isfinite(q):
        return ""
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else "n.s."


def plot_state_tests(
    tests: pd.DataFrame,
    seg: Optional[Segmentation] = None,
    palette: str = DEFAULT_PALETTE,
    figsize=(WIDTH_MID, 2.8),
):
    """Per state: change in genome share with its block-bootstrap 95% interval and
    significance stars from the replicate t-test (``q_rep``; ``q_block`` in brackets
    when no replicate test was possible).

    ``tests`` is :func:`chromstate.summary.state_tests` output.
    """
    conds = list(pd.unique(tests["condition"]))
    ref = str(tests["reference"].iloc[0])
    order = list(pd.unique(tests["state"]))
    colors = _label_colors(order, seg, palette)
    fig, ax = _new_fig(*figsize)
    n = len(conds)
    w = 0.8 / max(n, 1)
    x = np.arange(len(order))
    for i, c in enumerate(conds):
        sub = tests.loc[tests["condition"] == c].set_index("state").reindex(order)
        xi = x - 0.4 + w * (i + 0.5)
        d = sub["delta_pct"].to_numpy(dtype=float)
        lo = d - sub["delta_lo"].to_numpy(dtype=float)
        hi = sub["delta_hi"].to_numpy(dtype=float) - d
        ax.bar(xi, d, w, color=[colors[s] for s in order], edgecolor=INK, linewidth=0.4,
               alpha=1.0 - 0.45 * i / max(n - 1, 1), zorder=3)
        ax.errorbar(xi, d, yerr=np.vstack([np.clip(lo, 0, None), np.clip(hi, 0, None)]), fmt="none",
                    ecolor=INK, elinewidth=0.5, capsize=1.5, zorder=4)
        for xx, dd, qr, qb, h in zip(xi, d, sub["q_rep"], sub["q_block"], sub["delta_hi"]):
            star = _stars(qr) if np.isfinite(qr) else f"[{_stars(qb)}]"
            y = h if dd >= 0 else sub.loc[sub.index[list(xi).index(xx)], "delta_lo"]
            ax.text(xx, y, star, ha="center", va="bottom" if dd >= 0 else "top", fontsize=FS_CELL, color=INK)
    ax.axhline(0, color=INK, linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(state_labels(order), rotation=45, ha="right", fontsize=FS_TICK)
    ax.set_ylabel("Change in genome share (points)", fontsize=FS_LABEL)
    _title(ax, f"State share vs {ref}",
           "bars = difference, whiskers = 95% block-bootstrap interval, stars = replicate "
           "t-test FDR (brackets: block bootstrap when replicates are missing)"
           + (f"; shading: {', '.join(conds)}" if n > 1 else ""))
    fig.tight_layout()
    return fig


def plot_state_importance(
    table: pd.DataFrame,
    seg: Optional[Segmentation] = None,
    palette: str = DEFAULT_PALETTE,
    title: str = "State importance",
    figsize=(WIDTH_SINGLE, 2.8),
):
    """Unique held-out R² per state with its bootstrap interval, coloured by state and
    annotated with the direction of association (``+`` higher expression).

    ``table`` is ``StateImportance.table`` from
    :func:`chromstate.regulation.state_importance` (sorted by ``unique_r2``).
    """
    tab = table.sort_values("unique_r2").reset_index(drop=True)
    order = tab["state"].tolist()
    colors = _label_colors(order, seg, palette)
    fig, ax = _new_fig(*figsize)
    ax.grid(False)
    ax.grid(True, axis="x", color=GRID, linewidth=0.3, zorder=0)
    y = np.arange(len(tab))
    v = tab["unique_r2"].to_numpy(dtype=float)
    lo = v - tab["unique_lo"].to_numpy(dtype=float)
    hi = tab["unique_hi"].to_numpy(dtype=float) - v
    ax.barh(y, v, color=[colors[s] for s in order], edgecolor=INK, linewidth=0.4, zorder=3)
    ax.errorbar(v, y, xerr=np.vstack([np.clip(lo, 0, None), np.clip(hi, 0, None)]), fmt="none",
                ecolor=INK, elinewidth=0.5, capsize=1.5, zorder=4)
    if ANNOTATE_VALUES:
        for yy, vv, d, rho in zip(y, v, tab["direction"], tab["marginal_rho"]):
            ax.text(max(vv, 0) + 0.005, yy, f"{'+' if d >= 0 else '−'}  ρ={rho:.2f}", va="center",
                    ha="left", fontsize=FS_CELL, color=INK)
    ax.axvline(0, color=INK, linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(state_labels(order), fontsize=FS_TICK)
    ax.set_xlabel("Held-out R² lost on permutation", fontsize=FS_LABEL)
    _title(ax, title, "sign = direction of association; ρ = single-state held-out Spearman")
    fig.tight_layout()
    return fig


def plot_part_ranking(parts: pd.DataFrame, figsize=(WIDTH_SINGLE, 2.4)):
    """Held-out Spearman ρ (with 95% bootstrap interval) for each combination of gene parts.

    ``parts`` is :func:`chromstate.regions.rank_parts_for_expression` output. Reading
    ``P+E`` against ``P`` answers whether the distal enhancer space adds anything the
    promoter does not already carry.
    """
    tab = parts.sort_values("spearman").reset_index(drop=True)
    fig, ax = _new_fig(*figsize)
    ax.grid(False)
    ax.grid(True, axis="x", color=GRID, linewidth=0.3, zorder=0)
    y = np.arange(len(tab))
    v = tab["spearman"].to_numpy(dtype=float)
    lo = v - tab["rho_lo"].to_numpy(dtype=float)
    hi = tab["rho_hi"].to_numpy(dtype=float) - v
    names = {"P": "Promoter", "E": "enhancers", "B": "gene body"}
    labels = [" + ".join(names.get(p, p) for p in str(s).split("+")) for s in tab["parts"]]
    labels = [lbl[:1].upper() + lbl[1:] for lbl in labels]
    col = ["#AE231B" if "E" in str(s).split("+") else "#395493" for s in tab["parts"]]
    ax.barh(y, v, color=col, edgecolor=INK, linewidth=0.4, zorder=3)
    ax.errorbar(v, y, xerr=np.vstack([np.clip(lo, 0, None), np.clip(hi, 0, None)]), fmt="none",
                ecolor=INK, elinewidth=0.5, capsize=1.5, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FS_TICK)
    ax.set_xlabel("Spearman, held-out genes", fontsize=FS_LABEL)
    _title(ax, "Gene part", "red = includes the enhancer space; whiskers = 95% bootstrap over genes")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Composite (multi-panel) figures at journal size
# ---------------------------------------------------------------------------

#: Panel letters: 8 pt bold lowercase, the one piece of text allowed above the 7 pt scale.
FS_PANEL = 8.0
#: Deepest figure that still leaves room for a legend on a Nature page.
MAX_DEPTH_MM = 170.0
_NATURE_WIDTHS_MM = (89.0, 120.0, 183.0)


def figure_mm(width_mm: float = 183.0, height_mm: float = 120.0) -> plt.Figure:
    """An empty figure of an exact print size in millimetres (default: double column).

    Composite figures are laid out with ``fig.add_gridspec`` and saved with
    :func:`save_figure`, which keeps this size exactly; a tight bounding box would let a
    long label change the width the journal places the figure at.
    """
    return plt.figure(figsize=(width_mm * MM, height_mm * MM), facecolor=SURFACE)


def panel_letter(ax: plt.Axes, letter: str, dx_mm: float = 0.0, dy_mm: float = 0.5):
    """Put a bold lowercase panel letter at the top-left corner of the panel.

    The letter is aligned to the left edge of everything the axes draws (tick and axis
    labels included), so the letters of a row line up with the panels' visual edges
    rather than with their data areas. ``dx_mm`` / ``dy_mm`` nudge it from there.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    top = max(ax.get_position().y1, box.y1)
    x = box.x0 + dx_mm * MM / fig.get_figwidth()
    y = top + dy_mm * MM / fig.get_figheight()
    return fig.text(x, y, letter.lower(), fontsize=FS_PANEL, fontweight="bold", color=INK,
                    ha="left", va="bottom", gid="panel_letter")


def save_panels(fig: plt.Figure, groups: Dict[str, Sequence], outdir: str, stem: str,
                png_dpi: int = 600, pad_mm: float = 1.0) -> List[str]:
    """Write each group of panels in ``fig`` as its own figure, cropped to what it draws.

    ``groups`` maps a file suffix to the artists that make up one stand-alone figure:
    usually one axes, or a few axes that share a category axis, plus any panel letters,
    colour bars or headings that belong with them. Every other axes and figure-level text
    is hidden while a group is saved, so neighbours never leak into the crop. Each group
    is written as ``<stem>_<suffix>.pdf`` (vector, editable text) and, when ``png_dpi``
    is non-zero, a PNG beside it. The text keeps its point size, so a panel that was
    one-sixth of a composite page is read at full size on its own.
    """
    import os

    from matplotlib.transforms import Bbox

    os.makedirs(outdir, exist_ok=True)
    enforce_text_style(fig)
    everything = list(fig.axes) + list(fig.texts) + list(fig.legends)
    was_visible = {id(a): a.get_visible() for a in everything}
    paths: List[str] = []
    for suffix, artists in groups.items():
        keep = {id(a) for a in artists}
        for a in everything:
            a.set_visible(was_visible[id(a)] and id(a) in keep)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [a.get_tightbbox(renderer) for a in artists if a.get_visible()]
        # an axis label wider than its axes is not always inside the axes' tight box
        for a in artists:
            if isinstance(a, plt.Axes):
                boxes += [t.get_window_extent(renderer) for t in (a.xaxis.label, a.yaxis.label, a.title)
                          if t.get_text().strip()]
        boxes = [b for b in boxes if b is not None and b.width > 0 and b.height > 0]
        if not boxes:
            continue
        bbox = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted()).padded(pad_mm * MM)
        path = os.path.join(outdir, f"{stem}_{suffix}.pdf")
        fig.savefig(path, bbox_inches=bbox, facecolor=SURFACE)
        if png_dpi:
            fig.savefig(path[:-4] + ".png", bbox_inches=bbox, dpi=png_dpi, facecolor=SURFACE)
        paths.append(path)
    for a in everything:
        a.set_visible(was_visible[id(a)])
    plt.close(fig)
    return paths


def check_nature_style(fig: plt.Figure, max_depth_mm: float = MAX_DEPTH_MM) -> List[str]:
    """List every way ``fig`` departs from Nature's artwork rules; empty means it conforms.

    Checked: width is at most a double column (183 mm) and depth at most ``max_depth_mm``;
    every visible piece of text is black and 5-7 pt (panel letters: 8 pt bold); no drawn
    line is thinner than 0.25 pt or thicker than 1 pt; and text will be embedded as
    editable TrueType. The check reads the figure as drawn, so call it after layout.
    """
    from matplotlib.colors import to_hex
    from matplotlib.lines import Line2D
    from matplotlib.text import Text

    problems: List[str] = []
    width_mm = fig.get_figwidth() / MM
    height_mm = fig.get_figheight() / MM
    if width_mm > _NATURE_WIDTHS_MM[-1] + 0.05:
        problems.append(f"width {width_mm:.1f} mm exceeds the 183 mm double column")
    if height_mm > max_depth_mm + 0.05:
        problems.append(f"depth {height_mm:.1f} mm exceeds {max_depth_mm:.0f} mm")
    if mpl.rcParams["pdf.fonttype"] != 42:
        problems.append("pdf.fonttype is not 42: text would be saved as outlines")
    for t in fig.findobj(Text):
        text = t.get_text().strip()
        if not text or not t.get_visible():
            continue
        size = t.get_fontsize()
        is_letter = t.get_gid() == "panel_letter"
        if is_letter:
            if abs(size - FS_PANEL) > 0.01 or t.get_fontweight() not in ("bold", 700):
                problems.append(f"panel letter {text!r} is not 8 pt bold")
        elif size < 5.0 - 0.01 or size > 7.0 + 0.01:
            problems.append(f"text {text[:30]!r} is {size:.1f} pt (allowed 5-7 pt)")
        if to_hex(t.get_color()) != "#000000":
            problems.append(f"text {text[:30]!r} is not black")
    for line in fig.findobj(Line2D):
        if not line.get_visible() or line.get_linestyle() in ("None", "none", "", " "):
            continue
        if len(line.get_xdata()) == 0:
            continue
        width = line.get_linewidth()
        if width < 0.25 - 0.01 or width > 1.0 + 0.01:
            problems.append(f"line width {width:.2f} pt (allowed 0.25-1 pt)")
    return sorted(set(problems))


def save_figure(fig: plt.Figure, path: str, png: Optional[str] = None, dpi: int = 600) -> str:
    """Save a composite figure at its exact size as a vector PDF (text kept editable).

    ``png`` optionally writes a raster preview next to it (for screens and reviews; the
    PDF is the artwork). Raises ``ValueError`` if the figure breaks the Nature rules that
    :func:`check_nature_style` tests, so a non-conforming figure is never written quietly.
    """
    import os

    enforce_text_style(fig)
    problems = check_nature_style(fig)
    if problems:
        raise ValueError(f"{os.path.basename(path)} is not in Nature format: " + "; ".join(problems))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    if png:
        os.makedirs(os.path.dirname(os.path.abspath(png)), exist_ok=True)
        fig.savefig(png, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return path
