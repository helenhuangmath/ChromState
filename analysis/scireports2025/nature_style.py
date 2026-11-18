"""Nature-journal figure style for this analysis.

Nature's artwork guide: single column 89 mm, double column 183 mm, maximum depth
~170 mm; sans-serif type (Arial/Helvetica); lines 0.25-1 pt; no chart junk; editable
text in vector output. Panel letters are 8 pt bold lowercase. All text is Arial (or an
Arial-metric face where Arial is not installed) and pure black; colour is used only for
data marks.

Every lettered panel is saved as its own file (``save``), or with the panels it shares an
axis with, so the type is set at 7 pt: large for a panel read on its own, and still inside
the journal's range if the panels are later assembled.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1 / 25.4
SINGLE = 89 * MM
ONE_HALF = 120 * MM
DOUBLE = 183 * MM

# Cell states: the first three slots of the validated categorical palette, which pass
# colour-vision-deficiency separation for every pair (not just adjacent ones).
CONDITION_COLORS = {"Naive": "#2a78d6", "Arm": "#eb6834", "Cl13": "#1baf7a"}
CONDITION_NAMES = {"Naive": "TN", "Arm": "TMEM", "Cl13": "TEX"}

# Method comparison: baselines recede in grey, ChromState carries the one accent.
ACCENT = "#2a78d6"
BASELINE = "#9a9893"
BASELINE_DARK = "#52514e"
INK = "#000000"
INK_SECONDARY = "#52514e"
DIVERGING = "RdBu_r"
#: Arial first; the others have the same metrics and stand in where it is not installed.
FONT_STACK = ["Arial", "Liberation Sans", "Arimo", "Helvetica", "Nimbus Sans", "DejaVu Sans"]
#: Text size for anything set by hand in a figure script.
FS = 7


def apply() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "mathtext.fontset": "dejavusans",
        "text.color": INK,
        "legend.labelcolor": INK,
        "font.size": FS,
        "axes.titlesize": FS,
        "axes.labelsize": FS,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "legend.fontsize": FS,
        "legend.title_fontsize": FS,
        "legend.frameon": False,
        "axes.linewidth": 0.5,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlepad": 3,
        "axes.labelpad": 2,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5,
        "xtick.color": INK,
        "ytick.color": INK,
        "lines.linewidth": 0.75,
        "lines.markersize": 3,
        "patch.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.dpi": 150,
    })


def panel_label(ax: plt.Axes, letter: str, x: float = -0.18, y: float = 1.06) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="left", color=INK)
    # remembered so that save() can write this panel as its own figure
    ax.figure.__dict__.setdefault("_ns_panels", {})[letter] = ax


def active_first(labels):
    """State order for figures: active transcription start sites first, quiescent states
    last (the fitted model lists its states the other way, quiet to active, by signal)."""
    from chromstate.model.annotate import display_order

    return display_order(labels)


def _check_text(fig: plt.Figure, name: str) -> None:
    from matplotlib.colors import to_hex
    from matplotlib.text import Text

    bad = [
        (t.get_text(), to_hex(t.get_color()), t.get_fontname())
        for t in fig.findobj(Text)
        if t.get_visible() and t.get_text().strip()
        and (to_hex(t.get_color()) != INK or t.get_fontname() not in FONT_STACK)
    ]
    if bad:
        raise ValueError(f"{name}: text not in black Arial: {bad[:5]}")


def save(fig: plt.Figure, outdir: Path, name: str, groups=None, shared=None) -> None:
    """Vector PDF(s) with embedded TrueType text -- the only output format.

    A figure with lettered panels is written as one file per panel, ``<Fig>_<suffix>.pdf``,
    cropped to what the panel draws. ``groups`` maps a file suffix to the letters that
    belong together (``{"a-c_prediction": "abc", "d_single_mark": "d"}``); by default every
    letter is its own file. Unlettered axes (colour bars) go with the nearest lettered
    panel. ``shared`` names the suffixes that also carry the figure-level legend and text
    (default: none). ``groups=False`` keeps the figure whole.

    Refuses to save a figure carrying any visible text that is not black Arial, so the
    house style cannot drift silently when a panel is edited.
    """
    from matplotlib.transforms import Bbox

    _check_text(fig, name)
    outdir.mkdir(parents=True, exist_ok=True)
    panels = fig.__dict__.get("_ns_panels", {})
    if groups is False or len(panels) < 2:
        fig.savefig(outdir / f"{name}.pdf")
        plt.close(fig)
        return
    groups = groups or {letter: letter for letter in panels}
    shared = set(shared or [])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # every unlettered axes joins the lettered panel whose centre is nearest
    owner = {id(ax): letter for letter, ax in panels.items()}
    centres = {letter: ax.get_position() for letter, ax in panels.items()}
    for ax in fig.axes:
        if id(ax) in owner:
            continue
        p = ax.get_position()
        owner[id(ax)] = min(centres, key=lambda k: ((centres[k].x0 + centres[k].x1 - p.x0 - p.x1) ** 2
                                                    + (centres[k].y0 + centres[k].y1 - p.y0 - p.y1) ** 2))
    loose = list(fig.legends) + list(fig.texts)
    stem = name.split("_")[0]
    for old in outdir.glob(f"{stem}_*.pdf"):
        old.unlink()
    for suffix, letters in groups.items():
        members = [ax for ax in fig.axes if owner[id(ax)] in letters]
        extra = loose if suffix in shared else []
        for a in list(fig.axes) + loose:
            a.set_visible(a in members or a in extra)
        fig.canvas.draw()
        boxes = [a.get_tightbbox(renderer) for a in members + extra]
        # an axis label wider than its axes is not always inside the axes' tight box
        boxes += [t.get_window_extent(renderer) for ax in members
                  for t in (ax.xaxis.label, ax.yaxis.label, ax.title) if t.get_text().strip()]
        boxes = [b for b in boxes if b is not None and b.width > 0 and b.height > 0]
        bbox = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted()).padded(1.0 * MM)
        fig.savefig(outdir / f"{stem}_{suffix}.pdf", bbox_inches=bbox)
    plt.close(fig)
