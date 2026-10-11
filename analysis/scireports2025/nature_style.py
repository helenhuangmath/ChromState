"""Nature-journal figure style for this analysis.

Nature's artwork guide: single column 89 mm, double column 183 mm, maximum depth
~170 mm; sans-serif type (Arial/Helvetica) at 5-7 pt; lines 0.25-1 pt; no chart
junk; editable text in vector output. Panel letters are 8 pt bold lowercase.
All text is Arial and pure black; colour is used only for data marks.
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


def apply() -> None:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "text.color": INK,
        "legend.labelcolor": INK,
        "font.size": 6,
        "axes.titlesize": 7,
        "axes.labelsize": 6,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "legend.fontsize": 5.5,
        "legend.title_fontsize": 6,
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


def save(fig: plt.Figure, outdir: Path, name: str) -> None:
    """Vector PDF with embedded TrueType Arial -- the only output format.

    Refuses to save a figure carrying any visible text that is not black Arial, so the
    house style cannot drift silently when a panel is edited.
    """
    from matplotlib.colors import to_hex
    from matplotlib.text import Text

    bad = [
        (t.get_text(), to_hex(t.get_color()), t.get_fontname())
        for t in fig.findobj(Text)
        if t.get_visible() and t.get_text().strip()
        and (to_hex(t.get_color()) != INK or t.get_fontname() != "Arial")
    ]
    if bad:
        raise ValueError(f"{name}: text not in black Arial: {bad[:5]}")
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf")
    plt.close(fig)
