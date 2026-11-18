"""Composite-figure helpers: exact print size, panel letters, and the Nature-format check."""

import matplotlib

matplotlib.use("Agg")

import pytest

from chromstate import viz


def _figure():
    fig = viz.figure_mm(183, 60)
    ax = fig.add_axes([0.1, 0.2, 0.35, 0.6])
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("x")
    viz.panel_letter(ax, "A")
    return fig, ax


def test_conforming_figure_is_saved_at_exact_size(tmp_path):
    fig, _ = _figure()
    assert viz.check_nature_style(fig) == []
    assert fig.get_figwidth() == pytest.approx(183 * viz.MM)
    letters = [t for t in fig.texts if t.get_gid() == "panel_letter"]
    assert [t.get_text() for t in letters] == ["a"]
    path = viz.save_figure(fig, str(tmp_path / "fig.pdf"))
    assert (tmp_path / "fig.pdf").stat().st_size > 0 and path.endswith("fig.pdf")


def test_check_reports_each_departure(tmp_path):
    fig, ax = _figure()
    ax.set_title("too big", fontsize=12)
    ax.plot([0, 1], [1, 0], linewidth=2.0)
    fig.set_size_inches(200 * viz.MM, 180 * viz.MM)
    problems = " | ".join(viz.check_nature_style(fig))
    for expected in ("exceeds the 183 mm", "exceeds 170 mm", "12.0 pt", "line width 2.00"):
        assert expected in problems
    with pytest.raises(ValueError, match="not in Nature format"):
        viz.save_figure(fig, str(tmp_path / "bad.pdf"))


def test_save_panels_writes_one_cropped_file_per_group(tmp_path):
    fig = viz.figure_mm(183, 60)
    left = fig.add_axes([0.1, 0.2, 0.3, 0.6])
    right = fig.add_axes([0.6, 0.2, 0.3, 0.6])
    left.plot([0, 1], [0, 1])
    right.plot([0, 1], [1, 0])
    letter = viz.panel_letter(left, "a")
    paths = viz.save_panels(fig, {"a": [left, letter], "b": [right]}, str(tmp_path), "Fig0", png_dpi=100)
    assert [p.rsplit("/", 1)[-1] for p in paths] == ["Fig0_a.pdf", "Fig0_b.pdf"]
    from matplotlib.image import imread

    # each crop holds one panel, so it is well under half the 183 mm sheet
    for name in ("Fig0_a.png", "Fig0_b.png"):
        assert imread(str(tmp_path / name)).shape[1] < 0.5 * 183 * viz.MM * 100
