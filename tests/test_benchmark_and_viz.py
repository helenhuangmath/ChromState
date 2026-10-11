"""Tests for the benchmark harness, figures, and CLI."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import chromstate as cs
from chromstate.benchmark.baselines import METHODS, export_for_chromhmm
from chromstate.benchmark.metrics import (
    calibration_scores,
    clustering_scores,
    detection_scores,
    match_states,
    quantitative_sensitivity,
)
from chromstate.benchmark.run import (
    run_binarization_experiment,
    run_comparison,
    run_normalization_experiment,
)
from chromstate.model.binarize import poisson_binarize, poisson_threshold
from chromstate.simulate import simulate

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


# -- metrics -----------------------------------------------------------------


def test_match_states_is_invariant_to_relabelling():
    true = np.array([0, 0, 1, 1, 2, 2] * 20)
    permuted = np.array([2, 2, 0, 0, 1, 1] * 20)
    _, acc = match_states(true, permuted, 3, 3)
    assert acc == pytest.approx(1.0)
    assert clustering_scores(true, permuted, 3, 3)["ari"] == pytest.approx(1.0)


def test_calibration_flags_an_overconfident_model():
    n = 2000
    true = np.zeros(n, dtype=int)
    true[n // 2 :] = 1
    # Always claims 0.99 certainty, but is right only half the time.
    post = np.tile([0.99, 0.01], (n, 1))
    mapping = {0: 0, 1: 1}
    scores = calibration_scores(post, true, mapping)
    assert scores["accuracy"] == pytest.approx(0.5, abs=0.02)
    assert scores["overconfidence"] > 0.45
    assert scores["frac_saturated"] == pytest.approx(1.0)


def test_calibration_rewards_an_honest_model():
    n = 2000
    rng = np.random.default_rng(0)
    true = rng.integers(0, 2, size=n)
    post = np.full((n, 2), 0.5)
    scores = calibration_scores(post, true, {0: 0, 1: 1})
    assert abs(scores["overconfidence"]) < 0.05
    assert scores["mean_entropy"] == pytest.approx(1.0, abs=1e-6)


def test_detection_scores_on_a_perfect_ranking():
    score = np.arange(100.0)
    positive = np.arange(100) >= 90
    det = detection_scores(score, positive)
    assert det["auroc"] == pytest.approx(1.0)
    assert det["auprc"] == pytest.approx(1.0)


def test_quantitative_sensitivity_detects_a_flat_score():
    truth = np.arange(500.0)
    assert quantitative_sensitivity(truth, truth)["spearman"] == pytest.approx(1.0)
    assert np.isnan(quantitative_sensitivity(np.ones(500), truth)["spearman"])


# -- binarisation ------------------------------------------------------------


def test_poisson_threshold_increases_with_rate():
    assert poisson_threshold(0.1) < poisson_threshold(5.0) < poisson_threshold(100.0)


def test_binarize_produces_binary_output():
    ds, _ = simulate(n_bins=500, n_chroms=1, seed=30)
    B = poisson_binarize(ds.X)
    assert set(np.unique(B)) <= {0, 1}
    assert B.shape == ds.X.shape
    assert 0.0 < B.mean() < 0.5


def test_binarization_discards_magnitude():
    """The core claim about ChromHMM's preprocessing, measured rather than asserted."""
    ds, _ = simulate(n_bins=2000, n_chroms=1, seed=31)
    table = run_binarization_experiment(ds)
    assert (table["fold_range_collapsed"] > 2.0).all(), (
        "binarisation should collapse a wide count range onto the value 1"
    )


def test_export_for_chromhmm_writes_readable_files(tmp_path):
    ds, _ = simulate(n_bins=200, n_chroms=2, seed=32)
    paths = export_for_chromhmm(ds, str(tmp_path))
    assert len(paths) == len(ds.samples) * 2
    head = open(paths[0]).read().splitlines()
    assert len(head[0].split("\t")) == 2          # sample <TAB> chrom
    assert head[1].split("\t") == ds.marks        # mark header
    assert set(head[2].split("\t")) <= {"0", "1"}


# -- baselines ---------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(METHODS))
def test_every_method_returns_a_valid_result(key):
    ds, truth = simulate(n_bins=500, n_chroms=1, seed=33)
    cs.normalize(ds, method="depth")
    res = METHODS[key](ds, n_states=5, seed=0, n_restarts=1)
    assert res.states.shape == (ds.n_rows,)
    assert res.bivalency_score.shape == (ds.n_rows,)
    assert np.isfinite(res.bivalency_score).all()
    if res.posterior is not None:
        assert res.posterior.shape == (ds.n_rows, res.n_states)
        np.testing.assert_allclose(res.posterior.sum(axis=1), 1.0, atol=1e-6)


def test_comparison_table_has_a_row_per_method():
    ds, truth = simulate(n_bins=800, n_chroms=1, seed=34, blur=0.3)
    cs.normalize(ds, method="depth")
    table = run_comparison(ds, truth, seed=0, n_restarts=1)
    assert len(table) == 6
    assert "ChromState (NB-HMM)" in table["method"].values
    assert table["ari"].notna().sum() >= 4


def test_chromstate_beats_binarized_on_quantitative_sensitivity():
    """The headline claim: keeping counts should track signal magnitude better."""
    ds, truth = simulate(n_bins=2000, n_chroms=2, seed=35, blur=0.3)
    cs.normalize(ds, method="median_ratio")
    table = run_comparison(
        ds, truth, methods=("chromstate", "chromhmm_like"), seed=0, n_restarts=2
    ).set_index("method")
    assert (
        table.loc["ChromState (NB-HMM)", "signal_spearman"]
        > table.loc["ChromHMM-like (Bernoulli)", "signal_spearman"]
    )
    assert table.loc["ChromState (NB-HMM)", "ari"] > table.loc["ChromHMM-like (Bernoulli)", "ari"]


def test_normalization_experiment_separates_spikein_from_the_rest():
    table = run_normalization_experiment(n_bins=2000, seed=36).set_index("method")
    assert table.loc["spikein", "preserves_global_loss"]
    assert not table.loc["depth", "preserves_global_loss"]
    assert not table.loc["median_ratio", "preserves_global_loss"]
    assert table.loc["spikein", "biology_absorbed"] == pytest.approx(1.0, abs=0.1)


# -- figures -----------------------------------------------------------------


@pytest.fixture(scope="module")
def demo_seg():
    ds, truth = simulate(n_bins=600, n_chroms=2, seed=37,
                         perturbation={"responder_fraction": 0.5})
    cs.normalize(ds)
    return ds, truth, cs.fit(ds, n_states=truth.n_states, n_restarts=1, seed=0)


def test_all_figures_render(demo_seg, tmp_path):
    from chromstate import viz

    ds, _, seg = demo_seg
    table = cs.quantify(seg)
    a, b = ds.samples[0], ds.samples[1]
    figs = {
        "emissions": viz.plot_emission_heatmap(seg),
        "transitions": viz.plot_transition_heatmap(seg),
        "coverage": viz.plot_state_coverage(seg),
        "track": viz.plot_segmentation_track(seg, sample=a, max_bins=200),
        "confidence": viz.plot_confidence(table),
        "bivalency": viz.plot_bivalency(table, sample=a),
        "attribution": viz.plot_attribution(cs.state_mark_importance(seg, n_sample=500)),
        "flow": viz.plot_state_flow(cs.state_flow(seg, a, b), a, b),
        "occupancy": viz.plot_occupancy_change(cs.occupancy_change(seg)),
    }
    paths = viz.save_all(figs, str(tmp_path))
    assert len(paths) == len(figs)
    for p in paths:
        assert p.endswith(".pdf")
        assert (tmp_path / p.rsplit("/", 1)[-1]).stat().st_size > 3000


def test_figures_follow_nature_style(demo_seg):
    """All text black and 5-8 pt, and no figure wider than a double journal column."""
    from matplotlib.colors import to_hex
    from matplotlib.text import Text

    from chromstate import viz

    ds, _, seg = demo_seg
    table = cs.quantify(seg)
    figs = [viz.plot_emission_heatmap(seg), viz.plot_state_coverage(seg),
            viz.plot_confidence(table), viz.plot_bivalency(table, sample=ds.samples[0])]
    for fig in figs:
        viz.enforce_text_style(fig)
        assert fig.get_figwidth() <= viz.WIDTH_DOUBLE + 1e-9
        for t in fig.findobj(Text):
            if t.get_visible() and t.get_text().strip():
                assert to_hex(t.get_color()) == "#000000", t.get_text()
                assert 5.0 <= t.get_fontsize() <= 8.0, (t.get_text(), t.get_fontsize())
        plt.close(fig)


def test_cvd_palette_never_cycles_hues(demo_seg):
    from chromstate import viz

    _, _, seg = demo_seg
    colors = viz.state_colors(seg, palette="cvd")
    assert len(colors) == seg.n_states
    assert len(set(colors)) == len(colors), "hues must not repeat within the cap"


def test_unknown_palette_is_rejected(demo_seg):
    from chromstate import viz

    _, _, seg = demo_seg
    with pytest.raises(ValueError, match="Unknown palette"):
        viz.state_colors(seg, palette="rainbow")


def test_segmentation_bed_is_well_formed(demo_seg):
    ds, _, seg = demo_seg
    bed = seg.to_bed(sample=ds.samples[0], merge=True)
    assert (bed["end"] > bed["start"]).all()
    assert bed["name"].isin(seg.labels).all()
    assert bed["itemRgb"].str.count(",").eq(2).all()
    unmerged = seg.to_bed(sample=ds.samples[0], merge=False)
    assert len(bed) <= len(unmerged)


# -- CLI ---------------------------------------------------------------------


def test_cli_help_and_version():
    from chromstate.cli import main

    assert main(["--version"]) == 0
    assert main([]) == 1


def test_cli_fit_end_to_end(tmp_path):
    from chromstate.cli import main
    from chromstate.simulate import write_demo_files

    data = tmp_path / "data"
    manifest = write_demo_files(str(data), seed=38, n_bins=250)
    out = tmp_path / "out"
    code = main([
        "fit",
        "--manifest", manifest,
        "--chrom-sizes", str(data / "chrom.sizes"),
        "-k", "6",
        "--restarts", "1",
        "-o", str(out),
    ])
    assert code == 0
    for name in ("emissions.tsv", "transitions.tsv", "states.tsv", "quantification.tsv.gz"):
        assert (out / name).exists(), name
    assert (out / "figures" / "emissions.pdf").exists()
    assert (out / "control_segments.bed").exists()
