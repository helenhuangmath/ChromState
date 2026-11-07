"""Statistical figures of the differential report and the benchmark PR curves."""

from __future__ import annotations

import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import chromstate as cs
from chromstate import viz_stats as vs
from chromstate.compare import differential_report, permuted_region_state_tests, region_state_tests
from chromstate.simulate import simulate

GROUPS = {"ctrl": ["a1", "a2", "a3"], "drug": ["b1", "b2", "b3"], "other": ["c1", "c2", "c3"]}


@pytest.fixture(scope="module")
def report():
    ds, truth = simulate(
        n_bins=2000, n_chroms=3, samples=tuple(s for v in GROUPS.values() for s in v),
        treated=tuple(GROUPS["drug"] + GROUPS["other"]),
        perturbation={"responder_fraction": 0.6, "from_state": "ReprPC", "to_state": "TssA"},
        seed=21, blur=0.1,
    )
    cs.normalize(ds)
    seg = cs.fit(ds, n_states=truth.n_states, n_restarts=1, adaptive_restarts=False, seed=0, n_iter=60)
    rep = differential_report(seg, GROUPS, tile_bins=10, min_bins=5, genome_tests=False)
    regions = rep.regions["ctrl_vs_drug"][["chrom", "start", "end", "name"]].reset_index(drop=True)
    return ds, truth, seg, rep, regions


def test_region_state_heatmap(report):
    _, _, seg, rep, _ = report
    fig = vs.plot_region_state_heatmap(rep.region_states, contrast="ctrl_vs_drug", top=30, seg=seg)
    assert fig.axes[0].get_images()


def test_region_state_heatmap_keeps_untested_state(report):
    # A state with no testable region has all-NaN q; pivot_table used to drop it and the
    # column lookup raised KeyError (seen on real chr19 replicate data, state EnhA).
    _, _, seg, rep, _ = report
    rs = rep.region_states.loc[rep.region_states["contrast"] == "ctrl_vs_drug"].copy()
    gone = rs["state"].iloc[0]
    rs.loc[rs["state"] == gone, ["qvalue", "pvalue"]] = np.nan
    fig = vs.plot_region_state_heatmap(rs, top=30, seg=seg)
    assert fig.axes[0].get_images()


def test_state_gain_loss(report):
    _, _, seg, rep, _ = report
    fig = vs.plot_state_gain_loss(rep.states, seg=seg)
    assert len(fig.axes) >= 2  # two contrasts (plus twin axes)


def test_permuted_pvalues_are_uniform_and_observed_are_not(report):
    _, _, seg, rep, regions = report
    obs = rep.region_states.loc[rep.region_states["contrast"] == "ctrl_vs_drug"]
    perm = permuted_region_state_tests(seg, GROUPS["ctrl"], GROUPS["drug"], regions, seed=0)
    assert set(perm.attrs["permutation"][0]) != set(GROUPS["ctrl"])
    pm = perm["pvalue"].dropna()
    assert perm["tested"].sum() == len(pm) < len(perm)
    assert 0.35 < pm.mean() < 0.65
    assert (pm < 0.05).mean() < 0.15
    # the floor is the 95% quantile of the region statistic over every replicate pair
    nf = rep.noise_floor
    assert len(nf.pairs) == 9 and len(nf.distribution) == 9 * len(regions)
    assert np.isclose(nf.threshold, np.quantile(nf.distribution, 0.95))
    assert (rep.regions["ctrl_vs_drug"]["tv_distance"] > nf.threshold).mean() < 0.5
    # the perturbation moves ReprPC -> TssA, so those states carry real signal
    tssa = [l for l in seg.labels if "Tss" in l and "Biv" not in l][0]
    assert (obs.loc[obs["state"] == tssa, "pvalue"] < 0.05).mean() > (pm < 0.05).mean()
    fig = vs.plot_pvalue_histograms(obs, perm)
    assert fig.axes


def test_concordance_and_ecdf(report):
    _, _, seg, rep, regions = report
    tssa = [l for l in seg.labels if "Tss" in l and "Biv" not in l][0]
    fig = vs.plot_replicate_concordance(seg, regions, tssa, ("a1", "a2"), ("a1", "b1"), noise_floor=rep.noise_floor)
    assert len(fig.axes) == 2
    fig2 = vs.plot_change_ecdf(seg, GROUPS, regions, reference="ctrl", noise_floor=rep.noise_floor)
    assert fig2.axes[0].lines


def test_state_volcano(report):
    _, _, seg, rep, _ = report
    fig = vs.plot_region_state_volcano(rep.region_states, contrast="ctrl_vs_drug", seg=seg)
    assert fig.axes[0].collections


def test_anova_volcano(report):
    _, _, seg, rep, _ = report
    assert rep.anova is not None
    fig = vs.plot_anova_volcano(rep.anova, seg=seg)
    assert fig.axes[0].collections


def test_report_write_with_figures(report, tmp_path):
    _, _, seg, rep, regions = report
    files = rep.write(str(tmp_path), seg=seg, regions=regions)
    names = [os.path.basename(f) for f in files]
    assert "state_gain_loss.pdf" in names and not any(n.endswith(".png") for n in names)
    assert any(n.startswith("pvalue_calibration_") for n in names)
    assert any(n.startswith("replicate_concordance_") for n in names)
    assert "change_ecdf.pdf" in names and "anova_volcano.pdf" in names
    assert "volcano_ctrl_vs_drug.pdf" in names and "state_volcano_ctrl_vs_drug.pdf" in names


def test_perturbation_pr_curves(report):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    from toolbench.core import ToolRun
    from toolbench.report import perturbation_scores, plot_perturbation_curves

    ds, truth, seg, _, _ = report
    soft = ToolRun(tool="chromstate", display_name="ChromState", states=seg.states, posterior=seg.posterior,
                   n_states=seg.n_states)
    hard = ToolRun(tool="hard", display_name="hard", states=seg.states, n_states=seg.n_states)
    score, y = perturbation_scores(soft, ds, truth)
    assert len(score) == len(y) == int(ds.sample_mask("a1").sum())
    fig = plot_perturbation_curves([soft, hard], ds, truth)
    assert len(fig.axes) == 2
