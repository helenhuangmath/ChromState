"""Tests for the v0.3 layer: state definitions, region quantification, the differential
report, expression prediction, the perturbation benchmark metric and single-cell transfer."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import chromstate as cs
from chromstate import sc
from chromstate.compare import differential_report, moderated_f_test, region_state_tests
from chromstate.definitions import StateDefinitions, match_states_to_definitions, parse_patterns
from chromstate.predict import predict_expression
from chromstate.quantify import quantify_regions
from chromstate.simulate import simulate

GROUPS = {"ctrl": ["a1", "a2"], "drug": ["b1", "b2"]}


def FAST(seed):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(max_iter=50, learning_rate=0.15, random_state=seed)


@pytest.fixture(scope="module")
def simulated():
    ds, truth = simulate(
        n_bins=2500, n_chroms=3, samples=("a1", "a2", "b1", "b2"), treated=("b1", "b2"),
        perturbation={"responder_fraction": 0.7, "from_state": "ReprPC", "to_state": "TssA"},
        seed=11, blur=0.1,
    )
    cs.normalize(ds)
    return ds, truth


@pytest.fixture(scope="module")
def fitted(simulated):
    ds, truth = simulated
    seg = cs.fit(ds, n_states=truth.n_states, n_restarts=2, seed=0)
    return ds, truth, seg


def _features(ds, step=30_000, length=6_000):
    rows = []
    for c in pd.unique(ds.bins["chrom"]):
        n = int((ds.bins["chrom"] == c).sum() / len(ds.samples))
        for i, start in enumerate(range(4_000, n * 200 - length - 4_000, step)):
            rows.append({"gene_id": f"{c}_g{i}", "name": f"{c}G{i}", "chrom": c, "start": start,
                         "end": start + length, "strand": "+" if i % 2 else "-"})
    return cs.FeatureSet(pd.DataFrame(rows))


# -- 1. states from given marks ------------------------------------------------


def test_parse_patterns_and_default_vocabulary():
    pats = parse_patterns("TssA: H3K4me3+ H3K27me3-\nTssBiv: H3K4me3+ H3K27me3+\nQuies:\n")
    assert pats["TssA"] == {"H3K4me3": "+", "H3K27me3": "-"}
    assert pats["Quies"] == {}
    defs = StateDefinitions.from_dict(pats, marks=["H3K4me3", "H3K27me3", "H3K9me3"])
    assert defs.pattern.loc["TssBiv", "H3K27me3"] == 1
    assert defs.pattern.loc["TssA", "H3K9me3"] == 0
    assert defs.describe("TssA") == "H3K4me3+ H3K27me3-"
    d = StateDefinitions.default(["H3K4me3", "H3K27me3", "H3K27ac", "H3K9me3"])
    assert {"TssA", "TssBiv", "ReprPC", "Het", "Quies"} <= set(d.states)
    assert "Tx" not in d.states  # needs H3K36me3


def test_definitions_file_roundtrip(tmp_path):
    defs = StateDefinitions.default(["H3K4me3", "H3K27me3"])
    path = tmp_path / "defs.tsv"
    defs.to_frame().to_csv(path, sep="\t", index=False)
    back = StateDefinitions.from_file(str(path), marks=["H3K4me3", "H3K27me3"])
    assert back.pattern.equals(defs.pattern)
    txt = tmp_path / "defs.txt"
    txt.write_text("A: H3K4me3+ H3K27me3-\nB: H3K27me3+\n")
    t = StateDefinitions.from_file(str(txt))
    assert t.states == ["A", "B"]


def test_fit_with_definitions_names_and_recovers_states(simulated):
    ds, truth = simulated
    defs = StateDefinitions.default(ds.marks)
    seg = cs.fit(ds, n_states=truth.n_states, n_restarts=1, adaptive_restarts=False, seed=0,
                 definitions=defs)
    assert set(defs.states) <= set(seg.labels)
    assert len(set(seg.labels)) == seg.n_states
    assert "definition_assignment" in seg.meta
    # defined states point at the right emission pattern
    em = seg.emission_frame()
    assert em.loc["TssBiv", "H3K4me3"] > em.loc["ReprPC", "H3K4me3"]
    assert em.loc["ReprPC", "H3K27me3"] > em.loc["TssA", "H3K27me3"]
    assert em.loc["Het", "H3K9me3"] > em.loc["Quies", "H3K9me3"]
    # and the segmentation agrees with the truth about as well as an unsupervised fit
    from sklearn.metrics import adjusted_rand_score

    ari = adjusted_rand_score(truth.states, seg.states)
    assert ari > 0.6
    assert any(r.startswith("defined:") for r in seg.rules)


def test_fixed_anchor_pins_defined_means(simulated):
    ds, truth = simulated
    defs = StateDefinitions.from_dict({"Active": {"H3K4me3": "+", "H3K27me3": "-"},
                                       "Polycomb": {"H3K27me3": "+", "H3K4me3": "-"},
                                       "Other": {}}, marks=ds.marks)
    seg = cs.fit(ds, n_states=3, n_restarts=1, adaptive_restarts=False, seed=0, definitions=defs,
                 anchor="fixed", n_iter=20)
    em = seg.model.emissions
    assert em.active
    pinned = em.anchor
    assert np.allclose(em.mu[pinned], em.anchor_mu[pinned])
    assert seg.meta["anchor"] == "fixed"
    assert set(seg.labels) == {"Active", "Polycomb", "Other"}


def test_extra_undefined_states_are_allowed(simulated):
    ds, truth = simulated
    defs = StateDefinitions.from_dict({"Active": {"H3K4me3": "+"}, "Polycomb": {"H3K27me3": "+"}},
                                      marks=ds.marks)
    seg = cs.fit(ds, n_states=5, n_restarts=1, adaptive_restarts=False, seed=0, definitions=defs,
                 n_iter=30)
    assert seg.n_states == 5
    assert {"Active", "Polycomb"} <= set(seg.labels)


def test_match_states_to_definitions_is_one_to_one():
    mu = np.array([[20.0, 0.5], [0.5, 15.0], [0.3, 0.3], [12.0, 10.0]])
    defs = StateDefinitions.from_dict({"A": {"m1": "+", "m2": "-"}, "R": {"m2": "+", "m1": "-"},
                                       "B": {"m1": "+", "m2": "+"}, "Q": {}}, marks=["m1", "m2"])
    assign, _ = match_states_to_definitions(mu, ["m1", "m2"], defs)
    assert assign == [0, 1, 3, 2]


def test_quantify_regions_sums_to_one(fitted):
    ds, truth, seg = fitted
    regions = pd.DataFrame({"chrom": ["chr1", "chr1", "chr2"], "start": [1000, 50_000, 2000],
                            "end": [5000, 60_000, 2200], "name": ["r1", "r2", "tiny"]})
    tab = quantify_regions(seg, regions, min_bins=2)
    assert set(tab["name"]) == {"r1", "r2"}
    occ = tab[[c for c in tab.columns if c.startswith("occ_")]].sum(axis=1)
    assert np.allclose(occ, 1.0)
    assert (tab["n_bins"] > 0).all()
    assert tab.loc[tab["name"] == "r1", "sample"].nunique() == len(ds.samples)
    bp = tab[[c for c in tab.columns if c.startswith("bp_")]].sum(axis=1)
    assert np.allclose(bp.loc[tab["name"] == "r1"], 4000)


# -- 2. statistics by region and by state --------------------------------------


def test_moderated_f_test_detects_a_planted_shift():
    rng = np.random.default_rng(0)
    R = 200
    g1 = rng.normal(0, 1, (R, 3))
    g2 = rng.normal(0, 1, (R, 3))
    g3 = rng.normal(0, 1, (R, 3))
    g3[:20] += 4.0
    F, p, (df1, df2) = moderated_f_test([g1, g2, g3])
    assert df1 == 2
    assert p[:20].mean() < 0.01
    assert p[20:].mean() > 0.3


def test_region_state_tests_long_table(fitted):
    ds, truth, seg = fitted
    tiles = pd.DataFrame({"chrom": "chr1", "start": np.arange(0, 200_000, 4000), "end": np.arange(4000, 204_000, 4000)})
    tab = region_state_tests(seg, GROUPS["ctrl"], GROUPS["drug"], tiles)
    assert len(tab) == len(tiles) * seg.n_states
    assert {"delta", "pvalue", "qvalue", "significant", "share_a", "share_b"} <= set(tab.columns)
    assert tab.attrs["tested"]
    # delta sums to ~0 over states within a region (shares are compositional)
    s = tab.groupby("name")["delta"].sum()
    assert np.allclose(s, 0.0, atol=1e-6)


def test_differential_report_finds_responders(fitted, tmp_path):
    ds, truth, seg = fitted
    rep = differential_report(seg, GROUPS, tile_bins=10, min_bins=5, fdr=0.1)
    assert rep.reference == "ctrl"
    assert "ctrl_vs_drug" in rep.regions
    assert not rep.region_states.empty
    assert set(rep.states["contrast"]) == {"ctrl_vs_drug"}
    assert {"n_sig_gain", "n_sig_loss", "mean_delta"} <= set(rep.states.columns)
    # the planted perturbation moves ReprPC -> TssA in treated samples
    st = rep.states.set_index("state")
    tssa = [lbl for lbl in seg.labels if "TssA" in lbl or "Tss" in lbl]
    assert tssa
    assert st.loc[tssa, "mean_delta"].sum() > 0
    # genome tests merged
    assert "delta_pct" in rep.states.columns
    files = rep.write(str(tmp_path / "diff"))
    assert any(f.endswith("region_state_tests.tsv") for f in files)
    assert os.path.exists(tmp_path / "diff" / "NOTES.md")
    # regions enriched in responders have higher omnibus statistic
    omni = rep.regions["ctrl_vs_drug"]
    resp = truth.responder_mask[ds.sample_mask("a1")]
    bins = ds.bins.loc[ds.sample_mask("a1")].reset_index(drop=True)
    from chromstate.dynamics import _region_membership

    frac = np.array([resp[m].mean() for m in _region_membership(bins, omni)])
    from scipy.stats import spearmanr

    assert spearmanr(frac, omni["tv_distance"]).correlation > 0.5


def test_differential_report_three_conditions_runs_anova():
    ds, truth = simulate(n_bins=1500, n_chroms=2, samples=("a1", "a2", "b1", "b2", "c1", "c2"),
                         treated=("b1", "b2", "c1", "c2"), seed=5, blur=0.1)
    cs.normalize(ds)
    seg = cs.fit(ds, n_states=truth.n_states, n_restarts=1, adaptive_restarts=False, seed=0, n_iter=40)
    rep = differential_report(seg, {"a": ["a1", "a2"], "b": ["b1", "b2"], "c": ["c1", "c2"]},
                              tile_bins=10, min_bins=5, genome_tests=False)
    assert rep.anova is not None
    assert {"F", "pvalue", "qvalue", "share_a", "share_b", "share_c"} <= set(rep.anova.columns)
    assert set(rep.regions) == {"a_vs_b", "a_vs_c"}


# -- 3. predicting expression --------------------------------------------------


def test_predict_expression_level_and_change(fitted, tmp_path):
    ds, truth, seg = fitted
    features = _features(ds)
    gq = cs.gene_quantification(seg, features)
    tssa = next(lbl for lbl in seg.labels if "Tss" in lbl)
    # expression rule: promoter occupancy of the active state, plus noise
    rng = np.random.default_rng(0)
    wide = gq.pivot(index="gene_id", columns="sample", values=f"P_occ_{tssa}")
    expr = 3.0 * wide + rng.normal(0, 0.3, wide.shape)
    res = predict_expression(gq, expression=expr, model=FAST, n_boot=50, n_perm=2)
    assert res.kind == "expression level"
    assert res.summary["rho"] > 0.3
    assert res.summary["best_single_state"] == tssa or res.summary["most_unique_state"] == tssa
    assert {"gene_id", "sample", "observed", "predicted"} <= set(res.predictions.columns)
    assert len(res.predictions) == wide.notna().sum().sum()
    files = res.write(str(tmp_path / "pred"), seg=seg, figures=True)
    assert os.path.exists(tmp_path / "pred" / "REPORT.md")
    assert any("predicted_vs_observed" in f for f in files)
    # change task: a linear rule on few genes, so the linear reference model is the right tool
    features2 = _features(ds, step=10_000)
    gq2 = cs.gene_quantification(seg, features2)
    wide2 = gq2.pivot(index="gene_id", columns="sample", values=f"P_occ_{tssa}")
    lfc = (wide2[["b1", "b2"]].mean(axis=1) - wide2[["a1", "a2"]].mean(axis=1)) * 3.0
    lfc = lfc + rng.normal(0, 0.1, len(lfc))
    res2 = predict_expression(gq2, lfc=lfc, contrast=(["a1", "a2"], ["b1", "b2"]), model="ridge", n_boot=50, n_perm=2)
    assert res2.kind == "expression change"
    assert res2.summary["rho"] > 0.3
    assert "sample" not in res2.predictions.columns


def test_predict_cli(fitted, tmp_path):
    ds, truth, seg = fitted
    features = _features(ds)
    gq = cs.gene_quantification(seg, features)
    tssa = next(lbl for lbl in seg.labels if "Tss" in lbl)
    wide = gq.pivot(index="gene_id", columns="sample", values=f"P_occ_{tssa}")
    expr = (np.exp(3.0 * wide) - 1).rename_axis("gene_id").reset_index()
    gpath, epath = tmp_path / "gq.tsv", tmp_path / "expr.tsv"
    gq.to_csv(gpath, sep="\t", index=False)
    expr.to_csv(epath, sep="\t", index=False)
    from chromstate.cli import main

    rc = main(["predict", "--genes", str(gpath), "--expression", str(epath), "--model", "ridge",
               "--no-figures", "-o", str(tmp_path / "out")])
    assert rc == 0
    assert os.path.exists(tmp_path / "out" / "state_importance.tsv")


# -- 4. benchmark: perturbation detection metric --------------------------------


def test_perturbation_metric_rewards_posteriors(simulated, fitted):
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    from toolbench.core import ToolRun
    from toolbench.metrics import METRICS, EvalContext

    ds, truth, seg = fitted
    run = ToolRun(tool="chromstate", display_name="ChromState", states=seg.states, posterior=seg.posterior,
                  n_states=seg.n_states)
    ctx = EvalContext(run=run, ds=ds, truth=truth)
    m = METRICS["perturbation"]
    assert m.applicable(ctx)
    out = m.fn(ctx)
    assert out["perturb_auprc"] > out["perturb_baseline_auprc"] * 2
    assert out["perturb_auroc"] > 0.8
    hard = ToolRun(tool="hard", display_name="hard", states=seg.states, n_states=seg.n_states)
    out2 = m.fn(EvalContext(run=hard, ds=ds, truth=truth))
    assert out2["perturb_auroc"] > 0.7
    assert out2["perturb_auprc"] <= out["perturb_auprc"] + 0.05


# -- 5. single cell -------------------------------------------------------------


def test_pseudobulk_and_dataset(simulated):
    ds, truth = simulated
    rng = np.random.default_rng(0)
    n_bins = int(ds.sample_mask("a1").sum())
    cells = rng.poisson(0.2, size=(300, n_bins)).astype(float)
    groups = np.array(["T"] * 150 + ["B"] * 150)
    counts, n_cells = sc.pseudobulk(cells, groups, min_cells=10)
    assert list(counts.index) == ["B", "T"]
    assert np.allclose(counts.loc["T"].to_numpy(), cells[:150].sum(axis=0))
    assert np.allclose(counts.loc["B"].to_numpy(), cells[150:].sum(axis=0))
    bins = ds.bins.loc[ds.sample_mask("a1"), ["chrom", "start", "end"]].reset_index(drop=True)
    pds = sc.dataset_from_pseudobulk({"ATAC": counts}, bins)
    assert pds.samples == ["B", "T"]
    assert pds.n_rows == 2 * n_bins
    assert len(pds.chains) == 2 * bins["chrom"].nunique()
    mc = sc.metacells(rng.normal(size=(300, 5)), 6, labels=groups, seed=0)
    assert len(set(mc)) == 6
    assert all(m.startswith(("T_", "B_")) for m in mc)


def test_state_transfer_recovers_active_states(fitted):
    ds, truth, seg = fitted
    means = {"TssA": 30.0, "EnhA": 18.0, "TssBiv": 10.0, "Tx": 3.0, "ReprPC": 1.0, "Het": 0.8, "Quies": 1.0}
    bulk_proxy = sc.simulate_proxy(truth, ds, means, seed=1)
    tr = sc.StateTransfer(neighbourhood=1, seed=0).fit(seg, bulk_proxy)
    assert tr.cv_report is not None
    assert tr.cv_report.attrs["accuracy"] > 0.4
    # a new "single-cell" proxy on the same bins with a different noise realisation
    sc_proxy = sc.simulate_proxy(truth, ds, means, seed=2)
    proj = tr.project(sc_proxy)
    assert proj.meta["transferred"]
    assert proj.posterior.shape == (sc_proxy.n_rows, seg.n_states)
    assert np.allclose(proj.posterior.sum(axis=1), 1.0)
    assert proj.labels == seg.labels
    # active states, which the proxy carries, are recovered; repressed ones are not expected to be
    tssa = seg.labels.index(next(lbl for lbl in seg.labels if "Tss" in lbl and "Biv" not in lbl))
    truth_active = seg.states == tssa
    called = proj.states == tssa
    recall = (truth_active & called).sum() / max(truth_active.sum(), 1)
    assert recall > 0.5
    # downstream functions run on the transferred segmentation
    summ = cs.state_summary(proj)
    assert set(summ["sample"]) == set(ds.samples)
    cmp = cs.compare_conditions(proj, GROUPS)
    assert not cmp.differences.empty
