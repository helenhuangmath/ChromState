"""Tests for state summaries, condition comparison, gene patterns, alluvials and the pipeline."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import chromstate as cs
from chromstate import viz
from chromstate.patterns import (
    categorise,
    default_state_classes,
    flow_table,
    gene_dominant_states,
    pattern_counts,
    region_patterns,
)
from chromstate.pipeline import RunConfig, parse_groups, run_pipeline
from chromstate.regions import RegionScheme
from chromstate.simulate import DEFAULT_STATES, simulate
from chromstate.summary import compare_conditions, condition_summary, flow_bp, state_summary

GROUPS = {"ctrl": ["a1", "a2"], "drug": ["b1", "b2"]}


@pytest.fixture(scope="module")
def fitted():
    ds, truth = simulate(
        n_bins=2500, n_chroms=3, samples=("a1", "a2", "b1", "b2"), treated=("b1", "b2"),
        perturbation={"responder_fraction": 0.7, "from_state": "ReprPC", "to_state": "TssA"},
        seed=11, blur=0.1,
    )
    cs.normalize(ds)
    seg = cs.fit(ds, n_states=truth.n_states, n_restarts=2, seed=0)
    return ds, truth, seg


def _features(n_per_chrom=12, step=40_000, length=8_000, first=5_000):
    rows = []
    for c in range(3):
        for i in range(n_per_chrom):
            start = first + i * step
            rows.append({"gene_id": f"c{c}g{i}", "name": f"C{c}G{i}", "chrom": f"chr{c + 1}",
                         "start": start, "end": start + length, "strand": "+" if i % 2 else "-"})
    return cs.FeatureSet(pd.DataFrame(rows))


# -- summaries ----------------------------------------------------------------


def test_state_summary_percentages_sum_to_100(fitted):
    ds, _, seg = fitted
    summ = state_summary(seg)
    for s in ds.samples:
        sub = summ.loc[summ["sample"] == s]
        assert sub["pct_hard"].sum() == pytest.approx(100.0)
        assert sub["pct_soft"].sum() == pytest.approx(100.0, abs=1e-6)
        assert sub["bp_hard"].sum() == pytest.approx(summ.attrs["total_bp"][s])
        assert sub["n_segments"].sum() >= 3
    assert {"strength", "strength_sd", "mean_score", "confidence", "median_segment_bp"} <= set(summ.columns)
    assert summ["confidence"].dropna().between(0, 1).all()
    cond = condition_summary(summ, GROUPS)
    assert set(cond["condition"]) == set(GROUPS)
    assert "pct_soft_sd" in cond.columns and cond["pct_soft_sd"].notna().all()


def test_compare_conditions_detects_the_perturbation(fitted):
    ds, truth, seg = fitted
    cmp = compare_conditions(seg, GROUPS)
    assert cmp.reference == "ctrl"
    d = cmp.differences.set_index("state")
    assert np.isclose(d["delta_pct_soft"].sum(), 0.0, atol=1e-6)
    # active promoter mass goes up, Polycomb mass down, in the treated group
    active = [lbl for lbl in seg.labels if lbl.startswith("TssA")]
    polycomb = [lbl for lbl in seg.labels if lbl.startswith("ReprPC")]
    assert active and polycomb
    assert d.loc[active, "delta_pct_soft"].sum() > 0
    assert d.loc[polycomb, "delta_pct_soft"].sum() < 0
    assert d["z"].notna().all()
    flow = cmp.flow("ctrl", "drug")
    total = summ_total = cmp.per_sample.attrs["total_bp"]["a1"]
    assert flow.to_numpy().sum() == pytest.approx(total, rel=1e-6)
    assert flow_bp(seg, ["a1"], ["b1"]).to_numpy().sum() == pytest.approx(summ_total, rel=1e-6)
    assert "screening" in cmp.note


def test_summary_figures_render(fitted):
    ds, _, seg = fitted
    cmp = compare_conditions(seg, GROUPS)
    figs = {
        "pct": viz.plot_state_percentages(cmp.per_condition, seg),
        "strength": viz.plot_state_strength(cmp.per_condition, seg),
        "diff": viz.plot_condition_differences(cmp.differences, seg),
    }
    for f in figs.values():
        assert f is not None
    import matplotlib.pyplot as plt

    plt.close("all")


# -- patterns and alluvials ----------------------------------------------------


def test_categorise_rules():
    assert categorise(["active", "active"]) == "stable"
    assert categorise(["repressed", "active"]) == "activated"
    assert categorise(["active", "repressed"]) == "deactivated"
    assert categorise(["poised", "active"]) == "poised_to_active"
    assert categorise(["active", "poised", "active"]) == "transient"
    assert categorise(["background", "poised"]) == "poised"


def test_gene_dominant_states_and_flow(fitted):
    ds, _, seg = fitted
    f = _features()
    gq = cs.gene_regulatory_quantification(seg, f, RegionScheme(regulatory_flank=10_000))
    classes = default_state_classes(seg)
    dom = gene_dominant_states(gq, GROUPS, part="P", state_classes=classes)
    assert {"dom_ctrl", "dom_drug", "frac_ctrl", "pattern", "n_changes", "category"} <= set(dom.columns)
    assert len(dom) == len(f)
    assert dom["frac_ctrl"].between(0, 1).all()
    assert (dom["pattern"] == dom["dom_ctrl"] + ">" + dom["dom_drug"]).all()
    counts = pattern_counts(dom)
    assert counts["n_genes"].sum() == len(dom)
    flows = flow_table(dom, weight=None)
    assert flows["weight"].sum() == len(dom)
    assert flows.attrs["stages"] == ["ctrl", "drug"]
    cls_flows = flow_table(dom, by="class")
    assert set(cls_flows["ctrl"]) <= {"active", "poised", "repressed", "background"}

    reg = region_patterns(seg, GROUPS, state_classes=classes)
    rflows = flow_table(reg)
    assert rflows["weight"].sum() == pytest.approx(reg["bp"].sum())

    figs = [viz.plot_alluvial(flows, seg=seg), viz.plot_gene_alluvial(dom, seg, by="class"),
            viz.plot_region_alluvial(seg, GROUPS), viz.plot_pattern_counts(counts)]
    assert all(f is not None for f in figs)
    import matplotlib.pyplot as plt

    plt.close("all")


def test_alluvial_three_stages():
    rng = np.random.default_rng(0)
    tab = pd.DataFrame({"t0": rng.choice(["TssA", "ReprPC", "Quies"], 300),
                        "t1": rng.choice(["TssA", "ReprPC", "Quies"], 300),
                        "t2": rng.choice(["TssA", "ReprPC"], 300)})
    flows = tab.groupby(["t0", "t1", "t2"]).size().rename("weight").reset_index()
    fig = viz.plot_alluvial(flows, stages=["t0", "t1", "t2"])
    assert fig is not None
    with pytest.raises(ValueError):
        viz.plot_alluvial(flows.assign(weight=0.0), stages=["t0", "t1"])
    import matplotlib.pyplot as plt

    plt.close("all")


# -- pipeline -------------------------------------------------------------------


def test_parse_groups(tmp_path):
    assert parse_groups("A=a1,a2;B=b1") == {"A": ["a1", "a2"], "B": ["b1"]}
    p = tmp_path / "g.tsv"
    p.write_text("a1\tA\na2\tA\nb1\tB\n")
    assert parse_groups(str(p)) == {"A": ["a1", "a2"], "B": ["b1"]}
    assert parse_groups(None) is None


def _fine_sim(seed=3):
    scale = 10 / 200
    states = {k: tuple(v * scale for v in vals) for k, vals in DEFAULT_STATES.items()}
    return simulate(n_bins=5000, n_chroms=3, samples=("a1", "a2", "b1", "b2"), treated=("b1", "b2"),
                    states=states, self_transition=1 - 0.08 / 20, dispersion=0.15,
                    perturbation={"responder_fraction": 0.6, "from_state": "ReprPC", "to_state": "TssA"},
                    seed=seed, bin_size=10)


def test_pipeline_gene_mode_end_to_end(tmp_path):
    fine, truth = _fine_sim()
    f = _features(n_per_chrom=8, step=6_000, length=2_000)
    # expression that follows the active-promoter truth, so the importance step has signal
    rng = np.random.default_rng(0)
    expr = {}
    for s in fine.samples:
        st = truth.states_for(s)
        bins = fine.bins.loc[fine.sample_mask(s)].reset_index(drop=True)
        col = {}
        for g in f.table.itertuples():
            m = (bins["chrom"] == g.chrom) & (bins["start"] >= g.start - 1000) & (bins["start"] < g.start + 1000)
            frac = float((st[m.to_numpy()] == truth.labels.index("TssA")).mean()) if m.any() else 0.0
            col[g.gene_id] = 4.0 * frac + rng.normal(0, 0.3)
        expr[s] = col
    expression = pd.DataFrame(expr)
    expression.index.name = "gene_id"

    cfg = RunConfig(mode="gene", resolution=10, fit_bin_size=200, n_states=5, n_fit_states=7,
                    scheme=RegionScheme(regulatory_flank=3_000), n_restarts=1, max_restarts=1,
                    tracks=True, verbose=False, min_element_score=40)
    res = run_pipeline(fine, str(tmp_path / "run"), cfg, features=f, groups=GROUPS, expression=expression)
    out = str(tmp_path / "run")
    assert res.fine.n_states == 5 and res.coarse.meta["collapsed_from"] == 7
    for rel in ("REPORT.md", "model/emissions.tsv", "model/collapsed_members.tsv",
                "scores/state_scores.tsv.gz", "scores/a1_segments.bed",
                "summary/state_summary_per_sample.tsv", "summary/state_summary_per_condition.tsv",
                "summary/condition_differences.tsv", "summary/flow_bp_ctrl_to_drug.tsv",
                "genes/gene_quantification.tsv.gz", "genes/promoter_dominant_states.tsv",
                "genes/promoter_pattern_counts.tsv", "genes/candidate_elements.tsv",
                "figures/state_percentages.pdf", "figures/gene_alluvial.pdf",
                "figures/region_alluvial.pdf"):
        assert os.path.exists(os.path.join(out, rel)), rel
    tracks = os.listdir(os.path.join(out, "scores", "tracks"))
    assert len(tracks) == 5 * 4
    assert res.gene_table is not None and "E_n_bins" in res.gene_table.columns
    assert res.dominant is not None and "category" in res.dominant.columns
    assert "Resolution 10 bp" in res.report
    # expression steps ran (they are allowed to be skipped with a logged reason only when
    # data are too small; here they must have produced their tables)
    assert os.path.exists(os.path.join(out, "expression", "state_importance.tsv"))
    assert os.path.exists(os.path.join(out, "expression", "part_ranking.tsv"))
    parts = pd.read_csv(os.path.join(out, "expression", "part_ranking.tsv"), sep="\t")
    assert {"P", "E", "P+E"} <= set(parts["parts"])
    for rel in ("summary/state_tests.tsv", "differential/promoters_ctrl_vs_drug.tsv",
                "expression/state_rna_association_ctrl_vs_drug.tsv",
                "figures/state_tests.pdf", "figures/volcano_promoters_ctrl_vs_drug.pdf",
                "figures/expression_link_ctrl_vs_drug.pdf", "figures/state_importance.pdf",
                "figures/part_ranking.pdf"):
        assert os.path.exists(os.path.join(out, rel)), rel


def test_pipeline_genome_mode_without_genes(tmp_path):
    fine, truth = _fine_sim(seed=4)
    cfg = RunConfig(mode="genome", resolution=10, fit_bin_size=200, n_states=truth.n_states,
                    n_restarts=1, max_restarts=1, figures=False, verbose=False)
    assert cfg.write_scores is False
    res = run_pipeline(fine, str(tmp_path / "g"), cfg, groups=GROUPS)
    assert not os.path.exists(tmp_path / "g" / "scores" / "state_scores.tsv.gz")
    assert os.path.exists(tmp_path / "g" / "summary" / "condition_differences.tsv")
    assert res.gene_table is None
    with pytest.raises(ValueError):
        RunConfig(resolution=30, fit_bin_size=200)
    with pytest.raises(ValueError):
        RunConfig(n_states=10, n_fit_states=5)


def test_fine_binset_gene_mode_tiles_only_windows():
    from chromstate.genome import Genome
    from chromstate.pipeline import fine_binset

    genome = Genome({"chr1": 1_000_000, "chr2": 1_000_000, "chr3": 1_000_000})
    f = _features(n_per_chrom=2, step=200_000, first=100_000)
    cfg = RunConfig(mode="gene", resolution=10, scheme=RegionScheme(regulatory_flank=20_000))
    bs = fine_binset(cfg, genome, f)
    assert len(bs) == 6 * (8_000 + 40_000) // 10
    assert len(bs.chain_slices) == 6
    whole = fine_binset(RunConfig(mode="genome", resolution=10), genome, None)
    assert len(whole) == 300_000


# -- statistics, tensor, smoothing, palette -------------------------------------


def test_state_tests_find_the_perturbed_states(fitted):
    ds, _, seg = fitted
    from chromstate.summary import flow_test, state_tests

    tab = state_tests(seg, GROUPS, block_bp=20_000, n_boot=300)
    assert {"delta_pct", "delta_lo", "delta_hi", "p_rep", "q_rep", "p_block", "q_block"} <= set(tab.columns)
    assert tab["p_rep"].notna().all()  # two replicates a side
    t = tab.set_index("state")
    active = [l for l in seg.labels if l.startswith("TssA")]
    polycomb = [l for l in seg.labels if l.startswith("ReprPC")]
    assert (t.loc[active, "delta_lo"] > 0).all()
    assert (t.loc[polycomb, "delta_hi"] < 0).all()
    assert (t["delta_lo"] <= t["delta_pct"]).all() and (t["delta_pct"] <= t["delta_hi"]).all()
    assert t.loc[polycomb, "q_block"].max() < 0.05
    ft = flow_test(seg, GROUPS["ctrl"], GROUPS["drug"], block_bp=20_000, n_boot=100)
    top = ft.iloc[0]
    assert top["from"].startswith("ReprPC") and top["to"].startswith("TssA")
    assert top["bp_lo"] <= top["bp"] <= top["bp_hi"]
    assert (ft["share_of_source"].between(0, 1)).all()


def test_score_tensor_matches_in_memory_scores(tmp_path):
    from chromstate.resolution import ScoreTensor, coarsen_dataset, copy_size_factors, decode_at_resolution, decode_to_tensor, state_scores

    fine, truth = _fine_sim(seed=9)
    coarse = coarsen_dataset(fine, 20)
    cs.normalize(coarse)
    copy_size_factors(coarse, fine)
    seg = cs.fit(coarse, n_states=truth.n_states, n_restarts=1, adaptive_restarts=False, seed=0)
    fine_seg = decode_at_resolution(seg, fine)
    t = decode_to_tensor(seg, fine, str(tmp_path / "tensor"))
    assert set(t.chroms) == set(pd.unique(fine.bins["chrom"]))
    assert t.nbytes == 4 * fine.n_rows // 4 * truth.n_states
    t2 = ScoreTensor.open(str(tmp_path / "tensor"))
    scores = state_scores(fine_seg)
    sub = scores.loc[(scores["sample"] == "b1") & (scores["chrom"] == "chr2")]
    block = t2.window("chr2", 1000, 3000, sample="b1")
    expect = sub.loc[(sub["start"] >= 1000) & (sub["start"] < 3000),
                     [f"score_{l}" for l in seg.labels]].to_numpy()
    assert block.shape == expect.shape
    assert np.abs(block.astype(int) - expect.astype(int)).max() <= 1
    # region means and the summary agree with the in-memory summary
    regions = pd.DataFrame({"gene_id": ["r1", "r2"], "chrom": ["chr1", "chr3"],
                            "start": [0, 20_000], "end": [2_000, 25_000]})
    rm = t2.region_means(regions)
    assert len(rm) == 2 * 4 and rm["n_bins"].tolist()[:4] == [200] * 4
    from chromstate.summary import state_summary

    s_mem = state_summary(fine_seg, include_strength=False).set_index(["sample", "state"])
    s_ten = t2.summary().set_index(["sample", "state"])
    assert np.allclose(s_mem["pct_soft"], s_ten.loc[s_mem.index, "pct_soft"], atol=0.5)
    assert np.allclose(s_mem["pct_hard"], s_ten.loc[s_mem.index, "pct_hard"], atol=0.5)


def test_run_genome_streaming_with_loader(tmp_path):
    from chromstate.pipeline import run_genome_streaming

    fine, truth = _fine_sim(seed=12)
    sizes = tmp_path / "sizes.tsv"
    sizes.write_text("chr1\t50000\nchr2\t50000\nchr3\t50000\n")

    def loader(chroms):
        return fine.subset_rows(fine.bins["chrom"].isin(chroms).to_numpy())

    cfg = RunConfig(mode="genome", n_states=truth.n_states, n_restarts=1, max_restarts=1,
                    figures=False, verbose=False)
    res = run_genome_streaming("unused", str(sizes), str(tmp_path / "stream"), cfg,
                               fit_chroms=["chr1"], groups=GROUPS, loader=loader)
    assert set(res["tensor"].chroms) == {"chr1", "chr2", "chr3"}
    assert res["summary"]["pct_soft"].sum() == pytest.approx(100.0 * 4, abs=1.0)
    assert os.path.exists(tmp_path / "stream" / "scores" / "tensor" / "chr3.u8")
    assert os.path.exists(tmp_path / "stream" / "summary" / "state_summary_per_condition.tsv")


def test_smooth_segments_removes_short_runs(fitted):
    ds, _, seg = fitted
    from chromstate.collapse import smooth_segments
    from chromstate.summary import state_summary

    before = state_summary(seg, include_strength=False)
    sm = smooth_segments(seg, min_bp=1000)
    after = state_summary(sm, include_strength=False)
    assert after["n_segments"].sum() < before["n_segments"].sum()
    assert after["median_segment_bp"].min() >= before["median_segment_bp"].min()
    assert sm.meta["smoothed_min_bp"] == 1000
    assert np.shares_memory(sm.posterior, seg.posterior)


def test_palettes_are_distinct_and_default_is_scirep2025(fitted):
    _, _, seg = fitted
    from chromstate.model.annotate import OKABE_ITO, STATE_COLORS_NATURE, STATE_COLORS_SCIREP2025

    cols = viz.state_colors(seg)
    assert cols == viz.state_colors(seg, "scirep2025")
    assert len(cols) == seg.n_states and len(set(cols)) == seg.n_states
    assert any(c in STATE_COLORS_SCIREP2025.values() for c in cols)
    nat = viz.state_colors(seg, "nature")
    assert len(set(nat)) == seg.n_states
    assert all(c in set(STATE_COLORS_NATURE.values()) | set(OKABE_ITO) or c.startswith("#") for c in nat)
    assert viz.state_colors(seg, "roadmap") == list(seg.colors)
