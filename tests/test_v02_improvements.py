"""Tests for the v0.2 follow-ups: ring names, the small-table importance model, lfc input,
scoring at the fitted width, and distal-gene alluvials."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import chromstate as cs
from chromstate.enhancers import MonotoneLinear, make_importance_model, ring_columns
from chromstate.pipeline import RunConfig, read_lfc, run_pipeline
from chromstate.regions import RegionScheme, call_elements, default_rings, enhancer_contributions
from chromstate.resolution import coarsen_dataset, copy_size_factors, decode_at_resolution
from chromstate.simulate import DEFAULT_STATES, simulate

GROUPS = {"ctrl": ["a1", "a2"], "drug": ["b1", "b2"]}


def test_ring_columns_keep_kb_names_and_never_collide():
    assert ring_columns((0, 10_000, 50_000)) == ["enh_0_10kb", "enh_10_50kb"]
    cols = ring_columns(default_rings(4_000))
    assert len(set(cols)) == len(cols) == 3
    assert cols[0] == "enh_0_400bp"
    with pytest.raises(ValueError):
        ring_columns((0, 100, 100, 1000))


def test_monotone_linear_respects_sign_constraints():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.1 * rng.normal(size=300)
    # column 0 is forced non-positive although its true effect is positive
    m = MonotoneLinear(monotonic_cst=[-1, 0, 1]).fit(X, y)
    assert m.coef_[0] <= 1e-9 and m.coef_[1] < 0
    free = MonotoneLinear(monotonic_cst=[0, 0, 0]).fit(X, y)
    assert free.coef_[0] == pytest.approx(2.0, abs=0.05)
    assert free.predict(X[:5]).shape == (5,)


def test_make_importance_model_switches_on_table_size():
    mono = [1, 1, 0]
    assert make_importance_model("auto", mono, 50)[1] == "linear"
    assert make_importance_model("auto", mono, 500)[1] == "hgb"
    assert make_importance_model("hgb", mono, 50)[1] == "hgb"
    with pytest.raises(ValueError):
        make_importance_model("forest", mono, 50)


def test_read_lfc_accepts_two_column_and_deseq2_tables(tmp_path):
    two = tmp_path / "two.tsv"
    two.write_text("gene_id\tlog2fc\ng1\t1.5\ng2\t-0.5\ng2\t9\n")
    s = read_lfc(str(two))
    assert list(s.index) == ["g1", "g2"] and s["g2"] == -0.5
    de = tmp_path / "deseq.tsv"
    de.write_text("\tbaseMean\tlog2FoldChange\tpadj\ng1\t10\t2.0\t0.01\ng3\t5\tNA\t0.5\n")
    s = read_lfc(str(de))
    assert list(s.index) == ["g1"] and s["g1"] == 2.0


@pytest.fixture(scope="module")
def fine_sim():
    scale = 10 / 200
    states = {k: tuple(v * scale for v in vals) for k, vals in DEFAULT_STATES.items()}
    return simulate(n_bins=6000, n_chroms=3, samples=("a1", "a2", "b1", "b2"), treated=("b1", "b2"),
                    states=states, self_transition=1 - 0.08 / 20, dispersion=0.15, blur=0.1,
                    perturbation={"responder_fraction": 0.6, "from_state": "ReprPC", "to_state": "TssA"},
                    seed=7, bin_size=10)


def _features(n_per_chrom=8, step=6_000, length=2_000, first=4_000):
    rows = []
    for c in range(3):
        for i in range(n_per_chrom):
            start = first + i * step
            rows.append({"gene_id": f"c{c}g{i}", "name": f"C{c}G{i}", "chrom": f"chr{c + 1}",
                         "start": start, "end": start + length, "strand": "+" if i % 2 else "-"})
    return cs.FeatureSet(pd.DataFrame(rows))


def _expression(fine, truth, features, seed=0):
    rng = np.random.default_rng(seed)
    expr = {}
    for s in fine.samples:
        st = truth.states_for(s)
        bins = fine.bins.loc[fine.sample_mask(s)].reset_index(drop=True)
        col = {}
        for g in features.table.itertuples():
            m = ((bins["chrom"] == g.chrom) & (bins["start"] >= g.start - 1000)
                 & (bins["start"] < g.start + 1000)).to_numpy()
            frac = float((st[m] == truth.labels.index("TssA")).mean()) if m.any() else 0.0
            col[g.gene_id] = 4.0 * frac + rng.normal(0, 0.3)
        expr[s] = col
    out = pd.DataFrame(expr)
    out.index.name = "gene_id"
    return out


def test_enhancer_contributions_small_table_uses_linear_model_and_varies(fine_sim):
    fine, truth = fine_sim
    coarse = coarsen_dataset(fine, 20)
    cs.normalize(coarse)
    copy_size_factors(coarse, fine)
    seg = cs.fit(coarse, n_states=truth.n_states, n_restarts=1, seed=0)
    fs = decode_at_resolution(seg, fine)
    f = _features()
    scheme = RegionScheme(regulatory_flank=3_000)
    el = call_elements(fs, f, scheme, min_score=40, min_width=100)
    res = enhancer_contributions(fs, f, _expression(fine, truth, f), elements=el, scheme=scheme, seed=0)
    assert res["model_kind"] == "linear"
    assert "linear" in res["note"]
    imp = res["importance"]
    assert len(imp) > 0 and (imp["importance"] >= -1e-9).all()
    # a sign-constrained linear model moves with the element's own input: within a ring the
    # importances differ unless the ring coefficient is exactly zero
    assert imp["importance"].nunique() > 1 or np.allclose(imp["importance"], 0)
    forced = enhancer_contributions(fs, f, _expression(fine, truth, f), elements=el, scheme=scheme,
                                    seed=0, model="hgb")
    assert forced["model_kind"] == "hgb"


def test_pipeline_scores_at_the_fitted_width(tmp_path):
    ds, truth = simulate(n_bins=1500, n_chroms=2, samples=("a1", "a2", "b1", "b2"), treated=("b1", "b2"),
                         seed=3, blur=0.1)
    cfg = RunConfig(mode="genome", resolution=200, fit_bin_size=200, n_states=truth.n_states,
                    n_restarts=1, max_restarts=1, figures=False, verbose=False, write_scores=True)
    res = run_pipeline(ds, str(tmp_path / "r200"), cfg, groups=GROUPS)
    assert res.fine.meta["bin_size"] == 200 and res.fine.meta["fit_bin_size"] == 200
    assert len(res.fine.states) == ds.n_rows
    scores = pd.read_csv(tmp_path / "r200" / "scores" / "state_scores.tsv.gz", sep="\t")
    score_cols = [c for c in scores.columns if c.startswith("score_")]
    assert len(score_cols) == truth.n_states
    # each state is rounded on its own, so a row may miss 100 by up to about half a point per state
    assert np.abs(scores[score_cols].sum(axis=1) - 100).max() <= truth.n_states // 2 + 1
    assert "Resolution 200 bp" in res.report


def test_pipeline_writes_distal_gene_patterns_and_alluvial(tmp_path, fine_sim):
    fine, truth = fine_sim
    f = _features()
    cfg = RunConfig(mode="gene", resolution=10, fit_bin_size=200, n_states=truth.n_states,
                    scheme=RegionScheme(regulatory_flank=3_000), n_restarts=1, max_restarts=1,
                    verbose=False, min_element_score=40)
    out = str(tmp_path / "g")
    run_pipeline(fine, out, cfg, features=f, groups=GROUPS)
    for rel in ("genes/distal_dominant_states.tsv", "genes/distal_pattern_counts.tsv",
                "figures/gene_alluvial.pdf", "figures/gene_alluvial_distal.pdf"):
        assert os.path.exists(os.path.join(out, rel)), rel
