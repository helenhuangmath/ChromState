"""Tests for the quantification and dynamics layers."""

from __future__ import annotations

import numpy as np
import pytest

import chromstate as cs
from chromstate.benchmark.metrics import detection_scores
from chromstate.dynamics import benjamini_hochberg, sinkhorn, state_distance_matrix
from chromstate.quantify import (
    attribution,
    bivalency,
    effective_states,
    margin,
    posterior_bivalency,
    state_confidence,
    state_entropy,
)
from chromstate.simulate import simulate


@pytest.fixture(scope="module")
def fitted():
    ds, truth = simulate(
        n_bins=1500, n_chroms=2, seed=21,
        perturbation={"responder_fraction": 0.5, "from_state": "TssBiv", "to_state": "TssA"},
    )
    cs.normalize(ds)
    seg = cs.fit(ds, n_states=truth.n_states, n_restarts=2, seed=0)
    return ds, truth, seg


# -- composition -------------------------------------------------------------


def test_entropy_bounds_and_extremes(fitted):
    _, _, seg = fitted
    h = state_entropy(seg, normalize=True)
    assert h.min() >= -1e-9 and h.max() <= 1.0 + 1e-9

    K = seg.n_states
    uniform = np.full((1, K), 1.0 / K)
    onehot = np.zeros((1, K))
    onehot[0, 0] = 1.0
    from dataclasses import replace

    assert state_entropy(replace(seg, posterior=uniform))[0] == pytest.approx(1.0)
    assert state_entropy(replace(seg, posterior=onehot))[0] == pytest.approx(0.0, abs=1e-9)


def test_confidence_margin_and_effective_states_are_consistent(fitted):
    _, _, seg = fitted
    conf, marg, eff = state_confidence(seg), margin(seg), effective_states(seg)
    assert (conf >= 1.0 / seg.n_states - 1e-9).all()
    assert (marg >= -1e-9).all() and (marg <= conf + 1e-9).all()
    assert (eff >= 1.0 - 1e-6).all() and (eff <= seg.n_states + 1e-6).all()


def test_entropy_is_higher_where_the_model_is_wrong(fitted):
    """Entropy must carry information: misclassified bins should be less certain."""
    _, truth, seg = fitted
    from chromstate.benchmark.metrics import match_states

    mapping, _ = match_states(truth.states, seg.states, truth.n_states, seg.n_states)
    mapped = np.array([mapping[int(s)] for s in seg.states])
    wrong = mapped != truth.states
    h = state_entropy(seg)
    assert wrong.sum() > 10
    assert h[wrong].mean() > h[~wrong].mean()


# -- bivalency ---------------------------------------------------------------


def test_bivalency_score_is_the_harmonic_mean(fitted):
    ds, _, seg = fitted
    biv = bivalency(seg)
    a = biv["signal_H3K4me3"].to_numpy()
    r = biv["signal_H3K27me3"].to_numpy()
    expected = 2 * a * r / np.maximum(a + r, 1e-12)
    np.testing.assert_allclose(biv["bivalency_score"], expected, rtol=1e-10)
    # score == magnitude * balance, exactly
    np.testing.assert_allclose(
        biv["bivalency_score"], biv["bivalency_magnitude"] * biv["bivalency_balance"], rtol=1e-9
    )


def test_bivalency_is_zero_when_a_mark_is_absent(fitted):
    _, _, seg = fitted
    biv = bivalency(seg)
    absent = biv["signal_H3K27me3"] < 1e-9
    if absent.any():
        assert biv.loc[absent, "bivalency_score"].abs().max() < 1e-9


def test_bivalency_detects_true_bivalent_regions(fitted):
    _, truth, seg = fitted
    biv = bivalency(seg)
    det = detection_scores(biv["bivalency_score"].to_numpy(), truth.bivalent_mask)
    assert det["auprc"] > 5 * det["baseline_auprc"]
    assert det["auroc"] > 0.9


def test_bivalency_requires_both_marks(fitted):
    _, _, seg = fitted
    with pytest.raises(ValueError, match="needs both"):
        bivalency(seg, active_mark="H3K4me3", repressive_mark="H3K99me9")


def test_posterior_bivalency_separates_within_from_between_state(fitted):
    _, truth, seg = fitted
    pb = posterior_bivalency(seg)
    assert {"biv_within_state", "biv_between_state", "mixture_ambiguity"} <= set(pb.columns)
    # Truly bivalent bins score high on within-state, which is the unconfounded quantity.
    within = pb["biv_within_state"].to_numpy()
    assert within[truth.bivalent_mask].mean() > 3 * within[~truth.bivalent_mask].mean()
    assert (pb["mixture_ambiguity"] >= -1e-9).all() and (pb["mixture_ambiguity"] <= 1 + 1e-9).all()


# -- attribution -------------------------------------------------------------


def test_attribution_sums_to_the_total_llr(fitted):
    _, _, seg = fitted
    attr = attribution(seg, max_rows=500)
    mark_cols = [c for c in attr.columns if c.startswith("llr_")]
    np.testing.assert_allclose(attr[mark_cols].sum(axis=1), attr["total_llr"], rtol=1e-9)


def test_attribution_against_background_credits_the_defining_mark(fitted):
    """Measured against quiescent background, each state's namesake mark must dominate."""
    _, _, seg = fitted
    quies = next(i for i, lbl in enumerate(seg.labels) if lbl.startswith("Quies"))
    imp = cs.state_mark_importance(seg, against=quies)

    # Active promoters are defined by H3K4me3 *and* H3K27ac together, so either is a
    # correct top mark; the repressive and elongation states have a single namesake.
    expected = {
        "ReprPC": {"H3K27me3"},
        "Tx": {"H3K36me3"},
        "Het": {"H3K9me3"},
        "TssA": {"H3K4me3", "H3K27ac"},
    }
    checked = 0
    for label, marks in expected.items():
        rows = [i for i in imp.index if i.split("_")[0] == label]
        if not rows:
            continue
        top = imp.loc[rows[0]].idxmax()
        assert top in marks, f"{rows[0]} top mark was {top}, expected one of {sorted(marks)}"
        checked += 1
    assert checked >= 3, "too few states matched to make this a real test"


def test_attribution_against_runner_up_finds_the_discriminating_mark(fitted):
    """Against the runner-up the answer differs, and that is the intended semantics.

    A Polycomb state's nearest competitor is usually the bivalent state; both carry
    H3K27me3, so what separates them is the *absence* of H3K4me3, not the presence of
    H3K27me3. This asserts the two comparisons genuinely give different answers, so the
    `against` parameter is doing real work rather than being decorative.
    """
    _, _, seg = fitted
    quies = next(i for i, lbl in enumerate(seg.labels) if lbl.startswith("Quies"))
    vs_runner = cs.state_mark_importance(seg)
    vs_background = cs.state_mark_importance(seg, against=quies)

    shared = vs_runner.index.intersection(vs_background.index)
    top_runner = vs_runner.loc[shared].idxmax(axis=1)
    top_background = vs_background.loc[shared].idxmax(axis=1)
    assert (top_runner != top_background).any()


def test_attribution_against_a_fixed_state(fitted):
    _, _, seg = fitted
    attr = attribution(seg, against=0, max_rows=200)
    assert (attr["against_state"] == seg.labels[0]).all()


# -- dynamics ----------------------------------------------------------------


def test_state_flow_rows_are_conditional_distributions(fitted):
    ds, _, seg = fitted
    flow = cs.state_flow(seg, "control", "treated", normalize=True)
    np.testing.assert_allclose(flow.sum(axis=1), 1.0, atol=1e-9)


def test_state_flow_joint_marginals_match_occupancy(fitted):
    ds, _, seg = fitted
    joint = cs.state_flow(seg, "control", "treated", normalize=False)
    occ = seg.sample_state_matrix(soft=True)
    np.testing.assert_allclose(joint.sum(axis=1).to_numpy(), occ.loc["control"].to_numpy(), atol=1e-9)
    np.testing.assert_allclose(joint.sum(axis=0).to_numpy(), occ.loc["treated"].to_numpy(), atol=1e-9)


def test_flow_captures_the_simulated_switch(fitted):
    """Half the bivalent bins were moved to active promoters; the flow must show it."""
    _, _, seg = fitted
    flow = cs.state_flow(seg, "control", "treated")
    if "TssBiv" in flow.index and "TssA" in flow.columns:
        assert flow.loc["TssBiv", "TssA"] > 0.3
        assert flow.loc["TssBiv", "TssA"] < 0.7


def test_transition_propensity_finds_responders(fitted):
    _, truth, seg = fitted
    prop = cs.transition_propensity(seg, "control", "treated")
    responders = truth.responder_mask[: len(prop)]
    det = detection_scores(prop["tv_distance"].to_numpy(), responders)
    assert det["auprc"] > 0.7
    assert (prop["tv_distance"] >= -1e-9).all() and (prop["tv_distance"] <= 1 + 1e-9).all()
    assert (prop["jsd"] >= -1e-9).all() and (prop["jsd"] <= 1 + 1e-9).all()


def test_differential_state_refuses_pvalues_without_replicates(fitted):
    _, _, seg = fitted
    res = cs.differential_state(seg, ["control"], ["treated"])
    assert res.has_pvalues is False
    assert "pvalue" not in res.table.columns
    assert "2 replicates" in res.note


def test_differential_state_with_replicates_produces_qvalues():
    ds, truth = simulate(
        n_bins=800, n_chroms=1, samples=("c1", "c2", "t1", "t2"), seed=22,
        perturbation={"responder_fraction": 0.5, "from_state": "TssBiv", "to_state": "TssA"},
    )
    cs.normalize(ds)
    seg = cs.fit(ds, n_states=truth.n_states, n_restarts=2, seed=0)
    res = cs.differential_state(seg, ["c1", "c2"], ["t1", "t2"], n_permutations=50)
    assert res.has_pvalues
    assert res.table["pvalue"].between(0, 1).all()
    assert res.table["qvalue"].between(0, 1).all()
    assert (res.table["qvalue"] >= res.table["pvalue"] - 1e-12).all()


def test_permutation_pvalue_floor_reflects_distinct_splits_not_permutations():
    """A 2-vs-2 design can resolve p no lower than 1/3, and must say so.

    Counting permutations rather than group assignments inflates the resolution by
    ``n_a! * n_b!`` -- it reports a floor of 1/21 for a design whose honest floor is 1/3,
    which is the difference between "this cannot be tested" and a bin appearing to clear
    FDR.
    """
    ds, truth = simulate(
        n_bins=400, n_chroms=1, samples=("c1", "c2", "t1", "t2"), seed=23,
        perturbation={"responder_fraction": 0.5, "from_state": "TssBiv", "to_state": "TssA"},
    )
    cs.normalize(ds)
    seg = cs.fit(ds, n_states=truth.n_states, n_restarts=1, seed=0)
    res = cs.differential_state(seg, ["c1", "c2"], ["t1", "t2"], n_permutations=200)

    # Three unordered 2-vs-2 splits exist; one is the observed labelling.
    assert res.n_permutations == 2
    assert res.table["pvalue"].min() == pytest.approx(1 / 3)
    assert not res.table["significant"].any()
    assert "0.333" in res.note
    assert "no bin can be called significant" in res.note
    assert "groups of 4" in res.note


def test_benjamini_hochberg_is_monotone_and_bounded():
    p = np.array([0.001, 0.01, 0.02, 0.3, 0.9])
    q = benjamini_hochberg(p)
    assert (q >= p - 1e-12).all()
    assert (q <= 1.0).all()
    assert (np.diff(q) >= -1e-12).all()


def test_sinkhorn_respects_marginals():
    rng = np.random.default_rng(0)
    a = rng.dirichlet(np.ones(5))
    b = rng.dirichlet(np.ones(5))
    cost = rng.uniform(0, 1, size=(5, 5))
    plan = sinkhorn(a, b, cost, reg=0.05, n_iter=5000)
    np.testing.assert_allclose(plan.sum(axis=1), a, atol=1e-4)
    np.testing.assert_allclose(plan.sum(axis=0), b, atol=1e-4)


def test_state_distance_matrix_is_a_metric_shape(fitted):
    _, _, seg = fitted
    D = state_distance_matrix(seg)
    assert D.shape == (seg.n_states, seg.n_states)
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-9)
    np.testing.assert_allclose(D, D.T, atol=1e-9)


def test_paired_comparison_rejects_mismatched_binning(fitted):
    ds, _, seg = fitted
    from dataclasses import replace

    trimmed = ds.subset_rows(np.r_[np.ones(ds.n_rows - 5, bool), np.zeros(5, bool)])
    bad = replace(seg, dataset=trimmed, posterior=seg.posterior[: trimmed.n_rows],
                  states=seg.states[: trimmed.n_rows])
    with pytest.raises(ValueError, match="identical binning|same order"):
        cs.state_flow(bad, "control", "treated")


def test_quantify_produces_one_row_per_bin(fitted):
    ds, _, seg = fitted
    table = cs.quantify(seg)
    assert len(table) == ds.n_rows
    assert {"state", "confidence", "entropy", "bivalency_score"} <= set(table.columns)
    post_cols = [c for c in table.columns if c.startswith("post_")]
    assert len(post_cols) == seg.n_states
    np.testing.assert_allclose(table[post_cols].sum(axis=1), 1.0, atol=1e-9)
