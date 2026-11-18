"""Tests for state importance (states -> expression) and the algorithm benchmark."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from chromstate.benchmark.algorithms import compare_algorithms, speed_benchmark
from chromstate.features import gene_state_change, gene_state_matrix
from chromstate.importance import (
    _is_composition,
    compare_representations,
    state_importance,
    state_importance_change,
)
from chromstate.model import hmm as H
from chromstate.model.core import fit
from chromstate.model.emissions import build_emissions
from chromstate.simulate import simulate, simulate_genes_and_expression


@pytest.fixture(scope="module")
def sim():
    warnings.simplefilter("ignore")
    ds, truth = simulate(n_bins=4000, n_chroms=5, seed=3, blur=0.3)
    feats, expr, effects = simulate_genes_and_expression(ds, truth, n_genes=500, seed=1)
    seg = fit(ds, n_states=truth.n_states, n_restarts=2, adaptive_restarts=False)
    gs = gene_state_matrix(seg, feats, part="promoter", upstream=1000, downstream=1000)
    return ds, truth, feats, expr, effects, seg, gs


# ---------------------------------------------------------------------------
# Fused E-step and new emissions
# ---------------------------------------------------------------------------


def test_fused_e_step_matches_reference():
    ds, _ = simulate(n_bins=1500, n_chroms=3, seed=2)
    em = build_emissions("nb", ds.X, ds.size_factors, 5, np.random.default_rng(0))
    log_B = em.log_prob(ds.X, ds.size_factors)
    A = np.full((5, 5), 0.025)
    np.fill_diagonal(A, 0.9)
    pi = np.full(5, 0.2)
    chains = ds.chains[:-1]  # leave one chain uncovered on purpose
    g1, xi1, i1, ll1 = H.e_step_all_chains(log_B, A, pi, chains)
    g2 = np.zeros_like(g1)
    xi2 = np.zeros((5, 5))
    i2 = np.zeros(5)
    ll2 = 0.0
    for sl in chains:
        g, xi, ll = H.forward_backward(log_B[sl], A, pi)
        g2[sl] = g
        xi2 += xi
        i2 += g[0]
        ll2 += ll
    assert np.allclose(g1, g2, atol=1e-10)
    assert np.allclose(xi1, xi2, rtol=1e-8, atol=1e-8)
    assert np.allclose(i1, i2)
    assert abs(ll1 - ll2) < 1e-6
    # uncovered rows carry zero weight
    assert np.all(g1[ds.chains[-1]] == 0.0)


@pytest.mark.parametrize("kind", ["nb", "poisson", "zinb"])
def test_count_emissions_decompose_by_mark(kind):
    ds, _ = simulate(n_bins=800, n_chroms=2, seed=5)
    em = build_emissions(kind, ds.X, ds.size_factors, 4, np.random.default_rng(1))
    total = em.log_prob(ds.X, ds.size_factors)
    by_mark = np.stack(
        [em.log_prob_by_mark(ds.X, ds.size_factors, k).sum(axis=1) for k in range(4)], axis=1
    )
    assert np.allclose(total, by_mark, atol=1e-8)
    # M-step keeps parameters finite and in range
    gamma = np.random.default_rng(0).dirichlet(np.ones(4), size=ds.n_rows)
    em.m_step(ds.X, ds.size_factors, gamma)
    assert np.all(np.isfinite(em.mu)) and np.all(em.mu > 0)
    if kind == "zinb":
        assert np.all((em.zero > 0) & (em.zero < 1))


def test_compare_algorithms_ranks_nb_over_poisson():
    ds, truth = simulate(n_bins=3000, n_chroms=4, seed=7, blur=0.3)
    table = compare_algorithms(ds, n_states=truth.n_states, truth=truth,
                               emissions=("nb", "poisson"), n_restarts=1)
    t = table.set_index("emission")
    assert t.loc["nb", "heldout_loglik_per_bin"] > t.loc["poisson", "heldout_loglik_per_bin"]
    assert t.loc["nb", "ari"] > t.loc["poisson", "ari"]
    # Poisson fragments the segmentation
    assert t.loc["poisson", "mean_segment_len"] < t.loc["nb", "mean_segment_len"]


def test_speed_benchmark_runs():
    ds, _ = simulate(n_bins=2000, n_chroms=3, seed=1)
    table = speed_benchmark(ds, n_states=5, n_repeat=1)
    assert set(table["step"]) >= {"per-chain reference", "fused, chain-parallel"}
    assert (table["seconds"] > 0).all()


# ---------------------------------------------------------------------------
# State importance
# ---------------------------------------------------------------------------


def test_is_composition():
    comp = np.random.default_rng(0).dirichlet(np.ones(4), size=50)
    assert _is_composition(comp)
    assert not _is_composition(comp - comp.mean(axis=0))


def test_state_importance_recovers_planted_effects(sim):
    ds, truth, feats, expr, effects, seg, gs = sim
    one = gs.loc[gs["sample"] == ds.samples[0]]
    chrom = feats.table.set_index("gene_id")["chrom"]
    res = state_importance(one, expr, groups=chrom, model="ridge", n_permutations=5, n_boot=50)

    assert res.n_genes >= 400
    assert res.cv_spearman > 0.5
    table = res.table.set_index("state")
    # Fitted state labels come from annotate_states; match them to truth by emission
    # profile through the segmentation's label order versus the truth's effects.
    causal = set(effects.loc[effects["true_effect"] != 0, "state"])
    # map truth labels -> fitted labels via best-matching emission profile
    fitted_means = seg.model.emissions.mean_matrix()
    mapping = {}
    for i, lbl in enumerate(truth.labels):
        d = np.abs(np.log1p(fitted_means) - np.log1p(truth.emission_means[i])).sum(axis=1)
        mapping[lbl] = seg.labels[int(np.argmin(d))]
    causal_fitted = {mapping[s] for s in causal}
    null_fitted = {mapping[s] for s in truth.labels if s not in causal}
    top = set(res.table.head(len(causal_fitted))["state"])
    assert top == causal_fitted, (top, causal_fitted)
    assert table.loc[list(causal_fitted), "perm_importance"].min() > \
        table.loc[list(null_fitted), "perm_importance"].max()
    # signs
    up = mapping[effects.loc[effects["true_effect"] > 0, "state"].iloc[0]]
    assert table.loc[up, "direction"] == "up"
    # compositional drop-one is reported and positive for the strongest state
    assert table.loc[up, "drop_one_importance"] > 0


def test_state_importance_change_and_bad_inputs(sim):
    ds, truth, feats, expr, effects, seg, gs = sim
    change = gene_state_change(gs, [ds.samples[0]], [ds.samples[1]])
    rng = np.random.default_rng(0)
    # fake a fold change driven by the strongest state's delta
    strong = [c for c in change.columns if c.startswith("delta_")][0]
    rna = pd.DataFrame({"gene_id": change["gene_id"],
                        "log2fc": 2.0 * change[strong] + rng.normal(0, 0.1, len(change))})
    chrom = feats.table.set_index("gene_id")["chrom"]
    res = state_importance_change(change, rna, groups=chrom, n_permutations=3, n_boot=20)
    assert res.table.iloc[0]["state"] == strong.replace("delta_", "")
    assert np.isnan(res.table["drop_one_importance"]).all()  # no closure on deltas
    with pytest.raises(ValueError):
        state_importance_change(gs, rna)  # no delta columns
    with pytest.raises(ValueError):
        state_importance(gs.loc[gs["sample"] == ds.samples[0]],
                         pd.Series({"nope": 1.0, "nope2": 2.0}), groups=chrom)


def test_random_folds_warn_without_groups(sim):
    ds, truth, feats, expr, effects, seg, gs = sim
    one = gs.loc[gs["sample"] == ds.samples[0]]
    with pytest.warns(RuntimeWarning):
        state_importance(one, expr, model="ridge", n_permutations=2, n_boot=10)


def test_compare_representations(sim):
    ds, truth, feats, expr, effects, seg, gs = sim
    out = compare_representations(seg, feats, expr, sample=ds.samples[0], part="promoter",
                                  model="ridge", n_boot=50, upstream=1000, downstream=1000)
    assert set(out["representation"]) == {"quantitative", "hard_fractions", "tss_state", "raw_signal"}
    q = out.set_index("representation")
    assert q.loc["quantitative", "spearman"] > q.loc["tss_state", "spearman"]
    assert q.loc["tss_state", "diff_lo"] > 0


# ---------------------------------------------------------------------------
# New baselines standing in for EpiCSeg, EpiSegMix, diHMM and normR
# ---------------------------------------------------------------------------


def test_negative_multinomial_is_a_proper_likelihood_and_refuses_decomposition():
    from chromstate.model.emissions import NegativeMultinomialEmissions

    ds, _ = simulate(n_bins=600, n_chroms=2, seed=4)
    em = build_emissions("negative_multinomial", ds.X, ds.size_factors, 4, np.random.default_rng(0))
    lp = em.log_prob(ds.X, ds.size_factors)
    assert lp.shape == (ds.n_rows, 4) and np.all(np.isfinite(lp)) and np.all(lp <= 0)
    with pytest.raises(NotImplementedError):
        em.log_prob_by_mark(ds.X, ds.size_factors, 0)
    gamma = np.random.default_rng(1).dirichlet(np.ones(4), size=ds.n_rows)
    em.m_step(ds.X, ds.size_factors, gamma)
    assert isinstance(em, NegativeMultinomialEmissions)
    assert np.all(em.dispersion > 0) and np.all(np.isfinite(em.mu))
    # Likelihood of the exact mass function sums to <= 1 over a small count grid (1 mark).
    ds1 = simulate(n_bins=200, n_chroms=1, seed=4, marks=["H3K4me3"],
                   states={"a": (0.5,), "b": (8.0,)})[0]
    em1 = build_emissions("negative_multinomial", ds1.X, ds1.size_factors, 2, np.random.default_rng(0))
    grid = np.arange(0, 400)[:, None].astype(float)
    off = np.ones_like(grid)
    total = np.exp(em1.log_prob(grid, off)).sum(axis=0)
    assert np.all(total <= 1.0 + 1e-6) and np.all(total > 0.95)


def test_tied_emissions_fold_posterior():
    from chromstate.model.emissions import NegativeBinomialEmissions, TiedEmissions

    ds, _ = simulate(n_bins=500, n_chroms=2, seed=4)
    base = NegativeBinomialEmissions.initialize(ds.X, ds.size_factors, 3, np.random.default_rng(0))
    tied = TiedEmissions(base, 2)
    lp = tied.log_prob(ds.X, ds.size_factors)
    assert lp.shape == (ds.n_rows, 6)
    assert np.allclose(lp[:, :3], lp[:, 3:])
    gamma = np.random.default_rng(0).dirichlet(np.ones(6), size=ds.n_rows)
    bins, domains = tied.fold_posterior(gamma)
    assert bins.shape == (ds.n_rows, 3) and domains.shape == (ds.n_rows, 2)
    assert np.allclose(bins.sum(axis=1), 1.0) and np.allclose(domains.sum(axis=1), 1.0)


@pytest.mark.parametrize("key", ["epicseg_like", "episegmix_like", "dihmm_like", "normr_like"])
def test_new_baselines_produce_uniform_output(key):
    from chromstate.benchmark.baselines import METHODS

    ds, truth = simulate(n_bins=1500, n_chroms=3, seed=6)
    res = METHODS[key](ds, n_states=truth.n_states, seed=0, n_restarts=1)
    assert res.states.shape == (ds.n_rows,)
    assert res.posterior.shape == (ds.n_rows, truth.n_states)
    assert np.allclose(res.posterior.sum(axis=1), 1.0, atol=1e-6)
    assert res.bivalency_score.shape == (ds.n_rows,)
    assert res.runtime_s >= 0
    if key == "dihmm_like":
        assert res.meta["domain_posterior"].shape[0] == ds.n_rows


def test_run_comparison_includes_new_methods():
    from chromstate.benchmark.run import run_comparison

    ds, truth = simulate(n_bins=1500, n_chroms=3, seed=6, blur=0.3)
    table = run_comparison(ds, truth, n_restarts=1,
                           methods=("chromstate", "epicseg_like", "episegmix_like", "dihmm_like", "normr_like"))
    assert len(table) == 5
    t = table.set_index("method")
    # A segmenter with a chain must beat the no-chain enrichment caller on recovery.
    assert t.loc["ChromState (NB-HMM)", "ari"] > t.filter(like="normR", axis=0)["ari"].iloc[0]
