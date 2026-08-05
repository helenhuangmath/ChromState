"""Correctness tests for the HMM core and emission models.

The important tests here are parameter-recovery tests: data is generated from a known
model and the fitted parameters are checked against the values that produced it. A
segmentation that merely looks plausible proves nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import nbinom

from chromstate.model.core import fit, select_n_states
from chromstate.model.emissions import (
    BernoulliEmissions,
    GaussianEmissions,
    NegativeBinomialEmissions,
)
from chromstate.model.hmm import HMM, forward_backward, viterbi
from chromstate.simulate import simulate


# -- forward-backward --------------------------------------------------------


def test_forward_backward_matches_brute_force():
    """On a short chain, posteriors must equal exhaustive enumeration of all paths."""
    rng = np.random.default_rng(0)
    T, K = 6, 3
    A = rng.dirichlet(np.ones(K), size=K)
    pi = rng.dirichlet(np.ones(K))
    log_B = np.log(rng.uniform(0.05, 1.0, size=(T, K)))

    gamma, xi, loglik = forward_backward(log_B, A, pi)

    from itertools import product

    total = 0.0
    marg = np.zeros((T, K))
    for path in product(range(K), repeat=T):
        p = pi[path[0]] * np.exp(log_B[0, path[0]])
        for t in range(1, T):
            p *= A[path[t - 1], path[t]] * np.exp(log_B[t, path[t]])
        total += p
        for t, k in enumerate(path):
            marg[t, k] += p

    assert loglik == pytest.approx(np.log(total), rel=1e-10)
    np.testing.assert_allclose(gamma, marg / total, rtol=1e-9, atol=1e-12)


def test_forward_backward_posteriors_are_normalized():
    rng = np.random.default_rng(3)
    log_B = np.log(rng.uniform(0.01, 1.0, size=(500, 5)))
    A = rng.dirichlet(np.ones(5), size=5)
    pi = rng.dirichlet(np.ones(5))
    gamma, _, ll = forward_backward(log_B, A, pi)
    np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-10)
    assert np.isfinite(ll)


def test_forward_backward_survives_extreme_loglikelihoods():
    """Underflow guard: emissions of ~exp(-3000) must not produce nan or inf."""
    rng = np.random.default_rng(4)
    log_B = rng.uniform(-3000, -2900, size=(400, 4))
    A = np.full((4, 4), 0.25)
    pi = np.full(4, 0.25)
    gamma, xi, ll = forward_backward(log_B, A, pi)
    assert np.all(np.isfinite(gamma))
    assert np.isfinite(ll)
    np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-9)


def test_viterbi_finds_the_optimal_path():
    rng = np.random.default_rng(5)
    T, K = 8, 3
    A = rng.dirichlet(np.ones(K), size=K)
    pi = rng.dirichlet(np.ones(K))
    log_B = np.log(rng.uniform(0.05, 1.0, size=(T, K)))
    path = viterbi(log_B, A, pi)

    from itertools import product

    def score(p):
        s = np.log(pi[p[0]]) + log_B[0, p[0]]
        for t in range(1, T):
            s += np.log(A[p[t - 1], p[t]]) + log_B[t, p[t]]
        return s

    best = max(product(range(K), repeat=T), key=score)
    assert score(path) == pytest.approx(score(best), rel=1e-12)


def test_baum_welch_loglikelihood_is_monotone():
    """EM must never decrease the likelihood -- the canonical implementation check."""
    ds, _ = simulate(n_bins=600, n_chroms=1, samples=("s1",), seed=2)
    em = NegativeBinomialEmissions.initialize(ds.X, ds.size_factors, 5, np.random.default_rng(0))
    hmm = HMM(5, em)
    res = hmm.fit(ds.X, ds.size_factors, ds.chains, n_iter=25, tol=0.0)
    diffs = np.diff(res.history)
    assert (diffs > -1e-6).all(), f"log-likelihood decreased: min step {diffs.min()}"


def test_chains_are_inferred_independently():
    """Each chromosome's posterior must equal what it would be if run entirely alone."""
    ds, _ = simulate(n_bins=300, n_chroms=3, samples=("s1",), seed=6)
    assert len(ds.chains) == 3
    assert sum(sl.stop - sl.start for sl in ds.chains) == ds.n_rows

    seg = fit(ds, n_states=4, n_restarts=1, n_iter=20, seed=0)
    hmm = HMM(seg.n_states, seg.model.emissions, seg.model.transitions, seg.model.init_probs)
    joint = hmm.posterior(ds.X, ds.size_factors, ds.chains)

    for sl in ds.chains:
        alone = hmm.posterior(
            ds.X[sl], ds.size_factors[sl], [slice(0, sl.stop - sl.start)]
        )
        np.testing.assert_allclose(joint[sl], alone, atol=1e-10)


def test_joining_chains_changes_the_inference():
    """Sanity check on the above: chain boundaries must actually matter."""
    ds, _ = simulate(n_bins=300, n_chroms=3, samples=("s1",), seed=6)
    seg = fit(ds, n_states=4, n_restarts=1, n_iter=20, seed=0)
    hmm = HMM(seg.n_states, seg.model.emissions, seg.model.transitions, seg.model.init_probs)

    split = hmm.posterior(ds.X, ds.size_factors, ds.chains)
    joined = hmm.posterior(ds.X, ds.size_factors, [slice(0, ds.n_rows)])
    assert np.abs(split - joined).max() > 1e-6

    ll_split = hmm.score(ds.X, ds.size_factors, ds.chains)
    ll_joined = hmm.score(ds.X, ds.size_factors, [slice(0, ds.n_rows)])
    assert ll_split != pytest.approx(ll_joined)


# -- emission models ---------------------------------------------------------


def test_nb_log_prob_matches_scipy():
    rng = np.random.default_rng(7)
    mu = np.array([[3.0, 20.0], [0.5, 1.0]])
    disp = np.array([2.0, 5.0])
    em = NegativeBinomialEmissions(mu, disp)
    X = rng.poisson(5, size=(50, 2)).astype(float)
    offsets = np.full_like(X, 1.3)

    got = em.log_prob(X, offsets)
    for k in range(2):
        mean = offsets * mu[k]
        p = disp / (disp + mean)
        expect = nbinom.logpmf(X, disp, p).sum(axis=1)
        np.testing.assert_allclose(got[:, k], expect, rtol=1e-9)


def test_log_prob_by_mark_sums_to_log_prob():
    """Attribution is only valid if the per-mark decomposition is exact."""
    ds, _ = simulate(n_bins=200, n_chroms=1, samples=("s1",), seed=8)
    for cls in (NegativeBinomialEmissions, GaussianEmissions, BernoulliEmissions):
        em = cls.initialize(ds.X, ds.size_factors, 4, np.random.default_rng(0))
        total = em.log_prob(ds.X, ds.size_factors)
        for k in range(4):
            per_mark = em.log_prob_by_mark(ds.X, ds.size_factors, k)
            np.testing.assert_allclose(per_mark.sum(axis=1), total[:, k], rtol=1e-8,
                                       err_msg=f"{cls.__name__} state {k}")


def test_nb_m_step_recovers_known_means():
    """With known responsibilities, the M-step must return the generating means."""
    rng = np.random.default_rng(9)
    true_mu = np.array([[2.0, 30.0], [25.0, 1.5]])
    n = 20000
    labels = rng.integers(0, 2, size=n)
    disp = np.array([50.0, 50.0])
    mean = true_mu[labels]
    X = rng.poisson(mean).astype(float)
    offsets = np.ones_like(X)
    gamma = np.zeros((n, 2))
    gamma[np.arange(n), labels] = 1.0

    em = NegativeBinomialEmissions(np.ones((2, 2)), disp.copy())
    em.m_step(X, offsets, gamma)
    np.testing.assert_allclose(em.mu, true_mu, rtol=0.05)


def test_size_factors_enter_as_offsets_not_rescaling():
    """Doubling depth and doubling the offset must leave the fitted mu unchanged."""
    rng = np.random.default_rng(10)
    n = 8000
    labels = rng.integers(0, 2, size=n)
    true_mu = np.array([[3.0, 20.0], [18.0, 2.0]])
    gamma = np.zeros((n, 2))
    gamma[np.arange(n), labels] = 1.0

    X1 = rng.poisson(true_mu[labels]).astype(float)
    em1 = NegativeBinomialEmissions(np.ones((2, 2)), np.array([50.0, 50.0]))
    em1.m_step(X1, np.ones_like(X1), gamma)

    X2 = rng.poisson(2.0 * true_mu[labels]).astype(float)
    em2 = NegativeBinomialEmissions(np.ones((2, 2)), np.array([50.0, 50.0]))
    em2.m_step(X2, np.full_like(X2, 2.0), gamma)

    np.testing.assert_allclose(em1.mu, em2.mu, rtol=0.08)


# -- end-to-end recovery -----------------------------------------------------


def test_recovers_simulated_states():
    from chromstate.benchmark.metrics import clustering_scores

    ds, truth = simulate(n_bins=1500, n_chroms=2, seed=11)
    seg = fit(ds, n_states=truth.n_states, n_restarts=3, seed=0)
    scores = clustering_scores(truth.states, seg.states, truth.n_states, truth.n_states)
    assert scores["ari"] > 0.85, f"ARI too low: {scores}"
    assert scores["matched_accuracy"] > 0.85, f"accuracy too low: {scores}"


def test_recovers_emission_means():
    """Fitted emission means must match the generating profiles, not merely cluster."""
    from chromstate.benchmark.metrics import match_states

    ds, truth = simulate(n_bins=2000, n_chroms=2, samples=("s1",), seed=12)
    seg = fit(ds, n_states=truth.n_states, seed=0)
    assert seg.meta["restart_hits_at_best"] >= 2, "fit did not reach a reproducible optimum"
    mapping, _ = match_states(truth.states, seg.states, truth.n_states, truth.n_states)

    fitted = seg.model.emissions.mean_matrix()
    depth = np.array([truth.depth_factors[("s1", m)] for m in truth.marks])
    expected = truth.emission_means * depth[None, :]
    for pred_k, true_k in mapping.items():
        np.testing.assert_allclose(fitted[pred_k], expected[true_k], rtol=0.35, atol=1.0)


def test_state_labels_are_biologically_sensible():
    ds, truth = simulate(n_bins=1500, n_chroms=2, seed=13)
    seg = fit(ds, n_states=truth.n_states, n_restarts=3, seed=0)
    labels = set(seg.labels)
    for expected in ("TssA", "TssBiv", "ReprPC", "Quies"):
        assert expected in labels, f"{expected} missing from {sorted(labels)}"


def test_e_step_zeroes_rows_outside_the_fitted_chains():
    """Regression: partial chain coverage must not leak uninitialised memory.

    Held-out model selection fits on a subset of chromosomes, so `chains` does not cover
    every row of X. An earlier version allocated the responsibility matrix with np.empty
    and only filled the covered rows, feeding uninitialised memory into the M-step as if
    it were responsibility. Every emission estimate was corrupted, all states collapsed
    onto one profile, and every K returned an identical log-likelihood -- while reporting
    convergence.
    """
    ds, _ = simulate(n_bins=400, n_chroms=3, samples=("s1",), seed=17)
    em = NegativeBinomialEmissions.initialize(ds.X, ds.size_factors, 5, np.random.default_rng(0))
    hmm = HMM(5, em)

    partial = ds.chains[:2]  # deliberately leaves the third chromosome uncovered
    gamma, _, _, _ = hmm._e_step(ds.X, ds.size_factors, partial)

    covered = np.zeros(ds.n_rows, dtype=bool)
    for sl in partial:
        covered[sl] = True
    assert not covered.all(), "test needs genuinely uncovered rows"
    np.testing.assert_array_equal(gamma[~covered], 0.0)
    np.testing.assert_allclose(gamma[covered].sum(axis=1), 1.0, atol=1e-9)


def test_partial_chain_fit_does_not_collapse():
    """Fitting on a subset of chromosomes must still separate states."""
    ds, truth = simulate(n_bins=1000, n_chroms=4, samples=("s1",), seed=18)
    em = NegativeBinomialEmissions.initialize(ds.X, ds.size_factors, 7, np.random.default_rng(0))
    hmm = HMM(7, em)
    hmm.fit(ds.X, ds.size_factors, ds.chains[:3], n_iter=80)

    mu = hmm.emissions.mean_matrix()
    totals = np.sort(mu.sum(axis=1))
    assert totals[-1] / max(totals[0], 1e-9) > 3.0, f"states collapsed: {totals}"
    assert hmm.emissions.dispersion.max() < 1e4


def test_select_n_states_prefers_the_true_k():
    ds, truth = simulate(n_bins=1200, n_chroms=4, samples=("s1",), seed=14)
    table = select_n_states(ds, candidates=(3, 5, 7, 9), n_restarts=1, n_iter=60, seed=0)
    best = int(table.loc[table["heldout_loglik_per_bin"].idxmax(), "n_states"])
    assert best >= 5, f"held-out likelihood chose K={best}, expected >= 5\n{table}"


def test_nb_emission_refused_on_peak_input():
    ds, _ = simulate(n_bins=200, n_chroms=1, samples=("s1",), seed=15)
    ds.is_count = False
    with pytest.raises(ValueError, match="requires count input"):
        fit(ds, n_states=4, emission="nb")


def test_auto_emission_follows_input_type():
    ds, _ = simulate(n_bins=300, n_chroms=1, samples=("s1",), seed=16)
    assert fit(ds, n_states=3, n_restarts=1, n_iter=10).meta["emission"] == "nb"
    ds.is_count = False
    assert fit(ds, n_states=3, n_restarts=1, n_iter=10).meta["emission"] == "gaussian"
