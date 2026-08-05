"""Emission models.

Three are provided, and the choice of emission is the single biggest determinant of
whether the resulting states carry quantitative information:

:class:`BernoulliEmissions`
    ChromHMM's model. Signal is thresholded to present/absent and each mark contributes
    an independent Bernoulli. Everything above the threshold looks identical, so a bin
    with 5 reads and a bin with 500 reads are the same observation. Implemented here as
    a *baseline*, so ChromState can be compared against ChromHMM's actual statistical
    model without needing Java ChromHMM installed.

:class:`NegativeBinomialEmissions`
    ChromState's default. Counts are modelled directly with a per-mark dispersion, so
    magnitude is retained and the mean-variance relationship of sequencing count data is
    respected. Size factors enter as offsets on the mean, never as a rescaling of the
    data.

:class:`GaussianEmissions`
    Diagonal Gaussian on log-transformed signal, in the spirit of Segway. Required for
    peak-score input, where counts are not available.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

EPS = 1e-300
MIN_MEAN = 1e-6


class Emissions:
    """Interface every emission model implements."""

    n_states: int
    n_marks: int
    name: str = "base"

    def log_prob(self, X: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        """Return the ``(T, K)`` log-likelihood of each row under each state."""
        raise NotImplementedError

    def log_prob_by_mark(self, X: np.ndarray, offsets: np.ndarray, state: int) -> np.ndarray:
        """``(T, M)`` per-mark log-likelihood contributions under a single state.

        This is what makes ChromState's state calls traceable: the difference between
        two states' per-mark contributions says exactly how much each mark pushed a bin
        towards one call over the other. Marks are conditionally independent given the
        state in all emission models here, so these terms sum to :meth:`log_prob`.
        """
        raise NotImplementedError

    def m_step(self, X: np.ndarray, offsets: np.ndarray, gamma: np.ndarray) -> None:
        """Weighted maximum-likelihood update given responsibilities ``gamma``."""
        raise NotImplementedError

    def mean_matrix(self) -> np.ndarray:
        """``(K, M)`` summary used for the emission heat map."""
        raise NotImplementedError

    @property
    def n_free_params(self) -> int:
        raise NotImplementedError

    def reorder(self, order: np.ndarray) -> None:
        """Permute states in place (used to sort states for display)."""
        raise NotImplementedError


def seed_clusters(
    Z: np.ndarray, n_states: int, rng: np.random.Generator, n_init: int = 4
) -> np.ndarray:
    """Seed state assignments by k-means on the per-mark signal *profile*.

    Clustering on the full mark vector, not on total signal, is essential and was the
    cause of a real failure: an earlier version seeded states along quantiles of summed
    signal, which orders states only by overall magnitude. Two states with similar totals
    but opposite composition -- Polycomb (H3K27me3 high, total ~21) and heterochromatin
    (H3K9me3 high, total ~18) -- were therefore seeded on top of each other, and EM could
    not pull them apart from there. Every random restart converged to the same merged
    solution, ~3,900 log-likelihood units below the optimum reachable from a
    composition-aware start, while confidently reporting itself converged.

    The trap is self-reinforcing: merging two states inflates the apparent within-state
    variance of their defining marks, which drives those marks' dispersion estimates down,
    which flattens the likelihood for exactly the marks that would have separated them.

    ``rng`` seeds k-means, so restarts still explore different initialisations.
    """
    from sklearn.cluster import KMeans

    seed = int(rng.integers(0, 2**31 - 1))
    km = KMeans(n_clusters=n_states, n_init=n_init, random_state=seed).fit(Z)
    return km.labels_.astype(int)


def _cluster_moments(
    values: np.ndarray, labels: np.ndarray, n_states: int
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Per-cluster means, per-cluster variances, and cluster sizes."""
    n_marks = values.shape[1]
    means = np.zeros((n_states, n_marks))
    variances = np.zeros((n_states, n_marks))
    sizes = np.zeros(n_states)
    global_mean = values.mean(axis=0)
    global_var = values.var(axis=0)
    for k in range(n_states):
        sel = labels == k
        sizes[k] = sel.sum()
        if sizes[k] >= 2:
            means[k] = values[sel].mean(axis=0)
            variances[k] = values[sel].var(axis=0)
        else:  # empty or singleton cluster: fall back to the global moments
            means[k] = global_mean
            variances[k] = global_var
    return means, variances, sizes


def _pooled_nb_dispersion(
    means: np.ndarray, variances: np.ndarray, sizes: np.ndarray
) -> np.ndarray:
    """Method-of-moments NB dispersion pooled over clusters.

    Under NB, ``var = mu + mu^2 / r``, so ``r = mu^2 / (var - mu)``. Pooling numerator and
    denominator across clusters (weighted by size) is more stable than averaging per-
    cluster estimates, which blow up whenever ``var`` approaches ``mu``.
    """
    w = np.maximum(sizes, 0.0)[:, None]
    num = (w * means**2).sum(axis=0)
    den = (w * np.maximum(variances - means, 1e-9)).sum(axis=0)
    return np.clip(num / np.maximum(den, 1e-12), 0.5, 1e4)


class NegativeBinomialEmissions(Emissions):
    """Independent negative binomial per mark, with per-mark dispersion.

    Parameters
    ----------
    mu
        ``(K, M)`` expected counts per state and mark, at unit size factor.
    dispersion
        ``(M,)`` NB size parameter ``r``; variance is ``mu + mu^2 / r``. Shared across
        states because per-state dispersions are poorly identified when a state occupies
        a small fraction of the genome.
    """

    name = "nb"

    def __init__(self, mu: np.ndarray, dispersion: np.ndarray):
        self.mu = np.asarray(mu, dtype=np.float64)
        self.dispersion = np.asarray(dispersion, dtype=np.float64)
        self.n_states, self.n_marks = self.mu.shape

    @classmethod
    def initialize(
        cls, X: np.ndarray, offsets: np.ndarray, n_states: int, rng: np.random.Generator
    ) -> "NegativeBinomialEmissions":
        """Seed states by clustering the per-mark signal profile (see :func:`seed_clusters`).

        Dispersion is estimated from *within-cluster* variance. Using the marginal
        variance instead is badly wrong: it is dominated by the differences between
        states, implying enormous overdispersion, which flattens the likelihood until no
        state can be told from any other and EM collapses everything onto one profile.
        """
        scaled = X / np.maximum(offsets, 1e-12)
        labels = seed_clusters(np.log1p(scaled), n_states, rng)
        mu, var, sizes = _cluster_moments(scaled, labels, n_states)

        # Break exact ties so two clusters that landed identically can still separate.
        mu = np.maximum(mu * rng.uniform(0.85, 1.18, size=mu.shape), MIN_MEAN)
        dispersion = _pooled_nb_dispersion(mu, var, sizes)
        return cls(mu=mu, dispersion=dispersion)

    def log_prob(self, X: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        T, M = X.shape
        K = self.n_states
        out = np.zeros((T, K))
        r = self.dispersion
        gl_r = gammaln(r)
        r_log_r = r * np.log(r)
        # Terms independent of the state are dropped from the loop for speed but kept
        # in the total, so the returned value is a true log-likelihood (comparable
        # across models, which the benchmark relies on).
        const = (gammaln(X + r) - gl_r - gammaln(X + 1.0)).sum(axis=1)
        for k in range(K):
            mean = np.maximum(offsets * self.mu[k], MIN_MEAN)
            rp = r + mean
            out[:, k] = const + (r_log_r - r * np.log(rp) + X * (np.log(mean) - np.log(rp))).sum(
                axis=1
            )
        return out

    def log_prob_by_mark(self, X: np.ndarray, offsets: np.ndarray, state: int) -> np.ndarray:
        r = self.dispersion
        mean = np.maximum(offsets * self.mu[state], MIN_MEAN)
        rp = r + mean
        return (
            gammaln(X + r)
            - gammaln(r)
            - gammaln(X + 1.0)
            + r * np.log(r)
            - r * np.log(rp)
            + X * (np.log(mean) - np.log(rp))
        )

    def m_step(self, X: np.ndarray, offsets: np.ndarray, gamma: np.ndarray) -> None:
        weight = gamma.sum(axis=0)  # (K,)
        num = gamma.T @ X  # (K, M)
        den = gamma.T @ offsets  # (K, M)
        self.mu = np.maximum(num / np.maximum(den, 1e-12), MIN_MEAN)
        # Guard against a state that has lost all responsibility.
        dead = weight < 1e-6
        if dead.any():
            self.mu[dead] = np.maximum(X.mean(axis=0), MIN_MEAN)
        self._update_dispersion(X, offsets, gamma)

    def _update_dispersion(self, X: np.ndarray, offsets: np.ndarray, gamma: np.ndarray) -> None:
        """One-dimensional weighted MLE for ``r``, per mark."""
        for m in range(self.n_marks):
            x = X[:, m]
            means = np.maximum(offsets[:, m][:, None] * self.mu[:, m][None, :], MIN_MEAN)

            def neg_ll(log_r: float, x=x, means=means) -> float:
                r = np.exp(log_r)
                rp = r + means
                ll = gammaln(x[:, None] + r) - gammaln(r) - gammaln(x[:, None] + 1.0)
                ll = ll + r * np.log(r) - r * np.log(rp) + x[:, None] * (np.log(means) - np.log(rp))
                return -float((gamma * ll).sum())

            res = minimize_scalar(
                neg_ll, bounds=(np.log(1e-3), np.log(1e5)), method="bounded",
                options={"xatol": 1e-3},
            )
            if res.success:
                self.dispersion[m] = float(np.exp(res.x))

    def mean_matrix(self) -> np.ndarray:
        return self.mu.copy()

    @property
    def n_free_params(self) -> int:
        return self.n_states * self.n_marks + self.n_marks

    def reorder(self, order: np.ndarray) -> None:
        self.mu = self.mu[order]


class BernoulliEmissions(Emissions):
    """ChromHMM's emission model: independent Bernoulli over binarised signal.

    Included so the benchmark compares against ChromHMM's *statistical model*, not a
    strawman. :func:`chromstate.model.binarize.poisson_binarize` reproduces ChromHMM's
    binarisation rule.
    """

    name = "bernoulli"

    def __init__(self, p: np.ndarray):
        self.p = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1 - 1e-4)
        self.n_states, self.n_marks = self.p.shape

    @classmethod
    def initialize(
        cls, X: np.ndarray, offsets: np.ndarray, n_states: int, rng: np.random.Generator
    ) -> "BernoulliEmissions":
        # Seeded on the binary presence profile, for the same reason as the NB model:
        # states must be separated by *which* marks are present, not by how many.
        B = (X > 0).astype(np.float64)
        labels = seed_clusters(B, n_states, rng)
        p, _, _ = _cluster_moments(B, labels, n_states)
        p = np.clip(p + rng.uniform(-0.03, 0.03, size=p.shape), 0.02, 0.98)
        return cls(p=p)

    def log_prob(self, X: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        B = (X > 0).astype(np.float64)
        log_p = np.log(self.p)
        log_q = np.log1p(-self.p)
        return B @ log_p.T + (1.0 - B) @ log_q.T

    def log_prob_by_mark(self, X: np.ndarray, offsets: np.ndarray, state: int) -> np.ndarray:
        B = (X > 0).astype(np.float64)
        return B * np.log(self.p[state]) + (1.0 - B) * np.log1p(-self.p[state])

    def m_step(self, X: np.ndarray, offsets: np.ndarray, gamma: np.ndarray) -> None:
        B = (X > 0).astype(np.float64)
        weight = gamma.sum(axis=0)[:, None]
        self.p = np.clip((gamma.T @ B) / np.maximum(weight, 1e-12), 1e-4, 1 - 1e-4)

    def mean_matrix(self) -> np.ndarray:
        return self.p.copy()

    @property
    def n_free_params(self) -> int:
        return self.n_states * self.n_marks

    def reorder(self, order: np.ndarray) -> None:
        self.p = self.p[order]


class GaussianEmissions(Emissions):
    """Diagonal Gaussian on log-transformed signal (Segway-like).

    Used for peak-score input, where genuine counts are unavailable.
    """

    name = "gaussian"

    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = np.asarray(mu, dtype=np.float64)
        self.sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-3)
        self.n_states, self.n_marks = self.mu.shape

    @classmethod
    def initialize(
        cls, X: np.ndarray, offsets: np.ndarray, n_states: int, rng: np.random.Generator
    ) -> "GaussianEmissions":
        Z = np.log1p(X / np.maximum(offsets, 1e-12))
        labels = seed_clusters(Z, n_states, rng)
        mu, var, _ = _cluster_moments(Z, labels, n_states)
        mu = mu + rng.normal(0, 0.03, size=mu.shape)
        sigma = np.maximum(np.sqrt(var), 1e-2)
        return cls(mu=mu, sigma=sigma)

    def _transform(self, X: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        return np.log1p(X / np.maximum(offsets, 1e-12))

    def log_prob(self, X: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        Z = self._transform(X, offsets)
        out = np.zeros((Z.shape[0], self.n_states))
        for k in range(self.n_states):
            d = (Z - self.mu[k]) / self.sigma[k]
            out[:, k] = -0.5 * (d**2).sum(axis=1) - np.log(self.sigma[k]).sum()
        return out - 0.5 * self.n_marks * np.log(2 * np.pi)

    def log_prob_by_mark(self, X: np.ndarray, offsets: np.ndarray, state: int) -> np.ndarray:
        Z = self._transform(X, offsets)
        d = (Z - self.mu[state]) / self.sigma[state]
        return -0.5 * d**2 - np.log(self.sigma[state]) - 0.5 * np.log(2 * np.pi)

    def m_step(self, X: np.ndarray, offsets: np.ndarray, gamma: np.ndarray) -> None:
        Z = self._transform(X, offsets)
        weight = np.maximum(gamma.sum(axis=0)[:, None], 1e-12)
        self.mu = (gamma.T @ Z) / weight
        for k in range(self.n_states):
            d2 = (Z - self.mu[k]) ** 2
            self.sigma[k] = np.sqrt(np.maximum((gamma[:, k] @ d2) / weight[k], 1e-6))

    def mean_matrix(self) -> np.ndarray:
        return np.expm1(self.mu).clip(min=0.0)

    @property
    def n_free_params(self) -> int:
        return 2 * self.n_states * self.n_marks

    def reorder(self, order: np.ndarray) -> None:
        self.mu = self.mu[order]
        self.sigma = self.sigma[order]


EMISSION_REGISTRY = {
    "nb": NegativeBinomialEmissions,
    "bernoulli": BernoulliEmissions,
    "gaussian": GaussianEmissions,
}


def build_emissions(
    kind: str,
    X: np.ndarray,
    offsets: np.ndarray,
    n_states: int,
    rng: Optional[np.random.Generator] = None,
) -> Emissions:
    if kind not in EMISSION_REGISTRY:
        raise ValueError(f"Unknown emission {kind!r}; expected one of {sorted(EMISSION_REGISTRY)}")
    rng = rng if rng is not None else np.random.default_rng(0)
    return EMISSION_REGISTRY[kind].initialize(X, offsets, n_states, rng)
