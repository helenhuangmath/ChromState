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

try:  # pragma: no cover - depends on the environment
    from numba import njit, prange

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False


if _HAS_NUMBA:  # pragma: no cover - compiled

    @njit(cache=True, parallel=True)
    def _gather_sum_numba(table, starts, inv, out):
        """``out[t, k] = sum_m table[starts[m] + inv[t, m], k]`` in one pass over rows."""
        T, M = inv.shape
        K = table.shape[1]
        for t in prange(T):
            for k in range(K):
                acc = 0.0
                for m in range(M):
                    acc += table[starts[m] + inv[t, m], k]
                out[t, k] = acc

    @njit(cache=True)
    def _pair_weights_numba(inv, starts, gamma, W):
        """``W[starts[m] + inv[t, m], k] += gamma[t, k]`` for every row, mark and state."""
        T, M = inv.shape
        K = gamma.shape[1]
        for t in range(T):
            for m in range(M):
                p = starts[m] + inv[t, m]
                for k in range(K):
                    W[p, k] += gamma[t, k]


def _pair_index(X: np.ndarray, offsets: np.ndarray, m: int):
    """Distinct (count, offset) pairs in column ``m`` and each row's index into them.

    Count data has a few thousand distinct values per mark however many bins there are,
    so likelihood terms are computed once per pair and gathered back to rows.
    """
    x_vals, x_inv = np.unique(X[:, m], return_inverse=True)
    o_vals, o_inv = np.unique(offsets[:, m], return_inverse=True)
    key = x_inv.ravel().astype(np.int64) * len(o_vals) + o_inv.ravel()
    ukey, inv = np.unique(key, return_inverse=True)
    return x_vals[ukey // len(o_vals)], o_vals[ukey % len(o_vals)], inv.ravel()


class _PairCache:
    """Per-mark pair indices, reused while the same arrays are passed back in.

    EM calls the E- and M-steps with the identical ``X``/``offsets`` objects every
    iteration; recomputing the sort each time dominated the runtime on genome-scale data.
    The key includes a content fingerprint so a different array that happens to reuse a
    freed object's id is never served a stale index.
    """

    def __init__(self) -> None:
        self.key = None
        self.pairs = None
        self.inv_matrix = None  # (T, M) int32: each row's pair index per mark
        self.starts = None  # (M,) int64: offset of each mark's pair table when stacked

    def get(self, X: np.ndarray, offsets: np.ndarray):
        key = (id(X), id(offsets), X.shape,
               float(X[:: max(1, len(X) // 997)].sum()),
               float(offsets[:: max(1, len(offsets) // 997)].sum()))
        if key != self.key:
            self.pairs = [_pair_index(X, offsets, m) for m in range(X.shape[1])]
            self.inv_matrix = np.stack([p[2] for p in self.pairs], axis=1).astype(np.int32)
            sizes = np.array([len(p[0]) for p in self.pairs], dtype=np.int64)
            self.starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
            self.key = key
        return self.pairs


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
    Z: np.ndarray, n_states: int, rng: np.random.Generator, n_init: int = 4,
    max_rows: int = 200_000,
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

    k-means is fitted on at most ``max_rows`` randomly chosen rows and the remaining rows
    are assigned to the nearest centre. Seeding only needs the cluster centres, and on a
    genome-scale matrix (10^7 rows) a full k-means costs more than the EM it initialises.
    """
    from sklearn.cluster import KMeans

    seed = int(rng.integers(0, 2**31 - 1))
    n = Z.shape[0]
    if n > max_rows:
        idx = rng.choice(n, size=max_rows, replace=False)
        km = KMeans(n_clusters=n_states, n_init=n_init, random_state=seed).fit(Z[idx])
        return km.predict(Z).astype(int)
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

    def _pair_tables(self, X: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        """Stacked ``(sum_m P_m, K)`` log-likelihood table over distinct (count, offset) pairs."""
        tables = []
        for m, (x, off, _inv) in enumerate(self._cache().get(X, offsets)):
            r = self.dispersion[m]
            mean = np.maximum(off[:, None] * self.mu[None, :, m], MIN_MEAN)
            rp = r + mean
            tables.append(
                (gammaln(x + r) - gammaln(r) - gammaln(x + 1.0))[:, None]
                + (r * np.log(r) - r * np.log(rp) + x[:, None] * (np.log(mean) - np.log(rp)))
            )
        return np.ascontiguousarray(np.concatenate(tables, axis=0))

    def log_prob(self, X: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        T, M = X.shape
        # The full NB log-likelihood (state-independent terms included, so the value is
        # a true log-likelihood comparable across models), evaluated once per distinct
        # (count, offset) pair and gathered back to rows in a single pass.
        table = self._pair_tables(X, offsets)
        cache = self._cache()
        out = np.empty((T, self.n_states))
        if _HAS_NUMBA:
            _gather_sum_numba(table, cache.starts, cache.inv_matrix, out)
        else:
            out[:] = 0.0
            for m in range(M):
                out += table[cache.starts[m] + cache.inv_matrix[:, m]]
        return out

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_pairs", None)  # row indices for one dataset; never worth persisting
        return state

    def _cache(self) -> "_PairCache":
        if getattr(self, "_pairs", None) is None:
            self._pairs = _PairCache()
        return self._pairs

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
        """One-dimensional weighted MLE for ``r``, per mark.

        The likelihood depends on a row only through its (count, offset) pair, and real
        count data has a few thousand distinct pairs across millions of bins. Rows are
        collapsed onto those pairs with their responsibilities summed, which gives the
        same objective while making each optimiser step independent of genome size.
        """
        cache = self._cache()
        pairs = cache.get(X, offsets)
        total_pairs = int(sum(len(p[0]) for p in pairs))
        if _HAS_NUMBA:
            W = np.zeros((total_pairs, gamma.shape[1]))
            _pair_weights_numba(cache.inv_matrix, cache.starts, np.ascontiguousarray(gamma), W)
        else:
            W = np.zeros((total_pairs, gamma.shape[1]))
            for m, (x, _off, inv) in enumerate(pairs):
                for k in range(gamma.shape[1]):
                    W[cache.starts[m]:cache.starts[m] + len(x), k] = np.bincount(
                        inv, weights=gamma[:, k], minlength=len(x))
        for m, (x, off, _inv) in enumerate(pairs):
            weights = W[cache.starts[m]:cache.starts[m] + len(x)]
            means = np.maximum(off[:, None] * self.mu[:, m][None, :], MIN_MEAN)

            def neg_ll(log_r: float, x=x, means=means, weights=weights) -> float:
                r = np.exp(log_r)
                rp = r + means
                ll = gammaln(x[:, None] + r) - gammaln(r) - gammaln(x[:, None] + 1.0)
                ll = ll + r * np.log(r) - r * np.log(rp) + x[:, None] * (np.log(means) - np.log(rp))
                return -float((weights * ll).sum())

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


class PoissonEmissions(NegativeBinomialEmissions):
    """Independent Poisson per mark: the no-overdispersion limit of the NB.

    Included as an *algorithm choice to be tested*, not recommended. Sequencing counts
    are overdispersed, and a Poisson emission reads the extra variance as evidence for
    state changes, producing fragmented, over-confident segmentations. The benchmark
    (:func:`chromstate.benchmark.algorithms.compare_algorithms`) quantifies this on
    simulated data so the choice of NB is justified rather than asserted.
    """

    name = "poisson"

    def __init__(self, mu: np.ndarray):
        super().__init__(mu, dispersion=np.full(np.asarray(mu).shape[1], np.inf))

    @classmethod
    def initialize(
        cls, X: np.ndarray, offsets: np.ndarray, n_states: int, rng: np.random.Generator
    ) -> "PoissonEmissions":
        nb = NegativeBinomialEmissions.initialize(X, offsets, n_states, rng)
        return cls(mu=nb.mu)

    def _pair_tables(self, X: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        tables = []
        for m, (x, off, _inv) in enumerate(self._cache().get(X, offsets)):
            mean = np.maximum(off[:, None] * self.mu[None, :, m], MIN_MEAN)
            tables.append(x[:, None] * np.log(mean) - mean - gammaln(x + 1.0)[:, None])
        return np.ascontiguousarray(np.concatenate(tables, axis=0))

    def log_prob_by_mark(self, X: np.ndarray, offsets: np.ndarray, state: int) -> np.ndarray:
        mean = np.maximum(offsets * self.mu[state], MIN_MEAN)
        return X * np.log(mean) - mean - gammaln(X + 1.0)

    def _update_dispersion(self, X: np.ndarray, offsets: np.ndarray, gamma: np.ndarray) -> None:
        return  # no dispersion parameter

    @property
    def n_free_params(self) -> int:
        return self.n_states * self.n_marks


class ZeroInflatedNBEmissions(NegativeBinomialEmissions):
    """Zero-inflated negative binomial per mark, with a per-state zero-inflation weight.

    ``P(x) = z * 1[x = 0] + (1 - z) * NB(x; mu, r)`` with ``z[k, m]`` per state and mark.
    This stands in, in the algorithm benchmark, for the flexible per-mark count
    distributions of EpiSegMix (Bioinformatics 2024), which selects a distribution per
    mark by likelihood (Poisson, negative binomial, Sichel, beta-negative-binomial, ...).
    Zero inflation is the extension most relevant to sequencing bins. The extra zero
    mass models bins with no reads beyond what the NB tail expects, for example
    unmappable or deleted regions; whether it helps on a given dataset is an empirical
    question that held-out likelihood answers.

    Fitting is EM-within-EM: given the state responsibilities ``gamma``, the probability
    that each zero came from the point mass is computed in closed form and used to weight
    the NB mean and dispersion updates.
    """

    name = "zinb"

    def __init__(self, mu: np.ndarray, dispersion: np.ndarray, zero: Optional[np.ndarray] = None):
        super().__init__(mu, dispersion)
        if zero is None:
            zero = np.full(self.mu.shape, 0.02)
        self.zero = np.clip(np.asarray(zero, dtype=np.float64), 1e-6, 1 - 1e-6)

    @classmethod
    def initialize(
        cls, X: np.ndarray, offsets: np.ndarray, n_states: int, rng: np.random.Generator
    ) -> "ZeroInflatedNBEmissions":
        nb = NegativeBinomialEmissions.initialize(X, offsets, n_states, rng)
        return cls(mu=nb.mu, dispersion=nb.dispersion)

    def _nb_table(self, x, off, m):
        r = self.dispersion[m]
        mean = np.maximum(off[:, None] * self.mu[None, :, m], MIN_MEAN)
        rp = r + mean
        return (gammaln(x + r) - gammaln(r) - gammaln(x + 1.0))[:, None] + (
            r * np.log(r) - r * np.log(rp) + x[:, None] * (np.log(mean) - np.log(rp))
        )

    def _pair_tables(self, X: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        tables = []
        for m, (x, off, _inv) in enumerate(self._cache().get(X, offsets)):
            nb = self._nb_table(x, off, m) + np.log1p(-self.zero[None, :, m])
            is_zero = (x == 0)[:, None]
            with np.errstate(divide="ignore"):
                zero_term = np.where(is_zero, np.log(self.zero[None, :, m]), -np.inf)
            tables.append(np.logaddexp(nb, zero_term))
        return np.ascontiguousarray(np.concatenate(tables, axis=0))

    def log_prob_by_mark(self, X: np.ndarray, offsets: np.ndarray, state: int) -> np.ndarray:
        nb = super().log_prob_by_mark(X, offsets, state) + np.log1p(-self.zero[state])
        with np.errstate(divide="ignore"):
            zero_term = np.where(X == 0, np.log(self.zero[state]), -np.inf)
        return np.logaddexp(nb, zero_term)

    def m_step(self, X: np.ndarray, offsets: np.ndarray, gamma: np.ndarray) -> None:
        cache = self._cache()
        pairs = cache.get(X, offsets)
        K = gamma.shape[1]
        total_pairs = int(sum(len(p[0]) for p in pairs))
        W = np.zeros((total_pairs, K))
        if _HAS_NUMBA:
            _pair_weights_numba(cache.inv_matrix, cache.starts, np.ascontiguousarray(gamma), W)
        else:
            for m, (x, _off, inv) in enumerate(pairs):
                for k in range(K):
                    W[cache.starts[m]:cache.starts[m] + len(x), k] = np.bincount(
                        inv, weights=gamma[:, k], minlength=len(x))
        for m, (x, off, _inv) in enumerate(pairs):
            w = W[cache.starts[m]:cache.starts[m] + len(x)]  # (P, K)
            # responsibility of the point mass for each zero pair
            nb0 = np.exp(self._nb_table(x, off, m))  # (P, K) NB density
            z = self.zero[None, :, m]
            resp_zero = np.where((x == 0)[:, None], z / np.maximum(z + (1 - z) * nb0, 1e-300), 0.0)
            w_nb = w * (1.0 - resp_zero)
            tot = np.maximum(w.sum(axis=0), 1e-12)
            self.zero[:, m] = np.clip((w * resp_zero).sum(axis=0) / tot, 1e-6, 0.95)
            num = (w_nb * x[:, None]).sum(axis=0)
            den = np.maximum((w_nb * off[:, None]).sum(axis=0), 1e-12)
            self.mu[:, m] = np.maximum(num / den, MIN_MEAN)
            means = np.maximum(off[:, None] * self.mu[:, m][None, :], MIN_MEAN)

            def neg_ll(log_r: float, x=x, means=means, weights=w_nb) -> float:
                r = np.exp(log_r)
                rp = r + means
                ll = gammaln(x[:, None] + r) - gammaln(r) - gammaln(x[:, None] + 1.0)
                ll = ll + r * np.log(r) - r * np.log(rp) + x[:, None] * (np.log(means) - np.log(rp))
                return -float((weights * ll).sum())

            res = minimize_scalar(neg_ll, bounds=(np.log(1e-3), np.log(1e5)), method="bounded",
                                  options={"xatol": 1e-3})
            if res.success:
                self.dispersion[m] = float(np.exp(res.x))

    @property
    def n_free_params(self) -> int:
        return 2 * self.n_states * self.n_marks + self.n_marks

    def reorder(self, order: np.ndarray) -> None:
        self.mu = self.mu[order]
        self.zero = self.zero[order]


class NegativeMultinomialEmissions(Emissions):
    """EpiCSeg's emission model: one negative multinomial over all marks per state.

    The bin's total count follows a negative binomial with a *per-state* size ``r_k`` and
    the split across marks is multinomial with state-specific proportions, so marks are
    **not** conditionally independent given the state: a bin with more reads than
    expected in one mark is expected to have more in every mark. This is the published
    model of EpiCSeg (Mammana & Chung, Genome Biology 2015) and the nearest prior art to
    ChromState's count emission. Two consequences are relevant to the comparison:

    * a single shared overdispersion per state cannot describe marks with different noise
      levels, whereas ChromState's NB has a dispersion per mark;
    * because the marks are coupled, the log-likelihood ratio between two states does not
      decompose into per-mark terms. :meth:`log_prob_by_mark` therefore returns the
      multinomial part only and raises on request for an exact decomposition.

    With size factors ``s_m`` the expected count of mark ``m`` in state ``k`` is
    ``s_m * mu[k, m]``; the total mean is ``sum_m s_m mu[k, m]``.
    """

    name = "negative_multinomial"

    def __init__(self, mu: np.ndarray, dispersion: np.ndarray):
        self.mu = np.asarray(mu, dtype=np.float64)
        self.dispersion = np.asarray(dispersion, dtype=np.float64)  # (K,)
        self.n_states, self.n_marks = self.mu.shape

    @classmethod
    def initialize(cls, X, offsets, n_states, rng):
        nb = NegativeBinomialEmissions.initialize(X, offsets, n_states, rng)
        # per-state size from the pooled per-mark estimate, scaled to the total
        r = np.full(n_states, float(np.median(nb.dispersion)))
        return cls(mu=nb.mu, dispersion=r)

    def _terms(self, X, offsets):
        means = np.maximum(offsets[:, None, :] * self.mu[None, :, :], MIN_MEAN)  # (T, K, M)
        tot = means.sum(axis=2)  # (T, K)
        return means, tot

    def log_prob(self, X, offsets):
        n = X.sum(axis=1)  # (T,)
        lgx = gammaln(X + 1.0).sum(axis=1)  # (T,)
        means, tot = self._terms(X, offsets)
        r = self.dispersion[None, :]
        out = gammaln(n[:, None] + r) - gammaln(r) - lgx[:, None]
        out += r * (np.log(r) - np.log(r + tot))
        out += (X[:, None, :] * (np.log(means) - np.log(r + tot)[:, :, None])).sum(axis=2)
        return out

    def log_prob_by_mark(self, X, offsets, state):
        raise NotImplementedError(
            "The negative multinomial couples marks through a shared total, so its "
            "log-likelihood does not decompose into per-mark terms. This is a property of "
            "the EpiCSeg model, not a missing feature."
        )

    def m_step(self, X, offsets, gamma):
        num = gamma.T @ X
        den = gamma.T @ offsets
        self.mu = np.maximum(num / np.maximum(den, 1e-12), MIN_MEAN)
        n = X.sum(axis=1)
        _, tot = self._terms(X, offsets)
        for k in range(self.n_states):
            w = gamma[:, k]
            if w.sum() < 1e-6:
                continue
            tk = tot[:, k]

            def neg_ll(log_r, n=n, tk=tk, w=w):
                r = np.exp(log_r)
                ll = gammaln(n + r) - gammaln(r) + r * (np.log(r) - np.log(r + tk)) - n * np.log(r + tk)
                return -float((w * ll).sum())

            res = minimize_scalar(neg_ll, bounds=(np.log(1e-3), np.log(1e5)), method="bounded",
                                  options={"xatol": 1e-3})
            if res.success:
                self.dispersion[k] = float(np.exp(res.x))

    def mean_matrix(self):
        return self.mu.copy()

    @property
    def n_free_params(self):
        return self.n_states * self.n_marks + self.n_states

    def reorder(self, order):
        self.mu = self.mu[order]
        self.dispersion = self.dispersion[order]


class TiedEmissions(Emissions):
    """``n_tiles`` copies of a base emission model sharing parameters (a flattened hierarchy).

    Used to stand in for diHMM (Marco et al., Nature Communications 2017): bin-level
    chromatin states nested inside domain-level states. Flattening the hierarchy gives an
    ordinary HMM over ``(domain, bin-state)`` pairs whose emissions depend on the bin state
    only, so each domain state is free to have its own transition structure among bin
    states while sharing their meaning. The posterior over the flat states marginalises
    to a bin-state posterior (sum over domains) and a domain posterior (sum over bin
    states).
    """

    name = "tied"

    def __init__(self, base: Emissions, n_tiles: int):
        self.base = base
        self.n_tiles = int(n_tiles)
        self.n_states = base.n_states * self.n_tiles
        self.n_marks = base.n_marks

    def log_prob(self, X, offsets):
        return np.tile(self.base.log_prob(X, offsets), (1, self.n_tiles))

    def log_prob_by_mark(self, X, offsets, state):
        return self.base.log_prob_by_mark(X, offsets, state % self.base.n_states)

    def m_step(self, X, offsets, gamma):
        K = self.base.n_states
        folded = gamma.reshape(gamma.shape[0], self.n_tiles, K).sum(axis=1)
        self.base.m_step(X, offsets, folded)

    def mean_matrix(self):
        return np.tile(self.base.mean_matrix(), (self.n_tiles, 1))

    @property
    def n_free_params(self):
        return self.base.n_free_params

    def reorder(self, order):
        # Tied states cannot be permuted independently; keep the base order.
        return

    def fold_posterior(self, gamma):
        """Bin-state posterior ``(T, K)`` and domain posterior ``(T, D)`` from the flat one."""
        K = self.base.n_states
        g = gamma.reshape(gamma.shape[0], self.n_tiles, K)
        return g.sum(axis=1), g.sum(axis=2)


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
    "poisson": PoissonEmissions,
    "zinb": ZeroInflatedNBEmissions,
    "negative_multinomial": NegativeMultinomialEmissions,
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
