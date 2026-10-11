"""Hidden Markov model core: scaled forward-backward, Baum-Welch, Viterbi.

The chain is reset at every chromosome and sample boundary (the ``chains`` argument),
so a transition is never inferred across a centromere or between two different samples.

Numerics: emission log-likelihoods have their per-row maximum removed before
exponentiation, and the forward pass is scaled at every step. The returned
log-likelihood adds the removed maxima back, so it is a true log-likelihood on the
original scale and is directly comparable between emission models -- which the
benchmark's held-out likelihood comparison depends on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .emissions import Emissions

try:  # pragma: no cover - depends on the environment
    from numba import njit

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        def wrap(fn):
            return fn

        return wrap(args[0]) if args and callable(args[0]) else wrap


EPS = 1e-300


@njit(cache=True)
def _forward_numba(B, A, pi, alpha, c):  # pragma: no cover - compiled
    T, K = B.shape
    for k in range(K):
        alpha[0, k] = pi[k] * B[0, k]
    s = 0.0
    for k in range(K):
        s += alpha[0, k]
    if s <= 0.0:
        s = 1e-300
    c[0] = s
    for k in range(K):
        alpha[0, k] /= s
    for t in range(1, T):
        for j in range(K):
            acc = 0.0
            for i in range(K):
                acc += alpha[t - 1, i] * A[i, j]
            alpha[t, j] = acc * B[t, j]
        s = 0.0
        for j in range(K):
            s += alpha[t, j]
        if s <= 0.0:
            s = 1e-300
        c[t] = s
        for j in range(K):
            alpha[t, j] /= s
    return c


@njit(cache=True)
def _backward_numba(B, A, c, beta):  # pragma: no cover - compiled
    T, K = B.shape
    for k in range(K):
        beta[T - 1, k] = 1.0
    for t in range(T - 2, -1, -1):
        for i in range(K):
            acc = 0.0
            for j in range(K):
                acc += A[i, j] * B[t + 1, j] * beta[t + 1, j]
            beta[t, i] = acc / c[t + 1]
    return beta


def _forward_numpy(B, A, pi, alpha, c):
    T = B.shape[0]
    alpha[0] = pi * B[0]
    c[0] = max(alpha[0].sum(), EPS)
    alpha[0] /= c[0]
    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * B[t]
        c[t] = max(alpha[t].sum(), EPS)
        alpha[t] /= c[t]
    return c


def _backward_numpy(B, A, c, beta):
    T = B.shape[0]
    beta[T - 1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
    return beta


_forward = _forward_numba if _HAS_NUMBA else _forward_numpy
_backward = _backward_numba if _HAS_NUMBA else _backward_numpy


def forward_backward(
    log_B: np.ndarray, A: np.ndarray, pi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Posterior state probabilities and expected transition counts for one chain.

    Returns
    -------
    gamma
        ``(T, K)`` posterior probability of each state at each position.
    xi
        ``(K, K)`` expected number of transitions, summed over the chain.
    loglik
        Log-likelihood of the chain.
    """
    row_max = log_B.max(axis=1, keepdims=True)
    B = np.exp(log_B - row_max)
    T, K = B.shape

    alpha = np.empty((T, K))
    c = np.empty(T)
    _forward(B, A, pi, alpha, c)
    beta = np.empty((T, K))
    _backward(B, A, c, beta)

    gamma = alpha * beta
    total = gamma.sum(axis=1, keepdims=True)
    gamma /= np.maximum(total, EPS)

    if T > 1:
        scaled = (B[1:] * beta[1:]) / c[1:, None]
        xi = A * (alpha[:-1].T @ scaled)
    else:
        xi = np.zeros((K, K))

    loglik = float(np.log(np.maximum(c, EPS)).sum() + row_max.sum())
    return gamma, xi, loglik


@njit(cache=True)
def _viterbi_numba(log_B, log_A, log_pi):  # pragma: no cover - compiled
    T, K = log_B.shape
    delta = np.empty((T, K))
    psi = np.zeros((T, K), dtype=np.int32)
    for k in range(K):
        delta[0, k] = log_pi[k] + log_B[0, k]
    for t in range(1, T):
        for j in range(K):
            best = -np.inf
            arg = 0
            for i in range(K):
                v = delta[t - 1, i] + log_A[i, j]
                if v > best:
                    best = v
                    arg = i
            psi[t, j] = arg
            delta[t, j] = best + log_B[t, j]
    path = np.empty(T, dtype=np.int32)
    best = -np.inf
    arg = 0
    for k in range(K):
        if delta[T - 1, k] > best:
            best = delta[T - 1, k]
            arg = k
    path[T - 1] = arg
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


def _viterbi_numpy(log_B, log_A, log_pi):
    T, K = log_B.shape
    delta = np.empty((T, K))
    psi = np.zeros((T, K), dtype=np.int32)
    delta[0] = log_pi + log_B[0]
    for t in range(1, T):
        scores = delta[t - 1][:, None] + log_A
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = scores[psi[t], np.arange(K)] + log_B[t]
    path = np.empty(T, dtype=np.int32)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


def viterbi(log_B: np.ndarray, A: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Most likely state path for one chain."""
    log_A = np.log(np.maximum(A, EPS))
    log_pi = np.log(np.maximum(pi, EPS))
    impl = _viterbi_numba if _HAS_NUMBA else _viterbi_numpy
    return impl(np.ascontiguousarray(log_B, dtype=np.float64), log_A, log_pi)


@dataclass
class FitResult:
    loglik: float
    n_iter: int
    converged: bool
    history: List[float]


class HMM:
    """Multivariate HMM with a pluggable emission model.

    Parameters
    ----------
    n_states
        Number of chromatin states, K.
    emissions
        An :class:`~chromstate.model.emissions.Emissions` instance.
    transitions, init_probs
        Optional starting values; a sticky prior (high self-transition) is used by
        default because chromatin states form domains rather than alternating bin to bin.
    """

    def __init__(
        self,
        n_states: int,
        emissions: Emissions,
        transitions: Optional[np.ndarray] = None,
        init_probs: Optional[np.ndarray] = None,
        self_transition_prior: float = 0.9,
    ):
        self.n_states = n_states
        self.emissions = emissions
        if transitions is None:
            off = (1.0 - self_transition_prior) / max(n_states - 1, 1)
            transitions = np.full((n_states, n_states), off)
            np.fill_diagonal(transitions, self_transition_prior)
        self.transitions = np.asarray(transitions, dtype=np.float64)
        self.init_probs = (
            np.full(n_states, 1.0 / n_states) if init_probs is None else np.asarray(init_probs)
        )
        self.fit_result: Optional[FitResult] = None

    # -- E step ---------------------------------------------------------------

    def _e_step(
        self, X: np.ndarray, offsets: np.ndarray, chains: Sequence[slice]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        log_B = self.emissions.log_prob(X, offsets)
        # Zeros, not np.empty: `chains` need not cover every row of X. Held-out model
        # selection fits on a subset of chromosomes, and with np.empty the uncovered
        # rows would carry uninitialised memory into the M-step as if it were
        # responsibility -- silently corrupting every emission estimate. Zero weight is
        # both safe and correct: a row outside the fitted chains contributes nothing.
        gamma = np.zeros((X.shape[0], self.n_states))
        xi_sum = np.zeros((self.n_states, self.n_states))
        init_sum = np.zeros(self.n_states)
        loglik = 0.0
        for sl in chains:
            g, xi, ll = forward_backward(log_B[sl], self.transitions, self.init_probs)
            gamma[sl] = g
            xi_sum += xi
            init_sum += g[0]
            loglik += ll
        return gamma, xi_sum, init_sum, loglik

    # -- fitting --------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        offsets: np.ndarray,
        chains: Sequence[slice],
        n_iter: int = 200,
        tol: float = 1e-4,
        verbose: bool = False,
        update_transitions: bool = True,
    ) -> FitResult:
        """Baum-Welch. Stops when the relative log-likelihood gain falls below ``tol``."""
        history: List[float] = []
        prev = -np.inf
        converged = False
        it = 0
        for it in range(1, n_iter + 1):
            gamma, xi_sum, init_sum, loglik = self._e_step(X, offsets, chains)
            history.append(loglik)

            if update_transitions:
                denom = np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-12)
                self.transitions = xi_sum / denom
                self.init_probs = init_sum / max(init_sum.sum(), 1e-12)
            self.emissions.m_step(X, offsets, gamma)

            if verbose:
                print(f"  iter {it:3d}  loglik = {loglik:,.2f}")
            if np.isfinite(prev):
                rel = abs(loglik - prev) / max(abs(prev), 1.0)
                if rel < tol:
                    converged = True
                    prev = loglik
                    break
            prev = loglik

        self.fit_result = FitResult(
            loglik=float(prev), n_iter=it, converged=converged, history=history
        )
        return self.fit_result

    # -- inference ------------------------------------------------------------

    def posterior(
        self, X: np.ndarray, offsets: np.ndarray, chains: Sequence[slice]
    ) -> np.ndarray:
        """``(T, K)`` posterior state probabilities -- ChromState's primary output."""
        gamma, _, _, _ = self._e_step(X, offsets, chains)
        return gamma

    def decode(
        self, X: np.ndarray, offsets: np.ndarray, chains: Sequence[slice]
    ) -> np.ndarray:
        """Viterbi path -- the ChromHMM-style hard segmentation, for comparison."""
        log_B = self.emissions.log_prob(X, offsets)
        path = np.empty(X.shape[0], dtype=np.int32)
        for sl in chains:
            path[sl] = viterbi(log_B[sl], self.transitions, self.init_probs)
        return path

    def score(self, X: np.ndarray, offsets: np.ndarray, chains: Sequence[slice]) -> float:
        """Log-likelihood of data under the current parameters (used for held-out scoring)."""
        log_B = self.emissions.log_prob(X, offsets)
        return float(
            sum(forward_backward(log_B[sl], self.transitions, self.init_probs)[2] for sl in chains)
        )

    # -- model selection ------------------------------------------------------

    @property
    def n_free_params(self) -> int:
        k = self.n_states
        return self.emissions.n_free_params + k * (k - 1) + (k - 1)

    def bic(self, loglik: float, n_obs: int) -> float:
        return -2.0 * loglik + self.n_free_params * np.log(n_obs)

    def aic(self, loglik: float) -> float:
        return -2.0 * loglik + 2.0 * self.n_free_params

    # -- presentation ---------------------------------------------------------

    def reorder_states(self, order: np.ndarray) -> None:
        """Permute state indices in place (emissions, transitions, initial probs)."""
        order = np.asarray(order)
        self.emissions.reorder(order)
        self.transitions = self.transitions[np.ix_(order, order)]
        self.init_probs = self.init_probs[order]

    def sort_states_by_signal(self) -> np.ndarray:
        """Order states from quiet to active by total emission mean. Returns the order used."""
        order = np.argsort(self.emissions.mean_matrix().sum(axis=1))
        self.reorder_states(order)
        return order
