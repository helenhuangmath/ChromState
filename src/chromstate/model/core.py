"""Fitting orchestration: restarts, model selection, and the segmentation result object."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..dataset import ChromStateDataset
from .annotate import annotate_states
from .emissions import Emissions, build_emissions
from .hmm import HMM


@dataclass
class Segmentation:
    """A fitted ChromState model together with its inference on a dataset.

    ``posterior`` is the headline output. Unlike a hard segmentation it gives every bin a
    *distribution* over states, which is what the quantify layer turns into entropy,
    bivalency, and attribution scores.
    """

    posterior: np.ndarray
    states: np.ndarray
    labels: List[str]
    colors: List[str]
    rules: List[str]
    model: HMM
    dataset: ChromStateDataset
    loglik: float
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        return self.posterior.shape[1]

    @property
    def marks(self) -> List[str]:
        return list(self.dataset.marks)

    def emission_frame(self) -> pd.DataFrame:
        """``(K, M)`` emission means as a labelled DataFrame."""
        return pd.DataFrame(
            self.model.emissions.mean_matrix(), index=self.labels, columns=self.marks
        )

    def transition_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.model.transitions, index=self.labels, columns=self.labels)

    def state_table(self) -> pd.DataFrame:
        """Per-state summary: mnemonic, colour, rule matched, genome coverage."""
        hard = np.bincount(self.states, minlength=self.n_states) / len(self.states)
        soft = self.posterior.mean(axis=0)
        return pd.DataFrame(
            {
                "state": np.arange(self.n_states),
                "label": self.labels,
                "color": self.colors,
                "rule": self.rules,
                "coverage_hard": hard,
                "coverage_soft": soft,
            }
        )

    def sample_state_matrix(self, soft: bool = True) -> pd.DataFrame:
        """``(n_samples, K)`` state occupancy -- the basis of every cross-sample comparison."""
        rows = {}
        for s in self.dataset.samples:
            m = self.dataset.sample_mask(s)
            rows[s] = (
                self.posterior[m].mean(axis=0)
                if soft
                else np.bincount(self.states[m], minlength=self.n_states) / m.sum()
            )
        return pd.DataFrame(rows, index=self.labels).T

    def to_bed(self, sample: Optional[str] = None, merge: bool = True) -> pd.DataFrame:
        """ChromHMM-style dense BED of the hard segmentation (itemRgb coloured)."""
        bins = self.dataset.bins
        mask = np.ones(len(bins), dtype=bool) if sample is None else self.dataset.sample_mask(sample)
        sub = bins.loc[mask].reset_index(drop=True)
        st = self.states[mask]

        if not merge:
            out = sub[["chrom", "start", "end"]].copy()
            out["name"] = [self.labels[s] for s in st]
            out["score"] = 0
            out["strand"] = "."
            out["thickStart"] = out["start"]
            out["thickEnd"] = out["end"]
            out["itemRgb"] = [_hex_to_rgb(self.colors[s]) for s in st]
            return out

        rows = []
        chrom = sub["chrom"].to_numpy()
        start = sub["start"].to_numpy()
        end = sub["end"].to_numpy()
        i = 0
        n = len(sub)
        while i < n:
            j = i
            while j + 1 < n and st[j + 1] == st[i] and chrom[j + 1] == chrom[i] and start[j + 1] == end[j]:
                j += 1
            rows.append(
                {
                    "chrom": chrom[i],
                    "start": int(start[i]),
                    "end": int(end[j]),
                    "name": self.labels[st[i]],
                    "score": 0,
                    "strand": ".",
                    "thickStart": int(start[i]),
                    "thickEnd": int(end[j]),
                    "itemRgb": _hex_to_rgb(self.colors[st[i]]),
                }
            )
            i = j + 1
        return pd.DataFrame(rows)

    def posterior_frame(self, sample: Optional[str] = None) -> pd.DataFrame:
        bins = self.dataset.bins
        mask = np.ones(len(bins), dtype=bool) if sample is None else self.dataset.sample_mask(sample)
        out = bins.loc[mask, ["chrom", "start", "end", "sample"]].reset_index(drop=True)
        post = pd.DataFrame(self.posterior[mask], columns=self.labels)
        return pd.concat([out, post], axis=1)


def _hex_to_rgb(h: str) -> str:
    h = h.lstrip("#")
    return ",".join(str(int(h[i : i + 2], 16)) for i in (0, 2, 4))


def fit(
    ds: ChromStateDataset,
    n_states: int = 10,
    emission: str = "auto",
    n_restarts: int = 5,
    max_restarts: int = 20,
    adaptive_restarts: bool = True,
    n_iter: int = 200,
    tol: float = 1e-4,
    seed: int = 0,
    verbose: bool = False,
    self_transition_prior: float = 0.9,
) -> Segmentation:
    """Fit a ChromState model to a dataset.

    Parameters
    ----------
    n_states
        Number of chromatin states. Use :func:`select_n_states` if unsure.
    emission
        ``nb`` (counts, the default for read input), ``gaussian`` (peak-score input),
        ``bernoulli`` (ChromHMM's model, for benchmarking), or ``auto``.
    n_restarts
        Minimum number of random initialisations. Baum-Welch is not convex and the
        landscape here has real local optima: on test data a run that stopped at three
        restarts returned a fit ~1100 log-likelihood units worse than the global optimum,
        merging two states and missing heterochromatin entirely. The likelihood ranks the
        fits correctly -- there simply have to be enough of them.
    max_restarts, adaptive_restarts
        With ``adaptive_restarts``, restarting continues past ``n_restarts`` until the
        best log-likelihood has been reached independently at least twice, or
        ``max_restarts`` is hit. Two independent arrivals at the same optimum is weak but
        real evidence it is not a one-off local basin. ``meta['restart_hits_at_best']``
        records how many got there; a value of 1 at ``max_restarts`` means the fit should
        be treated with suspicion.
    """
    emission = _resolve_emission(emission, ds)
    if emission in ("nb", "poisson", "zinb", "negative_multinomial") and not ds.is_count:
        raise ValueError(
            f"emission={emission!r} requires count input (BAM or BED reads). This dataset was "
            "built from peak scores, which are already thresholded, so a negative "
            "binomial dispersion is not meaningful. Use emission='gaussian' instead."
        )

    X, offsets, chains = ds.X, ds.size_factors, ds.chains
    best: Optional[Tuple[float, HMM]] = None
    logliks: List[float] = []

    limit = max(n_restarts, max_restarts) if adaptive_restarts else n_restarts
    for r in range(limit):
        rng = np.random.default_rng(seed + r)
        em: Emissions = build_emissions(emission, X, offsets, n_states, rng)
        hmm = HMM(n_states, em, self_transition_prior=self_transition_prior)
        if verbose:
            print(f"restart {r + 1}")
        res = hmm.fit(X, offsets, chains, n_iter=n_iter, tol=tol, verbose=verbose)
        logliks.append(res.loglik)
        if best is None or res.loglik > best[0]:
            best = (res.loglik, hmm)

        if adaptive_restarts and (r + 1) >= n_restarts:
            hits = _hits_at_best(logliks)
            if hits >= 2:
                break

    assert best is not None
    loglik, hmm = best
    hits_at_best = _hits_at_best(logliks)
    # Only warn when the stability guarantee was requested: turning off adaptive
    # restarting is an explicit choice to accept a fixed budget.
    if hits_at_best < 2 and adaptive_restarts and len(logliks) > 1:
        warnings.warn(
            f"The best fit (loglik {loglik:,.1f}) was reached by only one of "
            f"{len(logliks)} restarts, and restart log-likelihoods span "
            f"{np.ptp(logliks):,.1f}. This optimum may be a local basin. Raise "
            "max_restarts, or reduce n_states.",
            RuntimeWarning,
            stacklevel=2,
        )
    hmm.sort_states_by_signal()

    posterior = hmm.posterior(X, offsets, chains)
    states = hmm.decode(X, offsets, chains)
    labels, colors, rules = annotate_states(hmm.emissions.mean_matrix(), ds.marks,
                                            occupancy=posterior.mean(axis=0))
    collapse = _collapse_diagnostic(hmm, states, n_states)

    return Segmentation(
        posterior=posterior,
        states=states,
        labels=labels,
        colors=colors,
        rules=rules,
        model=hmm,
        dataset=ds,
        loglik=loglik,
        meta={
            "emission": emission,
            "n_restarts": len(logliks),
            "restart_logliks": logliks,
            "restart_spread": float(np.ptp(logliks)) if len(logliks) > 1 else 0.0,
            "restart_hits_at_best": hits_at_best,
            "converged": hmm.fit_result.converged if hmm.fit_result else False,
            "n_iter": hmm.fit_result.n_iter if hmm.fit_result else 0,
            **collapse,
        },
    )


def _collapse_diagnostic(hmm: HMM, states: np.ndarray, n_states: int) -> Dict[str, object]:
    """Detect a degenerate fit where states are indistinguishable or unused.

    A collapsed fit still returns a segmentation, a likelihood, and pretty figures -- it
    just means nothing. This makes the failure loud instead of silent.
    """
    M = np.log1p(hmm.emissions.mean_matrix())
    scale = np.maximum(M.std(axis=0, keepdims=True), 1e-9)
    Mz = M / scale
    diff = Mz[:, None, :] - Mz[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_sep = float(dist.min()) if n_states > 1 else np.inf

    used = int(len(np.unique(states)))
    collapsed = bool(min_sep < 0.15 or used < max(2, n_states // 2))
    out: Dict[str, object] = {
        "min_state_separation": min_sep,
        "n_states_used": used,
        "collapsed": collapsed,
    }
    if collapsed:
        out["collapse_warning"] = (
            f"Fit looks degenerate: {used}/{n_states} states carry any bins and the "
            f"closest pair of emission profiles is {min_sep:.3g} apart. Treat these "
            "states as unreliable. Try more restarts, a smaller K, or check that the "
            "input is not dominated by a single signal level."
        )
        warnings.warn(out["collapse_warning"], RuntimeWarning, stacklevel=3)
    return out


def _hits_at_best(logliks: Sequence[float], rel_tol: float = 1e-4) -> int:
    """How many restarts landed within ``rel_tol`` of the best log-likelihood."""
    best = max(logliks)
    scale = max(abs(best), 1.0)
    return sum(1 for ll in logliks if abs(ll - best) / scale < rel_tol)


def _resolve_emission(emission: str, ds: ChromStateDataset) -> str:
    if emission != "auto":
        return emission
    return "nb" if ds.is_count else "gaussian"


def select_n_states(
    ds: ChromStateDataset,
    candidates: Sequence[int] = (4, 6, 8, 10, 12, 15, 18),
    emission: str = "auto",
    holdout_fraction: float = 0.25,
    n_restarts: int = 2,
    seed: int = 0,
    n_iter: int = 100,
    verbose: bool = False,
) -> pd.DataFrame:
    """Choose K by held-out log-likelihood, with BIC/AIC and a stability score alongside.

    Whole chains (chromosome arms) are held out rather than random bins, because
    neighbouring bins are strongly correlated and random holdout would leak the answer
    across the split and make every K look equally good.

    Returns a DataFrame with one row per candidate K; the caller decides. Held-out
    likelihood is the recommended criterion -- BIC systematically over-penalises here
    because adjacent bins are not independent observations.
    """
    emission = _resolve_emission(emission, ds)
    rng = np.random.default_rng(seed)
    n_chains = len(ds.chains)
    if n_chains < 2:
        raise ValueError(
            "Model selection needs at least 2 chains (chromosomes) to hold one out. "
            "Bin more than one chromosome, or pick K by BIC on the full fit."
        )
    n_test = max(1, int(round(holdout_fraction * n_chains)))
    test_idx = set(rng.choice(n_chains, size=n_test, replace=False).tolist())
    train_chains = [sl for i, sl in enumerate(ds.chains) if i not in test_idx]
    test_chains = [sl for i, sl in enumerate(ds.chains) if i in test_idx]
    n_test_obs = sum(sl.stop - sl.start for sl in test_chains)

    rows = []
    for k in candidates:
        best_ll, best_hmm = -np.inf, None
        restart_lls = []
        for r in range(n_restarts):
            rr = np.random.default_rng(seed + 1000 * k + r)
            em = build_emissions(emission, ds.X, ds.size_factors, k, rr)
            hmm = HMM(k, em)
            res = hmm.fit(ds.X, ds.size_factors, train_chains, n_iter=n_iter)
            restart_lls.append(res.loglik)
            if res.loglik > best_ll:
                best_ll, best_hmm = res.loglik, hmm
        assert best_hmm is not None
        heldout = best_hmm.score(ds.X, ds.size_factors, test_chains)
        n_train_obs = sum(sl.stop - sl.start for sl in train_chains)
        rows.append(
            {
                "n_states": k,
                "train_loglik": best_ll,
                "heldout_loglik": heldout,
                "heldout_loglik_per_bin": heldout / max(n_test_obs, 1),
                "bic": best_hmm.bic(best_ll, n_train_obs),
                "aic": best_hmm.aic(best_ll),
                "restart_spread": float(np.ptp(restart_lls)) if len(restart_lls) > 1 else 0.0,
                "n_params": best_hmm.n_free_params,
            }
        )
        if verbose:
            print(f"K={k:3d}  heldout/bin={rows[-1]['heldout_loglik_per_bin']:.4f}")
    return pd.DataFrame(rows)
