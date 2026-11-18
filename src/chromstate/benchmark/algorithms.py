"""Which inference algorithm should ChromState use? Measured, not asserted.

:func:`compare_algorithms` fits the same data with every candidate emission model and
scores each on

* **held-out log-likelihood per bin** on chromosomes the model never saw -- the one
  criterion that is fair across emission families on *count* input, because every count
  model is a probability mass function on the same observations (the Gaussian is on
  log-transformed data and is reported but marked as not comparable);
* **state recovery** against simulated truth (ARI, matched accuracy) when truth is given;
* **calibration** (expected calibration error, fraction of saturated posteriors);
* **runtime** and number of EM iterations to convergence.

:func:`speed_benchmark` times the E-step with the fused, chain-parallel kernel against
the per-chain reference, so the speed claim in the documentation is reproducible on the
machine it is made on.

The intended reading: the negative binomial is the default because, on overdispersed
counts, it has the best held-out likelihood among the models whose likelihoods are
comparable, and the zero-inflated variant is offered for data where ``zinb`` wins the
held-out comparison. Poisson is included so that its failure mode (fragmented,
over-confident segmentation) is on the record rather than assumed.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..dataset import ChromStateDataset
from ..model.emissions import EMISSION_REGISTRY, build_emissions
from ..model.hmm import HMM, e_step_all_chains, forward_backward
from ..simulate import SimulationTruth
from . import metrics as M

COUNT_EMISSIONS = ("nb", "zinb", "poisson")


def _split_chains(chains: Sequence[slice], holdout_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    n = len(chains)
    n_test = max(1, int(round(holdout_fraction * n))) if n > 1 else 0
    test_idx = set(rng.choice(n, size=n_test, replace=False).tolist()) if n_test else set()
    train = [sl for i, sl in enumerate(chains) if i not in test_idx]
    test = [sl for i, sl in enumerate(chains) if i in test_idx]
    return train, test


def compare_algorithms(
    ds: ChromStateDataset,
    n_states: int,
    truth: Optional[SimulationTruth] = None,
    emissions: Sequence[str] = ("nb", "zinb", "poisson", "gaussian"),
    holdout_fraction: float = 0.25,
    n_restarts: int = 2,
    seed: int = 0,
) -> pd.DataFrame:
    """Fit each emission model with an equal restart budget and score it.

    Returns one row per emission with ``heldout_loglik_per_bin`` (NaN for the Gaussian,
    whose likelihood is on a different scale), recovery and calibration metrics if
    ``truth`` is given, and runtime.
    """
    train, test = _split_chains(ds.chains, holdout_fraction, seed)
    n_test = sum(sl.stop - sl.start for sl in test)
    rows: List[Dict[str, object]] = []
    for em in emissions:
        if em not in EMISSION_REGISTRY:
            raise ValueError(f"unknown emission {em!r}; choose from {sorted(EMISSION_REGISTRY)}")
        if em in COUNT_EMISSIONS and not ds.is_count:
            continue
        t0 = time.time()
        # Fit on the training chains only so the held-out score is honest; the full
        # posterior is then computed on all chains for the truth-based metrics. Every
        # emission gets the same fixed number of restarts.
        best_ll, best_hmm = -np.inf, None
        for r in range(n_restarts):
            rng = np.random.default_rng(seed + r)
            hmm_r = HMM(n_states, build_emissions(em, ds.X, ds.size_factors, n_states, rng))
            res = hmm_r.fit(ds.X, ds.size_factors, train)
            if res.loglik > best_ll:
                best_ll, best_hmm = res.loglik, hmm_r
        runtime = time.time() - t0
        assert best_hmm is not None
        hmm: HMM = best_hmm
        heldout = hmm.score(ds.X, ds.size_factors, test) if n_test else np.nan
        row: Dict[str, object] = {
            "emission": em,
            "comparable_likelihood": em in COUNT_EMISSIONS,
            "heldout_loglik_per_bin": (heldout / n_test) if n_test else np.nan,
            "train_loglik": best_ll,
            "n_params": hmm.n_free_params,
            "n_iter": hmm.fit_result.n_iter if hmm.fit_result else None,
            "runtime_s": round(runtime, 2),
        }
        if truth is not None:
            post = hmm.posterior(ds.X, ds.size_factors, ds.chains)
            states = np.argmax(post, axis=1)
            row.update(M.clustering_scores(truth.states, states, truth.n_states, n_states))
            mapping, _ = M.match_states(truth.states, states, truth.n_states, n_states)
            cal = M.calibration_scores(post, truth.states, mapping)
            row.update({k: cal[k] for k in ("ece", "overconfidence", "frac_saturated", "mean_entropy")
                        if k in cal})
            # Segment fragmentation: mean segment length relative to the truth.
            row["mean_segment_len"] = _mean_segment_length(states, ds.chains)
            row["true_mean_segment_len"] = _mean_segment_length(truth.states, ds.chains)
        rows.append(row)
    table = pd.DataFrame(rows)
    front = ["emission", "heldout_loglik_per_bin", "ari", "matched_accuracy", "ece",
             "frac_saturated", "mean_segment_len", "runtime_s", "n_iter"]
    cols = [c for c in front if c in table.columns] + [c for c in table.columns if c not in front]
    return table[cols]


def _mean_segment_length(states: np.ndarray, chains: Sequence[slice]) -> float:
    lengths: List[int] = []
    for sl in chains:
        s = states[sl]
        if len(s) == 0:
            continue
        change = np.flatnonzero(np.diff(s) != 0)
        bounds = np.concatenate([[0], change + 1, [len(s)]])
        lengths.extend(np.diff(bounds).tolist())
    return float(np.mean(lengths)) if lengths else float("nan")


def speed_benchmark(ds: ChromStateDataset, n_states: int = 8, n_repeat: int = 3, seed: int = 0) -> pd.DataFrame:
    """Time one E-step: fused chain-parallel kernel versus the per-chain reference.

    Both paths compute identical posteriors (tested to 1e-12); this only measures time.
    The first fused call includes numba compilation and is excluded by a warm-up run.
    """
    from ..model.emissions import build_emissions

    em = build_emissions("nb" if ds.is_count else "gaussian", ds.X, ds.size_factors, n_states,
                         np.random.default_rng(seed))
    hmm = HMM(n_states, em)
    log_B = em.log_prob(ds.X, ds.size_factors)
    e_step_all_chains(log_B, hmm.transitions, hmm.init_probs, ds.chains)  # warm-up / compile

    def _reference():
        for sl in ds.chains:
            forward_backward(log_B[sl], hmm.transitions, hmm.init_probs)

    def _fused():
        e_step_all_chains(log_B, hmm.transitions, hmm.init_probs, ds.chains)

    def _emission():
        em.log_prob(ds.X, ds.size_factors)

    rows = []
    for name, fn in [("per-chain reference", _reference), ("fused, chain-parallel", _fused),
                     ("NB emission log-likelihood", _emission)]:
        times = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        rows.append({"step": name, "seconds": float(np.median(times)), "n_rows": ds.n_rows,
                     "n_states": n_states, "n_chains": len(ds.chains)})
    table = pd.DataFrame(rows)
    ref = table.loc[table["step"] == "per-chain reference", "seconds"].iloc[0]
    table["speedup_vs_reference"] = ref / table["seconds"]
    return table
