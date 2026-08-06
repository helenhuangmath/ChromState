"""Benchmark driver: run every method on identical data and score them against truth."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..dataset import ChromStateDataset
from ..model.annotate import canonical_mark
from ..model.binarize import poisson_binarize
from ..norm import compute_size_factors, normalize
from ..simulate import SimulationTruth, simulate
from . import metrics as M
from .baselines import METHODS, MethodResult


def run_comparison(
    ds: ChromStateDataset,
    truth: SimulationTruth,
    n_states: Optional[int] = None,
    methods: Sequence[str] = ("chromstate", "chromhmm_like", "segway_like", "gmm", "kmeans", "naive"),
    seed: int = 0,
    n_restarts: int = 3,
) -> pd.DataFrame:
    """Score every method on the same simulated dataset.

    Returns one row per method with recovery, calibration, quantitative-sensitivity, and
    bivalency-detection metrics. Methods that do not produce a posterior (naive signal)
    report NaN for calibration rather than a fabricated value.
    """
    n_states = n_states or truth.n_states
    true_states = truth.states
    results: List[MethodResult] = []

    for key in methods:
        if key not in METHODS:
            raise ValueError(f"Unknown method {key!r}; available: {sorted(METHODS)}")
        results.append(METHODS[key](ds, n_states=n_states, seed=seed, n_restarts=n_restarts))

    # The true per-bin amount of the active mark, used to test whether a method's output
    # tracks magnitude rather than mere presence.
    canon = [canonical_mark(m) for m in truth.marks]
    k4_idx = canon.index("K4me3") if "K4me3" in canon else 0
    true_signal = truth.emission_means[true_states, k4_idx]
    biv_truth = truth.states == truth.bivalent_state if truth.bivalent_state is not None else None

    rows = []
    for res in results:
        row: Dict[str, object] = {
            "method": res.name,
            "runtime_s": round(res.runtime_s, 2),
            "notes": res.notes,
        }

        if res.name.startswith("Naive"):
            row.update({k: np.nan for k in ("ari", "ami", "v_measure", "matched_accuracy")})
            row.update({k: np.nan for k in ("accuracy", "mean_confidence", "overconfidence",
                                            "brier", "ece", "mean_entropy", "frac_saturated")})
        else:
            clus = M.clustering_scores(true_states, res.states, truth.n_states, res.n_states)
            row.update(clus)
            if res.posterior is not None:
                mapping, _ = M.match_states(
                    true_states, np.argmax(res.posterior, axis=1), truth.n_states, res.n_states
                )
                row.update(M.calibration_scores(res.posterior, true_states, mapping))

        if res.posterior is not None and not res.name.startswith("Naive"):
            row.update(
                {
                    f"signal_{k}": v
                    for k, v in M.quantitative_sensitivity(res.activity_score, true_signal).items()
                }
            )
        else:
            row.update({"signal_spearman": np.nan, "signal_pearson": np.nan, "signal_n": np.nan})

        if biv_truth is not None:
            det = M.detection_scores(res.bivalency_score, biv_truth)
            row.update({f"biv_{k}": v for k, v in det.items()})

        rows.append(row)

    table = pd.DataFrame(rows)
    front = [
        "method", "ari", "ami", "matched_accuracy", "biv_auprc", "biv_auroc",
        "signal_spearman", "overconfidence", "frac_saturated", "mean_entropy", "runtime_s",
    ]
    ordered = [c for c in front if c in table.columns] + [
        c for c in table.columns if c not in front
    ]
    return table[ordered]


def run_binarization_experiment(ds: ChromStateDataset, p_threshold: float = 1e-4) -> pd.DataFrame:
    """Measure exactly what ChromHMM's binarisation step discards.

    For each mark: what fraction of bins survive as "present", and across those present
    bins, what range of read counts got collapsed onto the single value 1. The fold range
    is the quantitative statement of the information loss.
    """
    from ..model.binarize import binarization_information_loss

    B = poisson_binarize(ds.X, control=ds.control, p_threshold=p_threshold)
    stats = binarization_information_loss(ds.X, B)
    rows = []
    for j, mark in enumerate(ds.marks):
        rows.append({"mark": mark, **stats[j]})
    return pd.DataFrame(rows)


def run_normalization_experiment(
    n_bins: int = 4000,
    n_chroms: int = 2,
    retained: float = 0.35,
    seed: int = 0,
) -> pd.DataFrame:
    """Does each normalisation preserve a genome-wide loss of H3K27me3, or erase it?

    Simulates an EZH2-inhibitor-like perturbation where the treated sample retains only
    ``retained`` of its H3K27me3 genome-wide, on top of an unrelated random difference in
    sequencing depth.

    A size factor is supposed to absorb the *technical* depth difference and nothing else,
    so the correct answer is the true depth ratio. An estimator that instead returns
    ``depth_ratio * retained`` has folded the biology into the offset: after
    normalisation the treated sample looks untouched, and no downstream model can
    recover what was erased.

    ``biology_absorbed`` is the diagnostic. 1.0 means the estimator kept the biology
    intact; a value near ``retained`` means it normalised the drug effect away.
    """
    ds, truth = simulate(
        n_bins=n_bins,
        n_chroms=n_chroms,
        seed=seed,
        perturbation={"global_mark_loss": {"H3K27me3": retained}, "responder_fraction": 0.0},
    )
    control, treated = ds.samples[0], ds.samples[1]
    mark = "H3K27me3"
    j = ds.marks.index(mark)

    correct_factor = truth.depth_factors[(treated, mark)] / truth.depth_factors[(control, mark)]

    rows = []
    for method in ("depth", "median_ratio", "spikein"):
        kwargs = {"spikein": truth.spikein_counts} if method == "spikein" else {}
        factors = compute_size_factors(ds, method=method, reference=control, **kwargs)
        est = float(factors[ds.samples.index(treated), j])
        absorbed = est / max(correct_factor, 1e-9)
        log2_err = float(np.log2(max(est, 1e-9) / max(correct_factor, 1e-9)))
        rows.append(
            {
                "method": method,
                "estimated_factor": est,
                "correct_factor": float(correct_factor),
                "biology_absorbed": float(absorbed),
                "true_retention": retained,
                "log2_error": log2_err,
                "preserves_global_loss": bool(abs(log2_err) < 0.3),
            }
        )
    return pd.DataFrame(rows)


def run_perturbation_experiment(
    n_bins: int = 4000,
    n_chroms: int = 2,
    responder_fraction: float = 0.5,
    seed: int = 0,
    n_states: int = 7,
) -> pd.DataFrame:
    """Do methods detect which regions actually changed between conditions?

    Half the bivalent bins switch to active promoters, mimicking a drug releasing poised
    genes. Each method scores every bin for change, and the score is evaluated against
    the known responders.
    """
    ds, truth = simulate(
        n_bins=n_bins,
        n_chroms=n_chroms,
        seed=seed,
        perturbation={
            "responder_fraction": responder_fraction,
            "from_state": "TssBiv",
            "to_state": "TssA",
        },
    )
    control, treated = ds.samples[0], ds.samples[1]
    n_per = ds.sample_mask(control).sum()
    responders = truth.responder_mask[:n_per]

    rows = []
    for key in ("chromstate", "chromhmm_like", "segway_like"):
        res = METHODS[key](ds, n_states=n_states, seed=seed, n_restarts=2)
        assert res.posterior is not None
        pa = res.posterior[ds.sample_mask(control)]
        pb = res.posterior[ds.sample_mask(treated)]
        tv = 0.5 * np.abs(pa - pb).sum(axis=1)
        det = M.detection_scores(tv, responders)
        rows.append(
            {
                "method": res.name,
                "responder_auprc": det["auprc"],
                "responder_auroc": det["auroc"],
                "baseline_auprc": det["baseline_auprc"],
                "n_responders": det["n_positive"],
                "mean_tv_responder": float(tv[responders].mean()),
                "mean_tv_static": float(tv[~responders].mean()),
            }
        )
    return pd.DataFrame(rows)


def run_full_benchmark(
    n_bins: int = 4000,
    n_chroms: int = 3,
    seed: int = 0,
    n_restarts: int = 3,
) -> Dict[str, pd.DataFrame]:
    """Run every benchmark and return the tables as a dict."""
    ds, truth = simulate(
        n_bins=n_bins,
        n_chroms=n_chroms,
        seed=seed,
        perturbation={
            "responder_fraction": 0.5,
            "from_state": "TssBiv",
            "to_state": "TssA",
            "global_mark_loss": {"H3K27me3": 0.5},
        },
    )
    normalize(ds, method="median_ratio")

    return {
        "methods": run_comparison(ds, truth, seed=seed, n_restarts=n_restarts),
        "binarization": run_binarization_experiment(ds),
        "normalization": run_normalization_experiment(n_bins=n_bins, n_chroms=2, seed=seed),
        "perturbation": run_perturbation_experiment(n_bins=n_bins, n_chroms=2, seed=seed),
    }
