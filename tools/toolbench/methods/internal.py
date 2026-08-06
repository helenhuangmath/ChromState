"""The six in-process methods: ChromState and the reimplemented baselines.

These are the published *statistical models* -- negative-binomial HMM, binarised
Bernoulli HMM, Gaussian HMM, Gaussian mixture, k-means, and a no-model signal formula --
running on ChromState's own code from :mod:`chromstate.benchmark.baselines`. They always
run, they are exactly reproducible, and they are the right thing for a fast comparison of
*modelling choices*.

They are not evidence about the real software. A row here labelled "ChromHMM-like" says
what a binarised Bernoulli HMM does on these bins; it says nothing about ChromHMM's
heuristics, its binarisation, or its defaults. For that, enable the real tool in
:mod:`toolbench.methods.chromhmm`.

The implementations stay in the installed package rather than moving here, because
``chromstate compare`` ships with the wheel and cannot import from this repo-local
folder. What lives here is the adapter: the uniform interface, not the model.
"""

from __future__ import annotations

import numpy as np

from chromstate.benchmark.baselines import METHODS

from ..core import Availability, ToolAdapter, ToolRun

#: ``(registry key, baselines key, display name, what it stands for)``. The display names
#: are spelled out because they are printed directly onto figures.
INTERNAL_METHODS = [
    ("chromstate", "chromstate",
     "ChromState (negative-binomial hidden Markov model)",
     "this package"),
    ("chromhmm_like", "chromhmm_like",
     "ChromHMM-like (binarised Bernoulli hidden Markov model)",
     "ChromHMM's statistical model, reimplemented"),
    ("segway_like", "segway_like",
     "Segway-like (Gaussian hidden Markov model)",
     "Segway and epigenome-ssm's model family"),
    ("gmm", "gmm",
     "Gaussian mixture model (no spatial model)",
     "isolates what the Markov chain contributes"),
    ("kmeans", "kmeans",
     "k-means clustering (hard assignment)",
     "isolates what soft assignment contributes"),
    ("naive", "naive",
     "Naive signal combination (no model)",
     "the control: two tracks and one formula"),
]


class InternalAdapter(ToolAdapter):
    """Wraps one entry of :data:`chromstate.benchmark.baselines.METHODS`."""

    kind = "in-process"
    verified = True

    def __init__(self, key: str, method_key: str, display_name: str, stands_for: str):
        self.key = key
        self.method_key = method_key
        self.display_name = display_name
        self.stands_for = stands_for

    def availability(self) -> Availability:
        import chromstate

        return Availability(True, "bundled with chromstate",
                            getattr(chromstate, "__version__", ""))

    def run(self, ds, n_states: int, seed: int, workdir: str, timeout_s: float) -> ToolRun:
        try:
            res = METHODS[self.method_key](ds, n_states=n_states, seed=seed, n_restarts=2)
        except Exception as exc:  # a baseline that dies must not take the run down
            return self.failed(f"{type(exc).__name__}: {exc}")
        return ToolRun(
            tool=self.key,
            display_name=self.display_name,
            status="ok",
            states=np.asarray(res.states, dtype=np.int64),
            posterior=res.posterior,
            n_states=res.n_states,
            bivalency_score=res.bivalency_score,
            activity_score=res.activity_score,
            runtime_s=res.runtime_s,
            version="in-process",
            verified=True,
            notes=res.notes,
        )


def build() -> list:
    """One adapter per entry in :data:`INTERNAL_METHODS`."""
    return [InternalAdapter(*spec) for spec in INTERNAL_METHODS]
