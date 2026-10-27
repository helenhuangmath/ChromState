"""Real diHMM (Marco et al. 2017), via the C++ reimplementation github.com/ykai16/diHMM.

diHMM is a hierarchical HMM with two coupled chains: *nucleosome-level* (bin) states
with Bernoulli emissions on binarised marks, exactly as in ChromHMM, and *domain-level*
states that switch only at domain boundaries (``domain_size`` bins; the paper's 20 x
200 bp = 4 kb) and choose which bin-level transition matrix applies inside the domain.
It therefore reports two labels per bin.

This adapter scores the **bin-level** label against the truth like every other row --
that is the level the simulator's states live at -- and keeps the **domain-level** label
in ``ToolRun.meta["domain_states"]`` for descriptive reporting only: ChromState's
simulator has no domain layer, so there is no ground truth for it.

Input is the same ChromHMM-format binarisation the ChromHMM row receives, so the two
binarised-model rows see identical presence calls.

**Verified** against ykai16/diHMM HEAD (2020) built from source with the
``tools/setup/build_dihmm.sh`` recipe. What the real code required, all documented in
that script: (1) an off-by-one in ``Emissions_Dec`` that throws under Armadillo bounds
checks (fixed; no used value changes); (2) a row-vector ``each_col`` product that current
Armadillo rejects (made explicit with ``.t()``; same arithmetic); (3) a serial build,
because the OpenMP training loop has an unsynchronised ``push_back`` race; and (4)
**the number of bin states must equal the number of domain states** -- the
``intermediate_model_update`` matrices are sized assuming it, and unequal counts fail
with an Armadillo size error. Upstream's only shipped configuration is 30/30. So this
adapter asks for ``n_states`` at both levels; ``DIHMM_DOMAIN_SIZE`` (default 20) is the
only free setting. There is no seed (k-means initialisation with a static subset) and no
posterior output (annotation is Viterbi), so calibration metrics do not apply and the
bivalency/activity scores come from hard labels via :mod:`toolbench.scores`. diHMM
crops each chromosome to whole domains; up to ``domain_size - 1`` trailing bins per
chromosome stay unassigned and show up in the coverage column.

Environment: ``DIHMM_BUILD`` is the directory holding ``dihmm_ext.so``; ``DIHMM_PYTHON``
the interpreter it was built for (default: the harness's own).
"""

from __future__ import annotations

import os
import shlex
import sys
from typing import List, Sequence, Tuple

import numpy as np

from ..core import UNASSIGNED, Availability, ToolAdapter, ToolRun, probe_version, run_command
from ..io_formats import export_chromhmm_binary
from ..scores import posthoc_scores

DRIVER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "drivers",
                      "dihmm_run.py")


def export_rows(ds, sample: str, chrom: str) -> np.ndarray:
    """Dataset rows of one (sample, chrom), in the order ``export_for_chromhmm`` writes."""
    bins = ds.bins.reset_index(drop=True)
    return np.flatnonzero(((bins["sample"] == sample) & (bins["chrom"] == chrom)).to_numpy())


def read_dihmm_outputs(ds, outdir: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Read every ``<sample>_<chrom>_dihmm.tsv`` back onto the bin grid.

    Rows follow the binary file's bin order and may be *shorter* than it -- diHMM crops
    to whole domains -- but never longer; a longer file is an error. Returns
    ``(bin_states, domain_states, n_files_read)``.
    """
    bins = ds.bins.reset_index(drop=True)
    bin_states = np.full(ds.n_rows, UNASSIGNED, dtype=np.int64)
    dom_states = np.full(ds.n_rows, UNASSIGNED, dtype=np.int64)
    found = 0
    for sample in ds.samples:
        for chrom in bins.loc[bins["sample"] == sample, "chrom"].unique():
            path = os.path.join(outdir, f"{sample}_{chrom}_dihmm.tsv")
            if not os.path.exists(path):
                continue
            ann = np.loadtxt(path, skiprows=1, dtype=np.int64, ndmin=2)
            rows = export_rows(ds, sample, chrom)
            if ann.shape[0] > rows.size:
                raise ValueError(f"{path} has {ann.shape[0]} rows for {rows.size} bins")
            bin_states[rows[: ann.shape[0]]] = ann[:, 0]
            dom_states[rows[: ann.shape[0]]] = ann[:, 1]
            found += 1
    return bin_states, dom_states, found


class DiHMMAdapter(ToolAdapter):
    key = "dihmm"
    display_name = "diHMM (real, hierarchical hidden Markov model; nucleosome level)"
    kind = "external"
    verified = True
    stands_for = "two-scale published segmenter; bin level scored, domain level descriptive"

    def _python(self) -> str:
        return os.environ.get("DIHMM_PYTHON", sys.executable)

    def availability(self) -> Availability:
        how = ("build ykai16/diHMM with tools/setup/build_dihmm.sh, then: "
               "export DIHMM_BUILD=/dir/with/dihmm_ext.so DIHMM_PYTHON=/env/bin/python")
        build = os.environ.get("DIHMM_BUILD", "")
        if not build:
            return Availability(False, "DIHMM_BUILD is not set", how_to_enable=how)
        if not os.path.exists(os.path.join(build, "dihmm_ext.so")):
            return Availability(False, f"no dihmm_ext.so in {build}", how_to_enable=how)
        probe = probe_version([self._python(), "-c",
                               f"import sys; sys.path.insert(0, {build!r}); import dihmm_ext; "
                               "print('dihmm_ext ok')"])
        if probe != "dihmm_ext ok":
            return Availability(False, f"{self._python()} cannot import dihmm_ext ({probe})",
                                how_to_enable=how)
        return Availability(True, f"dihmm_ext.so in {build}", "ykai16/diHMM (C++), HEAD 2020")

    def _argv(self, n_states: int, workdir: str) -> List[str]:
        return [self._python(), DRIVER,
                "--build-dir", os.environ.get("DIHMM_BUILD", "<$DIHMM_BUILD-unset>"),
                "--input-dir", os.path.join(workdir, "binary"),
                "--outdir", os.path.join(workdir, "out"),
                "--n-bin-states", str(n_states), "--n-domain-states", str(n_states),
                "--domain-size", os.environ.get("DIHMM_DOMAIN_SIZE", "20")]

    def dry_run_command(self, ds, n_states: int, seed: int, workdir: str) -> str:
        return " ".join(shlex.quote(a) for a in self._argv(n_states, workdir))

    def run(self, ds, n_states: int, seed: int, workdir: str, timeout_s: float) -> ToolRun:
        avail = self.availability()
        if not avail.available:
            return self.skipped(avail.detail)
        export_chromhmm_binary(ds, os.path.join(workdir, "binary"))
        res = run_command(self._argv(n_states, workdir), cwd=workdir, timeout_s=timeout_s,
                          log_path=os.path.join(workdir, "dihmm.log"))
        if res.returncode != 0:
            return self.failed(f"dihmm_run.py exited {res.returncode}"
                               + (" (timed out)" if res.timed_out else ""),
                               command=res.command, workdir=workdir, runtime_s=res.runtime_s)
        try:
            states, domains, found = read_dihmm_outputs(ds, os.path.join(workdir, "out"))
        except (OSError, ValueError) as exc:
            return self.failed(f"could not read output: {exc}", command=res.command,
                               workdir=workdir, runtime_s=res.runtime_s)
        if found == 0:
            return self.failed("no *_dihmm.tsv written", command=res.command,
                               workdir=workdir, runtime_s=res.runtime_s)
        biv, act = posthoc_scores(ds, states, None, n_states)
        return ToolRun(
            tool=self.key, display_name=self.display_name, status="ok", states=states,
            posterior=None, n_states=n_states, bivalency_score=biv, activity_score=act,
            runtime_s=res.runtime_s, peak_rss_mb=res.peak_rss_mb, version=avail.version,
            command=res.command, workdir=workdir, verified=self.verified,
            notes=(f"bin level scored; {n_states} domain states x "
                   f"{os.environ.get('DIHMM_DOMAIN_SIZE', '20')}-bin domains kept in meta; "
                   "Viterbi labels, no posterior; serial build"),
            meta={"domain_states": domains},
        )


def build() -> Sequence[ToolAdapter]:
    return [DiHMMAdapter()]
