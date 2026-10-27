"""Real normR (Helmuth et al. 2016): per-mark enrichment calls, scored where comparable.

normR is **not a multi-mark segmenter**. ``enrichR`` fits, for one ChIP track against its
input, a two-component binomial mixture on each bin's (ChIP, input) counts and returns a
posterior of enrichment, a q-value and a class. ``diffR`` does the same between two
conditions and ``regimeR`` with more than two components. So this row is not "normR
segments the genome"; it is the two things normR can honestly be compared on here:

1. **Combinatorial patterns.** Every mark is called enriched or not at FDR 0.05, and each
   bin's on/off vector across marks becomes its label -- the ChromHMM-style "binary
   vector -> state" construction, with normR as the binariser and *no* model of how
   marks co-occur or of spatial dependence. The number of labels is however many
   patterns occur (up to 2^marks), not the requested state count. Label-agreement
   metrics therefore need care: the adjusted Rand index is comparable across rows, but
   matched accuracy maps every surplus pattern to its modal true state and so flatters
   this row.
2. **Bivalency.** The bivalency score is the product of normR's H3K4me3 and H3K27me3
   enrichment posteriors -- the probability both marks are enriched if the two fits are
   treated as independent -- and is scored with the same precision-recall metric as
   every other row. The activity score is normR's standardised enrichment averaged
   over the active marks present (H3K4me3, H3K4me1, H3K27ac).

``diffR`` is the natural normR comparator for ChromState's differential analysis
(condition A vs B per mark); the harness's metrics are segmentation metrics, so it is
noted in ``docs/tool_comparison.md`` rather than run here.

Control: ChromState's simulator has no input track, so ``r/normr_run.R`` draws a seeded
Poisson pseudo-input at the mark's mean depth unless a real control is supplied (see the
script's header for why a flat control does not work).

**Verified** against normR 1.36.0 (Bioconductor) on R 4.5.3. Environment:
``NORMR_RSCRIPT`` must name an ``Rscript`` that can load ``normr`` (explicit, so a system R
without it is never picked up by accident); ``NORMR_PROCS`` sets normR's ``procs``
(default: all cores).
"""

from __future__ import annotations

import os
import shlex
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from chromstate.model.annotate import canonical_mark

from ..core import UNASSIGNED, Availability, ToolAdapter, ToolRun, probe_version, run_command
from ..io_formats import export_count_matrix

R_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "r",
                        "normr_run.R")
ACTIVE = ("K4me3", "K4me1", "K27ac")


def read_normr_table(ds, path: str, sample: str) -> pd.DataFrame:
    """One ``normr_<sample>.tsv`` aligned to the dataset's rows for that sample.

    Aligned by (chrom, start), never by row order. Returns a frame indexed by dataset row
    position; bins the file does not mention are absent from the index.
    """
    table = pd.read_csv(path, sep="\t", dtype={"chrom": str})
    bins = ds.bins.reset_index(drop=True)
    mine = bins.loc[bins["sample"] == sample, ["chrom", "start"]].copy()
    mine["row"] = mine.index
    mine["chrom"] = mine["chrom"].astype(str)
    merged = mine.merge(table, on=["chrom", "start"], how="inner", validate="one_to_one")
    return merged.set_index("row").sort_index()


def patterns_to_states(calls: np.ndarray, marks: Sequence[str]
                       ) -> Tuple[np.ndarray, List[str]]:
    """Binary calls (bins x marks) -> integer label per distinct on/off pattern.

    Labels are numbered by decreasing frequency (ties by pattern), and named by the marks
    that are on, e.g. ``"H3K4me3+H3K27me3"``; the all-off pattern is ``"none"``.
    """
    calls = np.asarray(calls, dtype=bool)
    if calls.size == 0:
        return np.zeros(0, dtype=np.int64), []
    keys = calls.astype(np.int64) @ (1 << np.arange(calls.shape[1], dtype=np.int64))
    uniq, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    order = np.lexsort((uniq, -counts))
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    names = []
    for k in uniq[order]:
        on = [m for j, m in enumerate(marks) if (int(k) >> j) & 1]
        names.append("+".join(on) if on else "none")
    return rank[inverse].astype(np.int64), names


def scores_from_tables(ds, tables: Dict[str, pd.DataFrame]):
    """Build states, bivalency and activity arrays on the full bin grid."""
    marks = list(ds.marks)
    canon = [canonical_mark(m) for m in marks]
    n = ds.n_rows
    calls = np.zeros((n, len(marks)), dtype=bool)
    seen = np.zeros(n, dtype=bool)
    biv = np.zeros(n)
    act = np.zeros(n)
    for _, tab in tables.items():
        rows = tab.index.to_numpy()
        seen[rows] = True
        for j, m in enumerate(marks):
            calls[rows, j] = tab[f"{m}_class"].to_numpy() == 1
        if "K4me3" in canon and "K27me3" in canon:
            a = tab[f"{marks[canon.index('K4me3')]}_post"].to_numpy(dtype=float)
            r = tab[f"{marks[canon.index('K27me3')]}_post"].to_numpy(dtype=float)
            biv[rows] = np.nan_to_num(a) * np.nan_to_num(r)
        active = [marks[j] for j, c in enumerate(canon) if c in ACTIVE]
        if active:
            enr = np.column_stack([tab[f"{m}_enr"].to_numpy(dtype=float) for m in active])
            act[rows] = np.nan_to_num(enr).mean(axis=1)
    states = np.full(n, UNASSIGNED, dtype=np.int64)
    part, names = patterns_to_states(calls[seen], marks)
    states[seen] = part
    return states, names, biv, act


class NormRAdapter(ToolAdapter):
    key = "normr"
    display_name = "normR (real, per-mark enrichment; combinatorial patterns)"
    kind = "external"
    verified = True
    stands_for = "single-mark enrichment caller; patterns and bivalency only, not a segmenter"

    def _rscript(self) -> str:
        return os.environ.get("NORMR_RSCRIPT", "")

    def availability(self) -> Availability:
        how = ("install Bioconductor normr into an R library (see tools/README.md), then: "
               "export NORMR_RSCRIPT=/path/to/that/Rscript")
        rscript = self._rscript()
        if not rscript:
            return Availability(False, "NORMR_RSCRIPT is not set", how_to_enable=how)
        if not os.path.exists(rscript):
            return Availability(False, f"NORMR_RSCRIPT points at a missing file: {rscript}",
                                how_to_enable=how)
        version = probe_version([rscript, "-e", "cat(as.character(packageVersion('normr')))"])
        if not version or "rror" in version:
            return Availability(False, f"{rscript} cannot load normr ({version})",
                                how_to_enable=how)
        return Availability(True, f"normr via {rscript}", f"normr {version}")

    def _argv(self, workdir: str, seed: int) -> List[str]:
        return [self._rscript() or "<$NORMR_RSCRIPT-unset>", R_SCRIPT,
                os.path.join(workdir, "counts.tsv"), os.path.join(workdir, "out"),
                os.environ.get("NORMR_PROCS", str(os.cpu_count() or 1)), str(seed)]

    def dry_run_command(self, ds, n_states: int, seed: int, workdir: str) -> str:
        return " ".join(shlex.quote(a) for a in self._argv(workdir, seed))

    def run(self, ds, n_states: int, seed: int, workdir: str, timeout_s: float) -> ToolRun:
        avail = self.availability()
        if not avail.available:
            return self.skipped(avail.detail)
        export_count_matrix(ds, os.path.join(workdir, "counts.tsv"))
        argv = self._argv(workdir, seed)
        res = run_command(argv, cwd=workdir, timeout_s=timeout_s,
                          log_path=os.path.join(workdir, "normr.log"))
        if res.returncode != 0:
            return self.failed(f"normr_run.R exited {res.returncode}"
                               + (" (timed out)" if res.timed_out else ""),
                               command=res.command, workdir=workdir, runtime_s=res.runtime_s)
        try:
            tables = {s: read_normr_table(ds, os.path.join(workdir, "out", f"normr_{s}.tsv"), s)
                      for s in ds.samples}
            states, names, biv, act = scores_from_tables(ds, tables)
        except (OSError, ValueError, KeyError) as exc:
            return self.failed(f"could not read output: {exc}", command=res.command,
                               workdir=workdir, runtime_s=res.runtime_s)
        return ToolRun(
            tool=self.key, display_name=self.display_name, status="ok", states=states,
            posterior=None, n_states=len(names), bivalency_score=biv, activity_score=act,
            runtime_s=res.runtime_s, peak_rss_mb=res.peak_rss_mb, version=avail.version,
            command=res.command, workdir=workdir, verified=self.verified,
            notes=(f"{len(names)} on/off patterns at FDR 0.05 (not a segmentation); "
                   "Poisson pseudo-input; bivalency = P(K4me3 enr) x P(K27me3 enr)"),
            meta={"pattern_names": names},
        )


def build() -> Sequence[ToolAdapter]:
    return [NormRAdapter()]
