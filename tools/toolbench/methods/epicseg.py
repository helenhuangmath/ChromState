"""Real EpiCSeg (Mammana & Chung 2015): ``epicseg.R normalizecounts`` then ``segment``.

EpiCSeg is the closest published relative of ChromState's emission model: a hidden
Markov model whose emissions are *negative multinomial* -- one negative binomial for the
bin's total count across marks, multinomial for how that total splits between marks --
fitted to raw read counts. It ships an ``Rscript`` command line with ``getcounts`` (BAM ->
count matrix), ``normalizecounts`` (make several datasets comparable) and ``segment``.

The adapter replaces only ``getcounts``: it writes ChromState's bins in exactly the text
format ``getcounts`` produces (tab-separated, a header of mark names, one integer row per
bin) plus the BED of contiguous regions those bins tile. Then it runs EpiCSeg's own
documented workflow for several datasets: ``normalizecounts`` on all samples, then one
``segment`` call with every sample as a labelled ``--counts label:path``, which trains a
single model jointly and writes ``segmentation_<label>.bed`` per sample -- the same joint
design ChromState and ChromHMM get here.

**Verified** against EpiCSeg 1.0 (GitHub ``lamortenera/epicseg`` HEAD, with ``kfoots``
1.0) on R 4.5.3. Installing it on current R needs one build fix, recorded in
``tools/setup/epicseg_fcone.patch``: two BLAS calls lack the ``FCONE`` length arguments
R >= 4.3 requires for Fortran character arguments. Nothing in the model changes.

What the real tool told us that the documentation did not:

* Posteriors are not written as text. ``--save_rdata`` keeps them in
  ``rdata.Rdata`` (``nstates x nbins x ndatasets``); ``r/epicseg_posteriors.R`` exports
  them. That export step is excluded from the runtime, which covers only
  ``normalizecounts`` + ``segment``.
* ``segment`` always also renders its HTML report (PNG heatmaps etc.); that cost is part
  of the published command and is included in the runtime.
* There is no seed option; initialisation is deterministic.
* State numbers in the BED and posterior columns agree (both follow EpiCSeg's own state
  reordering), so state ``k`` is posterior column ``k``.

Environment: ``EPICSEG_R`` is the launcher ``epicseg:::getLauncher()`` writes; it is run
with the Rscript named in its own shebang line, so a system R on PATH without EpiCSeg
does not get used by accident. ``EPICSEG_NTHREADS`` (default: all cores) sets
``--nthreads``.
"""

from __future__ import annotations

import os
import shlex
import shutil
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core import UNASSIGNED, Availability, ToolAdapter, ToolRun, probe_version, run_command
from ..io_formats import labels_from_segmentation_bed
from ..scores import posthoc_scores

R_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "r")


def launcher_interpreter(launcher: str) -> str:
    """The Rscript a ``getLauncher()`` script names in its shebang, else ``Rscript``."""
    try:
        with open(launcher) as fh:
            first = fh.readline().strip()
    except OSError:
        return "Rscript"
    if first.startswith("#!"):
        path = first[2:].strip().split()[0]
        if os.path.exists(path):
            return path
    return shutil.which("Rscript") or "Rscript"


def sample_order(ds, sample: str) -> np.ndarray:
    """Dataset row indices of one sample, sorted by chromosome (first appearance) and start.

    This is the bin order of every file the adapter writes for that sample, so it is also
    the order in which posterior rows come back.
    """
    bins = ds.bins.reset_index(drop=True)
    rows = np.flatnonzero((bins["sample"] == sample).to_numpy())
    sub = bins.loc[rows]
    chrom_rank = {c: i for i, c in enumerate(dict.fromkeys(sub["chrom"]))}
    key = np.lexsort((sub["start"].to_numpy(), sub["chrom"].map(chrom_rank).to_numpy()))
    return rows[key]


def contiguous_regions(ds, rows: np.ndarray) -> List[Tuple[str, int, int]]:
    """BED regions tiled by the given (ordered) bins, split at gaps and chromosome ends.

    EpiCSeg derives the bin size from region widths and models transitions only inside a
    region, so a gap in the bin grid must start a new region rather than be bridged.
    """
    bins = ds.bins.reset_index(drop=True)
    chrom = bins["chrom"].to_numpy()[rows]
    start = bins["start"].to_numpy()[rows]
    end = bins["end"].to_numpy()[rows]
    regions: List[Tuple[str, int, int]] = []
    r0 = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or chrom[i] != chrom[i - 1] or start[i] != end[i - 1]:
            regions.append((str(chrom[r0]), int(start[r0]), int(end[i - 1])))
            r0 = i
    return regions


def write_epicseg_inputs(ds, workdir: str) -> Dict[str, object]:
    """Write one ``getcounts``-format matrix per sample plus the shared regions BED.

    Raises ``ValueError`` if samples do not share an identical bin grid: EpiCSeg's joint
    mode requires "count matrices ... on the same genomic regions".
    """
    os.makedirs(workdir, exist_ok=True)
    bins = ds.bins.reset_index(drop=True)
    orders = {s: sample_order(ds, s) for s in ds.samples}
    ref = orders[ds.samples[0]]
    ref_coords = bins.loc[ref, ["chrom", "start", "end"]].to_numpy()
    for s, rows in orders.items():
        if len(rows) != len(ref) or not (bins.loc[rows, ["chrom", "start", "end"]].to_numpy()
                                         == ref_coords).all():
            raise ValueError(f"sample {s!r} does not share the first sample's bin grid")

    regions_path = os.path.join(workdir, "regions.bed")
    with open(regions_path, "w") as fh:
        for chrom, start, end in contiguous_regions(ds, ref):
            fh.write(f"{chrom}\t{start}\t{end}\n")

    counts: Dict[str, str] = {}
    X = np.rint(ds.X).astype(np.int64)
    for s, rows in orders.items():
        path = os.path.join(workdir, f"counts_{_safe(s)}.txt")
        np.savetxt(path, X[rows], fmt="%d", delimiter="\t", header="\t".join(ds.marks),
                   comments="")
        counts[s] = path
    return {"regions": regions_path, "counts": counts, "orders": orders}


def _safe(name: str) -> str:
    """EpiCSeg uses --counts labels in file names and splits label:path on ':'."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))


def read_epicseg_outputs(ds, outdir: str, orders: Dict[str, np.ndarray], n_states: int
                         ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """Map EpiCSeg's per-sample BEDs (by coordinate) and posteriors (by written order).

    Single-sample runs write ``segmentation.bed`` / ``posterior.tsv``; multi-sample runs
    ``segmentation_<label>.bed`` / ``posterior_<label>.tsv``. A posterior file whose row
    count does not match the bins written for that sample is an error, not a truncation.
    Returns ``(states, posterior_or_None, vocabulary)``.
    """
    vocab = [str(k + 1) for k in range(n_states)]
    states = np.full(ds.n_rows, UNASSIGNED, dtype=np.int64)
    post = np.full((ds.n_rows, n_states), np.nan)
    single = len(orders) == 1
    for s, rows in orders.items():
        tag = "" if single else f"_{_safe(s)}"
        bed = os.path.join(outdir, f"segmentation{tag}.bed")
        part, vocab = labels_from_segmentation_bed(ds, bed, sample=s, vocabulary=vocab)
        keep = part != UNASSIGNED
        states[keep] = part[keep]
        ppath = os.path.join(outdir, f"posterior{tag}.tsv")
        if os.path.exists(ppath):
            P = np.loadtxt(ppath, ndmin=2)
            if P.shape != (len(rows), n_states):
                raise ValueError(f"{ppath} has shape {P.shape}, expected "
                                 f"{(len(rows), n_states)}")
            post[rows] = P
    if np.isnan(post).any():
        return states, None, vocab
    return states, post / np.maximum(post.sum(axis=1, keepdims=True), 1e-12), vocab


class EpiCSegAdapter(ToolAdapter):
    key = "epicseg"
    display_name = "EpiCSeg (real, negative multinomial hidden Markov model)"
    kind = "external"
    verified = True
    stands_for = "published count-based segmentation; closest prior art to ChromState's emissions"

    def availability(self) -> Availability:
        how = ("install kfoots + EpiCSeg (see tools/README.md), write the launcher with "
               "epicseg:::getLauncher('epicseg.R'), then: export EPICSEG_R=/path/to/epicseg.R")
        launcher = os.environ.get("EPICSEG_R", "")
        if not launcher:
            return Availability(False, "EPICSEG_R is not set", how_to_enable=how)
        if not os.path.exists(launcher):
            return Availability(False, f"EPICSEG_R points at a missing file: {launcher}",
                                how_to_enable=how)
        rscript = launcher_interpreter(launcher)
        version = probe_version([rscript, "-e",
                                 "cat(as.character(packageVersion('epicseg')))"])
        if not version or "rror" in version:
            return Availability(False, f"{rscript} cannot load epicseg ({version})",
                                how_to_enable=how)
        return Availability(True, f"launcher {launcher} via {rscript}", f"epicseg {version}")

    def _nthreads(self) -> str:
        return os.environ.get("EPICSEG_NTHREADS", str(os.cpu_count() or 1))

    def _commands(self, ds, n_states: int, workdir: str,
                  counts: Optional[Dict[str, str]] = None) -> List[List[str]]:
        launcher = os.environ.get("EPICSEG_R", "<$EPICSEG_R-unset>")
        rscript = launcher_interpreter(launcher)
        counts = counts or {s: os.path.join(workdir, f"counts_{_safe(s)}.txt")
                            for s in ds.samples}
        outdir = os.path.join(workdir, "out")
        cmds: List[List[str]] = []
        if len(counts) > 1:
            norm = [rscript, launcher, "normalizecounts", "--nthreads", self._nthreads()]
            for path in counts.values():
                norm += ["--counts", path]
            cmds.append(norm)
            seg_counts = [f"{_safe(s)}:{p[:-4]}_norm.txt" for s, p in counts.items()]
        else:
            seg_counts = list(counts.values())
        seg = [rscript, launcher, "segment", "--regions", os.path.join(workdir, "regions.bed"),
               "--nstates", str(n_states), "--outdir", outdir, "--nthreads", self._nthreads(),
               "--save_rdata"]
        for c in seg_counts:
            seg += ["--counts", c]
        cmds.append(seg)
        return cmds

    def dry_run_command(self, ds, n_states: int, seed: int, workdir: str) -> str:
        return " && ".join(" ".join(shlex.quote(a) for a in c)
                           for c in self._commands(ds, n_states, workdir))

    def run(self, ds, n_states: int, seed: int, workdir: str, timeout_s: float) -> ToolRun:
        avail = self.availability()
        if not avail.available:
            return self.skipped(avail.detail)
        try:
            inputs = write_epicseg_inputs(ds, workdir)
        except ValueError as exc:
            return self.failed(str(exc))
        outdir = os.path.join(workdir, "out")
        os.makedirs(outdir, exist_ok=True)

        runtime, peak, shown = 0.0, None, []
        for i, argv in enumerate(self._commands(ds, n_states, workdir, inputs["counts"])):
            res = run_command(argv, cwd=workdir, timeout_s=max(timeout_s - runtime, 1.0),
                              log_path=os.path.join(workdir, f"epicseg_{i}.log"))
            runtime += res.runtime_s
            peak = max(filter(None, [peak, res.peak_rss_mb]), default=None)
            shown.append(res.command)
            if res.returncode != 0:
                return self.failed(
                    f"{argv[2]} exited {res.returncode}" + (" (timed out)" if res.timed_out else ""),
                    command=" && ".join(shown), workdir=workdir, runtime_s=runtime)

        launcher = os.environ["EPICSEG_R"]
        exp = run_command([launcher_interpreter(launcher),
                           os.path.join(R_DIR, "epicseg_posteriors.R"),
                           os.path.join(outdir, "rdata.Rdata"), outdir],
                          cwd=workdir, timeout_s=600,
                          log_path=os.path.join(workdir, "epicseg_posteriors.log"))
        try:
            states, posterior, vocab = read_epicseg_outputs(ds, outdir, inputs["orders"],
                                                            n_states)
        except (OSError, ValueError) as exc:
            return self.failed(f"could not read output: {exc}", command=" && ".join(shown),
                               workdir=workdir, runtime_s=runtime)
        biv, act = posthoc_scores(ds, states, posterior, n_states)
        note = ("normalizecounts + joint segment over all samples; posteriors from "
                "--save_rdata" + ("" if exp.returncode == 0 and posterior is not None
                                  else " (posterior export failed)"))
        return ToolRun(
            tool=self.key, display_name=self.display_name, status="ok", states=states,
            posterior=posterior, n_states=max(len(vocab), n_states),
            bivalency_score=biv, activity_score=act, runtime_s=runtime, peak_rss_mb=peak,
            version=avail.version, command=" && ".join(shown), workdir=workdir,
            verified=self.verified, notes=note,
        )


def build() -> Sequence[ToolAdapter]:
    return [EpiCSegAdapter()]
