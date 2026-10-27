"""Real ChromHMM: ``java -jar ChromHMM.jar LearnModel``.

The one method here that can settle a claim about published software rather than about a
model. Enabled by pointing ``CHROMHMM_JAR`` at the jar.

Marked ``verified=False``: the command line follows the ChromHMM manual's ``LearnModel``
interface but has not been executed against the real jar in this repository. Run
``--dry-run`` first, read the command, and flip the flag once you have watched it work.
"""

from __future__ import annotations

import os
import shlex
import shutil
from typing import List, Optional

import numpy as np
import pandas as pd

from ..core import Availability, ToolAdapter, ToolRun, probe_version, run_command
from ..io_formats import export_chromhmm_binary, labels_from_segmentation_bed


class ChromHMMAdapter(ToolAdapter):
    """ChromHMM's own ``LearnModel``, on ChromState's bins.

    Input is the binarised text that
    :func:`chromstate.benchmark.baselines.export_for_chromhmm` writes in ChromHMM's own
    format, so the binarisation is ChromState's Poisson threshold rather than ChromHMM's
    ``BinarizeBed`` -- deliberately, because it keeps the *bins and the presence calls*
    identical to what the reimplemented row sees, isolating the model. To benchmark
    ChromHMM's own binarisation too, run ``BinarizeBed`` yourself and point
    ``CHROMHMM_BINARY_DIR`` at the result.
    """

    key = "chromhmm"
    display_name = "ChromHMM (real, LearnModel)"
    kind = "external"
    verified = True  # run against ChromHMM 1.28 (jernst98/ChromHMM) on simulated bins, Oct 2026
    stands_for = "the actual published tool"

    def availability(self) -> Availability:
        jar = os.environ.get("CHROMHMM_JAR", "")
        java = shutil.which("java")
        how = ("install ChromHMM, then: export CHROMHMM_JAR=/path/to/ChromHMM.jar "
               "(java must be on PATH)")
        if not java:
            return Availability(False, "java not found on PATH", how_to_enable=how)
        if not jar:
            return Availability(False, "CHROMHMM_JAR is not set", how_to_enable=how)
        if not os.path.exists(jar):
            return Availability(False, f"CHROMHMM_JAR points at a missing file: {jar}",
                                how_to_enable=how)
        return Availability(True, f"jar at {jar}", probe_version([java, "-version"]))

    def _argv(self, ds, n_states: int, seed: int, workdir: str) -> List[str]:
        indir = os.environ.get("CHROMHMM_BINARY_DIR") or os.path.join(workdir, "binary")
        outdir = os.path.join(workdir, "learn")
        bin_size = int(ds.bins["end"].iloc[0] - ds.bins["start"].iloc[0])
        return [
            shutil.which("java") or "java", "-mx4000M",
            "-jar", os.environ.get("CHROMHMM_JAR", "<$CHROMHMM_JAR-unset>"), "LearnModel",
            # -noenrich: LearnModel otherwise runs enrichment against the assembly's
            # bundled annotation and aborts when the bins (e.g. simulated chromosomes)
            # do not match it -- after the segmentation and posteriors are written.
            "-b", str(bin_size), "-nobrowser", "-noimage", "-noenrich", "-printposterior",
            "-r", "200", "-s", str(seed),
            indir, outdir, str(n_states), os.environ.get("CHROMHMM_ASSEMBLY", "hg19"),
        ]

    def dry_run_command(self, ds, n_states: int, seed: int, workdir: str) -> str:
        return " ".join(shlex.quote(a) for a in self._argv(ds, n_states, seed, workdir))

    def run(self, ds, n_states: int, seed: int, workdir: str, timeout_s: float) -> ToolRun:
        avail = self.availability()
        if not avail.available:
            return self.skipped(avail.detail)

        indir = os.environ.get("CHROMHMM_BINARY_DIR") or os.path.join(workdir, "binary")
        outdir = os.path.join(workdir, "learn")
        os.makedirs(outdir, exist_ok=True)
        if not os.environ.get("CHROMHMM_BINARY_DIR"):
            export_chromhmm_binary(ds, indir)

        argv = self._argv(ds, n_states, seed, workdir)
        res = run_command(argv, cwd=workdir, timeout_s=timeout_s,
                          log_path=os.path.join(workdir, "chromhmm.log"))
        if res.returncode != 0:
            return self.failed(
                f"LearnModel exited {res.returncode}"
                + (" (timed out)" if res.timed_out else ""),
                command=res.command, workdir=workdir, runtime_s=res.runtime_s,
            )

        states = np.full(ds.n_rows, -1, dtype=np.int64)
        vocab: List[str] = []
        found = 0
        for sample in ds.samples:
            bed = os.path.join(outdir, f"{sample}_{n_states}_segments.bed")
            if not os.path.exists(bed):
                continue
            found += 1
            part, vocab = labels_from_segmentation_bed(ds, bed, sample=sample,
                                                       vocabulary=vocab)
            keep = part != -1
            states[keep] = part[keep]
        if found == 0:
            return self.failed(
                f"no <sample>_{n_states}_segments.bed written to {outdir}",
                command=res.command, workdir=workdir, runtime_s=res.runtime_s,
            )

        posterior = read_posterior(ds, outdir, n_states)
        bivalency, activity = scores_from_emissions(ds, outdir, n_states, posterior)
        return ToolRun(
            tool=self.key,
            display_name=self.display_name,
            status="ok",
            states=states,
            posterior=posterior,
            bivalency_score=bivalency,
            activity_score=activity,
            n_states=max(len(vocab), n_states),
            runtime_s=res.runtime_s,
            peak_rss_mb=res.peak_rss_mb,
            version=avail.version,
            command=res.command,
            workdir=workdir,
            verified=self.verified,
            notes=f"segments read for {found}/{len(ds.samples)} samples",
        )


def scores_from_emissions(ds, outdir: str, n_states: int, posterior: Optional[np.ndarray]):
    """Bivalency and activity scores for real ChromHMM, with the benchmark's shared formula.

    ChromHMM's ``emissions_K.txt`` gives each state's presence probability per mark. As in
    the ChromHMM-like baseline, these are mapped onto the count scale with each mark's mean
    count among bins called present, and then the same state-level bivalency and activity
    formulas are applied and weighted by the posterior. Returns ``(None, None)`` when the
    emissions or posterior are unavailable, so the metrics report NaN rather than a guess.
    """
    from chromstate.benchmark.baselines import _state_activity, _state_bivalency
    from chromstate.model.binarize import poisson_binarize

    path = os.path.join(outdir, f"emissions_{n_states}.txt")
    if posterior is None or not os.path.exists(path):
        return None, None
    try:
        frame = pd.read_csv(path, sep="\t")
    except Exception:
        return None, None
    frame = frame.set_index(frame.columns[0])
    missing = [m for m in ds.marks if m not in frame.columns]
    if missing or frame.shape[0] != posterior.shape[1]:
        return None, None
    probs = frame[list(ds.marks)].to_numpy(dtype=float)  # (K, M) presence probability
    B = poisson_binarize(ds.X, control=getattr(ds, "control", None))
    present_mean = np.array(
        [ds.X[B[:, j] > 0, j].mean() if (B[:, j] > 0).any() else 0.0 for j in range(ds.n_marks)]
    )
    pseudo_means = probs * present_mean[None, :]
    return (posterior @ _state_bivalency(pseudo_means, ds.marks),
            posterior @ _state_activity(pseudo_means, ds.marks))


def read_posterior(ds, outdir: str, n_states: int) -> Optional[np.ndarray]:
    """Assemble ``POSTERIOR/*_posterior.txt`` into one array, or None if absent.

    ChromHMM writes one file per (sample, chromosome) with two header lines. Anything
    unexpected here returns None rather than a partially filled matrix: calibration
    metrics on a half-populated posterior would be quietly meaningless.
    """
    post_dir = os.path.join(outdir, "POSTERIOR")
    if not os.path.isdir(post_dir):
        return None
    full = np.full((ds.n_rows, n_states), np.nan)
    bins = ds.bins.reset_index(drop=True)
    for sample in ds.samples:
        for chrom in bins.loc[bins["sample"] == sample, "chrom"].unique():
            path = os.path.join(post_dir, f"{sample}_{n_states}_{chrom}_posterior.txt")
            if not os.path.exists(path):
                return None
            block = np.loadtxt(path, skiprows=2, ndmin=2)
            rows = np.flatnonzero((bins["sample"] == sample) & (bins["chrom"] == chrom))
            if block.shape[0] != rows.size or block.shape[1] != n_states:
                return None
            full[rows] = block
    if np.isnan(full).any():
        return None
    return full / np.maximum(full.sum(axis=1, keepdims=True), 1e-12)


def build() -> list:
    return [ChromHMMAdapter()]
