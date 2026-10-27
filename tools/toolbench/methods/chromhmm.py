"""Real ChromHMM: ``java -jar ChromHMM.jar LearnModel``.

The one method here that can settle a claim about published software rather than about a
model. Enabled by pointing ``CHROMHMM_JAR`` at the jar.

**Verified** against ChromHMM 1.28 (the official ``ChromHMM.zip`` release, OpenJDK 17) on
simulated data. What running the real jar changed relative to the manual-transcribed
command:

* ``-noenrich`` is required. Without it ``LearnModel`` finishes training, writes every
  segmentation and posterior, and *then* exits 1 because its post-hoc overlap enrichment
  looks up the assembly's annotation files (``COORDS/hg19/*.bed.gz``) and finds no
  coordinate on ChromState's chromosomes. On simulated chromosomes, or any bin grid that
  is not a real assembly, the enrichment is meaningless anyway; the assembly argument is
  still required positionally but is otherwise unused with ``-noenrich``.
* Java is taken from ``$JAVA_HOME/bin/java`` when set: conda's ``openjdk`` package puts
  the binary under ``$CONDA_PREFIX/lib/jvm/bin`` and relies on ``JAVA_HOME``, so
  ``java`` is often not on ``PATH`` even when a JDK is installed.
* Output names match the manual: ``<sample>_<K>_segments.bed`` (BED4, labels ``E1..EK``)
  and ``POSTERIOR/<sample>_<K>_<chrom>_posterior.txt`` (two header lines, one row per
  bin, columns ``E1..EK``, four decimals).

ChromHMM trains on all available cores by default (``-p 0`` means "up to one thread per
input file"); set ``CHROMHMM_PROCESSORS`` to cap it.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
from typing import List, Optional

import numpy as np

from ..core import Availability, ToolAdapter, ToolRun, probe_version, run_command
from ..io_formats import export_chromhmm_binary, labels_from_segmentation_bed
from ..scores import posthoc_scores


def java_binary() -> Optional[str]:
    """``$JAVA_HOME/bin/java`` if it exists, else whatever ``java`` is on PATH.

    macOS ships a ``/usr/bin/java`` stub that only prints "install Java" and exits 1, so a
    bare ``which java`` is not proof of a runtime; :meth:`ChromHMMAdapter.availability`
    therefore also requires the jar to answer ``Version``.
    """
    home = os.environ.get("JAVA_HOME", "")
    if home and os.path.exists(os.path.join(home, "bin", "java")):
        return os.path.join(home, "bin", "java")
    return shutil.which("java")


class ChromHMMAdapter(ToolAdapter):
    """ChromHMM's own ``LearnModel``, on ChromState's bins.

    Input is the binarised text that
    :func:`chromstate.benchmark.baselines.export_for_chromhmm` writes in ChromHMM's own
    format, so the binarisation is ChromState's Poisson threshold rather than ChromHMM's
    ``BinarizeBed`` -- deliberately, because it keeps the *bins and the presence calls*
    identical to what the reimplemented row sees, isolating the model. To benchmark
    ChromHMM's own binarisation too, run ``BinarizeBed`` yourself and point
    ``CHROMHMM_BINARY_DIR`` at the result.

    All samples are learned jointly (one model, one shared state vocabulary), which is
    ChromHMM's standard "concatenated" design and matches ChromState's joint fit.
    """

    key = "chromhmm"
    display_name = "ChromHMM (real, LearnModel)"
    kind = "external"
    verified = True
    stands_for = "the actual published tool"

    def availability(self) -> Availability:
        jar = os.environ.get("CHROMHMM_JAR", "")
        java = java_binary()
        how = ("install ChromHMM, then: export CHROMHMM_JAR=/path/to/ChromHMM.jar "
               "(and JAVA_HOME, or java on PATH)")
        if not java:
            return Availability(False, "java not found (JAVA_HOME unset, none on PATH)",
                                how_to_enable=how)
        if not jar:
            return Availability(False, "CHROMHMM_JAR is not set", how_to_enable=how)
        if not os.path.exists(jar):
            return Availability(False, f"CHROMHMM_JAR points at a missing file: {jar}",
                                how_to_enable=how)
        version = probe_version([java, "-jar", jar, "Version"])
        if "ChromHMM" not in version:
            return Availability(False, f"{java} could not run the jar ({version or 'no output'})",
                                how_to_enable=how)
        match = re.search(r"Version\s+([\w.]+)", version)
        return Availability(True, f"jar at {jar}",
                            f"ChromHMM {match.group(1)}" if match else version)

    def _argv(self, ds, n_states: int, seed: int, workdir: str) -> List[str]:
        indir = os.environ.get("CHROMHMM_BINARY_DIR") or os.path.join(workdir, "binary")
        outdir = os.path.join(workdir, "learn")
        bin_size = int(ds.bins["end"].iloc[0] - ds.bins["start"].iloc[0])
        argv = [
            java_binary() or "java", "-mx4000M",
            "-jar", os.environ.get("CHROMHMM_JAR", "<$CHROMHMM_JAR-unset>"), "LearnModel",
            "-b", str(bin_size), "-nobrowser", "-noimage", "-noenrich", "-printposterior",
            "-r", "200", "-s", str(seed),
        ]
        if os.environ.get("CHROMHMM_PROCESSORS"):
            argv += ["-p", os.environ["CHROMHMM_PROCESSORS"]]
        return argv + [indir, outdir, str(n_states),
                       os.environ.get("CHROMHMM_ASSEMBLY", "hg19")]

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

        # Fix the vocabulary to E1..EK so state k is posterior column k; first-appearance
        # order would scramble that correspondence.
        vocab: List[str] = [f"E{k + 1}" for k in range(n_states)]
        states = np.full(ds.n_rows, -1, dtype=np.int64)
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
        biv, act = posthoc_scores(ds, states, posterior, n_states)
        return ToolRun(
            tool=self.key,
            display_name=self.display_name,
            status="ok",
            states=states,
            posterior=posterior,
            n_states=max(len(vocab), n_states),
            bivalency_score=biv,
            activity_score=act,
            runtime_s=res.runtime_s,
            peak_rss_mb=res.peak_rss_mb,
            version=avail.version,
            command=res.command,
            workdir=workdir,
            verified=self.verified,
            notes=(f"segments read for {found}/{len(ds.samples)} samples; bivalency and "
                   "activity scored post hoc from the posterior (toolbench.scores)"),
        )


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
            # Rows of a posterior file follow bin order within the chromosome; ``rows`` is
            # in dataset order, which simulate/io keep sorted by start -- sort to be sure.
            rows = rows[np.argsort(bins.loc[rows, "start"].to_numpy(), kind="stable")]
            full[rows] = block
    if np.isnan(full).any():
        return None
    return full / np.maximum(full.sum(axis=1, keepdims=True), 1e-12)


def build() -> list:
    return [ChromHMMAdapter()]
