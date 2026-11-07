"""Real Segway: ``genomedata-load`` + ``segway train`` + ``segway annotate``.

Segway (Hoffman et al., 2012) is the other widely used segmenter and, unlike ChromHMM,
models continuous signal; it is the closest published relative of ChromState's model
family (a dynamic Bayesian network over asinh-transformed signal). It needs GMTK, which
is Linux-only, so the macOS runs in ``docs/tool_comparison.md`` could not include it.
On a Linux cluster it installs from bioconda (``conda create -c conda-forge -c bioconda
segway``); point ``SEGWAY_ENV`` at that env, or put ``segway`` and ``genomedata-load``
on ``PATH``.

Design, and where it departs from a stock Segway run:

* **One model for all samples, by pseudo-chromosomes.** Segway trains one archive at a
  time, so every (sample, chromosome) becomes a sequence named ``<chrom>__<sample>`` in
  a single genomedata archive. That gives one shared label vocabulary across samples,
  the same concatenated design ChromHMM and ChromState use, and the labels are mapped
  back onto the real bins by splitting the name again.
* **Signal is ChromState's depth-normalised count** (count over size factor) per bin,
  written as bedGraph at the bin resolution and loaded at ``--resolution`` equal to the
  bin width, so Segway sees exactly the bins every other row sees.
* **Training is capped** (``SEGWAY_MAX_ROUNDS``, default 10) and uses one instance
  (``SEGWAY_INSTANCES``, default 1) so a genome-scale run finishes in hours rather than
  days; Segway's own defaults would run several random restarts to convergence. Anything
  else can be appended through ``SEGWAY_TRAIN_OPTS``.
* **No posterior.** ``segway posterior`` writes one bedGraph per label and roughly
  doubles the runtime; the harness records Segway's hard labels only, so its calibration
  columns read "not measurable".
* Runs locally (``SEGWAY_CLUSTER=local``): the cluster already gave this job its cores.
"""

from __future__ import annotations

import gzip
import os
import shlex
import shutil
from typing import Dict, List, Optional

import numpy as np

from ..core import Availability, ToolAdapter, ToolRun, probe_version, run_command
from ..io_formats import labels_from_segmentation_bed
from ..scores import posthoc_scores

SEP = "__"


def _bin(name: str) -> Optional[str]:
    env = os.environ.get("SEGWAY_ENV", "")
    if env and os.path.exists(os.path.join(env, "bin", name)):
        return os.path.join(env, "bin", name)
    return shutil.which(name)


class SegwayAdapter(ToolAdapter):
    key = "segway"
    display_name = "Segway (real, train + annotate)"
    kind = "external"
    verified = False  # set to True once a run on this machine has been checked
    stands_for = "the other widely used segmenter; continuous signal, no binarisation"

    def availability(self) -> Availability:
        how = ("conda create -p <env> -c conda-forge -c bioconda segway; then "
               "export SEGWAY_ENV=<env>")
        segway, gdl = _bin("segway"), _bin("genomedata-load")
        if not segway:
            return Availability(False, "segway not found (SEGWAY_ENV unset, none on PATH)", how_to_enable=how)
        if not gdl:
            return Availability(False, "genomedata-load not found next to segway", how_to_enable=how)
        version = probe_version([segway, "--version"])
        if not version:
            return Availability(False, f"{segway} --version produced no output", how_to_enable=how)
        gmtk_env = os.environ.get("SEGWAY_GMTK_ENV") or os.environ.get("SEGWAY_ENV", "")
        gmtk = os.path.join(gmtk_env, "bin", "gmtkTriangulate") if gmtk_env else shutil.which("gmtkTriangulate")
        if not gmtk or not os.path.exists(gmtk):
            return Availability(False, "gmtkTriangulate not found (set SEGWAY_GMTK_ENV)", how_to_enable=how)
        if "error" in probe_version([gmtk, "-version"]).lower():
            return Availability(False, f"{gmtk} does not run (library mismatch?)", how_to_enable=how)
        return Availability(True, f"segway at {segway}; gmtk at {gmtk}", version)

    # -- input ---------------------------------------------------------------

    def _write_tracks(self, ds, workdir: str) -> Dict[str, str]:
        """bedGraph per mark over pseudo-chromosomes, plus a sizes file."""
        tracks = os.path.join(workdir, "tracks")
        os.makedirs(tracks, exist_ok=True)
        bins = ds.bins.reset_index(drop=True)
        bin_size = int(bins["end"].iloc[0] - bins["start"].iloc[0])
        Y = ds.X / np.maximum(ds.size_factors, 1e-12)
        pseudo = (bins["chrom"].astype(str) + SEP + bins["sample"].astype(str)).to_numpy()
        order = np.lexsort((bins["start"].to_numpy(), pseudo))
        sizes: Dict[str, int] = {}
        for p, e in zip(pseudo[order], bins["end"].to_numpy()[order]):
            sizes[p] = max(sizes.get(p, 0), int(e))
        sizes_path = os.path.join(tracks, "pseudo.chrom.sizes")
        with open(sizes_path, "w") as fh:
            for p, s in sizes.items():
                fh.write(f"{p}\t{s}\n")
        paths = {}
        starts = bins["start"].to_numpy()[order]
        ends = bins["end"].to_numpy()[order]
        for j, mark in enumerate(ds.marks):
            path = os.path.join(tracks, f"{mark}.bedGraph")
            vals = Y[order, j]
            with open(path, "w") as fh:
                for p, s, e, v in zip(pseudo[order], starts, ends, vals):
                    fh.write(f"{p}\t{s}\t{e}\t{v:.4f}\n")
            paths[mark] = path
        paths["__sizes__"] = sizes_path
        paths["__bin_size__"] = str(bin_size)
        return paths

    def _commands(self, ds, n_states: int, seed: int, workdir: str, paths: Dict[str, str]) -> List[List[str]]:
        archive = os.path.join(workdir, "archive.genomedata")
        traindir = os.path.join(workdir, "traindir")
        identdir = os.path.join(workdir, "identifydir")
        load = [_bin("genomedata-load") or "genomedata-load", "--sizes", "-s", paths["__sizes__"]]
        for mark in ds.marks:
            load += ["-t", f"{mark}={paths[mark]}"]
        load.append(archive)
        train = [_bin("segway") or "segway", "train",
                 f"--num-labels={n_states}", f"--resolution={paths['__bin_size__']}",
                 f"--max-train-rounds={os.environ.get('SEGWAY_MAX_ROUNDS', '10')}",
                 f"--num-instances={os.environ.get('SEGWAY_INSTANCES', '1')}"]
        extra = os.environ.get("SEGWAY_TRAIN_OPTS", "")
        if extra:
            train += shlex.split(extra)
        train += [archive, traindir]
        annotate = [_bin("segway") or "segway", "annotate", archive, traindir, identdir]
        return [load, train, annotate]

    def dry_run_command(self, ds, n_states: int, seed: int, workdir: str) -> str:
        fake = {m: os.path.join(workdir, "tracks", f"{m}.bedGraph") for m in ds.marks}
        fake["__sizes__"] = os.path.join(workdir, "tracks", "pseudo.chrom.sizes")
        fake["__bin_size__"] = str(int(ds.bins["end"].iloc[0] - ds.bins["start"].iloc[0]))
        return " && ".join(" ".join(shlex.quote(a) for a in c)
                           for c in self._commands(ds, n_states, seed, workdir, fake))

    # -- run -------------------------------------------------------------------

    def run(self, ds, n_states: int, seed: int, workdir: str, timeout_s: float) -> ToolRun:
        avail = self.availability()
        if not avail.available:
            return self.skipped(avail.detail)
        os.makedirs(workdir, exist_ok=True)
        for sub in ("archive.genomedata", "traindir", "identifydir"):
            shutil.rmtree(os.path.join(workdir, sub), ignore_errors=True)
        paths = self._write_tracks(ds, workdir)
        env = dict(os.environ)
        env.setdefault("SEGWAY_CLUSTER", "local")
        env.setdefault("SEGWAY_RAND_SEED", str(seed))
        env.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
        # GMTK may live in its own env (bioconda's gmtk links an older HDF5 than
        # genomedata needs); it goes first on PATH so segway finds gmtkTriangulate etc.
        for var in ("SEGWAY_ENV", "SEGWAY_GMTK_ENV"):
            if os.environ.get(var):
                env["PATH"] = os.path.join(os.environ[var], "bin") + os.pathsep + env.get("PATH", "")
        total = 0.0
        peak = None
        commands = []
        remaining = timeout_s
        for name, argv in zip(("load", "train", "annotate"), self._commands(ds, n_states, seed, workdir, paths)):
            res = run_command(argv, cwd=workdir, timeout_s=max(remaining, 1.0), env=env,
                              log_path=os.path.join(workdir, f"segway_{name}.log"))
            total += res.runtime_s
            remaining -= res.runtime_s
            commands.append(res.command)
            if res.peak_rss_mb is not None:
                peak = max(peak or 0.0, res.peak_rss_mb)
            if res.returncode != 0:
                return self.failed(f"segway {name} exited {res.returncode}"
                                   + (" (timed out)" if res.timed_out else "")
                                   + f"; see {os.path.join(workdir, f'segway_{name}.log')}",
                                   command=" && ".join(commands), workdir=workdir, runtime_s=total)

        bed_gz = os.path.join(workdir, "identifydir", "segway.bed.gz")
        if not os.path.exists(bed_gz):
            return self.failed(f"no segway.bed.gz in {os.path.join(workdir, 'identifydir')}",
                               command=" && ".join(commands), workdir=workdir, runtime_s=total)
        per_sample = self._split_bed(bed_gz, os.path.join(workdir, "segments"))
        vocab = [str(k) for k in range(n_states)]
        states = np.full(ds.n_rows, -1, dtype=np.int64)
        found = 0
        for sample in ds.samples:
            bed = per_sample.get(sample)
            if not bed:
                continue
            found += 1
            part, vocab = labels_from_segmentation_bed(ds, bed, sample=sample, vocabulary=vocab)
            keep = part != -1
            states[keep] = part[keep]
        if found == 0:
            return self.failed("segway.bed.gz held no pseudo-chromosome matching a sample",
                               command=" && ".join(commands), workdir=workdir, runtime_s=total)
        biv, act = posthoc_scores(ds, states, None, max(len(vocab), n_states))
        return ToolRun(
            tool=self.key, display_name=self.display_name, status="ok", states=states,
            posterior=None, n_states=max(len(vocab), n_states), bivalency_score=biv,
            activity_score=act, runtime_s=total, peak_rss_mb=peak, version=avail.version,
            command=" && ".join(commands), workdir=workdir, verified=self.verified,
            notes=(f"one model over {len(ds.samples)} samples as pseudo-chromosomes; hard labels "
                   f"only (no posterior); training capped at {os.environ.get('SEGWAY_MAX_ROUNDS', '10')} "
                   f"rounds, {os.environ.get('SEGWAY_INSTANCES', '1')} instance(s)"),
        )

    @staticmethod
    def _split_bed(bed_gz: str, outdir: str) -> Dict[str, str]:
        """``segway.bed.gz`` (pseudo-chromosomes) -> one BED4 per sample, real chromosomes."""
        os.makedirs(outdir, exist_ok=True)
        handles: Dict[str, object] = {}
        paths: Dict[str, str] = {}
        with gzip.open(bed_gz, "rt") as fh:
            for line in fh:
                if line.startswith(("track", "browser", "#")):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4 or SEP not in parts[0]:
                    continue
                chrom, sample = parts[0].rsplit(SEP, 1)
                if sample not in handles:
                    paths[sample] = os.path.join(outdir, f"{sample}_segments.bed")
                    handles[sample] = open(paths[sample], "w")
                handles[sample].write("\t".join([chrom, parts[1], parts[2], parts[3]]) + "\n")
        for h in handles.values():
            h.close()
        return paths


def build() -> list:
    return [SegwayAdapter()]
