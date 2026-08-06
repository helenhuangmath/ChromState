"""Methods described by a command template in ``tools.json`` rather than by Python.

This is how EpiCSeg, Segway, epigenome-ssm, and anything local join the comparison: give
the harness a command line, say where the output lands, and it handles export, execution,
parsing onto the bin grid, and scoring. No new module needed per tool.

Everything loaded this way starts ``verified=False``. The shipped entries' command lines
are transcribed from published documentation and have not been run against the real
software here.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
from typing import Dict, List

import numpy as np

from ..core import Availability, ToolAdapter, ToolRun, probe_version, run_command
from ..io_formats import (
    export_chromhmm_binary,
    export_count_matrix,
    export_regions_bed,
    labels_from_label_file,
    labels_from_segmentation_bed,
    posterior_from_matrix_file,
)

#: ``tools/tools.json`` -- two directories up from this module.
TOOLS_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tools.json",
)


class ExternalCommandAdapter(ToolAdapter):
    """A tool described by a command template and an output parser.

    The spec is a dict from ``tools.json``::

        {
          "key": "epicseg",
          "display_name": "EpiCSeg (negative multinomial hidden Markov model)",
          "stands_for": "count-based segmentation, published prior art",
          "enabled": false,
          "requires_env": ["EPICSEG_R"],
          "requires_binary": "Rscript",
          "input_format": "count_matrix",       # count_matrix | chromhmm_binary | regions_bed
          "command": ["Rscript", "{env:EPICSEG_R}", "segment",
                      "--counts", "{counts}", "--regions", "{regions}",
                      "--nstates", "{n_states}", "--outdir", "{outdir}"],
          "output": {"format": "segmentation_bed",
                     "path": "{outdir}/segmentation.bed",
                     "per_sample": true},
          "version_command": ["Rscript", "-e", "cat(as.character(packageVersion('epicseg')))"],
          "verified": false,
          "notes": "command line from the EpiCSeg README; not executed in this repo"
        }

    Placeholders available in ``command`` and ``output.path``: ``{workdir}``, ``{outdir}``,
    ``{counts}``, ``{regions}``, ``{binary_dir}``, ``{n_states}``, ``{seed}``,
    ``{bin_size}``, ``{sample}``, and ``{env:VAR}`` for any environment variable.
    """

    kind = "external"

    def __init__(self, spec: Dict[str, object]):
        self.spec = spec
        self.key = str(spec["key"])
        self.display_name = str(spec.get("display_name", self.key))
        self.stands_for = str(spec.get("stands_for", ""))
        self.verified = bool(spec.get("verified", False))
        self.notes = str(spec.get("notes", ""))

    # -- availability -------------------------------------------------------

    def availability(self) -> Availability:
        spec = self.spec
        how = str(spec.get("how_to_enable", "")) or (
            f'set "enabled": true in tools.json and provide '
            f'{", ".join(spec.get("requires_env", [])) or "the command"}'
        )
        if not spec.get("enabled", False):
            return Availability(False, 'disabled in tools.json ("enabled": false)',
                                how_to_enable=how)
        for var in spec.get("requires_env", []):  # type: ignore[union-attr]
            if not os.environ.get(str(var)):
                return Availability(False, f"environment variable {var} is not set",
                                    how_to_enable=how)
        binary = spec.get("requires_binary")
        if binary and not shutil.which(str(binary)):
            return Availability(False, f"{binary} not found on PATH", how_to_enable=how)
        version = ""
        if spec.get("version_command"):
            argv = [self._expand(str(a), {}) for a in spec["version_command"]]  # type: ignore[union-attr]
            version = probe_version(argv)
        return Availability(True, "configured in tools.json", version, how)

    # -- running ------------------------------------------------------------

    def _expand(self, template: str, fields: Dict[str, str]) -> str:
        out = template
        for key, value in fields.items():
            out = out.replace("{" + key + "}", str(value))
        while "{env:" in out:
            head, rest = out.split("{env:", 1)
            var, tail = rest.split("}", 1)
            # An unset variable leaves a visible marker rather than an empty string, so a
            # --dry-run of a not-yet-configured tool shows what is missing instead of
            # printing a command with a hole in it.
            out = head + os.environ.get(var, f"<${var}-unset>") + tail
        return out

    def _build_argv(self, fields: Dict[str, str]) -> List[str]:
        """Expand the command template into argv.

        An element that is *exactly* an ``{env:VAR}`` placeholder is word-split after
        expansion, so a variable can hold a multi-word launcher such as
        ``python -m epigenome_ssm``. Every other element stays one argument, so paths
        containing spaces survive.
        """
        argv: List[str] = []
        for raw in self.spec["command"]:  # type: ignore[union-attr]
            token = str(raw)
            expanded = self._expand(token, fields)
            if token.startswith("{env:") and token.endswith("}") and token.count("{") == 1:
                argv.extend(shlex.split(expanded))
            else:
                argv.append(expanded)
        return argv

    def _fields(self, ds, n_states: int, seed: int, workdir: str) -> Dict[str, str]:
        outdir = os.path.join(workdir, "out")
        return {
            "workdir": workdir,
            "outdir": outdir,
            "n_states": str(n_states),
            "seed": str(seed),
            "bin_size": str(int(ds.bins["end"].iloc[0] - ds.bins["start"].iloc[0])),
            "counts": os.path.join(workdir, "counts.tsv"),
            "regions": os.path.join(workdir, "regions.bed"),
            "binary_dir": os.path.join(workdir, "binary"),
            "sample": ds.samples[0],
        }

    def dry_run_command(self, ds, n_states: int, seed: int, workdir: str) -> str:
        """The exact command this adapter would execute, without running it."""
        fields = self._fields(ds, n_states, seed, workdir)
        return " ".join(shlex.quote(a) for a in self._build_argv(fields))

    def run(self, ds, n_states: int, seed: int, workdir: str, timeout_s: float) -> ToolRun:
        avail = self.availability()
        if not avail.available:
            return self.skipped(avail.detail)

        outdir = os.path.join(workdir, "out")
        os.makedirs(outdir, exist_ok=True)
        fields = self._fields(ds, n_states, seed, workdir)

        fmt = str(self.spec.get("input_format", "count_matrix"))
        if fmt == "count_matrix":
            export_count_matrix(ds, fields["counts"])
            export_regions_bed(ds, fields["regions"])
        elif fmt == "regions_bed":
            export_regions_bed(ds, fields["regions"])
        elif fmt == "chromhmm_binary":
            export_chromhmm_binary(ds, fields["binary_dir"])
        else:
            return self.failed(f"unknown input_format {fmt!r}")

        argv = self._build_argv(fields)
        res = run_command(argv, cwd=workdir, timeout_s=timeout_s,
                          log_path=os.path.join(workdir, f"{self.key}.log"))
        if res.returncode != 0:
            return self.failed(
                f"exited {res.returncode}" + (" (timed out)" if res.timed_out else ""),
                command=res.command, workdir=workdir, runtime_s=res.runtime_s,
            )

        try:
            states, vocab, posterior = self._parse_output(ds, fields)
        except (OSError, ValueError) as exc:
            return self.failed(f"could not read output: {exc}", command=res.command,
                               workdir=workdir, runtime_s=res.runtime_s)

        return ToolRun(
            tool=self.key,
            display_name=self.display_name,
            status="ok",
            states=states,
            posterior=posterior,
            n_states=max(len(vocab), n_states),
            runtime_s=res.runtime_s,
            peak_rss_mb=res.peak_rss_mb,
            version=avail.version,
            command=res.command,
            workdir=workdir,
            verified=self.verified,
            notes=self.notes,
        )

    def _parse_output(self, ds, fields: Dict[str, str]):
        out = self.spec.get("output", {})  # type: ignore[assignment]
        fmt = str(out.get("format", "segmentation_bed"))  # type: ignore[union-attr]
        per_sample = bool(out.get("per_sample", False))  # type: ignore[union-attr]
        template = str(out.get("path"))  # type: ignore[union-attr]

        states = np.full(ds.n_rows, -1, dtype=np.int64)
        vocab: List[str] = []
        samples = ds.samples if per_sample else [None]
        for sample in samples:
            local = dict(fields)
            if sample is not None:
                local["sample"] = sample
            path = self._expand(template, local)
            if fmt == "segmentation_bed":
                part, vocab = labels_from_segmentation_bed(ds, path, sample=sample,
                                                           vocabulary=vocab)
            elif fmt == "labels":
                part, vocab = labels_from_label_file(ds, path, sample=sample,
                                                     vocabulary=vocab)
            else:
                raise ValueError(f"unknown output format {fmt!r}")
            keep = part != -1
            states[keep] = part[keep]

        posterior = None
        post_path = out.get("posterior_path")  # type: ignore[union-attr]
        if post_path:
            posterior = posterior_from_matrix_file(
                ds, self._expand(str(post_path), fields),
                skiprows=int(out.get("posterior_skiprows", 0)),  # type: ignore[union-attr]
            )
        return states, vocab, posterior


def load_specs(path: str = TOOLS_JSON) -> List[Dict[str, object]]:
    """Read the tool specs from ``tools.json``; empty when the file is absent."""
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return json.load(fh).get("tools", [])


def build(path: str = TOOLS_JSON) -> List[ExternalCommandAdapter]:
    """One adapter per entry in ``tools.json``."""
    return [ExternalCommandAdapter(spec) for spec in load_specs(path)]
