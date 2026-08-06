"""One uniform interface over segmentation tools that otherwise share nothing.

ChromHMM is a Java jar that reads binarised text files and writes BED. EpiCSeg is an R
script that reads a count matrix. Segway needs a genomedata archive and GMTK. ChromState
is a Python function call. Comparing them means normalising three things:

**Input.** Every tool is handed the *same bins*, exported into whatever format it insists
on. Nothing is re-binned, re-normalised, or re-peak-called per tool, because a difference
that comes from preprocessing is not a difference between methods.

**Output.** Every tool returns a per-bin state label on ChromState's bin grid, plus a
posterior where it produces one. Bins a tool leaves uncovered stay ``UNASSIGNED`` rather
than being quietly imputed -- coverage is then reported as a metric instead of hiding as
a silent difference in denominator.

**Failure.** A tool that is not installed, times out, or crashes produces a row with a
status and a reason, never a missing row. A benchmark table that silently drops the tools
that failed reads as though everything ran, which is the single easiest way to publish a
misleading comparison.

The honesty rule for this package: an adapter whose command line has not been executed
against the real tool in this repository is marked ``verified=False`` and says so in
``toolbench check``. The command lines come from each tool's published documentation, but
documentation drifts, and an unverified adapter is a hypothesis about an interface.
"""

from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

#: Bin label used where a tool produced no call at all.
UNASSIGNED = -1


@dataclass
class Availability:
    """Whether a tool can run here, and what to do about it if not."""

    available: bool
    detail: str
    version: str = ""
    how_to_enable: str = ""


@dataclass
class ToolRun:
    """Uniform result from any tool, successful or not.

    ``states`` is on ChromState's bin grid: one entry per row of the dataset, in the same
    order, with :data:`UNASSIGNED` where the tool made no call.
    """

    tool: str
    display_name: str
    status: str = "ok"  # "ok" | "skipped" | "failed"
    reason: str = ""
    states: Optional[np.ndarray] = None
    posterior: Optional[np.ndarray] = None
    n_states: int = 0
    bivalency_score: Optional[np.ndarray] = None
    activity_score: Optional[np.ndarray] = None
    runtime_s: float = float("nan")
    peak_rss_mb: Optional[float] = None
    version: str = ""
    command: str = ""
    workdir: Optional[str] = None
    verified: bool = True
    notes: str = ""
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.states is not None

    @property
    def coverage(self) -> float:
        """Fraction of bins the tool actually assigned to a state."""
        if self.states is None:
            return float("nan")
        return float((self.states != UNASSIGNED).mean())


class ToolAdapter:
    """Base class. Subclasses implement :meth:`availability` and :meth:`run`."""

    #: Registry key, used on the command line.
    key: str = ""
    #: Name printed in tables and figures. Spelled out, per the figure conventions.
    display_name: str = ""
    #: "in-process" runs inside this interpreter; "external" shells out.
    kind: str = "external"
    #: False when the command line has never been executed against the real tool here.
    verified: bool = False
    #: One line on what this stands for in the comparison.
    stands_for: str = ""

    def availability(self) -> Availability:  # pragma: no cover - overridden
        raise NotImplementedError

    def run(  # pragma: no cover - overridden
        self, ds, n_states: int, seed: int, workdir: str, timeout_s: float
    ) -> ToolRun:
        raise NotImplementedError

    # -- helpers for subclasses ---------------------------------------------

    def skipped(self, reason: str) -> ToolRun:
        return ToolRun(
            tool=self.key,
            display_name=self.display_name,
            status="skipped",
            reason=reason,
            verified=self.verified,
        )

    def failed(self, reason: str, **kw) -> ToolRun:
        return ToolRun(
            tool=self.key,
            display_name=self.display_name,
            status="failed",
            reason=reason,
            verified=self.verified,
            **kw,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: "Dict[str, ToolAdapter]" = {}


def register(adapter: ToolAdapter) -> ToolAdapter:
    """Add an adapter to the registry, keyed by ``adapter.key``."""
    if not adapter.key:
        raise ValueError("adapter needs a key")
    if adapter.key in _REGISTRY:
        raise ValueError(f"duplicate tool key {adapter.key!r}")
    _REGISTRY[adapter.key] = adapter
    return adapter


def registry() -> Dict[str, ToolAdapter]:
    """All registered adapters, in registration order."""
    return dict(_REGISTRY)


def get_adapter(key: str) -> ToolAdapter:
    if key not in _REGISTRY:
        raise KeyError(f"unknown tool {key!r}; registered: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[key]


def available_keys() -> List[str]:
    return [k for k, a in _REGISTRY.items() if a.availability().available]


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------


@dataclass
class CommandResult:
    command: str
    returncode: int
    runtime_s: float
    stdout: str
    stderr: str
    peak_rss_mb: Optional[float]
    timed_out: bool = False


def run_command(
    argv: Sequence[str],
    cwd: Optional[str] = None,
    timeout_s: float = 3600.0,
    log_path: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> CommandResult:
    """Run an external tool, timing it and capturing everything it said.

    Peak memory is reported only when ``psutil`` is installed. The obvious alternative,
    ``resource.getrusage(RUSAGE_CHILDREN)``, returns a high-water mark across *all*
    children this process has ever reaped, so after the first heavyweight tool it reports
    that tool's memory for everything that follows. A wrong number is worse than none.
    """
    t0 = time.time()
    peak_mb: Optional[float] = None
    timed_out = False
    try:
        import psutil  # type: ignore

        proc = psutil.Popen(list(argv), cwd=cwd, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        peak = 0.0
        while proc.poll() is None:
            try:
                rss = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    rss += child.memory_info().rss
                peak = max(peak, rss)
            except Exception:  # process exited between poll and read
                pass
            if time.time() - t0 > timeout_s:
                proc.kill()
                timed_out = True
                break
            time.sleep(0.05)
        out, err = proc.communicate()
        peak_mb = peak / 1e6 if peak else None
        rc = proc.returncode
    except ImportError:
        try:
            done = subprocess.run(list(argv), cwd=cwd, env=env, capture_output=True,
                                  text=True, timeout=timeout_s)
            out, err, rc = done.stdout, done.stderr, done.returncode
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout or ""
            err = (exc.stderr or "") + f"\n[toolbench] timed out after {timeout_s:g}s"
            rc, timed_out = -1, True

    runtime = time.time() - t0
    command = " ".join(shlex.quote(str(a)) for a in argv)
    if log_path:
        with open(log_path, "w") as fh:
            fh.write(f"$ {command}\n\n--- stdout ---\n{out}\n--- stderr ---\n{err}\n")
    return CommandResult(command, rc, runtime, out, err, peak_mb, timed_out)


def probe_version(argv: Sequence[str], timeout_s: float = 30.0) -> str:
    """Best-effort version string; empty when the probe fails."""
    try:
        res = run_command(argv, timeout_s=timeout_s)
    except (OSError, ValueError):
        return ""
    text = (res.stdout or res.stderr).strip().splitlines()
    return text[0][:120] if text else ""


def occupancy(states: np.ndarray, n_states: int) -> np.ndarray:
    """Fraction of assigned bins in each state (ignores :data:`UNASSIGNED`)."""
    assigned = states[states != UNASSIGNED]
    if assigned.size == 0:
        return np.zeros(max(n_states, 1))
    counts = np.bincount(assigned, minlength=max(n_states, 1)).astype(float)
    return counts / counts.sum()
