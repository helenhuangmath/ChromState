"""Driver: run the selected tools on one dataset, then score them with the same metrics.

Everything here is deliberately boring and sequential. Parallelising the tool runs would
make the runtime and memory columns meaningless -- they are measurements of the tools, and
tools that share a machine measure each other.
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .core import ToolRun, get_adapter, registry
from .metrics import EvalContext, Metric, resolve

__all__ = ["run_tools", "evaluate_runs", "agreement_matrix", "availability_table"]


def availability_table() -> pd.DataFrame:
    """One row per registered tool: can it run here, and if not, what would fix it."""
    rows = []
    for key, adapter in registry().items():
        avail = adapter.availability()
        rows.append(
            {
                "tool": key,
                "display_name": adapter.display_name,
                "kind": adapter.kind,
                "available": avail.available,
                "verified": adapter.verified,
                "detail": avail.detail,
                "version": avail.version,
                "how_to_enable": "" if avail.available else avail.how_to_enable,
                "stands_for": adapter.stands_for,
            }
        )
    return pd.DataFrame(rows)


def run_tools(
    ds,
    tools: Sequence[str],
    n_states: int,
    seed: int = 0,
    workroot: Optional[str] = None,
    timeout_s: float = 3600.0,
    verbose: bool = True,
) -> List[ToolRun]:
    """Run each tool on the same dataset, one at a time, and collect uniform results.

    A tool that is unavailable or crashes yields a ``skipped``/``failed`` run rather than
    an exception: one missing dependency should not throw away the tools that did run.
    """
    # Without an explicit root, work goes to a temp directory rather than the current
    # one: in-process methods need no scratch space at all, and creating ./<tool>/ for
    # each of them litters whatever directory the caller happened to be in.
    made_root = workroot is None
    root = workroot or tempfile.mkdtemp(prefix="toolbench-")

    runs: List[ToolRun] = []
    for key in tools:
        adapter = get_adapter(key)
        workdir = os.path.join(root, key)
        if adapter.kind != "in-process":
            os.makedirs(workdir, exist_ok=True)
        if verbose:
            print(f"[toolbench] {key}: running")
        try:
            run = adapter.run(ds, n_states=n_states, seed=seed, workdir=workdir,
                              timeout_s=timeout_s)
        except Exception as exc:  # adapter bug, not tool failure -- report, do not crash
            run = adapter.failed(f"adapter raised {type(exc).__name__}: {exc}")
        if verbose:
            detail = run.reason or f"{run.runtime_s:.1f}s, coverage {run.coverage:.2f}"
            print(f"[toolbench] {key}: {run.status} ({detail})")
        runs.append(run)
    # An auto-created root that no external tool wrote into (in-process methods only,
    # which is every test run) is removed again; otherwise every run would leave an
    # empty ``toolbench-xxxx`` folder under TMPDIR.
    if made_root:
        try:
            for key in tools:  # a skipped external tool leaves its empty folder behind
                sub = os.path.join(root, key)
                if os.path.isdir(sub) and not os.listdir(sub):
                    os.rmdir(sub)
            if not os.listdir(root):
                os.rmdir(root)
        except OSError:
            pass
    return runs


def evaluate_runs(
    runs: Sequence[ToolRun],
    ds,
    truth: Optional[object] = None,
    metrics: Optional[Sequence[str]] = None,
    reference: Optional[str] = None,
) -> pd.DataFrame:
    """Score every run with every applicable metric. One row per tool, always.

    Metrics that do not apply to a run -- no truth, no posterior, tool failed -- leave
    NaN. A NaN here means "not measurable", never "zero"; the report says which.
    """
    selected: List[Metric] = resolve(metrics)
    ref_run = None
    if reference:
        ref_run = next((r for r in runs if r.tool == reference and r.ok), None)

    rows: List[Dict[str, object]] = []
    for run in runs:
        row: Dict[str, object] = {
            "tool": run.tool,
            "method": run.display_name,  # named for chromstate.viz.plot_method_comparison
            "status": run.status,
            "verified": run.verified,
            "n_states": run.n_states,
            "reason": run.reason,
            "notes": run.notes,
        }
        # A tool is never scored for agreement against itself: that is 1.0 by
        # construction and would read as a result.
        against = None if (ref_run is None or ref_run.tool == run.tool) else ref_run
        ctx = EvalContext(run=run, ds=ds, truth=truth, reference=against)
        for metric in selected:
            if not metric.applicable(ctx):
                continue
            try:
                row.update(metric.fn(ctx))
            except Exception as exc:  # a metric must never sink the table
                row[f"{metric.name}_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    table = pd.DataFrame(rows)
    front = ["tool", "method", "status", "verified", "n_states", "ari", "matched_accuracy",
             "biv_auprc", "signal_spearman", "overconfidence", "frac_saturated",
             "bin_coverage", "mean_segment_kb", "runtime_s"]
    ordered = [c for c in front if c in table.columns]
    return table[ordered + [c for c in table.columns if c not in ordered]]


def agreement_matrix(runs: Sequence[ToolRun]) -> pd.DataFrame:
    """Pairwise adjusted Rand index between every pair of successful runs.

    On real data this is most of what you have: it cannot tell you which tool is right,
    but a tool that agrees with nothing else is worth looking at before it is trusted.
    """
    from chromstate.benchmark.metrics import segmentation_agreement

    from .core import UNASSIGNED

    ok = [r for r in runs if r.ok]
    names = [r.display_name for r in ok]
    mat = np.full((len(ok), len(ok)), np.nan)
    for i, a in enumerate(ok):
        for j, b in enumerate(ok):
            if i == j:
                mat[i, j] = 1.0
                continue
            keep = (a.states != UNASSIGNED) & (b.states != UNASSIGNED)
            if keep.sum() >= 2:
                mat[i, j] = segmentation_agreement(a.states[keep], b.states[keep])["ari"]
    return pd.DataFrame(mat, index=names, columns=names)
