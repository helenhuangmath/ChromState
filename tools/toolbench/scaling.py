"""Runtime and memory against dataset size, measured rather than asserted.

    python tools/scaling.py -o scaling_out --sizes 6250,25000,100000 \
        --tools chromstate,chromhmm,epicseg,normr,dihmm

For each size the simulator is run once (``n_chroms`` chromosomes x ``size`` bins x two
conditions, so total bin-rows = 2 x n_chroms x size) and every tool is run on that same
dataset, one at a time.

How cost is measured, because the two kinds of tool cannot be measured identically:

* **External tools** (ChromHMM, EpiCSeg, normR, diHMM) run as subprocesses through their
  adapters, which time them and record the peak resident memory of the tool's whole
  process tree (``toolbench.core.run_command``). Export of input files and parsing of
  output are outside the timed region.
* **In-process methods** (ChromState and the reimplemented baselines) are run in a fresh
  child interpreter (``--child``), so their memory is not polluted by whatever the
  parent has already allocated. Runtime is the method's own wall-clock (fit + decode),
  excluding simulation. Peak memory is the child's whole process, which *includes* the
  Python interpreter, the simulated dataset and the simulator's own temporaries; the
  ``baseline_rss_mb`` column (a child that only simulates) is reported next to it so the
  method's own share can be read off.

A tool that exceeds ``--timeout`` at one size is not attempted at larger sizes; its row
says so. Single runs, single seed: the numbers carry run-to-run noise of a few percent
and are not a substitute for repeated timing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Dict, List, Optional, Sequence

import pandas as pd

from . import methods  # noqa: F401  (registers every compared method)
from .core import get_adapter, run_command
from .evaluate import evaluate_runs, run_tools

DEFAULT_TOOLS = "chromstate,chromhmm,epicseg,normr,dihmm"


def _simulate(size: int, n_chroms: int, seed: int, blur: float):
    import chromstate as cs
    from chromstate.simulate import simulate

    ds, truth = simulate(
        n_bins=size, n_chroms=n_chroms, seed=seed, blur=blur,
        perturbation={"responder_fraction": 0.5, "from_state": "TssBiv", "to_state": "TssA"},
    )
    cs.normalize(ds, method="median_ratio")
    return ds, truth


def _child(args: argparse.Namespace) -> int:
    """Run one in-process method (or nothing) on a fresh simulation; print JSON."""
    ds, truth = _simulate(args.size, args.n_chroms, args.seed, args.blur)
    out: Dict[str, object] = {"n_rows": int(ds.n_rows)}
    if args.child != "none":
        runs = run_tools(ds, [args.child], n_states=truth.n_states, seed=args.seed,
                         verbose=False)
        table = evaluate_runs(runs, ds, truth=truth, metrics=["recovery", "cost"])
        row = table.iloc[0]
        out.update(status=str(row["status"]), runtime_s=float(row.get("runtime_s", float("nan"))),
                   ari=float(row.get("ari", float("nan"))))
    print("SCALING_JSON " + json.dumps(out))
    return 0


def _run_child(method: str, size: int, args: argparse.Namespace, workdir: str) -> Dict:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    argv = [sys.executable, os.path.join(here, "scaling.py"), "--child", method,
            "--size", str(size), "--n-chroms", str(args.n_chroms), "--seed", str(args.seed),
            "--blur", str(args.blur)]
    res = run_command(argv, cwd=workdir, timeout_s=args.timeout,
                      log_path=os.path.join(workdir, f"child_{method}_{size}.log"))
    payload: Dict[str, object] = {}
    for line in res.stdout.splitlines():
        if line.startswith("SCALING_JSON "):
            payload = json.loads(line[len("SCALING_JSON "):])
    if res.returncode != 0 or not payload:
        payload.setdefault("status", "failed (timed out)" if res.timed_out else "failed")
    payload["peak_rss_mb"] = res.peak_rss_mb
    return payload


def run_scaling(sizes: Sequence[int], tools: Sequence[str], args: argparse.Namespace,
                workroot: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    gave_up: Dict[str, str] = {}
    for size in sizes:
        ds, truth = _simulate(size, args.n_chroms, args.seed, args.blur)
        base = _run_child("none", size, args, workroot)
        baseline = base.get("peak_rss_mb")
        print(f"[scaling] {ds.n_rows} bin-rows (baseline child {baseline} MB)", flush=True)
        for tool in tools:
            row: Dict[str, object] = {"tool": tool, "bins_per_chrom": size,
                                      "total_bin_rows": int(ds.n_rows)}
            adapter = get_adapter(tool)
            row["method"] = adapter.display_name
            if tool in gave_up:
                row.update(status="not attempted", reason=gave_up[tool])
                rows.append(row)
                continue
            if adapter.kind == "in-process":
                res = _run_child(tool, size, args, workroot)
                row.update(status=res.get("status"), runtime_s=res.get("runtime_s"),
                           peak_rss_mb=res.get("peak_rss_mb"), baseline_rss_mb=baseline,
                           ari=res.get("ari"), memory_scope="whole child process")
            else:
                workdir = os.path.join(workroot, f"{tool}_{size}")
                os.makedirs(workdir, exist_ok=True)
                run = run_tools(ds, [tool], n_states=truth.n_states, seed=args.seed,
                                workroot=workdir, timeout_s=args.timeout, verbose=False)[0]
                table = evaluate_runs([run], ds, truth=truth, metrics=["recovery", "cost"])
                r = table.iloc[0]
                row.update(status=run.status, reason=run.reason, runtime_s=run.runtime_s,
                           peak_rss_mb=run.peak_rss_mb, ari=r.get("ari"),
                           memory_scope="tool process tree")
            if str(row.get("status", "")).startswith("failed") or row.get("status") == "failed":
                if "timed out" in str(row.get("reason", "")) or "timed out" in str(row["status"]):
                    gave_up[tool] = f"timed out (> {args.timeout:g} s) at a smaller size"
            print(f"[scaling] {tool} @ {ds.n_rows}: {row.get('status')} "
                  f"{row.get('runtime_s')} s, {row.get('peak_rss_mb')} MB", flush=True)
            rows.append(row)
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Runtime and memory against dataset size.")
    p.add_argument("-o", "--out", default="scaling_out")
    p.add_argument("--sizes", default="6250,25000,100000",
                   help="bins per chromosome per sample (total = 2 x n_chroms x size)")
    p.add_argument("--n-chroms", type=int, default=4)
    p.add_argument("--tools", default=DEFAULT_TOOLS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--blur", type=float, default=0.3)
    p.add_argument("--timeout", type=float, default=1200.0, help="per tool and size, seconds")
    p.add_argument("--child", default=None, help=argparse.SUPPRESS)
    p.add_argument("--size", type=int, default=0, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.child:
        return _child(args)

    from chromstate import viz

    from .report import plot_scaling

    os.makedirs(args.out, exist_ok=True)
    sizes = [int(s) for s in args.sizes.split(",")]
    tools = [t.strip() for t in args.tools.split(",")]
    workroot = tempfile.mkdtemp(prefix="toolbench-scaling-", dir=args.out)
    table = run_scaling(sizes, tools, args, workroot)
    csv = os.path.join(args.out, "scaling.csv")
    table.to_csv(csv, index=False)
    paths = viz.save_all({"tool_scaling": plot_scaling(table)}, args.out)
    print(f"wrote {csv} and {', '.join(paths)}")
    return 0
