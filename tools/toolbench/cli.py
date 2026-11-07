"""Command line: ``check``, ``metrics``, ``run``.

    python tools/run_tools.py check
    python tools/run_tools.py metrics
    python tools/run_tools.py run --simulate 4000 --n-states 7 -o compare_out
    python tools/run_tools.py run --tools chromstate,chromhmm --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence

import pandas as pd

from . import methods  # noqa: F401  (registers every compared method)
from .core import get_adapter, registry
from .evaluate import agreement_matrix, availability_table, evaluate_runs, run_tools
from .io_formats import frame_from_runs
from .metrics import ALL_METRICS, METRICS, TRUTH_FREE
from .report import write_report


def _print_frame(frame: pd.DataFrame) -> None:
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.max_colwidth", 60):
        print(frame.to_string(index=False))


def cmd_check(args: argparse.Namespace) -> int:
    table = availability_table()
    _print_frame(table[["tool", "kind", "available", "verified", "detail", "version"]])
    missing = table.loc[~table["available"]]
    if not missing.empty:
        print("\nNot available here:")
        for _, row in missing.iterrows():
            print(f"  {row['tool']}: {row['detail']}")
            if row["how_to_enable"]:
                print(f"      -> {row['how_to_enable']}")
    unverified = table.loc[table["available"] & ~table["verified"], "tool"].tolist()
    if unverified:
        print("\nAvailable but unverified (command line not yet executed against the real "
              "tool in this repo): " + ", ".join(unverified))
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    for name in ALL_METRICS:
        metric = METRICS[name]
        needs = ", ".join(metric.requires) or "nothing"
        print(f"{name}\n    {metric.description}")
        print(f"    columns: {', '.join(metric.columns)}")
        print(f"    requires: {needs}\n")
    print("Truth-free metrics (usable on real data): " + ", ".join(TRUTH_FREE))
    return 0


def _load_dataset(args: argparse.Namespace):
    """Return ``(dataset, truth, note)``. Truth is None for real data."""
    import chromstate as cs

    if args.counts:
        frame = pd.read_csv(args.counts, sep="\t")
        ds = _dataset_from_frame(frame)
        cs.normalize(ds, method=args.normalize)
        note = (f"Dataset: {args.counts} — {ds.n_rows} bin-rows, {ds.n_marks} marks, "
                f"{len(ds.samples)} samples. No ground truth, so only truth-free metrics "
                f"are reported.")
        return ds, None, note

    from chromstate.simulate import simulate

    ds, truth = simulate(
        n_bins=args.simulate,
        n_chroms=args.n_chroms,
        seed=args.seed,
        blur=args.blur,
        perturbation={"responder_fraction": 0.5, "from_state": "TssBiv", "to_state": "TssA"},
    )
    cs.normalize(ds, method=args.normalize)
    note = (f"Dataset: simulated — {ds.n_rows} bin-rows, {ds.n_marks} marks, "
            f"{len(ds.samples)} samples, {truth.n_states} true states, "
            f"blur {args.blur:g}, seed {args.seed}. Ground truth is known, so every "
            f"metric applies. Note that the simulator draws from a negative-binomial HMM, "
            f"the same family ChromState fits — that biases the comparison in its favour "
            f"and no amount of blurring removes it.")
    return ds, truth, note


def _dataset_from_frame(frame: pd.DataFrame):
    """Rebuild a dataset from the count matrix :func:`export_count_matrix` writes."""
    import numpy as np

    from chromstate.dataset import ChromStateDataset

    meta_cols = ["chrom", "start", "end", "sample"]
    missing = [c for c in meta_cols if c not in frame.columns]
    if missing:
        raise SystemExit(f"count matrix is missing column(s): {', '.join(missing)}")
    marks = [c for c in frame.columns if c not in meta_cols]
    frame = frame.sort_values(["sample", "chrom", "start"]).reset_index(drop=True)

    chains, start = [], 0
    for _, grp in frame.groupby(["sample", "chrom"], sort=False):
        chains.append(slice(start, start + len(grp)))
        start += len(grp)
    X = frame[marks].to_numpy(dtype=np.float64)
    return ChromStateDataset(
        X=X,
        marks=list(marks),
        bins=frame[meta_cols],
        chains=chains,
        size_factors=np.ones_like(X),
    )


def cmd_run(args: argparse.Namespace) -> int:
    keys = [k.strip() for k in args.tools.split(",")] if args.tools else list(registry())
    unknown = [k for k in keys if k not in registry()]
    if unknown:
        print(f"unknown tool(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"registered: {', '.join(registry())}", file=sys.stderr)
        return 2

    ds, truth, note = _load_dataset(args)
    n_states = args.n_states or (truth.n_states if truth is not None else 7)
    # Absolute: external tools are launched with cwd=<their workdir>, so a relative
    # output path would make every file they are told to read or write point nowhere.
    args.out = os.path.abspath(args.out)
    workroot = os.path.abspath(args.workdir or os.path.join(args.out, "work"))
    os.makedirs(workroot, exist_ok=True)

    if args.dry_run:
        print(note + "\n")
        for key in keys:
            adapter = get_adapter(key)
            avail = adapter.availability()
            print(f"{key} ({adapter.kind}, available={avail.available}): {avail.detail}")
            if hasattr(adapter, "dry_run_command"):
                print("    " + adapter.dry_run_command(
                    ds, n_states, args.seed, os.path.join(workroot, key)))
        return 0

    runs = run_tools(ds, keys, n_states=n_states, seed=args.seed, workroot=workroot,
                     timeout_s=args.timeout)
    metrics: Optional[List[str]] = (
        [m.strip() for m in args.metrics.split(",")] if args.metrics else None
    )
    if truth is None and metrics is None:
        metrics = TRUTH_FREE
    table = evaluate_runs(runs, ds, truth=truth, metrics=metrics, reference=args.reference)
    agreement = agreement_matrix(runs)

    notes = {name: METRICS[name].description for name in (metrics or ALL_METRICS)}
    written = write_report(args.out, table, agreement, runs, note, metric_notes=notes,
                           truth=truth)

    if args.per_bin:
        path = os.path.join(args.out, "per_bin_calls.tsv.gz")
        frame_from_runs(ds, runs).to_csv(path, sep="\t", index=False)
        written.append(path)
    if args.save_runs:
        from .realdata import save_runs

        written += save_runs(runs, os.path.join(args.out, "runs"))

    _print_frame(table[[c for c in ["method", "status", "ari", "biv_auprc", "bin_coverage",
                                    "runtime_s"] if c in table.columns]])
    print("\nwrote:")
    for path in written:
        print(f"  {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toolbench",
        description="Run other chromatin segmentation tools on ChromState's bins and "
                    "score them all with the same metrics.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="which tools can run here, and how to enable the rest")
    p_check.set_defaults(func=cmd_check)

    p_metrics = sub.add_parser("metrics", help="list the available metrics")
    p_metrics.set_defaults(func=cmd_metrics)

    p_run = sub.add_parser("run", help="run tools and write a comparison")
    p_run.add_argument("-o", "--out", default="tool_comparison", help="output directory")
    p_run.add_argument("--tools", default=None,
                       help="comma-separated tool keys (default: every registered tool)")
    p_run.add_argument("--metrics", default=None,
                       help=f"comma-separated metric names (default: all applicable). "
                            f"Available: {', '.join(ALL_METRICS)}")
    p_run.add_argument("--n-states", type=int, default=None,
                       help="number of states each tool is asked for")
    p_run.add_argument("--counts", default=None,
                       help="TSV count matrix (chrom start end sample <marks...>) for a "
                            "real dataset; without it, data is simulated with known truth")
    p_run.add_argument("--simulate", type=int, default=3000,
                       help="bins per chromosome per sample when simulating")
    p_run.add_argument("--n-chroms", type=int, default=2)
    p_run.add_argument("--blur", type=float, default=0.3,
                       help="fraction of simulated bins blurred between two states")
    p_run.add_argument("--normalize", default="median_ratio",
                       choices=("depth", "median_ratio", "none"))
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--reference", default="chromstate",
                       help="tool key the agreement metric compares against")
    p_run.add_argument("--timeout", type=float, default=3600.0,
                       help="per-tool wall-clock limit in seconds")
    p_run.add_argument("--workdir", default=None,
                       help="where external tools run (default: <out>/work)")
    p_run.add_argument("--per-bin", action="store_true",
                       help="also write every tool's per-bin call")
    p_run.add_argument("--dry-run", action="store_true",
                       help="print the command each tool would run, then stop")
    p_run.add_argument("--save-runs", action="store_true",
                       help="write every tool's per-bin states and posterior to <out>/runs/ "
                            "so `realdata` can draw figures without re-running the tools")
    p_run.set_defaults(func=cmd_run)

    p_real = sub.add_parser("realdata", help="evaluation figures on real data from saved runs: "
                                             "state profiles, genome shares, promoter recovery, "
                                             "expression prediction, locus views")
    p_real.add_argument("--runs", required=True, action="append",
                        help="<out>/runs from `run --save-runs`; repeat to merge runs made "
                             "separately (a later directory replaces a tool from an earlier one)")
    p_real.add_argument("--counts", required=True, help="the count matrix the runs were made on")
    p_real.add_argument("--normalize", default="median_ratio", choices=("depth", "median_ratio", "none"))
    p_real.add_argument("--gtf", default=None, help="gene annotation (GENCODE/Ensembl GTF)")
    p_real.add_argument("--gene-types", default="protein_coding",
                        help="comma-separated gene_type values to keep from the GTF; 'all' keeps every gene")
    p_real.add_argument("--expression", default=None,
                        help="genes x samples table (log scale), columns named after the samples")
    p_real.add_argument("--contrast", default=None, help="two sample names 'A,B' for the change task")
    p_real.add_argument("--flank", type=int, default=1000, help="promoter half-width in bp")
    p_real.add_argument("--locus", action="append", default=[],
                        help="chrom:start-end[:name]; may be repeated")
    p_real.add_argument("--reference", default="chromstate")
    p_real.add_argument("--comparison", default=None, help="comparison.csv from `run`, for the report")
    p_real.add_argument("--seed", type=int, default=0)
    p_real.add_argument("-o", "--out", required=True)
    p_real.set_defaults(func=cmd_realdata)
    return parser


def cmd_realdata(args: argparse.Namespace) -> int:
    import chromstate as cs

    from .realdata import load_runs, read_expression, read_gtf_genes, write_realdata_report

    frame = pd.read_csv(args.counts, sep="\t")
    ds = _dataset_from_frame(frame)
    if args.normalize != "none":
        cs.normalize(ds, method=args.normalize)
    merged = {}
    for d in args.runs:
        for r in load_runs(d):
            merged[r.tool] = r
    runs = list(merged.values())
    chroms = sorted(ds.bins["chrom"].unique())
    features = expression = None
    if args.gtf:
        types = None if args.gene_types == "all" else tuple(args.gene_types.split(","))
        features = read_gtf_genes(args.gtf, chroms=chroms, gene_types=types)
        print(f"[realdata] {len(features):,} genes on {len(chroms)} chromosome(s)")
    if args.expression:
        expression = read_expression(args.expression, ds.samples)
    contrast = tuple(args.contrast.split(",")) if args.contrast else None
    loci = []
    for spec in args.locus:
        parts = spec.split(":")
        chrom, rng = parts[0], parts[1]
        a, b = (int(v.replace(",", "")) for v in rng.split("-"))
        loci.append((chrom, a, b, parts[2] if len(parts) > 2 else f"{chrom}_{a}"))
    comparison = pd.read_csv(args.comparison) if args.comparison else None
    note = (f"Dataset: {args.counts} — {ds.n_rows:,} bin-rows, {ds.n_marks} marks "
            f"({', '.join(ds.marks)}), samples {', '.join(ds.samples)}, chromosomes {', '.join(chroms)}.")
    written = write_realdata_report(args.out, ds, runs, features, expression, args.flank, loci,
                                    args.reference, contrast, comparison, note, args.seed)
    print("wrote:")
    for path in written:
        print(f"  {path}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
