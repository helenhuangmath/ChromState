"""Command line interface.

    chromstate fit      --manifest cellmarkfiletable.txt --chrom-sizes hg38.sizes -k 12 -o out/
    chromstate selectk  --manifest ... --chrom-sizes ... --candidates 6,8,10,12,15
    chromstate compare  --out benchmark/
    chromstate demo     --out demo/
    chromstate export-chromhmm --manifest ... --chrom-sizes ... -o binarized/
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import pandas as pd


def _add_input_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--manifest", required=True,
                   help="ChromHMM-style cellmarkfiletable: sample<TAB>mark<TAB>file[<TAB>control]")
    p.add_argument("--chrom-sizes", required=True, help="two-column chrom<TAB>size file")
    p.add_argument("--bin-size", type=int, default=200, help="bin width in bp (default 200)")
    p.add_argument("--chroms", default=None, help="comma-separated subset of chromosomes")
    p.add_argument("--blacklist", default=None, help="BED of regions to exclude")
    p.add_argument("--norm", default="median_ratio",
                   choices=["median_ratio", "depth", "spikein", "none"],
                   help="normalisation method (default median_ratio)")
    p.add_argument("--spikein", default=None,
                   help="TSV sample<TAB>mark<TAB>spikein_reads, required for --norm spikein")


def _load(args) -> "object":
    import chromstate as cs
    from chromstate.genome import BinSet, Genome

    genome = Genome.from_file(args.chrom_sizes)
    chroms = args.chroms.split(",") if args.chroms else None
    binset = BinSet.from_genome(genome, bin_size=args.bin_size, chroms=chroms)
    entries = cs.read_manifest(args.manifest)

    blacklist = cs.read_bed_regions(args.blacklist) if args.blacklist else None
    print(f"[chromstate] {len(entries)} input files, {len(binset):,} bins", file=sys.stderr)
    ds = cs.build_dataset(entries, binset, blacklist=blacklist)
    print(f"[chromstate] matrix {ds.X.shape[0]:,} x {ds.X.shape[1]} "
          f"({len(ds.samples)} samples)", file=sys.stderr)

    spikein = None
    if args.norm == "spikein":
        if not args.spikein:
            raise SystemExit("--norm spikein requires --spikein <file>")
        tab = pd.read_csv(args.spikein, sep="\t", header=None,
                          names=["sample", "mark", "reads"])
        spikein = {(r["sample"], r["mark"]): float(r["reads"]) for _, r in tab.iterrows()}
    if args.norm != "none":
        cs.normalize(ds, method=args.norm, spikein=spikein)
        if "norm_warning" in ds.meta:
            print(f"[chromstate] WARNING: {ds.meta['norm_warning']}", file=sys.stderr)
    return ds


def cmd_fit(args) -> int:
    import chromstate as cs
    from chromstate import viz

    ds = _load(args)
    print(f"[chromstate] fitting K={args.n_states} ({args.emission}) …", file=sys.stderr)
    seg = cs.fit(ds, n_states=args.n_states, emission=args.emission,
                 n_restarts=args.restarts, seed=args.seed, verbose=args.verbose)
    print(f"[chromstate] loglik {seg.loglik:,.1f}  converged={seg.meta['converged']} "
          f"restart spread {seg.meta['restart_spread']:,.1f}", file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    seg.emission_frame().to_csv(os.path.join(args.out, "emissions.tsv"), sep="\t")
    seg.transition_frame().to_csv(os.path.join(args.out, "transitions.tsv"), sep="\t")
    seg.state_table().to_csv(os.path.join(args.out, "states.tsv"), sep="\t", index=False)
    seg.sample_state_matrix().to_csv(os.path.join(args.out, "occupancy.tsv"), sep="\t")

    for sample in ds.samples:
        seg.to_bed(sample).to_csv(
            os.path.join(args.out, f"{sample}_segments.bed"),
            sep="\t", header=False, index=False,
        )

    table = cs.quantify(seg)
    table.to_csv(os.path.join(args.out, "quantification.tsv.gz"), sep="\t", index=False)
    cs.summarize(table).to_csv(os.path.join(args.out, "summary.tsv"), sep="\t")

    if not args.no_figures:
        figs = {
            "emissions": viz.plot_emission_heatmap(seg),
            "transitions": viz.plot_transition_heatmap(seg),
            "coverage": viz.plot_state_coverage(seg, palette=args.palette),
            "confidence": viz.plot_confidence(table),
            "track": viz.plot_segmentation_track(seg, palette=args.palette),
            "attribution": viz.plot_attribution(cs.state_mark_importance(seg)),
        }
        if "bivalency_score" in table.columns:
            figs["bivalency"] = viz.plot_bivalency(table, sample=ds.samples[0])
        if len(ds.samples) > 1:
            a, b = ds.samples[0], ds.samples[1]
            figs["state_flow"] = viz.plot_state_flow(cs.state_flow(seg, a, b), a, b)
            figs["occupancy"] = viz.plot_occupancy_change(cs.occupancy_change(seg))
        paths = viz.save_all(figs, os.path.join(args.out, "figures"))
        print(f"[chromstate] wrote {len(paths)} figures", file=sys.stderr)

    print(f"[chromstate] done -> {args.out}", file=sys.stderr)
    return 0


def cmd_selectk(args) -> int:
    import chromstate as cs
    from chromstate import viz

    ds = _load(args)
    cands = [int(x) for x in args.candidates.split(",")]
    table = cs.select_n_states(ds, candidates=cands, emission=args.emission,
                               seed=args.seed, verbose=True)
    os.makedirs(args.out, exist_ok=True)
    table.to_csv(os.path.join(args.out, "model_selection.tsv"), sep="\t", index=False)
    viz.save_all({"model_selection": viz.plot_model_selection(table)}, args.out)
    best = int(table.loc[table["heldout_loglik_per_bin"].idxmax(), "n_states"])
    print(table.to_string(index=False))
    print(f"\n[chromstate] best K by held-out likelihood: {best}", file=sys.stderr)
    return 0


def cmd_compare(args) -> int:
    from chromstate import viz
    from chromstate.benchmark.run import run_full_benchmark

    os.makedirs(args.out, exist_ok=True)
    print(f"[chromstate] running benchmark (n_bins={args.n_bins}) …", file=sys.stderr)
    results = run_full_benchmark(n_bins=args.n_bins, seed=args.seed, n_restarts=args.restarts)
    for name, table in results.items():
        table.to_csv(os.path.join(args.out, f"{name}.tsv"), sep="\t", index=False)
        print(f"\n=== {name} ===")
        print(table.round(3).to_string(index=False))

    figs = {
        "method_comparison": viz.plot_method_comparison(results["methods"]),
        "normalization": viz.plot_normalization_experiment(results["normalization"]),
        "binarization": viz.plot_binarization_loss(results["binarization"]),
    }
    viz.save_all(figs, os.path.join(args.out, "figures"))
    print(f"\n[chromstate] done -> {args.out}", file=sys.stderr)
    return 0


def cmd_demo(args) -> int:
    from chromstate.demo import run_demo

    run_demo(args.out, n_bins=args.n_bins, seed=args.seed)
    return 0


def cmd_export_chromhmm(args) -> int:
    from chromstate.benchmark.baselines import export_for_chromhmm

    ds = _load(args)
    paths = export_for_chromhmm(ds, args.out)
    print(f"[chromstate] wrote {len(paths)} binarised files to {args.out}", file=sys.stderr)
    print("[chromstate] run real ChromHMM with:", file=sys.stderr)
    print(f"  java -mx4000M -jar ChromHMM.jar LearnModel {args.out} chromhmm_out "
          f"{args.n_states} <assembly>", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chromstate",
        description="Quantitative chromatin state analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--version", action="store_true", help="print version and exit")
    sub = p.add_subparsers(dest="command")

    f = sub.add_parser("fit", help="fit a model and write states, scores, and figures")
    _add_input_args(f)
    f.add_argument("-k", "--n-states", type=int, default=10)
    f.add_argument("--emission", default="auto", choices=["auto", "nb", "gaussian", "bernoulli"])
    f.add_argument("--restarts", type=int, default=3)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--palette", default="roadmap", choices=["roadmap", "cvd"])
    f.add_argument("--no-figures", action="store_true")
    f.add_argument("--verbose", action="store_true")
    f.add_argument("-o", "--out", default="chromstate_out")
    f.set_defaults(func=cmd_fit)

    s = sub.add_parser("selectk", help="choose the number of states by held-out likelihood")
    _add_input_args(s)
    s.add_argument("--candidates", default="4,6,8,10,12,15,18")
    s.add_argument("--emission", default="auto", choices=["auto", "nb", "gaussian", "bernoulli"])
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("-o", "--out", default="chromstate_out")
    s.set_defaults(func=cmd_selectk)

    c = sub.add_parser("compare", help="benchmark ChromState against competing methods")
    c.add_argument("--n-bins", type=int, default=4000)
    c.add_argument("--restarts", type=int, default=3)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("-o", "--out", default="benchmark_out")
    c.set_defaults(func=cmd_compare)

    d = sub.add_parser("demo", help="run the full simulated demo end to end")
    d.add_argument("--n-bins", type=int, default=3000)
    d.add_argument("--seed", type=int, default=1)
    d.add_argument("-o", "--out", default="demo_out")
    d.set_defaults(func=cmd_demo)

    e = sub.add_parser("export-chromhmm", help="write ChromHMM-format binarised input")
    _add_input_args(e)
    e.add_argument("-k", "--n-states", type=int, default=10)
    e.add_argument("-o", "--out", default="binarized")
    e.set_defaults(func=cmd_export_chromhmm)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "version", False):
        from chromstate import __version__

        print(__version__)
        return 0
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
