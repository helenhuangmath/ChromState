"""Command line interface.

    chromstate fit      --manifest cellmarkfiletable.txt --chrom-sizes hg38.sizes -k 12 -o out/
    chromstate selectk  --manifest ... --chrom-sizes ... --candidates 6,8,10,12,15
    chromstate compare  --out benchmark/
    chromstate demo     --out demo/
    chromstate export-chromhmm --manifest ... --chrom-sizes ... -o binarized/
    chromstate run      --manifest ... --chrom-sizes ... --gtf genes.gtf --mode gene \
                        --resolution 10 --fit-bin-size 200 -k 10 --groups "A=a1,a2;B=b1,b2" -o run/
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
    definitions = None
    if getattr(args, "states", None):
        definitions = (cs.StateDefinitions.default(ds.marks) if args.states == "default"
                       else cs.StateDefinitions.from_file(args.states, ds.marks))
        print(f"[chromstate] {len(definitions)} defined states ({args.anchor}): "
              f"{', '.join(definitions.states)}", file=sys.stderr)
    seg = cs.fit(ds, n_states=args.n_states, emission=args.emission,
                 n_restarts=args.restarts, seed=args.seed, verbose=args.verbose,
                 definitions=definitions, anchor=getattr(args, "anchor", "seed"))
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


def cmd_run(args) -> int:
    import chromstate as cs
    from chromstate.pipeline import parse_groups, read_expression_matrix
    from chromstate.regions import RegionScheme

    scheme = RegionScheme(promoter_upstream=args.promoter_upstream,
                          promoter_downstream=args.promoter_downstream,
                          regulatory_flank=args.flank)
    cfg = cs.RunConfig(
        mode=args.mode, resolution=args.resolution, fit_bin_size=args.fit_bin_size,
        n_states=args.n_states, n_fit_states=args.fit_states, scheme=scheme, norm=args.norm,
        emission=args.emission, n_restarts=args.restarts, fit_fraction=args.fit_fraction,
        seed=args.seed, tracks=args.tracks, write_scores=None if not args.no_scores else False,
        figures=not args.no_figures, palette=args.palette, smooth_bp=args.smooth_bp, verbose=True,
    )
    features = None
    if args.gtf:
        features = cs.FeatureSet.from_gtf(args.gtf)
    elif args.genes_bed:
        features = cs.FeatureSet.from_bed(args.genes_bed)
    if cfg.mode == "gene" and features is None:
        raise SystemExit("--mode gene needs --gtf or --genes-bed")
    chroms = args.chroms.split(",") if args.chroms else None
    if args.norm == "spikein":
        if not args.spikein:
            raise SystemExit("--norm spikein requires --spikein <file>")
        tab = pd.read_csv(args.spikein, sep="\t", header=None, names=["sample", "mark", "reads"])
        cfg.spikein = {(r["sample"], r["mark"]): float(r["reads"]) for _, r in tab.iterrows()}
    from chromstate.dataset import estimate_memory
    from chromstate.genome import Genome

    genome = Genome.from_file(args.chrom_sizes)
    n_samples = len({e.sample for e in cs.read_manifest(args.manifest)})
    n_marks = len({e.mark for e in cs.read_manifest(args.manifest)})
    span = None
    if features is not None and cfg.mode == "gene":
        win = scheme.regulatory(features)
        span = int((win["end"] - win["start"]).sum())
    est = estimate_memory(sum(genome.sizes.values()), cfg.resolution, n_samples, n_marks,
                          cfg.n_states, mode=cfg.mode, gene_window_bp=span)
    print(f"[chromstate] {est['bins_per_sample']:,.0f} bins per sample at {cfg.resolution} bp; "
          f"in-memory run needs about {est['in_memory_total_gb']:.1f} GB "
          f"(counts {est['counts_gb']:.1f}, size factors {est['size_factors_gb']:.1f}, "
          f"scores {est['posterior_float32_gb']:.1f}); streaming peak about "
          f"{est['streaming_peak_gb']:.1f} GB", file=sys.stderr)
    if cfg.mode == "genome" and not args.stream and est["in_memory_total_gb"] > 32:
        print("[chromstate] hint: add --stream, or raise --resolution to 50", file=sys.stderr)
    groups = parse_groups(args.groups)
    if args.stream:
        if cfg.mode != "genome":
            raise SystemExit("--stream is for --mode genome")
        cs.run_genome_streaming(args.manifest, args.chrom_sizes, args.out, cfg, chroms=chroms,
                                blacklist=args.blacklist, groups=groups)
        return 0
    fine = cs.build_fine_dataset(args.manifest, args.chrom_sizes, cfg, features=features,
                                 blacklist=args.blacklist, chroms=chroms)
    expression = read_expression_matrix(args.expression, log=not args.expression_is_log) \
        if args.expression else None
    lfc = contrast = None
    if args.lfc:
        from chromstate.pipeline import read_lfc

        if not args.contrast or len(args.contrast.split(",")) != 2:
            raise SystemExit("--lfc needs --contrast A,B naming two conditions from --groups")
        contrast = tuple(c.strip() for c in args.contrast.split(","))
        if groups is None or any(c not in groups for c in contrast):
            raise SystemExit(f"--contrast {contrast} must name conditions defined by --groups")
        lfc = read_lfc(args.lfc)
    cs.run_pipeline(fine, args.out, cfg, features=features, groups=groups, expression=expression,
                    lfc=lfc, contrast=contrast)
    return 0


def cmd_predict(args) -> int:
    import chromstate as cs
    from chromstate.pipeline import parse_groups, read_expression_matrix, read_lfc

    gene_q = pd.read_csv(args.genes, sep="\t")
    expression = lfc = contrast = None
    groups = parse_groups(args.groups) if args.groups else None
    if args.lfc:
        if not args.contrast or len(args.contrast.split(",")) != 2:
            raise SystemExit("--lfc needs --contrast A,B (B over A)")
        a, b = [c.strip() for c in args.contrast.split(",")]
        if groups is not None:
            ga, gb = groups.get(a, [a]), groups.get(b, [b])
        else:
            ga, gb = [a], [b]
        lfc = read_lfc(args.lfc)
        contrast = (ga, gb)
    elif args.expression:
        expression = read_expression_matrix(args.expression, log=not args.expression_is_log)
    else:
        raise SystemExit("give --expression (level task) or --lfc with --contrast (change task)")
    res = cs.predict_expression(gene_q, expression=expression, lfc=lfc, contrast=contrast,
                                model=args.model, n_splits=args.folds, seed=args.seed,
                                parts=not args.no_parts)
    files = res.write(args.out, figures=not args.no_figures, palette=args.palette)
    print(res.note)
    print(f"[chromstate] wrote {len(files)} files to {args.out}")
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
    f.add_argument("--states", default=None,
                   help="state definitions file (TSV or 'State: mark+ mark- ...' lines); "
                        "'default' uses the Roadmap-style vocabulary for the marks present")
    f.add_argument("--anchor", default="seed", choices=["seed", "fixed"],
                   help="with --states: seed EM from the definitions, or pin the defined marks")
    f.add_argument("--palette", default="scirep2025", choices=["scirep2025", "nature", "roadmap", "cvd"])
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

    r = sub.add_parser("run", help="fit at 200 bp, score every 10 bp, summarise, compare, link genes")
    _add_input_args(r)
    r.add_argument("--mode", default="gene", choices=["gene", "genome"],
                   help="gene: tile each gene +/- flank; genome: tile every chromosome (default gene)")
    r.add_argument("--resolution", type=int, default=10, help="scoring resolution in bp (default 10)")
    r.add_argument("--fit-bin-size", type=int, default=200,
                   help="bin width the model is fitted at; a multiple of --resolution (default 200)")
    r.add_argument("-k", "--n-states", type=int, default=10, help="final number of states")
    r.add_argument("--fit-states", type=int, default=None,
                   help="fit this many states and collapse to -k (default: fit -k directly)")
    r.add_argument("--gtf", default=None, help="gene annotation (GTF/GFF, gene records)")
    r.add_argument("--genes-bed", default=None, help="gene annotation as BED6")
    r.add_argument("--promoter-upstream", type=int, default=1000)
    r.add_argument("--promoter-downstream", type=int, default=1000)
    r.add_argument("--flank", type=int, default=50_000,
                   help="regulatory window: gene +/- this many bp (default 50000)")
    r.add_argument("--groups", default=None,
                   help='conditions: "A=s1,s2;B=s3,s4" or a sample<TAB>condition file')
    r.add_argument("--expression", default=None,
                   help="genes x samples TSV (gene_id + one column per sample)")
    r.add_argument("--expression-is-log", action="store_true",
                   help="the expression table is already on a log scale")
    r.add_argument("--lfc", default=None,
                   help="per-gene log2 fold change (gene_id<TAB>log2fc, or a DESeq2/edgeR table "
                        "with a log2FoldChange/logFC column) for the expression-change task")
    r.add_argument("--contrast", default=None,
                   help="the two conditions --lfc compares, as 'A,B' (B over A); names from --groups")
    r.add_argument("--emission", default="auto", choices=["auto", "nb", "gaussian", "bernoulli"])
    r.add_argument("--restarts", type=int, default=3)
    r.add_argument("--fit-fraction", type=float, default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--stream", action="store_true",
                   help="genome mode: one chromosome in memory at a time, scores to a uint8 tensor on disk")
    r.add_argument("--smooth-bp", type=int, default=0,
                   help="reassign hard-call segments shorter than this to a neighbouring state (0 = off)")
    r.add_argument("--tracks", action="store_true", help="write a bedGraph score track per sample and state")
    r.add_argument("--no-scores", action="store_true", help="skip the per-bin score table")
    r.add_argument("--no-figures", action="store_true")
    r.add_argument("--palette", default="scirep2025", choices=["scirep2025", "nature", "roadmap", "cvd"])
    r.add_argument("-o", "--out", default="chromstate_run")
    r.set_defaults(func=cmd_run)

    q = sub.add_parser("predict", help="predict expression from per-gene state tables; rank states")
    q.add_argument("--genes", required=True,
                   help="gene_quantification table (genes/gene_quantification.tsv.gz from `chromstate run`)")
    q.add_argument("--expression", default=None, help="genes x samples TSV for the expression-level task")
    q.add_argument("--expression-is-log", action="store_true")
    q.add_argument("--lfc", default=None, help="per-gene log2 fold change (or DESeq2/edgeR table)")
    q.add_argument("--contrast", default=None, help="'A,B' (B over A): samples or --groups conditions")
    q.add_argument("--groups", default=None, help='conditions: "A=s1,s2;B=s3,s4"')
    q.add_argument("--model", default="hgb", choices=["hgb", "xgboost", "ridge"])
    q.add_argument("--folds", type=int, default=5)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--no-parts", action="store_true", help="skip the promoter/enhancer/body ranking")
    q.add_argument("--no-figures", action="store_true")
    q.add_argument("--palette", default="scirep2025", choices=["scirep2025", "nature", "roadmap", "cvd"])
    q.add_argument("-o", "--out", default="chromstate_predict")
    q.set_defaults(func=cmd_predict)

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
