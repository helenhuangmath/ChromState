"""End-to-end demo: files on disk in, states, scores, figures, and a benchmark out.

Deliberately routed through the real file readers rather than in-memory arrays, so the
demo exercises the same IO path a user's BAM or BED files take. Run it with::

    chromstate demo -o demo_out
"""

from __future__ import annotations

import os
import time
from typing import Dict

import numpy as np
import pandas as pd


def run_demo(outdir: str, n_bins: int = 3000, seed: int = 1, n_states: int = 7) -> Dict[str, object]:
    import chromstate as cs
    from chromstate import viz
    from chromstate.benchmark.metrics import clustering_scores, detection_scores
    from chromstate.benchmark.run import (
        run_binarization_experiment,
        run_comparison,
        run_normalization_experiment,
        run_perturbation_experiment,
    )
    from chromstate.genome import BinSet, Genome
    from chromstate.simulate import simulate, write_demo_files

    t0 = time.time()
    os.makedirs(outdir, exist_ok=True)
    figdir = os.path.join(outdir, "figures")
    datadir = os.path.join(outdir, "data")
    log = []

    def say(msg: str) -> None:
        print(f"[demo] {msg}")
        log.append(msg)

    # -- 1. write simulated reads to disk, then read them back through the real IO path
    say("writing simulated BED read files + ChromHMM-style manifest")
    manifest_path = write_demo_files(datadir, seed=seed, n_bins=n_bins)
    genome = Genome.from_file(os.path.join(datadir, "chrom.sizes"))
    binset = BinSet.from_genome(genome, bin_size=200)
    entries = cs.read_manifest(manifest_path)
    # The simulated reads are already fragment centres, so no shift is applied.
    ds = cs.build_dataset(entries, binset, reader_kwargs={"bed": {"shift": 0}})
    say(f"loaded {ds.X.shape[0]:,} bin-rows x {ds.n_marks} marks from {len(entries)} files")

    # Regenerate the matching truth (same seed and shape as write_demo_files used).
    _, truth = simulate(n_bins=n_bins, n_chroms=2, seed=seed)
    if truth.states.shape[0] != ds.n_rows:
        raise RuntimeError(
            f"Truth has {truth.states.shape[0]} rows but the round-tripped dataset has "
            f"{ds.n_rows}; the IO path changed the binning."
        )
    round_trip_r = float(np.corrcoef(ds.X.ravel(), _reference_counts(n_bins, seed).ravel())[0, 1])
    say(f"BED round-trip fidelity vs in-memory counts: r = {round_trip_r:.4f}")

    # -- 2. normalise and fit
    cs.normalize(ds, method="median_ratio")
    say("fitting ChromState (negative-binomial HMM, joint across both samples)")
    seg = cs.fit(ds, n_states=n_states, seed=seed)
    say(f"restarts run: {seg.meta['n_restarts']}, "
        f"{seg.meta['restart_hits_at_best']} reached the best optimum")
    scores = clustering_scores(truth.states, seg.states, truth.n_states, n_states)
    say(f"state recovery vs ground truth: ARI={scores['ari']:.3f} "
        f"accuracy={scores['matched_accuracy']:.3f}")
    say(f"states found: {', '.join(seg.labels)}")

    # -- 3. quantify
    table = cs.quantify(seg)
    summary = cs.summarize(table)
    say(f"quantified {len(table):,} bins; "
        f"{(table['confidence'] < 0.8).mean():.1%} of bins are low-confidence calls")

    if truth.bivalent_mask is not None:
        det = detection_scores(table["bivalency_score"].to_numpy(), truth.bivalent_mask)
        say(f"bivalency detection: AUPRC={det['auprc']:.3f} "
            f"(random baseline {det['baseline_auprc']:.3f})")

    # -- 4. dynamics
    a, b = ds.samples[0], ds.samples[1]
    flow = cs.state_flow(seg, a, b)
    occ = cs.occupancy_change(seg)
    diff = cs.differential_state(seg, [a], [b])
    say(f"differential testing: {diff.note}")
    plan, ot_cost = cs.transport_flux(seg, a, b)
    say(f"optimal-transport distance between conditions: {ot_cost:.4f}")

    # -- 5. benchmark
    say("benchmarking against ChromHMM-like, Segway-like, GMM, k-means, naive signal")
    ds_bench, truth_bench = simulate(
        n_bins=n_bins, n_chroms=2, seed=seed, blur=0.35,
        perturbation={"responder_fraction": 0.5, "from_state": "TssBiv", "to_state": "TssA"},
    )
    cs.normalize(ds_bench, method="median_ratio")
    comparison = run_comparison(ds_bench, truth_bench, seed=seed, n_restarts=2)
    binar = run_binarization_experiment(ds_bench)
    normex = run_normalization_experiment(n_bins=n_bins, seed=seed)
    perturb = run_perturbation_experiment(n_bins=n_bins, seed=seed, n_states=n_states)

    # -- 6. figures
    say("rendering figures")
    figs = {
        "01_emissions": viz.plot_emission_heatmap(seg),
        "02_transitions": viz.plot_transition_heatmap(seg),
        "03_coverage": viz.plot_state_coverage(seg),
        "04_track": viz.plot_segmentation_track(seg, sample=a, chrom="chr1", max_bins=400),
        "05_confidence": viz.plot_confidence(table),
        "06_bivalency": viz.plot_bivalency(table, sample=a),
        "07_attribution": viz.plot_attribution(cs.state_mark_importance(seg)),
        "08_state_flow": viz.plot_state_flow(flow, a, b),
        "09_occupancy_change": viz.plot_occupancy_change(occ),
        "10_method_comparison": viz.plot_method_comparison(comparison),
        "11_binarization_loss": viz.plot_binarization_loss(binar),
        "12_normalization": viz.plot_normalization_experiment(normex),
    }

    cal_post, cal_correct = _calibration_inputs(ds_bench, truth_bench, seed=seed)
    if cal_post:
        figs["13_calibration"] = viz.plot_calibration(cal_post, cal_correct)

    paths = viz.save_all(figs, figdir)
    say(f"wrote {len(paths)} figures to {figdir}")

    # -- 7. tables + report
    tables = {
        "emissions": seg.emission_frame(),
        "transitions": seg.transition_frame(),
        "states": seg.state_table(),
        "occupancy": occ,
        "state_flow": flow,
        "summary": summary,
        "method_comparison": comparison,
        "binarization_loss": binar,
        "normalization": normex,
        "perturbation_detection": perturb,
        "top_bivalent_regions": cs.top_regions(table, "bivalency_score", n=50, sample=a),
    }
    tabdir = os.path.join(outdir, "tables")
    os.makedirs(tabdir, exist_ok=True)
    for name, tab in tables.items():
        tab.to_csv(os.path.join(tabdir, f"{name}.tsv"), sep="\t",
                   index=name in ("emissions", "transitions", "occupancy", "state_flow", "summary"))
    table.to_csv(os.path.join(tabdir, "quantification.tsv.gz"), sep="\t", index=False)

    elapsed = time.time() - t0
    report = _write_report(outdir, seg, scores, comparison, binar, normex, perturb,
                           table, flow, ot_cost, elapsed, log)
    say(f"report written to {report}")
    say(f"finished in {elapsed:.1f}s")

    return {
        "segmentation": seg,
        "quantification": table,
        "tables": tables,
        "figures": paths,
        "report": report,
        "recovery": scores,
    }


def _verdict(comparison: pd.DataFrame, tie_margin: float = 0.02) -> list:
    """Per-metric winners, reporting ties and stating where ChromState does *not* win.

    A single "best method" headline would be misleading here: the methods differ by
    metric, and on some of them ChromState is beaten. Naming the loss is the point --
    a benchmark that only reports its author's wins is not evidence of anything.
    """
    me = "ChromState (NB-HMM)"
    metrics = [
        ("ari", "state recovery (ARI)", True),
        ("biv_auprc", "bivalency detection (AUPRC)", True),
        ("signal_spearman", "tracks signal magnitude", True),
        ("overconfidence", "calibration (lower is better)", False),
        ("frac_saturated", "fraction of saturated calls (lower is better)", False),
    ]
    lines = ["**Per-metric outcome:**", ""]
    losses = []
    for col, label, higher_better in metrics:
        if col not in comparison.columns:
            continue
        sub = comparison[["method", col]].dropna()
        if sub.empty or me not in sub["method"].values:
            continue
        best_val = sub[col].max() if higher_better else sub[col].min()
        winners = sub.loc[(sub[col] - best_val).abs() <= tie_margin, "method"].tolist()
        mine = float(sub.loc[sub["method"] == me, col].iloc[0])
        mark = "won" if me in winners else "LOST"
        if me not in winners:
            losses.append((label, winners[0], best_val, mine))
        tie = " (tied with " + ", ".join(w for w in winners if w != me) + ")" if len(winners) > 1 and me in winners else ""
        lines.append(f"- {label}: ChromState {mine:.3f} — {mark}{tie}; best {best_val:.3f}")

    lines.append("")
    if losses:
        lines.append("**Where ChromState is beaten:**")
        lines.append("")
        for label, winner, best_val, mine in losses:
            lines.append(f"- {label}: {winner} scores {best_val:.3f} vs ChromState's {mine:.3f}.")
    else:
        lines.append("ChromState was not beaten on any reported metric on this dataset.")
    return lines


def _md_table(df: pd.DataFrame, index: bool = False) -> str:
    """Render a DataFrame as a GitHub markdown table.

    Hand-rolled rather than using ``DataFrame.to_markdown``, which needs the optional
    ``tabulate`` package -- not worth a dependency for the demo report.
    """
    frame = df.reset_index() if index else df
    cols = [str(c) for c in frame.columns]

    def cell(v) -> str:
        if isinstance(v, float):
            return "" if pd.isna(v) else f"{v:g}"
        return str(v)

    rows = [[cell(v) for v in row] for row in frame.itertuples(index=False)]
    widths = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c)
              for i, c in enumerate(cols)]
    out = ["| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |",
           "| " + " | ".join("-" * w for w in widths) + " |"]
    out += ["| " + " | ".join(v.ljust(w) for v, w in zip(r, widths)) + " |" for r in rows]
    return "\n".join(out)


def _reference_counts(n_bins: int, seed: int) -> np.ndarray:
    from chromstate.simulate import simulate

    ds, _ = simulate(n_bins=n_bins, n_chroms=2, seed=seed)
    return ds.X


def _calibration_inputs(ds, truth, seed: int):
    """Posteriors and correctness masks for the reliability diagram."""
    from chromstate.benchmark.baselines import run_chromhmm_like, run_chromstate, run_segway_like
    from chromstate.benchmark.metrics import match_states

    posteriors, correct = {}, {}
    for fn in (run_chromstate, run_chromhmm_like, run_segway_like):
        res = fn(ds, n_states=truth.n_states, seed=seed, n_restarts=2)
        if res.posterior is None:
            continue
        pred = np.argmax(res.posterior, axis=1)
        mapping, _ = match_states(truth.states, pred, truth.n_states, res.n_states)
        posteriors[res.name] = res.posterior
        correct[res.name] = np.array([mapping.get(int(p), 0) for p in pred]) == truth.states
    return posteriors, correct


def _write_report(outdir, seg, scores, comparison, binar, normex, perturb,
                  table, flow, ot_cost, elapsed, log) -> str:
    path = os.path.join(outdir, "REPORT.md")
    verdict = _verdict(comparison)
    lines = [
        "# ChromState demo report",
        "",
        f"Generated in {elapsed:.1f}s on simulated data with known ground truth.",
        "",
        "## Run log",
        "",
        *[f"- {m}" for m in log],
        "",
        "## State recovery",
        "",
        f"- Adjusted Rand index vs truth: **{scores['ari']:.3f}**",
        f"- Accuracy after optimal state matching: **{scores['matched_accuracy']:.3f}**",
        f"- States discovered: {', '.join(seg.labels)}",
        "",
        "## Method comparison",
        "",
        _md_table(comparison.round(3)),
        "",
        *verdict,
        "",
        "## What binarisation discards",
        "",
        _md_table(binar.round(3)),
        "",
        "## Normalisation under genome-wide mark loss",
        "",
        _md_table(normex.round(3)),
        "",
        "## Detecting perturbation responders",
        "",
        _md_table(perturb.round(3)),
        "",
        "## State flow between conditions",
        "",
        _md_table(flow.round(3), index=True),
        "",
        f"Optimal-transport distance between conditions: {ot_cost:.4f}",
        "",
        "## Figures",
        "",
        *[f"- `figures/{n}.pdf`" for n in sorted(
            os.path.splitext(f)[0] for f in os.listdir(os.path.join(outdir, "figures"))
            if f.endswith(".pdf")
        )],
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path
