"""One call from reads to every table and figure, in gene-level or genome-wide mode.

``gene`` mode tiles only each gene's regulatory window (gene +/- 50 kb by default) at
the fine resolution (10 bp by default), so the whole run -- fitting, scoring, summaries,
gene tables, enhancer ranking -- stays small enough for a laptop on a few thousand
genes and for one node on a whole annotation. ``genome`` mode tiles every chromosome at
the fine resolution; the model is still fitted at the coarse width and the fine pass
streams one chromosome at a time.

Both modes run the same steps:

1. bin the input at ``resolution`` bp, pool to ``fit_bin_size`` bp, normalise there;
2. fit ``n_fit_states`` states on the pooled bins and, if asked, collapse them to
   ``n_states`` (:mod:`chromstate.collapse`);
3. score every fine bin on the 0-100 scale (:mod:`chromstate.resolution`);
4. summarise amount and strength per state, per sample and per condition, and compare
   conditions (:mod:`chromstate.summary`);
5. with genes: quantify promoter, distal and body per gene, find the dominant-state
   patterns and draw the alluvials (:mod:`chromstate.patterns`), call candidate
   elements from the state track (:mod:`chromstate.regions`);
6. with expression: rank states and gene parts by what they explain, and rank each
   gene's enhancers beyond its promoter.

Everything is written under ``out/`` with a ``REPORT.md`` that lists what was done and
with which settings.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .dataset import ChromStateDataset
from .features import FeatureSet
from .regions import RegionScheme
from .resolution import DEFAULT_FIT_BIN_SIZE, DEFAULT_RESOLUTION


@dataclass
class RunConfig:
    """Settings of one pipeline run; every default is the one the README names."""

    mode: str = "gene"
    resolution: int = DEFAULT_RESOLUTION
    fit_bin_size: int = DEFAULT_FIT_BIN_SIZE
    n_states: int = 10
    n_fit_states: Optional[int] = None
    scheme: RegionScheme = field(default_factory=RegionScheme)
    norm: str = "median_ratio"
    spikein: Optional[dict] = None
    emission: str = "auto"
    n_restarts: int = 3
    max_restarts: int = 12
    fit_fraction: Optional[float] = None
    seed: int = 0
    tracks: bool = False
    write_scores: Optional[bool] = None
    figures: bool = True
    palette: str = "scirep2025"
    score_scale: int = 100
    min_element_score: float = 50.0
    smooth_bp: int = 0
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ("gene", "genome"):
            raise ValueError(f"mode must be 'gene' or 'genome', got {self.mode!r}")
        if self.fit_bin_size % self.resolution:
            raise ValueError(
                f"fit_bin_size ({self.fit_bin_size}) must be a multiple of resolution ({self.resolution})")
        if self.n_fit_states is not None and self.n_fit_states < self.n_states:
            raise ValueError("n_fit_states must be >= n_states")
        if self.write_scores is None:
            self.write_scores = self.mode == "gene"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["scheme"] = asdict(self.scheme)
        return d


def _log(cfg: RunConfig, msg: str) -> None:
    if cfg.verbose:
        print(f"[chromstate] {msg}", file=sys.stderr, flush=True)


def _save(frame: pd.DataFrame, path: str, index: bool = False) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.to_csv(path, sep="\t", index=index, float_format="%.5g")
    return path


# ---------------------------------------------------------------------------
# Building the fine dataset
# ---------------------------------------------------------------------------


def fine_binset(cfg: RunConfig, genome, features: Optional[FeatureSet] = None,
                chroms: Optional[Sequence[str]] = None):
    """The tiling for the run: gene windows (gene mode) or whole chromosomes."""
    from .genome import BinSet

    if cfg.mode == "gene":
        if features is None:
            raise ValueError("gene mode needs a gene annotation (features)")
        windows = cfg.scheme.regulatory(features)
        if chroms is not None:
            windows = windows.loc[windows["chrom"].isin(list(chroms))]
        return BinSet.from_regions(windows, bin_size=cfg.resolution, genome=genome)
    return BinSet.from_genome(genome, bin_size=cfg.resolution, chroms=chroms)


def build_fine_dataset(
    manifest: str, chrom_sizes: str, cfg: RunConfig, features: Optional[FeatureSet] = None,
    blacklist: Optional[str] = None, chroms: Optional[Sequence[str]] = None,
) -> ChromStateDataset:
    """Read a manifest into a dataset tiled at ``cfg.resolution``."""
    from .dataset import build_dataset
    from .genome import Genome
    from .io.counts import read_bed_regions
    from .io.manifest import read_manifest

    genome = Genome.from_file(chrom_sizes)
    binset = fine_binset(cfg, genome, features, chroms)
    entries = read_manifest(manifest)
    bl = read_bed_regions(blacklist) if blacklist else None
    _log(cfg, f"{len(entries)} input files, {len(binset):,} bins of {cfg.resolution} bp "
              f"({cfg.mode} mode)")
    ds = build_dataset(entries, binset, blacklist=bl)
    ds.meta["chrom_sizes"] = dict(genome.sizes)
    return ds


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Handles to everything a run produced (tables are also on disk under ``out``)."""

    out: str
    config: RunConfig
    coarse: object
    fine: object
    summary: pd.DataFrame
    condition: Optional[object] = None
    gene_table: Optional[pd.DataFrame] = None
    dominant: Optional[pd.DataFrame] = None
    elements: Optional[pd.DataFrame] = None
    importance: Optional[object] = None
    parts: Optional[pd.DataFrame] = None
    contributions: Optional[dict] = None
    files: List[str] = field(default_factory=list)
    report: str = ""


def run_pipeline(
    fine: ChromStateDataset,
    out: str,
    cfg: Optional[RunConfig] = None,
    features: Optional[FeatureSet] = None,
    groups: Optional[Dict[str, Sequence[str]]] = None,
    expression: Optional[pd.DataFrame] = None,
    lfc: Optional[pd.Series] = None,
    contrast: Optional[Sequence[str]] = None,
) -> RunResult:
    """Run every step on a dataset already tiled at the fine resolution.

    Parameters
    ----------
    fine
        Dataset at ``cfg.resolution`` bp (from :func:`build_fine_dataset`, or any
        dataset whose bins are that wide).
    features
        Gene annotation. Required for gene-level tables; in genome mode it is optional.
    groups
        ``{condition: [sample, ...]}``. Defaults to one condition per sample.
    expression
        Genes x samples on a log scale (index or column ``gene_id``, columns named after
        samples). Enables the state-importance, part-ranking and enhancer-ranking steps.
    lfc, contrast
        Alternatively a per-gene log2 fold change (``Series`` indexed by ``gene_id``) and
        the two condition names it compares, for the expression-change task.
    """
    from . import viz
    from .collapse import collapse_states, merge_table
    from .model.core import fit
    from .norm import normalize
    from .patterns import default_state_classes, gene_dominant_states, pattern_counts
    from .resolution import (
        bin_width,
        coarsen_dataset,
        copy_size_factors,
        decode_at_resolution,
        state_scores,
        write_score_tracks,
    )
    from .summary import compare_conditions, state_summary

    cfg = cfg or RunConfig()
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    files: List[str] = []
    report: List[str] = [f"# ChromState run ({cfg.mode} mode)", "",
                         f"Resolution {cfg.resolution} bp, model fitted at {cfg.fit_bin_size} bp. "
                         f"{cfg.scheme.describe()}.", ""]
    groups = {k: list(v) for k, v in (groups or {s: [s] for s in fine.samples}).items()}
    for cond, reps in groups.items():
        for s in reps:
            if s not in fine.samples:
                raise ValueError(f"condition {cond!r} names unknown sample {s!r}; samples are {fine.samples}")

    # 1. pool and normalise ------------------------------------------------
    width = bin_width(fine)
    if width != cfg.resolution:
        warnings.warn(f"dataset bins are {width} bp, config says {cfg.resolution}; using {width}",
                      RuntimeWarning, stacklevel=2)
    factor = cfg.fit_bin_size // width
    coarse = coarsen_dataset(fine, factor)
    if cfg.norm != "none":
        normalize(coarse, method=cfg.norm, spikein=cfg.spikein)
        copy_size_factors(coarse, fine)
        if "norm_warning" in coarse.meta:
            _log(cfg, "WARNING: " + str(coarse.meta["norm_warning"]))
    _log(cfg, f"pooled {fine.n_rows:,} fine bins into {coarse.n_rows:,} bins of {cfg.fit_bin_size} bp")

    # 2. fit (and collapse) ---------------------------------------------------
    n_fit = cfg.n_fit_states or cfg.n_states
    _log(cfg, f"fitting {n_fit} states …")
    seg = fit(coarse, n_states=n_fit, emission=cfg.emission, n_restarts=cfg.n_restarts,
              max_restarts=cfg.max_restarts, seed=cfg.seed, fit_fraction=cfg.fit_fraction,
              verbose=False)
    report.append(f"Fitted {n_fit} states on {coarse.n_rows:,} pooled bins "
                  f"(log-likelihood {seg.loglik:,.1f}, {seg.meta['n_restarts']} restarts, "
                  f"best reached {seg.meta['restart_hits_at_best']} times).")
    if n_fit > cfg.n_states:
        seg = collapse_states(seg, cfg.n_states)
        files.append(_save(merge_table(seg), os.path.join(out, "model", "collapsed_members.tsv")))
        report.append(f"Collapsed to {cfg.n_states} states by emission similarity; see "
                      f"`model/collapsed_members.tsv`.")
    files.append(_save(seg.emission_frame(), os.path.join(out, "model", "emissions.tsv"), index=True))
    files.append(_save(seg.transition_frame(), os.path.join(out, "model", "transitions.tsv"), index=True))
    files.append(_save(seg.state_table(), os.path.join(out, "model", "states.tsv")))

    # 3. score fine ----------------------------------------------------------
    _log(cfg, f"scoring {fine.n_rows:,} bins at {width} bp …")
    fine_seg = decode_at_resolution(seg, fine)
    if cfg.smooth_bp > 0:
        from .collapse import smooth_segments

        fine_seg = smooth_segments(fine_seg, min_bp=cfg.smooth_bp)
    if cfg.write_scores:
        scores = state_scores(fine_seg, scale=cfg.score_scale)
        files.append(_save(scores, os.path.join(out, "scores", "state_scores.tsv.gz")))
        del scores
    if cfg.tracks:
        files += write_score_tracks(fine_seg, os.path.join(out, "scores", "tracks"), scale=cfg.score_scale)
    for s in fine.samples:
        files.append(_save(fine_seg.to_bed(s), os.path.join(out, "scores", f"{s}_segments.bed")))

    # 4. summaries and condition comparison --------------------------------
    summary = state_summary(fine_seg)
    files.append(_save(summary, os.path.join(out, "summary", "state_summary_per_sample.tsv")))
    figs = {}
    comparison = None
    if len(groups) >= 2:
        comparison = compare_conditions(fine_seg, groups)
        files.append(_save(comparison.per_condition, os.path.join(out, "summary", "state_summary_per_condition.tsv")))
        files.append(_save(comparison.differences, os.path.join(out, "summary", "condition_differences.tsv")))
        for (a, b), flow in comparison.flows.items():
            files.append(_save(flow, os.path.join(out, "summary", f"flow_bp_{a}_to_{b}.tsv"), index=True))
        report.append(comparison.note)
        from .summary import flow_test, state_tests

        tests = state_tests(fine_seg, groups, seed=cfg.seed)
        files.append(_save(tests, os.path.join(out, "summary", "state_tests.tsv")))
        report.append(tests.attrs["note"])
        if cfg.figures:
            figs["state_tests"] = viz.plot_state_tests(tests, fine_seg, palette=cfg.palette)
        ref = comparison.reference
        for c in groups:
            if c != ref:
                files.append(_save(flow_test(fine_seg, groups[ref], groups[c], seed=cfg.seed),
                                   os.path.join(out, "summary", f"flow_test_{ref}_to_{c}.tsv")))
    if cfg.figures:
        figs["emissions"] = viz.plot_emission_heatmap(seg)
        figs["state_percentages"] = viz.plot_state_percentages(
            comparison.per_condition if comparison else summary, fine_seg, palette=cfg.palette)
        if "strength" in summary.columns:
            figs["state_strength"] = viz.plot_state_strength(
                comparison.per_condition if comparison else summary, fine_seg, palette=cfg.palette)
        if comparison is not None:
            figs["condition_differences"] = viz.plot_condition_differences(
                comparison.differences, fine_seg, palette=cfg.palette)
            try:
                figs["region_alluvial"] = viz.plot_region_alluvial(fine_seg, groups, palette=cfg.palette)
                figs["region_alluvial_by_class"] = viz.plot_region_alluvial(
                    fine_seg, groups, by="class", palette=cfg.palette)
            except ValueError as exc:  # e.g. bins not matched across samples
                _log(cfg, f"region alluvial skipped: {exc}")

    # 5. genes ---------------------------------------------------------------
    gene_table = dominant = elements = importance = parts = contributions = None
    if features is not None:
        from .regions import (
            call_elements,
            enhancer_contributions,
            gene_regulatory_quantification,
            rank_parts_for_expression,
            summarise_contributions,
        )

        _log(cfg, f"quantifying {len(features):,} genes …")
        gene_table = gene_regulatory_quantification(fine_seg, features, cfg.scheme)
        files.append(_save(gene_table, os.path.join(out, "genes", "gene_quantification.tsv.gz")))
        classes = default_state_classes(fine_seg)
        dominant = gene_dominant_states(gene_table, groups, part="P", state_classes=classes)
        files.append(_save(dominant, os.path.join(out, "genes", "promoter_dominant_states.tsv")))
        files.append(_save(pattern_counts(dominant), os.path.join(out, "genes", "promoter_pattern_counts.tsv")))
        dom_e = None
        if "E_n_bins" in gene_table.columns:
            dom_e = gene_dominant_states(gene_table, groups, part="E",
                                         state_classes=default_state_classes(fine_seg, promoter=False))
            files.append(_save(dom_e, os.path.join(out, "genes", "distal_dominant_states.tsv")))
            files.append(_save(pattern_counts(dom_e), os.path.join(out, "genes", "distal_pattern_counts.tsv")))
            if "category" in dom_e.columns:
                files.append(_save(pattern_counts(dom_e, by="category"),
                                   os.path.join(out, "genes", "distal_category_counts.tsv")))
        if "category" in dominant.columns:
            files.append(_save(pattern_counts(dominant, by="category"),
                               os.path.join(out, "genes", "promoter_category_counts.tsv")))
        try:
            elements = call_elements(fine_seg, features, cfg.scheme, min_score=cfg.min_element_score)
            files.append(_save(elements, os.path.join(out, "genes", "candidate_elements.tsv")))
            report.append(f"{len(elements):,} candidate elements called from the state track "
                          f"({int(elements['is_promoter'].sum()):,} at promoters, the rest distal).")
        except ValueError as exc:
            _log(cfg, f"element calling skipped: {exc}")
        if cfg.figures and len(groups) >= 2:
            figs["gene_alluvial"] = viz.plot_gene_alluvial(dominant, fine_seg, palette=cfg.palette)
            if "category" in dominant.columns:
                figs["gene_alluvial_by_class"] = viz.plot_gene_alluvial(dominant, fine_seg, by="class",
                                                                        palette=cfg.palette)
            figs["pattern_counts"] = viz.plot_pattern_counts(pattern_counts(dominant))
            if dom_e is not None:
                figs["gene_alluvial_distal"] = viz.plot_gene_alluvial(dom_e, fine_seg, palette=cfg.palette)
                if "category" in dom_e.columns:
                    figs["gene_alluvial_distal_by_class"] = viz.plot_gene_alluvial(
                        dom_e, fine_seg, by="class", palette=cfg.palette)

        # 5b. differential state at promoters (replicate test) --------------
        reps_ok = [c for c in groups if len(groups[c]) >= 2]
        if len(groups) >= 2 and len(reps_ok) >= 2:
            from .dynamics import differential_regions, replicate_noise_floor

            ref_c = comparison.reference if comparison else list(groups)[0]
            try:
                floor = replicate_noise_floor(fine_seg, {c: groups[c] for c in reps_ok})
                prom = cfg.scheme.promoters(features).rename(columns={"name": "gene_name"})
                prom["name"] = prom["gene_id"]
                for c in groups:
                    if c == ref_c or len(groups[c]) < 2:
                        continue
                    diff = differential_regions(fine_seg, groups[ref_c], groups[c], regions=prom,
                                                noise_floor=floor, min_bins=1)
                    files.append(_save(diff.table, os.path.join(out, "differential",
                                                                f"promoters_{ref_c}_vs_{c}.tsv")))
                    report.append(f"Differential promoters {ref_c} vs {c}: {diff.note}")
                    if cfg.figures:
                        figs[f"volcano_promoters_{ref_c}_vs_{c}"] = viz.plot_change_volcano(
                            diff.table, noise_floor=floor.threshold)
                        figs[f"noise_floor_{ref_c}_vs_{c}"] = viz.plot_noise_floor(
                            floor, diff.table["tv_distance"].to_numpy())
            except ValueError as exc:
                _log(cfg, f"differential promoters skipped: {exc}")
            try:
                from .compare import differential_report

                prom = cfg.scheme.promoters(features).rename(columns={"name": "gene_name"})
                prom["name"] = prom["gene_id"]
                rep_groups = {c: groups[c] for c in groups if len(groups[c]) >= 2 or c == ref_c}
                dr = differential_report(fine_seg, rep_groups, regions=prom, reference=ref_c, min_bins=1,
                                         seed=cfg.seed)  # computes a region-level noise floor
                files += dr.write(os.path.join(out, "differential"), prefix="promoter_", seg=fine_seg,
                                  regions=prom, figures=cfg.figures, palette=cfg.palette)
                report.append("Per-region x per-state tests written to `differential/promoter_region_state_tests.tsv`; "
                              "per-state summary in `differential/promoter_state_summary.tsv`.")
            except Exception as exc:
                _log(cfg, f"differential report skipped: {exc}")

        # 6. expression ------------------------------------------------------
        if expression is not None or lfc is not None:
            from .regulation import rank_states_for_expression

            expr = None
            if expression is not None:
                expr = expression.set_index("gene_id") if "gene_id" in expression.columns else expression
                expr = expr[[c for c in expr.columns if c in fine.samples]]
            _log(cfg, "ranking states and gene parts for expression …")
            try:
                if lfc is not None:
                    if contrast is None or len(contrast) != 2:
                        raise ValueError("lfc needs contrast=(condition_a, condition_b)")
                    ga, gb = groups[contrast[0]], groups[contrast[1]]
                    importance, rep = rank_states_for_expression(gene_table, lfc=lfc, contrast=(ga, gb),
                                                                 seed=cfg.seed)
                    parts = rank_parts_for_expression(gene_table, lfc=lfc, contrast=(ga, gb), seed=cfg.seed)
                else:
                    importance, rep = rank_states_for_expression(gene_table, expression=expr, seed=cfg.seed)
                    parts = rank_parts_for_expression(gene_table, expression=expr, seed=cfg.seed)
                files.append(_save(importance.table, os.path.join(out, "expression", "state_importance.tsv")))
                files.append(_save(rep.scores, os.path.join(out, "expression", "representation_scores.tsv")))
                files.append(_save(parts, os.path.join(out, "expression", "part_ranking.tsv")))
                if cfg.figures:
                    figs["state_importance"] = viz.plot_state_importance(importance.table, fine_seg,
                                                                         palette=cfg.palette)
                    figs["part_ranking"] = viz.plot_part_ranking(parts)
                report.append(importance.note)
                top = importance.table.head(3)
                report.append("Top states for expression: " + ", ".join(
                    f"{r.state} (unique R² {r.unique_r2:.3f}, direction {r.direction:+.2f})"
                    for r in top.itertuples()) + ".")
            except Exception as exc:
                _log(cfg, f"state importance skipped: {exc}")
            try:
                from .features import gene_state_change, gene_state_matrix
                from .rna import expression_association, link_expression

                ref_c = comparison.reference if comparison else None
                pairs_to_test = []
                if lfc is not None and contrast is not None:
                    pairs_to_test.append((contrast[0], contrast[1], lfc))
                elif expr is not None and ref_c is not None:
                    for c in groups:
                        if c == ref_c:
                            continue
                        a = expr[[s for s in groups[ref_c] if s in expr.columns]].mean(axis=1)
                        b = expr[[s for s in groups[c] if s in expr.columns]].mean(axis=1)
                        pairs_to_test.append((ref_c, c, (b - a).dropna()))
                gsm = gene_state_matrix(fine_seg, features, part="promoter",
                                        upstream=cfg.scheme.promoter_upstream,
                                        downstream=cfg.scheme.promoter_downstream)
                for a_c, b_c, fold in pairs_to_test:
                    change = gene_state_change(gsm, groups[a_c], groups[b_c])
                    rna = pd.DataFrame({"gene_id": fold.index, "log2fc": fold.to_numpy(dtype=float)})
                    linked = link_expression(change, rna)
                    assoc = expression_association(linked, n_permutations=500, seed=cfg.seed)
                    files.append(_save(linked, os.path.join(out, "expression", f"promoter_change_vs_rna_{a_c}_vs_{b_c}.tsv")))
                    files.append(_save(assoc.table, os.path.join(out, "expression", f"state_rna_association_{a_c}_vs_{b_c}.tsv")))
                    report.append(f"Chromatin-vs-RNA change {a_c} vs {b_c}: {assoc.note}")
                    if cfg.figures:
                        figs[f"expression_link_{a_c}_vs_{b_c}"] = viz.plot_expression_link(linked, assoc.table)
            except Exception as exc:
                _log(cfg, f"chromatin-vs-RNA association skipped: {exc}")
            if expr is not None and elements is not None:
                try:
                    contributions = enhancer_contributions(fine_seg, features, expr, elements=elements,
                                                           scheme=cfg.scheme, seed=cfg.seed)
                    files.append(_save(contributions["importance"],
                                       os.path.join(out, "expression", "enhancer_importance.tsv")))
                    files.append(_save(summarise_contributions(contributions),
                                       os.path.join(out, "expression", "gene_enhancer_summary.tsv")))
                    report.append(contributions["note"])
                except Exception as exc:
                    _log(cfg, f"enhancer ranking skipped: {exc}")

    if cfg.figures and figs:
        files += viz.save_all(figs, os.path.join(out, "figures"))

    report.append("")
    report.append("## Settings")
    report.append("```json")
    report.append(json.dumps(cfg.to_dict(), indent=2, default=str))
    report.append("```")
    report.append("")
    report.append("## Files")
    report += [f"- `{os.path.relpath(f, out)}`" for f in files]
    text = "\n".join(report)
    with open(os.path.join(out, "REPORT.md"), "w") as fh:
        fh.write(text + "\n")
    _log(cfg, f"done in {time.time() - t0:.1f} s -> {out}")
    return RunResult(out, cfg, seg, fine_seg, summary, comparison, gene_table, dominant, elements,
                     importance, parts, contributions, files, text)


def parse_groups(spec: Optional[str]) -> Optional[Dict[str, List[str]]]:
    """``"TN=tn1,tn2;TEX=tex1,tex2"`` or a path to a two-column ``sample<TAB>condition`` file."""
    if not spec:
        return None
    if os.path.exists(spec):
        tab = pd.read_csv(spec, sep="\t", header=None, comment="#")
        out: Dict[str, List[str]] = {}
        for s, c in zip(tab.iloc[:, 0], tab.iloc[:, 1]):
            out.setdefault(str(c), []).append(str(s))
        return out
    out = {}
    for part in spec.split(";"):
        if not part.strip():
            continue
        cond, _, samples = part.partition("=")
        out[cond.strip()] = [s.strip() for s in samples.split(",") if s.strip()]
    return out


def read_expression_matrix(path: str, log: bool = True, pseudocount: float = 1.0) -> pd.DataFrame:
    """Genes x samples table (``gene_id`` first column) on a log scale."""
    tab = pd.read_csv(path, sep="\t")
    if "gene_id" not in tab.columns:
        tab = tab.rename(columns={tab.columns[0]: "gene_id"})
    tab = tab.set_index("gene_id")
    num = tab.select_dtypes("number")
    return np.log2(num + pseudocount) if log else num


def read_lfc(path: str) -> pd.Series:
    """Per-gene log2 fold change as a Series indexed by ``gene_id``.

    Accepts a two-column ``gene_id<TAB>log2fc`` file, or a DESeq2 / edgeR / limma result
    table (first column or ``gene_id`` column as the gene, and a ``log2FoldChange``,
    ``logFC`` or ``log2fc`` column as the effect).
    """
    tab = pd.read_csv(path, sep="\t")
    if tab.shape[1] < 2:
        tab = pd.read_csv(path, sep=None, engine="python")
    gene_col = "gene_id" if "gene_id" in tab.columns else tab.columns[0]
    for cand in ("log2FoldChange", "logFC", "log2fc", "log2FC", "lfc"):
        if cand in tab.columns:
            val_col = cand
            break
    else:
        num = [c for c in tab.columns if c != gene_col and pd.api.types.is_numeric_dtype(tab[c])]
        if not num:
            raise ValueError(f"{path}: no log2 fold change column found")
        val_col = num[0]
    out = pd.Series(tab[val_col].to_numpy(dtype=float), index=tab[gene_col].astype(str), name="log2fc")
    return out[~out.index.duplicated()].dropna()


# ---------------------------------------------------------------------------
# Genome-wide streaming: one chromosome in memory at a time
# ---------------------------------------------------------------------------


def run_genome_streaming(
    manifest: str,
    chrom_sizes: str,
    out: str,
    cfg: Optional[RunConfig] = None,
    fit_chroms: Optional[Sequence[str]] = None,
    chroms: Optional[Sequence[str]] = None,
    blacklist: Optional[str] = None,
    groups: Optional[Dict[str, Sequence[str]]] = None,
    loader=None,
) -> Dict[str, object]:
    """Genome-wide run that never holds more than one chromosome of fine bins.

    1. The model is fitted on ``fit_chroms`` (default: the three largest chromosomes),
       binned at the fine resolution, pooled to ``fit_bin_size`` and normalised there;
       the size factors are then kept for every chromosome. With ``n_fit_states`` the
       states are collapsed before scoring.
    2. Every chromosome in ``chroms`` is binned, scored, and written into a
       :class:`chromstate.resolution.ScoreTensor` under ``out/scores/tensor`` as
       ``uint8`` 0-100 scores (13 GB for a mouse genome at 10 bp with 12 states and 4
       samples, against 100 GB of float64 posteriors). Per-state base pairs and
       percentages are accumulated as it goes.

    ``loader(chroms) -> ChromStateDataset`` replaces the manifest reader (used by tests
    and by callers who already have binned data). Gene-level tables are not produced
    here; run ``gene`` mode for those, or call ``ScoreTensor.region_means`` on the
    tensor with any region table.
    """
    from .collapse import collapse_states, merge_table
    from .genome import Genome
    from .model.core import fit
    from .norm import normalize
    from .resolution import ScoreTensor, coarsen_dataset, copy_size_factors, decode_to_tensor

    cfg = cfg or RunConfig(mode="genome")
    os.makedirs(out, exist_ok=True)
    genome = Genome.from_file(chrom_sizes)
    all_chroms = list(chroms or genome.chroms)
    if fit_chroms is None:
        fit_chroms = sorted(all_chroms, key=lambda c: -genome.sizes[c])[:3]

    def load(cs_):
        if loader is not None:
            return loader(list(cs_))
        return build_fine_dataset(manifest, chrom_sizes, cfg, blacklist=blacklist, chroms=list(cs_))

    _log(cfg, f"fitting on {', '.join(fit_chroms)} …")
    fit_fine = load(fit_chroms)
    width = cfg.resolution
    coarse = coarsen_dataset(fit_fine, cfg.fit_bin_size // width)
    if cfg.norm != "none":
        normalize(coarse, method=cfg.norm, spikein=cfg.spikein)
    n_fit = cfg.n_fit_states or cfg.n_states
    seg = fit(coarse, n_states=n_fit, emission=cfg.emission, n_restarts=cfg.n_restarts,
              max_restarts=cfg.max_restarts, seed=cfg.seed, fit_fraction=cfg.fit_fraction)
    files = []
    if n_fit > cfg.n_states:
        seg = collapse_states(seg, cfg.n_states)
        files.append(_save(merge_table(seg), os.path.join(out, "model", "collapsed_members.tsv")))
    files.append(_save(seg.emission_frame(), os.path.join(out, "model", "emissions.tsv"), index=True))
    files.append(_save(seg.state_table(), os.path.join(out, "model", "states.tsv")))
    del fit_fine

    tensor = ScoreTensor(os.path.join(out, "scores", "tensor"), coarse.samples, seg.labels, width,
                         cfg.score_scale)
    for chrom in all_chroms:
        _log(cfg, f"scoring {chrom} …")
        fine = load([chrom])
        copy_size_factors(coarse, fine)
        decode_to_tensor(seg, fine, tensor.path, scale=cfg.score_scale, tensor=tensor)
        del fine
    summary = tensor.summary()
    files.append(_save(summary, os.path.join(out, "summary", "state_summary_per_sample.tsv")))
    groups = {k: list(v) for k, v in (groups or {s: [s] for s in tensor.samples}).items()}
    cond = None
    if len(groups) >= 2:
        from .summary import condition_summary

        cond = condition_summary(summary, groups)
        files.append(_save(cond, os.path.join(out, "summary", "state_summary_per_condition.tsv")))
    with open(os.path.join(out, "REPORT.md"), "w") as fh:
        fh.write(f"# ChromState genome-wide streaming run\n\nFitted on {', '.join(fit_chroms)}; "
                 f"scored {len(all_chroms)} chromosomes at {width} bp into a uint8 score tensor "
                 f"({tensor.nbytes / 1e9:.2f} GB on disk).\n\n" + "\n".join(f"- `{os.path.relpath(f, out)}`" for f in files) + "\n")
    _log(cfg, f"done -> {out} ({tensor.nbytes / 1e9:.2f} GB of scores)")
    return {"model": seg, "tensor": tensor, "summary": summary, "condition": cond, "files": files}
