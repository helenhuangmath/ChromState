"""ChromState -- quantitative chromatin state analysis.

Where a segmenter answers "which state is this bin in", ChromState answers "how much of
each state, how confident, how bivalent, and which mark is responsible".

Quick start
-----------
    import chromstate as cs

    ds = cs.build_dataset(cs.read_manifest("cellmarkfiletable.txt"), binset)
    cs.normalize(ds, method="median_ratio")
    seg = cs.fit(ds, n_states=10)
    table = cs.quantify(seg)
"""

from __future__ import annotations

__version__ = "1.0.0"

from . import sc
from .calibrate import (
    apply_temperature,
    fit_temperature,
    fit_temperature_from_replicates,
    temperature_scale,
)
from .collapse import collapse_states, fit_collapsed, match_states, smooth_segments
from .compare import DifferentialReport, differential_report, moderated_f_test, region_state_tests
from .dataset import ChromStateDataset, build_dataset, dataset_from_arrays, make_genome_binset
from .definitions import StateDefinitions
from .dynamics import (
    differential_regions,
    differential_state,
    occupancy_change,
    replicate_noise_floor,
    state_flow,
    transition_propensity,
    transport_flux,
)
from .enhancers import (
    MonotoneLinear,
    abc_change,
    abc_score,
    candidate_enhancers,
    enhancer_activity,
    enhancer_importance,
    enhancer_state_classes,
    gene_enhancer_features,
    link_enhancers,
    read_peaks,
)
from .features import (
    FeatureSet,
    gene_state_change,
    gene_state_matrix,
    map_bins_to_regions,
    trace_gene,
)
from .genome import BinSet, Genome
from .io.counts import load_signal, read_bed_regions
from .io.manifest import MarkFile, read_manifest
from .model.core import Segmentation, fit, select_n_states
from .norm import compute_size_factors, normalize
from .patterns import flow_table, gene_dominant_states, pattern_counts, region_patterns
from .pipeline import RunConfig, build_fine_dataset, read_lfc, run_genome_streaming, run_pipeline
from .predict import PredictionResult, predict_expression
from .quantify import (
    attribution,
    bivalency,
    posterior_bivalency,
    quantify,
    quantify_regions,
    state_entropy,
    state_mark_importance,
    summarize,
    top_regions,
)
from .regions import (
    RegionScheme,
    call_elements,
    enhancer_contributions,
    gene_regulatory_quantification,
    rank_parts_for_expression,
)
from .regulation import (
    compare_representations,
    expression_change_task,
    expression_level_task,
    gene_quantification,
    rank_states_for_expression,
    state_importance,
    state_intensity,
)
from .resolution import (
    ScoreTensor,
    coarsen_dataset,
    decode_at_resolution,
    decode_to_tensor,
    fit_and_decode,
    rescale_hmm,
    state_scores,
    write_bedgraph,
    write_score_tracks,
)
from .rna import (
    enhancer_expression_association,
    expression_association,
    expression_by_state_transition,
    expression_change,
    link_enhancer_expression,
    link_expression,
    rank_traceable_genes,
    read_expression,
)
from .simulate import simulate
from .summary import compare_conditions, condition_summary, flow_test, state_summary, state_tests

__all__ = [
    "__version__",
    "DifferentialReport",
    "PredictionResult",
    "StateDefinitions",
    "differential_report",
    "moderated_f_test",
    "predict_expression",
    "quantify_regions",
    "region_state_tests",
    "sc",
    "RegionScheme",
    "RunConfig",
    "ScoreTensor",
    "decode_to_tensor",
    "flow_test",
    "run_genome_streaming",
    "smooth_segments",
    "state_tests",
    "build_fine_dataset",
    "call_elements",
    "coarsen_dataset",
    "collapse_states",
    "compare_conditions",
    "condition_summary",
    "decode_at_resolution",
    "enhancer_contributions",
    "fit_and_decode",
    "fit_collapsed",
    "flow_table",
    "gene_dominant_states",
    "gene_regulatory_quantification",
    "match_states",
    "pattern_counts",
    "rank_parts_for_expression",
    "region_patterns",
    "rescale_hmm",
    "run_pipeline",
    "state_scores",
    "state_summary",
    "write_bedgraph",
    "write_score_tracks",
    "BinSet",
    "ChromStateDataset",
    "FeatureSet",
    "Genome",
    "MarkFile",
    "Segmentation",
    "abc_change",
    "abc_score",
    "apply_temperature",
    "attribution",
    "bivalency",
    "build_dataset",
    "candidate_enhancers",
    "compare_representations",
    "compute_size_factors",
    "dataset_from_arrays",
    "differential_regions",
    "differential_state",
    "enhancer_activity",
    "enhancer_expression_association",
    "enhancer_importance",
    "MonotoneLinear",
    "read_lfc",
    "enhancer_state_classes",
    "expression_association",
    "expression_by_state_transition",
    "expression_change",
    "expression_change_task",
    "expression_level_task",
    "fit",
    "fit_temperature",
    "fit_temperature_from_replicates",
    "gene_enhancer_features",
    "gene_quantification",
    "gene_state_change",
    "gene_state_matrix",
    "link_enhancer_expression",
    "link_enhancers",
    "link_expression",
    "load_signal",
    "make_genome_binset",
    "map_bins_to_regions",
    "normalize",
    "occupancy_change",
    "posterior_bivalency",
    "quantify",
    "rank_states_for_expression",
    "rank_traceable_genes",
    "read_bed_regions",
    "read_expression",
    "read_manifest",
    "read_peaks",
    "replicate_noise_floor",
    "select_n_states",
    "simulate",
    "state_entropy",
    "state_flow",
    "state_importance",
    "state_intensity",
    "state_mark_importance",
    "summarize",
    "temperature_scale",
    "top_regions",
    "trace_gene",
    "transition_propensity",
    "transport_flux",
]
