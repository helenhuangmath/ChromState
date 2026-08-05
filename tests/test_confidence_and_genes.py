"""Tests for change confidence (noise floor, region testing, calibration) and the gene
and RNA traceability layer.

The design principle these tests enforce: every number that could be mistaken for
evidence has to be checkable against something known. The simulator supplies the truth
for the chromatin side; the RNA side is generated from the simulated promoter states, so
the association test has an answer to recover and a null to fail against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import chromstate as cs
from chromstate.calibrate import (
    apply_temperature,
    confidence_summary,
    expected_calibration_error,
    fit_temperature,
    fit_temperature_from_replicates,
    temperature_scale,
)
from chromstate.features import FeatureSet, gene_state_matrix, map_bins_to_regions
from chromstate.simulate import simulate


# ---------------------------------------------------------------------------
# Fixtures: a replicated two-condition experiment, plus synthetic genes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def replicated():
    """Two conditions, three replicates each, with a known perturbation."""
    samples = ("c1", "c2", "c3", "t1", "t2", "t3")
    ds, truth = simulate(
        n_bins=600, n_chroms=2, samples=samples, seed=17, blur=0.2,
        # Without an explicit treated group the simulator perturbs every sample after the
        # first, which would make "control replicates" a control-versus-treated pair and
        # inflate the noise floor to meaninglessness.
        treated=["t1", "t2", "t3"],
        perturbation={"responder_fraction": 0.5, "from_state": "TssBiv", "to_state": "TssA"},
    )
    cs.normalize(ds, method="median_ratio")
    seg = cs.fit(ds, n_states=truth.n_states, n_restarts=1, seed=0)
    return ds, truth, seg, ["c1", "c2", "c3"], ["t1", "t2", "t3"]


def _synthetic_features(ds, n_genes: int = 120, seed: int = 0) -> FeatureSet:
    """Genes tiled across the simulated coordinates, alternating strand."""
    rng = np.random.default_rng(seed)
    bins = ds.bins.loc[ds.sample_mask(ds.samples[0])]
    rows = []
    for gi in range(n_genes):
        chrom = bins["chrom"].unique()[gi % bins["chrom"].nunique()]
        sub = bins.loc[bins["chrom"] == chrom]
        start_bin = int(rng.integers(5, len(sub) - 40))
        start = int(sub["start"].iloc[start_bin])
        rows.append(
            {
                "gene_id": f"GENE{gi:04d}",
                "name": f"Gene{gi}",
                "chrom": chrom,
                "start": start,
                "end": start + 6000,
                "strand": "+" if gi % 2 == 0 else "-",
            }
        )
    return FeatureSet.from_frame(pd.DataFrame(rows), source="synthetic")


# ---------------------------------------------------------------------------
# Noise floor
# ---------------------------------------------------------------------------


def test_simulator_perturbs_only_the_named_samples():
    """Replicates must be replicates: the noise floor is meaningless otherwise."""
    samples = ("c1", "c2", "t1", "t2")
    ds, truth = simulate(n_bins=200, n_chroms=1, samples=samples, seed=3,
                         treated=["t1", "t2"],
                         perturbation={"responder_fraction": 1.0, "from_state": "TssBiv",
                                       "to_state": "TssA"})
    assert truth.treated == ["t1", "t2"]
    st = {s: truth.states_for(s) for s in samples}
    np.testing.assert_array_equal(st["c1"], st["c2"])
    np.testing.assert_array_equal(st["t1"], st["t2"])
    assert (st["c1"] != st["t1"]).any(), "the treated group must actually differ"

    with pytest.raises(ValueError, match="not in the dataset"):
        simulate(n_bins=50, n_chroms=1, samples=samples, treated=["nope"])


def test_noise_floor_is_below_the_condition_contrast(replicated):
    ds, _, seg, a, b = replicated
    floor = cs.replicate_noise_floor(seg, {"control": a, "treated": b}, quantile=0.95)

    assert 0.0 <= floor.threshold <= 1.0
    assert len(floor.pairs) == 6  # 3 within-condition pairs per condition
    assert len(floor.per_condition) == 6

    contrast = cs.transition_propensity(seg, a[0], b[0])["tv_distance"].to_numpy()
    # The perturbation moves half the bivalent bins, so the contrast must have a heavier
    # tail than replicate noise -- otherwise the whole comparison is unusable.
    assert np.quantile(contrast, 0.95) > floor.threshold
    assert floor.exceeds(contrast).mean() > 0.05
    assert "replicate pair" in floor.note


def test_noise_floor_refuses_without_replicates(replicated):
    _, _, seg, a, b = replicated
    with pytest.raises(ValueError, match="2\\+ replicates"):
        cs.replicate_noise_floor(seg, {"control": [a[0]], "treated": [b[0]]})


def test_noise_floor_reports_single_sample_conditions(replicated):
    _, _, seg, a, b = replicated
    floor = cs.replicate_noise_floor(seg, {"control": a, "treated": [b[0]]})
    assert "single sample" in floor.note
    assert set(floor.per_condition["condition"]) == {"control"}


# ---------------------------------------------------------------------------
# Region-level testing
# ---------------------------------------------------------------------------


def test_region_testing_has_more_power_than_bin_testing(replicated):
    """The point of aggregating: same data, same permutations, usable q-values."""
    _, _, seg, a, b = replicated
    bins = cs.differential_state(seg, a, b, n_permutations=200)
    regions = cs.differential_regions(seg, a, b, tile_bins=10, n_permutations=200)

    assert bins.has_pvalues and regions.has_pvalues
    assert len(regions.table) < len(bins.table) / 5
    # Both are limited by the same permutation floor, but BH over far fewer, less
    # correlated tests is what turns the floor into significant calls.
    assert regions.table["qvalue"].min() <= bins.table["qvalue"].min()
    assert int(regions.table["significant"].sum()) >= 1


def test_region_test_carries_the_noise_floor_through(replicated):
    _, _, seg, a, b = replicated
    floor = cs.replicate_noise_floor(seg, {"control": a, "treated": b})
    res = cs.differential_regions(seg, a, b, tile_bins=10, noise_floor=floor,
                                  n_permutations=200)
    assert "above_noise_floor" in res.table.columns
    assert "confident_change" in res.table.columns
    # A confident change must clear both gates, never just one.
    conf = res.table["confident_change"].to_numpy()
    assert (conf <= res.table["significant"].to_numpy()).all()
    assert (conf <= res.table["above_noise_floor"].to_numpy()).all()
    assert "noise floor" in res.note


def test_regions_may_be_supplied_and_are_not_derived_from_the_effect(replicated):
    ds, _, seg, a, b = replicated
    features = _synthetic_features(ds)
    promoters = features.promoters(upstream=1000, downstream=1000)
    res = cs.differential_regions(seg, a, b, regions=promoters, min_bins=3,
                                  n_permutations=100)
    assert len(res.table) <= len(promoters)
    assert set(res.table["gene_id"]) <= set(promoters["gene_id"])
    assert (res.table["n_bins"] >= 3).all()


def test_region_test_rejects_regions_off_the_assembly(replicated):
    _, _, seg, a, b = replicated
    bogus = pd.DataFrame({"chrom": ["chrZZ"], "start": [0], "end": [10_000]})
    with pytest.raises(ValueError, match="same assembly"):
        cs.differential_regions(seg, a, b, regions=bogus)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_temperature_scaling_preserves_the_hard_call():
    rng = np.random.default_rng(0)
    post = rng.dirichlet(np.ones(5) * 0.2, size=500)
    for t in (0.5, 1.0, 3.0, 12.0):
        scaled = temperature_scale(post, t)
        np.testing.assert_array_equal(np.argmax(scaled, axis=1), np.argmax(post, axis=1))
        np.testing.assert_allclose(scaled.sum(axis=1), 1.0, atol=1e-12)
    assert temperature_scale(post, 5.0).max(axis=1).mean() < post.max(axis=1).mean()
    assert temperature_scale(post, 0.5).max(axis=1).mean() > post.max(axis=1).mean()


def test_supervised_calibration_reduces_overconfidence(replicated):
    _, truth, seg, _, _ = replicated
    result = fit_temperature(seg, truth.states, seed=0)
    assert result.temperature > 0
    assert result.after["ece"] <= result.before["ece"] + 1e-9
    assert abs(result.after["overconfidence"]) <= abs(result.before["overconfidence"]) + 1e-9
    assert "held-out" in result.note


def test_replicate_calibration_softens_a_saturated_posterior(replicated):
    _, _, seg, a, b = replicated
    result = fit_temperature_from_replicates(seg, {"control": a, "treated": b}, seed=0)
    assert result.temperature > 1.0, "saturated posteriors should be softened, not sharpened"
    assert result.after["mean_confidence"] < result.before["mean_confidence"]
    assert "reproducibility, not correctness" in result.note

    softened = apply_temperature(seg, result.temperature)
    assert softened.meta["temperature"] == pytest.approx(result.temperature)
    np.testing.assert_array_equal(softened.states, seg.states)
    before = confidence_summary(seg.posterior)
    after = confidence_summary(softened.posterior)
    assert after["frac_above_0.99"] < before["frac_above_0.99"]
    assert after["mean_entropy"] > before["mean_entropy"]


def test_replicate_calibration_needs_replicates(replicated):
    _, _, seg, a, b = replicated
    with pytest.raises(ValueError, match="2\\+ replicates"):
        fit_temperature_from_replicates(seg, {"control": [a[0]], "treated": [b[0]]})


def test_calibration_uncompresses_change_scores(replicated):
    """The reason calibration matters here: saturated posteriors binarise every effect."""
    _, _, seg, a, b = replicated
    raw = cs.transition_propensity(seg, a[0], b[0])["tv_distance"].to_numpy()
    warm = apply_temperature(seg, 4.0)
    scaled = cs.transition_propensity(warm, a[0], b[0])["tv_distance"].to_numpy()
    extreme_raw = np.mean((raw < 0.02) | (raw > 0.98))
    extreme_scaled = np.mean((scaled < 0.02) | (scaled > 0.98))
    assert extreme_scaled < extreme_raw


def test_expected_calibration_error_is_zero_for_a_perfect_model():
    n = 2000
    labels = np.zeros(n, dtype=int)
    post = np.zeros((n, 2))
    post[:, 0] = 1.0
    assert expected_calibration_error(post, labels) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Features and traceability
# ---------------------------------------------------------------------------


def test_promoters_are_strand_aware(replicated):
    ds, _, _, _, _ = replicated
    features = _synthetic_features(ds, n_genes=4)
    prom = features.promoters(upstream=2000, downstream=500).set_index("gene_id")
    plus = features.table.set_index("gene_id").loc["GENE0000"]
    minus = features.table.set_index("gene_id").loc["GENE0001"]

    # Plus strand: window runs from 2 kb before the start to 0.5 kb after it.
    assert prom.loc["GENE0000", "start"] == plus["start"] - 2000
    assert prom.loc["GENE0000", "end"] == plus["start"] + 500
    # Minus strand: the TSS is the gene end, and "upstream" is to its right.
    assert prom.loc["GENE0001", "end"] == minus["end"] - 1 + 2000
    assert prom.loc["GENE0001", "start"] == minus["end"] - 1 - 500


def test_gtf_and_bed_readers_agree(tmp_path):
    gtf = tmp_path / "genes.gtf"
    gtf.write_text(
        "#comment\n"
        'chr1\tsrc\tgene\t1001\t2000\t.\t+\t.\tgene_id "G1"; gene_name "Alpha";\n'
        'chr1\tsrc\texon\t1001\t1200\t.\t+\t.\tgene_id "G1";\n'
        'chr2\tsrc\tgene\t5001\t9000\t.\t-\t.\tgene_id "G2"; gene_name "Beta";\n'
    )
    fs = FeatureSet.from_gtf(str(gtf))
    assert list(fs.table["gene_id"]) == ["G1", "G2"]
    assert list(fs.table["name"]) == ["Alpha", "Beta"]
    # GTF is 1-based inclusive; the reader converts to 0-based half-open.
    assert fs.table.loc[0, "start"] == 1000 and fs.table.loc[0, "end"] == 2000
    assert list(fs.tss()) == [1000, 8999]

    bed = tmp_path / "genes.bed"
    bed.write_text("chr1\t1000\t2000\tG1\t0\t+\nchr2\t5000\t9000\tG2\t0\t-\n")
    fb = FeatureSet.from_bed(str(bed))
    pd.testing.assert_frame_equal(
        fs.table[["gene_id", "chrom", "start", "end", "strand"]],
        fb.table[["gene_id", "chrom", "start", "end", "strand"]],
    )


def test_bin_mapping_handles_shared_promoters_explicitly(replicated):
    ds, _, _, _, _ = replicated
    bins = ds.bins.loc[ds.sample_mask(ds.samples[0]), ["chrom", "start", "end"]].reset_index(drop=True)
    chrom = bins["chrom"].iloc[0]
    start = int(bins["start"].iloc[10])
    overlapping = pd.DataFrame(
        {
            "gene_id": ["A", "B"],
            "chrom": [chrom, chrom],
            "start": [start, start + 200],
            "end": [start + 1000, start + 1200],
            "part": ["promoter", "promoter"],
        }
    )
    every = map_bins_to_regions(bins, overlapping, assign="all")
    nearest = map_bins_to_regions(bins, overlapping, assign="nearest_tss")

    shared = every.groupby("bin_row").size()
    assert (shared > 1).any(), "a bin in both windows must appear twice under assign='all'"
    assert nearest["bin_row"].duplicated().sum() == 0, "nearest_tss must partition bins"


def test_mapping_reports_an_assembly_mismatch(replicated):
    ds, _, _, _, _ = replicated
    bins = ds.bins.loc[ds.sample_mask(ds.samples[0]), ["chrom", "start", "end"]]
    bogus = pd.DataFrame({"gene_id": ["X"], "chrom": ["1"], "start": [0], "end": [1000]})
    with pytest.raises(ValueError, match="naming mismatch"):
        map_bins_to_regions(bins.reset_index(drop=True), bogus)


def test_gene_state_matrix_composition_is_a_distribution(replicated):
    ds, _, seg, _, _ = replicated
    features = _synthetic_features(ds)
    genes = gene_state_matrix(seg, features, part="promoter", upstream=1000, downstream=1000)

    assert set(genes["sample"]) == set(ds.samples)
    comp = genes[seg.labels].to_numpy()
    np.testing.assert_allclose(comp.sum(axis=1), 1.0, atol=1e-9)
    assert (genes["n_bins"] > 0).all()
    assert genes["entropy"].between(0, 1).all()
    assert genes["top_state"].isin(seg.labels).all()
    # Promoter and body are different readouts and must not be silently merged.
    bodies = gene_state_matrix(seg, features, part="body", skip_promoter=1000, min_length=1000)
    assert set(bodies["part"]) == {"body"}


def test_trace_gene_reaches_the_raw_counts(replicated):
    ds, _, seg, _, _ = replicated
    features = _synthetic_features(ds)
    genes = gene_state_matrix(seg, features, part="promoter")
    gene_id = genes.iloc[0]["gene_id"]

    trace = cs.trace_gene(seg, features, gene_id, part="promoter")
    bins = trace["bins"]
    assert set(bins["sample"]) <= set(ds.samples)
    for mark in ds.marks:
        assert f"count_{mark}" in bins.columns
        assert f"signal_{mark}" in bins.columns
    assert any(c.startswith("llr_") for c in trace["attribution"].columns)

    # The gene-level composition must equal the mean of the bins it came from.
    sample = trace["composition"].iloc[0]["sample"]
    from_bins = bins.loc[bins["sample"] == sample, [f"post_{lbl}" for lbl in seg.labels]].mean()
    from_table = trace["composition"].iloc[0][seg.labels].to_numpy(dtype=float)
    np.testing.assert_allclose(from_bins.to_numpy(), from_table, atol=1e-9)


# ---------------------------------------------------------------------------
# RNA linkage
# ---------------------------------------------------------------------------


def _synthetic_rna(gene_change: pd.DataFrame, active_state: str, seed: int = 0,
                   effect: float = 3.0, noise: float = 0.6) -> pd.DataFrame:
    """Expression generated from the promoter's gain of an active state, plus noise.

    Ground truth by construction: the association test must recover a positive
    correlation for ``active_state`` and nothing for a shuffled control.
    """
    rng = np.random.default_rng(seed)
    delta = gene_change[f"delta_{active_state}"].to_numpy()
    lfc = effect * delta + rng.normal(0, noise, size=len(delta))
    return pd.DataFrame({"gene_id": gene_change["gene_id"], "log2fc": lfc})


def test_expression_association_recovers_a_planted_link(replicated):
    ds, _, seg, a, b = replicated
    features = _synthetic_features(ds, n_genes=150)
    genes = gene_state_matrix(seg, features, part="promoter", upstream=1000, downstream=1000)
    change = cs.gene_state_change(genes, a, b)

    active = next((lbl for lbl in seg.labels if lbl.startswith("TssA")), seg.labels[0])
    rna = _synthetic_rna(change, active, seed=1)
    linked = cs.link_expression(change, rna)
    assert linked.attrs["n_joined"] == len(change)

    assoc = cs.expression_association(linked, n_permutations=300, seed=0)
    row = assoc.table.set_index("state").loc[active]
    assert row["spearman"] > 0.3
    assert row["pvalue"] < 0.01
    assert bool(row["significant"])
    assert "does not establish" in assoc.note


def test_expression_association_finds_nothing_when_there_is_nothing(replicated):
    """The control that matters: unrelated expression must not produce a hit."""
    ds, _, seg, a, b = replicated
    features = _synthetic_features(ds, n_genes=150)
    genes = gene_state_matrix(seg, features, part="promoter")
    change = cs.gene_state_change(genes, a, b)

    rng = np.random.default_rng(7)
    rna = pd.DataFrame({"gene_id": change["gene_id"],
                        "log2fc": rng.normal(0, 1, size=len(change))})
    assoc = cs.expression_association(cs.link_expression(change, rna),
                                      n_permutations=300, seed=0)
    assert not assoc.table["significant"].any()


def test_link_warns_when_identifiers_barely_match(replicated):
    ds, _, seg, a, b = replicated
    features = _synthetic_features(ds, n_genes=60)
    change = cs.gene_state_change(gene_state_matrix(seg, features), a, b)
    rna = pd.DataFrame({"gene_id": [f"{g}.1" for g in change["gene_id"]],
                        "log2fc": np.zeros(len(change))})
    linked = cs.link_expression(change, rna)
    assert linked.attrs["n_joined"] == 0
    assert "identifier types" in linked.attrs["warning"]


def test_expression_change_matches_a_hand_computed_fold_change():
    expr = pd.DataFrame({"gene_id": ["G1", "G2"], "a1": [10.0, 100.0], "a2": [10.0, 100.0],
                         "b1": [40.0, 25.0], "b2": [40.0, 25.0]})
    out = cs.expression_change(expr, ["a1", "a2"], ["b1", "b2"], pseudocount=0.0)
    assert out.loc[0, "log2fc"] == pytest.approx(2.0)
    assert out.loc[1, "log2fc"] == pytest.approx(-2.0)


def test_rank_traceable_genes_requires_both_layers(replicated):
    ds, _, seg, a, b = replicated
    features = _synthetic_features(ds, n_genes=120)
    change = cs.gene_state_change(gene_state_matrix(seg, features), a, b)
    active = next((lbl for lbl in seg.labels if lbl.startswith("TssA")), seg.labels[0])
    linked = cs.link_expression(change, _synthetic_rna(change, active, seed=2))

    top = cs.rank_traceable_genes(linked, n=10)
    assert len(top) <= 10
    assert (top["joint_score"].diff().dropna() <= 1e-12).all(), "must be ranked"
    assert (top["tv_distance"] > 0).all() and (top["log2fc"].abs() > 0).all()
    assert set(top["direction"]) <= {"up", "down"}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def test_new_figures_render(replicated, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from chromstate import viz

    ds, _, seg, a, b = replicated
    features = _synthetic_features(ds, n_genes=120)
    floor = cs.replicate_noise_floor(seg, {"control": a, "treated": b})
    regions = cs.differential_regions(seg, a, b, tile_bins=10, noise_floor=floor,
                                      n_permutations=100)
    change = cs.gene_state_change(gene_state_matrix(seg, features), a, b)
    active = next((lbl for lbl in seg.labels if lbl.startswith("TssA")), seg.labels[0])
    linked = cs.link_expression(change, _synthetic_rna(change, active, seed=3))
    assoc = cs.expression_association(linked, n_permutations=100)
    trace = cs.trace_gene(seg, features, change.iloc[0]["gene_id"])

    figs = {
        "volcano": viz.plot_change_volcano(regions.table, noise_floor=floor.threshold),
        "floor": viz.plot_noise_floor(
            floor, cs.transition_propensity(seg, a[0], b[0])["tv_distance"].to_numpy()),
        "expression": viz.plot_expression_link(linked, assoc.table),
        "trace": viz.plot_gene_trace(trace),
    }
    paths = viz.save_all(figs, str(tmp_path))
    assert len(paths) == 4
    for p in paths:
        assert (tmp_path / p.rsplit("/", 1)[-1]).stat().st_size > 3000
