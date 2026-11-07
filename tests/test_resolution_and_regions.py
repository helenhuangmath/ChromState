"""Tests for multi-resolution scoring, region schemes, windowed tilings and state collapsing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import chromstate as cs
from chromstate.collapse import collapse_states, fit_collapsed, match_states
from chromstate.genome import BinSet, Genome
from chromstate.regions import RegionScheme, call_elements, gene_regulatory_quantification
from chromstate.resolution import (
    _matrix_root,
    coarsen_dataset,
    copy_size_factors,
    decode_at_resolution,
    rescale_hmm,
    score_track,
    state_scores,
)
from chromstate.simulate import DEFAULT_STATES, simulate

FINE = 10
FIT = 200


def _fine_states(scale=FINE / FIT):
    return {k: tuple(v * scale for v in vals) for k, vals in DEFAULT_STATES.items()}


@pytest.fixture(scope="module")
def fine_sim():
    """A 10 bp simulation: per-bin means and dispersion are the 200 bp ones divided by 20."""
    ds, truth = simulate(
        n_bins=6000, n_chroms=3, samples=("a1", "a2", "b1", "b2"), treated=("b1", "b2"),
        states=_fine_states(), self_transition=1 - (1 - 0.92) / 20, dispersion=3.0 / 20,
        perturbation={"responder_fraction": 0.6, "from_state": "TssBiv", "to_state": "TssA"},
        seed=5, bin_size=FINE,
    )
    return ds, truth


@pytest.fixture(scope="module")
def fitted(fine_sim):
    fine, truth = fine_sim
    coarse = coarsen_dataset(fine, FIT // FINE)
    cs.normalize(coarse)
    copy_size_factors(coarse, fine)
    seg = cs.fit(coarse, n_states=truth.n_states, n_restarts=2, seed=0)
    fine_seg = decode_at_resolution(seg, fine)
    return fine, truth, coarse, seg, fine_seg


# -- coarsening ---------------------------------------------------------------


def test_coarsen_sums_counts_and_keeps_chains(fine_sim):
    fine, _ = fine_sim
    coarse = coarsen_dataset(fine, 20)
    assert coarse.n_rows == fine.n_rows // 20
    assert np.allclose(coarse.X.sum(axis=0), fine.X.sum(axis=0))
    # first coarse bin of each sample sums the first 20 fine rows of that sample
    for s in fine.samples:
        fm = np.flatnonzero(fine.sample_mask(s))
        cm = np.flatnonzero(coarse.sample_mask(s))
        assert np.allclose(coarse.X[cm[0]], fine.X[fm[:20]].sum(axis=0))
    assert coarse.bins["end"].iloc[0] - coarse.bins["start"].iloc[0] == 200
    assert len(coarse.chains) == len(fine.chains)
    assert coarse.binset.bin_size == 200


def test_coarsen_averages_non_count_input(fine_sim):
    fine, _ = fine_sim
    from dataclasses import replace

    cov = replace(fine, is_count=False)
    coarse = coarsen_dataset(cov, 20)
    assert np.allclose(coarse.X[0], fine.X[:20].mean(axis=0))


# -- rescaling ----------------------------------------------------------------


def test_matrix_root_inverts_matrix_power():
    rng = np.random.default_rng(0)
    A = rng.dirichlet(np.ones(5) * 0.3, size=5) * 0.1
    np.fill_diagonal(A, 0.0)
    A = A / A.sum(axis=1, keepdims=True) * 0.1
    np.fill_diagonal(A, 0.9)
    R = _matrix_root(A, 1 / 20)
    assert np.allclose(R.sum(axis=1), 1.0)
    assert R.min() >= 0
    assert np.allclose(np.linalg.matrix_power(R, 20), A, atol=1e-2)  # clipping of tiny negatives
    assert np.allclose(np.diag(R), np.diag(A) ** (1 / 20), atol=0.02)


def test_rescale_hmm_divides_nb_parameters(fitted):
    _, _, _, seg, _ = fitted
    fine_hmm = rescale_hmm(seg.model, 0.05)
    assert np.allclose(fine_hmm.emissions.mu, seg.model.emissions.mu * 0.05)
    assert np.allclose(fine_hmm.emissions.dispersion, seg.model.emissions.dispersion * 0.05)
    assert np.allclose(fine_hmm.transitions.sum(axis=1), 1.0)
    assert (np.diag(fine_hmm.transitions) > np.diag(seg.model.transitions)).all()


# -- decoding at the fine width ------------------------------------------------


def test_fine_decoding_recovers_the_truth(fitted):
    fine, truth, coarse, seg, fine_seg = fitted
    from chromstate.benchmark.metrics import match_states as _match

    assert fine_seg.posterior.shape == (fine.n_rows, seg.n_states)
    assert fine_seg.posterior.dtype == np.float32
    assert np.allclose(fine_seg.posterior.sum(axis=1), 1.0, atol=1e-4)
    mapping, _ = _match(truth.states, fine_seg.states, truth.n_states, seg.n_states)
    mapped = np.array([mapping[int(s)] for s in fine_seg.states])
    acc = float((mapped == truth.states).mean())
    assert acc > 0.8, acc
    # the fine posterior averaged back to 200 bp agrees with the coarse posterior
    fine_avg = np.add.reduceat(np.asarray(fine_seg.posterior, dtype=float),
                               np.arange(0, fine.n_rows, 20), axis=0) / 20
    agree = float((fine_avg.argmax(axis=1) == seg.posterior.argmax(axis=1)).mean())
    assert agree > 0.85, agree
    assert fine_seg.meta["fit_bin_size"] == 200 and fine_seg.meta["bin_size"] == 10


def test_state_scores_and_tracks(fitted, tmp_path):
    fine, _, _, _, fine_seg = fitted
    scores = state_scores(fine_seg, scale=100)
    score_cols = [c for c in scores.columns if c.startswith("score_")]
    assert len(score_cols) == fine_seg.n_states
    total = scores[score_cols].to_numpy(dtype=int).sum(axis=1)
    assert np.abs(total - 100).max() <= fine_seg.n_states  # rounding only
    assert scores[score_cols].to_numpy().max() <= 100
    track = score_track(fine_seg, fine_seg.labels[0], fine.samples[0], merge=True)
    unmerged = score_track(fine_seg, fine_seg.labels[0], fine.samples[0], merge=False)
    assert len(track) < len(unmerged)
    assert (track["value"] * (track["end"] - track["start"])).sum() == pytest.approx(
        (unmerged["value"] * (unmerged["end"] - unmerged["start"])).sum())
    path = cs.write_bedgraph(fine_seg, fine_seg.labels[0], fine.samples[0], str(tmp_path / "s.bedgraph"))
    lines = open(path).read().splitlines()
    assert lines[0].startswith("track type=bedGraph")
    assert len(lines) == len(track) + 1


def test_fit_and_decode_one_call(fine_sim):
    fine, truth = fine_sim
    sub = fine.subset_rows(fine.bins["sample"].isin(["a1", "b1"]).to_numpy())
    seg, fine_seg = cs.fit_and_decode(sub, n_states=truth.n_states, fit_bin_size=200,
                                      n_restarts=1, adaptive_restarts=False, seed=0)
    assert fine_seg.n_states == truth.n_states
    assert fine_seg.dataset.n_rows == sub.n_rows


# -- windowed tiling ----------------------------------------------------------


def test_binset_from_regions_merges_overlaps_and_locates():
    genome = Genome({"chr1": 10_000, "chr2": 5_000})
    regions = pd.DataFrame({"chrom": ["chr1", "chr1", "chr2", "chr1"],
                            "start": [100, 900, 0, 5_000], "end": [1_000, 1_500, 10_000, 5_050]})
    bs = BinSet.from_regions(regions, bin_size=100, genome=genome)
    assert len(bs.chain_slices) == 3  # chr1 [100,1500), chr1 [5000,5050), chr2 [0,5000)
    assert not bs.is_genome_tiling
    assert bs.index_of("chr1", 150) == 0
    assert bs.index_of("chr1", 1_499) == 13
    assert bs.index_of("chr1", 2_000) == -1
    rows = bs.rows_for_positions("chr1", np.array([150, 2_000, 5_010, 1_600]))
    assert rows.tolist() == [0, -1, 14, -1]
    sl = bs.rows_overlapping("chr1", 950, 1_250)
    assert (sl.start, sl.stop) == (8, 12)
    assert bs.chrom_offsets()["chr2"][1] == 50
    mask = bs.apply_blacklist(pd.DataFrame({"chrom": ["chr1"], "start": [120], "end": [320]}))
    assert (~mask).sum() == 3


def test_count_bed_reads_on_windowed_tiling(tmp_path):
    from chromstate.io.counts import count_bed_reads

    genome = Genome({"chr1": 10_000})
    bs = BinSet.from_regions(pd.DataFrame({"chrom": ["chr1"], "start": [1_000], "end": [2_000]}),
                             bin_size=100, genome=genome)
    bed = tmp_path / "reads.bed"
    bed.write_text("chr1\t1000\t1050\tr1\t0\t+\nchr1\t1800\t1850\tr2\t0\t-\nchr1\t5000\t5050\tr3\t0\t+\n")
    counts = count_bed_reads(str(bed), bs, shift=10)
    assert counts.sum() == 2
    assert counts[0] == 1 and counts[8] == 1


# -- region scheme ------------------------------------------------------------


def _features():
    rows = [
        {"gene_id": "g1", "name": "G1", "chrom": "chr1", "start": 20_000, "end": 30_000, "strand": "+"},
        {"gene_id": "g2", "name": "G2", "chrom": "chr1", "start": 40_000, "end": 46_000, "strand": "-"},
        {"gene_id": "g3", "name": "G3", "chrom": "chr2", "start": 5_000, "end": 15_000, "strand": "+"},
        {"gene_id": "g4", "name": "G4", "chrom": "chr3", "start": 30_000, "end": 31_000, "strand": "+"},
    ]
    return cs.FeatureSet(pd.DataFrame(rows))


def test_region_scheme_parts_are_consistent():
    f = _features()
    scheme = RegionScheme(promoter_upstream=1_000, promoter_downstream=500, regulatory_flank=10_000)
    prom = scheme.promoters(f).set_index("gene_id")
    assert (prom.loc["g1", "start"], prom.loc["g1", "end"]) == (19_000, 20_500)
    assert (prom.loc["g2", "start"], prom.loc["g2", "end"]) == (45_499, 46_999)  # minus strand
    reg = scheme.regulatory(f).set_index("gene_id")
    assert (reg.loc["g1", "start"], reg.loc["g1", "end"]) == (10_000, 40_000)
    distal = scheme.distal(f)
    g1 = distal.loc[distal["gene_id"] == "g1"]
    assert len(g1) == 2
    assert g1["end"].min() == 19_000 and g1["start"].max() == 20_500
    for s, e in zip(g1["start"], g1["end"]):
        assert e <= 19_000 or s >= 20_500
    assert (distal["part"] == "distal").all()
    with pytest.raises(ValueError):
        RegionScheme(regulatory_flank=-1)
    assert "50,000" in RegionScheme().describe()


def test_gene_regulatory_quantification_has_three_parts(fitted):
    _, _, _, _, fine_seg = fitted
    f = _features()
    gq = gene_regulatory_quantification(fine_seg, f, RegionScheme(regulatory_flank=10_000))
    for pre in ("P", "E", "B"):
        assert any(c.startswith(f"{pre}_occ_") for c in gq.columns), pre
    assert set(gq["gene_id"]) == {"g1", "g2", "g3", "g4"}
    assert len(gq) == 4 * len(fine_seg.dataset.samples)
    # promoter and distal are disjoint: their bin counts add up to at most the window
    row = gq.loc[(gq["gene_id"] == "g1") & (gq["sample"] == "a1")].iloc[0]
    assert row["P_n_bins"] == 200 and row["E_n_bins"] == 3_000 - 200


def test_call_elements_from_the_state_track(fitted):
    _, _, _, _, fine_seg = fitted
    f = _features()
    el = call_elements(fine_seg, f, RegionScheme(), min_score=50, min_width=100)
    assert {"enhancer_id", "chrom", "start", "end", "n_samples", "max_score", "is_promoter"} <= set(el.columns)
    assert (el["end"] - el["start"] >= 100).all()
    assert el["max_score"].min() >= 50
    assert el["n_samples"].between(1, 4).all()


# -- collapsing -----------------------------------------------------------------


def test_collapse_states_sums_posteriors_and_aggregates_model(fitted):
    _, _, _, seg, _ = fitted
    merged = collapse_states(seg, 4)
    assert merged.n_states == 4
    assert np.allclose(merged.posterior.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(merged.posterior.sum(axis=0).sum(), seg.posterior.sum())
    assert sum(len(v) for v in merged.meta["members"].values()) == seg.n_states
    assert np.allclose(merged.model.transitions.sum(axis=1), 1.0)
    assert merged.model.emissions.mu.shape == (4, seg.dataset.n_marks)
    # the hard path is mapped through the groups, so it is consistent with the posterior
    groups = np.array(merged.meta["state_groups"])
    assert (merged.states == groups[seg.states]).all()
    # explicit members
    custom = collapse_states(seg, 2, members=[[0, 1, 2], list(range(3, seg.n_states))])
    assert custom.n_states == 2
    assert custom.meta["collapse_method"] == "members"
    with pytest.raises(ValueError):
        collapse_states(seg, seg.n_states + 1)


def test_fit_collapsed_and_match(fitted):
    _, truth, coarse, seg, _ = fitted
    merged = fit_collapsed(coarse, n_states=5, n_fit=8, n_restarts=1, adaptive_restarts=False, seed=1)
    assert merged.n_states == 5 and merged.meta["collapsed_from"] == 8
    tab = match_states(merged, seg)
    assert len(tab) == 5 and (tab["distance"] >= 0).all()
    # collapsed model rescales and decodes like any other
    fine_seg = decode_at_resolution(merged, fitted[0])
    assert fine_seg.n_states == 5
