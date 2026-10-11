"""IO round-trip and normalisation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import chromstate as cs
from chromstate.genome import BinSet, Genome
from chromstate.io.counts import count_bed_reads, infer_format, score_peaks
from chromstate.io.manifest import read_manifest, validate_manifest
from chromstate.norm import compute_size_factors
from chromstate.simulate import simulate, write_demo_files


@pytest.fixture
def small_binset():
    return BinSet.from_genome(Genome({"chr1": 2000, "chr2": 1000}), bin_size=200)


# -- binning -----------------------------------------------------------------


def test_binset_tiles_the_genome(small_binset):
    assert len(small_binset) == 10 + 5
    assert len(small_binset.chain_slices) == 2
    assert small_binset.table["start"].iloc[0] == 0
    assert small_binset.table["end"].iloc[9] == 2000


def test_binset_final_bin_is_truncated():
    bs = BinSet.from_genome(Genome({"chr1": 950}), bin_size=200)
    assert len(bs) == 5
    assert bs.table["end"].iloc[-1] == 950


def test_index_of_maps_positions_to_bins(small_binset):
    assert small_binset.index_of("chr1", 0) == 0
    assert small_binset.index_of("chr1", 199) == 0
    assert small_binset.index_of("chr1", 200) == 1
    assert small_binset.index_of("chr2", 0) == 10
    assert small_binset.index_of("chrX", 0) == -1
    assert small_binset.index_of("chr1", 999999) == -1


def test_blacklist_masks_overlapping_bins(small_binset):
    bl = pd.DataFrame({"chrom": ["chr1"], "start": [400], "end": [800]})
    mask = small_binset.apply_blacklist(bl)
    assert not mask[2] and not mask[3]
    assert mask[0] and mask[1] and mask[5]


# -- readers -----------------------------------------------------------------


def test_bed_reads_land_in_the_right_bins(tmp_path, small_binset):
    path = tmp_path / "reads.bed"
    path.write_text(
        "chr1\t10\t11\tr\t0\t+\n"      # bin 0
        "chr1\t250\t251\tr\t0\t+\n"    # bin 1
        "chr1\t260\t261\tr\t0\t+\n"    # bin 1
        "chr2\t5\t6\tr\t0\t+\n"        # bin 10
        "chrZ\t5\t6\tr\t0\t+\n"        # ignored: unknown chromosome
    )
    counts = count_bed_reads(str(path), small_binset, shift=0)
    assert counts[0] == 1
    assert counts[1] == 2
    assert counts[10] == 1
    assert counts.sum() == 4


def test_bed_shift_moves_reads_by_strand(tmp_path, small_binset):
    path = tmp_path / "reads.bed"
    path.write_text("chr1\t150\t151\tr\t0\t+\nchr1\t250\t251\tr\t0\t-\n")
    counts = count_bed_reads(str(path), small_binset, shift=100)
    assert counts[1] == 1  # + strand: 150 + 100 = 250 -> bin 1
    assert counts[0] == 1  # - strand: 251 - 100 = 151 -> bin 0


def test_peak_scores_use_signal_value(tmp_path, small_binset):
    path = tmp_path / "peaks.narrowPeak"
    path.write_text(
        "chr1\t0\t400\tp1\t100\t.\t7.5\t-1\t-1\t200\n"
        "chr1\t300\t500\tp2\t100\t.\t3.0\t-1\t-1\t100\n"
    )
    vals = score_peaks(str(path), small_binset)
    assert vals[0] == pytest.approx(7.5)
    assert vals[1] == pytest.approx(7.5)  # overlapping bin takes the max
    assert vals[2] == pytest.approx(3.0)
    assert vals[3] == 0.0


def test_bed3_peaks_default_to_one(tmp_path, small_binset):
    path = tmp_path / "peaks.bed"
    path.write_text("chr1\t0\t400\n")
    # A 3-column file named .bed is read as reads, so ask for peaks explicitly.
    vals = score_peaks(str(path), small_binset)
    assert vals[0] == 1.0 and vals[1] == 1.0 and vals[2] == 0.0


def test_format_inference():
    assert infer_format("a.bam") == "bam"
    assert infer_format("a.bed") == "bed"
    assert infer_format("a.bed.gz") == "bed"
    assert infer_format("x.narrowPeak") == "peak"
    assert infer_format("mypeaks.txt") == "peak"
    with pytest.raises(ValueError):
        infer_format("mystery.dat")


# -- manifest ----------------------------------------------------------------


def test_manifest_round_trip(tmp_path):
    for name in ("a_m1.bed", "a_m2.bed", "b_m1.bed", "b_m2.bed"):
        (tmp_path / name).write_text("chr1\t0\t1\tr\t0\t+\n")
    mpath = tmp_path / "manifest.txt"
    mpath.write_text(
        "# comment\n"
        "a\tm1\ta_m1.bed\n"
        "a\tm2\ta_m2.bed\n"
        "\n"
        "b\tm1\tb_m1.bed\n"
        "b\tm2\tb_m2.bed\n"
    )
    entries = read_manifest(str(mpath))
    assert len(entries) == 4
    assert entries[0].sample == "a" and entries[0].mark == "m1"
    validate_manifest(entries)


def test_manifest_rejects_mismatched_mark_panels(tmp_path):
    for name in ("a_m1.bed", "a_m2.bed", "b_m1.bed"):
        (tmp_path / name).write_text("chr1\t0\t1\tr\t0\t+\n")
    mpath = tmp_path / "manifest.txt"
    mpath.write_text("a\tm1\ta_m1.bed\na\tm2\ta_m2.bed\nb\tm1\tb_m1.bed\n")
    with pytest.raises(ValueError, match="same set of marks"):
        validate_manifest(read_manifest(str(mpath)))


def test_manifest_reports_missing_files(tmp_path):
    mpath = tmp_path / "manifest.txt"
    mpath.write_text("a\tm1\tnope.bed\n")
    with pytest.raises(FileNotFoundError):
        validate_manifest(read_manifest(str(mpath)))


def test_full_io_round_trip_reproduces_counts(tmp_path):
    """Simulated counts -> BED files -> readers must return the original matrix."""
    manifest = write_demo_files(str(tmp_path), seed=3, n_bins=300)
    ds_mem, _ = simulate(n_bins=300, n_chroms=2, seed=3)

    genome = Genome.from_file(str(tmp_path / "chrom.sizes"))
    binset = BinSet.from_genome(genome, bin_size=200)
    ds_disk = cs.build_dataset(
        cs.read_manifest(manifest), binset, reader_kwargs={"bed": {"shift": 0}}
    )

    assert ds_disk.X.shape == ds_mem.X.shape
    assert ds_disk.marks == ds_mem.marks
    np.testing.assert_array_equal(ds_disk.X, ds_mem.X)


# -- dataset -----------------------------------------------------------------


def test_subset_rows_splits_chains_at_gaps():
    ds, _ = simulate(n_bins=100, n_chroms=1, samples=("s1",), seed=4)
    mask = np.ones(ds.n_rows, dtype=bool)
    mask[50] = False  # punch a hole
    sub = ds.subset_rows(mask)
    assert len(sub.chains) == 2, "removing an interior bin must split the chain"
    assert sum(sl.stop - sl.start for sl in sub.chains) == sub.n_rows


def test_dataset_rejects_inconsistent_shapes():
    from chromstate.dataset import ChromStateDataset

    with pytest.raises(ValueError):
        ChromStateDataset(
            X=np.zeros((10, 2)),
            marks=["a", "b", "c"],
            bins=pd.DataFrame({"chrom": ["c"] * 10, "start": range(10), "end": range(10),
                               "sample": ["s"] * 10}),
            chains=[slice(0, 10)],
            size_factors=np.ones((10, 2)),
        )


# -- normalisation -----------------------------------------------------------


def test_median_ratio_recovers_depth_differences():
    ds, truth = simulate(n_bins=3000, n_chroms=2, seed=5)
    factors = compute_size_factors(ds, method="median_ratio", reference="control")
    for j, mark in enumerate(ds.marks):
        true_ratio = truth.depth_factors[("treated", mark)] / truth.depth_factors[("control", mark)]
        assert factors[1, j] == pytest.approx(true_ratio, rel=0.25), mark


def test_median_ratio_is_not_degenerate_at_200bp():
    """Guards the pooling fix: per-bin ratios quantise and collapse onto exactly 1.0."""
    ds, _ = simulate(n_bins=3000, n_chroms=2, seed=6)
    pooled = compute_size_factors(ds, method="median_ratio", window_bins=50)
    unpooled = compute_size_factors(ds, method="median_ratio", window_bins=1, min_count=1)
    assert np.sum(np.isclose(unpooled[1], 1.0)) > np.sum(np.isclose(pooled[1], 1.0)), (
        f"pooled={pooled[1]}\nunpooled={unpooled[1]}"
    )


def test_spikein_preserves_global_mark_loss():
    """The central normalisation claim: only spike-in leaves genome-wide loss visible."""
    ds, truth = simulate(
        n_bins=3000, n_chroms=2, seed=7,
        perturbation={"global_mark_loss": {"H3K27me3": 0.3}, "responder_fraction": 0.0},
    )
    j = ds.marks.index("H3K27me3")
    correct = truth.depth_factors[("treated", "H3K27me3")] / truth.depth_factors[("control", "H3K27me3")]

    spike = compute_size_factors(ds, method="spikein", spikein=truth.spikein_counts,
                                 reference="control")[1, j]
    depth = compute_size_factors(ds, method="depth", reference="control")[1, j]

    assert spike == pytest.approx(correct, rel=0.05)
    # Depth normalisation absorbs the biology: it lands near correct * 0.3 instead.
    assert depth == pytest.approx(correct * 0.3, rel=0.25)


def test_normalize_warns_about_global_change():
    ds, _ = simulate(n_bins=500, n_chroms=1, seed=8)
    cs.normalize(ds, method="depth")
    assert "norm_warning" in ds.meta
    assert "spike-in" in ds.meta["norm_warning"]


def test_spikein_requires_counts():
    ds, _ = simulate(n_bins=200, n_chroms=1, seed=9)
    with pytest.raises(ValueError, match="requires the spikein"):
        compute_size_factors(ds, method="spikein")


# -- bigWig coverage ---------------------------------------------------------


def test_bigwig_bin_means_match_per_base_average(tmp_path, small_binset):
    """Intervals that straddle bin edges, gaps, and a truncated final bin all average exactly."""
    pyBigWig = pytest.importorskip("pyBigWig")
    from chromstate.io.counts import bin_bigwig

    path = str(tmp_path / "cov.bw")
    bw = pyBigWig.open(path, "w")
    bw.addHeader([("chr1", 2000), ("chr2", 1000)])
    # chr1: gaps (zero coverage), intervals crossing the 200 bp edges
    bw.addEntries(["chr1"] * 4, [50, 150, 420, 1900], ends=[150, 330, 600, 2000],
                  values=[2.0, 5.0, 1.5, 4.0])
    bw.addEntries(["chr2"], [950], ends=[1000], values=[10.0])
    bw.close()

    per_base = {"chr1": np.zeros(2000), "chr2": np.zeros(1000)}
    per_base["chr1"][50:150] = 2.0
    per_base["chr1"][150:330] = 5.0
    per_base["chr1"][420:600] = 1.5
    per_base["chr1"][1900:2000] = 4.0
    per_base["chr2"][950:1000] = 10.0
    expected = np.array([
        per_base[r.chrom][r.start:r.end].mean() for r in small_binset.table.itertuples()
    ])

    got = bin_bigwig(path, small_binset)
    np.testing.assert_allclose(got, expected)
    assert infer_format(path) == "bigwig"
    np.testing.assert_allclose(cs.load_signal(path, small_binset), expected)
