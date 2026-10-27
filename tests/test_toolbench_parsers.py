"""Parsers for the real-tool adapters (ChromHMM, EpiCSeg, normR, diHMM), on fixture files.

The adapters were verified against the real software (docs/tool_comparison.md), but CI
cannot install Java, R/Bioconductor, or a patched C++ build. What can be pinned down
without them is the part that fails silently: reading each tool's output back onto the
right bins. Every fixture below is written in the exact layout the real tool produced
during verification, and each test checks the alignment, not just that parsing succeeds.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

from chromstate.simulate import simulate

TOOLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from toolbench.core import UNASSIGNED  # noqa: E402
from toolbench.methods import chromhmm, dihmm, epicseg, normr  # noqa: E402
from toolbench.scores import posthoc_scores, state_profiles  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    ds, truth = simulate(n_bins=60, n_chroms=2, seed=5, blur=0.0)
    return ds, truth


def _rows(ds, sample, chrom):
    b = ds.bins.reset_index(drop=True)
    return np.flatnonzero(((b["sample"] == sample) & (b["chrom"] == chrom)).to_numpy())


# -- ChromHMM -----------------------------------------------------------------


def test_chromhmm_posterior_files_assemble_in_bin_order(tiny, tmp_path):
    ds, _ = tiny
    K = 3
    post_dir = tmp_path / "POSTERIOR"
    post_dir.mkdir()
    expected = np.zeros((ds.n_rows, K))
    for sample in ds.samples:
        for chrom in ("chr1", "chr2"):
            rows = _rows(ds, sample, chrom)
            block = np.zeros((rows.size, K))
            block[np.arange(rows.size), np.arange(rows.size) % K] = 1.0  # cycling argmax
            expected[rows] = block
            with open(post_dir / f"{sample}_{K}_{chrom}_posterior.txt", "w") as fh:
                fh.write(f"{sample}\t{chrom}\nE1\tE2\tE3\n")
                for r in block:
                    fh.write("\t".join(f"{v:g}" for v in r) + "\n")
    P = chromhmm.read_posterior(ds, str(tmp_path), K)
    np.testing.assert_allclose(P, expected)


def test_chromhmm_posterior_with_wrong_row_count_is_rejected(tiny, tmp_path):
    ds, _ = tiny
    (tmp_path / "POSTERIOR").mkdir()
    for sample in ds.samples:
        for chrom in ("chr1", "chr2"):
            n = _rows(ds, sample, chrom).size - (1 if chrom == "chr2" else 0)
            with open(tmp_path / "POSTERIOR" / f"{sample}_2_{chrom}_posterior.txt", "w") as fh:
                fh.write(f"{sample}\t{chrom}\nE1\tE2\n" + "0.5\t0.5\n" * n)
    assert chromhmm.read_posterior(ds, str(tmp_path), 2) is None


def test_chromhmm_command_disables_assembly_enrichment(tiny, tmp_path, monkeypatch):
    """Without -noenrich the real jar exits 1 on non-assembly chromosomes."""
    ds, _ = tiny
    monkeypatch.setenv("CHROMHMM_JAR", "/x/ChromHMM.jar")
    cmd = chromhmm.ChromHMMAdapter().dry_run_command(ds, 5, 0, str(tmp_path))
    assert "-noenrich" in cmd and "-printposterior" in cmd and "LearnModel" in cmd


# -- EpiCSeg ------------------------------------------------------------------


def test_epicseg_inputs_match_getcounts_format(tiny, tmp_path):
    ds, _ = tiny
    out = epicseg.write_epicseg_inputs(ds, str(tmp_path))
    with open(out["regions"]) as fh:
        regions = [ln.split() for ln in fh]
    assert regions == [["chr1", "0", str(60 * 200)], ["chr2", "0", str(60 * 200)]]
    for sample, path in out["counts"].items():
        frame = pd.read_csv(path, sep="\t")
        assert list(frame.columns) == list(ds.marks)
        assert all(frame[c].dtype.kind == "i" for c in frame.columns)
        np.testing.assert_array_equal(frame.to_numpy(), np.rint(ds.X[out["orders"][sample]]))


def test_epicseg_regions_split_at_gaps(tiny):
    ds, _ = tiny
    rows = epicseg.sample_order(ds, ds.samples[0])
    keep = np.r_[rows[:10], rows[15:60], rows[60:]]  # drop 5 bins from chr1
    regions = epicseg.contiguous_regions(ds, keep)
    assert regions == [("chr1", 0, 2000), ("chr1", 3000, 12000), ("chr2", 0, 12000)]


def test_epicseg_outputs_map_by_coordinate_and_posterior_order(tiny, tmp_path):
    ds, _ = tiny
    K = 3
    orders = {s: epicseg.sample_order(ds, s) for s in ds.samples}
    for i, (sample, rows) in enumerate(orders.items()):
        labels = (np.arange(rows.size) // 7 + i) % K + 1  # 1-based like EpiCSeg
        P = np.full((rows.size, K), 0.1)
        P[np.arange(rows.size), labels - 1] = 0.8
        np.savetxt(tmp_path / f"posterior_{sample}.tsv", P, delimiter="\t")
        bins = ds.bins.reset_index(drop=True).loc[rows]
        with open(tmp_path / f"segmentation_{sample}.bed", "w") as fh:
            # BED9 as EpiCSeg writes it, one line per bin, in reverse order on purpose
            for (_, b), lab in reversed(list(zip(bins.iterrows(), labels))):
                fh.write(f"{b.chrom}\t{b.start}\t{b.end}\t{lab}\t0\t.\t{b.start}\t{b.end}\t0,0,0\n")
    states, post, vocab = epicseg.read_epicseg_outputs(ds, str(tmp_path), orders, K)
    assert vocab == ["1", "2", "3"]
    assert (states != UNASSIGNED).all()
    np.testing.assert_array_equal(states, post.argmax(axis=1))


def test_epicseg_posterior_shape_mismatch_is_an_error(tiny, tmp_path):
    ds, _ = tiny
    orders = {ds.samples[0]: epicseg.sample_order(ds, ds.samples[0])}
    (tmp_path / "segmentation.bed").write_text("chr1\t0\t200\t1\n")
    np.savetxt(tmp_path / "posterior.tsv", np.full((5, 2), 0.5))
    with pytest.raises(ValueError, match="shape"):
        epicseg.read_epicseg_outputs(ds, str(tmp_path), orders, 2)


def test_epicseg_launcher_interpreter_comes_from_shebang(tmp_path):
    fake_r = tmp_path / "Rscript"
    fake_r.write_text("")
    launcher = tmp_path / "epicseg.R"
    launcher.write_text(f"#!{fake_r}\ncat('loading epicseg\\n')\n")
    assert epicseg.launcher_interpreter(str(launcher)) == str(fake_r)


# -- normR --------------------------------------------------------------------


def test_normr_patterns_are_named_and_ranked_by_frequency():
    calls = np.array([[1, 0, 1], [1, 0, 1], [0, 0, 0], [1, 0, 1], [0, 1, 0], [0, 0, 0]])
    states, names = normr.patterns_to_states(calls, ["K4me3", "K4me1", "K27me3"])
    assert names[0] == "K4me3+K27me3" and names[1] == "none" and names[2] == "K4me1"
    np.testing.assert_array_equal(states, [0, 0, 1, 0, 2, 1])


def _normr_fixture(ds, sample, path, shuffle_seed=0):
    bins = ds.bins.reset_index(drop=True)
    sub = bins.loc[bins["sample"] == sample, ["chrom", "start", "end"]].copy()
    rng = np.random.default_rng(shuffle_seed)
    for j, m in enumerate(ds.marks):
        sub[f"{m}_post"] = (sub["start"] // 200 % (j + 2) == 0).astype(float) * 0.9
        sub[f"{m}_q"] = np.where(sub[f"{m}_post"] > 0.5, 0.01, np.nan)
        sub[f"{m}_class"] = (sub[f"{m}_post"] > 0.5).astype(int)
        sub[f"{m}_enr"] = sub[f"{m}_post"] * 2 - 1
    sub = sub.iloc[rng.permutation(len(sub))]  # rows out of order: must still align
    sub.to_csv(path, sep="\t", index=False, na_rep="NA")
    return sub


def test_normr_table_aligns_by_coordinate(tiny, tmp_path):
    ds, _ = tiny
    sample = ds.samples[1]
    written = _normr_fixture(ds, sample, tmp_path / "t.tsv")
    tab = normr.read_normr_table(ds, str(tmp_path / "t.tsv"), sample)
    bins = ds.bins.reset_index(drop=True)
    assert set(tab.index) == set(np.flatnonzero((bins["sample"] == sample).to_numpy()))
    lookup = written.set_index(["chrom", "start"])
    for row in tab.index[:20]:
        key = (bins.loc[row, "chrom"], bins.loc[row, "start"])
        assert tab.loc[row, f"{ds.marks[0]}_post"] == lookup.loc[key, f"{ds.marks[0]}_post"]


def test_normr_bivalency_is_product_of_k4me3_and_k27me3_posteriors(tiny, tmp_path):
    ds, _ = tiny
    tables = {}
    for s in ds.samples:
        _normr_fixture(ds, s, tmp_path / f"{s}.tsv")
        tables[s] = normr.read_normr_table(ds, str(tmp_path / f"{s}.tsv"), s)
    states, names, biv, act = normr.scores_from_tables(ds, tables)
    assert (states != UNASSIGNED).all() and len(names) == states.max() + 1
    k4, k27 = ds.marks.index("H3K4me3"), ds.marks.index("H3K27me3")
    tab = tables[ds.samples[0]]
    rows = tab.index.to_numpy()
    np.testing.assert_allclose(biv[rows], tab[f"{ds.marks[k4]}_post"] *
                               tab[f"{ds.marks[k27]}_post"])
    assert np.isfinite(act).all()


# -- diHMM --------------------------------------------------------------------


def test_dihmm_outputs_are_cropped_not_shifted(tiny, tmp_path):
    ds, _ = tiny
    domain = 20
    for sample in ds.samples:
        for chrom in ("chr1", "chr2"):
            n = _rows(ds, sample, chrom).size
            keep = n - (n % domain) - (domain if chrom == "chr2" else 0)  # chr2 cropped more
            with open(tmp_path / f"{sample}_{chrom}_dihmm.tsv", "w") as fh:
                fh.write("bin_state\tdomain_state\n")
                for i in range(keep):
                    fh.write(f"{i % 3}\t{i // domain % 2}\n")
    bins, doms, found = dihmm.read_dihmm_outputs(ds, str(tmp_path))
    assert found == 4
    r1 = _rows(ds, ds.samples[0], "chr1")
    np.testing.assert_array_equal(bins[r1], np.arange(r1.size) % 3)
    r2 = _rows(ds, ds.samples[0], "chr2")
    assert (bins[r2[-domain:]] == UNASSIGNED).all(), "cropped tail must stay unassigned"
    assert (doms[r2[:domain]] == 0).all() and (doms[r2[domain:2 * domain]] == 1).all()


def test_dihmm_output_longer_than_input_is_rejected(tiny, tmp_path):
    ds, _ = tiny
    n = _rows(ds, ds.samples[0], "chr1").size
    with open(tmp_path / f"{ds.samples[0]}_chr1_dihmm.tsv", "w") as fh:
        fh.write("bin_state\tdomain_state\n" + "0\t0\n" * (n + 1))
    with pytest.raises(ValueError, match="rows for"):
        dihmm.read_dihmm_outputs(ds, str(tmp_path))


def test_dihmm_asks_for_equal_bin_and_domain_states(tiny, tmp_path, monkeypatch):
    """The C++ implementation fails unless the two counts are equal."""
    ds, _ = tiny
    monkeypatch.setenv("DIHMM_BUILD", "/x")
    cmd = dihmm.DiHMMAdapter().dry_run_command(ds, 6, 0, str(tmp_path))
    assert "--n-bin-states 6" in cmd and "--n-domain-states 6" in cmd


# -- shared post-hoc scores and availability ----------------------------------


def test_posthoc_scores_from_hard_labels_and_posteriors_agree(tiny):
    ds, truth = tiny
    states = np.asarray(truth.states, dtype=np.int64)
    K = int(states.max()) + 1
    onehot = np.eye(K)[states]
    b1, a1 = posthoc_scores(ds, states, None, K)
    b2, a2 = posthoc_scores(ds, states, onehot, K)
    np.testing.assert_allclose(b1, b2)
    np.testing.assert_allclose(a1, a2)
    # the truly bivalent state should get the highest profile-based bivalency
    biv_state = truth.bivalent_state
    assert b1[states == biv_state].mean() > b1[states != biv_state].mean()


def test_unassigned_bins_contribute_nothing(tiny):
    ds, truth = tiny
    states = np.asarray(truth.states, dtype=np.int64).copy()
    states[:10] = UNASSIGNED
    profiles, W = state_profiles(ds, states, None, int(states.max()) + 1)
    assert (W[:10] == 0).all()
    assert np.isfinite(profiles).all()


@pytest.mark.parametrize("adapter,var", [
    (epicseg.EpiCSegAdapter(), "EPICSEG_R"),
    (normr.NormRAdapter(), "NORMR_RSCRIPT"),
    (dihmm.DiHMMAdapter(), "DIHMM_BUILD"),
])
def test_real_tool_adapters_skip_cleanly_when_not_configured(adapter, var, monkeypatch, tiny,
                                                             tmp_path):
    monkeypatch.delenv(var, raising=False)
    avail = adapter.availability()
    assert not avail.available and var in avail.detail and avail.how_to_enable
    run = adapter.run(tiny[0], 3, 0, str(tmp_path), 10.0)
    assert run.status == "skipped" and run.reason
