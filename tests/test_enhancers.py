"""Tests for the enhancer layer: candidate elements, activity, linking, ABC, importance.

Each test plants an answer and checks that it comes back: a peak whose distance to a TSS
is known, an element whose activity dominates its gene, an expression change generated
from near enhancers and not from far ones. The chromatin side uses the simulator, whose
true states say which bins are enhancers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import chromstate as cs
from chromstate.benchmark.metrics import match_states
from chromstate.enhancers import (
    DEFAULT_RINGS,
    _region_bin_matrix,
    powerlaw_contact,
    ring_columns,
)
from chromstate.features import FeatureSet
from chromstate.simulate import simulate

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _genes(rows) -> FeatureSet:
    return FeatureSet.from_frame(pd.DataFrame(
        rows, columns=["gene_id", "name", "chrom", "start", "end", "strand"]))


@pytest.fixture(scope="module")
def fitted():
    ds, truth = simulate(n_bins=900, n_chroms=2, samples=("naive", "exh"), seed=5)
    cs.normalize(ds, method="median_ratio")
    seg = cs.fit(ds, n_states=truth.n_states, n_restarts=1, seed=0)
    return ds, truth, seg


def _synthetic_genome(n_genes: int = 300, per_gene: int = 6, seed: int = 0):
    """Genes 300 kb apart on one chromosome, each with a promoter peak and distal peaks."""
    rng = np.random.default_rng(seed)
    genes, peaks = [], []
    for g in range(n_genes):
        tss = 1_000_000 + g * 300_000
        genes.append((f"G{g:04d}", f"Gene{g}", "chr1", tss, tss + 20_000, "+"))
        peaks.append(("chr1", tss - 200, tss + 300))
        offsets = rng.choice([-1, 1], per_gene) * rng.integers(3_000, 140_000, per_gene)
        for off in offsets:
            peaks.append(("chr1", tss + off - 250, tss + off + 250))
    features = _genes(genes)
    peaks = pd.DataFrame(peaks, columns=["chrom", "start", "end"]).drop_duplicates()
    return features, peaks, rng


def _activity(enh: pd.DataFrame, samples, rng, scale=None) -> pd.DataFrame:
    rows = []
    base = rng.lognormal(1.0, 1.0, len(enh))
    for s in samples:
        a = base * (scale[s] if scale is not None else 1.0)
        rows.append(pd.DataFrame({"enhancer_id": enh["enhancer_id"], "sample": s, "activity": a}))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Reading and candidates
# ---------------------------------------------------------------------------


def test_read_peaks_accepts_headed_tables_and_bed(tmp_path):
    headed = tmp_path / "atac.txt"
    headed.write_text(
        "Chr\tStart\tEnd\tpeakN\tGene.Name\tMean_Naive\tMean_Cl13\n"
        "chr1\t100\t400\tpeak1\tA\t5.0\t9.0\n"
        "chr1\t900\t1300\tpeak2\tB\t1.0\t0.5\n"
    )
    t = cs.read_peaks(str(headed))
    assert list(t["enhancer_id"]) == ["peak1", "peak2"]
    assert {"chrom", "start", "end", "Mean_Naive", "Mean_Cl13", "Gene.Name"} <= set(t.columns)

    bed = tmp_path / "peaks.bed"
    bed.write_text("track name=x\nchr2\t10\t60\tp1\nchr2\t80\t120\tp2\n")
    b = cs.read_peaks(str(bed))
    assert list(b["enhancer_id"]) == ["p1", "p2"] and b["start"].tolist() == [10, 80]

    bed3 = tmp_path / "peaks3.bed"
    bed3.write_text("chr2\t10\t60\n")
    assert cs.read_peaks(str(bed3))["enhancer_id"].iloc[0] == "chr2:10-60"


def test_duplicate_element_ids_are_rejected():
    peaks = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [0, 500], "end": [100, 600],
                          "enhancer_id": ["x", "x"]})
    with pytest.raises(ValueError, match="Duplicate"):
        cs.candidate_enhancers(peaks, _genes([("g", "g", "chr1", 0, 5000, "+")]))


def test_candidates_flag_promoters_by_distance_to_any_tss():
    features = _genes([
        ("plus", "Plus", "chr1", 10_000, 30_000, "+"),     # TSS 10,000
        ("minus", "Minus", "chr1", 50_000, 80_000, "-"),   # TSS 79,999
    ])
    peaks = pd.DataFrame({
        "chrom": ["chr1"] * 4 + ["chr2"],
        "start": [9_900, 11_500, 40_000, 80_500, 1_000],
        "end": [10_100, 11_800, 40_400, 80_900, 1_200],
        "enhancer_id": ["over_tss", "near", "distal", "minus_prom", "no_genes"],
    })
    c = cs.candidate_enhancers(peaks, features, tss_window=2000).set_index("enhancer_id")
    assert c.loc["over_tss", "nearest_tss_distance"] == 0
    assert c.loc["near", "nearest_tss_distance"] == 1_500
    assert c.loc["minus_prom", "nearest_gene"] == "minus"
    assert c.loc["minus_prom", "nearest_tss_distance"] == 80_500 - 79_999
    assert c.loc["over_tss", "is_promoter"] and c.loc["near", "is_promoter"]
    assert not c.loc["distal", "is_promoter"]
    assert np.isinf(c.loc["no_genes", "nearest_tss_distance"])
    assert not c.loc["no_genes", "is_promoter"]


# ---------------------------------------------------------------------------
# Linking
# ---------------------------------------------------------------------------


def test_link_reads_upstream_as_negative_on_either_strand():
    features = _genes([
        ("plus", "Plus", "chr1", 100_000, 150_000, "+"),    # TSS 100,000
        ("minus", "Minus", "chr1", 150_000, 200_001, "-"),  # TSS 200,000
    ])
    peaks = pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr1", "chr2"],
        "start": [76_900, 222_900, 99_900, 76_900],
        "end": [77_100, 223_100, 100_100, 77_100],
        "enhancer_id": ["up_plus", "up_minus", "prom_plus", "elsewhere"],
    })
    enh = cs.candidate_enhancers(peaks, features)
    pairs = cs.link_enhancers(enh, features, max_distance=100_000)
    p = pairs.set_index(["enhancer_id", "gene_id"])

    assert p.loc[("up_plus", "plus"), "signed_distance"] == -23_000
    assert p.loc[("up_minus", "minus"), "signed_distance"] == -23_000
    assert p.loc[("prom_plus", "plus"), "element_type"] == "own_promoter"
    assert p.loc[("prom_plus", "minus"), "element_type"] == "other_promoter"
    assert p.loc[("up_plus", "plus"), "element_type"] == "distal"
    assert ("up_plus", "minus") not in p.index          # 123 kb: outside the window
    assert "elsewhere" not in pairs["enhancer_id"].values


def test_powerlaw_contact_is_capped_and_decreasing():
    d = np.array([0, 1_000, 5_000, 10_000, 100_000])
    c = powerlaw_contact(d, gamma=1.0, min_distance=5_000)
    assert c[0] == c[1] == c[2] == 1.0
    assert c[3] == pytest.approx(0.5) and c[4] == pytest.approx(0.05)


def test_link_accepts_a_contact_function_for_hic():
    features, peaks, _ = _synthetic_genome(n_genes=5)
    enh = cs.candidate_enhancers(peaks, features)
    pairs = cs.link_enhancers(enh, features, contact=lambda t: np.full(len(t), 0.25))
    assert (pairs["contact"] == 0.25).all()
    with pytest.raises(ValueError, match="shape"):
        cs.link_enhancers(enh, features, contact=lambda t: np.ones(3))


# ---------------------------------------------------------------------------
# Activity on a fitted model
# ---------------------------------------------------------------------------


def test_peaks_narrower_than_a_bin_still_reach_their_bins():
    bins = pd.DataFrame({"chrom": "chr1", "start": np.arange(0, 2000, 200),
                         "end": np.arange(200, 2200, 200)})
    regions = pd.DataFrame({"chrom": ["chr1", "chr1", "chrX"],
                            "start": [1010, 1190, 0], "end": [1060, 1210, 100]})
    mat, n = _region_bin_matrix(bins, regions)
    assert n.tolist() == [1, 2, 0]
    assert np.allclose(np.asarray(mat.sum(axis=1)).ravel(), [1.0, 1.0, 0.0])


def test_state_classes_follow_the_emission_profile(fitted):
    ds, truth, seg = fitted
    mapping, _ = match_states(truth.states, seg.states, truth.n_states, seg.n_states)
    classes = cs.enhancer_state_classes(seg).set_index("state")["class"]
    by_truth = {truth.labels[t]: classes[p] for p, t in mapping.items()}
    assert by_truth["EnhA"] == "active"
    assert by_truth["TssA"] == "active"
    assert by_truth["TssBiv"] == "poised"
    assert by_truth["ReprPC"] == "repressed"
    assert by_truth["Het"] == "repressed"
    assert by_truth["Quies"] == "background"


def test_enhancer_peaks_carry_active_mass_and_quiet_peaks_do_not(fitted):
    ds, truth, seg = fitted
    sample = ds.samples[0]
    bins = ds.bins.loc[ds.sample_mask(sample)].reset_index(drop=True)
    true = truth.states_for(sample)
    pick = {}
    for label in ("EnhA", "Quies"):
        idx = np.flatnonzero(true == truth.labels.index(label))[:60]
        pick[label] = bins.iloc[idx][["chrom", "start", "end"]].assign(
            start=lambda f: f["start"] + 50, end=lambda f: f["start"] + 100)
    peaks = pd.concat(pick.values(), ignore_index=True)
    peaks["enhancer_id"] = [f"{k}_{i}" for k, v in pick.items() for i in range(len(v))]

    act = cs.enhancer_activity(seg, peaks)
    act = act.loc[act["sample"] == sample]
    is_enh = act["enhancer_id"].str.startswith("EnhA")
    assert (act["n_bins"] == 1).all()
    assert act.loc[is_enh, "active_mass"].mean() > 0.8
    assert act.loc[~is_enh, "active_mass"].mean() < 0.2
    assert act.loc[is_enh, "activity"].mean() > 5 * act.loc[~is_enh, "activity"].mean()
    post = act[[c for c in act.columns if c.startswith("post_")]].sum(axis=1)
    assert np.allclose(post, 1.0)


def test_activity_with_atac_is_the_geometric_mean(fitted):
    ds, _, seg = fitted
    peaks = pd.DataFrame({"chrom": [ds.bins["chrom"].iloc[0]] * 2,
                          "start": [1_000, 5_000], "end": [1_400, 5_200],
                          "enhancer_id": ["e1", "e2"]})
    atac = pd.DataFrame({"enhancer_id": ["e1", "e2"], "naive": [4.0, 9.0], "exh": [1.0, 0.0]})
    act = cs.enhancer_activity(seg, peaks, atac=atac)
    expected = np.sqrt(act["atac"] * act["signal_H3K27ac"])
    assert np.allclose(act["activity"], expected)
    assert act.loc[(act["sample"] == "exh") & (act["enhancer_id"] == "e2"), "activity"].iloc[0] == 0

    with pytest.raises(ValueError, match="no column for sample"):
        cs.enhancer_activity(seg, peaks, atac=atac[["enhancer_id", "naive"]])


def test_off_genome_elements_are_reported_not_invented(fitted):
    _, _, seg = fitted
    peaks = pd.DataFrame({"chrom": ["chrUn"], "start": [0], "end": [500]})
    act = cs.enhancer_activity(seg, peaks)
    assert (act["n_bins"] == 0).all() and act["activity"].isna().all()
    assert act.attrs["n_off_genome"] == len(act)


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


def test_abc_shares_sum_to_one_per_gene_and_rank_the_dominant_element():
    features, peaks, rng = _synthetic_genome(n_genes=40)
    enh = cs.candidate_enhancers(peaks, features)
    pairs = cs.link_enhancers(enh, features, max_distance=200_000)
    activity = _activity(enh, ["a", "b"], rng)

    target = pairs.loc[(pairs["gene_id"] == "G0010") & (pairs["element_type"] == "distal")]
    star = target.sort_values("distance").iloc[0]["enhancer_id"]
    activity.loc[activity["enhancer_id"] == star, "activity"] *= 1000

    scores = cs.abc_score(pairs, activity)
    totals = scores.groupby("gene_id")[["abc_a", "abc_b"]].sum()
    assert np.allclose(totals.to_numpy(), 1.0)
    top = scores.loc[scores["gene_id"] == "G0010"].sort_values("abc_a").iloc[-1]
    assert top["enhancer_id"] == star
    assert np.allclose(scores["abc_raw_a"], scores["activity_a"] * scores["contact"])


def test_abc_change_reports_both_share_and_absolute_input():
    features, peaks, rng = _synthetic_genome(n_genes=10)
    enh = cs.candidate_enhancers(peaks, features)
    pairs = cs.link_enhancers(enh, features)
    scores = cs.abc_score(pairs, _activity(enh, ["a", "b"], rng, scale={"a": 1.0, "b": 2.0}))
    change = cs.abc_change(scores, ["a"], ["b"])
    # Doubling every element doubles every input but leaves every share unchanged.
    assert np.allclose(change["delta_abc"], 0.0, atol=1e-12)
    assert np.allclose(change["abc_raw_b"], 2 * change["abc_raw_a"])
    assert (change["log2fc_activity"] > 0).all()
    with pytest.raises(ValueError, match="not in the ABC table"):
        cs.abc_change(scores, ["a"], ["c"])


# ---------------------------------------------------------------------------
# Gene features and importance
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained():
    features, peaks, rng = _synthetic_genome(n_genes=300, seed=3)
    enh = cs.candidate_enhancers(peaks, features)
    pairs = cs.link_enhancers(enh, features)
    activity = _activity(enh, ["a"], rng)
    scores = cs.abc_score(pairs, activity)
    feats = cs.gene_enhancer_features(scores)
    cols = ring_columns(DEFAULT_RINGS) + ["prom_input"]
    weights = np.array([2.0, 1.0, 0.4, 0.1, 1.5])
    y = feats[cols].to_numpy() @ weights + rng.normal(0, 0.05, len(feats))
    model = LinearRegression().fit(feats[cols], y)
    return scores, feats, cols, model


def test_gene_features_are_exact_ring_sums(trained):
    scores, feats, _, _ = trained
    g = "G0007"
    sub = scores.loc[(scores["gene_id"] == g) & (scores["element_type"] == "distal")
                     & (scores["distance"] < 10_000)]
    row = feats.loc[feats["gene_id"] == g].iloc[0]
    assert row["enh_0_10kb"] == pytest.approx(np.log1p(sub["abc_raw_a"].sum()))


def test_importance_is_an_exact_deletion_through_the_model(trained):
    scores, feats, cols, model = trained
    imp = cs.enhancer_importance(model, feats, scores, feature_columns=cols)
    row = imp.iloc[0]
    f = feats.loc[feats["gene_id"] == row["gene_id"]].iloc[0]
    ring = int(np.searchsorted(DEFAULT_RINGS, row["distance"], side="right") - 1)
    col = ring_columns(DEFAULT_RINGS)[ring]
    before = f[col]
    after = np.log1p(max(np.expm1(before) - row["abc_raw"], 0.0))
    expected = model.coef_[cols.index(col)] * (before - after)
    assert row["importance"] == pytest.approx(expected, rel=1e-9)
    assert (imp["importance"] >= -1e-9).all()          # positive weights: deletion only lowers
    assert set(imp.groupby("gene_id")["rank_in_gene"].min()) == {1}


def test_importance_puts_a_planted_enhancer_first(trained):
    scores, feats, cols, model = trained
    scores = scores.copy()
    # A 10-50 kb element made dominant must outrank the gene's nearer elements, which
    # the model weights more heavily per unit of input.
    sel = ((scores["gene_id"] == "G0100") & (scores["element_type"] == "distal")
           & scores["distance"].between(10_000, 49_999))
    assert sel.any()
    star = scores.loc[sel].index[0]
    scores.loc[star, "abc_raw_a"] *= 1e6
    feats2 = cs.gene_enhancer_features(scores)
    imp = cs.enhancer_importance(model, feats2, scores, feature_columns=cols)
    gene = imp.loc[imp["gene_id"] == "G0100"].sort_values("rank_in_gene")
    assert gene.iloc[0]["enhancer_id"] == scores.loc[star, "enhancer_id"]


def test_importance_requires_the_ring_features(trained):
    scores, feats, _, _ = trained
    model = LinearRegression().fit(feats[["prom_input"]], np.zeros(len(feats)))
    with pytest.raises(ValueError, match="ring column"):
        cs.enhancer_importance(model, feats, scores, feature_columns=["prom_input"])


# ---------------------------------------------------------------------------
# Association with expression
# ---------------------------------------------------------------------------


def _changed_genome(seed: int = 0):
    features, peaks, rng = _synthetic_genome(n_genes=300, seed=seed)
    enh = cs.candidate_enhancers(peaks, features)
    pairs = cs.link_enhancers(enh, features)
    base = rng.lognormal(1.0, 1.0, len(enh))
    fold = rng.lognormal(0.0, 1.0, len(enh))
    activity = pd.concat([
        pd.DataFrame({"enhancer_id": enh["enhancer_id"], "sample": "a", "activity": base}),
        pd.DataFrame({"enhancer_id": enh["enhancer_id"], "sample": "b", "activity": base * fold}),
    ], ignore_index=True)
    change = cs.abc_change(cs.abc_score(pairs, activity), ["a"], ["b"])
    return change, rng


def test_association_recovers_near_enhancers_and_not_far_ones():
    change, rng = _changed_genome()
    near = change.loc[(change["element_type"] == "distal") & (change["distance"] < 10_000)]
    per_gene = near.groupby("gene_id")[["abc_raw_a", "abc_raw_b"]].sum()
    lfc = np.log2(per_gene["abc_raw_b"] / per_gene["abc_raw_a"]) + rng.normal(0, 0.3, len(per_gene))
    rna = pd.DataFrame({"gene_id": per_gene.index, "log2fc": lfc.to_numpy()})

    linked = cs.link_enhancer_expression(change, rna)
    res = cs.enhancer_expression_association(linked, n_permutations=200)
    t = res.table.set_index("set")
    assert t.loc["0-10 kb", "spearman"] > 0.7 and t.loc["0-10 kb", "significant"]
    assert abs(t.loc["200-500 kb", "spearman"]) < 0.35    # ~3 SD of the null at n ~ 80
    assert "does not show" in res.note


def test_association_finds_nothing_when_expression_is_unrelated():
    change, rng = _changed_genome(seed=1)
    genes = change["gene_id"].unique()
    rna = pd.DataFrame({"gene_id": genes, "log2fc": rng.normal(0, 1, len(genes))})
    res = cs.enhancer_expression_association(cs.link_enhancer_expression(change, rna),
                                             n_permutations=200)
    assert (res.table["spearman"].abs() < 0.35).all()
    assert (res.table["pvalue"] > 0.01).all()


def test_link_warns_when_genes_barely_join():
    change, _ = _changed_genome()
    rna = pd.DataFrame({"gene_id": ["G0000"] + [f"ENSMUSG{i}" for i in range(9)],
                        "log2fc": np.zeros(10)})
    linked = cs.link_enhancer_expression(change, rna)
    assert linked.attrs["n_genes_joined"] == 1 and "warning" in linked.attrs


def test_importance_warns_when_the_model_uses_the_peak_count(trained):
    scores, feats, cols, _ = trained
    use = cols + ["n_distal"]
    model = LinearRegression().fit(feats[use], feats[cols].sum(axis=1))
    with pytest.warns(RuntimeWarning, match="n_distal"):
        cs.enhancer_importance(model, feats, scores, feature_columns=use)
