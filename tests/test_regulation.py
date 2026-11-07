"""Tests for per-state gene quantification and state importance.

Expression here is generated from the simulated chromatin with a known rule, so the
ranking has an answer: the state the rule uses must come first, with the right sign, and
a state the rule ignores must carry no unique importance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chromstate import regulation as rg
from chromstate.features import FeatureSet
from chromstate.model.core import fit
from chromstate.simulate import simulate


def FAST(seed):
    """A small booster: the tests check recovery of a planted rule, not peak accuracy."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(max_iter=60, learning_rate=0.15, random_state=seed)


@pytest.fixture(scope="module")
def fitted():
    ds, truth = simulate(n_bins=4000, n_chroms=5, samples=("a", "b"), seed=3, blur=0.15)
    seg = fit(ds, n_states=7, n_restarts=2, seed=0)
    rng = np.random.default_rng(0)
    rows = []
    for c in pd.unique(ds.bins["chrom"]):
        for i, start in enumerate(range(5_000, 4000 * 200 - 20_000, 12_000)):
            rows.append({"gene_id": f"{c}_g{i}", "chrom": c, "start": start,
                         "end": start + 6_000, "strand": "+" if rng.random() < 0.5 else "-"})
    features = FeatureSet(pd.DataFrame(rows))
    gq = rg.gene_quantification(seg, features)
    return seg, features, gq


def _state(seg, key):
    return next(lbl for lbl in seg.labels if key(lbl))


def test_intensity_is_zero_at_the_state_mean(fitted):
    seg, _, _ = fitted
    ds = seg.dataset
    mu = seg.model.emissions.mean_matrix()
    k = int(np.argmax(mu.sum(axis=1)))
    # A synthetic row sitting exactly on state k's expected signal has intensity 0 for k.
    x = mu[k][None, :] * ds.size_factors[:1]
    Z = np.log1p(x / ds.size_factors[:1])
    w = rg.state_mark_weights(seg)
    inten = Z @ w.T - (w * np.log1p(mu)).sum(axis=1)[None, :]
    assert abs(inten[0, k]) < 1e-9
    assert np.allclose(w.sum(axis=1), 1.0)


def test_intensity_tracks_magnitude_within_a_state(fitted):
    seg, _, _ = fitted
    inten = rg.state_intensity(seg)
    k = int(np.argmax(seg.model.emissions.mean_matrix().sum(axis=1)))
    in_k = seg.states == k
    total = np.log1p(seg.dataset.X[in_k]).sum(axis=1)
    # Among bins called state k, stronger signal means higher intensity for k.
    assert np.corrcoef(total, inten[in_k, k])[0, 1] > 0.5


def test_gene_table_shape_and_columns(fitted):
    seg, features, gq = fitted
    assert set(gq["sample"]) == {"a", "b"}
    assert gq["gene_id"].is_unique is False  # one row per gene per sample
    for lbl in seg.labels:
        for q in rg.STATE_QUANTITIES:
            assert f"P_{q}_{lbl}" in gq.columns
            assert f"B_{q}_{lbl}" in gq.columns
    occ = gq[[f"P_occ_{lbl}" for lbl in seg.labels]].to_numpy()
    assert np.allclose(occ.sum(axis=1), 1.0, atol=1e-6)
    assert gq["chrom"].notna().all()


def test_fixed_mark_weights_are_used(fitted):
    seg, features, gq = fitted
    w = np.full((seg.n_states, len(seg.marks)), 1.0 / len(seg.marks))
    gq2 = rg.gene_quantification(seg, features, mark_weights=w)
    col = f"P_int_{seg.labels[-1]}"
    assert not np.allclose(gq[col].to_numpy(), gq2[col].to_numpy())


def test_state_importance_recovers_the_rule(fitted):
    seg, _, gq = fitted
    act = _state(seg, lambda s: s.startswith("TssA"))
    rep = _state(seg, lambda s: "Repr" in s)
    sub = gq.loc[gq["sample"] == "a"].set_index("gene_id")
    rng = np.random.default_rng(1)
    y = 3.0 * sub[f"P_occ_{act}"] - 2.0 * sub[f"P_occ_{rep}"] + rng.normal(0, 0.2, len(sub))
    task = rg.expression_level_task(gq, pd.DataFrame({"a": y}), ["a"])
    imp = rg.state_importance(task, model=FAST, n_boot=50, n_perm=2)
    top2 = set(imp.table["state"].head(2))
    assert top2 == {act, rep}
    t = imp.table.set_index("state")
    assert t.loc[act, "direction"] > 0 > t.loc[rep, "direction"]
    ignored = [s for s in seg.labels if s not in (act, rep)]
    assert (t.loc[ignored, "unique_r2"] < 0.05).all()
    assert imp.full_r2 > 0.5


def test_change_task_and_representations(fitted):
    seg, _, gq = fitted
    act = _state(seg, lambda s: s.startswith("TssA"))
    a = gq.loc[gq["sample"] == "a"].set_index("gene_id")
    b = gq.loc[gq["sample"] == "b"].set_index("gene_id")
    rng = np.random.default_rng(2)
    lfc = 4.0 * (b[f"P_dose_{act}"] - a[f"P_dose_{act}"]) + rng.normal(0, 0.05, len(a))
    task = rg.expression_change_task(gq, lfc, ["a"], ["b"])
    assert task.kind == "change"
    assert len(task.y) == len(a)
    rep = rg.compare_representations(task, model=FAST, n_boot=50)
    scores = rep.scores.set_index("family")
    assert {"top_state", "occupancy", "quantitative", "signal"} <= set(scores.index)
    assert scores.loc["quantitative", "spearman"] >= scores.loc["top_state", "spearman"]
    assert not rep.contrasts.empty


def test_ridge_backend_runs(fitted):
    _, _, gq = fitted
    sub = gq.loc[gq["sample"] == "a"].set_index("gene_id")
    y = sub.filter(like="P_occ_").iloc[:, -1]
    task = rg.expression_level_task(gq, pd.DataFrame({"a": y}), ["a"])
    pred = rg.cross_validated_predictions(task.X.to_numpy(float), task.y.to_numpy(float),
                                          task.groups, model="ridge")
    assert np.isfinite(pred).all()


def test_bad_inputs_raise(fitted):
    _, _, gq = fitted
    with pytest.raises(ValueError):
        rg.expression_level_task(gq, pd.DataFrame({"zzz": [1.0]}))
    with pytest.raises(ValueError):
        rg.make_model("nope")
    with pytest.raises(ValueError):
        rg.rank_states_for_expression(gq)


def test_folds_keep_stacked_gene_copies_together():
    # Level task stacks samples: gene g appears once per sample, all on one chromosome.
    # Fallback blocks must be cut over unique genes, or a gene's copies leak across folds.
    import numpy as np
    from chromstate.regulation import _folds

    genes = np.array([f"g{i}" for i in range(60)])
    keys = np.tile(genes, 4)                      # 4 stacked samples
    groups = np.repeat("chr19", len(keys))
    for tr, te in _folds(groups, 5, keys=keys):
        assert not set(keys[tr]) & set(keys[te])
