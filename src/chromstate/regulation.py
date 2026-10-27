"""Quantify every chromatin state per gene, and rank states by how much expression they explain.

A segmentation says which state a promoter is in. To ask which states *regulate*
expression, each state needs to be a number per gene that can sit in a regression next to
RNA-seq. This module produces those numbers and the held-out tests that rank them.

**Three quantities per state, because "how much of the state" is two questions.**

``occ`` (occupancy)
    Mean posterior of the state over the gene's bins: *how much of the region* is in it.
``int`` (intensity)
    Posterior-weighted log fold of the observed signal over the state's own expected
    signal, on the marks that define the state: *how strongly* the region expresses the
    state where it is in it. Zero means "a typical instance of this state". Occupancy is
    blind to this by construction -- two Active TSS promoters with 5 and 50 H3K4me3 reads
    both have occupancy near 1 -- and it is exactly the information raw signal carried
    over the state composition in the CD8 T-cell analysis.
``dose``
    Occupancy weighted by the fold, ``mean_b gamma_bk * exp(I_bk)``: the amount of state,
    counting a strong instance for more than a weak one. One number when a model or a
    reader wants a single column per state.

**Importance is reported two ways, because they answer different questions.**
``marginal_rho`` is how well a state predicts expression on its own; ``unique_r2`` is how
much held-out R^2 is lost when that state's columns are permuted inside the full model,
i.e. what no other state can stand in for. Redundant states (Active TSS and Flanking TSS
carry overlapping information) score high on the first and low on the second, and that
disagreement is informative rather than a contradiction. ``direction`` is the sign of the
state's association with expression, so an important repressive state reads as negative.

Every score is computed on held-out chromosomes (genes on one chromosome are never on both
sides of a split), and comparisons between representations are paired bootstrap
differences over genes. As everywhere in ChromState, these are associations: a state that
predicts expression is not thereby shown to cause it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse

from .features import FeatureSet, map_bins_to_regions
from .model.annotate import canonical_mark
from .model.core import Segmentation

EPS = 1e-12

#: Column prefix for each gene part.
PART_PREFIX = {"promoter": "P", "body": "B", "gene": "G"}

#: Per-state quantities, in column order.
STATE_QUANTITIES = ("occ", "int", "dose")

#: Occupancy below which a state's intensity is shrunk towards 0 ("typical") rather than
#: estimated from a sliver of posterior mass.
INTENSITY_SHRINK = 0.05


# ---------------------------------------------------------------------------
# Per-bin quantities
# ---------------------------------------------------------------------------


def state_mark_weights(seg: Segmentation, occupancy: Optional[np.ndarray] = None) -> np.ndarray:
    """``(K, M)`` weight of each mark in defining each state; rows sum to 1.

    Weights are fold enrichment of the state's emission mean over the genome-wide
    (occupancy-weighted) mean of that mark, not raw means: marks differ several-fold in
    reads per bin (sharp H3K4me3 versus broad H3K27me3), and raw means would let the
    deepest mark define every state.

    ``occupancy`` (length K) overrides the occupancy taken from ``seg``. Pass the
    genome-wide occupancy when ``seg`` covers only part of the genome -- one chromosome
    decoded at a time -- so every chromosome gets the same weights.
    """
    mu = np.maximum(seg.model.emissions.mean_matrix(), EPS)
    occ = seg.posterior.mean(axis=0) if occupancy is None else np.asarray(occupancy, float)
    occ = occ / max(occ.sum(), EPS)
    genome = np.maximum(occ @ mu, EPS)
    enr = mu / genome[None, :]
    return enr / enr.sum(axis=1, keepdims=True)


def state_intensity(seg: Segmentation, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """``(n_rows, K)`` log fold of each bin's signal over each state's expected signal.

    ``I[b, k] = sum_m w[k, m] * (log1p(x[b, m] / s[b, m]) - log1p(mu[k, m]))`` with ``w``
    from :func:`state_mark_weights`. Size factors divide the observation here only to put
    it on the emission scale; the model itself never rescales counts.

    Bernoulli (binarised) emissions have no magnitude to compare against, so intensity is
    0 everywhere for them. ``weights`` overrides :func:`state_mark_weights`.
    """
    ds = seg.dataset
    if seg.model.emissions.name == "bernoulli":
        return np.zeros((ds.n_rows, seg.n_states))
    w = state_mark_weights(seg) if weights is None else np.asarray(weights, float)
    mu = np.maximum(seg.model.emissions.mean_matrix(), 0.0)
    Z = np.log1p(ds.X / np.maximum(ds.size_factors, EPS))
    return Z @ w.T - (w * np.log1p(mu)).sum(axis=1)[None, :]


def _bin_entropy(post: np.ndarray) -> np.ndarray:
    k = post.shape[1]
    h = -(post * np.log(np.clip(post, EPS, 1.0))).sum(axis=1)
    return h / np.log(k) if k > 1 else h


def _bivalency_columns(seg: Segmentation, Z: np.ndarray) -> Dict[str, np.ndarray]:
    """Signal-level bivalency per bin, if the panel has both promoter marks."""
    canon = [canonical_mark(m) for m in seg.marks]
    a_key, r_key = canonical_mark("H3K4me3"), canonical_mark("H3K27me3")
    if a_key not in canon or r_key not in canon:
        return {}
    a, r = Z[:, canon.index(a_key)], Z[:, canon.index(r_key)]
    denom = np.maximum(a + r, EPS)
    magnitude = np.sqrt(np.maximum(a * r, 0.0))
    return {
        "biv_score": 2.0 * a * r / denom,
        "biv_magnitude": magnitude,
        "biv_balance": 2.0 * magnitude / denom,
    }


# ---------------------------------------------------------------------------
# Gene-level quantification
# ---------------------------------------------------------------------------


def _averaging_matrix(bins: pd.DataFrame, regions: pd.DataFrame, assign: str):
    """Sparse ``(genes x bins)`` matrix whose rows average each gene's bins."""
    pairs = map_bins_to_regions(bins, regions, assign=assign)
    genes = pd.Index(pd.unique(pairs["gene_id"]))
    rows = genes.get_indexer(pairs["gene_id"])
    counts = np.bincount(rows, minlength=len(genes))
    G = sparse.csr_matrix(
        (1.0 / counts[rows], (rows, pairs["bin_row"].to_numpy())), shape=(len(genes), len(bins))
    )
    return G, genes, counts


def gene_quantification(
    seg: Segmentation,
    features: FeatureSet,
    parts: Sequence[str] = ("promoter", "body"),
    part_kwargs: Optional[Dict[str, dict]] = None,
    assign: str = "all",
    include_signal: bool = True,
    mark_weights: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Quantify every state in every gene part, per sample: the table that meets RNA-seq.

    One row per (gene, sample). For each part (prefix ``P_`` promoter, ``B_`` body) and
    each state: ``occ_<state>``, ``int_<state>``, ``dose_<state>`` (see the module
    docstring), plus ``entropy``, signal-level bivalency (``biv_score``,
    ``biv_magnitude``, ``biv_balance``) when H3K4me3 and H3K27me3 are both present, and
    with ``include_signal`` the mean ``log1p`` normalised signal per mark (``sig_<mark>``)
    as the raw-signal comparator. ``chrom`` is carried for chromosome-grouped
    cross-validation; ``gene_id`` is the join key for expression.

    ``part_kwargs`` passes window settings through, e.g.
    ``{"promoter": {"upstream": 1000, "downstream": 1000}}``. A gene with no bins in a
    part (a short gene's trimmed body) has NaN for that part, which the gradient-boosted
    models here handle natively.

    ``mark_weights`` fixes the intensity weights (see :func:`state_mark_weights`); pass
    them when quantifying one chromosome at a time so intensities are comparable across
    chromosomes.
    """
    ds = seg.dataset
    part_kwargs = part_kwargs or {}
    labels = list(seg.labels)
    intensity = state_intensity(seg, mark_weights)
    Z = np.log1p(ds.X / np.maximum(ds.size_factors, EPS))
    regions = {p: features.regions(p, **part_kwargs.get(p, {})) for p in parts}

    frames = []
    for sample in ds.samples:
        mask = ds.sample_mask(sample)
        bins = ds.bins.loc[mask, ["chrom", "start", "end"]].reset_index(drop=True)
        post = seg.posterior[mask]
        inten = intensity[mask]
        z = Z[mask]
        per_bin_extra = {"entropy": _bin_entropy(post), **_bivalency_columns(seg, z)}

        sample_frame = None
        for part in parts:
            G, genes, n_bins = _averaging_matrix(bins, regions[part], assign)
            pre = PART_PREFIX.get(part, part)
            occ = np.asarray(G @ post)
            weighted = np.asarray(G @ (post * inten))
            inten_gene = weighted / np.maximum(occ, INTENSITY_SHRINK)
            dose = np.asarray(G @ (post * np.exp(np.clip(inten, -20.0, 20.0))))
            cols = {f"{pre}_n_bins": n_bins}
            for k, lbl in enumerate(labels):
                cols[f"{pre}_occ_{lbl}"] = occ[:, k]
                cols[f"{pre}_int_{lbl}"] = inten_gene[:, k]
                cols[f"{pre}_dose_{lbl}"] = dose[:, k]
            for name, v in per_bin_extra.items():
                cols[f"{pre}_{name}"] = np.asarray(G @ v)
            if include_signal:
                sig = np.asarray(G @ z)
                for j, m in enumerate(ds.marks):
                    cols[f"{pre}_sig_{m}"] = sig[:, j]
            frame = pd.DataFrame(cols, index=pd.Index(genes, name="gene_id"))
            sample_frame = frame if sample_frame is None else sample_frame.join(frame, how="outer")
        sample_frame.insert(0, "sample", sample)
        frames.append(sample_frame.reset_index())

    out = pd.concat(frames, ignore_index=True)
    meta = features.table.set_index("gene_id")[["name", "chrom"]]
    out.insert(1, "name", out["gene_id"].map(meta["name"]))
    out.insert(2, "chrom", out["gene_id"].map(meta["chrom"]))
    out.attrs.update({"states": labels, "marks": list(ds.marks), "parts": list(parts),
                      "assign": assign})
    return out


# ---------------------------------------------------------------------------
# Expression tasks
# ---------------------------------------------------------------------------


@dataclass
class ExpressionTask:
    """Features, target, and CV groups for one question about expression."""

    X: pd.DataFrame
    y: pd.Series
    groups: np.ndarray
    kind: str
    description: str


def _feature_columns(table: pd.DataFrame) -> List[str]:
    skip = {"gene_id", "name", "chrom", "sample"}
    return [c for c in table.columns
            if c not in skip and not c.endswith("_n_bins") and pd.api.types.is_numeric_dtype(table[c])]


def expression_level_task(
    gene_q: pd.DataFrame, expression: pd.DataFrame, samples: Optional[Sequence[str]] = None
) -> ExpressionTask:
    """Expression *level* across genes, from each sample's own chromatin.

    ``expression`` is genes x samples (index ``gene_id``), already on a log scale. Several
    samples are stacked; grouping is by chromosome, so a gene never appears on both sides
    of a split even when it is stacked.
    """
    samples = list(samples or [s for s in gene_q["sample"].unique() if s in expression.columns])
    if not samples:
        raise ValueError("no sample is in both the gene table and the expression columns")
    cols = _feature_columns(gene_q)
    Xs, ys, gs = [], [], []
    for s in samples:
        sub = gene_q.loc[gene_q["sample"] == s].set_index("gene_id")
        genes = sub.index.intersection(expression.index[expression[s].notna()])
        Xs.append(sub.loc[genes, cols])
        ys.append(expression.loc[genes, s])
        gs.append(sub.loc[genes, "chrom"].to_numpy())
    X = pd.concat(Xs)
    return ExpressionTask(X, pd.concat(ys).rename("y"), np.concatenate(gs), "level",
                          f"expression level in {', '.join(samples)}")


def expression_change_task(
    gene_q: pd.DataFrame,
    lfc: pd.Series,
    group_a: Sequence[str],
    group_b: Sequence[str],
    include_baseline: bool = False,
) -> ExpressionTask:
    """Expression *change* (``lfc``, b over a) from the change in chromatin.

    Features are group-mean(b) minus group-mean(a) of every quantity. With
    ``include_baseline`` the group-a values are added as ``base_`` columns, which asks
    whether the starting chromatin conditions how much a gene can move.
    """
    cols = _feature_columns(gene_q)
    a = gene_q.loc[gene_q["sample"].isin(group_a)].groupby("gene_id")[cols].mean()
    b = gene_q.loc[gene_q["sample"].isin(group_b)].groupby("gene_id")[cols].mean()
    genes = a.index.intersection(b.index).intersection(lfc.dropna().index)
    X = b.loc[genes] - a.loc[genes]
    if include_baseline:
        X = X.join(a.loc[genes].add_prefix("base_"))
    chrom = gene_q.drop_duplicates("gene_id").set_index("gene_id")["chrom"]
    return ExpressionTask(X, lfc.loc[genes].rename("y"), chrom.loc[genes].to_numpy(), "change",
                          f"log2 fold change {'+'.join(group_b)} vs {'+'.join(group_a)}")


# ---------------------------------------------------------------------------
# Models and cross-validation
# ---------------------------------------------------------------------------


ModelSpec = Union[str, Callable[[int], object]]


def make_model(model: ModelSpec = "hgb", seed: int = 0):
    """``hgb`` (scikit-learn histogram gradient boosting, default), ``xgboost``, ``ridge``,
    or a callable ``seed -> estimator``.

    ``hgb`` and XGBoost are the same family of model and give near-identical accuracy on
    tables this size; ``hgb`` is the default because it has no extra dependency and
    handles NaN natively. ``ridge`` (standardised, NaN filled with 0) is the linear
    reference: if boosting barely beats it, the relationship is close to additive.
    """
    if callable(model):
        return model(seed)
    if model == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed)
    if model == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("model='xgboost' needs `pip install xgboost`") from exc
        return XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                            colsample_bytree=0.8, random_state=seed, n_jobs=0)
    if model == "ridge":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0),
                             StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    raise ValueError(f"model must be 'hgb', 'xgboost', 'ridge' or a callable, got {model!r}")


def _folds(groups: np.ndarray, n_splits: int):
    from sklearn.model_selection import GroupKFold

    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError("cross-validation needs genes on at least 2 chromosomes")
    return list(GroupKFold(n_splits=min(n_splits, n_groups)).split(groups, groups=groups))


def cross_validated_predictions(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, model: ModelSpec = "hgb",
    n_splits: int = 5, seed: int = 0,
) -> np.ndarray:
    """Out-of-fold predictions with folds grouped by chromosome."""
    pred = np.full(len(y), np.nan)
    for tr, te in _folds(groups, n_splits):
        pred[te] = make_model(model, seed).fit(X[tr], y[tr]).predict(X[te])
    return pred


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1.0 - np.sum((y - p) ** 2) / max(np.sum((y - y.mean()) ** 2), EPS))


def _rank(v: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(v)


def _spearman_boot(y: np.ndarray, p: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Spearman rho on each bootstrap resample (rows of ``idx``)."""
    out = np.empty(len(idx))
    for i, ii in enumerate(idx):
        a, b = _rank(y[ii]), _rank(p[ii])
        a, b = a - a.mean(), b - b.mean()
        out[i] = (a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), EPS)
    return out


# ---------------------------------------------------------------------------
# Representation comparison
# ---------------------------------------------------------------------------


def default_families(columns: Sequence[str]) -> Dict[str, List[str]]:
    """Feature families for the representation comparison, from column names.

    ``top_state`` is the hard-label comparator: one-hot of the state with the most
    occupancy in each part, built by :func:`compare_representations` from the ``occ_``
    columns, i.e. what a discrete segmentation reports.
    """
    cols = list(columns)

    def pick(*pats: str) -> List[str]:
        rx = re.compile("|".join(pats))
        return [c for c in cols if rx.search(c)]

    occ = pick(r"(^|_)occ_")
    fam = {
        "top_state": occ,
        "occupancy": occ,
        "occupancy+intensity": pick(r"(^|_)occ_", r"(^|_)int_"),
        "quantitative": pick(r"(^|_)occ_", r"(^|_)int_", r"(^|_)dose_", r"(^|_)entropy$",
                             r"(^|_)biv_"),
        "signal": pick(r"(^|_)sig_"),
    }
    fam["quantitative+signal"] = fam["quantitative"] + fam["signal"]
    return {k: v for k, v in fam.items() if v}


def _top_state_onehot(X: pd.DataFrame, occ_cols: Sequence[str]) -> np.ndarray:
    """One-hot of the most-occupied state per part prefix (``P_``, ``B_``, ``base_P_`` ...)."""
    blocks = []
    by_prefix: Dict[str, List[str]] = {}
    for c in occ_cols:
        by_prefix.setdefault(c.split("occ_")[0], []).append(c)
    for _, cs in by_prefix.items():
        v = X[cs].to_numpy(dtype=float)
        ok = np.isfinite(v).all(axis=1)
        oh = np.zeros_like(v)
        oh[np.flatnonzero(ok), np.nanargmax(np.where(ok[:, None], v, 0.0), axis=1)[ok]] = 1.0
        blocks.append(oh)
    return np.hstack(blocks)


@dataclass
class RepresentationResult:
    """Held-out accuracy of each feature family, and paired differences to a reference."""

    scores: pd.DataFrame
    contrasts: pd.DataFrame
    predictions: pd.DataFrame


def compare_representations(
    task: ExpressionTask,
    families: Optional[Dict[str, Sequence[str]]] = None,
    reference: str = "quantitative",
    model: ModelSpec = "hgb",
    n_splits: int = 5,
    n_boot: int = 500,
    seed: int = 0,
) -> RepresentationResult:
    """Which representation of chromatin predicts expression best, on held-out chromosomes?

    Every family is scored on the same genes and folds. Confidence intervals are 2.5-97.5%
    bootstrap quantiles over genes of the out-of-fold predictions, and ``contrasts`` are
    paired differences in Spearman rho (``reference`` minus each other family) on the same
    resamples, with ``p_le_0`` the bootstrap fraction where the reference is not better.
    """
    families = dict(families or default_families(task.X.columns))
    y = task.y.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    preds, boots, rows = {}, {}, []
    for name, cols in families.items():
        cols = list(cols)
        if name == "top_state":
            X = _top_state_onehot(task.X, cols)
        else:
            X = task.X[cols].to_numpy(dtype=float)
        p = cross_validated_predictions(X, y, task.groups, model, n_splits, seed)
        preds[name] = p
        boots[name] = _spearman_boot(y, p, boot_idx)
        rows.append({"family": name, "n_features": X.shape[1],
                     "spearman": float(_spearman_boot(y, p, np.arange(len(y))[None])[0]),
                     "rho_lo": float(np.quantile(boots[name], 0.025)),
                     "rho_hi": float(np.quantile(boots[name], 0.975)),
                     "r2": _r2(y, p), "n_genes": len(y)})
    contrasts = []
    if reference in boots:
        for name, b in boots.items():
            if name == reference:
                continue
            d = boots[reference] - b
            contrasts.append({"contrast": f"{reference} - {name}", "delta_rho": float(d.mean()),
                              "lo": float(np.quantile(d, 0.025)),
                              "hi": float(np.quantile(d, 0.975)),
                              "p_le_0": float(np.mean(d <= 0))})
    scores = pd.DataFrame(rows).sort_values("spearman", ascending=False).reset_index(drop=True)
    scores.insert(0, "task", task.description)
    return RepresentationResult(scores, pd.DataFrame(contrasts),
                                pd.DataFrame(preds, index=task.X.index).assign(y=y))


# ---------------------------------------------------------------------------
# State importance
# ---------------------------------------------------------------------------


def state_column_groups(
    columns: Sequence[str], states: Sequence[str],
    quantities: Sequence[str] = STATE_QUANTITIES,
) -> Dict[str, List[str]]:
    """Map each state to its columns (every part, every quantity, baseline included)."""
    out: Dict[str, List[str]] = {}
    for s in states:
        rx = re.compile(r"(^|_)(" + "|".join(quantities) + r")_" + re.escape(s) + r"$")
        cols = [c for c in columns if rx.search(c)]
        if cols:
            out[s] = cols
    return out


def _direction_column(cols: Sequence[str]) -> str:
    """The column whose sign best represents a state: promoter occupancy if present."""
    for pref in ("P_occ_", "G_occ_", "B_occ_"):
        for c in cols:
            if c.startswith(pref):
                return c
    return cols[0]


@dataclass
class StateImportance:
    """Per-state importance for one expression task."""

    table: pd.DataFrame
    full_r2: float
    full_spearman: float
    task: str
    note: str


def state_importance(
    task: ExpressionTask,
    states: Optional[Sequence[str]] = None,
    quantities: Sequence[str] = STATE_QUANTITIES,
    model: ModelSpec = "hgb",
    n_splits: int = 5,
    n_perm: int = 5,
    n_boot: int = 500,
    seed: int = 0,
) -> StateImportance:
    """Rank chromatin states by how much held-out expression variance each explains.

    For each state, using all of its columns (every part, the chosen ``quantities``):

    ``marginal_rho``
        Held-out Spearman rho of a model that sees only that state.
    ``unique_r2`` (with ``unique_lo``/``unique_hi``)
        Fall in held-out R^2 of the full model when the state's columns are permuted
        jointly across genes, averaged over ``n_perm`` permutations. CI from bootstrapping
        genes of the held-out squared errors. What only this state carries.
    ``direction``
        Spearman rho between the state's promoter occupancy (or its change) and the
        target: positive for states associated with higher expression.

    The full model sees every state's columns. Shared columns (entropy, bivalency, raw
    signal) are left out on purpose, so that importance is divided among states and not
    absorbed by the raw marks every state is built from.
    """
    X_df = task.X
    if states is None:
        states = sorted({m.group(2) for c in X_df.columns
                         for m in [re.search(r"(^|_)occ_(.+)$", c)] if m})
    groups_map = state_column_groups(X_df.columns, states, quantities)
    if not groups_map:
        raise ValueError("no state columns found; build the table with gene_quantification")
    all_cols = [c for cols in groups_map.values() for c in cols]
    col_idx = {c: i for i, c in enumerate(all_cols)}
    X = X_df[all_cols].to_numpy(dtype=float)
    y = task.y.to_numpy(dtype=float)
    folds = _folds(task.groups, n_splits)
    rng = np.random.default_rng(seed)

    base_pred = np.full(len(y), np.nan)
    perm_pred = {s: np.zeros(len(y)) for s in groups_map}
    for tr, te in folds:
        m = make_model(model, seed).fit(X[tr], y[tr])
        base_pred[te] = m.predict(X[te])
        for s, cols in groups_map.items():
            ci = [col_idx[c] for c in cols]
            acc = np.zeros(len(te))
            for _ in range(n_perm):
                Xp = X[te].copy()
                Xp[:, ci] = Xp[rng.permutation(len(te))][:, ci]
                acc += m.predict(Xp)
            perm_pred[s][te] = acc / n_perm

    sst = max(np.sum((y - y.mean()) ** 2), EPS)
    base_se = (y - base_pred) ** 2
    full_r2 = _r2(y, base_pred)
    boot_idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    rows = []
    for s, cols in groups_map.items():
        perm_se = (y - perm_pred[s]) ** 2
        unique = float((perm_se.sum() - base_se.sum()) / sst)
        diff = perm_se - base_se
        boot = np.array([diff[ii].sum() / max(np.sum((y[ii] - y[ii].mean()) ** 2), EPS)
                         for ii in boot_idx])
        single = cross_validated_predictions(X_df[cols].to_numpy(dtype=float), y, task.groups,
                                             model, n_splits, seed)
        dcol = _direction_column(cols)
        dv = X_df[dcol].to_numpy(dtype=float)
        ok = np.isfinite(dv)
        rows.append({
            "state": s,
            "marginal_rho": float(_spearman_boot(y, single, np.arange(len(y))[None])[0]),
            "unique_r2": unique,
            "unique_lo": float(np.quantile(boot, 0.025)),
            "unique_hi": float(np.quantile(boot, 0.975)),
            "direction": float(_spearman_boot(y[ok], dv[ok], np.arange(ok.sum())[None])[0]),
            "direction_column": dcol,
            "n_columns": len(cols),
        })
    table = pd.DataFrame(rows).sort_values("unique_r2", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    full_rho = float(_spearman_boot(y, base_pred, np.arange(len(y))[None])[0])
    note = (
        f"{task.description}: {len(y)} genes, folds grouped by chromosome. unique_r2 is the "
        f"held-out R^2 lost when a state's columns are permuted in the full model "
        f"(R^2 {full_r2:.3f}); marginal_rho is a single-state model. Associations, not "
        f"causal effects."
    )
    return StateImportance(table, full_r2, full_rho, task.description, note)


def rank_states_for_expression(
    gene_q: pd.DataFrame,
    expression: Optional[pd.DataFrame] = None,
    lfc: Optional[pd.Series] = None,
    contrast: Optional[Tuple[Sequence[str], Sequence[str]]] = None,
    samples: Optional[Sequence[str]] = None,
    model: ModelSpec = "hgb",
    seed: int = 0,
    **kwargs,
) -> Tuple[StateImportance, RepresentationResult]:
    """One call: state importance and the representation comparison for one task.

    Give ``expression`` (genes x samples, log scale) for the level task, or ``lfc`` plus
    ``contrast=(group_a, group_b)`` for the change task.
    """
    if lfc is not None:
        if contrast is None:
            raise ValueError("the change task needs contrast=(group_a, group_b)")
        task = expression_change_task(gene_q, lfc, contrast[0], contrast[1])
    elif expression is not None:
        task = expression_level_task(gene_q, expression, samples)
    else:
        raise ValueError("give expression= (level) or lfc= with contrast= (change)")
    imp = state_importance(task, model=model, seed=seed, **kwargs)
    rep = compare_representations(task, model=model, seed=seed)
    return imp, rep
