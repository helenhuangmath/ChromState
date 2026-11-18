"""Which chromatin states matter for gene expression, and by how much.

:mod:`chromstate.rna` asks whether *gaining* a state tracks expression change, one state
at a time. This module asks the multivariate question: given a gene's full state
composition, which states carry independent information about its expression, in which
direction, and how much is lost if that state is removed. Three measures are reported for
every state, each with a bootstrap interval, because no single one is adequate:

**Standardised coefficient** (linear model only). The change in expression per standard
deviation of a state's share, holding the other states fixed. It has a sign, so it says
whether a state is associated with higher or lower expression; it is unstable when two
states are nearly collinear, which the composition's unit-sum constraint guarantees for
at least one pair, so ridge regularisation is used and the interval is reported.

**Permutation importance** (any model). Out-of-fold loss in predictive correlation when
one state's column is shuffled across genes. It measures how much the *fitted* model
relies on that state and is model-agnostic, but a state that duplicates another's
information scores low even if both matter.

**Drop-one importance** (any model). Out-of-fold loss when the model is refitted
without the state. On a composition this needs care: the shares sum to one, so a dropped
state is recoverable from the others and naive drop-one is identically zero. The remaining
shares are therefore re-closed (renormalised to sum to one) before refitting, which
removes the dropped state's share and leaves only the relative composition of the rest.
For change tables (``delta_*`` columns, which sum to zero) no closure exists and this
measure is not reported. ``rank`` is by permutation importance, which handles both.

All cross-validation holds out whole chromosomes. Neighbouring genes share domains,
enhancers and copy-number, so a random split leaks the answer across folds and every
state looks predictive.

What this does not show: that a state *causes* expression. The direction of the
association is not identifiable from paired chromatin and RNA measurements, and the
functions here name their outputs as associations.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .model.core import Segmentation

EPS = 1e-12
META_COLUMNS = {"gene_id", "name", "sample", "part", "n_bins", "entropy", "top_state",
                "top_state_fraction", "tv_distance", "top_gain", "top_loss", "log2fc"}


@dataclass
class ImportanceResult:
    """Per-state importance for expression, with the out-of-fold score it is relative to."""

    table: pd.DataFrame
    cv_spearman: float
    cv_r2: float
    model: str
    n_genes: int
    n_states: int
    target: str
    note: str
    fold_scores: pd.DataFrame = field(default_factory=pd.DataFrame)

    def top(self, n: int = 5) -> pd.DataFrame:
        return self.table.head(n)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    from scipy.stats import rankdata

    xr = rankdata(x[ok]) - (ok.sum() + 1) / 2.0
    yr = rankdata(y[ok]) - (ok.sum() + 1) / 2.0
    denom = np.sqrt((xr**2).sum() * (yr**2).sum())
    return float((xr * yr).sum() / denom) if denom > EPS else float("nan")


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    ss = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ((y - p) ** 2).sum() / ss) if ss > EPS else float("nan")


def _make_model(model: str, seed: int):
    if model == "ridge":
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    if model == "gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_leaf_nodes=15,
                                             l2_regularization=1.0, random_state=seed)
    raise ValueError(f"model must be 'ridge' or 'gbm', got {model!r}")


def _folds(groups: Optional[np.ndarray], n: int, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold, KFold

    if groups is None:
        warnings.warn(
            "No gene groups (chromosomes) given; using a random K-fold split. Neighbouring "
            "genes will sit on both sides of a split, so scores are optimistic.",
            RuntimeWarning, stacklevel=3,
        )
        return list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.arange(n)))
    n_groups = len(np.unique(groups))
    k = min(n_splits, n_groups)
    if k < 2:
        raise ValueError("need at least two gene groups (chromosomes) for grouped cross-validation")
    return list(GroupKFold(n_splits=k).split(np.arange(n), groups=groups))


def _is_composition(X: np.ndarray, tol: float = 0.05) -> bool:
    s = X.sum(axis=1)
    return bool(np.all(X >= -1e-9) and np.all(np.abs(s - 1.0) < tol))


def _oof_predictions(X: np.ndarray, y: np.ndarray, folds, model: str, seed: int,
                     permute_col: Optional[int] = None, drop_col: Optional[int] = None,
                     rng: Optional[np.random.Generator] = None, close: bool = False) -> np.ndarray:
    """Out-of-fold predictions, optionally with one column shuffled (test fold only) or dropped.

    With ``close=True`` the columns that remain after a drop are renormalised to sum to
    one, so a compositional table does not leak the dropped column through the others.
    """
    pred = np.empty(len(y))
    cols = [j for j in range(X.shape[1]) if j != drop_col]
    Xw = X[:, cols]
    if drop_col is not None and close:
        Xw = Xw / np.maximum(Xw.sum(axis=1, keepdims=True), EPS)
    for tr, te in folds:
        est = _make_model(model, seed)
        est.fit(Xw[tr], y[tr])
        Xte = Xw[te].copy()
        if permute_col is not None and rng is not None:
            j = cols.index(permute_col)
            Xte[:, j] = rng.permutation(Xte[:, j])
        pred[te] = est.predict(Xte)
    return pred


def _state_columns(frame: pd.DataFrame, states: Optional[Sequence[str]], prefix: str = "") -> List[str]:
    if states is not None:
        cols = [f"{prefix}{s}" if f"{prefix}{s}" in frame.columns else s for s in states]
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise ValueError(f"state column(s) not in table: {missing}")
        return cols
    cols = [c for c in frame.columns if c not in META_COLUMNS and not c.startswith("signal_")
            and pd.api.types.is_numeric_dtype(frame[c])]
    if prefix:
        cols = [c for c in cols if c.startswith(prefix)]
    if not cols:
        raise ValueError("no state columns found; pass states= explicitly")
    return cols


def _bootstrap_ci(fn, n: int, n_boot: int, rng: np.random.Generator, level: float = 0.95) -> Tuple[float, float]:
    vals = np.array([fn(rng.integers(0, n, size=n)) for _ in range(n_boot)], dtype=float)
    lo, hi = np.nanpercentile(vals, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def state_importance(
    composition: pd.DataFrame,
    expression: pd.Series,
    states: Optional[Sequence[str]] = None,
    groups: Optional[pd.Series] = None,
    model: str = "ridge",
    target: str = "expression",
    n_splits: int = 5,
    n_permutations: int = 20,
    n_boot: int = 200,
    seed: int = 0,
    min_genes: int = 50,
) -> ImportanceResult:
    """Rank chromatin states by how much each adds to predicting expression.

    Parameters
    ----------
    composition
        One row per gene with a ``gene_id`` column and one numeric column per state: the
        output of :func:`chromstate.features.gene_state_matrix` for a single sample (or
        a group mean), or of :func:`gene_state_change` (``delta_<state>`` columns) when
        ``expression`` is a fold change. Columns named ``signal_*`` and the metadata
        columns are ignored unless named in ``states``.
    expression
        Per-gene value indexed by ``gene_id``: log-scale expression for a level model, or
        log2 fold change for a change model. Genes absent from either side are dropped and
        the join is reported.
    groups
        Per-gene group label indexed by ``gene_id`` (normally the chromosome, from
        ``FeatureSet.table``). Cross-validation never splits a group.
    model
        ``ridge`` (linear; gives signed coefficients) or ``gbm`` (gradient boosting;
        captures interactions between states, no coefficients).
    """
    comp = composition.drop_duplicates("gene_id").set_index("gene_id")
    cols = _state_columns(comp.reset_index(), states)
    expr = pd.Series(expression).dropna()
    expr.index = expr.index.astype(str)
    shared = comp.index.astype(str).intersection(expr.index)
    n_total = len(comp)
    if len(shared) < min_genes:
        raise ValueError(f"only {len(shared)} genes shared between composition ({n_total}) and "
                         f"expression ({len(expr)}); check identifier types")
    comp = comp.loc[comp.index.astype(str).isin(shared)]
    comp.index = comp.index.astype(str)
    comp = comp.loc[shared]
    X = comp[cols].to_numpy(dtype=float)
    y = expr.loc[shared].to_numpy(dtype=float)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y, genes = X[ok], y[ok], np.asarray(shared)[ok]
    grp = None
    if groups is not None:
        g = pd.Series(groups)
        g.index = g.index.astype(str)
        grp = g.reindex(genes).to_numpy()
        if pd.isna(grp).any():
            keep = ~pd.isna(grp)
            X, y, genes, grp = X[keep], y[keep], genes[keep], grp[keep]
    n = len(y)
    rng = np.random.default_rng(seed)
    folds = _folds(grp, n, n_splits, seed)
    compositional = _is_composition(X)

    # Full model, out of fold.
    pred = _oof_predictions(X, y, folds, model, seed)
    rho_full = _spearman(pred, y)
    r2_full = _r2(y, pred)
    fold_rows = []
    for i, (_tr, te) in enumerate(folds):
        fold_rows.append({"fold": i, "n_test": len(te), "spearman": _spearman(pred[te], y[te]),
                          "r2": _r2(y[te], pred[te])})

    rows = []
    for j, col in enumerate(cols):
        # Marginal association, for reference.
        rho_marg = _spearman(X[:, j], y)

        # Permutation importance: repeat the shuffle, average the loss, bootstrap genes.
        perm_preds = np.stack([_oof_predictions(X, y, folds, model, seed, permute_col=j, rng=rng)
                               for _ in range(n_permutations)])
        perm_loss = np.array([rho_full - _spearman(p, y) for p in perm_preds])
        perm_mean = float(perm_loss.mean())
        def _perm_stat(idx, perm_preds=perm_preds):
            return np.mean([_spearman(pred[idx], y[idx]) - _spearman(p[idx], y[idx])
                            for p in perm_preds])

        perm_lo, perm_hi = _bootstrap_ci(_perm_stat, n, n_boot, rng)

        # Drop-one importance, with closure on compositions; undefined on deltas.
        if compositional:
            drop_pred = _oof_predictions(X, y, folds, model, seed, drop_col=j, close=True)
            drop_loss = rho_full - _spearman(drop_pred, y)
            drop_r2_loss = r2_full - _r2(y, drop_pred)
            def _drop_stat(idx, drop_pred=drop_pred):
                return _spearman(pred[idx], y[idx]) - _spearman(drop_pred[idx], y[idx])

            drop_lo, drop_hi = _bootstrap_ci(_drop_stat, n, n_boot, rng)
        else:
            drop_loss = drop_r2_loss = drop_lo = drop_hi = float("nan")

        row: Dict[str, object] = {
            "state": col,
            "marginal_spearman": rho_marg,
            "perm_importance": perm_mean, "perm_lo": perm_lo, "perm_hi": perm_hi,
            "drop_one_importance": float(drop_loss), "drop_lo": drop_lo, "drop_hi": drop_hi,
            "drop_one_r2": float(drop_r2_loss),
            "mean_share": float(X[:, j].mean()),
        }
        rows.append(row)

    table = pd.DataFrame(rows)

    if model == "ridge":
        coefs = _ridge_coefficients(X, y, seed)
        table["coef_std"] = coefs
        boot = np.stack([_ridge_coefficients(X[idx], y[idx], seed)
                         for idx in (rng.integers(0, n, size=n) for _ in range(n_boot))])
        table["coef_lo"] = np.percentile(boot, 2.5, axis=0)
        table["coef_hi"] = np.percentile(boot, 97.5, axis=0)
        table["direction"] = np.where(table["coef_lo"] > 0, "up",
                                      np.where(table["coef_hi"] < 0, "down", "unclear"))
    else:
        table["direction"] = np.where(table["marginal_spearman"] > 0, "up (marginal)", "down (marginal)")

    table["significant"] = table["perm_lo"] > 0
    table = table.sort_values("perm_importance", ascending=False).reset_index(drop=True)
    table.insert(1, "rank", np.arange(1, len(table) + 1))

    note = (
        f"{model} model of {target} from {len(cols)} state shares across {n} genes, scored "
        f"out of fold on {len(folds)} {'chromosome-grouped' if grp is not None else 'random'} "
        f"splits (Spearman {rho_full:.3f}, R^2 {r2_full:.3f}). Importance is the out-of-fold "
        f"loss in Spearman when a state is shuffled ({n_permutations}x)"
        f"{' or dropped with the remaining shares re-closed and refitted' if compositional else ''}; "
        f"intervals are 95% bootstrap over genes ({n_boot}). These are associations between "
        f"chromatin state and {target}; they do not establish direction or causation."
    )
    if len(shared) < 0.5 * min(n_total, len(expr)):
        note += (f" Only {len(shared)} of {n_total} composition rows joined to {len(expr)} "
                 f"expression values; check identifier types.")
    return ImportanceResult(table=table, cv_spearman=rho_full, cv_r2=r2_full, model=model,
                            n_genes=n, n_states=len(cols), target=target, note=note,
                            fold_scores=pd.DataFrame(fold_rows))


def _ridge_coefficients(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    est = _make_model("ridge", seed).fit(X, y)
    return np.asarray(est[-1].coef_, dtype=float)


def state_importance_change(
    gene_change: pd.DataFrame,
    rna_change: pd.DataFrame,
    lfc_col: str = "log2fc",
    groups: Optional[pd.Series] = None,
    **kwargs,
) -> ImportanceResult:
    """Which states' *changes* carry expression change: :func:`state_importance` on deltas.

    ``gene_change`` is :func:`chromstate.features.gene_state_change` output and
    ``rna_change`` a per-gene table with ``gene_id`` and a log2 fold-change column.
    """
    if "gene_id" not in rna_change.columns or lfc_col not in rna_change.columns:
        raise ValueError(f"rna_change needs 'gene_id' and {lfc_col!r} columns")
    delta_cols = [c for c in gene_change.columns if c.startswith("delta_")]
    if not delta_cols:
        raise ValueError("gene_change has no delta_<state> columns; run gene_state_change first")
    expr = rna_change.set_index("gene_id")[lfc_col]
    res = state_importance(gene_change, expr, states=delta_cols, groups=groups,
                           target="expression change", **kwargs)
    res.table["state"] = res.table["state"].str.replace("^delta_", "", regex=True)
    return res


# ---------------------------------------------------------------------------
# Quantitative versus discrete representations
# ---------------------------------------------------------------------------


def gene_hard_state_fractions(seg: Segmentation, features, sample: str, part: str = "promoter",
                              assign: str = "all", **region_kwargs) -> pd.DataFrame:
    """Fraction of a gene's bins in each Viterbi state (the ChromHMM-style representation)."""
    from .features import map_bins_to_regions

    ds = seg.dataset
    mask = ds.sample_mask(sample)
    bins = ds.bins.loc[mask, ["chrom", "start", "end"]].reset_index(drop=True)
    hard = seg.states[mask]
    pairs = map_bins_to_regions(bins, features.regions(part, **region_kwargs), assign=assign)
    K = seg.n_states
    rows = []
    for gene_id, idx in pairs.groupby("gene_id")["bin_row"]:
        counts = np.bincount(hard[idx.to_numpy()], minlength=K) / len(idx)
        row = {"gene_id": gene_id}
        row.update({lbl: float(v) for lbl, v in zip(seg.labels, counts)})
        row["tss_state"] = seg.labels[int(hard[idx.to_numpy()[len(idx) // 2]])]
        rows.append(row)
    return pd.DataFrame(rows)


def compare_representations(
    seg: Segmentation,
    features,
    expression: pd.Series,
    sample: str,
    part: str = "promoter",
    model: str = "gbm",
    n_splits: int = 5,
    n_boot: int = 500,
    seed: int = 0,
    **region_kwargs,
) -> pd.DataFrame:
    """Does the quantitative state composition predict expression better than discrete labels?

    Four representations of the same genes and bins are scored out of fold:
    ``quantitative`` (mean posterior per state, plus entropy), ``hard_fractions``
    (fraction of bins per Viterbi state), ``tss_state`` (one-hot state at the TSS bin),
    and ``raw_signal`` (mean normalised signal per mark). Differences from
    ``quantitative`` are reported with paired bootstrap intervals over genes.
    """
    from .features import gene_state_matrix

    quant = gene_state_matrix(seg, features, part=part, include_marks=True, **region_kwargs)
    quant = quant.loc[quant["sample"] == sample].set_index("gene_id")
    hard = gene_hard_state_fractions(seg, features, sample, part=part, **region_kwargs).set_index("gene_id")
    expr = pd.Series(expression).dropna()
    expr.index = expr.index.astype(str)
    genes = quant.index.astype(str).intersection(hard.index.astype(str)).intersection(expr.index)
    quant.index, hard.index = quant.index.astype(str), hard.index.astype(str)
    quant, hard = quant.loc[genes], hard.loc[genes]
    y = expr.loc[genes].to_numpy(dtype=float)

    chrom = features.table.set_index("gene_id")["chrom"]
    chrom.index = chrom.index.astype(str)
    grp = chrom.reindex(genes).to_numpy()
    folds = _folds(grp if not pd.isna(grp).any() else None, len(y), n_splits, seed)

    reps: Dict[str, np.ndarray] = {
        "quantitative": quant[list(seg.labels) + ["entropy"]].to_numpy(dtype=float),
        "hard_fractions": hard[list(seg.labels)].to_numpy(dtype=float),
        "tss_state": pd.get_dummies(hard["tss_state"]).reindex(columns=seg.labels, fill_value=0)
                       .to_numpy(dtype=float),
        "raw_signal": np.log1p(quant[[c for c in quant.columns if c.startswith("signal_")]]
                               .to_numpy(dtype=float)),
    }
    preds = {name: _oof_predictions(X, y, folds, model, seed) for name, X in reps.items()}
    rng = np.random.default_rng(seed)
    idxs = [rng.integers(0, len(y), size=len(y)) for _ in range(n_boot)]
    ref = preds["quantitative"]
    rows = []
    for name, p in preds.items():
        rho = _spearman(p, y)
        boot = np.array([_spearman(p[i], y[i]) for i in idxs])
        diff = np.array([_spearman(ref[i], y[i]) - _spearman(p[i], y[i]) for i in idxs])
        rows.append({
            "representation": name, "n_features": reps[name].shape[1],
            "spearman": rho, "spearman_lo": float(np.percentile(boot, 2.5)),
            "spearman_hi": float(np.percentile(boot, 97.5)), "r2": _r2(y, p),
            "quantitative_minus_this": float(np.mean(diff)),
            "diff_lo": float(np.percentile(diff, 2.5)), "diff_hi": float(np.percentile(diff, 97.5)),
        })
    out = pd.DataFrame(rows)
    out.attrs["n_genes"] = int(len(y))
    out.attrs["model"] = model
    out.attrs["note"] = (
        f"{model} model of expression in {sample} from {len(y)} genes ({part}), scored out of "
        f"fold on chromosome-grouped splits. 'quantitative_minus_this' is the paired bootstrap "
        f"difference in Spearman between the posterior composition and each alternative."
    )
    return out
