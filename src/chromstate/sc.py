"""Chromatin states in single cells: pseudobulk fits and state transfer to scATAC + scRNA.

Three routes, in order of how directly the states are measured.

1. **Single-cell histone data** (scCUT&Tag, nano-CT, multimodal CUT&Tag). Aggregate cells
   into clusters or metacells with :func:`pseudobulk` and build a dataset with
   :func:`dataset_from_pseudobulk`; every pseudobulk is a *sample* of the joint model, so
   :func:`chromstate.fit`, :func:`chromstate.quantify`, condition comparison and the
   gene-expression link work unchanged. States mean the same thing in every metacell
   because the emissions are shared.

2. **scATAC + scRNA (multiome) with a matched bulk histone dataset.** Histone marks are
   not measured in the single cells, so a state cannot be *fitted* there. It can be
   *transferred*: :class:`StateTransfer` learns, on the bulk bins, how the bulk state
   posterior relates to proxies that *are* measured in the cells -- accessibility in and
   around the bin, and the expression of the nearest gene -- and then scores every
   metacell bin with the same states. The result is a :class:`Segmentation` over the
   bulk vocabulary, carrying per-metacell state scores and a model in proxy space, so
   the summary, comparison and gene functions apply. Held-out accuracy of the transfer
   (chromosome-grouped folds on the bulk) is reported, and it is the ceiling on what the
   single-cell scores can mean: states that accessibility and expression cannot tell
   apart in bulk (Polycomb versus heterochromatin, say) are not told apart in the cells.

3. **scATAC + scRNA without bulk histone data.** Fit the negative-binomial HMM directly on
   pseudobulk accessibility plus gene-activity or expression tracks (each as a "mark").
   These are *accessibility states*, not histone states; the vocabulary is learned, not
   transferred, and should be named accordingly.

Bulk and single-cell data must be on the same bins (same genome, same bin width).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .dataset import ChromStateDataset, _chains_from_runs
from .model.core import Segmentation

# ---------------------------------------------------------------------------
# Pseudobulk
# ---------------------------------------------------------------------------


def metacells(embedding: np.ndarray, n_metacells: int, labels: Optional[Sequence] = None,
              seed: int = 0) -> np.ndarray:
    """Group cells into ``n_metacells`` metacells by k-means on an embedding (LSI, PCA).

    With ``labels`` (cell types or conditions) the metacells are formed *within* each
    label, in proportion to its cell count, so no metacell mixes labels. Returns a
    string label per cell, ``<label>_mc<i>``.
    """
    from sklearn.cluster import KMeans

    embedding = np.asarray(embedding, dtype=float)
    n = embedding.shape[0]
    labels = np.asarray(["all"] * n) if labels is None else np.asarray(labels).astype(str)
    out = np.empty(n, dtype=object)
    uniq, counts = np.unique(labels, return_counts=True)
    alloc = np.maximum(1, np.round(n_metacells * counts / n).astype(int))
    rng = np.random.default_rng(seed)
    for lab, k in zip(uniq, alloc):
        idx = np.flatnonzero(labels == lab)
        k = int(min(k, len(idx)))
        if k <= 1:
            out[idx] = f"{lab}_mc0"
            continue
        km = KMeans(n_clusters=k, n_init=4, random_state=int(rng.integers(0, 2**31 - 1)))
        assign = km.fit_predict(embedding[idx])
        out[idx] = [f"{lab}_mc{a}" for a in assign]
    return out.astype(str)


def pseudobulk(matrix, cell_groups: Sequence, min_cells: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
    """Sum a cells x features count matrix (dense or scipy sparse) within cell groups.

    Returns ``(counts, n_cells)``: ``counts`` is groups x features, ``n_cells`` the number
    of cells per group. Groups with fewer than ``min_cells`` cells are dropped.
    """
    cell_groups = np.asarray(cell_groups).astype(str)
    uniq, inv = np.unique(cell_groups, return_inverse=True)
    n_groups = len(uniq)
    try:
        import scipy.sparse as sp

        is_sparse = sp.issparse(matrix)
    except ImportError:  # pragma: no cover
        is_sparse = False
    inv = np.asarray(inv).ravel()
    if is_sparse:
        import scipy.sparse as sp

        ind = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(n_groups, len(inv)))
        prod = ind @ matrix
        agg = np.asarray(prod.todense() if sp.issparse(prod) else prod)
    else:
        matrix = np.asarray(matrix, dtype=np.float64)
        agg = np.vstack([matrix[inv == g].sum(axis=0) for g in range(n_groups)])
    n_cells = pd.Series(np.bincount(inv, minlength=n_groups), index=uniq)
    keep = n_cells >= min_cells
    counts = pd.DataFrame(agg, index=uniq)
    return counts.loc[keep], n_cells.loc[keep]


def dataset_from_pseudobulk(
    counts: Dict[str, pd.DataFrame],
    bins: pd.DataFrame,
    sample_meta: Optional[pd.DataFrame] = None,
) -> ChromStateDataset:
    """Build a joint dataset from pseudobulk counts.

    ``counts`` maps a mark (e.g. ``"H3K27me3"``, or ``"ATAC"``) to a groups x bins table
    whose rows are the pseudobulk samples (the same rows in every mark) and whose
    columns align with ``bins`` (``chrom``, ``start``, ``end``). The samples are stacked
    as in any ChromState dataset; ``sample_meta`` (indexed by sample) is kept in ``meta``.
    """
    marks = list(counts)
    first = counts[marks[0]]
    samples = list(first.index)
    for m in marks[1:]:
        if list(counts[m].index) != samples:
            raise ValueError(f"mark {m!r} has different pseudobulk samples than {marks[0]!r}")
        if counts[m].shape[1] != len(bins):
            raise ValueError(f"mark {m!r} has {counts[m].shape[1]} bins, bins table has {len(bins)}")
    bins = bins.reset_index(drop=True)
    order = np.lexsort((bins["start"].to_numpy(), bins["chrom"].to_numpy().astype(str)))
    bins = bins.iloc[order].reset_index(drop=True)
    width = int(np.median(bins["end"] - bins["start"]))
    frames, blocks = [], []
    for s in samples:
        b = bins[["chrom", "start", "end"]].copy()
        b["sample"] = s
        frames.append(b)
        blocks.append(np.stack([counts[m].loc[s].to_numpy(dtype=np.float64)[order] for m in marks], axis=1))
    allbins = pd.concat(frames, ignore_index=True)
    X = np.vstack(blocks)
    ds = ChromStateDataset(X=X, marks=marks, bins=allbins, chains=_chains_from_runs(allbins, width),
                           size_factors=np.ones_like(X), is_count=True)
    ds.meta["pseudobulk"] = True
    if sample_meta is not None:
        ds.meta["sample_meta"] = sample_meta.loc[[s for s in samples if s in sample_meta.index]]
    return ds


# ---------------------------------------------------------------------------
# State transfer: bulk histone states -> single-cell proxies
# ---------------------------------------------------------------------------


def _nearest_gene(bins: pd.DataFrame, features) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest TSS gene_id and signed distance (bp) for each bin of one sample."""
    tss = features.tss().to_numpy()
    tab = features.table
    gene_id = np.empty(len(bins), dtype=object)
    dist = np.full(len(bins), np.nan)
    mid = ((bins["start"] + bins["end"]) // 2).to_numpy()
    for chrom, grp in bins.groupby("chrom", sort=False):
        g = np.flatnonzero(tab["chrom"].to_numpy() == chrom)
        if len(g) == 0:
            continue
        order = np.argsort(tss[g])
        pos = tss[g][order]
        ids = tab["gene_id"].to_numpy()[g][order]
        m = mid[grp.index.to_numpy()]
        j = np.searchsorted(pos, m)
        left = np.clip(j - 1, 0, len(pos) - 1)
        right = np.clip(j, 0, len(pos) - 1)
        pick = np.where(np.abs(pos[left] - m) <= np.abs(pos[right] - m), left, right)
        gene_id[grp.index.to_numpy()] = ids[pick]
        dist[grp.index.to_numpy()] = m - pos[pick]
    return gene_id, dist


def proxy_features(
    ds: ChromStateDataset,
    expression: Optional[pd.DataFrame] = None,
    features=None,
    neighbourhood: int = 2,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Per-bin transfer features from a proxy dataset (accessibility, expression).

    For each mark of ``ds``: ``log1p`` of the depth-normalised count in the bin and the
    mean over the ``neighbourhood`` bins on each side (within the chain). With
    ``expression`` (genes x samples, log scale) and ``features`` (gene annotation): the
    nearest gene's expression in the bin's sample and ``log10`` distance to its TSS.
    Returns ``(X, names, chrom_labels)``.
    """
    Z = np.log1p(ds.X / np.maximum(ds.size_factors, 1e-12))
    cols, names = [], []
    for j, m in enumerate(ds.marks):
        cols.append(Z[:, j])
        names.append(f"{m}_bin")
        if neighbourhood > 0:
            nb = np.zeros(ds.n_rows)
            for sl in ds.chains:
                z = Z[sl, j]
                c = np.cumsum(np.r_[0.0, z])
                n = len(z)
                i = np.arange(n)
                lo = np.maximum(i - neighbourhood, 0)
                hi = np.minimum(i + neighbourhood + 1, n)
                nb[sl] = (c[hi] - c[lo]) / (hi - lo)
            cols.append(nb)
            names.append(f"{m}_nbhd{neighbourhood}")
    if expression is not None and features is not None:
        if "gene_id" in expression.columns:
            expression = expression.set_index("gene_id")
        expr_col = np.full(ds.n_rows, np.nan)
        dist_col = np.full(ds.n_rows, np.nan)
        for s in ds.samples:
            m = ds.sample_mask(s)
            bins = ds.bins.loc[m].reset_index(drop=True)
            gid, dist = _nearest_gene(bins, features)
            if s in expression.columns:
                e = expression[s]
                vals = pd.Series(gid).map(e).to_numpy(dtype=float)
            else:
                vals = np.full(len(bins), np.nan)
            expr_col[m] = vals
            dist_col[m] = np.log10(np.abs(dist) + 1.0)
        cols += [expr_col, dist_col]
        names += ["nearest_gene_expr", "log10_tss_distance"]
    X = np.column_stack(cols)
    return X, names, ds.bins["chrom"].to_numpy().astype(str)


@dataclass
class StateTransfer:
    """Learn bulk state posteriors from proxies; score them in single cells.

    Parameters
    ----------
    model
        ``"hgb"`` (histogram gradient boosting classifier, default) or a callable
        ``seed -> classifier`` with ``fit(X, y, sample_weight)`` and ``predict_proba``.
    neighbourhood
        Bins on each side pooled into the neighbourhood feature.
    """

    model: object = "hgb"
    neighbourhood: int = 2
    seed: int = 0
    labels: List[str] = field(default_factory=list)
    colors: List[str] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)
    classes_: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    estimator_: object = None
    cv_report: Optional[pd.DataFrame] = None
    note: str = ""

    def _make(self):
        if callable(self.model):
            return self.model(self.seed)
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=self.seed)

    def fit(self, bulk_seg: Segmentation, proxy: ChromStateDataset, expression: Optional[pd.DataFrame] = None,
            features=None, n_splits: int = 5, max_rows: int = 300_000) -> "StateTransfer":
        """``proxy`` must have the same bins and samples (rows) as ``bulk_seg.dataset``."""
        if proxy.n_rows != bulk_seg.dataset.n_rows:
            raise ValueError("proxy dataset must have exactly the rows (samples x bins) of the bulk dataset")
        if not np.array_equal(proxy.bins["start"].to_numpy(), bulk_seg.dataset.bins["start"].to_numpy()):
            raise ValueError("proxy bins do not line up with the bulk bins")
        X, names, chrom = proxy_features(proxy, expression, features, self.neighbourhood)
        post = bulk_seg.posterior
        y = post.argmax(axis=1)
        w = post.max(axis=1)
        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(y))
        if len(idx) > max_rows:
            idx = rng.choice(idx, size=max_rows, replace=False)
        self.labels = list(bulk_seg.labels)
        self.colors = list(bulk_seg.colors)
        self.feature_names = names

        # Chromosome-grouped CV: how much of the bulk state is recoverable from proxies.
        uniq = np.unique(chrom[idx])
        folds = np.array_split(rng.permutation(uniq), min(n_splits, len(uniq)))
        pred = np.full(len(idx), -1)
        for held in folds:
            te = np.isin(chrom[idx], held)
            if te.all() or not te.any():
                continue
            est = self._make().fit(X[idx][~te], y[idx][~te], sample_weight=w[idx][~te])
            pred[te] = est.classes_[est.predict_proba(X[idx][te]).argmax(axis=1)]
        ok = pred >= 0
        rows = []
        for k, lbl in enumerate(self.labels):
            truth = y[idx][ok] == k
            called = pred[ok] == k
            tp = float((truth & called).sum())
            rows.append({"state": lbl, "n_bulk_bins": int(truth.sum()),
                         "recall": tp / max(truth.sum(), 1), "precision": tp / max(called.sum(), 1)})
        self.cv_report = pd.DataFrame(rows)
        acc = float((pred[ok] == y[idx][ok]).mean()) if ok.any() else float("nan")
        from sklearn.metrics import adjusted_rand_score

        ari = float(adjusted_rand_score(y[idx][ok], pred[ok])) if ok.any() else float("nan")
        self.cv_report.attrs.update({"accuracy": acc, "ari": ari})
        self.estimator_ = self._make().fit(X[idx], y[idx], sample_weight=w[idx])
        self.classes_ = np.asarray(self.estimator_.classes_)
        self.note = (f"State transfer trained on {len(idx):,} bulk bins with features {names}. "
                     f"Chromosome-held-out accuracy of the bulk top state from proxies: {acc:.3f} "
                     f"(ARI {ari:.3f}). This is the ceiling for what the single-cell scores resolve.")
        return self

    def predict_posterior(self, proxy: ChromStateDataset, expression: Optional[pd.DataFrame] = None,
                          features=None) -> np.ndarray:
        X, names, _ = proxy_features(proxy, expression, features, self.neighbourhood)
        if names != self.feature_names:
            raise ValueError(f"feature mismatch: trained on {self.feature_names}, got {names}")
        proba = self.estimator_.predict_proba(X)
        post = np.zeros((proxy.n_rows, len(self.labels)))
        post[:, self.classes_] = proba
        return post

    def project(self, proxy: ChromStateDataset, expression: Optional[pd.DataFrame] = None,
                features=None) -> Segmentation:
        """Score a single-cell (pseudobulk) proxy dataset with the bulk states.

        The returned :class:`Segmentation` has the bulk labels, the transferred posterior
        as its state scores, and a negative-binomial model in *proxy* space (per-state
        posterior-weighted proxy means), so the summary, comparison and gene functions
        run on it. ``meta['transferred']`` records that these are transferred scores.
        """
        from .model.emissions import NegativeBinomialEmissions, _pooled_nb_dispersion
        from .model.hmm import HMM

        post = self.predict_posterior(proxy, expression, features)
        states = post.argmax(axis=1)
        K = len(self.labels)
        scaled = proxy.X / np.maximum(proxy.size_factors, 1e-12)
        w = post.sum(axis=0)
        mu = (post.T @ scaled) / np.maximum(w[:, None], 1e-9)
        mu = np.maximum(mu, 1e-6)
        var = (post.T @ (scaled ** 2)) / np.maximum(w[:, None], 1e-9) - mu ** 2
        disp = _pooled_nb_dispersion(mu, np.maximum(var, 1e-9), w)
        em = NegativeBinomialEmissions(mu=mu, dispersion=disp)
        A = np.full((K, K), 0.1 / max(K - 1, 1))
        np.fill_diagonal(A, 0.9)
        hmm = HMM(K, em, transitions=A)
        rules = ["transferred from bulk" for _ in self.labels]
        return Segmentation(posterior=post, states=states, labels=list(self.labels), colors=list(self.colors),
                            rules=rules, model=hmm, dataset=proxy, loglik=float("nan"),
                            meta={"transferred": True, "transfer_note": self.note,
                                  "transfer_cv": None if self.cv_report is None else self.cv_report.attrs})


def transfer_states(bulk_seg: Segmentation, bulk_proxy: ChromStateDataset, sc_proxy: ChromStateDataset,
                    bulk_expression: Optional[pd.DataFrame] = None, sc_expression: Optional[pd.DataFrame] = None,
                    features=None, **kwargs) -> Tuple[Segmentation, StateTransfer]:
    """One call: fit the transfer on bulk, project onto the single-cell pseudobulks."""
    tr = StateTransfer(**kwargs).fit(bulk_seg, bulk_proxy, bulk_expression, features)
    return tr.project(sc_proxy, sc_expression, features), tr


# ---------------------------------------------------------------------------
# Simulation support
# ---------------------------------------------------------------------------


def simulate_proxy(truth, ds: ChromStateDataset, state_means: Dict[str, float], mark: str = "ATAC",
                   dispersion: float = 4.0, seed: int = 0) -> ChromStateDataset:
    """A proxy (accessibility-like) dataset on the bins of a simulated dataset.

    ``state_means[label]`` is the expected proxy count per bin in that true state. Used
    to test the transfer: the proxy carries partial information about the state.
    """
    rng = np.random.default_rng(seed)
    means = np.array([state_means.get(lbl, 1.0) for lbl in truth.labels])
    mu = means[np.asarray(truth.states)]
    lam = rng.gamma(shape=dispersion, scale=mu / dispersion)
    x = rng.poisson(lam).astype(np.float64)[:, None]
    return ChromStateDataset(X=x, marks=[mark], bins=ds.bins.copy(), chains=list(ds.chains),
                             size_factors=np.ones_like(x), is_count=True, binset=ds.binset)
