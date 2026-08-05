"""Joining chromatin state to expression, without overclaiming what the join shows.

The mechanics are simple -- both sides are per-gene tables, ``gene_id`` is the key -- and
the interpretation is where this goes wrong in practice. Three rules are built into the
functions here rather than left to the discussion section:

**Association is not causation, and the direction is not identifiable from this data.**
A promoter gaining active mass as its gene goes up is equally consistent with the mark
driving the transcription, the transcription depositing the mark, and a third factor
doing both. The output of :func:`expression_association` is a correlation with a null,
and it is named as one.

**The null has to be a gene null, not a bin null.** Expression change is spatially
autocorrelated (neighbouring genes, shared domains) and so is chromatin change, so
shuffling values across genes independently makes almost any association look
significant. The permutation here shuffles whole genes, which preserves each gene's own
magnitude and only breaks the pairing.

**A test on genes you selected for changing is circular.** If the gene set was chosen
because those promoters moved, the association is guaranteed. :func:`expression_association`
takes all genes by default and reports the number used, and any filtering should happen
before -- and be stated -- rather than inside the test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

EPS = 1e-12


def read_expression(
    path: str,
    gene_col: str = "gene_id",
    sep: str = "\t",
) -> pd.DataFrame:
    """Read a gene x sample expression matrix (counts, TPM, or normalised values)."""
    frame = pd.read_csv(path, sep=sep)
    if gene_col not in frame.columns:
        raise ValueError(f"{path} has no {gene_col!r} column; found {list(frame.columns)[:8]}")
    return frame.rename(columns={gene_col: "gene_id"})


def expression_change(
    expression: pd.DataFrame,
    group_a: Sequence[str],
    group_b: Sequence[str],
    log: bool = True,
    pseudocount: float = 1.0,
) -> pd.DataFrame:
    """Per-gene log2 fold change between two groups of RNA samples.

    Use this only when you do not already have a differential-expression result. If you
    have DESeq2 or edgeR output, pass that instead -- it models the count dispersion,
    which a ratio of means does not, and it comes with the standard errors this does not
    attempt to invent.
    """
    group_a, group_b = list(group_a), list(group_b)
    missing = [s for s in group_a + group_b if s not in expression.columns]
    if missing:
        raise ValueError(f"Expression table has no column(s): {missing}")

    a = expression[group_a].to_numpy(dtype=float).mean(axis=1)
    b = expression[group_b].to_numpy(dtype=float).mean(axis=1)
    if log:
        lfc = np.log2((b + pseudocount) / (a + pseudocount))
    else:
        lfc = b - a
    return pd.DataFrame(
        {
            "gene_id": expression["gene_id"].to_numpy(),
            "mean_a": a,
            "mean_b": b,
            "log2fc": lfc,
        }
    )


def link_expression(
    gene_change: pd.DataFrame,
    rna_change: pd.DataFrame,
    lfc_col: str = "log2fc",
    how: str = "inner",
) -> pd.DataFrame:
    """Join per-gene chromatin change to per-gene expression change on ``gene_id``.

    The join is reported, not assumed: the result carries ``n_chromatin``, ``n_rna``, and
    ``n_joined`` in ``.attrs``, because a silent 30% drop-out from an annotation mismatch
    looks exactly like a weak biological association.
    """
    if "gene_id" not in gene_change.columns or "gene_id" not in rna_change.columns:
        raise ValueError("both tables need a 'gene_id' column")
    if lfc_col not in rna_change.columns:
        raise ValueError(f"rna_change has no {lfc_col!r} column; found {list(rna_change.columns)}")

    merged = gene_change.merge(
        rna_change[["gene_id", lfc_col]].rename(columns={lfc_col: "log2fc"}),
        on="gene_id", how=how,
    )
    merged.attrs["n_chromatin"] = int(len(gene_change))
    merged.attrs["n_rna"] = int(len(rna_change))
    merged.attrs["n_joined"] = int(len(merged))
    if len(merged) < 0.5 * min(len(gene_change), len(rna_change)):
        merged.attrs["warning"] = (
            f"Only {len(merged)} genes joined from {len(gene_change)} chromatin and "
            f"{len(rna_change)} expression rows. Check identifier types (Ensembl versus "
            f"symbol, versioned versus unversioned) before reading anything into the result."
        )
    return merged


@dataclass
class AssociationResult:
    """Per-state association between chromatin change and expression change."""

    table: pd.DataFrame
    n_genes: int
    n_permutations: int
    note: str


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    from scipy.stats import rankdata

    xr = rankdata(x[ok])
    yr = rankdata(y[ok])
    xr = xr - xr.mean()
    yr = yr - yr.mean()
    denom = np.sqrt((xr**2).sum() * (yr**2).sum())
    return float((xr * yr).sum() / denom) if denom > EPS else float("nan")


def expression_association(
    linked: pd.DataFrame,
    state_columns: Optional[Sequence[str]] = None,
    n_permutations: int = 1000,
    seed: int = 0,
    fdr: float = 0.05,
) -> AssociationResult:
    """Does gaining each state predict expression change? Correlation plus a gene null.

    For every ``delta_<state>`` column, the Spearman correlation with ``log2fc`` across
    genes, and a p-value from shuffling gene labels -- which breaks the pairing while
    leaving each side's own distribution intact.

    A positive correlation for an active promoter state and a negative one for a Polycomb
    state is the expected sanity check. If it does not appear, the usual causes are an
    identifier mismatch in the join, the wrong gene part (body states will not track
    expression the way promoter states do), or a contrast where the RNA and the chromatin
    were not measured on the same conditions.
    """
    if "log2fc" not in linked.columns:
        raise ValueError("linked table needs a 'log2fc' column; run link_expression first")
    if state_columns is None:
        state_columns = [c for c in linked.columns if c.startswith("delta_")]
    if not state_columns:
        raise ValueError("no delta_<state> columns found to test")

    lfc = linked["log2fc"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    for col in state_columns:
        x = linked[col].to_numpy(dtype=float)
        rho = _spearman(x, lfc)
        if not np.isfinite(rho):
            rows.append({"state": col.replace("delta_", ""), "spearman": np.nan,
                         "pvalue": np.nan, "n": int(np.isfinite(x).sum())})
            continue
        null = np.empty(n_permutations)
        for i in range(n_permutations):
            null[i] = _spearman(x, lfc[rng.permutation(len(lfc))])
        pvalue = (np.sum(np.abs(null) >= abs(rho)) + 1.0) / (n_permutations + 1.0)
        rows.append(
            {
                "state": col.replace("delta_", ""),
                "spearman": rho,
                "pvalue": float(pvalue),
                "n": int(np.isfinite(x).sum()),
                "null_sd": float(np.std(null)),
            }
        )

    table = pd.DataFrame(rows)
    from .dynamics import benjamini_hochberg

    ok = table["pvalue"].notna()
    table.loc[ok, "qvalue"] = benjamini_hochberg(table.loc[ok, "pvalue"].to_numpy())
    table["significant"] = table["qvalue"] < fdr
    table = table.sort_values("spearman", ascending=False).reset_index(drop=True)

    note = (
        f"Spearman correlation between per-gene change in each state's occupancy and "
        f"change in expression, across {len(linked)} genes, with a null from "
        f"{n_permutations} gene-label shuffles. This is an association: it does not "
        f"establish that the chromatin change caused the expression change, or that the "
        f"causality runs in that direction at all."
    )
    if "warning" in linked.attrs:
        note += " " + str(linked.attrs["warning"])
    return AssociationResult(table, int(len(linked)), n_permutations, note)


def expression_by_state_transition(
    linked: pd.DataFrame,
    min_genes: int = 10,
) -> pd.DataFrame:
    """Expression change grouped by which state each gene gained and lost.

    The categorical companion to the correlation: "genes whose promoters moved from
    Polycomb repressed to Active transcription start site went up by a median of X".
    Groups smaller than ``min_genes`` are dropped rather than reported as a median of
    three genes.
    """
    if not {"top_gain", "top_loss", "log2fc"} <= set(linked.columns):
        raise ValueError("linked table needs 'top_gain', 'top_loss', and 'log2fc'")
    grouped = linked.groupby(["top_loss", "top_gain"])["log2fc"]
    out = grouped.agg(n="size", median_log2fc="median", mean_log2fc="mean").reset_index()
    out = out.loc[out["n"] >= min_genes].sort_values("median_log2fc", ascending=False)
    return out.reset_index(drop=True)


def rank_traceable_genes(
    linked: pd.DataFrame,
    min_tv: float = 0.0,
    min_abs_lfc: float = 0.0,
    n: int = 50,
) -> pd.DataFrame:
    """Genes that changed in both layers, ranked -- the shortlist to trace and validate.

    Ranking is by the product of the two effect sizes, so a gene needs both a chromatin
    move and an expression move to reach the top. Feed the ``gene_id`` column to
    :func:`chromstate.features.trace_gene` to see the bins, marks, and counts behind it.
    """
    sub = linked.loc[
        (linked["tv_distance"] >= min_tv) & (linked["log2fc"].abs() >= min_abs_lfc)
    ].copy()
    sub["joint_score"] = sub["tv_distance"] * sub["log2fc"].abs()
    sub["direction"] = np.where(sub["log2fc"] >= 0, "up", "down")
    cols = [c for c in ["gene_id", "name", "tv_distance", "log2fc", "joint_score",
                        "direction", "top_gain", "top_loss", "n_bins"] if c in sub.columns]
    return sub.sort_values("joint_score", ascending=False).head(n)[cols].reset_index(drop=True)
