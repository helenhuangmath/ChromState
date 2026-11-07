"""Which chromatin states predict gene expression, and how well: one call, one report.

:func:`predict_expression` takes the per-gene state table from
:func:`chromstate.gene_quantification` (or the pipeline's ``genes/gene_quantification.tsv.gz``)
and an expression target -- a genes x samples matrix (expression *level*) or a per-gene
log2 fold change with the two conditions it compares (expression *change*) -- and reports:

* **held-out predictions** for every gene (folds grouped by chromosome, so no gene is
  predicted from its own neighbourhood), with Spearman rho, R^2 and bootstrap intervals;
* **which state predicts best**: a single-state model per state (``marginal_rho``) and
  the held-out R^2 lost when that state is permuted inside the full model
  (``unique_r2``), with its direction of association;
* **which quantity of a state matters**: occupancy alone, occupancy + intensity, the full
  quantitative set, raw mark signal, and the hard top-state label, scored on the same
  genes and folds with paired bootstrap differences;
* **which gene part matters** (promoter, enhancer space, body) when the table has them.

Everything is an association on held-out genes, not a causal effect.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .regulation import (
    ExpressionTask,
    ModelSpec,
    RepresentationResult,
    StateImportance,
    _spearman_boot,
    compare_representations,
    expression_change_task,
    expression_level_task,
    state_importance,
)


@dataclass
class PredictionResult:
    task: str
    kind: str
    importance: pd.DataFrame
    representations: pd.DataFrame
    representation_contrasts: pd.DataFrame
    predictions: pd.DataFrame
    parts: Optional[pd.DataFrame]
    summary: Dict[str, object]
    note: str
    figures: Dict[str, object] = field(default_factory=dict)

    @property
    def best_single_state(self) -> str:
        return str(self.summary["best_single_state"])

    @property
    def most_unique_state(self) -> str:
        return str(self.summary["most_unique_state"])

    def report_markdown(self) -> str:
        s = self.summary
        lines = [f"# Expression prediction: {self.task}", "", self.note, "",
                 "## Held-out accuracy of the full quantitative model", "",
                 f"- genes: {s['n_genes']:,}", f"- Spearman rho: {s['rho']:.3f} "
                 f"(95% bootstrap {s['rho_lo']:.3f}-{s['rho_hi']:.3f})", f"- R^2: {s['r2']:.3f}", "",
                 "## Which state predicts expression best", "",
                 f"- best single state (its own model): **{s['best_single_state']}** "
                 f"(rho {s['best_single_rho']:.3f})",
                 f"- most irreplaceable state (unique held-out R^2): **{s['most_unique_state']}** "
                 f"({s['most_unique_r2']:.3f}, direction {s['most_unique_direction']:+.2f})", "",
                 _md(self.importance[[c for c in ("rank", "state", "marginal_rho", "unique_r2", "unique_lo",
                                                   "unique_hi", "direction") if c in self.importance.columns]]),
                 "", "## Which representation of chromatin predicts best", "",
                 _md(self.representations[[c for c in ("family", "n_features", "spearman", "rho_lo", "rho_hi", "r2")
                                           if c in self.representations.columns]])]
        if not self.representation_contrasts.empty:
            lines += ["", "Paired bootstrap differences (quantitative minus each family):", "",
                      _md(self.representation_contrasts)]
        if self.parts is not None and not self.parts.empty:
            lines += ["", "## Which gene part predicts best", "",
                      _md(self.parts[[c for c in ("parts", "n_features", "spearman", "rho_lo", "rho_hi")
                                      if c in self.parts.columns]])]
        return "\n".join(lines) + "\n"

    def write(self, outdir: str, seg=None, figures: bool = True, palette: str = "scirep2025") -> List[str]:
        os.makedirs(outdir, exist_ok=True)
        files = []

        def save(frame, name, index=False):
            path = os.path.join(outdir, name)
            frame.to_csv(path, sep="\t", index=index)
            files.append(path)

        save(self.importance, "state_importance.tsv")
        save(self.representations, "representation_scores.tsv")
        save(self.representation_contrasts, "representation_contrasts.tsv")
        save(self.predictions, "gene_predictions.tsv")
        if self.parts is not None:
            save(self.parts, "part_ranking.tsv")
        pd.Series(self.summary).to_frame("value").rename_axis("key").to_csv(
            os.path.join(outdir, "summary.tsv"), sep="\t")
        files.append(os.path.join(outdir, "summary.tsv"))
        with open(os.path.join(outdir, "REPORT.md"), "w") as fh:
            fh.write(self.report_markdown())
        files.append(os.path.join(outdir, "REPORT.md"))
        if figures:
            from . import viz

            figs = {"state_importance": viz.plot_state_importance(self.importance, seg, palette=palette,
                                                                   title=f"Which states predict {self.kind}"),
                    "predicted_vs_observed": plot_predicted_vs_observed(self.predictions, self.summary),
                    "representations": plot_representations(self.representations)}
            if self.parts is not None and not self.parts.empty:
                figs["part_ranking"] = viz.plot_part_ranking(self.parts)
            self.figures = figs
            files += viz.save_all(figs, os.path.join(outdir, "figures"))
        return files


def _md(frame: pd.DataFrame) -> str:
    f = frame.copy()
    for c in f.columns:
        if pd.api.types.is_float_dtype(f[c]):
            f[c] = f[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "")
    head = "| " + " | ".join(map(str, f.columns)) + " |"
    sep = "|" + "---|" * len(f.columns)
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in f.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    ss = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((y - p) ** 2) / max(ss, 1e-12))


def _sample_labels_for_level_task(gene_q: pd.DataFrame, expression: pd.DataFrame,
                                  samples: Sequence[str]) -> np.ndarray:
    out = []
    for s in samples:
        sub = gene_q.loc[gene_q["sample"] == s].set_index("gene_id")
        genes = sub.index.intersection(expression.index[expression[s].notna()])
        out.append(np.repeat(s, len(genes)))
    return np.concatenate(out) if out else np.array([], dtype=object)


def predict_expression(
    gene_q: pd.DataFrame,
    expression: Optional[pd.DataFrame] = None,
    lfc: Optional[pd.Series] = None,
    contrast: Optional[Tuple[Sequence[str], Sequence[str]]] = None,
    samples: Optional[Sequence[str]] = None,
    model: ModelSpec = "hgb",
    n_splits: int = 5,
    n_boot: int = 500,
    n_perm: int = 5,
    parts: bool = True,
    seed: int = 0,
) -> PredictionResult:
    """Predict expression from chromatin states and report which state does it best.

    Parameters
    ----------
    gene_q
        Output of :func:`chromstate.gene_quantification` or
        :func:`chromstate.gene_regulatory_quantification` (one row per gene and sample).
    expression
        Genes x samples on a log scale (index or column ``gene_id``), for the level task.
    lfc, contrast
        Per-gene log2 fold change (``Series`` indexed by ``gene_id``) and
        ``(samples_a, samples_b)`` (b over a), for the change task.
    parts
        Also rank promoter / enhancer / body parts when the table has more than one.
    """
    if expression is not None and "gene_id" in expression.columns:
        expression = expression.set_index("gene_id")
    if lfc is not None:
        if contrast is None:
            raise ValueError("the change task needs contrast=(samples_a, samples_b)")
        task = expression_change_task(gene_q, lfc, contrast[0], contrast[1])
        kind = "expression change"
        sample_labels = None
    elif expression is not None:
        samples = list(samples or [s for s in gene_q["sample"].unique() if s in expression.columns])
        task = expression_level_task(gene_q, expression, samples)
        kind = "expression level"
        sample_labels = _sample_labels_for_level_task(gene_q, expression, samples)
    else:
        raise ValueError("give expression= (level) or lfc= with contrast= (change)")

    imp: StateImportance = state_importance(task, model=model, n_splits=n_splits, n_perm=n_perm,
                                            n_boot=n_boot, seed=seed)
    rep: RepresentationResult = compare_representations(task, model=model, n_splits=n_splits,
                                                        n_boot=n_boot, seed=seed)
    y = task.y.to_numpy(dtype=float)
    pred_col = "quantitative" if "quantitative" in rep.predictions.columns else rep.scores["family"].iloc[0]
    p = rep.predictions[pred_col].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    rho_boot = _spearman_boot(y, p, boot_idx)
    rho = float(_spearman_boot(y, p, np.arange(len(y))[None])[0])

    predictions = pd.DataFrame({"gene_id": task.X.index.to_numpy(), "chrom": task.groups,
                                "observed": y, "predicted": p, "residual": y - p})
    if sample_labels is not None and len(sample_labels) == len(predictions):
        predictions.insert(1, "sample", sample_labels)
    for fam in rep.predictions.columns:
        if fam not in ("y", pred_col):
            predictions[f"predicted_{fam}"] = rep.predictions[fam].to_numpy(dtype=float)

    tab = imp.table
    best_single = tab.loc[tab["marginal_rho"].idxmax()]
    most_unique = tab.loc[tab["unique_r2"].idxmax()]
    summary = {
        "task": task.description, "kind": kind, "n_genes": int(len(y)), "model": str(model),
        "rho": rho, "rho_lo": float(np.quantile(rho_boot, 0.025)), "rho_hi": float(np.quantile(rho_boot, 0.975)),
        "r2": _r2(y, p), "prediction_family": pred_col,
        "best_single_state": str(best_single["state"]), "best_single_rho": float(best_single["marginal_rho"]),
        "most_unique_state": str(most_unique["state"]), "most_unique_r2": float(most_unique["unique_r2"]),
        "most_unique_direction": float(most_unique["direction"]),
    }
    part_tab = None
    if parts:
        prefixes = {c.split("_")[0] for c in gene_q.columns if "_occ_" in c}
        if len(prefixes) >= 2:
            from .regions import rank_parts_for_expression

            try:
                part_tab = rank_parts_for_expression(
                    gene_q, expression=expression if lfc is None else None, lfc=lfc, contrast=contrast,
                    samples=samples, parts=tuple(sorted(prefixes, key=lambda s: "PEB".find(s) if s in "PEB" else 9)),
                    model=model, n_splits=n_splits, seed=seed)
                summary["best_parts"] = str(part_tab.sort_values("spearman", ascending=False)["parts"].iloc[0])
            except Exception as exc:  # pragma: no cover - depends on table layout
                part_tab = None
                summary["parts_note"] = f"part ranking skipped: {exc}"
    note = (f"{task.description}: {len(y):,} genes, {n_splits} folds grouped by chromosome, model {model}. "
            f"Held-out Spearman rho {rho:.3f} (R^2 {summary['r2']:.3f}) for the full quantitative "
            f"representation. Best single state {summary['best_single_state']} (rho "
            f"{summary['best_single_rho']:.3f}); most irreplaceable state {summary['most_unique_state']} "
            f"(unique R^2 {summary['most_unique_r2']:.3f}). Associations on held-out genes, not causal effects.")
    return PredictionResult(task.description, kind, tab, rep.scores, rep.contrasts, predictions, part_tab,
                            summary, note)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_predicted_vs_observed(predictions: pd.DataFrame, summary: Dict[str, object], figsize=None):
    from . import viz

    figsize = figsize or (viz.WIDTH_SINGLE, viz.WIDTH_SINGLE * 0.95)
    fig, ax = viz._new_fig(*figsize)
    x = predictions["observed"].to_numpy(dtype=float)
    y = predictions["predicted"].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() > 2000:
        ax.hexbin(x[ok], y[ok], gridsize=40, mincnt=1, cmap="Blues", linewidths=0.1)
    else:
        ax.scatter(x[ok], y[ok], s=4, color="#395493", alpha=0.5, linewidths=0, zorder=3)
    lo, hi = np.nanmin(np.r_[x[ok], y[ok]]), np.nanmax(np.r_[x[ok], y[ok]])
    ax.plot([lo, hi], [lo, hi], color=viz.INK, linewidth=0.5, linestyle="--")
    ax.set_xlabel(f"observed {summary.get('kind', '')}", fontsize=viz.FS_LABEL)
    ax.set_ylabel("held-out prediction", fontsize=viz.FS_LABEL)
    viz._title(ax, "Predicted versus observed",
               f"rho {summary['rho']:.3f} ({summary['rho_lo']:.3f}-{summary['rho_hi']:.3f}), "
               f"R² {summary['r2']:.3f}, n = {summary['n_genes']:,}")
    fig.tight_layout()
    return fig


def plot_representations(scores: pd.DataFrame, figsize=None):
    from . import viz

    figsize = figsize or (viz.WIDTH_SINGLE, 2.2)
    tab = scores.sort_values("spearman").reset_index(drop=True)
    fig, ax = viz._new_fig(*figsize)
    ax.grid(False)
    ax.grid(True, axis="x", color=viz.GRID, linewidth=0.3, zorder=0)
    yy = np.arange(len(tab))
    v = tab["spearman"].to_numpy(dtype=float)
    lo = v - tab["rho_lo"].to_numpy(dtype=float)
    hi = tab["rho_hi"].to_numpy(dtype=float) - v
    col = ["#AE231B" if f == "quantitative" else ("#9E9E9E" if f in ("top_state", "signal") else "#395493")
           for f in tab["family"]]
    ax.barh(yy, v, color=col, edgecolor=viz.INK, linewidth=0.4, zorder=3)
    ax.errorbar(v, yy, xerr=np.vstack([np.clip(lo, 0, None), np.clip(hi, 0, None)]), fmt="none",
                ecolor=viz.INK, elinewidth=0.5, capsize=1.5, zorder=4)
    ax.set_yticks(yy)
    ax.set_yticklabels(tab["family"], fontsize=viz.FS_TICK)
    ax.set_xlabel("held-out Spearman ρ with expression", fontsize=viz.FS_LABEL)
    viz._title(ax, "Representations of chromatin", "red = ChromState quantitative; grey = hard labels / raw signal")
    fig.tight_layout()
    return fig
