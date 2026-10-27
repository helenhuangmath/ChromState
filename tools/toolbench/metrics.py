"""The metrics, as a registry you can select from and extend.

Two families, and mixing them up is the usual way a benchmark misleads:

**Truth-based** metrics need a known answer, so they only exist on simulated data (or
data with an orthogonal ground truth such as reChIP-seq bivalency). They are the ones
that say a tool is *right*.

**Truth-free** metrics run anywhere, including real data. They say a tool is *consistent*,
*confident*, or *cheap* -- never that it is right. Segment length and occupancy entropy
describe the shape of a segmentation; agreement with a reference tool says two methods
concur, which is equally consistent with both being wrong.

Every metric declares what it requires, so a run without truth silently drops the
truth-based columns instead of filling them with something invented. Adding a metric
means adding one entry to :data:`METRICS`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from chromstate.benchmark import metrics as M
from chromstate.model.annotate import canonical_mark

from .core import UNASSIGNED, ToolRun, occupancy


@dataclass
class EvalContext:
    """Everything a metric may look at for one tool's run."""

    run: ToolRun
    ds: object
    truth: Optional[object] = None
    reference: Optional[ToolRun] = None

    # -- convenience --------------------------------------------------------

    @property
    def assigned(self) -> np.ndarray:
        """Mask of bins the tool actually called; metrics score only these."""
        states = self.run.states
        return np.zeros(0, dtype=bool) if states is None else states != UNASSIGNED

    def true_states(self) -> Optional[np.ndarray]:
        return None if self.truth is None else np.asarray(self.truth.states)


@dataclass
class Metric:
    """One metric. ``fn`` returns a dict so related numbers stay together."""

    name: str
    description: str
    columns: Tuple[str, ...]
    requires: Tuple[str, ...]  # any of: truth, posterior, reference, bivalency, activity
    fn: Callable[[EvalContext], Dict[str, float]]
    higher_is_better: Optional[bool] = True

    def applicable(self, ctx: EvalContext) -> bool:
        if not ctx.run.ok:
            return False
        if "truth" in self.requires and ctx.truth is None:
            return False
        if "posterior" in self.requires and ctx.run.posterior is None:
            return False
        if "reference" in self.requires and (ctx.reference is None or not ctx.reference.ok):
            return False
        if "bivalency" in self.requires and ctx.run.bivalency_score is None:
            return False
        if "activity" in self.requires and ctx.run.activity_score is None:
            return False
        return True


# ---------------------------------------------------------------------------
# Truth-based
# ---------------------------------------------------------------------------


def _recovery(ctx: EvalContext) -> Dict[str, float]:
    true = ctx.true_states()
    assert true is not None and ctx.run.states is not None
    keep = ctx.assigned
    scores = M.clustering_scores(
        true[keep], ctx.run.states[keep], int(ctx.truth.n_states), max(ctx.run.n_states, 1)
    )
    return {k: float(v) for k, v in scores.items()}


def _bivalency(ctx: EvalContext) -> Dict[str, float]:
    truth = ctx.truth
    if getattr(truth, "bivalent_state", None) is None:
        return {}
    positive = np.asarray(truth.states) == truth.bivalent_state
    det = M.detection_scores(np.asarray(ctx.run.bivalency_score), positive)
    return {f"biv_{k}": float(v) for k, v in det.items()}


def _magnitude(ctx: EvalContext) -> Dict[str, float]:
    truth = ctx.truth
    canon = [canonical_mark(m) for m in truth.marks]
    j = canon.index("K4me3") if "K4me3" in canon else 0
    true_signal = truth.emission_means[np.asarray(truth.states), j]
    out = M.quantitative_sensitivity(np.asarray(ctx.run.activity_score), true_signal)
    return {"signal_spearman": float(out["spearman"]), "signal_pearson": float(out["pearson"])}


def _bivalency_posthoc(ctx: EvalContext) -> Dict[str, float]:
    """Bivalency AUPRC with one scoring rule for every tool, from its partition alone.

    The ``bivalency`` metric scores whatever ``bivalency_score`` a run carries: in-process
    methods derive it from their own fitted emission means, external tools from the
    post-hoc profile in :mod:`toolbench.scores`. Those are different inputs to the same
    formula, so a gap between an internal and an external row could come from the
    scoring route rather than the segmentation. This metric removes that: every run --
    internal or external -- is re-scored with :func:`toolbench.scores.posthoc_scores` on
    its own states (and posterior, where it has a proper state posterior).
    """
    from .scores import posthoc_scores

    truth = ctx.truth
    if getattr(truth, "bivalent_state", None) is None or ctx.run.n_states <= 1:
        return {}
    post = ctx.run.posterior
    if post is not None and post.shape[1] != ctx.run.n_states:
        post = None
    biv, _ = posthoc_scores(ctx.ds, ctx.run.states, post, ctx.run.n_states)
    positive = np.asarray(truth.states) == truth.bivalent_state
    det = M.detection_scores(biv, positive)
    return {"biv_posthoc_auprc": float(det["auprc"]), "biv_posthoc_auroc": float(det["auroc"])}


def _domains(ctx: EvalContext) -> Dict[str, float]:
    """diHMM's domain-level labels, described -- the simulator has no domain truth.

    Reports how many domain states are used and how long domain segments are, plus the
    adjusted Rand index of domain labels against the *bin-level* truth, which is only a
    description of how much bin-state information the coarser layer carries.
    """
    dom = ctx.run.meta.get("domain_states") if ctx.run.meta else None
    if dom is None:
        return {}
    dom = np.asarray(dom)
    keep = dom != UNASSIGNED
    bins = ctx.ds.bins.reset_index(drop=True)
    bin_size = int(bins["end"].iloc[0] - bins["start"].iloc[0])
    n_seg = 0
    for _, grp in bins.groupby(["sample", "chrom"], sort=False):
        seq = dom[grp.index.to_numpy()]
        seq = seq[seq != UNASSIGNED]
        n_seg += (1 + int((seq[1:] != seq[:-1]).sum())) if seq.size else 0
    out = {"domain_states_used": float(len(np.unique(dom[keep]))),
           "domain_mean_segment_kb": float(keep.sum() * bin_size / max(n_seg, 1) / 1e3)}
    if ctx.truth is not None:
        out["domain_ari_vs_bin_truth"] = float(M.segmentation_agreement(
            np.asarray(ctx.truth.states)[keep], dom[keep])["ari"])
    return out


def _calibration(ctx: EvalContext) -> Dict[str, float]:
    true = ctx.true_states()
    post = ctx.run.posterior
    assert true is not None and post is not None
    mapping, _ = M.match_states(
        true, np.argmax(post, axis=1), int(ctx.truth.n_states), post.shape[1]
    )
    out = M.calibration_scores(post, true, mapping)
    return {k: float(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Truth-free
# ---------------------------------------------------------------------------


def _segmentation_shape(ctx: EvalContext) -> Dict[str, float]:
    """Segment count, mean segment length, and how evenly states are used.

    A segmentation that flips state every bin and one that assigns the whole genome to a
    single state both score terribly on any downstream use, and both can look fine on a
    likelihood. These are the two failure shapes, measured directly.
    """
    ds, states = ctx.ds, ctx.run.states
    assert states is not None
    bins = ds.bins.reset_index(drop=True)
    bin_size = int(bins["end"].iloc[0] - bins["start"].iloc[0])

    n_segments = 0
    for _, grp in bins.groupby(["sample", "chrom"], sort=False):
        seq = states[grp.index.to_numpy()]
        if seq.size == 0:
            continue
        n_segments += 1 + int((seq[1:] != seq[:-1]).sum())

    n_assigned = int((states != UNASSIGNED).sum())
    occ = occupancy(states, max(ctx.run.n_states, 1))
    nz = occ[occ > 0]
    entropy = float(-(nz * np.log(nz)).sum() / np.log(len(occ))) if len(occ) > 1 else 0.0
    return {
        "n_segments": float(n_segments),
        "mean_segment_kb": float(n_assigned * bin_size / max(n_segments, 1) / 1e3),
        "states_used": float((occ > 0).sum()),
        "occupancy_entropy": entropy,
        "largest_state_share": float(occ.max()) if occ.size else float("nan"),
    }


def _coverage(ctx: EvalContext) -> Dict[str, float]:
    return {"bin_coverage": float(ctx.run.coverage)}


def _cost(ctx: EvalContext) -> Dict[str, float]:
    out = {"runtime_s": float(ctx.run.runtime_s)}
    if ctx.run.peak_rss_mb is not None:
        out["peak_rss_mb"] = float(ctx.run.peak_rss_mb)
    return out


def _agreement(ctx: EvalContext) -> Dict[str, float]:
    """Adjusted Rand index against a reference tool, on bins both of them called."""
    ref = ctx.reference
    assert ref is not None and ref.states is not None and ctx.run.states is not None
    keep = (ctx.run.states != UNASSIGNED) & (ref.states != UNASSIGNED)
    if keep.sum() < 2:
        return {"agreement_ari": float("nan")}
    scores = M.segmentation_agreement(ctx.run.states[keep], ref.states[keep])
    return {f"agreement_{k}": float(v) for k, v in scores.items()}


def _confidence_shape(ctx: EvalContext) -> Dict[str, float]:
    """How decisive the posterior is, with no reference to whether it is correct.

    Reported separately from calibration because it is available on real data: a model
    whose posteriors sit at 0.999 everywhere is telling you nothing about which regions
    to trust, whether or not it happens to be right.
    """
    post = ctx.run.posterior
    assert post is not None
    conf = post.max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(post * np.log(np.maximum(post, 1e-12))).sum(axis=1) / np.log(post.shape[1])
    return {
        "mean_max_posterior": float(conf.mean()),
        "frac_above_0.99": float((conf > 0.99).mean()),
        "mean_posterior_entropy": float(np.nanmean(ent)),
    }


METRICS: Dict[str, Metric] = {
    "recovery": Metric(
        "recovery",
        "Agreement with the known state labels: adjusted Rand, adjusted mutual "
        "information, V-measure, and accuracy under the best state matching.",
        ("ari", "ami", "v_measure", "matched_accuracy"),
        ("truth",),
        _recovery,
    ),
    "bivalency": Metric(
        "bivalency",
        "Detection of the truly bivalent bins from each tool's bivalency score "
        "(area under the precision-recall and ROC curves).",
        ("biv_auprc", "biv_auroc", "biv_baseline_auprc"),
        ("truth", "bivalency"),
        _bivalency,
    ),
    "magnitude": Metric(
        "magnitude",
        "Whether the tool's activity score tracks the true amount of the active mark "
        "(Spearman and Pearson correlation) rather than only its presence.",
        ("signal_spearman", "signal_pearson"),
        ("truth", "activity"),
        _magnitude,
    ),
    "bivalency_posthoc": Metric(
        "bivalency_posthoc",
        "Bivalency AUPRC/AUROC with one scoring rule for every tool: each state's "
        "posterior-weighted mean normalised count, then the shared harmonic-mean "
        "formula. Separates segmentation quality from how a tool's score was derived.",
        ("biv_posthoc_auprc", "biv_posthoc_auroc"),
        ("truth",),
        _bivalency_posthoc,
    ),
    "domains": Metric(
        "domains",
        "Descriptive summary of a second, domain-level label where a tool has one "
        "(diHMM): domain states used, mean domain segment length, and agreement of the "
        "domain label with the bin-level truth. Not a correctness score.",
        ("domain_states_used", "domain_mean_segment_kb", "domain_ari_vs_bin_truth"),
        (),
        _domains,
        higher_is_better=None,
    ),
    "calibration": Metric(
        "calibration",
        "Whether reported confidence matches observed accuracy: expected calibration "
        "error, overconfidence, Brier score, and the fraction of saturated calls.",
        ("accuracy", "mean_confidence", "overconfidence", "brier", "ece",
         "mean_entropy", "frac_saturated"),
        ("truth", "posterior"),
        _calibration,
        higher_is_better=None,
    ),
    "segmentation": Metric(
        "segmentation",
        "Shape of the segmentation with no ground truth: number of segments, mean "
        "segment length, how many states are used, and how evenly.",
        ("n_segments", "mean_segment_kb", "states_used", "occupancy_entropy",
         "largest_state_share"),
        (),
        _segmentation_shape,
        higher_is_better=None,
    ),
    "coverage": Metric(
        "coverage",
        "Fraction of the dataset's bins the tool assigned to any state.",
        ("bin_coverage",),
        (),
        _coverage,
    ),
    "confidence": Metric(
        "confidence",
        "How decisive the posterior is, independent of correctness: mean winning "
        "probability, fraction of calls above 0.99, and mean normalised entropy.",
        ("mean_max_posterior", "frac_above_0.99", "mean_posterior_entropy"),
        ("posterior",),
        _confidence_shape,
        higher_is_better=None,
    ),
    "agreement": Metric(
        "agreement",
        "Adjusted Rand index against a reference tool, on the bins both of them called. "
        "Says the two concur, not that either is right.",
        ("agreement_ari", "agreement_ami"),
        ("reference",),
        _agreement,
    ),
    "cost": Metric(
        "cost",
        "Wall-clock runtime, and peak resident memory when psutil is installed.",
        ("runtime_s", "peak_rss_mb"),
        (),
        _cost,
        higher_is_better=False,
    ),
}

#: Everything, in the order they are printed.
ALL_METRICS: List[str] = list(METRICS)

#: The truth-free subset, i.e. what a real-data run can report.
TRUTH_FREE: List[str] = [k for k, m in METRICS.items() if "truth" not in m.requires]


def resolve(names: Optional[Sequence[str]]) -> List[Metric]:
    """Look up metric names, defaulting to all of them."""
    if not names:
        return [METRICS[k] for k in ALL_METRICS]
    unknown = [n for n in names if n not in METRICS]
    if unknown:
        raise KeyError(f"unknown metric(s) {unknown}; available: {', '.join(ALL_METRICS)}")
    return [METRICS[n] for n in names]
