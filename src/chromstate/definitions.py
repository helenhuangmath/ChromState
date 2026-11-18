"""States defined from the marks you give, not only discovered from the data.

:func:`chromstate.fit` learns states without supervision and names them afterwards. That
is right when the mark panel is unfamiliar, and wrong when the biologist already knows
what they want to measure: *how much* active promoter, bivalent and Polycomb chromatin
each region carries, with those three words meaning the same thing in every dataset.

A :class:`StateDefinitions` table fixes the vocabulary. Each state is a pattern over the
marks -- ``+`` (high), ``-`` (low), or ``.`` (free) -- and the fit is then either

``anchor="seed"``
    the pattern seeds the emission means and the names, and EM is free to move them
    (the data win; the definitions only pick the basin and the labels), or
``anchor="fixed"``
    the ``+`` and ``-`` entries stay at their seeded level through every M-step and
    only the free entries, the dispersions and the transitions are learned (the
    definitions win; what is measured is exactly the stated pattern).

Either way the posterior for every bin is a quantity over *named* states, so the same
``TssBiv`` column can be compared between experiments that were fitted separately.

Extra, undefined states are allowed (``n_states`` larger than the table): they are
seeded from the data as usual and named automatically, which is how to say "measure these
four things and tell me what else is there".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .model.annotate import canonical_mark
from .model.emissions import NegativeBinomialEmissions

HIGH, LOW, FREE = 1, -1, 0
_SYMBOLS = {"+": HIGH, "high": HIGH, "1": HIGH, "-": LOW, "low": LOW, "-1": LOW,
            ".": FREE, "any": FREE, "0": FREE, "": FREE, "*": FREE}

#: Roadmap-style default patterns for the common histone panel. Marks not in the data
#: are ignored, states that end up with no informative mark are dropped.
DEFAULT_PATTERNS: Dict[str, Dict[str, str]] = {
    "TssA":   {"H3K4me3": "+", "H3K27ac": "+", "H3K27me3": "-", "H3K9me3": "-", "H3K4me1": "."},
    "TssBiv": {"H3K4me3": "+", "H3K27me3": "+", "H3K27ac": "-", "H3K9me3": "-"},
    "EnhA":   {"H3K27ac": "+", "H3K4me1": "+", "H3K4me3": "-", "H3K27me3": "-", "H3K9me3": "-"},
    "Tx":     {"H3K36me3": "+", "H3K4me3": "-", "H3K27me3": "-", "H3K9me3": "-"},
    "ReprPC": {"H3K27me3": "+", "H3K4me3": "-", "H3K27ac": "-", "H3K9me3": "-"},
    "Het":    {"H3K9me3": "+", "H3K4me3": "-", "H3K27ac": "-", "H3K27me3": "-"},
    "Quies":  {},
}


@dataclass
class StateDefinitions:
    """A states x marks pattern table with entries ``+1`` (high), ``-1`` (low), ``0`` (free)."""

    pattern: pd.DataFrame
    colors: Dict[str, str] = field(default_factory=dict)

    # -- constructors ---------------------------------------------------------

    @classmethod
    def from_dict(cls, patterns: Mapping[str, Mapping[str, Union[str, int]]],
                  marks: Optional[Sequence[str]] = None) -> "StateDefinitions":
        """``{"TssA": {"H3K4me3": "+", "H3K27me3": "-"}, ...}``; unnamed marks are free."""
        marks = list(marks) if marks is not None else sorted({m for p in patterns.values() for m in p})
        rows = {}
        for state, pat in patterns.items():
            row = {}
            lookup = {canonical_mark(m) or m: v for m, v in pat.items()}
            for m in marks:
                key = canonical_mark(m) or m
                v = lookup.get(key, lookup.get(m, FREE))
                row[m] = _code(v)
            rows[state] = row
        frame = pd.DataFrame(rows, index=marks).T.astype(int)
        return cls(frame)

    @classmethod
    def from_file(cls, path: str, marks: Optional[Sequence[str]] = None) -> "StateDefinitions":
        """TSV with a ``state`` column and one column per mark holding ``+``, ``-`` or ``.``.

        A plain-text form is accepted too, one state per line::

            TssA:   H3K4me3+ H3K27ac+ H3K27me3-
            TssBiv: H3K4me3+ H3K27me3+
            Quies:
        """
        with open(path) as fh:
            text = fh.read()
        if "\t" in text.splitlines()[0]:
            tab = pd.read_csv(path, sep="\t", dtype=str).fillna(".")
            col = "state" if "state" in tab.columns else tab.columns[0]
            pats = {r[col]: {m: r[m] for m in tab.columns if m != col} for _, r in tab.iterrows()}
        else:
            pats = parse_patterns(text)
        return cls.from_dict(pats, marks)

    @classmethod
    def default(cls, marks: Sequence[str]) -> "StateDefinitions":
        """The Roadmap-style vocabulary restricted to the marks present (see DEFAULT_PATTERNS)."""
        canon = {canonical_mark(m) or m: m for m in marks}
        pats = {}
        for state, pat in DEFAULT_PATTERNS.items():
            row = {canon[canonical_mark(k)]: v for k, v in pat.items() if canonical_mark(k) in canon}
            informative = any(_code(v) == HIGH for v in row.values())
            if informative or state == "Quies":
                pats[state] = row
        defs = cls.from_dict(pats, marks)
        # Quiescent = every mark low: the explicit background pattern.
        if "Quies" in defs.pattern.index:
            defs.pattern.loc["Quies", :] = LOW
        return defs

    # -- views ------------------------------------------------------------------

    @property
    def states(self) -> List[str]:
        return list(self.pattern.index)

    @property
    def marks(self) -> List[str]:
        return list(self.pattern.columns)

    def __len__(self) -> int:
        return len(self.pattern)

    def restrict(self, marks: Sequence[str]) -> "StateDefinitions":
        """The same definitions over the marks of a dataset (missing marks become free)."""
        marks = list(marks)
        frame = pd.DataFrame(FREE, index=self.pattern.index, columns=marks, dtype=int)
        canon_data = {canonical_mark(m) or m: m for m in marks}
        for m in self.pattern.columns:
            key = canonical_mark(m) or m
            target = m if m in marks else canon_data.get(key)
            if target is not None:
                frame[target] = self.pattern[m].to_numpy()
        return StateDefinitions(frame, dict(self.colors))

    def describe(self, state: str) -> str:
        parts = []
        for m, v in self.pattern.loc[state].items():
            if v == HIGH:
                parts.append(f"{m}+")
            elif v == LOW:
                parts.append(f"{m}-")
        return " ".join(parts) if parts else "all free"

    def to_frame(self) -> pd.DataFrame:
        sym = {HIGH: "+", LOW: "-", FREE: "."}
        return self.pattern.replace(sym).rename_axis("state").reset_index()


def _code(v) -> int:
    if isinstance(v, (int, np.integer)):
        return int(np.sign(v))
    s = str(v).strip().lower()
    if s not in _SYMBOLS:
        raise ValueError(f"pattern entry {v!r} not understood; use +, -, or .")
    return _SYMBOLS[s]


def parse_patterns(text: str) -> Dict[str, Dict[str, str]]:
    """Parse ``State: MarkA+ MarkB- ...`` lines."""
    out: Dict[str, Dict[str, str]] = {}
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"expected 'State: mark+ mark- ...', got {line!r}")
        state, rest = line.split(":", 1)
        pat = {}
        for tok in rest.split():
            m = re.match(r"^(.+?)([+\-.])$", tok)
            if not m:
                raise ValueError(f"token {tok!r} in state {state.strip()!r} needs a trailing +, - or .")
            pat[m.group(1)] = m.group(2)
        out[state.strip()] = pat
    return out


# ---------------------------------------------------------------------------
# Seeding and anchoring emissions
# ---------------------------------------------------------------------------


def mark_levels(X: np.ndarray, offsets: np.ndarray, high_q: float = 0.95, low_q: float = 0.50
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Per mark, the expected normalised count of a 'high' and of a 'low' bin.

    High = mean of the bins above the ``high_q`` quantile (the enriched tail), low =
    mean of the bins below the ``low_q`` quantile (background). Both are computed on
    depth-corrected counts so the seed does not depend on which sample is deepest.
    """
    scaled = X / np.maximum(offsets, 1e-12)
    high = np.zeros(X.shape[1])
    low = np.zeros(X.shape[1])
    for m in range(X.shape[1]):
        v = scaled[:, m]
        hq = np.quantile(v, high_q)
        lq = np.quantile(v, low_q)
        hv = v[v >= hq]
        lv = v[v <= lq]
        high[m] = float(hv.mean()) if hv.size else float(v.mean())
        low[m] = float(lv.mean()) if lv.size else float(v.mean())
        if high[m] <= low[m]:
            high[m] = low[m] * 4.0 + 1.0
    return np.maximum(high, 1e-3), np.maximum(low, 1e-3)


def seed_from_definitions(
    X: np.ndarray,
    offsets: np.ndarray,
    defs: StateDefinitions,
    n_states: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    jitter: float = 0.0,
):
    """Negative-binomial emissions seeded from a pattern table.

    Defined entries take the mark's high or low level (:func:`mark_levels`); free
    entries take the mark's global mean. States beyond ``len(defs)`` are seeded by
    k-means on the residual bins, exactly as an unsupervised fit would seed them.
    Returns ``(emissions, anchor_mask)`` where ``anchor_mask`` marks the defined entries.
    """
    from .model.emissions import MIN_MEAN, _cluster_moments, _pooled_nb_dispersion, seed_clusters

    rng = rng if rng is not None else np.random.default_rng(0)
    n_def = len(defs)
    n_states = int(n_states or n_def)
    if n_states < n_def:
        raise ValueError(f"n_states={n_states} is smaller than the {n_def} defined states")
    P = defs.pattern.to_numpy(dtype=int)
    high, low = mark_levels(X, offsets)
    scaled = X / np.maximum(offsets, 1e-12)
    gmean = np.maximum(scaled.mean(axis=0), 1e-3)
    mu = np.empty((n_states, X.shape[1]))
    anchor = np.zeros((n_states, X.shape[1]), dtype=bool)
    for k in range(n_def):
        mu[k] = np.where(P[k] == HIGH, high, np.where(P[k] == LOW, low, gmean))
        anchor[k] = P[k] != FREE

    # Assign bins to their nearest defined profile (log scale) to get within-state moments.
    Z = np.log1p(scaled)
    d = ((Z[:, None, :] - np.log1p(mu[None, :n_def, :])) ** 2).sum(axis=2)
    labels = d.argmin(axis=1)
    if n_states > n_def:
        # The residual: bins far from every definition seed the free states.
        resid = d.min(axis=1)
        cut = np.quantile(resid, 0.5)
        far = np.flatnonzero(resid > cut)
        extra = n_states - n_def
        if len(far) >= extra * 5:
            sub = seed_clusters(Z[far], extra, rng)
            labels = labels.copy()
            labels[far] = n_def + sub
        else:  # tiny data: random split
            labels[far] = n_def + rng.integers(0, extra, size=len(far))
        for k in range(n_def, n_states):
            sel = labels == k
            mu[k] = scaled[sel].mean(axis=0) if sel.sum() >= 2 else gmean
    _, var, sizes = _cluster_moments(scaled, labels, n_states)
    dispersion = _pooled_nb_dispersion(mu, var, sizes)
    if jitter > 0:
        mu = mu * rng.uniform(1.0 - jitter, 1.0 + jitter, size=mu.shape)
    mu = np.maximum(mu, MIN_MEAN)
    em = AnchoredNegativeBinomial(mu=mu, dispersion=dispersion, anchor=anchor)
    return em, anchor


class AnchoredNegativeBinomial(NegativeBinomialEmissions):
    """Negative binomial whose anchored means are pinned after every M-step.

    ``anchor`` is a ``(K, M)`` boolean mask; ``anchor_mu`` the values to pin them to.
    With ``active=False`` it behaves as an ordinary NB (the ``seed`` mode).
    """

    name = "nb"

    def __init__(self, mu, dispersion, anchor=None, active=False):
        super().__init__(mu, dispersion)
        self.anchor = (np.zeros(self.mu.shape, dtype=bool) if anchor is None
                       else np.asarray(anchor, dtype=bool))
        self.anchor_mu = self.mu.copy()
        self.active = bool(active)

    def m_step(self, X, offsets, gamma):
        super().m_step(X, offsets, gamma)
        if self.active and self.anchor.any():
            self.mu[self.anchor] = self.anchor_mu[self.anchor]

    def reorder(self, order):
        super().reorder(order)
        order = np.asarray(order)
        self.anchor = self.anchor[order]
        self.anchor_mu = self.anchor_mu[order]

    def n_free_params(self) -> int:
        n = super().n_free_params()
        return int(n - self.anchor.sum()) if self.active else n


#: With state occupancy known, a mark also counts as *high* in a state at this fold over
#: its genome-wide mean, so that one dominant state (active promoters on H3K27ac) cannot
#: push the midpoint above a real enhancer state's level.
DEFINITION_HIGH_FOLD = 2.0


def match_states_to_definitions(mean_matrix: np.ndarray, marks: Sequence[str],
                                defs: StateDefinitions,
                                occupancy: Optional[np.ndarray] = None) -> Tuple[List[int], np.ndarray]:
    """Pair fitted states with defined states by the best one-to-one pattern agreement.

    Each state's emission profile is scored against each definition as the mean over
    defined marks of (+1 if the mark lies on the right side of the mark's midpoint on
    the log scale, -1 otherwise). With ``occupancy`` (state coverage), a mark at
    ``DEFINITION_HIGH_FOLD`` x its occupancy-weighted genome mean is also high. The
    Hungarian assignment maximises total agreement.
    Returns ``(assignment, scores)``: ``assignment[j]`` is the fitted state index for
    definition ``j``.
    """
    from scipy.optimize import linear_sum_assignment

    defs = defs.restrict(marks)
    P = defs.pattern.to_numpy(dtype=int)
    M = np.asarray(mean_matrix, dtype=float)
    L = np.log1p(M)
    mid = (L.max(axis=0) + L.min(axis=0)) / 2.0
    high = L > mid[None, :]
    if occupancy is not None:
        occ = np.asarray(occupancy, dtype=float)
        occ = occ / max(occ.sum(), 1e-12)
        background = np.maximum((occ[:, None] * M).sum(axis=0), 1e-12)
        high |= M / background[None, :] >= DEFINITION_HIGH_FOLD
    side = np.where(high, 1.0, -1.0)  # (K, M)
    # Graded strength in [-1, 1] breaks ties between states with the same discrete pattern
    # (two H3K4me3+/H3K27ac+ states: the stronger one is the promoter).
    rng = np.maximum(L.max(axis=0) - L.min(axis=0), 1e-9)
    graded = np.tanh(2.0 * (L - mid[None, :]) / rng[None, :])
    scores = np.zeros((len(defs), L.shape[0]))
    tie = np.zeros_like(scores)
    for j in range(len(defs)):
        m = P[j] != FREE
        if m.any():
            scores[j] = (side[:, m] * P[j][m][None, :]).mean(axis=1)
            tie[j] = (graded[:, m] * P[j][m][None, :]).mean(axis=1)
        else:  # the free state: prefers the lowest total signal
            scores[j] = -L.sum(axis=1) / max(L.sum(axis=1).max(), 1e-9)
    # one discrete agreement step is 2 / n_defined_marks >= 0.2 for <= 10 marks; the
    # tie-break is bounded by 0.05 so it only ever orders states with equal agreement
    rows, cols = linear_sum_assignment(-(scores + 0.05 * tie))
    assignment = [0] * len(defs)
    for r, c in zip(rows, cols):
        assignment[r] = int(c)
    return assignment, scores


def apply_definitions_labels(seg, defs: StateDefinitions) -> None:
    """Relabel a fitted segmentation in place so defined states carry their given names.

    States that were matched keep the definition's name and record ``defined: <pattern>``
    as their rule; unmatched (extra) states keep their automatic name.
    """
    try:
        occupancy = seg.state_table()["coverage_soft"].to_numpy(dtype=float)
    except Exception:  # pragma: no cover - a segmentation without posteriors
        occupancy = None
    assignment, scores = match_states_to_definitions(seg.model.emissions.mean_matrix(), seg.marks, defs,
                                                     occupancy=occupancy)
    labels = list(seg.labels)
    rules = list(seg.rules)
    colors = list(seg.colors)
    used = set()
    for j, k in enumerate(assignment):
        name = defs.states[j]
        labels[k] = name
        rules[k] = f"defined: {defs.describe(name)} (agreement {scores[j, k]:+.2f})"
        if name in defs.colors:
            colors[k] = defs.colors[name]
        used.add(k)
    # Keep every label unique: an automatic name equal to a defined one gets a suffix.
    seen = {}
    for k, lbl in enumerate(labels):
        if k in used:
            seen[lbl] = seen.get(lbl, 0) + 1
    for k, lbl in enumerate(labels):
        if k not in used and lbl in seen:
            labels[k] = f"{lbl}_x"
    seg.labels, seg.rules, seg.colors = labels, rules, colors
    seg.meta["definitions"] = defs.to_frame().to_dict("records")
    seg.meta["definition_assignment"] = {defs.states[j]: int(k) for j, k in enumerate(assignment)}
