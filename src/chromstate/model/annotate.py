"""Automatic state mnemonics and colours.

ChromHMM users read segmentations through mnemonics (TssA, EnhA, ReprPC, Quies...) and
the Roadmap colour scheme. ChromState assigns the same vocabulary automatically from the
fitted emission profile, so output is legible to anyone who already reads ChromHMM
tracks -- and so the two can be placed side by side in a figure.

Assignment is a transparent rule set over z-scored emission means, not a black box: the
rules are listed in :data:`STATE_RULES` and the matched rule is reported alongside the
label so a user can see why a state got its name.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Roadmap Epigenomics 18-state colour conventions, keyed by mnemonic.
STATE_COLORS: Dict[str, str] = {
    "TssA": "#FF0000",
    "TssFlnk": "#FF4500",
    "TssBiv": "#CD5C5C",
    "BivFlnk": "#E9967A",
    "EnhA": "#FFC34D",
    "EnhWk": "#FFFF00",
    "EnhBiv": "#BDB76B",
    "Tx": "#008000",
    "TxWk": "#90EE90",
    "ReprPC": "#808080",
    "ReprPCWk": "#C0C0C0",
    "Het": "#8A91D0",
    "HetWk": "#C5C9E8",
    "ZNF/Rpts": "#66CDAA",
    "Quies": "#F5F5F5",
    "Unknown": "#DDDDDD",
}

# Spelled-out state names, keyed by mnemonic. Figures print these rather than the
# mnemonics: the abbreviations are only legible to people already fluent in ChromHMM
# output, and a figure has to stand on its own.
STATE_FULL_NAMES: Dict[str, str] = {
    "TssA": "Active transcription start site",
    "TssFlnk": "Flanking transcription start site",
    "TssBiv": "Bivalent transcription start site",
    "BivFlnk": "Flanking bivalent region",
    "EnhA": "Active enhancer",
    "EnhWk": "Weak enhancer",
    "EnhBiv": "Bivalent enhancer",
    "Tx": "Strong transcription",
    "TxWk": "Weak transcription",
    "ReprPC": "Polycomb repressed",
    "ReprPCWk": "Weakly Polycomb repressed",
    "Het": "Heterochromatin",
    "HetWk": "Weak heterochromatin",
    "ZNF/Rpts": "Zinc-finger genes and repeats",
    "Quies": "Quiescent, low signal",
    "Unknown": "Unclassified state",
}

# Canonical mark aliases, so H3K4me3 / K4me3 / h3k4me3 all resolve.
MARK_ALIASES = {
    "h3k4me3": "K4me3",
    "k4me3": "K4me3",
    "h3k4me1": "K4me1",
    "k4me1": "K4me1",
    "h3k27ac": "K27ac",
    "k27ac": "K27ac",
    "h3k27me3": "K27me3",
    "k27me3": "K27me3",
    "h3k36me3": "K36me3",
    "k36me3": "K36me3",
    "h3k9me3": "K9me3",
    "k9me3": "K9me3",
    "h3k9ac": "K9ac",
    "atac": "ATAC",
    "dnase": "ATAC",
    "ctcf": "CTCF",
}

# (label, required-high marks, required-low marks). Evaluated in order; first match wins.
STATE_RULES: List[Tuple[str, Sequence[str], Sequence[str]]] = [
    ("TssBiv", ("K4me3", "K27me3"), ()),
    ("EnhBiv", ("K4me1", "K27me3"), ("K4me3",)),
    ("TssA", ("K4me3",), ("K27me3",)),
    ("TssFlnk", ("K4me3", "K4me1"), ("K27me3",)),
    ("EnhA", ("K4me1", "K27ac"), ("K4me3",)),
    ("EnhWk", ("K4me1",), ("K27ac", "K4me3")),
    # Panels without H3K4me1: acetylation away from H3K4me3 is the enhancer signature.
    ("EnhA", ("K27ac",), ("K4me3",)),
    ("Tx", ("K36me3",), ()),
    ("ReprPC", ("K27me3",), ("K4me3", "K4me1")),
    ("Het", ("K9me3",), ()),
]


def full_state_name(label: str) -> str:
    """Spell out a state mnemonic for display: ``EnhA_2`` -> ``Active enhancer (2)``.

    Labels that are not recognised mnemonics are returned unchanged, so a user-supplied
    label is never mangled.
    """
    base, _, suffix = label.partition("_")
    full = STATE_FULL_NAMES.get(base)
    if full is None:
        return label
    return f"{full} ({suffix})" if suffix else full


def canonical_mark(name: str) -> str:
    key = name.strip().lower().replace("-", "").replace("_", "")
    return MARK_ALIASES.get(key, name)


#: A mark is *present* in a state at >= PRESENT_FOLD x its genome-wide mean, or at
#: >= PRESENT_FRAC of its strongest state; *strong* at STRONG_FOLD / STRONG_FRAC.
#: Two scales because each fails alone: on a real genome active states cover ~1-3%, so
#: the genome mean is near background and fold enrichment reads naturally; when states
#: cover similar shares (small panels, simulations) the mean is inflated by the active
#: states, and the fraction of the strongest state is what still separates them.
PRESENT_FOLD = 1.5
STRONG_FOLD = 4.0
PRESENT_FRAC = 0.25
STRONG_FRAC = 0.35


def _annotate_by_enrichment(
    M: np.ndarray, marks: Sequence[str], occupancy: np.ndarray
) -> Tuple[List[str], List[str]]:
    """Name states from each mark's fold enrichment over its genome-wide mean.

    The genome-wide mean of a mark is the occupancy-weighted average of the state means,
    i.e. what an average bin carries. Enrichment relative to that background is how a
    biologist reads a state ("H3K27me3 4x over background"), and unlike scaling to the
    strongest state it is not distorted when one state (an active promoter) dominates a
    mark's dynamic range. Rules run in order; the first match names the state.
    """
    occ = np.asarray(occupancy, dtype=np.float64)
    occ = occ / max(occ.sum(), 1e-12)
    background = np.maximum((occ[:, None] * M).sum(axis=0), 1e-12)
    F = M / background
    R = M / np.maximum(M.max(axis=0, keepdims=True), 1e-12)
    canon = [canonical_mark(m) for m in marks]
    idx = {c: j for j, c in enumerate(canon)}

    def f(k: int, mark: str) -> float:
        return float(F[k, idx[mark]]) if mark in idx else 0.0

    def r(k: int, mark: str) -> float:
        return float(R[k, idx[mark]]) if mark in idx else 0.0

    def strongest(k: int) -> str:
        return canon[int(np.argmax(F[k]))]

    labels: List[str] = []
    rules: List[str] = []
    for k in range(M.shape[0]):
        present = lambda m, k=k: f(k, m) >= PRESENT_FOLD or r(k, m) >= PRESENT_FRAC  # noqa: E731
        strong = lambda m, k=k: f(k, m) >= STRONG_FOLD or r(k, m) >= STRONG_FRAC  # noqa: E731
        if not any(present(m) for m in canon):
            name = "Quies"
        elif strong("K4me3") and strong("K27me3"):
            name = "TssBiv"
        elif strong("K9me3") and strongest(k) == "K9me3":
            name = "Het"
        elif strong("K4me3"):
            name = "TssA"
        elif strong("K36me3") and strongest(k) == "K36me3":
            name = "Tx"
        elif present("K27ac") and present("K4me1") and f(k, "K4me1") > f(k, "K4me3"):
            name = "EnhA"
        elif present("K4me3") and (present("K27ac") or present("K4me1")):
            name = "TssFlnk"
        elif present("K27ac") and not present("K4me3"):
            name = "EnhA"
        elif present("K4me1"):
            name = "EnhWk"
        elif strong("K27me3"):
            name = "ReprPC"
        elif present("K4me3") and present("K27me3"):
            name = "BivFlnk"
        elif present("K27me3"):
            name = "ReprPCWk"
        elif present("K9me3"):
            name = "HetWk"
        elif present("K36me3"):
            name = "TxWk"
        else:
            name = "Unknown"
        enriched = [f"{marks[j]} {F[k, j]:.1f}x" for j in np.argsort(-F[k])
                    if F[k, j] >= PRESENT_FOLD or R[k, j] >= PRESENT_FRAC]
        labels.append(name)
        rules.append("fold over genome mean: " + (", ".join(enriched) if enriched else "no mark enriched"))
    return labels, rules


def annotate_states(
    mean_matrix: np.ndarray,
    marks: Sequence[str],
    high_frac: float = 0.35,
    low_frac: float = 0.20,
    quiet_frac: float = 0.15,
    occupancy: Optional[np.ndarray] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """Assign a mnemonic, colour, and matched-rule string to each state.

    With ``occupancy`` (each state's share of the genome; ``fit`` always supplies it),
    states are named from fold enrichment over the genome-wide background -- the
    recommended mode, see :func:`_annotate_by_enrichment`. Without it, the legacy rules
    below scale each mark to its strongest state.

    Parameters
    ----------
    mean_matrix
        ``(K, M)`` emission means from the fitted model.
    marks
        Mark names, one per column.
    high_frac, low_frac
        A mark is "high" for a state when its emission reaches ``high_frac`` of that
        mark's maximum across states, and "low" when it falls below ``low_frac``.
        Normalising within each mark keeps the rules independent of sequencing depth and
        antibody efficiency, which differ wildly between marks.

        Fixed fractions are used rather than across-state quantiles because quantiles
        make every threshold depend on K and on how the other states happened to land: a
        mark present in only one state still has 40% of states sitting "above the 40th
        percentile", so quantile rules mislabel exactly the sparse, informative marks
        (H3K9me3, H3K36me3) that identify a state most cleanly.
    quiet_frac
        A state in which *every* mark is below this fraction of that mark's maximum is
        called Quiescent before any other rule is tried. The test is per mark, not on the
        summed emission: on real data one mark (H3K4me3 at active promoters) can dwarf
        the rest, and a summed threshold then calls a clear Polycomb or heterochromatin
        state quiescent.

    Returns
    -------
    ``(labels, colors, rules)`` -- three lists of length K. Labels are made unique by
    appending an index when a rule matches more than one state.
    """
    M = np.asarray(mean_matrix, dtype=np.float64)
    if occupancy is not None:
        raw, rules = _annotate_by_enrichment(M, marks, occupancy)
        labels = _uniquify(raw)
        colors = [STATE_COLORS.get(lbl.split("_")[0], STATE_COLORS["Unknown"]) for lbl in labels]
        return labels, colors, rules

    K = M.shape[0]
    canon = [canonical_mark(m) for m in marks]

    # Column-normalise so marks with different dynamic range are comparable.
    denom = np.maximum(M.max(axis=0, keepdims=True), 1e-12)
    Z = M / denom

    idx = {c: j for j, c in enumerate(canon)}

    raw_labels: List[str] = []
    rules: List[str] = []
    for k in range(K):
        if Z[k].max() < quiet_frac:
            raw_labels.append("Quies")
            rules.append(f"every mark below {quiet_frac:.0%} of its maximum across states")
            continue
        matched = None
        for label, highs, lows in STATE_RULES:
            if not all(h in idx for h in highs):
                continue
            if not all(Z[k, idx[h]] >= high_frac for h in highs):
                continue
            if not all(Z[k, idx[low]] <= low_frac for low in lows if low in idx):
                continue
            matched = (label, highs, lows)
            break
        if matched is None:
            top = int(np.argmax(Z[k]))
            raw_labels.append("Unknown")
            rules.append(f"no rule matched; strongest mark {marks[top]}")
        else:
            label, highs, lows = matched
            raw_labels.append(label)
            detail = "high " + "+".join(highs)
            if lows:
                detail += ", low " + "+".join(lows)
            rules.append(detail)

    labels = _uniquify(raw_labels)
    colors = [STATE_COLORS.get(lbl.split("_")[0], STATE_COLORS["Unknown"]) for lbl in labels]
    return labels, colors, rules


def _uniquify(labels: Sequence[str]) -> List[str]:
    counts: Dict[str, int] = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    seen: Dict[str, int] = {}
    out: List[str] = []
    for lbl in labels:
        if counts[lbl] == 1:
            out.append(lbl)
        else:
            seen[lbl] = seen.get(lbl, 0) + 1
            out.append(f"{lbl}_{seen[lbl]}")
    return out


def classify_marks(marks: Sequence[str]) -> Dict[str, List[int]]:
    """Group mark indices into activating / repressive / elongation sets.

    Used by the bivalency score, which needs to know which marks pull in which
    direction. Marks it does not recognise are returned under ``other`` and are excluded
    from the bivalency calculation rather than being guessed at.
    """
    active = {"K4me3", "K4me1", "K27ac", "K9ac", "ATAC"}
    repressive = {"K27me3", "K9me3"}
    elongation = {"K36me3"}
    groups: Dict[str, List[int]] = {"active": [], "repressive": [], "elongation": [], "other": []}
    for j, m in enumerate(marks):
        c = canonical_mark(m)
        if c in active:
            groups["active"].append(j)
        elif c in repressive:
            groups["repressive"].append(j)
        elif c in elongation:
            groups["elongation"].append(j)
        else:
            groups["other"].append(j)
    return groups


def promoter_marks(marks: Sequence[str]) -> Tuple[Optional[int], Optional[int]]:
    """Index of (H3K4me3, H3K27me3), the canonical bivalency pair, or ``None`` if absent."""
    canon = [canonical_mark(m) for m in marks]
    k4 = canon.index("K4me3") if "K4me3" in canon else None
    k27 = canon.index("K27me3") if "K27me3" in canon else None
    return k4, k27
