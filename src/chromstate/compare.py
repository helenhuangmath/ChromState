"""Statistical comparison of chromatin states across conditions, by region and by state.

:func:`differential_report` is the one call that answers both of the questions a
condition comparison raises, with the statistics kept separate:

**By region** -- for every predefined region (promoters, peaks, tiles, any BED) and every
contrast against the reference condition: the change in each state's share, a
variance-moderated *t*-test on the logit share across replicates (per state, with
Benjamini-Hochberg across regions *within* each state), an omnibus per-region test
(the most-changed state, Bonferroni-corrected across states), the replicate noise floor,
and with three or more conditions a moderated one-way *F*-test per region and state.

**By state** -- for every state and contrast: how many regions gain or lose it at the
chosen FDR, the mean and median change over regions, and the genome-wide tests from
:func:`chromstate.summary.state_tests` (replicates as units, genome blocks as units).

Regions must be defined without looking at the effect (see
:func:`chromstate.dynamics.differential_regions`); the default tiles are data-independent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .dynamics import (
    NoiseFloor,
    _logit,
    _region_membership,
    _tile_regions,
    benjamini_hochberg,
    differential_regions,
    moderated_t_test,
)
from .model.core import Segmentation
from .summary import state_tests


def moderated_f_test(groups: Sequence[np.ndarray], prior_df: float = 4.0,
                     prior_mask: Optional[np.ndarray] = None,
                     ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """One-way ANOVA per feature with the within-group variance shrunk across features.

    ``groups`` is a list of ``(n_features, n_replicates_g)`` arrays. The within-group
    mean square is shrunk toward its across-feature median with ``prior_df`` prior
    degrees of freedom, as in :func:`chromstate.dynamics.moderated_t_test`. Returns
    ``(F, p, (df1, df2))``.
    """
    from scipy.stats import f as f_dist

    sizes = np.array([g.shape[1] for g in groups])
    if (sizes < 2).any():
        raise ValueError("every condition needs 2+ replicates for an F-test")
    n_total = int(sizes.sum())
    g = len(groups)
    means = np.stack([x.mean(axis=1) for x in groups], axis=1)  # (R, g)
    grand = (means * sizes[None, :]).sum(axis=1) / n_total
    ss_between = ((means - grand[:, None]) ** 2 * sizes[None, :]).sum(axis=1)
    ss_within = sum(((x - x.mean(axis=1, keepdims=True)) ** 2).sum(axis=1) for x in groups)
    df1 = g - 1
    df2 = n_total - g
    ms_within = ss_within / df2
    ok = np.isfinite(ms_within) & (ms_within > 0)
    if prior_mask is not None and (ok & prior_mask).any():
        ok &= prior_mask  # prior from the tested features only, as in moderated_t_test
    prior = float(np.median(ms_within[ok])) if ok.any() else 1.0
    shrunk = (prior_df * prior + df2 * ms_within) / (prior_df + df2)
    F = (ss_between / df1) / np.maximum(shrunk, 1e-12)
    p = f_dist.sf(F, df1, df2 + prior_df)
    return F, p, (float(df1), float(df2 + prior_df))


@dataclass
class DifferentialReport:
    """Everything :func:`differential_report` produced, with writers."""

    regions: Dict[str, pd.DataFrame]          # contrast -> per-region table (omnibus)
    region_states: pd.DataFrame               # long: region x state x contrast
    states: pd.DataFrame                      # per state x contrast summary
    genome_state_tests: Optional[pd.DataFrame]
    anova: Optional[pd.DataFrame]             # region x state F-test over all conditions
    noise_floor: Optional[NoiseFloor]
    reference: str
    groups: Dict[str, List[str]]
    notes: List[str] = field(default_factory=list)

    def write(self, outdir: str, prefix: str = "", seg: Optional[Segmentation] = None,
              regions: Optional[pd.DataFrame] = None, figures: bool = True, palette: str = "scirep2025",
              png: bool = False) -> List[str]:
        """Write every table, the notes, and (with ``seg``) the figures as publication PDFs.
        ``png=True`` adds raster previews, which are for screen review only."""
        os.makedirs(outdir, exist_ok=True)
        files = []
        if figures and seg is not None:
            from . import viz

            figs = self.figures(seg, regions=regions, palette=palette)
            figdir = os.path.join(outdir, f"{prefix}figures")
            if png:
                os.makedirs(figdir, exist_ok=True)
                for name, fig in figs.items():
                    path = os.path.join(figdir, f"{name}.png")
                    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=viz.SURFACE)
                    files.append(path)
            files += viz.save_all(figs, figdir)

        def save(frame, name):
            path = os.path.join(outdir, f"{prefix}{name}")
            frame.to_csv(path, sep="\t", index=False)
            files.append(path)

        for contrast, tab in self.regions.items():
            save(tab, f"regions_{contrast}.tsv")
        save(self.region_states, "region_state_tests.tsv")
        save(self.states, "state_summary.tsv")
        if self.genome_state_tests is not None:
            save(self.genome_state_tests, "genome_state_tests.tsv")
        if self.anova is not None:
            save(self.anova, "region_state_anova.tsv")
        with open(os.path.join(outdir, f"{prefix}NOTES.md"), "w") as fh:
            fh.write("\n\n".join(self.notes) + "\n")
        files.append(os.path.join(outdir, f"{prefix}NOTES.md"))
        return files

    def figures(self, seg: Segmentation, regions: Optional[pd.DataFrame] = None, palette: str = "scirep2025",
                permuted_seed: int = 0) -> Dict[str, object]:
        """The statistical figures for this report (see :mod:`chromstate.viz_stats`)."""
        from . import viz_stats as vs

        figs: Dict[str, object] = {}
        if regions is None:
            first = next(iter(self.regions.values()))
            regions = first[["chrom", "start", "end", "name"]].reset_index(drop=True)
        tested = "qvalue" in self.region_states.columns
        from . import viz

        thr = None if self.noise_floor is None else float(self.noise_floor.threshold)
        for contrast, tab in self.regions.items():
            if "qvalue" in tab.columns:
                figs[f"volcano_{contrast}"] = viz.plot_change_volcano(tab, noise_floor=thr, label_top=3)
            if tested:
                try:
                    figs[f"region_state_heatmap_{contrast}"] = vs.plot_region_state_heatmap(
                        self.region_states, contrast=contrast, seg=seg, palette=palette)
                    figs[f"state_volcano_{contrast}"] = vs.plot_region_state_volcano(
                        self.region_states, contrast=contrast, seg=seg, palette=palette)
                except ValueError:
                    pass
        if tested and not self.states.empty:
            figs["state_gain_loss"] = vs.plot_state_gain_loss(self.states, seg=seg, palette=palette)
            ref = self.groups[self.reference]
            for c, reps in self.groups.items():
                if c == self.reference or len(reps) < 2 or len(ref) < 2:
                    continue
                contrast = f"{self.reference}_vs_{c}"
                obs = self.region_states.loc[self.region_states["contrast"] == contrast]
                perm = permuted_region_state_tests(seg, ref, reps, regions, seed=permuted_seed)
                figs[f"pvalue_calibration_{contrast}"] = vs.plot_pvalue_histograms(obs, perm)
                state = self.states.loc[self.states["contrast"] == contrast].sort_values(
                    "median_abs_delta", ascending=False)["state"].iloc[0]
                figs[f"replicate_concordance_{contrast}"] = vs.plot_replicate_concordance(
                    seg, regions, state, (ref[0], ref[1]), (ref[0], reps[0]), noise_floor=self.noise_floor)
        if any(len(v) >= 2 for v in self.groups.values()):
            figs["change_ecdf"] = vs.plot_change_ecdf(seg, self.groups, regions, reference=self.reference,
                                                      noise_floor=self.noise_floor)
        if self.anova is not None:
            figs["anova_volcano"] = vs.plot_anova_volcano(self.anova, seg=seg, palette=palette)
        return figs

    def significant_regions(self, contrast: str, fdr: float = 0.05) -> pd.DataFrame:
        tab = self.regions[contrast]
        col = "confident_change" if "confident_change" in tab.columns else "significant"
        return tab.loc[tab[col]] if col in tab.columns else tab.iloc[0:0]


def _region_occupancy(seg: Segmentation, sample: str, members: List[np.ndarray]) -> np.ndarray:
    post = seg.posterior[seg.dataset.sample_mask(sample)]
    return np.stack([post[m].mean(axis=0) for m in members])


def region_state_tests(
    seg: Segmentation,
    group_a: Sequence[str],
    group_b: Sequence[str],
    regions: pd.DataFrame,
    members: Optional[List[np.ndarray]] = None,
    prior_df: float = 4.0,
    fdr: float = 0.05,
    min_delta: float = 0.0,
    min_share: float = 0.01,
) -> pd.DataFrame:
    """Per region *and* state: share in each group, change, moderated t, p and q.

    ``q`` is Benjamini-Hochberg across regions within each state (each state is its own
    family of hypotheses), unlike the omnibus per-region test, which corrects across
    states within a region. ``significant`` requires ``q < fdr`` and
    ``|delta| >= min_delta`` (in share units, 0-1).

    Shares are tested on the empirical-logit scale of the region's own bin count (see
    :func:`chromstate.dynamics._logit`). A state that is essentially absent from a region
    in *both* groups (mean share below ``min_share`` and below half a bin, ``0.5 / n``,
    on each side) is not tested: its shares are below what the region can resolve, the
    t-statistic is 0 by construction and the p-value 1.
    Those rows get ``tested = False`` and NaN p and q, so they neither inflate the
    multiple-testing burden nor pile up at p = 1 in a calibration plot.
    """
    ds = seg.dataset
    bins = ds.bins.loc[ds.sample_mask(group_a[0]), ["chrom", "start", "end"]].reset_index(drop=True)
    if members is None:
        members = _region_membership(bins, regions)
    occ_a = np.stack([_region_occupancy(seg, s, members) for s in group_a], axis=1)  # (R, na, K)
    occ_b = np.stack([_region_occupancy(seg, s, members) for s in group_b], axis=1)
    can_test = occ_a.shape[1] >= 2 and occ_b.shape[1] >= 2
    sizes = np.maximum(np.array([len(m) for m in members]), 1)
    floor = np.maximum(min_share, 0.5 / sizes)
    rows = []
    names = regions["name"].to_numpy() if "name" in regions.columns else np.array(
        [f"{c}:{s}-{e}" for c, s, e in zip(regions["chrom"], regions["start"], regions["end"])])
    for k, lbl in enumerate(seg.labels):
        a, b = occ_a[:, :, k], occ_b[:, :, k]
        ma, mb = a.mean(axis=1), b.mean(axis=1)
        frame = pd.DataFrame({"name": names, "chrom": regions["chrom"].to_numpy(),
                              "start": regions["start"].to_numpy(), "end": regions["end"].to_numpy(),
                              "state": lbl, "share_a": ma, "share_b": mb, "delta": mb - ma,
                              "log2_ratio": np.log2((mb + 1e-3) / (ma + 1e-3))})
        if can_test:
            informative = (ma >= floor) | (mb >= floor)
            t, p, df = moderated_t_test(_logit(a, n_bins=sizes), _logit(b, n_bins=sizes),
                                        prior_df=prior_df, prior_mask=informative)
            t = np.where(informative, t, np.nan)
            p = np.where(informative, p, np.nan)
            q = np.full(len(p), np.nan)
            if informative.any():
                q[informative] = benjamini_hochberg(p[informative])
            frame["tested"] = informative
            frame["t"] = t
            frame["pvalue"] = p
            frame["qvalue"] = q
            frame["significant"] = informative & (q < fdr) & (np.abs(frame["delta"]) >= min_delta)
        rows.append(frame)
    out = pd.concat(rows, ignore_index=True)
    out.attrs["tested"] = can_test
    return out


def permuted_region_state_tests(
    seg: Segmentation,
    group_a: Sequence[str],
    group_b: Sequence[str],
    regions: pd.DataFrame,
    seed: int = 0,
    **kwargs,
) -> pd.DataFrame:
    """:func:`region_state_tests` after swapping one sample between the groups.

    A null for the calibration check: with the labels scrambled there is no condition
    effect, so the p-values should be uniform. Only splits that mix the groups are used.
    """
    rng = np.random.default_rng(seed)
    a, b = list(group_a), list(group_b)
    i, j = int(rng.integers(len(a))), int(rng.integers(len(b)))
    pa, pb = list(a), list(b)
    pa[i], pb[j] = b[j], a[i]
    out = region_state_tests(seg, pa, pb, regions, **kwargs)
    out.attrs["permutation"] = (pa, pb)
    return out


def region_noise_floor(
    seg: Segmentation,
    groups: Dict[str, Sequence[str]],
    regions: pd.DataFrame,
    members: Optional[List[np.ndarray]] = None,
    quantile: float = 0.95,
) -> NoiseFloor:
    """The replicate noise floor at *region* level.

    :func:`chromstate.dynamics.replicate_noise_floor` is a quantile of the per-bin
    total-variation distance between replicates. Averaging over a region shrinks that
    distance, so the bin-level threshold is far too permissive when it is applied to
    region statistics. This version computes the same quantity on the per-region mean
    composition of every within-condition replicate pair, and is what the region tests
    and figures should use.
    """
    ds = seg.dataset
    groups = {c: list(v) for c, v in groups.items() if len(v) >= 2}
    if not groups:
        raise ValueError("a noise floor needs a condition with two or more replicates")
    bins = ds.bins.loc[ds.sample_mask(next(iter(groups.values()))[0]), ["chrom", "start", "end"]].reset_index(drop=True)
    if members is None:
        members = _region_membership(bins, regions)
    occ = {}
    pooled, pairs, rows = [], [], []
    for c, reps in groups.items():
        for s in reps:
            occ[s] = _region_occupancy(seg, s, members)
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                d = 0.5 * np.abs(occ[reps[i]] - occ[reps[j]]).sum(axis=1)
                d = d[np.isfinite(d)]
                pooled.append(d)
                pairs.append((reps[i], reps[j]))
                rows.append({"condition": c, "sample_a": reps[i], "sample_b": reps[j],
                             "median": float(np.median(d)), f"q{int(quantile * 100)}": float(np.quantile(d, quantile))})
    pooled_arr = np.concatenate(pooled)
    threshold = float(np.quantile(pooled_arr, quantile))
    note = (f"Region-level noise floor from {len(pairs)} replicate pair(s) over {len(members):,} regions: the "
            f"{int(quantile * 100)}% quantile of region total-variation distance between replicates is "
            f"{threshold:.3f}. Condition differences at or below it are within replicate noise.")
    return NoiseFloor(threshold, quantile, pooled_arr, pairs, pd.DataFrame(rows), note)


def differential_report(
    seg: Segmentation,
    groups: Dict[str, Sequence[str]],
    regions: Optional[pd.DataFrame] = None,
    reference: Optional[str] = None,
    tile_bins: int = 10,
    min_bins: int = 3,
    fdr: float = 0.05,
    min_delta: float = 0.0,
    prior_df: float = 4.0,
    noise_floor: Optional[NoiseFloor] = None,
    genome_tests: bool = True,
    seed: int = 0,
) -> DifferentialReport:
    """Compare every condition with the reference, by region and by state.

    Parameters
    ----------
    groups
        ``{condition: [sample, ...]}``. Conditions with one sample get effect sizes
        only; two or more replicates on both sides enable the tests.
    regions
        ``chrom``/``start``/``end`` (+ ``name``). Defaults to fixed tiles of
        ``tile_bins`` bins. Must not be chosen from the observed effect.
    reference
        The condition every other one is compared with (default: the first).
    min_delta
        A change in share below this (0-1) is never called significant, however small
        its p-value; it separates statistical from biological significance.
    noise_floor
        Replicate noise floor on the *region* statistic (:func:`region_noise_floor`);
        computed from the replicated conditions when omitted. Do not pass the bin-level
        floor from :func:`chromstate.dynamics.replicate_noise_floor` here.
    genome_tests
        Also run :func:`chromstate.summary.state_tests` (genome share per state).
    """
    groups = {c: list(v) for c, v in groups.items()}
    reference = reference or next(iter(groups))
    if reference not in groups:
        raise ValueError(f"reference {reference!r} is not one of {list(groups)}")
    ds = seg.dataset
    for c, reps in groups.items():
        for s in reps:
            if s not in ds.samples:
                raise ValueError(f"condition {c!r} names unknown sample {s!r}")
    bins = ds.bins.loc[ds.sample_mask(groups[reference][0]), ["chrom", "start", "end"]].reset_index(drop=True)
    if regions is None:
        regions = _tile_regions(bins, tile_bins)
    regions = regions.reset_index(drop=True)
    if "name" not in regions.columns:
        regions = regions.assign(name=[f"{c}:{s}-{e}" for c, s, e in
                                       zip(regions["chrom"], regions["start"], regions["end"])])
    members = _region_membership(bins, regions)
    keep = np.array([len(m) >= min_bins for m in members])
    regions = regions.loc[keep].reset_index(drop=True)
    members = [m for m, k in zip(members, keep) if k]
    if not members:
        raise ValueError(f"no region covers {min_bins}+ bins")

    notes = [f"Reference condition: {reference}. {len(regions):,} regions of >= {min_bins} bins."]
    replicated = {c: reps for c, reps in groups.items() if len(reps) >= 2}
    if noise_floor is None and len(replicated) >= 1:
        try:
            noise_floor = region_noise_floor(seg, replicated, regions, members=members)
            notes.append(noise_floor.note)
        except Exception as exc:  # pragma: no cover - depends on data
            notes.append(f"Noise floor not computed: {exc}")

    per_region: Dict[str, pd.DataFrame] = {}
    long_frames = []
    state_rows = []
    for cond, reps in groups.items():
        if cond == reference:
            continue
        contrast = f"{reference}_vs_{cond}"
        omni = differential_regions(seg, groups[reference], reps, regions=regions, min_bins=min_bins,
                                    noise_floor=noise_floor, prior_df=prior_df, fdr=fdr)
        per_region[contrast] = omni.table
        notes.append(f"[{contrast}] {omni.note}")
        rs = region_state_tests(seg, groups[reference], reps, regions, members=members,
                                prior_df=prior_df, fdr=fdr, min_delta=min_delta)
        rs.insert(0, "contrast", contrast)
        long_frames.append(rs)
        tested = bool(rs.attrs["tested"])
        for lbl, sub in rs.groupby("state", sort=False):
            d = sub["delta"].to_numpy()
            row = {"contrast": contrast, "state": lbl, "n_regions": len(sub),
                   "mean_delta": float(d.mean()), "median_abs_delta": float(np.median(np.abs(d))),
                   "n_gain": int((d > 0).sum()), "n_loss": int((d < 0).sum())}
            if tested:
                sig = sub["significant"].to_numpy()
                inf = sub["tested"].to_numpy()
                row["n_tested"] = int(inf.sum())
                row["n_sig_gain"] = int((sig & (d > 0)).sum())
                row["n_sig_loss"] = int((sig & (d < 0)).sum())
                row["frac_sig"] = float(sig[inf].mean()) if inf.any() else np.nan
                row["min_q"] = float(np.nanmin(sub["qvalue"])) if inf.any() else np.nan
            if noise_floor is not None and "driver_state" in omni.table.columns:
                drv = omni.table["driver_state"] == lbl
                col = "confident_change" if "confident_change" in omni.table.columns else "significant"
                row["n_regions_driven"] = int((drv & omni.table[col]).sum())
            state_rows.append(row)
    region_states = pd.concat(long_frames, ignore_index=True) if long_frames else pd.DataFrame()
    states = pd.DataFrame(state_rows)

    genome = None
    if genome_tests and len(groups) >= 2:
        try:
            genome = state_tests(seg, groups, reference=reference, seed=seed, fdr=fdr)
            notes.append(genome.attrs.get("note", ""))
            if not states.empty:
                g = genome.rename(columns={"condition": "cond"})
                if "cond" in g.columns and "state" in g.columns:
                    g["contrast"] = [f"{reference}_vs_{c}" for c in g["cond"]]
                    keepc = [c for c in ("contrast", "state", "delta_pct", "delta_lo", "delta_hi",
                                         "p_block", "q_block", "t", "p_rep", "q_rep") if c in g.columns]
                    states = states.merge(g[keepc], on=["contrast", "state"], how="left")
        except Exception as exc:  # pragma: no cover - depends on data
            notes.append(f"Genome-wide state tests skipped: {exc}")

    anova = None
    if len(replicated) >= 3:
        occ = {c: np.stack([_region_occupancy(seg, s, members) for s in reps], axis=1)
               for c, reps in replicated.items()}
        rows = []
        sizes = np.maximum(np.array([len(m) for m in members]), 1)
        floor = np.maximum(0.01, 0.5 / sizes)
        for k, lbl in enumerate(seg.labels):
            means = np.stack([occ[c][:, :, k].mean(axis=1) for c in replicated], axis=1)
            # as in region_state_tests: a state absent from the region in every condition
            # is not tested
            informative = means.max(axis=1) >= floor
            F, p, (df1, df2) = moderated_f_test(
                [_logit(occ[c][:, :, k], n_bins=sizes) for c in replicated], prior_df=prior_df,
                prior_mask=informative)
            p = np.where(informative, p, np.nan)
            q = np.full(len(p), np.nan)
            if informative.any():
                q[informative] = benjamini_hochberg(p[informative])
            frame = pd.DataFrame({"name": regions["name"], "chrom": regions["chrom"],
                                  "start": regions["start"], "end": regions["end"], "state": lbl,
                                  "F": np.where(informative, F, np.nan), "pvalue": p, "qvalue": q})
            for j, c in enumerate(replicated):
                frame[f"share_{c}"] = means[:, j]
            frame["max_delta"] = means.max(axis=1) - means.min(axis=1)
            rows.append(frame)
        anova = pd.concat(rows, ignore_index=True)
        anova["significant"] = anova["qvalue"] < fdr
        notes.append(f"Moderated one-way F-test across {len(replicated)} replicated conditions "
                     f"({', '.join(replicated)}), df ({df1:g}, {df2:g}); q is BH within state. "
                     f"{int(anova['significant'].sum())} region-state pairs pass FDR {fdr:g}.")

    return DifferentialReport(per_region, region_states, states, genome, anova, noise_floor,
                              reference, groups, notes)
