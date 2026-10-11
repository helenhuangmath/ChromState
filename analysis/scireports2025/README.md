# ChromState on the Sci Rep 2025 CD8 T-cell data

First real-data test of ChromState: Huang, Baxter et al., *Sci Rep* 2025
(GSE285245 CUT&RUN, GSE285248 RNA-seq). P14 CD8 T cells: naive (TN), memory
(TMEM, LCMV Armstrong d30), exhausted (TEX, LCMV clone 13 d32). Marks: H3K4me3,
H3K27ac, H3K27me3, H3K9me3; RNA-seq with 4 replicates per group (TN sampled at d30
and d32).

## Run

```bash
cd analysis/scireports2025
export PYTHONPATH=../../src:.
# CHROMSTATE_SCIREP_DATA=/path/to/CutRun_mm10   (default: the lab folder)
python s01_bin_tracks.py     # pooled bigWigs -> 200 bp bins          (~8 min)
python s02_fit_model.py      # choose K, fit joint TN/TMEM/TEX model  (~30 min)
python s03_rna.py            # pydeseq2 differential expression       (~1 min)
python s04_gene_features.py  # decode genome, per-gene features       (~6 min)
python s05_predict.py        # prediction benchmark                   (~40 min)
python s06_bivalency.py      # bivalency vs published poised calls    (~1 min)
python s07_figures.py        # Nature-style figures -> out/figures/
python s08_importance.py     # state importance; additive vs combinatorial  (~25 min)
python s09_cross_lag.py      # chromatin-first vs RNA-first cross-lag test  (~2 min)
python s10_figures_order.py  # Fig. 4 (ordering) and Fig. 5 (importance)
```

Outputs go to `out/` (git-ignored).

## Input limitation

No BAMs were available locally, so chromatin input is the **replicate-pooled,
DESeq2-scaled coverage tracks**, rounded to integers for the negative-binomial
model. One track per condition means no chromatin replicates: the replicate noise
floor and differential-state tests are not run here. Per-replicate counts from the
BAMs (`/home/hhua/Berger_Zhen/Berger_Zhen_CutRun_d30_3Conds_All_mm10/` on the
cluster) would remove both limitations. There is no spike-in, so global changes in a
mark between cell states are normalised away.

## Results (K = 15, fit on chr11/17/19, decoded genome-wide)

Prediction is scored by Spearman ρ on held-out chromosomes (5-fold, grouped by
chromosome), with 95% CIs from 1,000 gene bootstraps. Δρ below is ChromState
quantitative minus the named feature set, paired on the same resample.

| Task | ChromState quantitative | Raw signal | ChromHMM composition | ChromHMM TSS label |
|---|---|---|---|---|
| Expression level, TN | **0.821** | 0.802 | 0.793 | 0.632 |
| Expression change, TMEM vs TN | 0.511 | **0.529** | 0.407 | 0.162 |
| Expression change, TEX vs TN | **0.553** | 0.542 | 0.442 | 0.179 |
| Expression change, TEX vs TMEM | **0.432** | 0.411 | 0.293 | 0.083 |
| Later change from TN chromatin, TEX vs TN | **0.263** | 0.257 | 0.239 | 0.174 |

* **Quantitative state beats hard labels by a wide margin** (Δρ ≈ +0.2 for level,
  +0.35 for change, all CIs exclude 0). This is the clearest result.
* **Against raw signal it is a small, inconsistent edge**: better for level (Δρ +0.02,
  CI 0.016–0.022), TEX vs TMEM (+0.022) and TEX vs TN (+0.011, CI −0.001–0.023);
  *worse* for TMEM vs TN (−0.018, CI −0.030 to −0.005). Combining quantitative state
  with raw signal is best in every change task (ρ 0.549 / 0.581 / 0.468).
* **H3K27ac change tracks expression change most closely** (single-mark ρ 0.51–0.52),
  then H3K27me3 and H3K4me3; H3K9me3 change barely tracks it (ρ < 0.09).
* **TN chromatin anticipates later change only modestly** (ρ ≈ 0.19 TMEM, 0.26 TEX),
  and adds to TN expression level rather than replacing it (TN expression alone
  0.19; with ChromState 0.27 / 0.34).
* **Bivalency**: ChromState's bivalency score ranks the peak-called TN bivalent
  promoters first (AUROC 0.92; same data underneath, so a consistency check). Among
  TN bivalent promoters, H3K27me3-dominant ones rise later, by median log2FC +0.63
  (TMEM) and +1.32 (TEX). **About two thirds of that is regression to the mean**:
  compared with genes of matched TN expression the excess is +0.17 and +0.45.

### Which states matter, and is it additive or combinatorial? (Fig. 5)

* **One state dominates**: active TSS (H3K4me3 26×, H3K27ac 29×) is the only state
  whose removal costs much accuracy (held-out R² drops 0.35 for level, 0.42 for change).
  For *change*, the next are bivalent TSS, the H3K4me3+H3K27ac enhancer-like state and
  the Polycomb states (drops 0.04–0.09). Heterochromatin and the H3K9me3 states add
  nothing unique.
* **Single marks**: H3K27ac is the most informative alone (ρ 0.74 level, 0.52 change);
  H3K9me3 alone is nearly uninformative for change (ρ < 0.09).
* **Mostly additive, modestly combinatorial**: allowing marks to interact adds Δρ
  +0.016 to +0.023 over the additive model, all CIs above 0. The strongest pairs are
  H3K4me3 × H3K27me3 (bivalency; level, TMEM) and H3K27ac × H3K27me3 (TEX).
  ChromState states, which encode combinations by construction, gain +0.033 (level) and
  +0.030 (TEX) but not for TMEM (−0.008, CI spans 0).

### Which comes first? (Fig. 4, cross-lagged test)

With TN as the precursor of TMEM/TEX: does TN chromatin predict later expression
beyond TN expression (chromatin leads), or the reverse (RNA leads)? Effects are small
(standardised asymmetries ≤ 0.09):

* **Lean chromatin-first** (TEX): H3K27me3 (promoter and body), Polycomb and bivalent
  TSS promoter states, and the H3K4me3+H3K27ac states. This is consistent with poised or
  Polycomb promoters being set before their genes change, and it survives the noise bias
  below.
* **Lean RNA-first**: H3K9me3 (promoter, both contrasts; the largest asymmetry) and
  H3K4me3 (promoter, plus gene body in TMEM), which fits H3K4me3's known coupling to
  transcription.
* **Caveat**: chromatin here has no replicates and is noisier than RNA (TN RNA day 30
  vs 32 ρ = 0.994). That biases this test toward "RNA leads", so the RNA-first calls
  (especially H3K9me3) need per-replicate chromatin before they can be trusted. Two
  endpoints are not a time course, and none of this is causal.

## Known issues

* The best K = 15 optimum was reached by 1 of 8 restarts, and held-out likelihood was
  still rising at K = 15. States may shift with more restarts or a larger K.
* Automatic state names under-call weak states: several states named "Quiescent"
  carry H3K9me3 or weak H3K4me3 + H3K27me3 (see the log-scale emission heatmap,
  Fig. 1a). Read the emission profile, not the name.
* None of this orders chromatin and RNA changes in time; all conditions are
  endpoints.
