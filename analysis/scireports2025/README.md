# ChromState on the Sci Rep 2025 CD8 T-cell data


Browse the included [example PDFs](../../docs/figures/README.md#cd8-t-cell-application) without downloading the input data.

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
python s11_reliability.py    # chromatin/RNA reliability from per-replicate tracks (~1 min)
python s09_cross_lag.py      # rerun after s11: measurement-error-corrected ordering
python s10_figures_order.py  # Fig. 4 (ordering) and Fig. 5 (importance)
python s12_marker_genes.py   # Fig. 6 (marker genes)
python s13_tracks.py Pdcd1 Tigit Tox Tcf7   # Fig. 7, one PDF per gene (any genes)
python s14_enhancers.py      # ATAC elements, ABC enhancer-gene links       (~3 min)
python s15_chromatin_programs.py  # Figs. 8-9: poised genes, chromatin ahead of RNA, gene clusters (~2 min)
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

## Results

The full results, with interpretation and figure legends, are in
**[RESULTS.md](RESULTS.md)**. In brief:

1. **Prediction.** Quantitative chromatin states predict expression level (ρ 0.82) and
   its change between cell states (ρ 0.51–0.55; 0.55–0.58 with raw signal) better than
   discrete states (ρ 0.16–0.18 for change).
2. **Marker genes.** They reproduce known biology. Naive genes (*Sell*, *Ccr7*) acquire
   Polycomb or bivalent marking as they are silenced. Effector and exhaustion genes are
   activated from repressed or poised promoters. *Pdcd1* and *Tigit* carry weak active
   marking in TN that is amplified 4- to 9-fold in TEX (Fig. 6).
3. **Bivalent promoters.** H3K27me3-dominant bivalent TN promoters anticipate later
   activation, beyond expression-matched controls (Fig. 3).
4. **State importance.** The active-promoter state dominates. Change additionally
   depends on bivalent, flanking-TSS and Polycomb promoters. Marks act mostly
   additively, with a modest bivalent and active-versus-Polycomb combinatorial
   component (Fig. 5).
5. **Ordering.** The test was corrected for measured reliability (H3K9me3 0.67,
   H3K27me3 0.93, others ≥ 0.99):
   - active promoter states in TN lean chromatin-first (TEX);
   - H3K4me3, H3K9me3 and Polycomb in memory lean RNA-first;
   - H3K27me3 and bivalency show no genome-wide lead (Fig. 4).

## Known issues

* **Fit stability.** The 15-state optimum was reached by 1 of 16 restarts, but it is
  47,600 log-likelihood units better than any of the other 15. Held-out likelihood was
  still rising at K = 15.
* **Proxy reliabilities.** Reliabilities for H3K27me3 (TEX) and H3K9me3 (TMEM, 2
  replicates) come from cell states other than TN, and state-level corrections use the
  least reliable enriched mark as a proxy.
* **Endpoints only.** All conditions are endpoints; the ordering analysis describes
  predictiveness across a 30-day step, not timing or causation.
