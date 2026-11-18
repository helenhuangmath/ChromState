# Benchmarks

ChromState was compared with the published ChromHMM 1.28, EpiCSeg 1.0, normR 1.36,
diHMM and Segway 3.0.7 software. Every tool was run on the same bins. The harness is in
`tools/toolbench`, and the figures are drawn by `tools/make_paper_figures.py` from the
saved result tables.

EpiSegMix, epigenome-ssm and chromstaR have not been run yet.

## Summary

| Metric (chromosome 19) | ChromState | ChromHMM | EpiCSeg | Raw signal |
|---|---|---|---|---|
| Simulation: state recovery (ARI) | 0.84 | 0.74 | 0.67 | – |
| CD8 CUT&RUN: active promoters (AUPRC) | 0.24 | 0.19 | 0.25 | – |
| CD8 CUT&RUN: expression change (Spearman) | 0.36 | 0.23 | 0.30 | 0.34 |
| ENCODE ChIP-seq: active promoters (AUPRC) | 0.41 | 0.39 | 0.43 | – |
| ENCODE ChIP-seq: expression change (Spearman) | 0.49 | 0.50 | 0.53 | 0.51 |
| Run time, CD8 / ENCODE (s) | 25 / 35 | 65 / 81 | 51 / 59 | – |

How to read this:

- **Simulation favours ChromState.** The simulated counts are drawn from a
  negative-binomial model, the family ChromState fits.
- **On real data ChromState and EpiCSeg are close.** EpiCSeg finds active promoters
  slightly better in both data sets, and the raw mark signal predicts expression about
  as well as any tool's states.
- **Speed.** ChromState is the fastest tool on one chromosome with three samples. A
  full fit of 16 samples on five chromosomes took 3.2 hours.

## Simulated data

Nine methods on 200,000 simulated bins with seven true states. Tools marked * are
reimplementations of the published model.

![Figure 1, state recovery](../_generated/Fig1_a-c_state_recovery.png)

![Figure 1, perturbation calibration](../_generated/Fig1_d-e_perturbation_calibration.png)

![Figure 1, run time](../_generated/Fig1_f_run_time.png)

## Real data

Six tools on CD8 T-cell CUT&RUN (4 marks, 15 states) and ENCODE ChIP-seq (GM12878, K562,
H1; 6 marks, 18 states), scored on genes and RNA-seq that the tools never saw.

![Figure 2, cd8 tcells](../_generated/Fig2_a-c_cd8_tcells.png)

![Figure 2, encode](../_generated/Fig2_d-f_encode.png)

![Figure 2, tool agreement](../_generated/Fig2_g-h_tool_agreement.png)

![Figure 2, roadmap agreement](../_generated/Fig2_i_roadmap_agreement.png)

Agreement with the Roadmap 18-state annotation is lower for ChromState (ARI 0.23–0.26)
than for ChromHMM (0.33–0.44). That annotation was made with ChromHMM.

## Condition comparison with replicates

One 15-state model on 16 CD8 T-cell samples. The contrast between two batches of naive
cells is the null: it calls 0.015% of tiles, against 26.7% for exhausted versus naive.
Genome-wide state shares differ between the two naive batches by up to about 2
percentage points, so shifts of that size are not evidence of a condition effect.

![Figure 3, states and changes](../_generated/Fig3_a-d_states_and_changes.png)

![Figure 3, change per tile](../_generated/Fig3_e_change_per_tile.png)

![Figure 3, confident changes](../_generated/Fig3_f_confident_changes.png)

## Expression prediction

Whole-gene state tables predict expression level at Spearman 0.80 (CD8) and 0.83
(ENCODE), and expression change at 0.53 and 0.68. A hard state label loses most of the
information about change.

![Figure 4, predicted vs measured](../_generated/Fig4_a_predicted_vs_measured.png)

![Figure 4, representations](../_generated/Fig4_b-c_representations.png)

![Figure 4, importance cd8](../_generated/Fig4_d_importance_cd8.png)

![Figure 4, importance encode](../_generated/Fig4_e_importance_encode.png)

![Figure 4, gene part](../_generated/Fig4_f_gene_part.png)

## External validation on ENCODE

STARR-seq activity is enriched up to about 9-fold in the active promoter states. For
CRISPRi-validated K562 enhancers, the raw H3K27ac signal (AUPRC 0.30) is not improved
on by the ChromState scores (0.28), and the states named as enhancers rank below
chance.

![Figure 5, states and reporter activity](../_generated/Fig5_a-c_states_and_reporter_activity.png)

![Figure 5, segmentation agreement](../_generated/Fig5_d_segmentation_agreement.png)

![Figure 5, crispri enhancers](../_generated/Fig5_e_crispri_enhancers.png)

![Figure 5, posterior calibration](../_generated/Fig5_f_posterior_calibration.png)

## Reproducing

```bash
python tools/run_tools.py --help          # run the external tools on a dataset
python tools/make_paper_figures.py        # redraw the figures from saved tables
```

Installation notes for the external tools are in `tools/README.md`. More detail on the
simulated comparison is in `docs/tool_comparison.md` in the repository.
