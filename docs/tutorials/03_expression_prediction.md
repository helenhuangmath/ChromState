# Tutorial 3 — Which chromatin states predict gene expression? (`chromstate predict`)

This tutorial runs the v0.3 prediction report on real data: CD8 T-cell CUT&RUN
(Huang, Baxter *et al.*, *Sci Rep* 2025; naive **Naive**, memory **Arm**, exhausted
**Cl13**; H3K4me3, H3K27ac, H3K27me3, H3K9me3) with the matched RNA-seq. It asks three
questions and answers each with a held-out number:

1. How well do quantitative chromatin states predict expression level, and its change
   from naive to exhausted cells?
2. Which state does it best, alone and as the irreplaceable part of the full model?
3. Does a vocabulary of *defined* states (section 5.11 of the README) predict as well as
   states learned without supervision?

Figures: [docs/figures/expression_prediction/](https://github.com/helenhuangmath/ChromState/tree/main/docs/figures/expression_prediction).
Code: [`tools/realdata/predict_cd8.py`](https://github.com/helenhuangmath/ChromState/blob/main/tools/realdata/predict_cd8.py); run with
`../testing/sbatch/predict_cd8.sbatch` on the cluster.

## 1. Data and run

Chromosomes 17, 18 and 19 (3.7 million bin-rows at 200 bp, three conditions), the same
pooled, DESeq2-scaled tracks rounded to counts as in Tutorial 2. Expression level is
log2(mean normalised count + 1) per condition; expression change is the DESeq2 log2 fold
change Cl13 versus Naive. Promoters are TSS ± 1 kb (`P_`), gene bodies the rest (`B_`).

```bash
sbatch --export=ALL,CHROMS=chr17+chr18+chr19,K=12,RUNTAG=chr17_19_k12 ../testing/sbatch/predict_cd8.sbatch
sbatch --export=ALL,CHROMS=chr17+chr18+chr19,K=12,RUNTAG=chr17_19_defs,EXTRA="--definitions default" ../testing/sbatch/predict_cd8.sbatch
```

Each job fits the model, quantifies every protein-coding gene (2,289 genes; 4,743 gene x
condition rows for the level task, 1,581 genes with a DESeq2 result for the change task),
and calls `predict_expression` twice. The same from the command line on a finished
`chromstate run`:

```bash
chromstate predict --genes run_out/genes/gene_quantification.tsv.gz --expression rna.tsv -o pred_level
chromstate predict --genes run_out/genes/gene_quantification.tsv.gz --lfc de_Cl13_vs_Naive.tsv \
                   --contrast Naive,Cl13 -o pred_change
```

Folds are grouped by chromosome; with only three chromosomes the package falls back to two
consecutive gene blocks per chromosome (it says so in a warning), so a gene is never
predicted from its own neighbourhood. Wall time was 17 min (unsupervised; 813 s of it the
fit with adaptive restarts) and 13 min (defined states).

## 2. How well does chromatin predict expression?

Held-out Spearman ρ of the full quantitative representation (occupancy, intensity, dose,
entropy and bivalency of every state in promoter and body):

| Task | Unsupervised K = 12 | Defined states (6 defined + 6 learned) |
|---|---|---|
| Expression level (4,743 gene x condition rows) | **0.774** (0.761–0.786), R² 0.64 | **0.778** (0.765–0.790), R² 0.63 |
| Expression change Cl13 vs Naive (1,581 genes) | **0.385** (0.334–0.435), R² 0.15 | **0.409** (0.359–0.457), R² 0.18 |

Representations, same genes and folds (unsupervised fit; `representation_scores.tsv`):

| Representation | Features | Level ρ | Change ρ |
|---|---|---|---|
| quantitative + raw signal | 88 | 0.780 | 0.432 |
| quantitative (ChromState) | 80 | 0.774 | 0.385 |
| occupancy only | 24 | 0.766 | 0.350 |
| raw mark signal, no states | 8 | 0.762 | 0.370 |
| hard top state (what a segmentation gives) | 24 | 0.652 | 0.126 |

Two things hold on both tasks and both fits. Hard labels lose most of the information
about change (ρ 0.13 against 0.39) and a fair amount about level. The quantitative states
are on a par with the raw marks for level, where any readout of H3K4me3/H3K27ac at the
promoter works, and ahead of them for change; adding the raw signal on top still helps for
change (0.43), so the states do not yet carry everything the marks do about a *shift*.

Gene parts (`part_ranking.tsv`): for level, body beats promoter (0.746 vs 0.657) and the
two together reach 0.780, which reflects H3K27ac/H3K4me3 spreading into the body of
highly expressed genes. For change the order flips (promoter 0.368, body 0.328, both
0.432).

## 3. Which state predicts best?

`state_importance.tsv` gives two numbers per state. `marginal_rho` is a model that sees
only that state; `unique_r2` is the held-out R² lost when that state's columns are
permuted inside the full model, i.e. what no other state can replace.

Unsupervised fit, expression level:

| State (rule) | marginal ρ | unique R² (95% CI) | direction |
|---|---|---|---|
| TssA (H3K27ac 24x, H3K4me3 23x) | 0.635 | **0.227** (0.208–0.245) | + |
| Quies_4 (no mark) | **0.648** | 0.041 | ~0 |
| TssFlnk (H3K27ac 4.7x, H3K4me3 3.0x) | 0.640 | 0.028 | + |
| ReprPC_1 (H3K27me3 4.1x) | 0.622 | 0.018 | − |

Expression change:

| State | marginal ρ | unique R² (95% CI) | direction |
|---|---|---|---|
| TssA | **0.362** | **0.163** (0.099–0.228) | + |
| every other state | ≤ 0.21 | ≤ 0.006, CI spans 0 | |

Reading the two columns together is the point. Several states predict *level* about
equally well on their own (ρ 0.62–0.65), including a quiescent state whose occupancy
simply says "no active mark here"; but only the active-promoter state is irreplaceable
(unique R² 0.23 against ≤ 0.04). For *change*, the active state is the only state with
signal at all: a gene's expression moves between naive and exhausted cells to the extent
that the amount and intensity of TssA at its promoter and body moves. Polycomb states
predict level negatively, as expected, but carry nothing unique about change on this
contrast.

## 4. Defined states versus learned states

The second run gave the model a vocabulary (`StateDefinitions.default`): TssA, TssBiv,
EnhA, ReprPC, Het and Quies as patterns over the four marks, with six more states free.
What happened is instructive.

- The six defined states were seeded, named and matched one-to-one (agreement +1.0 for
  five of them, +0.5 for EnhA, which needs H3K4me1 the panel does not have).
- EM then put the strongest active promoters into one of the *free* states
  (`TssA_x`: H3K27ac 25x, H3K4me3 23x, 1.6% of the genome) and left the defined `TssA`
  as a weaker active state (3.3%). Likewise `ReprPC_x` is the strong Polycomb state.
  With `anchor="seed"` the data are allowed to do this; `anchor="fixed"` would have
  pinned the defined means and absorbed the strong promoters into `TssA`.
- Prediction was equal or slightly better than the unsupervised fit (level 0.778 vs
  0.774, change 0.409 vs 0.385), and the most irreplaceable state was again the strong
  active state (`TssA_x`, unique R² 0.29 for level and 0.21 for change). `TssBiv` covered
  0.6% of the genome and was not among the predictive states; with pooled tracks and no
  H3K4me1 the bivalent promoters are few on these chromosomes.

So a defined vocabulary costs nothing in accuracy and makes the names stable across
datasets, with one caveat the `rule` column makes visible: under `seed` the defined name
marks the *basin EM started from*, not necessarily the strongest state of that kind. Use
`fixed` when the definition itself is the quantity to be measured, and read `states.tsv`
either way.

## 5. Region quantification

The run also writes `promoter_quantification.tsv.gz` from `quantify_regions`: for every
promoter and condition the fraction and base pairs in each state, the dominant state and
the composition entropy. Dominant-state counts over the 2,289 promoters (unsupervised fit):

| Condition | TssA | TssFlnk | ReprPC (both) | Het (all) |
|---|---|---|---|---|
| Naive | 1,078 | 69 | 526 | 201 |
| Arm | 996 | 101 | 564 | 189 |
| Cl13 | 1,029 | 76 | 498 | 131 |

## 6. What this does and does not show

- Chromatin input is one pooled track per condition, so no differential test is run here
  (`differential_report` needs two replicates per condition); the per-replicate BAMs noted
  in `analysis/scireports2025/README.md` would enable it.
- All numbers are associations on held-out genes. They say which state *carries* the
  information about expression, not that the state causes it.
- Three chromosomes were used to keep the fit at minutes. The chromosome-19-only run
  fails by design: cross-validation needs more than one chromosome, or the block fallback.
