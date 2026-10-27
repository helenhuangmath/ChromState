# Tutorial 2 — Application: chromatin states and gene expression in CD8 T-cell differentiation


Pre-generated [CD8 T-cell example figures](../figures/README.md#cd8-t-cell-application) are included for browsing.

This tutorial applies ChromState to primary mouse CD8 T cells in three differentiation
states. It works through each analysis step: building the chromatin state model, linking
it to RNA-seq, and asking four biological questions.

1. Does quantitative chromatin state predict gene expression, and its change between
   cell states, better than discrete states or raw signal?
2. Do bivalent promoters in naive cells anticipate later expression?
3. Which chromatin states matter most, and do marks act additively or in combination?
4. Do chromatin changes tend to precede or follow transcriptional change?

All code is in [`analysis/scireports2025/`](../../analysis/scireports2025/). The figures
are Nature-style PDFs written to `analysis/scireports2025/out/figures/`.

---

## 1. Data

Huang H, Baxter AE, *et al.* Deciphering the role of histone modifications in memory and
exhausted CD8 T cells. *Scientific Reports* **15**, 17359 (2025).
GEO: [GSE285245](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE285245) (CUT&RUN),
[GSE285248](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE285248) (RNA-seq).

| Cell state | Abbreviation | Infection and day | Chromatin (CUT&RUN) | RNA-seq |
|---|---|---|---|---|
| Naive P14 CD8 T cells | TN | Uninfected; harvested with each infected group (days 30 and 32) | H3K4me3, H3K27ac, H3K27me3, H3K9me3 | 8 replicates (4 per day) |
| Memory | TMEM | LCMV Armstrong, day 30 | same | 4 replicates |
| Exhausted | TEX | LCMV clone 13, day 32 | same | 4 replicates |

Genome: mm10.

**Input used here.** Aligned reads were not available locally, so chromatin input is the
replicate-pooled, DESeq2-scaled coverage tracks (`pool_<mark>_<condition>.bw`),
averaged into 200 bp bins and rounded to integers for the negative-binomial model. RNA
input is the raw featureCounts table. The published 15-state ChromHMM segmentation of the
same data is used as a baseline. Section 9 explains what pooled input does and does not
allow.

---

## 2. Running the analysis

```bash
cd analysis/scireports2025
export PYTHONPATH=../../src:.
export CHROMSTATE_SCIREP_DATA=/path/to/CutRun_mm10      # processed-data folder

python s01_bin_tracks.py      # coverage tracks -> 200 bp bins               ~8 min
python s02_fit_model.py       # choose K, fit joint TN/TMEM/TEX model         ~30 min
python s03_rna.py             # differential expression (pydeseq2)            ~1 min
python s04_gene_features.py   # decode genome, summarise every gene           ~6 min
python s05_predict.py         # expression-prediction benchmark               ~40 min
python s06_bivalency.py       # bivalency vs published poised promoters       ~1 min
python s07_figures.py         # Figs 1-3
python s08_importance.py      # state importance; additive vs combinatorial   ~25 min
python s09_cross_lag.py       # chromatin-first vs RNA-first test             ~2 min
python s11_reliability.py     # chromatin/RNA reliability from replicate tracks ~1 min
python s09_cross_lag.py       # rerun: ordering corrected for measurement error
python s10_figures_order.py   # Figs 4-5
python s12_marker_genes.py    # Fig 6
```

Results with interpretation and Nature-style figure legends are collected in
[`analysis/scireports2025/RESULTS.md`](../../analysis/scireports2025/RESULTS.md).

Shared paths and constants are in `common.py`, and the figure style in `nature_style.py`.

---

## 3. Building the chromatin state model (steps 1–2, 4; Fig. 1)

### 3.1 Binning

`s01_bin_tracks.py` averages each coverage track over 200 bp bins with
`chromstate.io.bin_bigwig`, an exact base-weighted mean. The output covers chromosomes
1–19 and X: 13.2 million bins per cell state × 4 marks.

### 3.2 Choosing the number of states

`s02_fit_model.py` fits candidate models on chromosomes 11, 17 and 19 (4.2 million
condition × bin rows). It scores each on whole chromosomes held out from fitting:

| K | Held-out log-likelihood per bin |
|---|---|
| 6 | −7.108 |
| 8 | −6.903 |
| 10 | −6.744 |
| 12 | −6.675 |
| 15 | −6.572 |

The rule chooses the smallest K that captures 95% of the held-out gain, which gave
**K = 15**, the same number of states as the published ChromHMM model. The final model
is refitted with adaptive restarts. One emission profile is shared by TN, TMEM and TEX,
so a state means the same thing in every cell state.

### 3.3 Decoding and naming states

`s04_gene_features.py` applies the fitted model to every chromosome and names each
state from its **fold enrichment**: the state's mean read count for a mark divided by the
genome-wide average for that mark (occupancy-weighted, TN). "H3K4me3 26×" means the
state's bins carry 26 times the genome-average H3K4me3. A mark counts as present at
≥ 1.5× and as strong at ≥ 4×. The figure labels list every mark at ≥ 1.5×.

| State | Enriched marks | Share of genome in TN |
|---|---|---|
| Active TSS | H3K4me3 26×, H3K27ac 29× | 1.3% |
| Bivalent TSS | H3K4me3 14×, H3K27me3 8×, H3K27ac 4× | 0.5% |
| Flanking TSS | H3K27ac 6.0×, H3K4me3 3.8× | 2.4% |
| Polycomb repressed (2) | H3K27me3 10×, H3K4me3 2× | 2.8% |
| Polycomb repressed (1) | H3K27me3 4.2× | 6.6% |
| Weak heterochromatin (3) | H3K9me3 3.2× | 16.5% |
| Heterochromatin (high signal) | H3K9me3 19×, H3K27ac 7.8×, H3K4me3 4.9× | 0.1% |
| Weakly Polycomb repressed | H3K27me3 1.8× | 9.7% |
| Weak heterochromatin (1) and (2) | H3K9me3 1.5–1.9× | 21.2% |
| Five background-level states | none ≥ 1.5× | 38.8% |

**Fig. 1** shows the log-scaled emissions, genome occupancy in each cell state, the
model-selection curve, and TN expression of genes grouped by dominant promoter state.
Active-TSS promoters carry the most highly expressed genes, and Polycomb promoters the
least.

### 3.4 Per-gene features

For each of ~54,000 annotated genes in each cell state, step 4 summarises two windows:
the **promoter** (TSS ± 1 kb) and the **gene body** (TSS + 1 kb to TES). It records:
- the posterior state composition (15 values per window);
- state entropy;
- posterior- and signal-level bivalency;
- mean log signal per mark;
- the ChromState state at the TSS;
- the published ChromHMM state at the TSS and its composition over each window.

All feature families come from the same genes and bins, so later comparisons are
between representations, not between preprocessing pipelines.

---

## 4. Gene expression (step 3)

`s03_rna.py` runs pydeseq2 on the raw counts.

| Contrast | Genes with \|log2FC\| > 1 and padj < 0.05 |
|---|---|
| TMEM vs TN | 3,925 |
| TEX vs TN | 5,405 |
| TEX vs TMEM | 2,236 |
| TN day 32 vs TN day 30 (batch reference) | 390 |

TN cells were harvested on two days. The 390 genes that differ between them set a
practical floor for what counts as a biological change.

---

## 5. Question 1 — Does chromatin state predict expression? (step 5; Fig. 2)

**Design.**
- **Model:** gradient-boosted regression.
- **Validation:** five-fold cross-validation **grouped by chromosome**, so neighbouring,
  co-regulated genes never fall on both sides of a split.
- **Score:** Spearman ρ of out-of-fold predictions, with 95% CIs from 1,000 gene
  bootstraps.

There are three tasks:
- **Level:** expression in one cell state from that cell state's chromatin.
- **Change:** log2 fold change from the change in chromatin between the two cell states.
- **Prepattern:** later log2 fold change from TN chromatin only.

Each task is run with six feature families:

| Feature family | Content |
|---|---|
| Raw signal | Mean log signal per mark, promoter and body |
| ChromHMM label at TSS | Published 15-state call at the TSS |
| ChromHMM composition | Fraction of promoter and body in each ChromHMM state |
| ChromState label at TSS | ChromState Viterbi state at the TSS |
| **ChromState quantitative** | Posterior composition, entropy and bivalency, promoter and body |
| ChromState + raw | Both |

**Results** (Spearman ρ; 95% CI half-width about 0.005 for level and 0.015 for change):

| Task | ChromState quantitative | ChromState + raw | Raw signal | ChromHMM composition | ChromHMM label | ChromState label |
|---|---|---|---|---|---|---|
| Level, TN | 0.821 | **0.824** | 0.802 | 0.793 | 0.632 | 0.600 |
| Level, TMEM | 0.810 | **0.815** | 0.792 | 0.789 | 0.611 | 0.598 |
| Level, TEX | 0.811 | **0.815** | 0.793 | 0.793 | 0.617 | 0.598 |
| Change, TMEM vs TN | 0.511 | **0.549** | 0.529 | 0.407 | 0.162 | 0.148 |
| Change, TEX vs TN | 0.553 | **0.581** | 0.542 | 0.442 | 0.179 | 0.157 |
| Change, TEX vs TMEM | 0.432 | **0.468** | 0.411 | 0.293 | 0.083 | 0.078 |

**Interpretation.**
- **Quantitative versus discrete.** Quantitative states predict both expression level
  and expression change far better than discrete state labels (Δρ ≈ +0.2 for level,
  +0.35 for change). They are also better than the published ChromHMM state composition.
- **Versus raw signal.** Against raw mark signal, ChromState is better for expression
  level (Δρ +0.019, CI 0.016–0.022) and for TEX vs TMEM (+0.022). It is comparable for
  TEX vs TN and slightly lower for TMEM vs TN (−0.018). Combining the two is best for
  every change task, so state composition carries information that the marks alone do
  not.
- **Which mark tracks change.** In single-mark models, change in H3K27ac tracks
  expression change most closely (ρ ≈ 0.52), followed by H3K27me3 (0.34–0.37) and
  H3K4me3 (0.30–0.36). H3K9me3 change barely tracks expression change (ρ < 0.09; Fig.
  2d).

**Prepattern** (later change from TN alone):

| Predictors | TMEM vs TN | TEX vs TN |
|---|---|---|
| TN expression only | 0.188 | 0.185 |
| TN chromatin (ChromState) only | 0.188 | 0.263 |
| TN expression + TN chromatin (ChromState) | 0.265 | **0.336** |

TN chromatin anticipates later expression change, especially towards exhaustion. It
adds to what TN expression level already predicts rather than duplicating it.

---

## 6. Question 2 — Do bivalent promoters in naive cells anticipate later expression? (step 6; Fig. 3)

The original study called TN promoter classes from peak overlap: bivalent, H3K4me3 only,
H3K27me3 only, or neither.

**Concordance.** ChromState's continuous bivalency score ranks the peak-called bivalent
promoters first (AUROC 0.92; Fig. 3a). Both are derived from the same data, so this
is a consistency check rather than independent validation.

**Anticipation.** The 5,068 expressed genes with bivalent TN promoters were divided into
quartiles by the balance of H3K4me3 and H3K27me3:

| TN bivalent quartile | Median log2FC, TMEM vs TN | Median log2FC, TEX vs TN | Beyond expression-matched genes, TMEM | Beyond expression-matched genes, TEX |
|---|---|---|---|---|
| Q1 (H3K27me3-dominant) | +0.63 | +1.32 | **+0.17** | **+0.45** |
| Q2 | +0.04 | +0.06 | +0.03 | +0.06 |
| Q3 | 0.00 | +0.02 | 0.00 | +0.03 |
| Q4 (H3K4me3-dominant) | −0.01 | +0.01 | 0.00 | +0.03 |

H3K27me3-dominant bivalent promoters in TN are the ones whose genes rise later, most
strongly in exhaustion (61% rise more than expression-matched controls). The
**expression-matched** column compares each gene with genome-wide genes of the same TN
expression level. Lowly expressed genes have the most room to rise, and this correction
shows that about two thirds of the raw effect is regression to the mean. The remainder
is a genuine property of the poised promoters. This supports the study's observation
that promoters poised in TN are later activated in TMEM or TEX.

---

## 7. Question 3 — Which states matter, and are marks additive or combinatorial? (step 8; Fig. 5)

### 7.1 State importance

A model of the full state composition is fitted, and each state is then permuted on
held-out data. The drop in R² is the state's **unique** contribution, the information
no other state can substitute for.

This analysis is now a package function. `chromstate.state_importance` takes the
per-gene composition from `gene_state_matrix` and a per-gene expression vector, and
returns for every state the permutation importance, the drop-one importance with the
remaining shares re-closed, a signed standardised coefficient (ridge) with bootstrap
intervals, and a rank; `state_importance_change` does the same for `delta_<state>`
columns against a log2 fold change, and `compare_representations` scores the quantitative
composition against hard state fractions, the TSS state and raw signal on the same genes.
From the command line, `chromstate fit --gtf genes.gtf --expression expr.tsv` writes a
`state_importance_<sample>.tsv` per sample.

```python
genes = cs.gene_state_matrix(seg, features, part="promoter")
chrom = features.table.set_index("gene_id")["chrom"]
level = cs.state_importance(genes[genes["sample"] == "TN"], log2_tpm["TN"], groups=chrom)
change = cs.state_importance_change(cs.gene_state_change(genes, ["TN"], ["TEX"]),
                                    deseq_tex_vs_tn, groups=chrom)
level.table[["state", "rank", "perm_importance", "perm_lo", "perm_hi", "coef_std", "direction"]]
```

Cross-validation is grouped by chromosome, and the result's `note` states the model,
the number of genes joined, and that the quantities are associations. The table below was
produced by the analysis script; its numbers are permutation losses in R².

| State | Level (TN) | Change, TMEM vs TN | Change, TEX vs TN |
|---|---|---|---|
| Active TSS | **0.347** | **0.416** | **0.415** |
| Bivalent TSS | 0.022 | 0.077 | 0.091 |
| Flanking TSS | 0.083 | 0.056 | 0.077 |
| Polycomb repressed (2) | 0.005 | 0.041 | 0.077 |
| Weakly Polycomb repressed | 0.002 | 0.040 | 0.041 |
| Background states (4) and (5) | 0.129, 0.084 | ≤ 0.005 | ≤ 0.004 |
| H3K9me3 and heterochromatin states | ≤ 0.002 | ≤ 0.005 | ≤ 0.004 |

- **One dominant state.** The active-TSS state carries most of the information in
  every task.
- **Change depends on poised and repressed states.** Predicting *change* relies
  additionally on bivalent and Polycomb promoters, the poised compartments from which
  genes are activated or silenced.
- **Level uses background states.** Expression *level* also draws on two
  background-level states, which separate lowly expressed from silent genes.
- **H3K9me3 adds nothing unique.**

### 7.2 Additive versus combinatorial

Models on the four marks (promoter and body) are compared under three conditions:
- marks combined only **additively** (interactions forbidden);
- with interactions allowed **within one pair** of marks;
- with **all interactions** allowed.

ChromState states encode combinations by construction.

| Model | Level (TN) | Change, TMEM vs TN | Change, TEX vs TN |
|---|---|---|---|
| H3K27ac alone | 0.738 | 0.509 | 0.521 |
| All marks, additive | 0.782 | 0.512 | 0.519 |
| All marks, interacting | 0.802 | 0.529 | 0.542 |
| ChromState states | **0.815** | 0.505 | **0.548** |

| Gain over additive (Δρ, 95% CI) | Level (TN) | TMEM vs TN | TEX vs TN |
|---|---|---|---|
| H3K4me3 × H3K27me3 | **+0.014** (0.012–0.016) | **+0.010** (0.004–0.016) | +0.004 (−0.001–0.009) |
| H3K27ac × H3K27me3 | +0.012 (0.010–0.014) | −0.001 (−0.007–0.005) | **+0.014** (0.008–0.021) |
| H3K4me3 × H3K27ac | +0.009 | +0.008 | +0.007 |
| All interactions | +0.019 | +0.016 | +0.023 |
| ChromState states | +0.033 | −0.008 (n.s.) | +0.030 |

Chromatin acts on expression **mostly additively**, with a modest but significant
combinatorial component. The informative combinations are those that define poised
chromatin:
- **H3K4me3 × H3K27me3**, the bivalent configuration, which matters most for expression
  level and for TN → TMEM;
- **H3K27ac × H3K27me3**, the active-versus-Polycomb balance, which matters most for
  TN → TEX.

---

## 8. Question 4 — Does chromatin change precede or follow transcription? (step 9; Fig. 4)

TN cells are the precursors of TMEM and TEX, so TN → later is a real step in
differentiation. For each mark or state *V*, all variables are rank-normalised and two
regressions are fitted across 12,287 expressed genes:

- **chromatin → RNA:** RNA_later ~ RNA_TN + **β**·V_TN. Does TN chromatin predict later
  expression beyond TN expression?
- **RNA → chromatin:** V_later ~ V_TN + **γ**·RNA_TN. Does TN expression predict later
  chromatin beyond TN chromatin?

The asymmetry |β| − |γ| (x-axis of Fig. 4) indicates which way the relationship leans:
- above 0: chromatin leads;
- below 0: RNA leads;
- a call is made only if the 95% bootstrap CI excludes 0.

For a ChromState state (Fig. 4b), *V* is the fraction of the promoter (TSS ± 1 kb)
occupied by that state.

**Correcting for measurement error.** A noisy predictor attenuates its own slope and
leaves the other predictor to absorb what it failed to control for. With correlated
predictors this bias can go either way. Step 11 therefore measures reliability from the
per-replicate tracks available locally: the mean correlation between single replicates,
converted to the reliability of the pooled track by the Spearman–Brown formula.

| Signal | Single replicate r | Pooled-track reliability |
|---|---|---|
| RNA (TN, 8 replicates) | 0.995 | 0.999 |
| H3K4me3 (TN) | 0.956 | 0.994 |
| H3K27ac (TN) | 0.949 | 0.993 |
| H3K27me3 (TEX; proxy) | 0.766 | 0.929 |
| H3K9me3 (TMEM, 2 replicates; proxy) | 0.340 | 0.673 |

The regressions are then refitted as errors-in-variables regressions: the error variance
(1 − reliability) is removed from the predictor covariance before solving. States use
the reliability of their least reliable enriched mark. In Fig. 4, filled symbols are
corrected and open symbols (background states) are uncorrected.

**Results after correction:**

| Leans chromatin first | Leans RNA first | No clear order |
|---|---|---|
| Active-TSS and flanking-TSS promoter states (TEX); H3K27ac in the gene body (TMEM, TEX) | H3K4me3 promoter and gene body (TMEM); H3K9me3 promoter (TMEM, TEX) and gene body (TEX); Polycomb promoter states (TMEM) | H3K27me3 promoter and body; bivalent TSS; H3K27ac promoter |

- **Active promoter chromatin primes exhaustion.** Active promoter configurations in TN
  anticipate expression change towards exhaustion. In Fig. 6, *Pdcd1* and *Tigit* carry
  weak active marking in TN that is amplified 4- to 9-fold as their expression rises.
- **H3K4me3 follows transcription**, consistent with its transcription-coupled
  deposition.
- **H3K9me3 gain follows transcription.** This survives the correction despite its low
  reliability, and offers a reading of the study's observation that narrow H3K9me3
  peaks associate with *increased* expression in TEX.
- **Polycomb in memory follows transcription.** Polycomb states in TMEM follow
  transcription, as H3K27me3 is laid on genes such as *Sell* and *Ccr7* after they are
  switched off.
- **The correction changes the H3K27me3 result.** Before correction H3K27me3 appeared
  to lead (+0.022 promoter, +0.051 body, TEX); after it, there is no clear order. The
  bivalent anticipation of section 6 holds *within* bivalent promoters but does not make
  H3K27me3 a genome-wide predictor.

**Effect sizes are small** (|asymmetry| ≤ 0.10 SD).

### 8.1 Marker genes (step 12; Fig. 6)

For 25 canonical genes, Fig. 6 shows three panels in TN, TMEM and TEX:
- **a**, promoter state composition;
- **b**, each mark's promoter signal;
- **c**, expression.

**How a gene's state is measured.**
- The promoter (TSS ± 1 kb) is ten 200 bp bins, and each bin has a probability for
  every one of the 15 states.
- The gene's composition is the mean over the ten bins, so a gene usually carries
  several states. *Tox* in TN is 70% active TSS (central bins) and 30% flanking TSS
  (downstream bins). Each individual bin here is called with probability ≈ 1.0, so the
  mixtures arise mainly from different bins sitting in different states.
- Panel a sums the 15 states into five classes for display. That hides magnitude, which
  panel b restores.
- These are descriptive per-gene readings, not per-gene statistical tests.

- **Naive genes are silenced by Polycomb.** *Sell* and *Ccr7* go from fully active
  promoters in TN to bivalent or Polycomb-repressed promoters, with 20- to 60-fold lower
  expression.
- **Memory keeps, exhaustion poises.** *Tcf7*, *Lef1* and *Il7r* stay active in TMEM;
  *Tcf7* and *Lef1* become bivalent in TEX.
- **Effector genes are activated from repressed promoters.** *Klrg1* goes from fully
  Polycomb in TN to fully active in TMEM and TEX.
- **Primed exhaustion genes.** *Pdcd1* and *Tigit* (log2 expression ≈ 5 in TN) carry a
  focal, weak active configuration in TN (*Pdcd1* H3K4me3 48, H3K27ac 13 reads per bin).
  It is amplified 4- to 9-fold in TEX (H3K4me3 215, H3K27ac 116) as expression rises
  about 100-fold. *Havcr2* instead starts near background and is activated de novo.
- ***Tox* is held bivalent in memory.** Its H3K4me3 stays constant (53–67 reads), while
  H3K27me3 rises from 1 to 14 in TMEM and falls back in TEX, where *Tox* is most
  expressed.
- **Bivalent genes resolve.** *Eomes* resolves from bivalent in TN to active in TEX.

---

## 9. Scope and caveats

- **Pooled chromatin.** The analysis tracks are pooled. Replicate noise floors and
  region-level differential tests (`cs.differential_regions`) were therefore not run.
  Measurement reliability was estimated from the per-replicate tracks available
  locally. For H3K27me3 and H3K9me3 it comes from another cell state, and for states
  from their least reliable mark (section 8).
- **Endpoints, not a time course.** TMEM and TEX are sampled 30 days after TN. The
  ordering analysis describes which layer is predictive across that step; it does not
  measure timing or establish causality.
- **No spike-in.** Global differences in a mark between cell states are normalised away.
- **Model fit.** The best 15-state optimum was reached by 1 of 16 restarts but is 47,600
  log-likelihood units better than any other. The held-out likelihood was still rising
  at K = 15.

**Recommended next steps:**
- count per-replicate reads from the BAM files, which enables noise floors,
  differential testing and an unbiased ordering test;
- profile chromatin and RNA (ideally nascent RNA) at several time points during chronic
  infection;
- test candidate chromatin-first loci by targeted perturbation.

---

## 10. Adapting this analysis to your data

1. **Inputs.** Replace the paths in `common.py`: coverage tracks or, preferably, a
   ChromHMM-style manifest of BAM files (`cs.build_dataset`), and a raw gene-count
   table.
2. **Marks and conditions.** Set `MARKS`, `CONDITIONS` and `TRAIN_CHROMS`. Bivalency
   needs H3K4me3 and H3K27me3.
3. **Steps.** Run steps 1–12. Steps 5, 8 and 9 need only the per-gene feature table from
   step 4 and the RNA tables from step 3, so they work for any panel of marks and any
   number of conditions with a common precursor.
