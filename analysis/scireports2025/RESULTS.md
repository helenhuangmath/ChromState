# Results: quantitative chromatin states and gene expression in CD8 T-cell differentiation


Copies of the figures are included in [the example figure gallery](../../docs/figures/README.md#cd8-t-cell-application).

Data: Huang H, Baxter AE, *et al.*, *Scientific Reports* 15, 17359 (2025). The data are
CUT&RUN for H3K4me3, H3K27ac, H3K27me3 and H3K9me3, plus RNA-seq, in P14 CD8 T cells:
naive (TN), memory (TMEM, LCMV Armstrong day 30) and exhausted (TEX, LCMV clone 13
day 32). The methods and code are in [`README.md`](README.md) and
[Tutorial 2](../../docs/tutorials/02_cd8_tcell_application.md). The figures are in
`out/figures/`.

---

## 1. A quantitative chromatin state map of naive, memory and exhausted CD8 T cells

A single 15-state model was fitted jointly to all three cell states. Fifteen states were
chosen by held-out likelihood, matching the published ChromHMM model. States were named
from their fold enrichment over the genome-wide average of each mark (Fig. 1a):

| Group | States | Defining enrichment | Share of the TN genome |
|---|---|---|---|
| Active promoters | Active TSS; flanking TSS | H3K4me3 26×, H3K27ac 29×; H3K27ac 6×, H3K4me3 4× | 3.8% |
| Bivalent promoters | Bivalent TSS | H3K4me3 14×, H3K27me3 8× | 0.5% |
| Polycomb | Polycomb repressed (2 states); weakly Polycomb repressed | H3K27me3 4–10×; 1.8× | 19.1% |
| H3K9me3 | Heterochromatin; three weak-heterochromatin states | H3K9me3 19×; 1.5–3.2× | 37.8% |
| Background | Five states | no mark ≥ 1.5× | 38.8% |

Promoter state explains expression level as expected (Fig. 1d):
- **Active TSS promoters** carry the most highly expressed genes (median log2 expression
  ≈ 10).
- **Flanking-TSS and bivalent promoters** are intermediate.
- **Polycomb promoters** carry the least expressed genes.

Genome-wide state occupancy is similar in the three cell states (Fig. 1b). Differentiation
redistributes states at specific loci rather than remodelling the genome globally. Two
shifts stand out:
- the Polycomb states expand in TMEM (6.6% → 8.9% for Polycomb repressed (1));
- the heterochromatin state nearly doubles in TEX (0.09% → 0.17%).

---

## 2. Quantitative states predict expression better than discrete states

Expression was predicted from chromatin features with gradient boosting and scored on
held-out chromosomes (Fig. 2). Values are Spearman ρ; the 95% CI is about ±0.005 for
level and ±0.015 for change.

| | ChromState quantitative | + raw signal | Raw signal | ChromHMM composition | ChromHMM state at TSS |
|---|---|---|---|---|---|
| Expression level, TN | 0.821 | **0.824** | 0.802 | 0.793 | 0.632 |
| Expression change, TMEM vs TN | 0.511 | **0.549** | 0.529 | 0.407 | 0.162 |
| Expression change, TEX vs TN | 0.553 | **0.581** | 0.542 | 0.442 | 0.179 |
| Expression change, TEX vs TMEM | 0.432 | **0.468** | 0.411 | 0.293 | 0.083 |

**Interpretation.**
- **Graded chromatin, graded expression.** Representing chromatin as a graded state
  composition rather than a single label recovers most of the information that
  discrete segmentation discards. The gain is largest for expression *change*, where ρ
  rises from about 0.17 (state at the TSS) to 0.55. Quantitative changes in chromatin,
  not switches between named states, accompany most transcriptional change during
  differentiation.
- **States and signal are complementary.** Adding raw signal to the state composition
  improves every change task, so the two carry complementary information. Single-mark
  models show that H3K27ac change tracks expression change most closely (ρ ≈ 0.52),
  followed by H3K27me3 and H3K4me3. H3K9me3 change barely tracks it (ρ < 0.09).
- **Naive chromatin anticipates later change.** Naive chromatin alone predicts later
  expression change modestly (ρ 0.19 for TMEM, 0.26 for TEX). Combined with naive
  expression it reaches ρ 0.27 and 0.34. Part of the transcriptional programme of
  differentiation, especially towards exhaustion, is already encoded in the naive
  chromatin state.

---

## 3. Chromatin state at marker genes tracks cell identity (Fig. 6)

**How a gene's state is measured.**
- **Bins.** The promoter (TSS ± 1 kb) is ten 200 bp bins. For each bin, the model gives
  a probability for each of the 15 states from the bin's four read counts and its
  neighbours.
- **Gene composition.** The gene's promoter composition is the **mean of the ten bins'
  probabilities**: 15 numbers summing to 1. A gene therefore usually has several
  states. *Tox* in TN is 70% active TSS (its central bins) and 30% flanking TSS (the
  bins downstream of the TSS).
- **Display classes.** Fig. 6a sums the 15 states into five classes (active = active +
  flanking TSS; bivalent; Polycomb; H3K9me3; background). This grouping hides
  magnitude, so Fig. 6b gives each mark's promoter signal directly (mean reads per bin).
- **Not statistically tested per gene.** The statements below are descriptive readings
  of Fig. 6. Pooled chromatin gives no replicates for per-gene tests; the statistical
  support comes from the genome-wide analyses in sections 2, 4, 5 and 6.

The promoter state composition of canonical marker genes reproduces the known biology
of CD8 T-cell differentiation, gene by gene. Read in the figure as stacked state shares
for TN, TMEM and TEX:

- **Naive genes are silenced by Polycomb.**
  - *Sell* and *Ccr7* change from fully active promoters in TN (0.99–1.00) to bivalent
    or Polycomb-repressed promoters in TMEM and TEX.
  - Their expression falls 20- to 60-fold (*Sell* log2 15.6 → 9.7 in TEX).
  - Silencing of naive-homing genes is therefore not just loss of activation, but
    acquisition of H3K27me3.
- **Memory retains the stem-like programme, exhaustion places it in a poised state.**
  - *Tcf7*, *Lef1* and *Il7r* keep active promoters in TMEM.
  - In TEX, *Tcf7* (0.86) and *Lef1* (1.00) become bivalent, and expression falls.
  - This matches the progressive loss of TCF1-driven stemness in exhaustion. In bulk
    TEX, the bivalent call may partly reflect a mixture of TCF1⁺ progenitor and
    terminally exhausted cells.
- **Effector genes are activated from repressed or poised promoters.**
  - *Klrg1* changes from fully Polycomb-repressed in TN to fully active in TMEM and TEX
    (expression about 145-fold higher).
  - *Id2* and *Gzmb* change from bivalent or background to active.
- **Some exhaustion genes carry weak active marking in naive cells that is later
  amplified.**
  - *Pdcd1* and *Tigit* are barely expressed in TN (log2 ≈ 5), yet their promoters
    already carry a focal, low-level active configuration. *Pdcd1* is mostly
    flanking TSS, with H3K4me3 48 and H3K27ac 13 reads per bin; *Tigit* has H3K4me3 15
    and H3K27ac 17.
  - In TEX the same promoters are fully active and 4- to 9-fold more strongly marked
    (*Pdcd1* H3K4me3 215, H3K27ac 116; *Tigit* 121 and 75). Expression rises about
    100-fold (log2 12–13).
  - These promoters are primed rather than active in naive cells. Exhaustion amplifies
    a pre-existing mark rather than creating one de novo, consistent with the
    genome-wide ordering result for active promoter states (section 6).
  - In contrast, *Havcr2* starts from near-background marking (H3K4me3 7) and is
    activated de novo.
  - **Genome-browser tracks (Fig. 7)** show what the promoter averages compress.
    - In TN, both genes carry only a focal H3K4me3/H3K27ac peak at the TSS.
    - In TMEM, and much more in TEX, the active domain **spreads**: H3K4me3 and
      H3K27ac extend across the gene body and upstream region, and the bins change
      from background to active or flanking TSS across most of the locus.
    - H3K27me3 and H3K9me3 stay near background (≤ 14 and ≤ 5 reads per bin) in all
      three states. These genes are therefore activated by gain and spreading of active
      marks, not by release from Polycomb or H3K9me3 repression.
- **Bivalent states resolve towards the cell-type programme.**
  - *Tox* is bivalent in TMEM but active in TEX, where it is most expressed. The
    exhaustion master regulator is held in a poised, H3K27me3-marked configuration in
    memory cells.
  - *Eomes* resolves from bivalent in TN to fully active in TEX.
  - *Il2ra* and *Epas1* resolve to active in TMEM but stay bivalent in TEX.

---

## 4. Bivalent promoters in naive cells anticipate later activation (Fig. 3)

ChromState's continuous bivalency score agrees with the peak-based bivalent promoter
calls of the original study (AUROC 0.92; Fig. 3a). Among the 5,068 expressed genes with
bivalent TN promoters, the balance between the two marks predicts the later expression
change (Fig. 3b):

| TN bivalent promoters | TMEM vs TN: median log2FC | TMEM: beyond expression-matched genes | TEX vs TN: median log2FC | TEX: beyond expression-matched genes |
|---|---|---|---|---|
| H3K27me3-dominant (Q1) | +0.63 | **+0.17** | +1.32 | **+0.45** |
| Q2 | +0.04 | +0.03 | +0.06 | +0.06 |
| Q3 | 0.00 | 0.00 | +0.02 | +0.03 |
| H3K4me3-dominant (Q4) | −0.01 | 0.00 | +0.01 | +0.03 |

**Interpretation.**
- **Poised promoters are activated later.** Bivalent promoters dominated by H3K27me3 in
  naive cells are the ones whose genes are activated later, most strongly in
  exhaustion: 61% rise more than expression-matched controls.
- **About two thirds of the raw effect is regression to the mean.** Lowly expressed
  genes have the most room to rise. The remainder is a property of the poised promoter
  itself.
- **This supports the original study.** Promoters held in a repressed-but-poised
  configuration in TN are a reservoir for activation in memory and exhaustion.

---

## 5. Which chromatin states matter, and do marks act additively or combinatorially? (Fig. 5)

**Unique contribution of each state** (drop in held-out R² when the state is permuted):

| State | Expression level (TN) | Change, TMEM vs TN | Change, TEX vs TN |
|---|---|---|---|
| Active TSS | **0.347** | **0.416** | **0.415** |
| Bivalent TSS | 0.022 | 0.077 | 0.091 |
| Flanking TSS | 0.083 | 0.056 | 0.077 |
| Polycomb repressed (2) | 0.005 | 0.041 | 0.077 |
| Weakly Polycomb repressed | 0.002 | 0.040 | 0.041 |
| Background states (4) and (5) | 0.129, 0.084 | ≤ 0.005 | ≤ 0.004 |
| Heterochromatin and weak heterochromatin | ≤ 0.002 | ≤ 0.005 | ≤ 0.004 |

**Additive versus combinatorial** (Spearman ρ): best single mark (H3K27ac) 0.74 / 0.51 /
0.52; all marks, additive 0.78 / 0.51 / 0.52; all marks, interacting 0.80 / 0.53 / 0.54;
ChromState states 0.82 / 0.51 / 0.55 (level / TMEM change / TEX change). Interactions
add Δρ +0.016 to +0.023 over the additive model, with all 95% CIs above 0.

**Interpretation.**
- **One dominant state.** The active-promoter state is the single dominant determinant
  of both expression level and expression change.
- **Change draws on poised and repressed promoters.** Predicting *change* relies in
  addition on bivalent, flanking-TSS and Polycomb promoters: the poised compartments
  from which genes are activated or silenced during differentiation.
- **Level draws on background states.** Two background states matter for *level* only;
  they separate lowly expressed from silent genes.
- **H3K9me3 contributes nothing unique** at promoters.
- **Mostly additive, partly combinatorial.** Chromatin acts on expression mostly
  additively, with a modest but significant combinatorial component. The informative
  combinations are those that define poised chromatin:
  - **H3K4me3 × H3K27me3** (bivalency; Δρ +0.014 for level, +0.010 for TMEM);
  - **H3K27ac × H3K27me3** (active-versus-Polycomb balance; Δρ +0.014 for TEX).

---

## 6. Do chromatin changes precede or follow transcription? (Fig. 4)

TN cells are the precursors of TMEM and TEX, so a cross-lagged test can ask which layer
is predictive across the TN → TMEM/TEX step:
- whether TN chromatin predicts later expression beyond TN expression (chromatin leads);
- or whether TN expression predicts later chromatin beyond TN chromatin (RNA leads).

The regressions were **corrected for measurement error** using reliabilities measured
from replicates:

| Signal | Reliability of the pooled track |
|---|---|
| RNA | 0.999 |
| H3K4me3 | 0.994 |
| H3K27ac | 0.993 |
| H3K27me3 | 0.93 |
| H3K9me3 | 0.67 |

H3K9me3 promoter signal is the least reproducible, so the correction matters most for it.

**Results after correction:**

| Leans chromatin first | Leans RNA first | No clear order |
|---|---|---|
| Active-TSS and flanking-TSS promoter states (TEX); H3K27ac in the gene body (TMEM, TEX) | H3K4me3 at promoter and gene body (TMEM); H3K9me3 at promoter (TMEM, TEX) and gene body (TEX); Polycomb promoter states (TMEM) | H3K27me3 at promoter and gene body; bivalent TSS; H3K27ac at promoter |

**Interpretation.**
- **Active promoter chromatin primes exhaustion.** Active promoter configurations in
  naive cells anticipate expression changes towards exhaustion. The marker genes show
  this at single genes: *Pdcd1* and *Tigit* carry weak active marking in TN that is
  amplified 4- to 9-fold as their expression rises (section 3).
- **H3K4me3 follows transcription.** This fits its deposition by transcription-coupled
  machinery.
- **H3K9me3 gain follows transcription.** Genes highly expressed in TN gain promoter
  H3K9me3 later. This offers a mechanistic reading of the original observation that
  narrow H3K9me3 peaks associate with increased expression in TEX: H3K9me3 at these
  genes may be a consequence of their activity rather than a cause of repression.
- **The exception is rare.** The rare high-signal heterochromatin state (0.1% of the
  genome, which also carries H3K27ac and H3K4me3) leans the other way; it is too small
  to interpret on its own.
- **Polycomb in memory follows transcription.** Polycomb promoter states in TMEM also
  follow transcription, consistent with H3K27me3 being laid down on genes, such as
  *Sell* and *Ccr7*, after they are switched off.
- **Bivalency does not lead on its own.** After correction, H3K27me3 and the bivalent
  state show no clear genome-wide lead. The bivalent anticipation in section 4 holds
  *within* bivalent promoters and depends on mark balance, but does not extend to
  H3K27me3 as a genome-wide predictor.

**Strength of evidence.** All effects are small (≤ 0.10 SD). They come from two time
points, so they describe which layer is predictive across a 30-day step, not timing or
causation. State-level corrections use the reliability of each state's least reliable
mark as a proxy. The reliabilities of H3K27me3 and H3K9me3 were estimated in TEX and
TMEM respectively, where replicates were available.

---

## 7. Summary

1. **Quantitative states improve prediction.** Quantitative chromatin states predict
   expression and its change during CD8 T-cell differentiation better than discrete
   states, and complement raw mark signal.
2. **One dominant state, plus poised compartments.** The active promoter state carries
   most of the information. Expression *change* additionally depends on poised
   (bivalent, flanking) and Polycomb promoters.
3. **Mostly additive, partly combinatorial.** Marks act largely additively. The
   combinatorial contribution sits in the bivalent (H3K4me3 × H3K27me3) and
   active-versus-Polycomb (H3K27ac × H3K27me3) combinations.
4. **Naive chromatin carries a partial blueprint.** Weakly marked, primed promoters
   (*Pdcd1*, *Tigit*) and H3K27me3-dominant bivalent promoters (*Eomes*) anticipate
   later activation.
5. **Some marks follow transcription.** H3K4me3, H3K9me3 and Polycomb in memory
   behave as consequences of transcriptional change rather than causes.

**Recommended follow-up:**
- Per-replicate chromatin counts from the BAM files, to measure reliability for every
  mark and cell state.
- A chromatin and RNA time course during chronic infection.
- Targeted epigenome editing at primed promoters (e.g. *Pdcd1*, *Tox*) to test
  causality.

---

## Figure legends

**Figure 1 | A joint 15-state chromatin model of naive, memory and exhausted CD8 T
cells.**
- **a**, Emission profiles: log-scaled mean read count per 200 bp bin, divided by each
  mark's maximum. States are named from fold enrichment over the genome-wide average.
- **b**, Fraction of the genome in each state in TN, TMEM and TEX (log scale).
- **c**, Held-out log-likelihood per bin for 6–15 states, evaluated on whole held-out
  chromosomes. K = 15 was selected as the smallest K capturing 95% of the gain.
- **d**, TN expression of genes grouped by the dominant state of their promoter
  (TSS ± 1 kb). Boxes show median and interquartile range; whiskers extend to 1.5× IQR.

Active promoters carry the most expressed genes and Polycomb promoters the least.
Global state occupancy changes little between cell states.

**Figure 2 | Quantitative chromatin states predict gene expression and its change.**
Spearman ρ between observed values and out-of-fold predictions from gradient-boosted
models, with 5-fold cross-validation grouped by chromosome. Bars show 95% bootstrap CIs
(1,000 gene resamples).
- **a**, Expression level in each cell state.
- **b**, log2 fold change, predicted from the chromatin change.
- **c**, Later log2 fold change, predicted from TN chromatin and/or TN expression only.
- **d**, Single-mark models of expression change.

Graded state composition outperforms discrete state labels, and H3K27ac change tracks
expression change most closely.

**Figure 3 | Bivalent promoters in naive cells anticipate later activation.**
- **a**, ChromState bivalency score for TN promoters grouped by peak-called class (AUROC
  for bivalent vs all other promoters, 0.92).
- **b**, Median log2 fold change of genes with bivalent TN promoters, by quartile of
  H3K4me3 − H3K27me3. Open circles show raw values. Filled circles show the excess over
  genes of matched TN expression, which removes regression to the mean.

H3K27me3-dominant bivalent promoters are preferentially activated, most strongly in TEX.

**Figure 4 | Chromatin–transcription ordering across differentiation (cross-lagged
analysis).**
The asymmetry |chromatin → RNA| − |RNA → chromatin| of standardised cross-lagged
coefficients for TN → TMEM (orange) and TN → TEX (green), across 12,287 expressed genes.
Positive values mean TN chromatin predicts later expression beyond TN expression.
Negative values mean TN expression predicts later chromatin beyond TN chromatin.
- **a**, Histone marks at promoter and gene body.
- **b**, ChromState promoter states (fraction of the promoter in each state).

Filled symbols are corrected for measurement error using replicate-derived reliabilities;
open symbols are uncorrected background states. Bars show 95% bootstrap CIs.

**Figure 5 | Which states matter, and how marks combine.**
- **a**, Unique contribution of each state: drop in held-out R² when the state's
  promoter and body columns are permuted together.
- **b**, Held-out Spearman ρ for single marks, all marks additively (interactions
  forbidden), all marks with interactions, and ChromState states.
- **c**, Gain over the additive model when interactions are allowed within one pair of
  marks, or among all marks. Bars show paired 95% bootstrap CIs.

Light blue: expression level (TN). Mid blue: change, TMEM vs TN. Dark blue: change,
TEX vs TN.

**Figure 6 | Promoter chromatin state and expression of CD8 T-cell marker genes.**
- **a**, Promoter (TSS ± 1 kb) state composition in TN, TMEM and TEX: the mean of the
  ten 200 bp bins' posterior probabilities over the 15 states, summed into active
  (active + flanking TSS), bivalent, Polycomb, H3K9me3 and background classes.
- **b**, Promoter signal of each mark, as mean reads per bin. Colour is log-scaled to
  each mark's maximum (floor 20 reads). N, TN; M, TMEM; E, TEX.
- **c**, Expression, as log2 normalised count + 1.

Naive genes acquire Polycomb or bivalent marking as they are silenced. Effector and
exhaustion genes are activated from repressed or poised promoters. *Pdcd1* and *Tigit*
carry weak active marking in TN that is amplified 4- to 9-fold in TEX.

**Figure 7 | Genome-browser tracks of *Pdcd1*, *Tigit*, *Tox* and *Tcf7*.** One file
per gene: `Fig7_tracks_<gene>.pdf`.
- **Window:** the gene ± 5 kb.
- **Mark tracks:** mean reads per 200 bp bin for H3K4me3, H3K27ac, H3K27me3 and
  H3K9me3 in TN (blue), TMEM (orange) and TEX (green). The y-axis is shared within each
  mark; its maximum is shown at the top right.
- **RNA tracks:** pooled RNA-seq coverage.
- **State ribbons:** the most probable ChromState state of every bin.
- **Shading:** the TSS ± 1 kb promoter window.
- **Arrow:** the TSS and the direction of transcription.

Observations:
- ***Pdcd1* and *Tigit*.** A focal TN promoter peak expands into a broad active domain
  in TEX, without repressive marking in any state.
- ***Tox* (305 kb).**
  - In TMEM, a broad H3K27me3 domain covers the 5′ ~70 kb of the gene, so the promoter
    region is called Polycomb or bivalent.
  - In TEX that domain is gone, and large H3K27ac/H3K4me3 domains appear across the
    intronic region (about 100–200 kb downstream of the TSS). These are candidate
    exhaustion-specific intronic enhancers, and nearly the whole gene is called active
    or flanking TSS.
- ***Tcf7*.**
  - A broad active domain in TN narrows in TMEM.
  - In TEX, H3K27ac is lost and H3K27me3 covers the whole locus (up to ~140 reads per
    bin), so the bins are called Polycomb or bivalent.
  - Expression falls about 6-fold. In bulk TEX this may partly reflect the mix of
    TCF1⁺ progenitor and terminally exhausted cells.
