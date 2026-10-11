# ChromState: quantitative chromatin state analysis

**Helen Huang** · Version 0.1.0 (research release) · Python package `chromstate`, R interface `chromstatR` · MIT licence

---

## Abstract

Chromatin states — combinations of histone modifications that mark active promoters,
enhancers, Polycomb-repressed and bivalent regions — are a powerful way to read the
regulatory genome. ChromState turns them from labels into **quantities**. Each genomic
region can then be described by how much of each state it carries, how confidently that
is known, how bivalent it is, and which mark drives the call. Every one of these numbers
can be compared directly between conditions and linked to gene expression.

ChromState models read counts with a hidden Markov model using negative-binomial
emissions, fitted jointly across all samples. Each state therefore means the same thing
in every condition, and sequencing depth or spike-in normalisation enters the model as an
offset rather than a rescaling of the data.

For every genomic bin, ChromState reports:
- the full state composition;
- call confidence and entropy;
- a continuous bivalency score resolved into magnitude and balance;
- an exact per-mark attribution of the evidence for each call.

Between conditions, it measures state flow and transport, sets a replicate-derived
noise floor, calibrates posteriors and tests regions for change. At the gene level, it
connects state composition to RNA-seq and traces any result back to its bins, marks and
read counts.

**Performance.**
- **Simulation with known ground truth.** Retaining signal magnitude substantially
  improves the detection of bivalent regions and of regions that respond to a
  perturbation, compared with segmentation of binarised signal. ChromState also gives
  the best-calibrated state probabilities of the models tested.
- **Primary CD8 T cells.** In naive, memory and exhausted cells, ChromState's
  quantitative states predict gene expression and its changes between cell states more
  accurately than discrete state labels, and they add information beyond raw mark
  signal.
- **Poised promoters.** Bivalent and Polycomb-marked promoters identified in naive
  cells anticipate genes that change expression later in differentiation.

ChromState is available as a Python package with an R interface for Bioconductor
workflows. It reads ChromHMM's input format unchanged and produces publication-ready
figures.

---

## Contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Quick start](#3-quick-start)
4. [Input data](#4-input-data)
5. [Standard workflow](#5-standard-workflow)
6. [Statistical model](#6-statistical-model)
7. [Benchmarks on simulated data](#7-benchmarks-on-simulated-data)
8. [Application: CD8 T-cell differentiation](#8-application-cd8-t-cell-differentiation)
9. [Comparison with external tools](#9-comparison-with-external-tools)
10. [Figures](#10-figures)
11. [Limitations](#11-limitations)
12. [Frequently asked questions](#12-frequently-asked-questions)
13. [Repository layout and development](#13-repository-layout-and-development)
14. [References](#14-references)
15. [Licence and contact](#15-licence-and-contact)

---

## 1. Overview

### 1.1 Motivation

Chromatin state segmentation (Ernst & Kellis, 2012; Hoffman et al., 2012) reduces the
joint distribution of histone modifications along the genome to a small number of
recurrent combinatorial patterns. This representation has become the standard way to
annotate regulatory elements, but it was designed to *classify* the genome, not to
*measure* it. Four properties of the conventional approach limit its use for
quantitative and comparative questions.

1. **Loss of dosage information.** Binarising each mark at a significance threshold
   maps every enriched bin onto the same observation. Regions differing several-fold in
   modification density become indistinguishable, and quantities defined by the
   *relative* abundance of two marks, such as bivalency, cannot be recovered.
2. **Degenerate posterior uncertainty.** HMM posteriors over states are usually
   concentrated near 0 or 1. They therefore carry little information about which calls
   are reliable, or about regions that genuinely lie between states.
3. **Compositional normalisation.** ChIP-seq and CUT&RUN libraries measure the relative
   distribution of reads, not absolute modification levels. Depth-based scaling absorbs
   genome-wide changes, such as global loss of H3K27me3 after EZH2 inhibition, so
   occupancy differences between conditions are confounded with library composition.
4. **Non-identifiability across conditions.** States learned separately per sample, or
   without explicit sample-specific scaling, need not correspond across conditions.
   Differences in state occupancy then mix biological change with differences in how
   the model was parameterised.

ChromState addresses each of these within a single generative model:

| Requirement | Discrete segmentation (ChromHMM) | ChromState |
|---|---|---|
| Modification dosage | Binarised (present/absent) | Read counts, negative-binomial likelihood |
| Uncertainty of state assignment | Posteriors, typically saturated | State composition, entropy, margin; temperature calibration |
| Bivalency | Membership of a "bivalent" state | Continuous score resolved into magnitude and balance; mixture diagnostic |
| Evidence for a call | Not decomposed | Exact per-mark log-likelihood-ratio decomposition |
| Comparability across conditions | Not guaranteed | Shared emission parameters; depth, median-of-ratios or spike-in offsets |
| Significance of change | Not addressed | Replicate noise floor; region-level moderated *t*-test with FDR control |
| Relation to transcription | Not addressed | Gene-level state composition, RNA-seq linkage, per-gene evidence tracing |

### 1.2 Design principles

**Likelihood-based modelling of counts.** Observations are modelled as
negative-binomial counts. This respects the mean–variance relationship of sequencing
data and lets the dispersion of each mark be estimated from the data. Normalisation
enters as a multiplicative offset on the expected count, never as a transformation of
the observations, so the model remains a proper count likelihood under any size-factor
scheme, including spike-in calibration.

**Shared parameterisation across conditions.** A state is defined by its emission
vector, and this vector is estimated once from all samples jointly. Conditions differ
only in how the genome is distributed among states. A change in occupancy can
therefore be read as a change in the genome, not in the definition of the state, and no
post hoc matching of states between separately fitted models is needed.

**Probabilistic rather than categorical output.** The posterior distribution over states
is treated as the measurement, and the maximum a posteriori segmentation as a summary of
it. Uncertainty is reported explicitly (entropy, margin, effective number of states)
and can be calibrated against replicate agreement, so that partial and transitional
configurations are retained rather than rounded away.

**Decomposable evidence.** Marks are conditionally independent given the state, so the
log-likelihood ratio between any two states is an exact sum of per-mark terms. Every
state assignment can therefore be attributed quantitatively to the modifications that
support or oppose it, without approximate saliency methods.

**Distinct readouts for distinct mechanisms.** Quantities with different biological
meanings are kept separate rather than combined:
- bivalency magnitude versus balance;
- bivalency carried by a single state versus posterior mass shared between an active
  and a repressive state (compatible with a mixture of cell populations);
- promoter versus gene-body composition.

**Separation of effect size from statistical confidence.** The size of a change is
measured on the state composition. Its reliability is assessed against two references:
the distribution of the same statistic between biological replicates (the noise
floor), and a variance-moderated test at the level of predefined regions with
false-discovery-rate control.

**Validation against known ground truth.** Correctness is established on simulated data
whose states and parameters are known. Competing models are reimplemented on identical
input, with an equal computational budget, so that differences can be attributed to the
model rather than to preprocessing (section 7).

---

## 2. Installation

ChromState requires Python ≥ 3.9.

```bash
git clone https://github.com/helenhuangmath/ChromState.git
cd ChromState
pip install -e ".[all]"        # numpy, scipy, pandas, scikit-learn, matplotlib,
                               # plus pysam (BAM), pyBigWig (bigWig), anndata
```

`pysam` is required only for BAM input and `pyBigWig` only for bigWig input. Numba is
used automatically if installed and accelerates the forward–backward and Viterbi
recursions.

The R interface calls the Python package through `reticulate`:

```r
install.packages("reticulate")
# from the repository root, after installing the Python package:
R CMD INSTALL R/chromstatR
```

Check the installation by running the test suite (`pytest -q`) or the simulated demo
(next section).

---

## 3. Quick start

### 3.1 Simulated end-to-end run

```bash
chromstate demo -o demo_out
```

The demo simulates a two-condition, six-mark experiment with known states. It writes the
input files to disk, reads them back, fits the model, quantifies every bin, benchmarks
five competing models and writes 13 figures, 12 tables and a report
(`demo_out/REPORT.md`). It takes about 15 s.

### 3.2 Real data from the command line

ChromState reads ChromHMM's `cellmarkfiletable` format without modification:

```text
# cellmarkfiletable.txt — sample <TAB> mark <TAB> file [<TAB> control]
CD8_naive       H3K4me3    naive_K4me3.bam    naive_IgG.bam
CD8_naive       H3K27me3   naive_K27me3.bam   naive_IgG.bam
CD8_exhausted   H3K4me3    exh_K4me3.bam      exh_IgG.bam
CD8_exhausted   H3K27me3   exh_K27me3.bam     exh_IgG.bam
```

```bash
# 1. choose the number of states by held-out likelihood
chromstate selectk --manifest cellmarkfiletable.txt --chrom-sizes mm10.chrom.sizes \
                   --candidates 6,8,10,12,15 -o selectk_out

# 2. fit, quantify and plot
chromstate fit --manifest cellmarkfiletable.txt --chrom-sizes mm10.chrom.sizes \
               -k 12 --norm median_ratio --blacklist mm10.blacklist.bed -o fit_out
```

`chromstate fit` writes the following files:

| File | Content |
|---|---|
| `emissions.tsv`, `transitions.tsv` | Fitted emission means (reads per bin) and transition matrix |
| `states.tsv` | State mnemonic, colour, the rule that named it, and hard and soft coverage |
| `occupancy.tsv` | Sample × state occupancy (mean posterior) |
| `<sample>_segments.bed` | ChromHMM-style dense BED of the Viterbi path, with Roadmap colours |
| `quantification.tsv.gz` | One row per bin and sample: every per-bin quantity in section 5.5 |
| `summary.tsv` | Per-sample summary of the quantification table |
| `figures/*.pdf` | Emissions, transitions, coverage, browser track, confidence, bivalency, attribution, state flow, occupancy change |

Other commands:
- `chromstate compare` benchmarks the competing models on simulated data.
- `chromstate export-chromhmm` writes binarised input for the real ChromHMM.
- Run `chromstate <command> -h` to see all options.

---

## 4. Input data

### 4.1 Why un-normalised counts

ChromState models read counts. The model must be given **raw counts**, not normalised
or smoothed tracks, for the same reason as count-based RNA-seq methods: the
negative-binomial likelihood describes sampling noise in counts, and sequencing-depth
differences are accounted for through size factors (section 6.2). A rescaled coverage
track does not have count noise. Fitting a count model to it gives over- or
under-confident posteriors without any warning.

### 4.2 Supported formats

The format is taken from an optional fifth manifest column or inferred from the file
extension.

| Format | Extension | Read as | Default model |
|---|---|---|---|
| Aligned reads | `.bam` | One count per fragment midpoint (paired-end) or shifted 5′ end (single-end); MAPQ ≥ 30; duplicates removed | Negative binomial |
| Aligned reads as BED | `.bed`, `.tagAlign` | As for BAM | Negative binomial |
| Coverage track | `.bw`, `.bigWig` | Base-weighted mean coverage per bin | Gaussian on log scale |
| Called peaks | `narrowPeak`, `broadPeak`, BED3/6 | Peak score per bin | Gaussian or Bernoulli |

Coverage tracks and peak calls are already-processed summaries. ChromState accepts them
but refuses to fit the negative-binomial model to them, because a dispersion estimated
from scaled or thresholded values has no meaning. If only replicate-pooled bigWig tracks
are available, rounding a depth-scaled track to integers is a usable approximation for
an exploratory analysis. Record that choice, as the analysis in section 8 does.

### 4.3 Genome tiling, blacklist and controls

- **Bins.** The genome is tiled into fixed bins, 200 bp by default to match ChromHMM, so
  segmentations can be compared bin for bin.
- **Chains.** Each chromosome in each sample is a separate Markov chain. Transitions are
  never inferred across a chromosome or sample boundary.
- **Blacklist.** Regions in a blacklist BED are removed.
- **Controls.** An input or IgG control given in the fourth manifest column is binned
  and stored with the dataset. It is used for diagnostics, not as an emission offset.

### 4.4 Experimental design

ChromState needs **biological replicates** to separate a condition effect from
sample-to-sample variation. A label-permutation test cannot give a *p*-value below
1/(number of distinct group assignments):

| Replicates per group | Smallest permutation p-value |
|---|---|
| 2 | 0.33 |
| 3 | 0.10 |
| 4 | 0.029 |

After Benjamini–Hochberg correction across many regions, no permutation result can
reach significance below about eight replicates per group. `differential_regions`
therefore uses a variance-moderated *t*-test by default (section 6.6). It reports the
permutation floor when the assumption-free route is requested instead.

When a perturbation may change a mark globally (for example an EZH2 inhibitor), use
**spike-in normalisation** (section 6.2).

---

## 5. Standard workflow

The examples use the Python API. The R function that does the same job is given where
one exists.

### 5.1 Building a dataset

```python
import chromstate as cs
from chromstate.genome import BinSet, Genome

binset = BinSet.from_genome(Genome.from_file("mm10.chrom.sizes"), bin_size=200)
ds = cs.build_dataset(cs.read_manifest("cellmarkfiletable.txt"), binset,
                      blacklist=cs.read_bed_regions("mm10.blacklist.bed"))
```

```r
library(chromstatR)
ds <- cs_load("cellmarkfiletable.txt", "mm10.chrom.sizes",
              blacklist = "mm10.blacklist.bed")
```

The dataset stacks every (sample, bin) row into one matrix. This is what allows a single
model to be fitted across all samples.

### 5.2 Normalisation

```python
cs.normalize(ds, method="median_ratio")          # or "depth", "spikein", "none"
cs.normalize(ds, method="spikein", spikein={("drug", "H3K27me3"): 41_250, ...})
```

ChIP-seq and CUT&RUN libraries are **compositional**: they report the relative
distribution of reads, not the absolute amount of a mark. If a treatment removes 65% of
H3K27me3 genome-wide, depth or median-of-ratios normalisation rescales the remaining
signal back up, and the treated sample appears unchanged. In simulation with a known 35%
retention, the fraction of the true effect that survives normalisation is:

| Method | Fraction of the effect surviving normalisation |
|---|---|
| Sequencing depth | 0.36 (effect erased) |
| Median of ratios | 0.35 (effect erased) |
| Spike-in | 1.00 (effect preserved) |

`normalize` warns when a non-spike-in method is applied to a design in which a global
change is possible.

### 5.3 Choosing the number of states

```python
sel = cs.select_n_states(ds, candidates=(6, 8, 10, 12, 15))
```

```r
sel <- cs_select_k(ds, candidates = c(6, 8, 10, 12, 15))
```

Whole chromosomes are held out, because neighbouring bins are strongly correlated and
holding out random bins would leak information across the split. The function reports
held-out log-likelihood per bin together with BIC and AIC. Held-out likelihood is the
recommended criterion; BIC over-penalises because adjacent bins are not independent.
On genome-scale data the held-out likelihood often keeps rising with *K*. Choose the
smallest *K* that captures most of the gain, and confirm that the extra states are
biologically distinct.

### 5.4 Fitting the model

```python
seg = cs.fit(ds, n_states=12)        # negative binomial for count input
seg.emission_frame()                 # states × marks, mean reads per bin
seg.sample_state_matrix()            # samples × states occupancy
seg.meta["restart_hits_at_best"]     # number of restarts that reached the best optimum
```

```r
seg <- cs_fit(ds, n_states = 12)
cs_emissions(seg); cs_states(seg)
```

Baum–Welch is not convex. ChromState uses multiple restarts seeded by k-means on the
per-mark profile. By default it keeps restarting until the best log-likelihood has been
reached at least twice (`adaptive_restarts=True`). It records `restart_hits_at_best`,
`min_state_separation` and a `collapsed` flag, and warns when a fit appears degenerate.

States are named automatically from their emission profiles with the Roadmap
Epigenomics vocabulary (TssA, TssBiv, ReprPC, Het, Quies, …). The rule that produced
each name is recorded in `states.tsv`. **Names are a convenience, not a result.** With
small mark panels, weak states can be named "Quiescent" even though they carry signal,
so always check the emission profile.

### 5.5 Per-region quantification

```python
table = cs.quantify(seg)                         # one row per (sample, bin)
cs.top_regions(table, "bivalency_score", n=100)
```

```r
q <- cs_quantify(seg)
gr <- cs_to_granges(q)                           # GenomicRanges for Bioconductor
```

| Column | Meaning |
|---|---|
| `post_<state>` | Posterior probability of each state (the state composition) |
| `state`, `confidence`, `margin` | Most probable state; its posterior; gap to the runner-up |
| `entropy`, `effective_states` | Normalised Shannon entropy of the posterior; its perplexity |
| `bivalency_score` | Harmonic mean of normalised active and repressive signal |
| `bivalency_magnitude`, `bivalency_balance` | Geometric-mean magnitude and the 0–1 balance of the two marks |
| `biv_within_state`, `biv_between_state` | Bivalency carried by one state, versus posterior mass split between an active and a repressive state |
| `mixture_ambiguity` | Share of bivalency that is between states, i.e. compatible with a mixture of two cell populations |
| `llr_<mark>`, `total_llr`, `top_mark` | Per-mark log-likelihood ratio supporting the call over the runner-up, which sums exactly to `total_llr` |

### 5.6 Comparing conditions

```python
flow  = cs.state_flow(seg, "TN", "TEX")                 # P(state after | state before)
flux, cost = cs.transport_flux(seg, "TN", "TEX")        # optimal transport between states
occ   = cs.occupancy_change(seg, reference="TN")

# confidence in a change needs replicates
floor = cs.replicate_noise_floor(seg, {"TN": tn_reps, "TEX": tex_reps})
cal   = cs.fit_temperature_from_replicates(seg, {"TN": tn_reps, "TEX": tex_reps})
seg   = cs.apply_temperature(seg, cal.temperature)
res   = cs.differential_regions(seg, tn_reps, tex_reps,
                                regions=features.promoters(), noise_floor=floor)
res.table.query("confident_change")      # clears both the noise floor and the FDR
```

Three safeguards are applied by default:
- **A noise floor.** It defines what an unchanged region looks like, using replicates
  of the same condition.
- **Calibration.** Saturated posteriors compress change scores towards 0 or 1, so
  partial shifts are lost unless the posteriors are calibrated.
- **Region-level testing.** Regions are defined without reference to the effect, and
  testing is done at that level rather than per bin.

### 5.7 Genes and expression

```python
features = cs.FeatureSet.from_gtf("gencode.vM25.annotation.gtf")
genes    = cs.gene_state_matrix(seg, features, part="promoter")   # gene × sample composition
change   = cs.gene_state_change(genes, tn_reps, tex_reps)
linked   = cs.link_expression(change, rna_lfc)        # rna_lfc: gene_id, log2fc (e.g. DESeq2)
assoc    = cs.expression_association(linked)          # per state, with a gene-shuffle null
shortlist = cs.rank_traceable_genes(linked, n=50)
```

- **Promoter and gene body are separate readouts.** H3K4me3 marks the transcription
  start site and H3K36me3 spans transcribed bodies; averaging the two produces a value
  without a clear meaning.
- **Shared bins are visible.** A bin overlapping two genes is assigned to both by
  default, so double counting is visible; `assign="nearest_tss"` partitions bins
  instead.
- **The null preserves each gene.** Association tests shuffle whole genes, which breaks
  the pairing while keeping each gene's own distribution.
- **Association, not causation.** Results are reported as associations, not as evidence
  that chromatin caused the expression change or of its direction.

### 5.8 Tracing a single gene

```python
trace = cs.trace_gene(seg, features, "ENSMUSG00000025902")
```

`trace_gene` returns the full chain of evidence for one gene: its bins, each bin's
posterior, the per-mark log-likelihood ratios that produced the call, and the
underlying read counts. No quantity in it is a summary of another summary.

---

## 6. Statistical model

### 6.1 Data

Let $Y_{bms}$ be the read count in bin $b$ for mark $m$ in sample $s$, and $z_{bs} \in
\{1, \dots, K\}$ the latent chromatin state of bin $b$ in sample $s$. Each
(sample, chromosome) pair is an independent Markov chain with a shared transition matrix
$A$ and initial distribution $\pi$.

### 6.2 Emissions and normalisation

Conditional on the state, marks are independent negative-binomial variables. A
sample- and mark-specific size factor $s_{sm}$ multiplies the mean:

$$
Y_{bms} \mid z_{bs}=k \;\sim\; \mathrm{NB}\!\left(\mu_{km}\, s_{sm},\; r_m\right),
\qquad \mathrm{Var} = \mu + \mu^2 / r_m .
$$

The state means $\mu_{km}$ are shared by all samples. This joint fit is what makes
state $k$ the same in every condition. Dispersion $r_m$ is estimated per mark by
weighted maximum likelihood. The size factor $s_{sm}$ is one of:
- **library depth;**
- **median of ratios**, computed over ~10 kb windows, because 200 bp bins hold too few
  reads for a stable ratio;
- **spike-in**, the ratio of exogenous reads (Orlando et al., 2014).

Alternative emissions are a diagonal Gaussian on $\log(1 + Y/s)$ (Segway-like) and a
Bernoulli on Poisson-binarised calls (ChromHMM's model). Both are provided for
non-count input and for benchmarking.

### 6.3 Inference

Parameters are estimated by Baum–Welch (EM) with scaled forward–backward recursions.
Initial states come from k-means on log per-mark signal, and dispersion is initialised
from within-cluster variance. Initialising from the marginal variance overstates
overdispersion and causes states to collapse (see `docs/PLAN.md` §9). The posterior
$\gamma_{bs}(k) = P(z_{bs} = k \mid Y)$ is the primary output. The Viterbi path gives a
hard segmentation for compatibility.

### 6.4 Per-bin quantities

- **Entropy** $H_b = -\sum_k \gamma_b(k)\log\gamma_b(k) / \log K \in [0,1]$.
  **Margin** is $\gamma_{(1)} - \gamma_{(2)}$, the gap between the best and second-best
  state.
- **Bivalency.** Let $a$ and $r$ be the normalised log signal of the active and
  repressive marks (H3K4me3 and H3K27me3 by default). Then:
  - magnitude $=\sqrt{ar}$;
  - balance $=2\sqrt{ar}/(a+r)$;
  - score $= 2ar/(a+r)$, the product of magnitude and balance.

  Magnitude and balance are reported separately: a weak but balanced region and a
  strong but lopsided one can share a score while meaning different things. At the
  posterior level, `biv_within_state` measures bivalency carried by a single state.
  `biv_between_state` measures posterior mass split between an active and a repressive
  state, which in bulk data cannot be distinguished from a mixture of two cell
  populations.
- **Attribution.** Marks are conditionally independent given the state, so the
  log-likelihood ratio between the called state $k$ and a comparison state $j$
  decomposes exactly into per-mark terms:

$$
\log\frac{p(Y_b\mid k)}{p(Y_b\mid j)} \;=\; \sum_m \Big[\log p(Y_{bm}\mid k) - \log p(Y_{bm}\mid j)\Big].
$$

  This is an identity of the model, not a saliency approximation.

### 6.5 Calibration

HMM posteriors are typically over-confident. A temperature $T$ rescales each posterior
as $\gamma^{1/T}$, renormalised; this changes confidence without changing the hard
calls (Guo et al., 2017). $T$ is fitted either against known labels (simulation) or
against agreement between biological replicates (real data). Calibration reduces
saturation but cannot fix a bias shared by every replicate. In particular it does not
correct the conditional-independence assumption.

### 6.6 Change between conditions

- **State flow** is the matrix $P_a^\top P_b / n$ over matched bins, optionally
  conditioned on the earlier state. It treats the two samples' posteriors as
  independent given position.
- **Transport flux** is computed by entropically regularised optimal transport
  (Sinkhorn; Cuturi, 2013). The cost is the distance between state emission profiles,
  for samples that cannot be matched bin to bin.
- **Region-level differential test.** Each region's occupancy is moved to the logit
  scale and tested with a two-sample *t*-statistic whose variance is shrunk towards the
  across-region median. This is an empirical-Bayes moderation in the spirit of limma
  (Smyth, 2004). *p*-values are adjusted by Benjamini–Hochberg. A region is called a
  confident change only if it also clears the replicate noise floor.

---

## 7. Benchmarks on simulated data

`chromstate demo` (seed 1) simulates 12,000 bin-rows: 2 conditions × 2 chromosomes ×
3,000 bins, 6 marks and 7 true states. In 35% of bins the emission profile is blended
between two states, so some calls are genuinely ambiguous. Competing methods are
reimplemented **as their statistical models on identical bins**, with an equal restart
budget, so that differences reflect the model rather than preprocessing.

| Method | Adjusted Rand index | Accuracy | Bivalency AUPRC | Signal Spearman ρ | Calibration error |
|---|---|---|---|---|---|
| **ChromState** (negative-binomial HMM) | **0.841** | **0.916** | **0.786** | **0.893** | **0.054** |
| Segway-like (Gaussian HMM) | **0.841** | 0.915 | 0.769 | 0.886 | 0.060 |
| ChromHMM-like (binarised Bernoulli HMM) | 0.742 | 0.843 | 0.219 | 0.838 | 0.063 |
| Gaussian mixture (no spatial model) | 0.605 | 0.732 | 0.774 | 0.856 | 0.189 |
| k-means (hard assignment) | 0.689 | 0.845 | 0.567 | 0.860 | 0.155 |
| Naive two-track signal (no model) | – | – | 0.674 | – | – |

The random-baseline bivalency AUPRC is 0.078.

**Detecting regions that respond to a perturbation** (half of the bivalent bins switched
to active promoters; random-baseline AUPRC 0.052):

| Method | Responder AUPRC | Responder AUROC |
|---|---|---|
| **ChromState** | **0.991** | 0.999 |
| Segway-like | 0.984 | 0.999 |
| ChromHMM-like | 0.712 | 0.957 |

### 7.1 Interpretation

1. **Binarisation is the dominant loss.**
   - Among bins called "present", the binarised model maps a 4.9–7.9-fold range of read
     counts onto the single value 1.
   - Its bivalency AUPRC (0.219) is below that of the no-model two-track control
     (0.674). Bivalency depends on the relative amount of two marks, which a
     binarised representation cannot recover.
   - The same loss appears in perturbation detection (0.712 versus 0.991).
2. **The negative binomial adds little over a log-Gaussian model in this setting.**
   ChromState and the Segway-like model tie on segmentation accuracy and differ only
   slightly on bivalency and calibration. Most of the improvement comes from not
   binarising; the specific count distribution is a refinement.
3. **Absolute calibration remains limited.** ChromState is the best-calibrated method
   here, yet 78% of its calls exceed a posterior of 0.99. Temperature scaling (section
   7.5) reduces this, and entropy should not be read as a calibrated measure of
   plasticity without it.
4. **The simulator favours ChromState.** Data are generated from the model family that
   ChromState fits. Real-data validation (sections 8 and 9) is therefore required, and
   the results above should be read as tests of correctness rather than of superiority.

---

## 8. Application: CD8 T-cell differentiation

`analysis/scireports2025/` applies ChromState to the data of Huang, Baxter et al.
(*Sci Rep* 2025; GEO GSE285245 and GSE285248):
- P14 CD8 T cells in three states: naive (TN), memory (TMEM, LCMV Armstrong day 30)
  and exhausted (TEX, LCMV clone 13 day 32).
- CUT&RUN for H3K4me3, H3K27ac, H3K27me3 and H3K9me3, and RNA-seq with four replicates
  per group.
- A 15-state model was selected by held-out likelihood.

Expression was predicted from chromatin by gradient boosting, scored on held-out
chromosomes. Results are Spearman ρ, with 95% bootstrap CIs in the analysis README:

| Prediction target | ChromState quantitative states | Raw signal | ChromHMM state composition | ChromHMM state at TSS |
|---|---|---|---|---|
| Expression level, TN | **0.821** | 0.802 | 0.793 | 0.632 |
| Expression change, TMEM vs TN | 0.511 | **0.529** | 0.407 | 0.162 |
| Expression change, TEX vs TN | **0.553** | 0.542 | 0.442 | 0.179 |

Main observations:
- **Quantitative states and labels.** Quantitative state features predict expression
  far better than hard state labels. Their advantage over raw signal is small and not
  uniform, and combining the two is best for every change task.
- **One dominant state.** The active-TSS state carries most of the predictive
  information. Mark effects are largely additive, with a modest combinatorial
  contribution concentrated in H3K4me3 × H3K27me3 and H3K27ac × H3K27me3.
- **Bivalent promoters.** H3K27me3-dominant bivalent promoters in TN gain expression in
  TMEM and TEX. About two-thirds of the raw effect is explained by regression to the
  mean; the remainder persists against expression-matched controls.
- **Ordering.** A cross-lagged analysis suggests that Polycomb and bivalent states lead
  later expression change, and that H3K9me3 and H3K4me3 follow it. The effects are small
  and come from two time points without chromatin replicates, so they generate
  hypotheses rather than establish order.

The analysis used replicate-pooled coverage tracks because aligned reads were not
available. The scripts, figures and complete results, including all caveats, are in
[`analysis/scireports2025/README.md`](analysis/scireports2025/README.md).

---

## 9. Comparison with external tools

Section 8 compares statistical models on identical input. `tools/` runs the **actual
software** on ChromState's bins:
- Real ChromHMM is enabled by setting `CHROMHMM_JAR=/path/to/ChromHMM.jar`.
- EpiCSeg, Segway, epigenome-ssm and other tools are configured through command
  templates in `tools/tools.json`.

Each tool's output is mapped back onto the bin grid by coordinate and scored with a
common metric registry:
- **Truth-based metrics** (simulation): adjusted Rand index, matched accuracy,
  bivalency AUPRC, magnitude correlation and calibration.
- **Truth-free metrics** (real data): segment length and number, coverage, posterior
  decisiveness, pairwise agreement, runtime and memory.

```bash
python tools/run_tools.py check                            # which tools can run here
python tools/run_tools.py run -o compare_out               # simulated data, all metrics
python tools/run_tools.py run -o out --counts counts.tsv   # real data, truth-free metrics
```

The adapters for EpiCSeg, Segway and epigenome-ssm are **disabled and unverified** until
each has been run against the real software. See [`tools/README.md`](tools/README.md).

---

## 10. Figures

All figures follow Nature's artwork guidelines:
- vector PDF at final print size (89, 120 or 183 mm wide);
- Arial at 5–7 pt, with all text in black;
- 0.5 pt lines and a white background;
- text embedded as editable TrueType.

Colour encodes data only, never text. State labels are written out in full ("Bivalent
transcription start site" rather than `TssBiv`), so a figure can be read without the
mnemonic table.

The default state colours follow the Roadmap Epigenomics scheme, which is familiar but
not safe for colour-vision deficiency: TssA and TssFlnk differ by ΔE 3.0.
`--palette cvd` selects a validated alternative suitable for publication.

---

## 11. Limitations

- **No published real-data validation yet.** The benchmarks in section 7 are
  simulations from the model family ChromState fits. Validation against real ChromHMM,
  EpiCSeg and epigenome-ssm, and against orthogonal bivalency data (reChIP-seq), is
  still outstanding.
- **Conditional independence of marks.** The emission model treats marks as
  independent given the state. This is the main cause of over-confident posteriors and
  also defines what the attribution measures.
- **Bulk bivalency may be a mixture.** In bulk data, a population of cells that are
  either H3K4me3-positive or H3K27me3-positive looks the same as a truly bivalent
  nucleosome. `biv_between_state` and `mixture_ambiguity` flag the ambiguity but cannot
  resolve it; single-cell or reChIP data are required.
- **Automatic state names** are rule-based and can mislabel weak states.
- **Order and causality.** Association between chromatin and expression change does not
  establish causality or temporal order. Ordering requires time-course or perturbation
  data.
- **Not yet implemented:** the landscape and reversibility layer (potential, barriers,
  reversibility index), single-cell input and bigWig *output*. See `docs/PLAN.md`.
- **Name.** The name ChromState is close to the Bioconductor package *chromstaR*, and
  the R interface `chromstatR` differs from it by one letter. Renaming is under
  consideration before a public release.

---

## 12. Frequently asked questions

**Can I use normalised bigWig tracks instead of BAM files?**
Yes for exploration. ChromState then uses the Gaussian model by default (section 4.2).
For inference, use raw counts from BAM files.

**How many replicates do I need?**
At least three per condition for the moderated test; more if effects are small. With
two replicates per group no permutation result can be significant (section 4.4).

**How many states should I fit?**
Use `selectk` and choose the smallest *K* that captures most of the held-out gain, then
check that each additional state has a distinct emission profile.

**Why do the posteriors look so certain?**
HMM posteriors saturate. Use `fit_temperature_from_replicates` and `apply_temperature`
before interpreting entropy or partial changes.

**Can I compare a sample fitted in one run with a sample fitted in another run?**
No. State definitions are only shared within a joint fit. Fit all samples to be
compared together.

**How do I get results into R and Bioconductor?**
`cs_quantify()` returns a data frame and `cs_to_granges()` converts it to `GRanges`.

---

## 13. Repository layout and development

```
src/chromstate/      Python package
  io/                BAM, BED, peak and bigWig readers; manifest parsing
  model/             emissions, HMM, fitting, model selection, state naming
  norm.py            depth, median-of-ratios and spike-in size factors
  quantify.py        composition, entropy, bivalency, attribution
  dynamics.py        state flow, transport, noise floor, differential testing
  calibrate.py       temperature scaling
  features.py        genes, promoters, bin-to-gene mapping, per-gene tracing
  rna.py             expression linkage and association
  viz.py             figures (Nature style, PDF)
  benchmark/         reimplemented competing models and metrics
R/chromstatR/        R interface (reticulate)
tools/               harness for running external segmentation software
analysis/            real-data analyses (scripts and results)
docs/PLAN.md         development plan, prior art and implementation status
tests/               unit and ground-truth tests
```

```bash
pip install -e ".[dev]"
pytest -q                       # 131 tests, including recovery of simulated ground truth
ruff check src tests
```

Continuous integration runs the Python tests on Python 3.9, 3.11 and 3.12, and the R
interface tests on R release.

---

## 14. References

- Anders S, Huber W. Differential expression analysis for sequence count data.
  *Genome Biology* 11, R106 (2010).
- Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and
  powerful approach to multiple testing. *J. R. Stat. Soc. B* 57, 289–300 (1995).
- Cuturi M. Sinkhorn distances: lightspeed computation of optimal transport.
  *Advances in Neural Information Processing Systems* 26 (2013).
- Ernst J, Kellis M. ChromHMM: automating chromatin-state discovery and
  characterization. *Nature Methods* 9, 215–216 (2012).
- Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks.
  *Proceedings of ICML* (2017).
- Hoffman MM, et al. Unsupervised pattern discovery in human chromatin structure
  through genomic segmentation. *Nature Methods* 9, 473–476 (2012).
- Huang H, Baxter AE, et al. Deciphering the role of histone modifications in memory
  and exhausted CD8 T cells. *Scientific Reports* 15, 17359 (2025).
- Love MI, Huber W, Anders S. Moderated estimation of fold change and dispersion for
  RNA-seq data with DESeq2. *Genome Biology* 15, 550 (2014).
- Mammana A, Chung H-R. Chromatin segmentation based on a probabilistic model for read
  counts explains a large portion of the epigenome. *Genome Biology* 16, 151 (2015).
- Orlando DA, et al. Quantitative ChIP-Seq normalization reveals global modulation of
  the epigenome. *Cell Reports* 9, 1163–1170 (2014).
- Roadmap Epigenomics Consortium. Integrative analysis of 111 reference human
  epigenomes. *Nature* 518, 317–330 (2015).
- Smyth GK. Linear models and empirical Bayes methods for assessing differential
  expression in microarray experiments. *Stat. Appl. Genet. Mol. Biol.* 3, Article 3
  (2004).

---

## 15. Licence and contact

ChromState is released under the MIT licence (see [`LICENSE`](LICENSE)). It is
developed by Helen Huang (Berger and Wherry laboratories, University of Pennsylvania).
Please report problems and requests through the
[GitHub issue tracker](https://github.com/helenhuangmath/ChromState/issues).
