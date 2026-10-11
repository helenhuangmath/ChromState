# ChromState: quantitative chromatin state analysis

**Hua Huang** · Version 0.1.0 (research release) 
Python package `chromstate`, R interface `chromstatR` · MIT licence

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

### Project summary

Epigenetics researchers often want to know how chromatin states move from one 
regulatory program to another, which regions are bivalent
or poised, and which epigenetic features are most likely to respond to treatment.
ChromState is designed for that use case: it upgrades chromatin-state analysis from a
qualitative segmentation into quantitative, traceable measurements that can be compared
across conditions.

The goal is to help identify regulatory regions, marks and transcriptional programs that
may be reversible under perturbation, such as epigenetic drugs in cancer, chronic disease
or metabolic disease settings. Instead of asking only "which state is this region in?",
ChromState asks how much of each state is present, how confident the call is, how strong
and balanced a bivalent signal is, and which histone marks drive the inference.


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
6. [Tutorials](#6-tutorials)
7. [Comparison with external tools](#7-comparison-with-external-tools)
8. [Repository layout and development](#8-repository-layout-and-development)
9. [Licence and contact](#9-licence-and-contact)

---

## 1. Overview

### 1.1 Motivation

Chromatin state segmentation ([ChromHMM](https://doi.org/10.1038/nmeth.1906), Ernst & Kellis 2012;
[Segway](https://doi.org/10.1038/nmeth.1937), Hoffman et al. 2012) reduces the
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
model rather than to preprocessing ([Tutorial 1](docs/tutorials/01_simulation.md)).

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
differences are accounted for through size factors (section 5.2). A rescaled coverage
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
an exploratory analysis. Record that choice, as
[Tutorial 2](docs/tutorials/02_cd8_tcell_application.md) does.

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
therefore uses a variance-moderated *t*-test by default (section 5.6). It reports the
permutation floor when the assumption-free route is requested instead.

When a perturbation may change a mark globally (for example an EZH2 inhibitor), use
**spike-in normalisation** (section 5.2).

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

## 6. Tutorials

Two tutorials work through ChromState in detail.

| Tutorial | Content |
|---|---|
| [**1. Simulation**](docs/tutorials/01_simulation.md) | Simulated two-condition data with known ground truth. Covers state recovery, bivalency, perturbation detection, why spike-in normalisation matters, what binarisation discards, a benchmark against ChromHMM-, Segway-, GMM- and k-means-style models, and the statistical model in brief. |
| [**2. CD8 T-cell differentiation**](docs/tutorials/02_cd8_tcell_application.md) | Naive, memory and exhausted CD8 T cells (four histone marks plus RNA-seq). Covers building the state model, predicting expression and its change, bivalent promoters in naive cells, state importance, additive versus combinatorial marks, and whether chromatin change precedes or follows transcription. |

---

## 7. Comparison with external tools

[Tutorial 1](docs/tutorials/01_simulation.md) compares statistical models on identical input. `tools/` runs the **actual
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

## 8. Repository layout and development

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
docs/tutorials/      simulation and CD8 T-cell tutorials
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

## 9. Licence and contact

ChromState is released under the MIT licence (see [`LICENSE`](LICENSE)). It is
developed by Hua Huang.
Please report problems and requests through the
[GitHub issue tracker](https://github.com/helenhuangmath/ChromState/issues).
