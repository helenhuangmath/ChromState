# ChromState user guide

This is the full reference walk-through of ChromState: the motivation and design, the
input data, and every step of the standard workflow. For a short introduction see the
[README](https://github.com/helenhuangmath/ChromState/blob/main/README.md); the same material is on the
[documentation site](https://helenhuangmath.github.io/ChromState/).

**Contents:** [1. Overview](#1-overview) · [2. Installation](#2-installation) ·
[3. Quick start](#3-quick-start) · [4. Input data](#4-input-data) ·
[5. Standard workflow](#5-standard-workflow) · [6. Tutorials](#6-tutorials) ·
[7. Comparison with external tools](#7-comparison-with-external-tools) ·
[8. Repository layout and development](#8-repository-layout-and-development)

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
model rather than to preprocessing ([Tutorial 1](https://github.com/helenhuangmath/ChromState/blob/main/docs/tutorials/01_simulation.md)).

---

## 2. Installation

See the [installation guide](https://helenhuangmath.github.io/ChromState/docs/INSTALL.html) for pip, conda and the R interface.

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

With a gene annotation and an expression table, `fit` also quantifies every gene and
ranks the states by their contribution to expression:

```bash
chromstate fit --manifest cellmarkfiletable.txt --chrom-sizes mm10.chrom.sizes -k 12 \
               --gtf gencode.vM25.annotation.gtf --expression expression.tsv -o fit_out
# expression.tsv: gene_id <TAB> CD8_naive <TAB> CD8_exhausted ...  (counts/TPM; log2(x+1) applied)
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
| `gene_states_promoter.tsv.gz` | With `--gtf`: state composition, entropy and mark signal per gene and sample |
| `state_importance_<sample>.tsv` | With `--expression`: per-state permutation and drop-one importance, signed coefficient, intervals, rank |

Other commands:
- `chromstate predict` predicts expression from the per-gene state table and ranks states (section 5.11).
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
[Tutorial 2](https://github.com/helenhuangmath/ChromState/blob/main/docs/tutorials/02_cd8_tcell_application.md) does.

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

Fitting is multithreaded across chromosomes (`n_jobs`, or `CHROMSTATE_THREADS`;
results do not depend on the thread count). On CD8 T-cell chr19 (921,474 bins, 15
states) a fit takes 64 s, against 173 s with the previous implementation, at the same
log-likelihood (−5,720,329 vs −5,720,122).

For exploratory fits on very large data, the parameters can be estimated on part of the
genome and every bin decoded with them:

```python
seg = cs.fit(ds, n_states=12, fit_fraction=0.25)  # EM on 25% of the genome, decode all
```

`fit_fraction` draws whole genomic blocks (`block_bins`, 2,500 bins = 500 kb by default),
the same blocks in every sample. It costs accuracy on real data: on chr19 it halved the
time but agreed with a full fit at an adjusted Rand index of only 0.74. Rare states have
to be present in the sample. Refit on all bins for final results.

Baum–Welch is not convex. ChromState uses multiple restarts seeded by k-means on the
per-mark profile. By default it keeps restarting until the best log-likelihood has been
reached at least twice (`adaptive_restarts=True`). It records `restart_hits_at_best`,
`min_state_separation` and a `collapsed` flag, and warns when a fit appears degenerate.

`emission` defaults to the negative binomial. `zinb` (zero-inflated NB) is available for
data with excess empty bins, `poisson` and `negative_multinomial` (EpiCSeg's model) for
comparison, and `gaussian` for peak-score input. `chromstate.benchmark.algorithms.compare_algorithms`
scores the count models by held-out likelihood on your data; the default is justified in
[Tutorial 1 §7.2](https://github.com/helenhuangmath/ChromState/blob/main/docs/tutorials/01_simulation.md).

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

### 5.8 Which states regulate expression?

`chromstate.regulation` turns every state into numbers per gene and ranks states by how
much held-out expression variance they explain.

```python
gq = cs.gene_quantification(seg, features)          # gene x sample, promoter (P_) and body (B_)

# expression level: genes x samples, log scale
imp, rep = cs.rank_states_for_expression(gq, expression=log_expr, samples=["TN"])
# expression change: log2 fold change of group b over group a
imp, rep = cs.rank_states_for_expression(gq, lfc=rna_lfc, contrast=(tn_reps, tex_reps))

imp.table      # per state: marginal_rho, unique_r2 (95% CI), direction
rep.scores     # held-out accuracy of each representation; rep.contrasts: paired differences
```

For every state and gene part there are three quantities:

| Column | Meaning |
|---|---|
| `occ_<state>` | Occupancy: mean posterior over the region (how much of it is in the state) |
| `int_<state>` | Intensity: posterior-weighted log fold of the signal over the state's own expected signal, on the marks that define the state (how strongly; 0 = typical) |
| `dose_<state>` | Occupancy weighted by that fold |

Entropy, bivalency and mean raw signal per mark (`sig_<mark>`) are included as well.

- **Two kinds of importance.** `marginal_rho` scores a model that sees one state only.
  `unique_r2` is the held-out R² lost when that state is permuted inside the full
  model, i.e. what no other state can replace. Redundant states score high on the first
  and low on the second.
- **Held-out chromosomes.** Every score uses folds grouped by chromosome. Representations
  (`top_state`, `occupancy`, `quantitative`, `signal`, …) are compared on the same genes
  with paired bootstrap differences.
- **Models.** Gradient boosting (`model="hgb"`, the default, no extra dependency),
  `"xgboost"`, `"ridge"` as a linear reference, or any `seed -> estimator` callable.

In the CD8 T-cell data, adding intensity and dose raises held-out prediction of
expression change by 0.020–0.025 Spearman ρ over occupancy-based features, and beats raw
mark signal on two of three contrasts (tie on the third)
([results](https://github.com/helenhuangmath/ChromState/blob/main/analysis/scireports2025/RESULTS.md)).

The composition-only API also reports drop-one scores and signed coefficients.

```python
chrom = features.table.set_index("gene_id")["chrom"]        # fold groups
imp   = cs.composition_state_importance(genes[genes["sample"] == "TN"], log2_tpm_tn, groups=chrom)
imp.table[["state", "rank", "perm_importance", "perm_lo", "perm_hi", "coef_std", "direction"]]
dimp  = cs.state_importance_change(change, rna_lfc, groups=chrom)
reps  = cs.compare_composition_representations(seg, features, log2_tpm_tn, sample="TN")
```

### 5.9 Tracing a single gene

```python
trace = cs.trace_gene(seg, features, "ENSMUSG00000025902")
```

`trace_gene` returns the full chain of evidence for one gene: its bins, each bin's
posterior, the per-mark log-likelihood ratios that produced the call, and the
underlying read counts. No quantity in it is a summary of another summary.

---

### 5.10 Fine-resolution scores, promoters versus enhancers, conditions, and patterns

Version 0.2 adds a second layer on top of the 200 bp model: the model is still fitted at
200 bp (or any ``fit_bin_size``), and every 10 bp bin (or any ``resolution``) is then
scored with the same states on a 0-100 scale. One command runs everything:

```bash
# gene-level mode (default): tile each gene +/- 50 kb at 10 bp, fit at 200 bp, score at 10 bp
chromstate run --manifest cellmarkfiletable.txt --chrom-sizes mm10.chrom.sizes \
               --gtf gencode.vM25.annotation.gtf --mode gene \
               --resolution 10 --fit-bin-size 200 -k 10 --fit-states 16 \
               --promoter-upstream 1000 --promoter-downstream 1000 --flank 50000 \
               --groups "TN=tn1,tn2;TEX=tex1,tex2" --expression rna_log_tpm.tsv --tracks -o run_out

# the same run scored every 200 bp (no finer bins), and an RNA contrast instead of a matrix
chromstate run --manifest ... --chrom-sizes ... --gtf ... --resolution 200 --fit-bin-size 200 \
               --groups "TN=tn1,tn2;TEX=tex1,tex2" --lfc deseq2_TEX_vs_TN.tsv --contrast TN,TEX -o run_out

# genome-wide mode: every chromosome at 10 bp, fitted at 200 bp
chromstate run --manifest ... --chrom-sizes ... --mode genome --resolution 10 -k 10 -o genome_out
```

| Setting | Default | Meaning |
|---|---|---|
| `--resolution` | 10 bp | width of the scored bins; `--fit-bin-size` (200 bp) must be a multiple |
| `--promoter-upstream/-downstream` | 1,000 / 1,000 bp | the promoter is the TSS plus these flanks |
| `--flank` | 50,000 bp | a gene's regulatory window is the gene +/- this; the window minus the promoter is the **distal** (enhancer) space |
| `--mode` | `gene` | `gene` tiles only the regulatory windows; `genome` tiles every chromosome |
| `-k` / `--fit-states` | 10 / none | fit `--fit-states` states and collapse to `-k` by emission similarity (ChromHMM-style merging, done reproducibly) |
| `--expression` / `--lfc --contrast A,B` | none | genes x samples (log scale, or counts without `--expression-is-log`) for the expression-level task; or a per-gene log2 fold change (two columns, or a DESeq2/edgeR table) with the two conditions it compares, for the expression-change task |

The same steps from Python:

```python
import chromstate as cs
scheme = cs.RegionScheme(promoter_upstream=1000, promoter_downstream=1000, regulatory_flank=50_000)
cfg    = cs.RunConfig(mode="gene", resolution=10, fit_bin_size=200, n_states=10, n_fit_states=16,
                      scheme=scheme, tracks=True)
fine   = cs.build_fine_dataset("cellmarkfiletable.txt", "mm10.chrom.sizes", cfg, features=genes)
res    = cs.run_pipeline(fine, "run_out", cfg, features=genes,
                         groups={"TN": ["tn1", "tn2"], "TEX": ["tex1", "tex2"]}, expression=log_expr)

# or step by step
coarse = cs.coarsen_dataset(fine, 20); cs.normalize(coarse)
seg    = cs.fit_collapsed(coarse, n_states=10, n_fit=16)      # fit 16, merge to 10
fine_seg = cs.decode_at_resolution(seg, fine)                 # 0-100 scores every 10 bp
scores = cs.state_scores(fine_seg)                             # score_<state> columns, 0-100
cs.write_score_tracks(fine_seg, "tracks/")                     # one bedGraph per sample and state

summ  = cs.state_summary(fine_seg)                             # bp, %, strength per state
cmp   = cs.compare_conditions(fine_seg, groups)                # per-condition deltas, bp flows
gq    = cs.gene_regulatory_quantification(fine_seg, genes, scheme)   # P_ / E_ / B_ per gene
dom   = cs.gene_dominant_states(gq, groups, part="P", state_classes=...)  # patterns, categories
imp, rep = cs.rank_states_for_expression(gq, expression=log_expr)         # which states matter
parts = cs.rank_parts_for_expression(gq, expression=log_expr)             # P vs E vs P+E
enh   = cs.enhancer_contributions(fine_seg, genes, log_expr, scheme=scheme)  # which enhancers
```

What each piece does, and the choices behind it:

- **Scores at 10 bp without refitting** (`chromstate.resolution`). A negative-binomial
  count over 200 bp is the sum of 20 independent counts over 10 bp with the mean and
  the dispersion divided by 20, and a transition matrix per 200 bp becomes its 20th
  matrix root per 10 bp. The fine pass is one forward-backward run per chromosome, so
  it costs about one EM iteration, not a refit. Scores are `100 x posterior`, integers,
  and sum to 100 across states in every bin.
- **Promoters and enhancers are separate columns** (`chromstate.regions`). The
  `RegionScheme` fixes the promoter flanks and the regulatory window; the distal space is
  the window minus the promoter. Gene tables carry `P_`, `E_` and `B_` prefixes, so
  `rank_states_for_expression` reports promoter and enhancer states separately and
  `rank_parts_for_expression` says whether enhancers add held-out accuracy beyond the
  promoter (`P+E` versus `P`). `call_elements` finds candidate elements from the state
  track itself (runs of active or poised states), so no peak file is required, and
  `enhancer_contributions` ranks each gene's elements by in-silico deletion through a
  promoter-plus-enhancer model that is monotone in enhancer input. The model is
  gradient boosting once there are at least 200 gene x sample rows and a sign-constrained
  linear model below that (`model="auto"`); with few genes a tree model sees one split
  per distance ring and reports the same importance for every element, which the linear
  model avoids. Distance rings are named in kilobases (`enh_0_10kb`) when every edge is a
  whole kilobase and in base pairs otherwise (`enh_0_400bp`), so short windows never
  produce two rings with one name.
- **Conditions** (`chromstate.summary`). `state_summary` gives base pairs and
  percentage per state by hard call and by posterior mass, and the state's *strength*
  (posterior-weighted log fold of signal over the state's own expectation); amount and
  strength can move independently and are kept apart. `compare_conditions` averages over
  replicates, reports percentage-point and strength differences against a reference
  with a replicate-based z for screening, and the base-pair flow matrix between
  conditions. `differential_regions` (section 5.6) remains the test.
- **Patterns and alluvials** (`chromstate.patterns`, `viz.plot_alluvial`). Per gene,
  the dominant state of the promoter (or distal space) in each condition gives a pattern
  (`ReprPC>TssBiv>TssA`) and a category from the state classes (`activated`,
  `deactivated`, `poised_to_active`, `stable`, ...). The same reduction per bin,
  weighted by base pairs, gives the regional flow. Both are drawn as alluvial plots with
  any number of conditions, and the gene-level alluvial is drawn twice: once by the
  promoter's dominant state (`gene_alluvial`) and once by the distal space's
  (`gene_alluvial_distal`), with pattern and category counts for each.
- **Statistics for state-level comparison** (`state_tests`, `flow_test`). Per state,
  a Welch t-test on the per-replicate genome share (the test that generalises to new
  samples; needs two replicates a side) and a 100 kb block bootstrap of the share
  difference (a confidence interval even without replicates, answering only whether the
  change is spread across the genome). Flows between states get block-bootstrap
  intervals too. Region-level testing stays with `differential_regions`.
- **Genome-wide without the memory** (`ScoreTensor`, `run_genome_streaming`,
  `chromstate run --mode genome --stream`). The model is fitted on a few chromosomes,
  then each chromosome is binned, scored and written into a `uint8` score tensor on disk
  (samples x bins x states, one memory-mapped shard per chromosome): 13 GB for a mouse
  genome at 10 bp with 12 states and 4 samples, never more than one chromosome in RAM.
  `ScoreTensor.window` and `region_means` read it back by locus or by region table.
- **Colours** follow the scheme of Huang, Baxter *et al.*, Sci Rep 2025 (Supplementary
  Fig. S4a) by default (`palette="scirep2025"`): active states in reds, poised in magenta,
  repressed in blues, heterochromatin in violet, low signal in grey. `nature` (Okabe-Ito,
  colourblind-safe), `roadmap` and `cvd` remain available; segment BED files keep Roadmap
  colours for browsers.
- **Where the results are.** Every `chromstate run` writes:

  | Folder | Tables | Figures |
  |---|---|---|
  | `model/` | emissions, transitions, states, collapsed members | `emissions.pdf` |
  | `scores/` | `state_scores.tsv.gz` (0-100 per bin), `<sample>_segments.bed`, `tracks/*.bedgraph` | |
  | `summary/` | per-sample and per-condition bp / % / strength, `condition_differences.tsv`, `state_tests.tsv`, `flow_bp_*.tsv`, `flow_test_*.tsv` | `state_percentages`, `state_strength`, `condition_differences`, `state_tests`, `region_alluvial*` |
  | `differential/` | `promoters_<ref>_vs_<cond>.tsv` (moderated t-test, FDR, noise floor) | `volcano_promoters_*`, `noise_floor_*` |
  | `genes/` | `gene_quantification.tsv.gz` (P_/E_/B_ per state), promoter and distal dominant states, pattern and category counts, candidate elements | `gene_alluvial*`, `gene_alluvial_distal*`, `pattern_counts` |
  | `expression/` | state importance, part ranking, promoter-change-vs-RNA with association test, enhancer importance | `state_importance`, `part_ranking`, `expression_link_*` |

- **Memory.** `chromstate run` prints an estimate before loading. Counts and size factors
  are stored as float32 and scores as float32 (or uint8 on disk with `--stream`). At 10 bp
  a mouse genome with 4 marks and 4 samples is about 17 GB of counts plus 17 GB of size
  factors plus 52 GB of float32 scores for 12 states in memory; at 50 bp every one of
  those numbers divides by five, and with `--stream` the peak is one chromosome. For
  promoter and enhancer questions 50 bp loses nothing visible: the bins are still
  four times finer than a nucleosome footprint and twenty times finer than a promoter.
- **A stable number of states** (`chromstate.collapse`). Fit more states than wanted,
  cluster them by emission profile (optionally also by how often they neighbour each
  other), cut at the target number and sum the posteriors. The merge is recorded in
  `meta['members']` and the collapsed model can be rescaled and decoded like any other;
  `match_states` pairs the states of two independent fits. `smooth_segments`
  (`--smooth-bp`) is the spatial counterpart: hard-call runs shorter than a threshold
  are handed to the flanking state with the higher posterior over the run.


### 5.11 Version 0.3: defined states, region x state statistics, expression prediction, single cells

Four additions, one per question the package is asked most often. The design notes
and open questions for each are in [docs/ROADMAP_v03.md](https://github.com/helenhuangmath/ChromState/blob/main/docs/ROADMAP_v03.md).

**States from the marks you give** (`chromstate.definitions`). A `StateDefinitions`
table says which states to measure as patterns over marks (`+` high, `-` low, `.` free).
The fit then uses those states, with their names, in every dataset:

```python
defs = cs.StateDefinitions.default(ds.marks)                 # Roadmap vocabulary for the marks present
defs = cs.StateDefinitions.from_file("states.txt")           # "TssBiv: H3K4me3+ H3K27me3+" lines, or a TSV
seg  = cs.fit(ds, definitions=defs, anchor="seed")           # patterns seed the emissions and fix the names
seg  = cs.fit(ds, definitions=defs, anchor="fixed")          # defined marks are pinned; only free entries learn
seg  = cs.fit(ds, n_states=10, definitions=defs)             # 10 states: the defined ones plus 4 learned
comp = cs.quantify_regions(seg, peaks_bed)                   # fraction and bp of each state in any BED region
```

```bash
chromstate fit --manifest ... --chrom-sizes ... --states default --anchor seed -o out
```

`seed` lets the data move the means (the definitions choose the basin and the labels);
`fixed` makes the measured quantity exactly the stated pattern, which is what cross-dataset
comparison needs. `match_states_to_definitions` aligns any fitted model to a vocabulary.

**Statistics by region and by state** (`chromstate.compare`). One call compares every
condition with a reference and writes both views:

```python
rep = cs.differential_report(seg, groups={"TN": [...], "TEX": [...]}, regions=promoters, reference="TN")
rep.regions["TN_vs_TEX"]        # per region: omnibus moderated t, FDR, noise floor, driver state
rep.region_states               # per region x state: share in each group, delta, t, p, q (BH within state)
rep.states                      # per state: regions gaining / losing at FDR, mean change, genome-wide tests
rep.anova                       # with >= 3 replicated conditions: moderated one-way F per region x state
rep.write("differential/")
```

`chromstate run` writes the same tables for promoters under `differential/`, together
with the statistical figures (`rep.write(outdir, seg=seg)`; module `chromstate.viz_stats`):
an omnibus volcano with the noise floor per contrast, a per-state volcano (which state
moved and by how much), a region x state heatmap of signed −log10 q, gain/loss counts
per state, observed-versus-permuted p-value histograms (test calibration), a
replicate-concordance scatter with the noise-floor band, ECDFs of the change statistic
for replicate pairs against condition pairs, and the ANOVA volcano with three or more
conditions. Examples on simulated data with three replicates per condition are in
[docs/figures/statistics/](https://github.com/helenhuangmath/ChromState/tree/main/docs/figures/statistics/). The noise floor used at region
level is the quantile of the *region* statistic between replicate pairs
(`region_noise_floor`), not the per-bin one, which would be far too permissive once bins
are averaged. Every test needs two or more replicates per condition; with fewer, effect
sizes are reported and the noise floor is the only evidence.

**Which state predicts expression** (`chromstate.predict`, `chromstate predict`):

```python
res = cs.predict_expression(gene_q, expression=log_expr)                       # level
res = cs.predict_expression(gene_q, lfc=rna_lfc, contrast=(tn_reps, tex_reps))  # change
res.summary["rho"], res.best_single_state, res.most_unique_state
res.predictions                 # held-out prediction for every gene (folds grouped by chromosome)
res.importance                  # per state: marginal_rho (its own model), unique_r2 (what only it carries)
res.representations             # quantitative vs occupancy vs hard top state vs raw signal
res.write("predict_out/", seg=seg)   # tables, REPORT.md, figures
```

```bash
chromstate predict --genes run_out/genes/gene_quantification.tsv.gz --expression rna_tpm.tsv -o predict_out
chromstate predict --genes ... --lfc deseq2.tsv --contrast TN,TEX --groups "TN=tn1,tn2;TEX=tex1,tex2" -o out
```

[Tutorial 3](https://github.com/helenhuangmath/ChromState/blob/main/docs/tutorials/03_expression_prediction.md) runs this on the CD8 T-cell
data.

**Single cells** (`chromstate.sc`). Histone states can be *fitted* in single-cell
histone data (scCUT&Tag, nano-CT) after aggregation to metacells, and *transferred* to
scATAC + scRNA when a matched bulk histone dataset exists:

```python
from chromstate import sc
mc      = sc.metacells(lsi, n_metacells=40, labels=cell_type)        # k-means within label
counts, n = sc.pseudobulk(atac_counts, mc)                            # cells x bins -> metacells x bins
sc_ds   = sc.dataset_from_pseudobulk({"ATAC": counts}, bins)         # metacells are samples
tr      = sc.StateTransfer().fit(bulk_seg, bulk_atac_ds, bulk_expr, genes)   # bulk posterior <- proxies
tr.cv_report, tr.note                                                 # held-out ceiling of the transfer
proj    = tr.project(sc_ds, sc_pseudobulk_expr, genes)                # Segmentation over the bulk states
cs.state_summary(proj); cs.differential_report(proj, groups); cs.predict_expression(...)
```

For scCUT&Tag the first three lines plus `cs.fit` are all that is needed. The transfer
reports its chromosome-held-out accuracy on the bulk, which is the ceiling on what the
single-cell scores resolve: states that accessibility and expression cannot separate in
bulk are not separated in cells either.

## 6. Tutorials

The three worked tutorials are on the documentation site:
[simulation](https://helenhuangmath.github.io/ChromState/docs/tutorial_simulation.html),
[CD8 T-cell differentiation](https://helenhuangmath.github.io/ChromState/docs/tutorial_cd8.html) and
[expression prediction](https://helenhuangmath.github.io/ChromState/docs/tutorial_expression.html).

---

## 7. Comparison with external tools

[Tutorial 1](https://github.com/helenhuangmath/ChromState/blob/main/docs/tutorials/01_simulation.md) compares statistical models on identical input. `tools/` runs the **actual
software** on ChromState's bins and scores every tool with one metric registry. The
measured comparison, with caveats, is in [docs/tool_comparison.md](https://github.com/helenhuangmath/ChromState/blob/main/docs/tool_comparison.md).

| Tool | Status in `tools/` |
|---|---|
| ChromHMM 1.28 | real software, verified |
| EpiCSeg 1.0 | real software, verified (two compatibility patches for current R) |
| normR 1.36.0 | real software, verified; per-mark enrichment, scored on mark patterns and bivalency |
| diHMM (C++) | real software, verified after three upstream bug fixes; a lower bound |
| Segway 3.0.7 | real software, run on real data on a Linux cluster (adapter `methods/segway.py`; not yet verified beyond that run) |
| epigenome-ssm | command template only, disabled |

On simulated data (200,000 bins, 7 true states) ChromState had the highest state recovery
(adjusted Rand index 0.84 against 0.74 for ChromHMM and 0.72 for EpiCSeg) and the best
calibrated posteriors. EpiCSeg detected bivalent bins better (AUPRC 0.685 against
0.634). On the same run, ChromState ranks the bins the simulated perturbation moved best
(responder AUPRC 0.63 against 0.40 for EpiCSeg and 0.21 for ChromHMM; [§2.1b](https://github.com/helenhuangmath/ChromState/blob/main/docs/tool_comparison.md)).
The simulation draws counts from ChromState's own model family, so it favours
ChromState; read [§3 of the comparison](https://github.com/helenhuangmath/ChromState/blob/main/docs/tool_comparison.md) before quoting a number.

On **real data** (CD8 T-cell CUT&RUN, chromosome 19, 15 states, all six tools including
Segway; [§2.3](https://github.com/helenhuangmath/ChromState/blob/main/docs/tool_comparison.md)) there is no state truth, so each tool is scored
on genes and RNA-seq it never saw. Every tool predicts expression *level* from its
promoter states about as well as the raw marks do (Spearman 0.56-0.61); expression
*change* between naive and exhausted cells separates them (ChromState 0.36, EpiCSeg
0.30, binarised tools 0.23, Segway 0.12), and EpiCSeg finds promoters best (AUPRC 0.23;
ChromState 0.21). `python tools/run_tools.py realdata` draws the eight figures.

Each tool's output is mapped back onto the bin grid by coordinate and scored with:
- **Truth-based metrics** (simulation): adjusted Rand index, matched accuracy,
  bivalency AUPRC, magnitude correlation and calibration.
- **Truth-free metrics** (real data): segment length and number, coverage, posterior
  decisiveness, pairwise agreement, runtime and memory.

```bash
bash tools/setup/setup_env.sh                              # conda env with the external tools
python tools/run_tools.py check                            # which tools can run here
python tools/run_tools.py run -o compare_out               # simulated data, all metrics
python tools/run_tools.py run -o out --counts counts.tsv   # real data, truth-free metrics
python tools/scaling.py                                    # runtime and memory against size
```

See [`tools/README.md`](https://github.com/helenhuangmath/ChromState/blob/main/tools/README.md) for per-tool setup.

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
  regulation.py      per-state gene quantification; state importance for expression
  resolution.py      fit at 200 bp, score every 10 bp on a 0-100 scale; bedGraph tracks
  regions.py         promoter / distal / body scheme; elements from the state track;
                     which enhancers matter beyond the promoter
  summary.py         bp and % per state, state strength, condition comparison
  patterns.py        dominant-state patterns per gene and per region; flow tables
  collapse.py        merge similar states to a stable number; match states between fits
  pipeline.py        `chromstate run`: gene-level or genome-wide, end to end
  definitions.py     states defined from given marks (pattern table, seeded or pinned emissions)
  compare.py         differential report: per region x state tests, per-state summary, ANOVA
  predict.py         expression prediction and state ranking with a written report
  sc.py              single cells: metacells, pseudobulk datasets, bulk-to-scATAC/scRNA state transfer
  viz.py             figures (Nature style, PDF)
  importance.py      which states matter for expression: permutation, drop-one, coefficients
  benchmark/         reimplemented competing models, algorithm selection, metrics
R/chromstatR/        R interface (reticulate)
tools/               harness for running external segmentation software
analysis/            real-data analyses (scripts and results)
docs/tutorials/      simulation and CD8 T-cell tutorials
docs/PLAN.md         development plan, prior art and implementation status
tests/               unit and ground-truth tests
```

```bash
pip install -e ".[dev]"
pytest -q                       # unit tests, including recovery of simulated ground truth
ruff check src tests
```

Continuous integration runs the Python tests on Python 3.9, 3.11 and 3.12, and the R
interface tests on R release.
