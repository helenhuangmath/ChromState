# Chromatin-state tools compared: ChromHMM, Segway, EpiCSeg, diHMM, normR, EpiSegMix, ChromState

This document compares the published tools on what they model and what they output, then
on identical data. Section 1 is from the papers and software documentation (citations at
the end); section 2 is the benchmark from `tools/run_tools.py` and
`chromstate.benchmark`, reproduced in [Tutorial 1 §7](tutorials/01_simulation.md). Where a
fact could not be confirmed from the primary source it is marked *not stated*.

## 1. What each tool is

| | ChromHMM | Segway | EpiCSeg | diHMM | normR | EpiSegMix | **ChromState** |
|---|---|---|---|---|---|---|---|
| Reference | Ernst & Kellis, *Nat Methods* 2012; *Nat Protoc* 2017 | Hoffman *et al.*, *Nat Methods* 2012 | Mammana & Chung, *Genome Biol* 2015 | Marco *et al.*, *Nat Commun* 2017 | Helmuth *et al.*, bioRxiv 2016; Bioconductor `normr` | Schmitz, Zentgraf & Rahmann, *Bioinformatics* 2024 | this package (v0.1) |
| Task | genome segmentation into combinatorial states | genome segmentation | genome segmentation | two-scale segmentation (nucleosome- and domain-level states) | per-mark enrichment and regime calling; differential enrichment | genome segmentation | segmentation **plus** per-bin quantification, cross-condition testing, gene/expression linkage |
| Graph | multivariate HMM | dynamic Bayesian network (HMM with segment-length ruler) | HMM | hierarchical HMM: domain states (blocks of 20 bins) condition nucleosome-level transitions | none (mixture per bin) | HMM with **duration modelling**: each state expanded into a left-to-right sub-chain giving negative-binomial segment lengths | HMM |
| Emission | independent Bernoulli per mark | Gaussian per track (asinh-transformed signal) | negative multinomial over all marks; by default **one dispersion shared by all states** ("dependent" mode), optionally per state | independent Bernoulli per mark (ChromHMM-style binarised input) | binomial mixture of ChIP vs control counts per mark | per-mark choice among count distributions (Poisson, negative binomial, Sichel, beta-negative-binomial, …), selected by likelihood | independent negative binomial per mark, per-mark dispersion; also ZINB, Poisson, negative multinomial, Gaussian |
| Input | binarised presence/absence (Poisson threshold on reads, or peaks) | continuous signal (bigWig/genomedata), missing data allowed | raw read counts per bin | binarised, as ChromHMM | raw counts, ChIP and control | raw read counts per bin | raw read counts per bin (BAM/BED); bigWig/peaks via Gaussian/Bernoulli |
| Resolution | 200 bp default | 1 bp (no downsampling) | 200 bp in the paper | 200 bp nucleosome level; domains ≥ 4 kb (20-bin blocks) | 250 bp default | 200 bp default | 200 bp default (ChromHMM-compatible) |
| Multiple samples / cell types | concatenated (shared states) or stacked (joint states) | concatenated | shared states across datasets stated as possible; method not detailed | one model trained on three cell lines together (chr17 only, for cost) | per sample; `diffR` compares two tracks | each dataset processed separately, with per-dataset distribution choice | always jointly fitted with shared emissions; conditions differ only in occupancy; size factors as likelihood offsets |
| Normalisation across samples | per-sample binarisation threshold | signal transform | *not stated* | per-sample binarisation | background component estimated jointly with enrichment (ChIP vs control) | *not stated* | depth, median-of-ratios, or **spike-in** offsets inside the count likelihood |
| Posteriors | yes (`-printposterior`), typically saturated | yes (posterior wiggle) | posterior decoding used internally; export *not stated* | computed in training; export *not stated* | enrichment q-values per mark (T-filter FDR) | *not stated*; output is one label per bin plus an HTML report | yes, with entropy/margin and **temperature calibration** against replicates |
| Per-mark attribution of a call | no | no | not decomposable (marks coupled) | no | per mark by construction, no joint state | no | exact per-mark log-likelihood-ratio decomposition |
| Bivalency | a "bivalent" state if learned | a label if learned | a state if learned | bivalent promoter states in Polycomb domains reported | definable as joint enrichment of two marks (not described in the vignette) | a state if learned | continuous score split into magnitude and balance; within- vs between-state diagnostic |
| Differential / cross-condition | no built-in (external tools: ChromDiff, SCIDDO, chromswitch) | no | no | states compared across cell types in the paper | `diffR` (two tracks, one mark, no control needed) | *not stated* | state flow, optimal-transport flux, replicate noise floor, region-level moderated test with FDR |
| Link to gene expression | enrichment of states in expression quantiles | aggregation via Segtools | R² of per-bin log RNA-seq coverage on state label (slightly above ChromHMM) | mean expression z-score of genes with TSS within ±2 kb of each state | no | R² of per-bin log(FPKM+1) on state label; "highest predictive power" among compared methods | gene-level composition, expression association with gene-shuffle null, **state importance** (permutation, drop-one, coefficients, CIs) |
| Choosing K | user; `CompareModels` across K | user | user (BIC/AIC tried, uninformative in the paper) | user; 30 + 30 used, 20–35 similar | number of regimes (user) | user; 10 in benchmarks | held-out likelihood on whole chromosomes, BIC/AIC reported |
| Scalability | fast; hundreds of epigenomes (Roadmap) | slow; training on ~1% of genome recommended | 15–30 min genome-wide, similar to ChromHMM; C++/OpenMP core | ~10 compute-days to train on chr17; < 2 h to annotate genome-wide | fast | *not stated*; C++ core in a Snakemake workflow, trained on pilot regions by default | numba, chain-parallel E-step; 400k bins × 6 marks × 8 states ≈ 0.2 s per EM iteration on two cores |
| Language / availability | Java; compbio.mit.edu/ChromHMM, github.com/jernst98/ChromHMM | Python + GMTK; segway.hoffmanlab.org; GPL-2 | R + C++ (Rcpp); github.com/lamortenera/epicseg; GPLv3 | source stated to be on GitHub (URL and language *not stated* in the paper) | R/Bioconductor `normr` | C++ + Snakemake; gitlab.com/rahmannlab/episegmix; licence *not stated* | Python + R; MIT |

## 2. What the differences mean

**Binarise or not.** ChromHMM and diHMM threshold each mark before modelling, so a bin
with 5 reads and one with 500 are the same observation. That is what makes them fast and
robust across hundreds of epigenomes, and what removes the H3K4me3 : H3K27me3 ratio on
which bivalency depends. Segway keeps continuous signal; EpiCSeg, EpiSegMix and
ChromState keep counts.

**How counts are modelled.** EpiCSeg couples all marks in a bin through one negative
multinomial, which captures correlated depth but imposes a single overdispersion (shared
by all states in its default mode) on marks with very different noise, and makes the
likelihood non-decomposable by mark. EpiSegMix selects a count distribution per mark by
likelihood and models segment lengths as negative binomial by expanding each state into a
short left-to-right sub-chain; its paper reports the highest expression R² among
ChromHMM, EpiCSeg and Segway.
ChromState uses an independent NB per mark with a per-mark dispersion, which is what
makes the per-mark attribution exact, and lets size factors (including spike-ins) enter
as offsets so that a global loss of a mark remains visible.

**Segmenter versus caller.** normR is not a segmenter: it decides, mark by mark, whether a
bin is enriched over its control (`enrichR`), differs between two tracks (`diffR`) or
falls in one of several enrichment regimes (`regimeR`), with q-values after a T-filter.
It is the right tool for "is H3K27me3 present here", and bivalent regions can be defined
from it as the intersection of two enrichment calls; it has no notion of a joint state or
of spatial continuity.

**Hierarchy.** diHMM is the only tool with two scales: domain states over 20-bin (4 kb)
blocks condition the transitions among nucleosome-level states, so domain context shapes
bin-level calls. It inherits ChromHMM's binarisation, and training is expensive (about ten
compute-days for one chromosome in the paper). Its flattened equivalent in the benchmark
(bin states × domain states with tied emissions) reproduces the mechanism on a count
emission without the cost.

**Quantification and comparison.** None of the six treats the posterior as the
measurement, calibrates it, tests state changes against a replicate noise floor, or ranks
states by their contribution to expression. That is the ground ChromState occupies; the
segmentation itself is at parity with the best count-based models (section 3).

## 3. On identical data

Mean ± s.d. over three simulations (10,000 bins × 4 chromosomes × 2 samples, 6 marks,
7 true states, 35 % of bins blurred between two states; seeds 11–13; every method asked
for 7 states with two restarts). Real ChromHMM is the actual software (version 1.27);
every other competitor is its published statistical model reimplemented on the same bins
(`chromstate.benchmark.baselines`), so differences are attributable to the model, not to
preprocessing. Full table and reading guide: [Tutorial 1 §7.1](tutorials/01_simulation.md).

| Method | ARI | Accuracy | Bivalency AUPRC | Calibration error | Runtime (s) |
|---|---|---|---|---|---|
| **ChromState** (NB-HMM) | **0.803 ± 0.018** | **0.904 ± 0.005** | 0.785 ± 0.058 | **0.060 ± 0.003** | 1.7 |
| EpiSegMix-like (ZINB HMM, geometric durations) | **0.803 ± 0.018** | **0.904 ± 0.005** | 0.779 ± 0.061 | **0.060 ± 0.003** | 0.8 |
| diHMM-like (hierarchical, flattened) | 0.798 ± 0.016 | 0.902 ± 0.004 | **0.818 ± 0.054** | 0.064 ± 0.003 | 2.3 |
| Segway-like (Gaussian HMM) | 0.786 ± 0.020 | 0.895 ± 0.006 | 0.743 ± 0.046 | 0.074 ± 0.005 | 2.2 |
| **ChromHMM 1.27 (real)** | 0.704 ± 0.058 | 0.804 ± 0.070 | 0.414 ± 0.278 | 0.093 ± 0.068 | 7.9 |
| EpiCSeg-like (negative multinomial HMM) | 0.690 ± 0.012 | 0.848 ± 0.004 | 0.796 ± 0.022 | 0.123 ± 0.002 | 5.6 |
| normR-like (per-mark enrichment, no chain) | 0.606 ± 0.007 | 0.802 ± 0.003 | 0.699 ± 0.028 | 0.198 ± 0.003 | 5.6 |

Three things to keep in view when reading this. The data are generated from the model
family that the count-based HMMs fit, so the table tests correctness and the cost of
binarisation, not superiority on real chromatin. The "-like" rows are models, not
software: EpiSegMix's duration model and diHMM's binarisation are not reproduced, and the
real tools may do better or worse. And the comparison is at the level of segmentation,
which is the one thing all seven do; the quantities ChromState adds on top (calibrated
composition, noise floor, state importance) have no counterpart to compare against and
are validated separately ([Tutorial 1 §7.3](tutorials/01_simulation.md),
[Tutorial 2](tutorials/02_cd8_tcell_application.md)).

## 4. Running the real software

`tools/run_tools.py` runs any of these tools on ChromState's bins and scores them with the
same metrics. ChromHMM is enabled by `CHROMHMM_JAR` and has been verified. Segway,
EpiCSeg, epigenome-ssm, EpiSegMix, diHMM and normR have command-template adapters in
`tools/tools.json` that are **disabled and unverified** until each has been run against
the installed software (`python tools/run_tools.py check` lists what is needed). The
most valuable next runs are real EpiSegMix and EpiCSeg, which are the closest prior art
to ChromState's emission model, and then all tools on Roadmap data with reChIP-seq
bivalency as orthogonal truth.

## Sources

- Ernst J, Kellis M. ChromHMM: automating chromatin-state discovery and characterization. *Nat Methods* 9:215–216 (2012). https://doi.org/10.1038/nmeth.1906
- Ernst J, Kellis M. Chromatin-state discovery and genome annotation with ChromHMM. *Nat Protoc* 12:2478–2492 (2017). https://doi.org/10.1038/nprot.2017.124
- Hoffman MM, Buske OJ, Wang J, Weng Z, Bilmes JA, Noble WS. Unsupervised pattern discovery in human chromatin structure through genomic segmentation. *Nat Methods* 9:473–476 (2012). https://doi.org/10.1038/nmeth.1937
- Mammana A, Chung H-R. Chromatin segmentation based on a probabilistic model for read counts explains a large portion of the epigenome. *Genome Biol* 16:151 (2015). https://doi.org/10.1186/s13059-015-0708-z
- Marco E, Meuleman W, Huang J, Glass K, Pinello L, Wang J, Kellis M, Yuan G-C. Multi-scale chromatin state annotation using a hierarchical hidden Markov model. *Nat Commun* 8:15011 (2017). https://doi.org/10.1038/ncomms15011
- Helmuth J, Li N, Arrigoni L, Gianmoena K, Cadenas C, Gasparoni G, Sinha A, Rosenstiel P, Walter J, Hengstler JG, Manke T, Chung H-R. normR: Regime enrichment calling for ChIP-seq data. *bioRxiv* 082263 (2016). https://doi.org/10.1101/082263 ; Bioconductor package `normr`.
- Schmitz J, Zentgraf J, Rahmann S. EpiSegMix: a flexible distribution hidden Markov model with duration modeling for chromatin state discovery. *Bioinformatics* 40(4):btae178 (2024). https://doi.org/10.1093/bioinformatics/btae178 ; preprint https://doi.org/10.1101/2023.09.07.556549 ; software https://gitlab.com/rahmannlab/episegmix
- EpiCSeg software: https://github.com/lamortenera/epicseg (GPLv3)
- normR package: https://bioconductor.org/packages/normr
- Segway software: https://segway.hoffmanlab.org ; https://github.com/hoffmangroup/segway
- ChromHMM software: https://compbio.mit.edu/ChromHMM ; https://github.com/jernst98/ChromHMM
