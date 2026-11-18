# ChromState against the published tools

This page compares ChromState with the *real* published chromatin segmentation tools
(not reimplementations of their models), each run on ChromState's own bins through the
harness in [`tools/`](../tools/README.md). It has three parts: what each tool is
(§1), what we measured (§2), and what the measurements do and do not support (§3).

**Bottom line, with the caveat that comes first.** The test data are simulated from a
negative-binomial HMM, which is the model family ChromState fits (§3.1). That favours
ChromState by construction. Under that caveat:

* **State recovery.** ChromState recovers the true states best (adjusted Rand index
  0.84). The best real tools are ChromHMM 1.28 (0.74) and EpiCSeg 1.0 (0.72). Real
  ChromHMM does slightly better than our ChromHMM-like reimplementation (0.70) on the
  same binarised input.
* **Bivalency.** ChromState does **not** win. EpiCSeg detects bivalent bins better
  (precision-recall AUC 0.68 vs 0.63, with the same scoring rule for both). The
  binarised models do poorly (ChromHMM 0.15), and a no-model harmonic mean of the two
  marks reaches 0.59.
* **Speed.** ChromState is the fastest tool measured at every size: 8 s for 800,000
  bins, against 23–31 s for ChromHMM, EpiCSeg and normR, and 204 s for diHMM. The
  ChromState timing uses numba, which is optional. Memory use is comparable once the
  data ChromState holds in-process is accounted for.
* **What these numbers rest on.** All four real tools in this table ran and were checked
  against their own outputs. Segway did not run on the Mac (§3.4) but did on real data
  on a Linux cluster (§2.3). diHMM needed bug fixes to run at all, and its row is a
  lower bound (§3.3).
* **On real data (§2.3, CD8 T-cell CUT&RUN, six tools including Segway).** Expression
  *level* is predicted equally well by every tool's promoter states (rho 0.56-0.61,
  raw marks 0.60). Expression *change* between naive and exhausted cells separates them:
  ChromState 0.36, EpiCSeg 0.30, the binarised tools 0.23, Segway 0.12. EpiCSeg finds
  promoters best (AUPRC 0.23), ChromState second (0.21). ChromState is the fastest
  (25 s for 921k bin-rows; Segway and diHMM an hour each).

## 1. What each tool is

| | ChromState | ChromHMM | Segway | EpiCSeg | diHMM | normR |
|---|---|---|---|---|---|---|
| **Model class** | HMM, fitted jointly across samples | HMM | dynamic Bayesian network (GMTK) | HMM | hierarchical HMM: bin chain + domain chain | two-component mixture per bin; no spatial model |
| **Emission / likelihood** | negative binomial per mark, size factors as offsets | multivariate Bernoulli (marks independent given state) | Gaussian on transformed signal | negative multinomial (NB total × multinomial split across marks) | multivariate Bernoulli | binomial: ChIP vs input share of each bin's reads |
| **Input** | raw read counts per bin (+ optional control, spike-in) | binarised marks (`BinarizeBam/Bed`, Poisson threshold vs input) | continuous signal tracks in a genomedata archive | raw read counts per bin (`getcounts` from BAM) | binarised marks (ChromHMM format) | read counts for ChIP and input (BAM or integer vectors) |
| **Resolution** | any fixed bin (200 bp here) | fixed bin, 200 bp default | down to 1 bp | fixed bin, 200 bp default | 200 bp bins + domains of 20 bins (4 kb) | fixed bins, user-chosen |
| **Several conditions in one fit** | yes: one model, shared states, per-sample offsets | yes (concatenated / stacked designs) | not tested here | yes: labelled `--counts` matrices after `normalizecounts` | yes (all input files train one model) | no: one fit per mark and sample; `diffR` compares two |
| **Per-bin posterior** | yes, plus calibration diagnostics | yes (`-printposterior`) | yes (`segway posterior`) | yes (only inside the R archive) | no in the C++ build (Viterbi labels) | yes, per mark (enriched vs background) |
| **Bivalency** | explicit continuous score (magnitude and balance) | only as a state whose emissions show both marks | same, post hoc | same, post hoc | same, post hoc | only by combining two marks' calls (our construction) |
| **Differential testing** | yes (state flow, replicate noise floor, region tests) | no built-in test | no | no | no | yes, per mark (`diffR`) |
| **Language** | Python (numba optional) | Java | Python + GMTK (C++) | R + C++ (Rcpp, kfoots) | C++ with Python bindings (original: MATLAB) | R + C++ (Rcpp) |
| **Citation** | this repository | Ernst & Kellis, *Nat Methods* 9:215 (2012) | Hoffman et al., *Nat Methods* 9:473 (2012) | Mammana & Chung, *Genome Biol* 16:151 (2015) | Marco et al., *Nat Commun* 8:15011 (2017) | Helmuth et al., bioRxiv 082263 (2016) |
| **Run here** | yes | yes, 1.28, verified | **no**: Docker CLI hung (§3.4) | yes, 1.0, verified | yes, ykai16 C++ build, verified | yes, 1.36.0, verified |

"Verified" means that the adapter was run against the installed software and its output
mapping was checked against the tool's own files. Examples: ChromHMM's segment BED
against its posterior argmax (100% agreement), EpiCSeg's BED against its posterior
argmax (100%), and diHMM's labels against its own learned emission matrix (no bin
assigned a state under which it has zero probability). Versions, commands and the
changes each tool needed are in the adapter docstrings under
`tools/toolbench/methods/` and in [`tools/README.md`](../tools/README.md).

## 2. Results

### 2.1 Accuracy on simulated data with known states

Data: `chromstate.simulate`. Four chromosomes × 25,000 bins of 200 bp × two
conditions (control, treated; half of the bivalent bins become active promoters in the
treated sample) = 200,000 bin-rows. Six marks, seven true states, 30% of bins blurred
between two states, seed 0. Every tool was asked for 7 states. Command:

```bash
bash tools/setup/benchrun.sh python tools/run_tools.py run -o compare_sim \
    --simulate 25000 --n-chroms 4 --blur 0.3 --seed 0 --timeout 1800
```

| Tool | Kind | ARI | Matched accuracy | Bivalency AUPRC (one rule) | Bivalency AUPRC (own score) | Magnitude (Spearman) | Overconfidence | ECE | Saturated calls | Mean segment (kb) | Runtime (s) | Peak memory (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ChromState | this package | **0.844** | **0.917** | 0.634 | 0.634 | **0.896** | **0.055** | **0.055** | 0.81 | 1.05 | 2.9 | — |
| ChromHMM 1.28 | real, verified | 0.743 | 0.815 | 0.148 | 0.148 | 0.864 | 0.105 | 0.105 | 0.57 | 2.59 | 7.7 | 461 |
| EpiCSeg 1.0 | real, verified | 0.724 | 0.862 | **0.685** | **0.685** | 0.853 | 0.113 | 0.113 | 0.82 | 0.69 | 14.5 | 503 |
| normR 1.36.0 | real, verified; 61 mark patterns, not a segmentation | 0.483 | (0.811)* | 0.494 | 0.554 | 0.823 | — | — | — | 0.32 | 17.0 | 1081 |
| diHMM (C++), nucleosome level | real, verified, patched (§3.3) | 0.302 | 0.557 | 0.141 | 0.141 | 0.591 | — | — | — | 0.65 | 110.5 | 439 |
| Segway | real | not run: Docker CLI hung on this Mac (linux/amd64 emulation required) | | | | | | | | | | |
| ChromHMM-like | reimplemented model | 0.705 | 0.807 | 0.191 | 0.199 | 0.836 | 0.087 | 0.087 | 0.47 | 2.80 | 1.7 | — |
| Segway-like (Gaussian HMM) | reimplemented model | 0.832 | 0.907 | 0.504 | 0.504 | 0.886 | 0.067 | 0.067 | 0.82 | 0.95 | 2.8 | — |
| Gaussian mixture | reimplemented model | 0.654 | 0.815 | 0.411 | 0.411 | 0.858 | 0.119 | 0.119 | 0.55 | 0.53 | 0.7 | — |
| k-means | reimplemented model | 0.697 | 0.848 | 0.443 | 0.443 | 0.865 | 0.152 | 0.152 | 1.00 | 0.59 | 0.5 | — |
| Naive two-mark signal | no model | — | — | — | 0.591 | 0.739 | — | — | — | — | 0.01 | — |

\* normR's matched accuracy maps 61 patterns onto 7 states many-to-one and is not
comparable (§3.2). The bivalency baseline (fraction of truly bivalent bins) is 0.044.
"—" means not measurable: no posterior (normR, diHMM), no states (naive), or memory not
recorded for in-process methods in this run (see §2.2 for in-process memory).
Calibration: "overconfidence" is mean winning posterior minus accuracy, and "saturated
calls" is the fraction of bins with posterior > 0.99. All columns:
[`figures/tool_comparison/comparison.csv`](figures/tool_comparison/comparison.csv).

![Accuracy](figures/tool_comparison/accuracy.pdf)

[`accuracy.pdf`](figures/tool_comparison/accuracy.pdf) shows the truth-based metrics;
[`calibration.pdf`](figures/tool_comparison/calibration.pdf) the reliability curves for
every tool with a posterior; [`agreement.pdf`](figures/tool_comparison/agreement.pdf) the
pairwise adjusted Rand index between tools.

What the table supports:

* **On this simulation, counts beat binarisation for state recovery.** The rows that
  model signal magnitude (ChromState, the Gaussian HMM, EpiCSeg) recover states better
  than the binarised ones (ChromHMM, the ChromHMM-like reimplementation, diHMM), which
  see the same presence calls. The gap between ChromState (0.84) and real ChromHMM
  (0.74) is close to the gap with the reimplementation (0.70). So the reimplemented
  ChromHMM row in `chromstate compare` slightly *understates* the real tool, but it does
  not change the conclusion.
* **EpiCSeg, the closest prior art, is the strongest external tool.** It edges out
  ChromHMM on matched accuracy and beats every method, ChromState included, on
  bivalency detection. Its ARI is slightly below ChromHMM's: it splits states more
  finely (0.69 kb mean segments against 2.6 kb).
* **ChromState's posteriors are the best calibrated of the tools that report one**
  (overconfidence 0.055 against 0.105 for ChromHMM and 0.113 for EpiCSeg). This is
  also the metric most exposed to the simulation bias of §3.1, because ChromState's
  likelihood is the generating one.
* **A Gaussian HMM on log counts gets within 0.01 ARI of ChromState here.** The
  advantage over binarised tools comes from keeping magnitude more than from the
  negative-binomial form specifically.
* **diHMM's domain level** (descriptive only; no truth exists for it): all 7 domain
  states were used, the mean domain segment was 7.5 kb, and domain labels agree with the
  bin-level truth at ARI 0.19.


### 2.1b Detecting the perturbation (condition change), simulated truth

The condition-comparison claim rests on detecting which bins moved between the control and
the treated sample. Added in v0.3 as the `perturbation` metric: each tool's control-versus-
treated change per bin (total-variation distance of its posteriors, or a changed hard
call when it has none) is scored against the simulated responders (3.0% of bins). Same
data and command as §2.1 (`sbatch/simbench.sbatch`, 2026-11-02; K = 7).

| Tool | Change score | Responder AUPRC | AUROC | Fraction of bins called changed | ARI (same run) |
|---|---|---|---|---|---|
| ChromState | posterior TV | **0.632** | **0.966** | 0.09 | 0.844 |
| ChromHMM-like (reimplementation) | posterior TV | 0.581 | 0.946 | 0.27 | 0.705 |
| Segway-like (reimplementation) | posterior TV | 0.536 | 0.957 | 0.09 | 0.832 |
| EpiCSeg 1.0 | posterior TV | 0.396 | 0.887 | 0.24 | 0.672 |
| GMM | posterior TV | 0.337 | 0.915 | 0.19 | 0.654 |
| ChromHMM 1.28 | posterior TV | 0.214 | 0.910 | 0.28 | 0.743 |
| k-means | hard call | 0.127 | 0.873 | 0.20 | 0.697 |
| diHMM | hard call | 0.070 | 0.785 | 0.38 | 0.302 |
| normR | per-mark calls | 0.057 | 0.748 | 0.50 | 0.483 |

Baseline AUPRC (random ranking) is 0.030. ChromState ranks the responders best and calls
the fewest bins changed while doing so; the binarised tools call a quarter of the genome
changed because a one-bin flip of a saturated posterior counts as much as a real shift.
The caveat of §3.1 applies in full: the simulation is drawn from ChromState's model
family. Real-data differential testing needs chromatin replicates, which the CD8 tracks
at hand do not have (Tutorial 3, §6).

### 2.2 Runtime and memory against dataset size

Same simulator, four chromosomes, two conditions, three sizes (6,250 / 25,000 / 100,000
bins per chromosome per sample, so 50,000 / 200,000 / 800,000 bin-rows). One run per tool
per size, run sequentially, with a 20-minute cap (no tool reached it). Command:

```bash
bash tools/setup/benchrun.sh python tools/scaling.py -o scaling \
    --sizes 6250,25000,100000 --tools chromstate,chromhmm,epicseg,normr,dihmm
```

| Tool | 50k: runtime (s) | 50k: peak MB | 200k: runtime (s) | 200k: peak MB | 800k: runtime (s) | 800k: peak MB | ARI at 800k |
|---|---|---|---|---|---|---|---|
| ChromState† | **1.3** | 335 (floor 253) | **3.0** | 503 (floor 339) | **8.4** | 1124 (floor 658) | 0.845 |
| ChromHMM 1.28 | 3.9 | 372 | 7.8 | 515 | 24.9 | 540 | 0.757 |
| EpiCSeg 1.0 | 18.9 | 415 | 14.8 | 481 | 22.8 | 726 | 0.728 |
| normR 1.36.0 | 15.0 | 1051 | 20.5 | 1042 | 31.0 | 1343 | 0.481 |
| diHMM (C++, serial) | 12.5 | 119 | 109.8 | 440 | 204.5 | 1607 | 0.354 |

† ChromState runs in-process. Its peak is the whole child interpreter, including the
simulated dataset, and "floor" is the same interpreter after simulating but before
fitting. ChromState's own share at 800k is therefore about 470 MB. For the external
tools, peak memory is the tool's process tree. Raw numbers:
[`figures/tool_comparison/scaling.csv`](figures/tool_comparison/scaling.csv).

![Scaling](figures/tool_comparison/scaling.pdf)

* ChromState scales roughly linearly and is 2.6–3× faster than ChromHMM at every size
  measured. It uses numba-compiled forward–backward with 2 restarts; without numba it
  falls back to NumPy and would be slower (not measured here).
* EpiCSeg and normR are dominated by fixed costs at small sizes: R start-up, Bioconductor
  loading, and EpiCSeg's report rendering. That is why EpiCSeg took *longer* at 50k than
  at 200k. Its iteration count also differs between datasets.
* diHMM is the slowest by an order of magnitude and the only tool whose memory grows
  steeply. It stores forward and backward cubes over 7 × 7 joint states per bin and runs
  single-threaded (§3.3).
* normR's ~1 GB is mostly the R/Bioconductor session it runs in, not its model.


### 2.3 Real data: CD8 T-cell CUT&RUN, every tool on the same bins

Simulation cannot settle how the tools behave on a real genome, so the same six were run
on real data: the Sci Rep 2025 P14 CD8 T-cell CUT&RUN (naive, memory/Arm, exhausted/Cl13;
H3K4me3, H3K27ac, H3K27me3, H3K9me3; replicate-pooled DESeq2-scaled coverage per 200 bp
bin, rounded to counts), chromosome 19 in full: 307,158 bins x 3 conditions = 921,474
bin-rows. Every tool was asked for 15 states, the number the published ChromState model
used, and run on a Linux cluster node (8 cores). Segway could run here (Linux), so this
is the first row for it. Commands and the figure definitions are in
[`tools/README.md`](../tools/README.md#real-data-the-same-tools-on-the-cd8-t-cell-cutrun-bins);
results, figures (PDF and PNG), the tables and `REPORT.md` are in
[`figures/tool_comparison/realdata_cd8_chr19/`](figures/tool_comparison/realdata_cd8_chr19/)
(the raw run folders stay under the git-ignored `tmp/realbench/`).

There is no state truth on real data. Each tool is therefore scored on *external*
facts it never saw -- GENCODE vM25 protein-coding genes and the RNA-seq of the same
cells -- with identical cross-validation for every row.

| Tool | Segments (mean kb) | Promoter recovery AUPRC (any TSS / active TSS; chance 0.022 / 0.007) | Expression level rho | Expression change rho (Cl13 - Naive) | Agreement with ChromState (ARI) | Runtime (s) | Peak memory (MB) |
|---|---|---|---|---|---|---|---|
| ChromState | 3.40 | 0.205 / **0.238** | 0.58 | **0.36** | -- | **25** | (in-process) |
| ChromHMM 1.28 (`-init random`) | 2.33 | 0.171 / 0.189 | 0.56 | 0.23 | 0.13 | 65 | 1527 |
| EpiCSeg 1.0 | 3.81 | **0.233** / **0.251** | 0.56 | 0.30 | 0.62 | 51 | 827 |
| normR 1.28 (16 mark patterns) | 1.20 | 0.143 / 0.143 | 0.59 | 0.23 | 0.22 | 22 | 1043 |
| diHMM (C++) | 2.71 | 0.161 / 0.183 | 0.58 | 0.23 | 0.05 | 3501 | 6549 |
| Segway 3.0.7 (10 training rounds) | 2.88 | 0.149 / 0.168 | **0.61** | 0.12 | 0.48 | 3498 | 4667 |
| Raw mark signal, no states | -- | -- | 0.60 | 0.34 | -- | -- | -- |

*Promoter recovery*: per bin the score is its state's promoter rate learned on the other
half of the 1 Mb blocks (two-fold), so a tool gains nothing from merely having many
states; "active TSS" positives are TSS bins of the top-30 % expressed genes in that
condition. *Expression*: gradient boosting on the promoter's (TSS +/- 1 kb) state
composition, five-fold by 1 Mb block, Spearman on held-out genes (1,500 gene x condition
rows; 500 genes for change); bootstrap 95 % intervals are about +/- 0.04 for level and
+/- 0.09 for change. Figures: `state_profiles` (what each state of each tool is, in
marks), `state_class_fractions`, `segment_lengths`, `state_overlap`, `tool_agreement`,
`promoter_recovery`, `expression_prediction`, and three locus views
(*Cd274*/*Pdcd1lg2*/*Jak2*, *Fas*, *Cd5*/*Cd6*).

What the real-data rows support:

* **The count-based tools agree with each other and the binarised tools do not.**
  ChromState, EpiCSeg and Segway form one cluster (pairwise ARI 0.44-0.62); ChromHMM and
  diHMM form another (0.51), with normR between. The split is the input, not the
  algorithm: ChromHMM and diHMM see the same presence calls.
* **Expression *level* does not separate the tools.** Every segmentation predicts level
  about as well as the raw marks at the promoter (rho 0.56-0.61 against 0.60): the
  level is carried by H3K4me3/H3K27ac at the TSS, and any 15-state partition keeps it.
  This is the ceiling of the task, not a tie between methods.
* **Expression *change* does.** Predicting how much a gene's expression moves from naive
  to exhausted from how its promoter composition moves, ChromState (0.36) is the only
  tool at the level of the raw marks (0.34); EpiCSeg follows (0.30), the binarised
  tools and normR sit at 0.23, Segway at 0.12. Change is where quantitative posteriors
  matter: a hard-called or presence-based state cannot move by a little.
* **Promoters: EpiCSeg first, ChromState second**, on both the any-TSS and the
  active-TSS task (0.233/0.251 against 0.205/0.238); the rest are 0.14-0.19. This is the
  same ordering as bivalency on the simulation: EpiCSeg's finer promoter states are its
  strength.
* **Cost.** ChromState 25 s, EpiCSeg 51 s, ChromHMM 65 s for 921k bin-rows; diHMM and
  Segway each needed 58 minutes and 4.7-6.5 GB. Segway was capped at 10 training rounds
  and one instance (its defaults would run several restarts to convergence, which would
  take days on this input); its row is therefore a lower bound on Segway, as diHMM's is.
* **Class shares differ more than the biology does.** On the same bins the tools put
  between 2.8 % (ChromHMM) and 6 % (EpiCSeg, normR) of the genome in active states and
  between 7 % and 29 % in Polycomb states, because each cuts the mark continuum at
  different points; condition differences (exhausted cells lose H3K9me3 and gain
  quiescent share) are consistent across tools.

## 3. Caveats: read these before quoting a number

### 3.1 The simulation favours ChromState

The simulator (`chromstate.simulate`) draws counts from a 7-state HMM with
negative-binomial emissions, which is exactly the model family ChromState fits. Blurring
30% of bins between two states makes the data less separable but does not remove that
advantage. These numbers therefore bound what each tool can do *when ChromState's
assumptions hold*. They are not evidence that ChromState is more accurate on real
chromatin. On real data the harness can report only truth-free metrics
(`run --counts`), and those say whether tools are consistent, decisive or cheap, never
whether they are right. EpiCSeg's negative multinomial is the nearest relative of the
generating model, and it is also the strongest external tool here, which is what that
bias predicts.

### 3.2 What each row is actually given

* **ChromHMM and diHMM** both receive ChromState's Poisson binarisation
  (`export_for_chromhmm`, p < 1e-4), not ChromHMM's own `BinarizeBed`. That keeps
  the presence calls identical between the two binarised-model rows and the
  ChromHMM-like reimplementation, so their differences come from the models. It also
  means the information lost in binarisation is lost for all three in the same way.
* **EpiCSeg** receives raw integer counts and runs its own `normalizecounts`.
  ChromState, the Segway-like reimplementation and the post-hoc profiles use
  ChromState's median-ratio size factors.
* **normR** has no input track in the simulation, so it is given a seeded
  Poisson(mean-depth) pseudo-input. A flat control made normR's own q-value step fail
  (`qvalue::pi0est` cannot handle p-values that discrete). normR is a per-mark
  enrichment caller, not a segmenter: its "states" are the observed on/off patterns
  across marks (61 here, not 7). Its ARI is comparable with the other rows. Its matched
  accuracy is not, because every surplus pattern is mapped to its modal true state,
  which flatters it.
* **Bivalency scores** for external tools are post hoc: each state's posterior-weighted
  mean normalised count goes through the same harmonic-mean formula the in-process
  baselines use (`toolbench/scores.py`). The in-process methods use their *fitted*
  emission means instead. The `biv_posthoc_auprc` column re-scores every row through
  the same post-hoc route. Use that column to compare internal and external rows, and
  `biv_auprc` only within a family.
* **States asked for**: 7 (the truth) for every tool. diHMM was given 7 bin states and
  7 domain states (§3.3). ChromState was run with 2 restarts, as the harness does for
  every in-process method. The external tools ran once with their own defaults
  (ChromHMM: 200 iterations maximum, all cores; EpiCSeg: `--nthreads 10`; normR:
  `procs = 10`; diHMM: 500 iterations maximum, single-threaded).

### 3.3 diHMM: what had to change, and how far to trust the row

The C++ reimplementation (ykai16/diHMM) would not run as published on a current
toolchain. Every change is a patch under `tools/setup/`, and none changes the model:

1. an off-by-one loop in `Emissions_Dec` that reads past the emission matrix;
2. a row-vector `each_col` product that only worked with Armadillo's checks off;
3. **a Viterbi backtrace that stored each bin's predecessor state**, which shifted every
   label one bin downstream. Before the fix, 2,128 of 8,000 labels in a check run gave
   their bin zero probability under the model's own emissions; after it, none did;
4. a serial build, because the OpenMP training loop appends to a shared vector without a
   lock.

The bin and domain state counts must also be equal: unequal counts fail with a matrix
size error, and upstream only ever ran 30/30. Even after the fixes, diHMM converges to a
solution in which one catch-all state holds a large share of bins and several emission
probabilities are exactly 0 or 1. That is why its bin-level accuracy is low. We could
not tell whether this is a poor local optimum from upstream's k-means initialisation or
another defect in the training code, and we did not tune it. Treat the diHMM row as a
lower bound on what the diHMM *model* can do, not as an estimate. Its domain-level
labels have no ground truth in this simulation and are reported only descriptively.

### 3.4 What did not run

* **Segway did not run in the macOS simulation benchmark (§2.1-2.2)**: GMTK, its
  inference engine, is Linux-only, and the Docker CLI hung on that machine. It *did* run
  on the real data (§2.3) on a Linux cluster, through the adapter in
  `tools/toolbench/methods/segway.py` (one genomedata archive with the samples as
  pseudo-chromosomes, `segway train` capped at 10 rounds, `segway annotate`; hard labels
  only). The adapter is marked `verified: false` until its output mapping has been
  checked against Segway's own files beyond the one run reported here. The Segway-like
  row of §2.1 is ChromState's own Gaussian-HMM reimplementation and says nothing about
  Segway the software.
* **epigenome-ssm** was not attempted. It is shipped as a disabled, unverified template.
* **The original diHMM** (MATLAB, gcyuan/diHMM) was not run, because no MATLAB was
  available. Only the C++ reimplementation was.

### 3.5 Measurement limits

Every number comes from a single run with one seed on one simulated genome, on a
10-core Apple-silicon laptop with 16 GB of RAM. Expect a few percent of run-to-run
timing noise, and treat accuracy differences of about 0.02 as noise too. Runtimes include
each tool's own I/O and, for EpiCSeg, the HTML report its `segment` command always
renders. Peak memory for external tools is their whole process tree. For in-process
methods in the scaling run it is the whole child interpreter, *including* the simulated
dataset; `baseline_rss_mb` gives that floor. Several tools stop at a convergence
tolerance, so their runtime depends on how many EM iterations a given dataset needs. It
is not a fixed cost per bin.

## 4. Published methods and model benchmark

### Published methods and their capabilities

| | ChromHMM | Segway | EpiCSeg | diHMM | normR | EpiSegMix | **ChromState** |
|---|---|---|---|---|---|---|---|
| Reference | Ernst & Kellis, *Nat Methods* 2012; *Nat Protoc* 2017 | Hoffman *et al.*, *Nat Methods* 2012 | Mammana & Chung, *Genome Biol* 2015 | Marco *et al.*, *Nat Commun* 2017 | Helmuth *et al.*, bioRxiv 2016; Bioconductor `normr` | Schmitz, Zentgraf & Rahmann, *Bioinformatics* 2024 | this package (v0.1) |
| Task | genome segmentation into combinatorial states | genome segmentation | genome segmentation | two-scale segmentation (nucleosome- and domain-level states) | per-mark enrichment and regime calling; differential enrichment | genome segmentation | segmentation **plus** per-bin quantification, cross-condition testing, gene/expression linkage |
| Graph | multivariate HMM | dynamic Bayesian network (HMM with segment-length ruler) | HMM | hierarchical HMM: domain states (blocks of 20 bins) condition nucleosome-level transitions | none (mixture per bin) | HMM with **duration modelling**: each state expanded into a left-to-right sub-chain giving negative-binomial segment lengths | HMM |
| Emission | independent Bernoulli per mark | Gaussian per track on asinh-transformed signal (Segway 2.0: Gaussian mixtures) | negative multinomial over all marks; by default **one dispersion shared by all states** ("dependent" mode), optionally per state | independent Bernoulli per mark (ChromHMM-style binarised input) | binomial mixture of ChIP vs control counts per mark | per-mark choice among binomial, Poisson, negative binomial, beta-binomial, beta-negative-binomial and Sichel, selected by likelihood | independent negative binomial per mark, per-mark dispersion; also ZINB, Poisson, negative multinomial, Gaussian |
| Input | binarised presence/absence (Poisson threshold on reads, or peaks) | continuous signal (bigWig/genomedata), missing data allowed | raw read counts per bin | binarised, as ChromHMM | raw counts, ChIP and control | raw read counts per bin | raw read counts per bin (BAM/BED); bigWig/peaks via Gaussian/Bernoulli |
| Resolution | 200 bp default | 1 bp (no downsampling); segments median 124 bp | 200 bp in the paper | 200 bp nucleosome level; domains ≥ 4 kb (20-bin blocks) | 250 bp default (paper: 500 bp–1 kb) | 200 bp default | 200 bp default (ChromHMM-compatible) |
| Multiple samples / cell types | concatenated (shared states) or stacked (joint states) | concatenated | shared states across datasets stated as possible; method not detailed | one model trained on three cell lines together (chr17 only, for cost) | per sample; `diffR` compares two tracks | each dataset processed separately, with per-dataset distribution choice | always jointly fitted with shared emissions; conditions differ only in occupancy; size factors as likelihood offsets |
| Normalisation across samples | per-sample binarisation threshold | signal transform | *not stated* | per-sample binarisation | background component estimated jointly with enrichment (ChIP vs control) | *not stated* | depth, median-of-ratios, or **spike-in** offsets inside the count likelihood |
| Posteriors | yes (`-printposterior`), typically saturated | yes (posterior wiggle) | posterior decoding used internally; export *not stated* | computed in training; export *not stated* | enrichment q-values per mark (T-filter FDR) | *not stated*; output is one label per bin plus an HTML report | yes, with entropy/margin and **temperature calibration** against replicates |
| Per-mark attribution of a call | no | no | not decomposable (marks coupled) | no | per mark by construction, no joint state | no | exact per-mark log-likelihood-ratio decomposition |
| Bivalency | a "bivalent" state if learned | a label if learned | a state if learned | bivalent promoter states in Polycomb domains reported | definable as joint enrichment of two marks (not described in the vignette) | a state if learned | continuous score split into magnitude and balance; within- vs between-state diagnostic |
| Differential / cross-condition | no built-in (external tools: ChromDiff, SCIDDO, chromswitch) | no | no | states compared across cell types in the paper | `diffR` (two tracks, one mark, no control needed) | *not stated* | state flow, optimal-transport flux, replicate noise floor, region-level moderated test with FDR |
| Link to gene expression | enrichment of states in expression quantiles | aggregation via Segtools | R² of per-bin log RNA-seq coverage on state label (slightly above ChromHMM) | mean expression z-score of genes with TSS within ±2 kb of each state | no | R² of per-bin log(FPKM+1) on state label; "highest predictive power" among compared methods | gene-level composition, expression association with gene-shuffle null, **state importance** (permutation, drop-one, coefficients, CIs) |
| Choosing K | user; `CompareModels` across K | user | user (BIC/AIC tried, uninformative in the paper) | user; 30 + 30 used, 20–35 similar | number of regimes (user) | user; 10 in benchmarks | held-out likelihood on whole chromosomes, BIC/AIC reported |
| Scalability | fast; hundreds of epigenomes (Roadmap); protocol quotes ~4 h learning | slow; trained on ~1% of genome (pilot regions); Segway 2.0 adds minibatch training | 15–30 min genome-wide, similar to ChromHMM; C++/OpenMP core | ~10 compute-days to train on chr17; < 2 h to annotate genome-wide | fast | *not stated*; C++ core in a Snakemake workflow, trained on pilot regions by default | numba, chain-parallel E-step; 400k bins × 6 marks × 8 states ≈ 0.2 s per EM iteration on two cores |
| Language / availability | Java; compbio.mit.edu/ChromHMM, github.com/jernst98/ChromHMM (v1.28); GPL-3 | Python + GMTK; segway.hoffmanlab.org, github.com/hoffmangroup/segway; GPL-2 | R + C++ (Rcpp/OpenMP); github.com/lamortenera/epicseg; GPLv3 | github.com/gcyuan/diHMM; language and licence *not stated* | R + C++; bioconductor.org/packages/normr; GPL-2 | C++ + Snakemake; gitlab.com/rahmannlab/episegmix; licence *not stated* | Python + R; MIT |

### How the models differ

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

### Three-seed comparison of statistical models

Mean ± s.d. over three simulations (10,000 bins × 4 chromosomes × 2 samples, 6 marks,
7 true states, 35 % of bins blurred between two states; seeds 11–13; every method asked
for 7 states with two restarts). Real ChromHMM is the actual software (version 1.28);
every other competitor is its published statistical model reimplemented on the same bins
(`chromstate.benchmark.baselines`), so differences are attributable to the model, not to
preprocessing. Full table and reading guide: [Tutorial 1 §7.1](tutorials/01_simulation.md).

| Method | ARI | Accuracy | Bivalency AUPRC | Calibration error | Runtime (s) |
|---|---|---|---|---|---|
| **ChromState** (NB-HMM) | **0.803 ± 0.018** | **0.904 ± 0.005** | 0.785 ± 0.058 | **0.060 ± 0.003** | 1.7 |
| EpiSegMix-like (ZINB HMM, geometric durations) | **0.803 ± 0.018** | **0.904 ± 0.005** | 0.779 ± 0.061 | **0.060 ± 0.003** | 0.8 |
| diHMM-like (hierarchical, flattened) | 0.798 ± 0.016 | 0.902 ± 0.004 | **0.818 ± 0.054** | 0.064 ± 0.003 | 2.3 |
| Segway-like (Gaussian HMM) | 0.786 ± 0.020 | 0.895 ± 0.006 | 0.743 ± 0.046 | 0.074 ± 0.005 | 2.2 |
| **ChromHMM 1.28 (real)** | 0.704 ± 0.058 | 0.804 ± 0.070 | 0.414 ± 0.278 | 0.093 ± 0.068 | 7.9 |
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
- Chan RCW *et al.* Segway 2.0: Gaussian mixture models and minibatch training. *Bioinformatics* 34(4) (2018); preprint https://www.biorxiv.org/content/10.1101/147470v1
- Segway software: https://segway.hoffmanlab.org ; https://github.com/hoffmangroup/segway
- diHMM software: https://github.com/gcyuan/diHMM
- ChromHMM software: https://compbio.mit.edu/ChromHMM ; https://github.com/jernst98/ChromHMM
