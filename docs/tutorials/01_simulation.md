# Tutorial 1 — Simulation: checking ChromState against known ground truth


Pre-generated [simulation figures](../figures/README.md#simulation) are included for browsing.

This tutorial simulates a two-condition chromatin experiment in which every state, every
bivalent region and every perturbed region is known in advance. It then runs the
ChromState workflow on the simulated data and checks each result against that truth.

You will:
- recover the true states;
- detect bivalent regions;
- find the regions changed by a perturbation;
- see why normalisation and binarisation matter;
- benchmark ChromState against competing statistical models on identical input.

**Time:** about 2 minutes on a laptop. **Requires:** `pip install -e ".[all]"` (see the
main [README](../../README.md)).

---

## 1. The simulator

`cs.simulate()` draws read counts from a hidden Markov model whose parameters are known.
It returns two objects:
- **`ds`**, the dataset: only what a real analysis would see, namely binned read counts
  for each sample and mark;
- **`truth`**, the answer key: true states, emission means, the bivalent and responder
  regions, depth factors and spike-in counts.

By default it simulates six marks (H3K4me3, H3K4me1, H3K27ac, H3K36me3, H3K27me3,
H3K9me3) and seven states: quiescent, active TSS, active enhancer, transcribed,
Polycomb-repressed, heterochromatin and bivalent TSS. Bins in the two samples share one
underlying state path, so they can be compared one to one.

Three options make the simulation realistic:

| Option | Effect |
|---|---|
| `blur=0.35` | 35% of bins get a profile mixed from two states (weight 0.25–0.5). Real chromatin shades between states, and without blur every method looks perfectly confident. |
| `perturbation={"responder_fraction": 0.5, "from_state": "TssBiv", "to_state": "TssA"}` | In the treated sample, half of the bivalent bins switch to active promoters, like a drug releasing poised genes. |
| `perturbation={"global_mark_loss": {"H3K27me3": 0.35}}` | Only 35% of H3K27me3 remains genome-wide, as after EZH2 inhibition. |

With replicates, pass `treated=[...]` to say which samples receive the perturbation.

```python
import chromstate as cs

ds, truth = cs.simulate(
    n_bins=3000, n_chroms=2, seed=1, blur=0.35,
    perturbation={"responder_fraction": 0.5, "from_state": "TssBiv", "to_state": "TssA"},
)
ds.samples      # ['control', 'treated']
truth.labels    # ['Quies', 'TssA', 'EnhA', 'Tx', 'ReprPC', 'Het', 'TssBiv']
```

The dataset has 2 samples × 2 chromosomes × 3,000 bins = 12,000 rows.

---

## 2. Fit and check state recovery

```python
cs.normalize(ds, method="median_ratio")
seg = cs.fit(ds, n_states=truth.n_states, seed=0)

from chromstate.benchmark.metrics import clustering_scores
clustering_scores(truth.states, seg.states, truth.n_states, seg.n_states)
# {'ari': 0.841, 'ami': 0.823, 'v_measure': 0.823, 'matched_accuracy': 0.916, ...}
```

ChromState recovers the seven states with an adjusted Rand index of 0.84 and 92%
matched accuracy. Because 35% of bins are deliberately mixed, perfect accuracy is not
attainable. The automatic names match the true ones (`seg.labels` contains `TssA`,
`TssBiv`, `ReprPC`, `Het`, `Tx`, `EnhA` and `Quies`).

---

## 3. Bivalency

```python
from chromstate.benchmark.metrics import bivalency_truth_from_states, detection_scores

table = cs.quantify(seg)
is_bivalent = bivalency_truth_from_states(truth.states, truth.bivalent_state)

detection_scores(table["biv_within_state"].to_numpy(), is_bivalent)
# {'auroc': 0.979, 'auprc': 0.786, 'baseline_auprc': 0.078, ...}
detection_scores(table["bivalency_score"].to_numpy(), is_bivalent)
# {'auroc': 0.953, 'auprc': 0.674, ...}
```

ChromState provides two bivalency scores:

| Score | What it measures | AUPRC |
|---|---|---|
| Posterior-level `biv_within_state` | Bivalency carried by a single inferred state | 0.786 |
| Signal-level `bivalency_score` | Harmonic mean of normalised H3K4me3 and H3K27me3 | 0.674 |

Both are far above the random baseline of 0.078. The posterior-level score is the better
one here because it uses all six marks through the fitted states.

---

## 4. Detecting the perturbation

```python
prop = cs.transition_propensity(seg, "control", "treated")    # one row per matched bin
is_responder = truth.responder_mask[ds.sample_mask("treated")]

detection_scores(prop["tv_distance"].to_numpy(), is_responder)
# {'auroc': 0.959, 'auprc': 0.698, 'baseline_auprc': 0.052, ...}

cs.state_flow(seg, "control", "treated").loc["TssBiv"]
# TssBiv 0.56, TssA 0.34, ReprPC 0.04, ...
```

- **Detection.** `tv_distance` is the fraction of a bin's state composition that moved
  between conditions. It ranks the true responder bins with AUROC 0.96.
- **Flow.** The state-flow row for bivalent TSS recovers the simulated switch: 34% of
  bivalent mass moves to active TSS, close to the simulated 50% responder fraction, the
  rest diluted by blurred bins.
- **Noise.** Without blur, detection is almost perfect (`run_perturbation_experiment`
  below gives AUPRC 0.99). Ambiguous bins are what make the task hard.

---

## 5. Why spike-in normalisation matters

ChIP-seq and CUT&RUN libraries report the *relative* distribution of reads. A global
loss of a mark is therefore rescaled away by depth-based normalisation.

```python
from chromstate.benchmark.run import run_normalization_experiment

run_normalization_experiment(n_bins=3000, seed=1)[["method", "biology_absorbed"]]
#          method  biology_absorbed
#           depth             0.356
#    median_ratio             0.351
#         spikein             1.000
```

`biology_absorbed` is the fraction of the true 65% H3K27me3 loss that survives
normalisation. Depth and median-of-ratios scaling erase the effect, returning about the
35% that a complete erasure would give. Spike-in normalisation preserves it exactly.
Use `cs.normalize(ds, method="spikein", spikein=...)` whenever a treatment may change a
mark genome-wide.

---

## 6. What binarisation discards

ChromHMM binarises each mark before modelling. This experiment measures how much
quantitative range that removes:

```python
from chromstate.benchmark.run import run_binarization_experiment

run_binarization_experiment(ds)[["mark", "frac_present", "fold_range_collapsed"]]
#     mark  frac_present  fold_range_collapsed
#  H3K4me3          0.13                  7.33
#  H3K4me1          0.10                  7.95
#  H3K27ac          0.12                  7.33
# H3K36me3          0.12                  6.50
# H3K27me3          0.10                  4.87
#  H3K9me3          0.08                  6.42
```

Among bins called "present", read counts span a 4.9–7.9-fold range, all collapsed onto
the single value 1. ChromState keeps that range.

---

## 7. Benchmarking against other models

`run_comparison` fits each competing method's *statistical model* on exactly the same
bins, with an equal restart budget:

```python
from chromstate.benchmark.run import run_comparison

res = run_comparison(ds, truth, seed=1, n_restarts=2)
res[["method", "ari", "matched_accuracy", "biv_auprc", "signal_spearman", "ece"]]
```

| Method | ARI | Accuracy | Bivalency AUPRC | Signal Spearman ρ | Calibration error |
|---|---|---|---|---|---|
| **ChromState** (negative-binomial HMM) | **0.841** | **0.916** | **0.786** | **0.893** | **0.054** |
| Segway-like (Gaussian HMM) | **0.841** | 0.915 | 0.769 | 0.886 | 0.060 |
| ChromHMM-like (binarised Bernoulli HMM) | 0.742 | 0.843 | 0.219 | 0.838 | 0.063 |
| Gaussian mixture (no spatial model) | 0.605 | 0.732 | 0.774 | 0.856 | 0.189 |
| k-means (hard assignment) | 0.689 | 0.845 | 0.567 | 0.860 | 0.155 |
| Naive two-track signal (no model) | – | – | 0.674 | – | – |

Perturbation detection without blur, from `run_perturbation_experiment(n_bins=3000,
seed=1)`:

| Method | Responder AUPRC | AUROC |
|---|---|---|
| **ChromState** | **0.991** | 0.999 |
| Segway-like | 0.984 | 0.999 |
| ChromHMM-like | 0.712 | 0.957 |

The same analyses run end to end, with figures and a report, using:

```bash
chromstate demo -o demo_out          # this tutorial's settings (seed 1, blur 0.35)
chromstate compare -o benchmark_out  # benchmark only, without blur
```

### 7.1 The full field: EpiCSeg, EpiSegMix, diHMM, normR and real ChromHMM

`run_comparison` also carries reimplementations of the other published chromatin-state
methods, and `tools/run_tools.py` runs the *actual* ChromHMM when `CHROMHMM_JAR` is set.
The table below is the mean ± s.d. over three simulations (`--simulate 10000 --n-chroms 4
--blur 0.35`, seeds 11–13, every method asked for the true number of states, two
restarts), produced by:

```bash
export CHROMHMM_JAR=/path/to/ChromHMM.jar
python tools/run_tools.py run -o out --simulate 10000 --n-chroms 4 --blur 0.35 --seed 11
```

| Method | Stands for | ARI | Accuracy | Bivalency AUPRC | Signal ρ | Calibration error | Runtime (s) |
|---|---|---|---|---|---|---|---|
| **ChromState** (NB-HMM) | this package | **0.803 ± 0.018** | **0.904 ± 0.005** | 0.785 ± 0.058 | 0.881 ± 0.006 | **0.060 ± 0.003** | 1.7 |
| EpiSegMix-like (ZINB HMM) | a flexible per-mark count emission, geometric durations | **0.803 ± 0.018** | **0.904 ± 0.005** | 0.779 ± 0.061 | **0.882 ± 0.006** | **0.060 ± 0.003** | 0.8 |
| diHMM-like (hierarchical, flattened) | diHMM's two-level structure | 0.798 ± 0.016 | 0.902 ± 0.004 | **0.818 ± 0.054** | 0.879 ± 0.006 | 0.064 ± 0.003 | 2.3 |
| Segway-like (Gaussian HMM) | Segway | 0.786 ± 0.020 | 0.895 ± 0.006 | 0.743 ± 0.046 | 0.869 ± 0.007 | 0.074 ± 0.005 | 2.2 |
| **ChromHMM 1.27 (real `LearnModel`)** | ChromHMM | 0.704 ± 0.058 | 0.804 ± 0.070 | 0.414 ± 0.278 | 0.832 ± 0.022 | 0.093 ± 0.068 | 7.9 |
| EpiCSeg-like (negative multinomial HMM) | EpiCSeg | 0.690 ± 0.012 | 0.848 ± 0.004 | 0.796 ± 0.022 | 0.832 ± 0.017 | 0.123 ± 0.002 | 5.6 |
| ChromHMM-like (Bernoulli HMM) | ChromHMM's model, reimplemented | 0.673 ± 0.063 | 0.794 ± 0.067 | 0.407 ± 0.265 | 0.810 ± 0.041 | 0.089 ± 0.069 | 1.5 |
| k-means | hard assignment | 0.655 ± 0.011 | 0.829 ± 0.003 | 0.574 ± 0.024 | 0.838 ± 0.006 | 0.171 ± 0.003 | 1.5 |
| normR-like (per-mark enrichment, no chain) | normR | 0.606 ± 0.007 | 0.802 ± 0.003 | 0.699 ± 0.028 | 0.757 ± 0.025 | 0.198 ± 0.003 | 5.6 |
| Gaussian mixture (no chain) | – | 0.574 ± 0.023 | 0.763 ± 0.046 | 0.654 ± 0.103 | 0.838 ± 0.008 | 0.159 ± 0.039 | 0.7 |
| Naive two-track signal | no model | – | – | 0.681 ± 0.019 | 0.744 ± 0.014 | – | 0.0 |

How to read it:

- **Real ChromHMM behaves like its reimplementation**, which validates the Bernoulli
  baseline, and it is better than the reimplementation on every metric (its
  initialisation is tuned). Its bivalency detection is the least stable number in the
  table (0.21, 0.30 and 0.73 on the three seeds): binarisation removes the H3K4me3 to
  H3K27me3 ratio, so whether a bivalent state is found depends on where the local
  optimum lands.
- **The count-based HMMs form one tier.** ChromState, the zero-inflated variant and the
  hierarchical variant are indistinguishable on segmentation; the hierarchy helps
  bivalency slightly (domain context keeps bivalent bins together), and zero inflation
  does nothing here because the simulation has no excess zeros. Held-out likelihood,
  not this table, is how to choose between them on real data (section 7.2).
- **EpiCSeg's negative multinomial pays for coupling the marks.** One dispersion per
  state for all marks gives the worst calibration among the HMMs and an ARI below the
  independent NB; it also forfeits the per-mark attribution, since the likelihood no
  longer decomposes by mark.
- **normR is not a segmenter** and the comparison is not to its discredit: per-mark
  enrichment without a chain lands between k-means and the GMM, and its joint-enrichment
  bivalency score (0.70) is close to the naive two-track formula (0.68), which is what it is.
- EpiSegMix's and diHMM's *software* has not been run here; the rows marked "-like" are
  their published models on identical input. `tools/tools.json` carries disabled
  adapters for both, and for normR, Segway and EpiCSeg, to be enabled when the software
  is installed. [docs/tool_comparison.md](../tool_comparison.md) compares the tools
  themselves, not just their models.

### 7.2 Choosing the inference algorithm

`compare_algorithms` fits each emission family on the same training chromosomes and
scores it on held-out ones. On count input every count model is a probability mass
function on the same observations, so their held-out log-likelihoods are directly
comparable; the Gaussian is on log-transformed data and is shown but not comparable.

```python
from chromstate.benchmark.algorithms import compare_algorithms, speed_benchmark

compare_algorithms(ds, n_states=truth.n_states, truth=truth)
```

| Emission | Held-out log-lik / bin | ARI | ECE | Saturated posteriors | Mean segment (bins; truth 8.1) | Runtime (s) |
|---|---|---|---|---|---|---|
| **nb** (default) | **−13.77** | **0.777** | **0.063** | 0.755 | 4.60 | 0.83 |
| zinb | **−13.77** | **0.777** | **0.063** | 0.756 | 4.60 | 0.53 |
| poisson | −17.87 | 0.640 | 0.152 | 0.878 | 2.83 | 0.50 |
| gaussian (log signal) | not comparable | 0.776 | 0.070 | 0.800 | 4.35 | 1.23 |

The Poisson row is the reason the default is a negative binomial: without a dispersion
parameter, ordinary count noise is read as state change, segments are a third as long as
the truth, and posteriors are both more saturated and less calibrated. Zero inflation is
free here (one extra parameter per state and mark, no change in fit) and becomes useful
on data with unmappable or deleted regions, where `zinb` wins the held-out comparison;
run `compare_algorithms` on your own data before changing the default.

**Speed.** The E-step runs all chromosomes of all samples in one fused, chain-parallel
forward–backward kernel (numba `prange`); the negative-binomial likelihood is evaluated
once per distinct (count, size-factor) pair and gathered to bins in a single pass; k-means
seeding uses a 200,000-bin subsample. `speed_benchmark(ds)` times these on your machine.
On a 400,000-bin, 6-mark, 8-state problem the per-iteration cost fell from 0.34 s to
0.19 s on two cores, with the fused E-step scaling with the number of cores and the
single-threaded path no slower than before. Posteriors are identical to the reference
implementation to 1e-12 (tested).

### 7.3 Which states matter for expression: validation with planted effects

`simulate_genes_and_expression` places genes on the simulated genome and generates
expression from a *known* linear effect of the promoter's true state fractions: here
TssA +3, ReprPC −2, TssBiv −1, and zero for the other four states. `state_importance`
then has a right answer to be checked against:

```python
from chromstate import simulate_genes_and_expression, state_importance, gene_state_matrix

feats, expr, effects = simulate_genes_and_expression(ds, truth, n_genes=800, seed=1)
seg = cs.fit(ds, n_states=7)
genes = gene_state_matrix(seg, feats, part="promoter", upstream=1000, downstream=1000)
genes = genes[genes["sample"] == "control"]
res = state_importance(genes, expr, groups=feats.table.set_index("gene_id")["chrom"])
res.table
```

| State | Rank | Permutation importance (Δρ) | 95% CI | Standardised coefficient | 95% CI | Direction | True effect |
|---|---|---|---|---|---|---|---|
| TssA | 1 | 0.355 | 0.33–0.39 | +0.92 | 0.86–0.96 | up | +3 |
| ReprPC | 2 | 0.231 | 0.20–0.27 | −0.53 | −0.58–−0.47 | down | −2 |
| TssBiv | 3 | 0.190 | 0.16–0.23 | −0.35 | −0.39–−0.30 | down | −1 |
| Quies, Het, EnhA, Tx | 4–7 | ≤ 0.003 | include 0 | ≤ 0.03 | include 0 | unclear | 0 |

Out-of-fold Spearman 0.81 on chromosome-grouped folds. The three planted states are
recovered in the right order, with the right signs and with magnitudes in the planted
ratio; the four null states are flat and non-significant. The same ranking is obtained
with `model="gbm"`. One thing this exercise exposed, which is now built into the method:
a state's **share** cannot be dropped from a composition without the other shares
giving it away (they sum to one), so naive drop-one importance is identically zero. The
remaining shares are re-closed before refitting, and the ranking is by permutation
importance, which has no such problem.

---

## 8. How to read these results

1. **Keeping magnitude is what matters most.**
   - The binarised model has the lowest bivalency AUPRC of the probabilistic models
     (0.219), below even the no-model two-track signal (0.674). Bivalency is defined by
     the relative amount of two marks, which binarisation removes.
   - The same loss shows in perturbation detection (0.712 versus 0.991).
2. **The count model is a refinement, not the main gain.** ChromState and the
   log-Gaussian Segway-like model tie on segmentation and differ only slightly on
   bivalency and calibration.
3. **Calibration is better, not perfect.** ChromState has the lowest calibration error,
   but 78% of its calls exceed a posterior of 0.99 (`(table["confidence"] > 0.99).mean()`).
   Use `cs.fit_temperature_from_replicates` and `cs.apply_temperature` before reading
   entropy as plasticity.
4. **The simulation favours ChromState.** The data come from the model family ChromState
   fits, so these benchmarks test correctness, not superiority. Real-data evaluation is
   in [Tutorial 2](02_cd8_tcell_application.md).

---

## 9. Statistical model in brief

- **Emissions.** Read counts follow a negative binomial,
  $Y_{bms} \mid z_{bs}=k \sim \mathrm{NB}(\mu_{km} s_{sm}, r_m)$. The state means
  $\mu_{km}$ are shared across samples, the size factor $s_{sm}$ is an offset, and the
  dispersion $r_m$ is estimated per mark.
- **Fitting.** Parameters are fitted by Baum–Welch with k-means initialisation and
  adaptive restarts. The posterior over states is the primary output.
- **Bivalency.** For normalised active and repressive log signal $a$ and $r$:
  magnitude $= \sqrt{ar}$, balance $= 2\sqrt{ar}/(a+r)$, and score $= 2ar/(a+r)$.
- **Attribution.** Marks are conditionally independent given the state, so the
  log-likelihood ratio between two states is an exact sum of per-mark terms.
- **Calibration.** Temperature scaling, $\gamma^{1/T}$ renormalised, softens
  over-confident posteriors without changing the hard calls.

Full derivations are in [`docs/readme_details.txt`](../readme_details.txt), section 7.
