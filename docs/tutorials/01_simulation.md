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
