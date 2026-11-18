# ChromState — Development Plan

**Goal:** a Python package that makes chromatin states *quantitative, comparable across
conditions, traceable to individual marks, and predictive of which regions can be moved
by perturbation.*

**Decisions locked in:** Python (`chromstate`) with an R wrapper (`chromstatR`); bulk
histone ChIP-seq/CUT&RUN + ATAC multi-omic for v1, single-cell later; all three novelty
layers shipped in stages.

**Status:** v0.1 is implemented and tested — see §9 for what exists and what does not.

ChromState directly describes the core aim: quantitative analysis of chromatin states and their dynamics.
We will compare it with current methods including ChromHMM, ChromQuant, StateQuant,
DynaChrom, ChromTrace, ChromDynamics, EpiLandscape, ChromLandscape, GenomeLandscape and
ChromSpace. 

---

## 0. Reality check — what is already published

This matters more than anything else in the plan. Several things that feel like the
novelty are already taken, and a reviewer will know them:

| Idea | Already done by | Implication |
|---|---|---|
| Count-based (non-binarized) emissions in an HMM | **EpiCSeg** (negative multinomial, Genome Biol 2015); **EpiSegMix** (flexible distributions + duration modeling, Bioinformatics 2024) | Do not claim "we don't binarize" as the contribution. It's table stakes. |
| Continuous chromatin state features instead of discrete labels | **epigenome-ssm** (Bioinformatics 2022) — continuous state-space model, beats ChromHMM/Segway on TSS/enhancer AUC | Direct prior art for "quantitative state scores." Must benchmark against it. |
| Posterior probabilities per bin | **ChromHMM itself** (`-printposterior`) | ChromHMM is not strictly qualitative. Its real weakness is that posteriors are saturated near 0/1, so they carry little quantitative information — say *that*, precisely. |
| Differential / combinatorial states across conditions | **chromstaR** (Bioconductor) | Name collision risk too — see §8. |
| Multi-scale / hierarchical states | **diHMM** (Nat Commun 2017) | Don't reinvent. |
| Bivalency detection | **normR** (used for reChIP-seq bivalency), SVM poised-gene classifiers | Bivalency *calling* exists; bivalency *quantification with a reversibility interpretation* does not. |
| Epigenetic potential landscape from chromatin | 2026 bioRxiv: dual-modality nano-CUT&Tag (H3K27ac + H3K27me3) defining an "epigenetic potential" metric visualizing repressive barriers, **in single cells** | The landscape idea is being actively claimed. Move fast, and differentiate on *bulk + genome-wide + perturbation-predictive*. |

**Where the open ground actually is — the three claims worth defending:**

1. **Cross-condition quantitative comparability.** Existing segmenters fit per-sample or
   assume states mean the same thing everywhere. A jointly-fit model with sample-specific
   size factors and *spike-in-aware* normalization makes state occupancy a number you can
   compare between a treated and untreated sample. ChIP-seq is compositional; almost every
   tool ignores this. This is the least glamorous and most defensible contribution.
2. **Traceability.** Per-region, per-mark attribution: *why* was this bin called bivalent,
   and how much did each mark contribute. This directly answers your "hard to trace back
   which specific region is important."
3. **Reversibility index with a falsifiable test.** Quantify a barrier between chromatin
   states and then **predict, from control data alone, which regions an epigenetic drug
   will move** — validated on held-out EZH2i/LSD1i/DNMTi datasets. If this works it is the
   paper. Nothing else in the table above attempts prediction.

---

## 1. The quantities to define (the scientific core)

### Layer 0 — signal model

Binned counts `Y[b, m, s]` for bin *b*, mark *m*, sample *s* (default 200 bp bins, matching
ChromHMM for comparability).

- Normalization: sequencing depth size factors, input/IgG control, GC, and **spike-in
  scaling when available** (Drosophila S2 / barcoded nucleosome spike-ins in CUT&RUN).
  Spike-in is what turns "relative enrichment" into "amount of mark," and it is the honest
  foundation for the word *quantitative* in the package name. Make it optional with a
  reference-distribution fallback for public data that lacks it.
- Emissions: negative binomial / negative multinomial per state per mark. Overdispersion
  estimated per mark.
- **Joint fitting across all samples and conditions with shared emission parameters**, so
  state *k* means the same thing in every sample and only occupancy changes. This is the
  design decision that makes everything downstream comparable.

### Layer 1 — per-region quantitative readouts

For each bin, posterior `γ_b ∈ Δ^(K-1)` over K states.

- **State composition vector** — soft membership, not argmax. The primary output.
- **State entropy** `H(γ_b) = -Σ γ log γ` — ambiguity / plasticity proxy. Report a
  calibrated version, since raw HMM posteriors saturate (the epigenome-ssm critique).
- **Bivalency score** — not "was the bivalent state assigned," but a continuous score
  combining active mass A and repressive mass R, e.g. a balance term `2√(AR)/(A+R)` scaled
  by total mark magnitude. Balance × magnitude, reported as two separate numbers so a
  weak-but-balanced region is distinguishable from a strong-but-balanced one.
- **Mark attribution** — decompose the log-likelihood ratio between the called state and
  the runner-up into per-mark contributions. Output as a per-bin × per-mark table. This is
  the traceability feature; it is cheap to implement and will be disproportionately loved
  by bench biologists.

### Layer 2 — dynamics

- **Differential state** — Δγ between conditions with a replicate-aware test
  (Dirichlet-multinomial or bootstrap over replicates) and FDR control. Most existing
  differential-chromatin work is under-powered on replicates; do this properly.
- **Transition flux** — estimate the state-to-state flow between conditions using
  **optimal transport in emission space** (cost = distance between state emission
  profiles). This is the Waddington-OT idea moved from cells to genomic bins, and gives an
  interpretable "which state feeds which" matrix rather than a naive confusion table.
- **Per-region transition propensity** — how much of a region's posterior mass moved, and
  toward what.

### Layer 3 — landscape and reversibility

- Embed bins in a continuous latent space (start with PCA/NMF on normalized signal; a VAE
  only if it demonstrably beats it — resist the temptation).
- Density `ρ(z)` over bins → statistical potential `U(z) = -log ρ(z)`. States are basins.
- **Barrier height** between basins via minimum-energy-path / saddle-point search.
- **Reversibility index** per region = distance to the nearest saddle relative to barrier
  height.
- **The validation that makes this real:** fit on control/untreated data, predict the
  ranked set of regions that will change under EZH2i / LSD1i / DNMTi, evaluate against
  held-out treated data (AUPRC vs. baselines: raw signal magnitude, ChromHMM state change,
  distance-to-TSS).

> **Language discipline:** `U(z)` is a *statistical* potential, not a thermodynamic free
> energy. Call it a potential, define it operationally, and let the perturbation-prediction
> result carry the physical claim. Overclaiming here is the fastest route to a hostile
> review.

---

## 2. Staged roadmap

### v0.1 — "Quantitative states" (target: ~months 1–4)
Scope: `io`, `norm`, `model`, `quantify`. Outputs: state composition, entropy, bivalency
score, mark attribution. Deliverables: bigWig/BED tracks + AnnData tensor; head-to-head
benchmark vs ChromHMM, EpiCSeg, epigenome-ssm on Roadmap/ENCODE.
Success gate: bivalency score recovers reChIP-seq-validated bivalent promoters better than
ChromHMM state assignment.

### v0.2 — "Dynamics" (~months 4–7)
Scope: `dynamics`. Replicate-aware differential state, OT transition flux, transition
propensity. Applied to a differentiation timecourse and to a T-cell exhaustion timecourse.
Success gate: recovers known state transitions in ESC→NPC with FDR control, and finds
exhaustion-associated regions consistent with published TOX/PD-1 biology.

### v0.3 — "Landscape / reversibility" (~months 7–12)
Scope: `landscape`. Potential, barriers, reversibility index, and the perturbation
prediction benchmark.
Success gate: predicts drug-responsive regions above all baselines on held-out data.

### v0.4 — single-cell (~months 12–15)
scATAC / scCUT&Tag. Resolves the single biggest scientific confound in Layer 1: **bulk
bivalency can be a mixture artifact** — two cell populations, one K4me3+ and one K27me3+,
look identical to a truly bivalent nucleosome in bulk. Metacell-level state composition
plus a mixture-vs-bivalency test. Note the 2026 nano-CUT&Tag preprint is working this
ground; scope accordingly.

---

## 3. Package architecture

```
chromstate/
  io/          # BAM/bigWig/BED → binned counts; genome, blacklist, chunking
  norm/        # size factors, spike-in, input control, GC
  model/
    emissions.py   # NB / negative multinomial / Gaussian
    hmm.py         # forward-backward, Baum-Welch, posterior decoding (numba)
    joint.py       # multi-sample joint fit, shared emissions
    selection.py   # K selection: held-out likelihood, BIC, stability across seeds
  quantify/
    composition.py entropy.py bivalency.py attribution.py
  dynamics/
    differential.py transport.py propensity.py
  landscape/
    embed.py potential.py barrier.py
  annotate/    # genes, TSS, TF motifs, enrichment
  viz/         # tracks, heatmaps, landscape plots, state-flow river plots
  cli.py api.py
```

**Interop choices that drive adoption:**
- Emit **bigWig** for every continuous score so it opens in IGV. Non-negotiable — this is
  how bench biologists will actually evaluate the tool.
- Store the bin × sample × state tensor as **AnnData/zarr** for scanpy/muon interop and a
  clean path to the single-cell version.
- Ship a **ChromHMM-compatible mode**: same 200 bp bins, same input file layout, and a
  converter, so people can drop it into an existing pipeline.

**Engineering baseline:** `pyproject.toml` (hatch or uv), ruff + mypy, pytest with
simulation-based correctness tests (generate from a known HMM, recover parameters),
GitHub Actions CI, mkdocs-material docs with notebook tutorials, example Snakemake/Nextflow
workflow, MIT or BSD license, Zenodo DOI, release to PyPI **and bioconda**.

---

## 4. Benchmark datasets

- **Reference:** Roadmap/ENCODE 127 epigenomes with published 18-state ChromHMM models.
- **Bivalency ground truth:** ESC H3K4me3/H3K27me3; reChIP-seq bivalency in CD4+ memory
  T cells (Kinkley et al. 2016) — a real orthogonal validation set, not a circular one.
- **Differentiation:** ESC→NPC / mesoderm timecourse.
- **Perturbation (the key set):** EZH2 inhibitors (GSK126, tazemetostat), EED inhibitors,
  LSD1i, DNMTi — before/after histone-mark profiling.
- **Lab-adjacent applications:** T cell exhaustion timecourse (LCMV cl13, TOX, PD-1
  blockade partial reversibility) — exhaustion reversibility *is* the reversibility-index
  question, and you have local expertise and possibly unpublished data. Aging/senescence
  H3K27me3 loss is a second natural application.

---

## 4b. Method comparison — design and current results

Implemented in `chromstate.benchmark`. Run it with `chromstate compare -o benchmark_out`.

### How the comparison is set up

Competing methods are reimplemented **as their published statistical models, on
ChromState's own bins**, rather than shelled out to the original tools:

| Row | Model | Stands for |
|---|---|---|
| ChromState | Negative-binomial HMM on raw counts, joint across samples | this package |
| ChromHMM-like | Poisson binarisation → multivariate Bernoulli HMM | ChromHMM |
| Segway-like | Diagonal Gaussian HMM on log signal | Segway, epigenome-ssm |
| GMM | Gaussian mixture, no spatial model | isolates what the HMM chain contributes |
| k-means | Hard assignment, no probability | isolates what soft assignment contributes |
| Naive signal | Harmonic mean of two normalised tracks, no model at all | the control that matters |

Three deliberate choices, each of which can be argued with:

1. **Reimplementation, not the real tools.** Buys byte-identical input and a benchmark
   that runs anywhere, so differences are attributable to the *model* rather than to
   preprocessing. The cost is that tool-specific heuristics are not reproduced. Before
   publication the ChromHMM row must be re-run against real ChromHMM —
   `chromstate export-chromhmm` writes binarised input in ChromHMM's own format for
   exactly this. **Treat the current ChromHMM numbers as provisional until that is done.**
2. **Equal compute budget.** Every HMM method gets the same fixed number of restarts
   (`adaptive_restarts=False`). ChromState's own default keeps restarting until an
   optimum is hit twice, which is right for analysis but would hand it a larger budget
   than the baselines and inflate the comparison.
3. **No cross-model likelihood comparison.** NB and Bernoulli likelihoods are computed
   over different data — counts versus binarised calls — so they are not on the same
   scale. Held-out likelihood is only used to compare models fit to the same
   representation.

The `Naive signal` row is the honest control: if a fitted model cannot beat two tracks
and one formula at finding bivalent regions, the model is not earning its complexity.

### Results (simulated data, ground truth known)

16,000 bin-rows, 6 marks, 2 conditions, 7 true states, 35% of bins blurred between two
states to create genuinely ambiguous regions. Seed 1.

| method | ARI | accuracy | bivalency AUPRC | tracks magnitude (ρ) | overconfidence | saturated calls |
|---|---|---|---|---|---|---|
| **ChromState (NB-HMM)** | **0.823** | **0.910** | **0.816** | **0.883** | **0.054** | 0.758 |
| ChromHMM-like (Bernoulli) | 0.679 | 0.777 | 0.224 | 0.755 | 0.092 | 0.437 |
| Segway-like (Gaussian) | 0.814 | 0.904 | 0.736 | 0.875 | 0.067 | 0.780 |
| GMM (no spatial model) | 0.575 | 0.783 | 0.527 | 0.836 | 0.136 | 0.459 |
| k-means | 0.657 | 0.830 | 0.537 | 0.837 | 0.170 | 1.000 |
| Naive signal (no model) | — | — | 0.682 | — | — | — |

Random-baseline bivalency AUPRC is 0.081.

**Detecting which regions responded to a perturbation** (half of bivalent bins switched
to active promoters):

| method | responder AUPRC | AUROC |
|---|---|---|
| **ChromState** | **0.988** | 0.999 |
| Segway-like | 0.982 | 0.999 |
| ChromHMM-like | 0.403 | 0.900 |

Random baseline 0.054.

### What these results actually say

**The binarisation result is the strongest finding.** ChromHMM-like scores 0.224 on
bivalency detection — *worse than the naive two-track formula at 0.682*, and 3.6× worse
than ChromState. Binarisation is measured directly: among bins it calls "present", it
collapses a 4.9–7.9× range of read counts onto the single value 1. Bivalency is
intrinsically about the *relative amount* of two marks, so a representation that discards
magnitude cannot recover it, and the perturbation result (0.403 vs 0.988) says the same
thing about detecting change.

**Spike-in normalisation is not optional.** Under a simulated EZH2-inhibitor-like loss
where H3K27me3 drops to 35% genome-wide, depth and median-of-ratios both return a size
factor that absorbs the biology (`biology_absorbed` ≈ 0.35 against a correct value of
1.0). The drug effect is erased before any model sees it. Only spike-in recovers it
exactly.

**Where ChromState does not win, and it matters:**

- **Segway-like Gaussian is nearly as good** — ARI 0.814 vs 0.823, responder AUPRC 0.982
  vs 0.988. On this simulation the gain from a proper count model over a log-Gaussian is
  real but small. The honest reading is that *not binarising* is what buys most of the
  improvement; the specific choice of count distribution is a refinement. Do not oversell
  the negative binomial.
- **ChromState is badly calibrated in absolute terms.** 75.8% of its calls sit above 0.99
  posterior, and mean entropy is 0.049. It is the *best* on overconfidence (0.054) and
  better than ChromHMM-like (0.092), but "least bad" is not "good". Since one of the
  package's selling points is knowing which calls to trust, this needs explicit work —
  temperature scaling or a hierarchical model — before the entropy score can be presented
  as a calibrated plasticity measure. **This is currently the weakest claim in the plan.**
- **The simulation is generated from the same model family ChromState fits.** That biases
  in ChromState's favour and no amount of blurring fully removes it. Real-data validation
  is not a nice-to-have; it is what makes any of this credible.

### Running the real tools

`tools/` (see `tools/README.md`) is the harness for exactly this: it runs real ChromHMM,
and any tool described by a command template in `tools/tools.json`, on ChromState's own
bins, maps each tool's output back onto the bin grid by coordinate, and scores everything
with one metric registry — truth-based metrics on simulated data, truth-free ones
(segment shape, coverage, posterior decisiveness, pairwise agreement, runtime, memory) on
real data. Tools that are absent or crash stay in the table with a reason.

**Real ChromHMM has been run through the harness** (ChromHMM 1.28 from
`jernst98/ChromHMM`, October 2026): the adapter needed `-noenrich`, because `LearnModel`
otherwise aborts in its post-hoc enrichment step on chromosomes that do not match the
named assembly, *after* writing the segmentation. Results over three seeds are in
Tutorial 1 §7.1; real ChromHMM scores above its reimplementation on every metric, and its
bivalency detection is the least stable number in the table (AUPRC 0.21–0.73).

The adapters for EpiCSeg, Segway, epigenome-ssm, EpiSegMix, diHMM and normR ship
**disabled and unverified**:
their command lines are transcribed from published documentation and have not been run
against the real software here. That flag is printed by `toolbench check` and repeated in
every report, and it should stay set until someone has watched each tool run.

### Still required before publication

1. ~~Re-run the ChromHMM row against **real ChromHMM**~~ — done on simulated data
   (Tutorial 1 §7.1); still to be done on real data (item 2).
2. Re-run the whole comparison on **real data** — Roadmap/ENCODE with published
   segmentations, and reChIP-seq bivalency as orthogonal ground truth.
3. Add **epigenome-ssm** and **EpiCSeg** as rows. They are the closest prior art (§0) and
   their absence is the most obvious gap a reviewer will name. *Partly addressed:*
   EpiCSeg's negative multinomial, EpiSegMix's zero-inflated NB, diHMM's two-level
   structure and normR's enrichment calling are reimplemented as `epicseg_like`,
   `episegmix_like`, `dihmm_like` and `normr_like` on identical bins, and disabled
   adapters for the real EpiSegMix, diHMM and normR are in `tools/tools.json`.
   epigenome-ssm and the real EpiSegMix/diHMM/normR software still need to be run.
4. Fix or honestly caveat **posterior calibration**. *Partly addressed:* `calibrate.py`
   fits a temperature against replicate reproducibility (real data) or known labels
   (simulation) and cuts saturated calls from ~85% to ~65-73% with the calibration error
   roughly halved. It does not fix the underlying cause — the model's conditional
   independence across marks — and a reproducibility criterion cannot detect a bias that
   every replicate shares.
5. Report **variability across seeds and simulation parameters**, not a single run.
   *Partly addressed:* Tutorial 1 §7.1 reports mean ± s.d. over three seeds; simulation
   parameters (blur, dispersion, depth) have not been swept.
6. Run `compare_algorithms` and `state_importance` on the **CD8 data** and report the
   held-out likelihood of `nb` versus `zinb` and the state ranking with intervals, so
   that the method-level validation in Tutorial 1 §7.2–7.3 has a real-data counterpart.

---

## 5. Validation strategy, per claim

| Claim | Test | Failure mode it guards against |
|---|---|---|
| States are quantitative | Recover known spike-in-scaled mark amounts; correlate state scores with qPCR/ChIP-qPCR magnitude | Scores are just re-scaled read depth |
| States are comparable across conditions | Simulated global mark loss: occupancy should shift, state definitions shouldn't | Compositional artifact masquerading as biology |
| Bivalency score is real | reChIP-seq concordance; single-cell mixture test | Bulk mixture artifact |
| Attribution is meaningful | Ablate a mark, check attribution collapses for that mark only | Post-hoc storytelling |
| Reversibility index predicts | Held-out drug-response AUPRC vs. 3 baselines | The landscape is a pretty picture with no predictive content |

---

## 6. Risks, ranked

1. **Prior-art overlap** (highest). Mitigation: benchmark explicitly against EpiCSeg,
   EpiSegMix, epigenome-ssm, chromstaR in the first month, and position on prediction.
   Do a proper systematic literature + bioRxiv sweep before writing any modeling code.
2. **The landscape layer doesn't predict anything.** Mitigation: v0.1 and v0.2 are
   independently publishable and useful; the package does not depend on v0.3 succeeding.
3. **Bulk bivalency is a mixture artifact.** Mitigation: report it as a caveat from day 1,
   build the single-cell test into the roadmap rather than bolting it on after review.
4. **K selection instability.** Mitigation: stability analysis across seeds/subsamples
   shipped as a first-class diagnostic, not a footnote.
5. **Spike-in data is scarce in public archives.** Mitigation: optional with a documented
   fallback, and be explicit in the docs about what "quantitative" means without it.
6. **Scope creep across three layers.** Mitigation: hard gates between versions; do not
   start v0.2 until v0.1's success gate is met.

---

## 7. First two weeks — concrete

1. Systematic prior-art sweep (PubMed + bioRxiv, last 3 years) on quantitative chromatin
   state models. Write it up as `docs/prior-art.md`. **Do this before writing model code.**
2. Scaffold the repo: `pyproject.toml`, package skeleton, CI, ruff/mypy/pytest.
3. Implement `io` + `norm` end to end on one real dataset; produce a binned count matrix.
4. Reproduce a published ChromHMM segmentation on that dataset — this is the baseline
   everything is measured against, and it forces the I/O layer to be correct.
5. Implement the NB-HMM with posterior decoding; validate by parameter recovery on
   simulated data.
6. Only then: the first quantitative score (state composition + entropy) and a look at
   whether it says anything ChromHMM's posteriors don't.

---

## 8. Naming

`ChromState` is close to **chromstaR**, an existing Bioconductor package for combinatorial
and differential chromatin state analysis. Same domain, near-identical name — that is a
real citation-confusion and search-collision problem. From your shortlist, `ChromTrace`
(matches the traceability contribution) and `ChromSpace` (matches the landscape
contribution) are cleaner. Decide before the first public release; renaming after a DOI is
painful.

The R wrapper is named `chromstatR`, which collides even harder with `chromstaR` — one
letter apart. Strongly consider renaming both together.

---

## 9. Implementation status (v0.1)

174 Python tests and 24 R tests pass. `chromstate demo -o demo_out` runs the whole thing
end to end in ~9 s and writes 13 figures, 12 tables, and a report.

### Built

| Layer | Module | State |
|---|---|---|
| Genome binning, blacklist | `genome.py` | done — 200 bp default, ChromHMM-compatible |
| IO: BAM / BED reads / peak BED | `io/` | done — reads ChromHMM's `cellmarkfiletable` unchanged |
| Normalisation: depth, median-of-ratios, spike-in | `norm.py` | done |
| NB / ZINB / Poisson / negative-multinomial / Bernoulli / Gaussian emissions, tied (hierarchical) wrapper | `model/emissions.py` | done — single-pass pair tables |
| HMM: fused chain-parallel forward-backward, Baum-Welch, Viterbi | `model/hmm.py` | done, numba `prange` over chromosomes; reference path kept and tested equal |
| Algorithm selection (held-out likelihood, calibration, fragmentation, runtime) and E-step speed benchmark | `benchmark/algorithms.py` | done |
| State importance for expression: permutation, closed drop-one, ridge coefficients, bootstrap CIs; representation comparison; planted-effect simulator | `importance.py`, `simulate.py` | done — validated against planted effects |
| Automatic state mnemonics + Roadmap colours | `model/annotate.py` | done |
| Model selection by held-out likelihood | `model/core.py` | done |
| Composition, entropy, margin, bivalency, attribution | `quantify.py` | done |
| Differential state, OT flux, transition propensity | `dynamics.py` | done |
| Replicate noise floor, region-level testing (moderated t) | `dynamics.py` | done — v0.2 |
| Posterior calibration by temperature scaling | `calibrate.py` | done — v0.2 |
| Genes, promoters, bin-to-feature mapping, per-gene trace | `features.py` | done — v0.2 |
| RNA join, per-state association, traceable-gene ranking | `rna.py` | done — v0.2 |
| Benchmark harness + 10 methods (incl. EpiCSeg-, EpiSegMix-, diHMM-, normR-like) + metrics | `benchmark/` | done |
| Figures (ChromHMM-standard + ChromState-specific) | `viz.py` | done — 13 figures |
| CLI (`fit [--gtf --expression]`, `selectk`, `compare`, `demo`, `export-chromhmm`) | `cli.py` | done |
| R wrapper over reticulate | `R/chromstatR/` | done |

### Not built (per the staged roadmap)

- **Landscape / reversibility (v0.3)** — no `landscape.py`. The potential, barrier
  heights, and reversibility index are designed in §1 but not implemented. This is the
  headline scientific claim and it is still entirely unwritten.
- **Single-cell (v0.4)** — nothing.
- **bigWig output** — scores currently emit as TSV/BED; `pyBigWig` was not available in
  this environment. IGV-viewable continuous tracks are important for adoption and should
  be added early.
- **AnnData export** — `to_anndata()` exists but is untested against the wider ecosystem.

### Five real bugs the tests caught, worth remembering

All were silent — the fit converged, reported a likelihood, and produced clean figures
while being wrong. None would have been visible without ground-truth simulation.

1. **Dispersion initialised from the marginal variance.** The genome-wide variance of a
   mark is dominated by differences *between* states, so it implied enormous
   overdispersion, flattened the likelihood, and collapsed every state onto one profile.
   Fixed by estimating dispersion from within-cluster residual variance.
2. **States seeded by quantiles of total signal.** This orders states only by overall
   magnitude, so Polycomb (H3K27me3-high, total ≈ 21) and heterochromatin (H3K9me3-high,
   total ≈ 18) were seeded on top of each other and EM could never separate them. *Every*
   random restart converged to the same merged solution ~3,900 log-likelihood units below
   the true optimum, while reporting itself converged. Fixed by seeding with k-means on
   the per-mark profile; ARI went 0.857 → 0.993 and the recovered dispersion matched the
   simulator's true value.

3. **Uninitialised memory fed into the M-step.** The responsibility matrix was allocated
   with `np.empty` and filled only for rows covered by the fitted chains. Held-out model
   selection fits on a subset of chromosomes, so the uncovered rows carried arbitrary
   memory contents into the M-step *as if they were responsibilities*. Every emission
   estimate was corrupted and all states collapsed — while EM reported convergence, and
   every candidate K returned a log-likelihood identical to six decimal places. Fixed by
   zero-initialising. This one only ever fired in `select_n_states`, which is precisely
   the code path a user relies on to choose K.
4. **The permutation null counted permutations, not group assignments.** Every split of
   2n samples into two groups is reachable by `n_a! · n_b!` permutations, and the
   statistic is symmetric under swapping the groups, so each distinct null value was
   counted many times. The exceedance *ratio* stayed unbiased, but the `+1` pseudo-count
   in `(exceed+1)/(done+1)` was diluted by the same factor: a 2-versus-2 design reported
   a p-value floor of 1/21 = 0.048 when the honest floor is 1/3 = 0.333. That is the
   difference between "this design cannot resolve significance at all" and a bin
   appearing to clear FDR. Fixed by deduplicating on the unordered split. The honest
   floors are 0.333 (2v2), 0.100 (3v3), 0.029 (4v4) — so **four replicates per group is
   the floor-imposed minimum** before any bin can reach q < 0.05, and that is before
   multiple testing over hundreds of thousands of correlated bins, which is why
   differential calling has to aggregate to regions.

5. **The simulator perturbed every sample after the first.** `simulate()` had no notion
   of a condition group: `is_treated = si > 0`, so in a six-sample design named
   `ctrl1..3, drug1..3` five of the six samples were treated. Nothing failed — the fit
   converged, the contrast looked strong — but the "within-condition replicate pairs"
   used to measure the noise floor were really control-versus-treated comparisons, which
   put the floor at 0.71 and made every real change look like noise. Fixed by an explicit
   `treated=` argument, recorded on the truth object. The general lesson matches the
   others: a replicate-aware analysis silently produces a plausible number when the data
   has no replicates in it.

The trap in (2) is self-reinforcing — merging two states inflates the within-state
variance of their defining marks, which drives those marks' dispersion down, which
flattens the likelihood for exactly the marks that would have separated them.

The common thread across all three: **the fit reported success every time.** Convergence
flags, likelihood values, restart agreement and clean figures were all produced by
thoroughly broken models. Nothing but comparison against known ground truth would have
caught them. Two of the three were invisible in the main `fit()` path and only appeared in
a specific configuration — held-out selection, or a mark panel where two states share a
similar total signal. `fit()` now records `restart_hits_at_best`, `min_state_separation`
and a `collapsed` flag, and warns on a degenerate fit, but the real defence is that every
claim in §4b is measured against simulated data where the answer is known.
