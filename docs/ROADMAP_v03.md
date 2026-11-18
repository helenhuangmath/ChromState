# ChromState v0.3 — five questions, what exists, what was added, what is next

Written 2026-11-02 against the five asks for the package:

1. quantify chromatin states from a given set of marks;
2. compare conditions statistically, by region and by state;
3. measure which state predicts gene expression, and show it on real data;
4. benchmark against existing tools;
5. carry the states into single cells (scRNA + scATAC).

Each section says what the code already did before this round (v0.2), what v0.3 adds,
how it is tested, and the open questions that are worth a decision.

---

## 1. Quantify chromatin states from given marks

**Before.** `fit` learned K states without supervision from whatever marks the manifest
lists, then named them after the fact (`model/annotate.py`). Every bin got a posterior over
those states; `quantify` turned it into composition, entropy, bivalency and per-mark
attribution. Two gaps: (a) the *user* could not say which states to measure, so the
vocabulary changed from dataset to dataset and two experiments fitted separately were not
comparable; (b) there was no plain "quantify these states in these BED regions" call.

**Added (v0.3).**

| Piece | Where | What it does |
|---|---|---|
| `StateDefinitions` | `definitions.py` | states x marks pattern table (`+` high, `-` low, `.` free); from a dict, a TSV, a text file (`TssA: H3K4me3+ H3K27me3-`), or `StateDefinitions.default(marks)` (Roadmap vocabulary restricted to the marks present) |
| `fit(..., definitions=, anchor="seed"\|"fixed")` | `model/core.py` | the patterns seed the NB emissions and fix the names; `fixed` pins every `+`/`-` mean through EM so the measured quantity is exactly the stated pattern; `n_states` may exceed the table (extra states are learned and named automatically) |
| `AnchoredNegativeBinomial` | `definitions.py` | NB emission with an anchor mask re-imposed after each M-step |
| `match_states_to_definitions` | `definitions.py` | Hungarian one-to-one matching of fitted states to patterns; also useful to align two independent fits to one vocabulary |
| `quantify_regions(seg, bed)` | `quantify.py` | composition (fraction and bp per state), dominant state, entropy for any BED table, per sample |
| `chromstate fit --states default\|FILE --anchor seed\|fixed` | `cli.py` | the same from the command line |

**Tests.** `tests/test_v03_features.py`: parsing and default vocabulary; file round trip;
fit with definitions recovers the simulated states (ARI > 0.6) with the right names and
emission ordering; `fixed` leaves anchored means untouched; extra states allowed;
one-to-one matching; region quantification sums to 1 and to the region's bp.

**Open questions.**
- *Which anchor should be the default?* `seed` keeps the likelihood honest; `fixed` makes
  cross-dataset comparison exact but can mis-fit a dataset whose marks behave differently.
  A middle road is a Gaussian prior on the log-mean (strength as a parameter). Not done.
- *Levels.* `+` currently means "the mean of the top 5% of bins", `-` "the mean of the
  bottom 50%". A richer vocabulary (`++` for very high, or numbers) is a two-line change
  in `_SYMBOLS` once there is a use for it.
- *Given marks only.* If the question is "use only these marks, ignore the others",
  that is a manifest edit; nothing in the model needs to change.

## 2. Compare conditions statistically, by region and by state

**Before.** `differential_regions` (per region: omnibus moderated *t* on the most-changed
state, Bonferroni across states, BH across regions, replicate noise floor),
`state_tests` (per state, genome share: Welch *t* on replicates and a block bootstrap),
`compare_conditions` (per-condition deltas, bp flows), `flow_test`. The per-region x
per-state table (every state's change and its own test) did not exist, there was no
multi-condition test, and the results were spread over three functions.

**Added (v0.3).**

| Piece | What it reports |
|---|---|
| `region_state_tests` | long table: region x state x contrast with share in each group, delta, log2 ratio, moderated *t*, *p*, *q* (BH across regions within each state), `significant` with an optional effect-size floor `min_delta` |
| `moderated_f_test` | limma-style one-way *F* per feature across >= 3 replicated conditions, shrunk within-group variance |
| `differential_report(seg, groups, regions, reference)` | one call: per-region omnibus tables per contrast, the region x state long table, a per-state summary (regions gaining / losing at FDR, mean and median change, regions where the state drives a confident change) merged with the genome-wide `state_tests`, the ANOVA table when there are three or more replicated conditions, the noise floor, and a notes file. `.write(outdir)` saves everything. |
| pipeline | `chromstate run` now writes `differential/promoter_region_state_tests.tsv`, `promoter_state_summary.tsv`, `promoter_genome_state_tests.tsv` (and `promoter_region_state_anova.tsv` with >= 3 replicated conditions) |

**Figures (added 2026-11-02, `viz_stats.py`, wired into `DifferentialReport.write`).**
Omnibus volcano with the noise floor; per-state volcano; region x state heatmap of signed
−log10 q; gain/loss per state; observed-versus-permuted p-value histograms; replicate
concordance with the noise-floor band; ECDF of the change statistic for replicate versus
condition pairs; ANOVA volcano; and in the benchmark, precision-recall / ROC curves for the
perturbation metric (`toolbench.report.plot_perturbation_curves`). Examples in
`docs/figures/statistics/`. Two corrections came out of drawing them: the bin-level noise
floor is too permissive for region statistics, so `region_noise_floor` (quantile of the
region statistic over replicate pairs) is now used by the report and the pipeline; and a
state absent from a region in both groups is no longer tested (p = 1 by construction),
which removes the spike at 1 and shrinks the multiple-testing burden.

**Tests.** planted shift detected by the *F*-test; compositional deltas sum to zero
within a region; the report's region statistic tracks the fraction of true responders in
a region (Spearman > 0.5); three-condition ANOVA runs.

**Open questions.**
- *Replicates.* All tests need two or more chromatin replicates per condition. The
  CD8 data at hand are pooled tracks, so the real-data tutorial cannot exercise this;
  the per-replicate BAMs noted in `analysis/scireports2025/README.md` would.
- *Compositional test.* Per-state tests on logit shares are correlated across states.
  A Dirichlet-multinomial or CLR-Hotelling omnibus would test the whole composition at
  once; the current omnibus (Bonferroni over the most-changed state) is conservative.
- *Effect-size shrinkage* (apeglm-style) for ranking regions with few replicates.

## 3. Which state predicts gene expression

**Before.** `regulation.rank_states_for_expression` (state importance: single-state
`marginal_rho`, permutation `unique_r2`; representation comparison), `rank_parts_for_expression`
(promoter vs enhancer vs body), the Sci Rep analysis scripts. Scattered; no per-gene
predictions; no single report.

**Added (v0.3).**

| Piece | What it does |
|---|---|
| `predict_expression(gene_q, expression= \| lfc=, contrast=)` | held-out predictions for every gene (chromosome-grouped folds), Spearman with bootstrap CI and R^2, state importance, representation comparison with paired bootstrap differences, part ranking when the table has P/E/B, a `summary` naming the best single state and the most irreplaceable state, `REPORT.md` and figures (`state_importance`, `predicted_vs_observed`, `representations`, `part_ranking`) |
| `chromstate predict --genes gene_quantification.tsv.gz --expression X \| --lfc F --contrast A,B` | the same from the command line, on the table `chromstate run` already writes |
| `tools/realdata/predict_cd8.py` + `../testing/sbatch/predict_cd8.sbatch` | the real-data test: CD8 T-cell CUT&RUN (chr19), RNA level in three cell states and the Cl13-vs-Naive change; unsupervised K=12 and the default definitions side by side |
| [Tutorial 3](tutorials/03_expression_prediction.md) | the walk-through with the numbers from that run: level rho 0.77, change rho 0.39-0.41, TssA the only irreplaceable state, hard labels at 0.13 for change, defined states equal to learned ones |

**Open questions.**
- Expression is bulk RNA per condition; with scRNA (section 5) the same call works on
  pseudobulk expression per metacell.
- The change task uses DESeq2 log2 fold change as target; a shrunken LFC or a signed
  -log10 *p* would weight genes differently. Make it an option if it matters.

## 4. Benchmark against existing tools

**Before.** `tools/toolbench`: real ChromHMM, EpiCSeg, normR, diHMM, Segway run on the
same bins; simulated truth metrics (ARI, bivalency, calibration, magnitude) and
truth-free metrics (shape, agreement, cost); real-data metrics (promoter recovery,
expression level and change prediction); `docs/tool_comparison.md`.

**Added (v0.3).** A `perturbation` metric: detection of the simulated responder bins
from each tool's control-versus-treated change (posterior total variation, or a changed
hard call), AUPRC / AUROC against the known responders. This is the metric the
condition-comparison claim rests on and it was missing from the harness; tools without
posteriors reach one point on the PR curve, which is the point.

**Run (2026-11-02, `../testing/sbatch/simbench.sbatch`, 200k bins, K = 7, nine tools).** Responder
AUPRC: ChromState 0.63, ChromHMM-like 0.58, Segway-like 0.54, EpiCSeg 0.40, ChromHMM 1.28
0.21, k-means 0.13, diHMM 0.07, normR 0.06 (baseline 0.03). Table in
`docs/tool_comparison.md` §2.1b.

**Proposed next (not done, each is a job).**
1. ~~Re-run the simulated comparison with the perturbation metric~~ done; repeat at
   several responder fractions and effect sizes to get a curve rather than one point.
2. A *differential* real-data metric: split-half reproducibility of each tool's
   Naive-vs-Cl13 changes once per-replicate BAMs are binned.
3. epigenome-ssm (continuous states; the nearest prior art for "quantitative states")
   is still a command template only. chromstaR (Bioconductor, differential combinatorial
   states) is the closest tool to item 2 and is not in the harness. Both are install
   work, not code.
4. Given-mark definitions (item 1) versus ChromHMM's fixed model: compare a `fixed`
   anchor fit with the same patterns imposed on ChromHMM emissions.

## 5. Single cells: scRNA + scATAC

**Before.** Nothing (a `to_anndata` export only).

**What is and is not possible.** Histone marks are not measured in a multiome
experiment, so a histone state cannot be *fitted* in those cells. Three routes are
implemented in `chromstate.sc`, in order of directness:

1. **Single-cell histone data** (scCUT&Tag, nano-CT, multimodal CUT&Tag): aggregate to
   metacells (`metacells`, `pseudobulk`), build a joint dataset (`dataset_from_pseudobulk`),
   and every existing function applies with metacells as samples. States are shared across
   metacells because the emissions are. This is the honest version of "ChromState in single
   cells".
2. **scATAC + scRNA with a matched bulk histone dataset** (the CD8 case: bulk CUT&RUN exists,
   multiome could be generated): `StateTransfer` learns on the bulk bins how the bulk state
   posterior relates to proxies measurable in cells (accessibility in and around the bin,
   nearest-gene expression, TSS distance), reports the chromosome-held-out accuracy of that
   mapping (the ceiling on what the single-cell scores mean), and `project` scores every
   metacell bin with the bulk vocabulary, returning a `Segmentation` with a model in proxy
   space so `state_summary`, `compare_conditions`, `differential_report`,
   `gene_quantification` and `predict_expression` (with pseudobulk scRNA) run unchanged.
3. **scATAC + scRNA alone**: fit the NB-HMM on pseudobulk accessibility and gene-activity
   tracks as "marks". These are accessibility states, and should be named as such.

**Tests.** pseudobulk sums and dataset construction; metacells respect labels; transfer
from a simulated accessibility proxy recovers the active promoter state (recall > 0.5)
and the downstream summary / comparison functions run on the projected segmentation.

**Open questions, in order of importance.**
- Which single-cell dataset to use first. For CD8 T cells the published LCMV multiome
  or scATAC atlases would pair with the Sci Rep bulk marks; the transfer ceiling will be
  modest for repressive states (Polycomb vs heterochromatin are both closed) and good for
  active ones.
- Transfer features: adding motif deviations or gene-body accessibility is a one-line
  change in `proxy_features`; a per-cell (not metacell) version needs smoothing over a
  kNN graph, which is the next step once metacell results look right.
- Validation: hold out a bulk cell type (fit transfer on TN + TMEM, project TEX
  accessibility, compare with the TEX bulk states) is the test to run before trusting
  any single-cell map.

---

## Running everything on PARCC

All jobs go through `sbatch` (see `../testing/sbatch/`, outside the repository):

```bash
SB="SLURM_CONF=/cm/shared/apps/slurm/etc/slurm/slurm.conf /cm/local/apps/slurm/24.11/bin/sbatch"
sbatch ../testing/sbatch/run_tests.sbatch                                     # full test suite
sbatch --export=ALL,PYTEST_ARGS="tests/test_v03_features.py" ../testing/sbatch/run_tests.sbatch
sbatch --export=ALL,CHROMS=chr19,K=12,RUNTAG=chr19_k12 ../testing/sbatch/predict_cd8.sbatch
sbatch --export=ALL,CHROMS=chr19,K=12,RUNTAG=chr19_defs,EXTRA="--definitions default" ../testing/sbatch/predict_cd8.sbatch
```
