# Changelog

All notable changes to ChromState are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [1.1.0] - 2026-11-18

### Added
- Documentation site built with Sphinx (`docs/source`), with one page per subcommand,
  file formats, tutorials, benchmarks, a Q&A and an API reference, deployed to GitHub
  Pages by `.github/workflows/docs.yml`.
- Paper figures in Nature format (`tools/make_paper_figures.py`) and the helpers behind
  them: `viz.figure_mm`, `viz.panel_letter`, `viz.check_nature_style`, `viz.save_figure`.
- Real-data validation on ENCODE ChIP-seq (GM12878, K562, H1) and a replicate-level CD8
  T-cell CUT&RUN comparison, including a null contrast between two batches.
- Project files: `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `environment.yml`, issue and pull-request templates, and a release workflow.
- `fast` extra (`pip install "chromstate[fast]"`) that installs Numba.

## [1.0.0]

### Changed
- Version raised to 1.0.0; package metadata now lists project URLs and supported Python
  versions.

### Changed (statistics)
- Region × state tests, the per-region test and the ANOVA now use the empirical logit of
  the region's own bin count instead of a logit clipped at 1e-4, take the variance prior
  from the tested regions only, and skip states below half a bin in every group. This
  removes the flat ceiling in volcano plots and the pile-up of p-values at 1. Numbers
  from differential analyses run before this change will shift when rerun.
- The ANOVA plot shows the largest difference between conditions against q, and the
  region × state heatmap shows the change in share.
- Figures no longer carry explanatory subtitles (`viz.SHOW_SUBTITLES`).
- `simulate(perturbation={"sample_dose": ...})` gives a treated sample a weaker response.

### Fixed
- `quantify.py` used `Sequence` without importing it.
- All lint errors in `src` and `tests`.

## [0.3.0]

### Added
- States defined from given marks (`StateDefinitions`, `--states`, `--anchor`).
- Region × state statistics (`differential_report`): moderated t-tests, ANOVA for three
  or more conditions, region-level noise floor.
- Expression prediction and state ranking (`chromstate predict`, `predict_expression`).
- Single-cell module (`chromstate.sc`): metacells, pseudobulk, bulk-to-single-cell
  state transfer.

## [0.2.0]

### Added
- `chromstate run`: fit at 200 bp, score every 10 bp, gene-level and genome-wide modes,
  promoter / distal / body quantification, condition comparison, alluvial plots.
- Harness for running external segmentation tools (`tools/toolbench`): ChromHMM,
  EpiCSeg, normR, diHMM, Segway.

## [0.1.0]

### Added
- Negative-binomial HMM on raw counts with posterior-based quantification, bivalency
  scoring, mark attribution, temperature calibration, and the R interface `chromstatR`.
