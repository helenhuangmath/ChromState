[![CI](https://github.com/helenhuangmath/ChromState/actions/workflows/ci.yml/badge.svg)](https://github.com/helenhuangmath/ChromState/actions/workflows/ci.yml)
[![Docs](https://github.com/helenhuangmath/ChromState/actions/workflows/docs.yml/badge.svg)](https://github.com/helenhuangmath/ChromState/actions/workflows/docs.yml)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)

# ChromState: quantitative chromatin state analysis

**Latest release:** 1.1.0 —
[GitHub](https://github.com/helenhuangmath/ChromState) ·
[Changelog](CHANGELOG.md)

## Introduction

Histone modifications combine into chromatin states: active promoters, enhancers,
Polycomb-repressed and bivalent regions, heterochromatin. Tools such as ChromHMM assign
one state label to each genomic bin. ChromState keeps the same idea but treats the
states as **quantities**. It fits a hidden Markov model with negative-binomial
emissions to raw read counts, and reports for every bin the probability of each state.
From those probabilities it computes how much of a promoter, gene or any region is in
each state, tests how that changes between conditions with replicates, and measures
which states predict gene expression.

ChromState takes ChIP-seq, CUT&RUN, CUT&Tag or ATAC-seq data as BAM, BED, bigWig or
peak files, in the same sample-by-mark table that ChromHMM uses.

## What it does

| Task | Command | Python |
|---|---|---|
| Learn chromatin states and segment the genome | [`chromstate fit`](docs/fit.md) | `cs.fit` |
| Choose the number of states | [`chromstate selectk`](docs/selectk.md) | `cs.select_n_states` |
| Quantify states per gene, promoter or region | [`chromstate run`](docs/run.md) | `cs.gene_quantification`, `cs.quantify_regions` |
| Compare conditions, with alluvial plots | [`chromstate run --groups`](docs/run.md) | `cs.differential_report`, `cs.compare_conditions` |
| Predict expression and rank states | [`chromstate predict`](docs/predict.md) | `cs.predict_expression` |
| Benchmark against other models | [`chromstate compare`](docs/compare.md) | `chromstate.benchmark` |

## Changes for ChromState 1.1.0

### Features

- One command, [`chromstate run`](docs/run.md), fits at 200 bp, scores every 10 bp,
  quantifies promoters, gene bodies and distal regions, compares conditions and links
  the result to RNA-seq.
- Region × state statistics with replicates: moderated t-tests, an ANOVA for three or
  more conditions, and a noise floor estimated from replicate pairs.
- Expression prediction on held-out chromosomes with a ranking of the states that
  carry the signal ([`chromstate predict`](docs/predict.md)).
- States can be learned or defined from a pattern table (`--states`).

### Validation

- Compared with the published ChromHMM, EpiCSeg, normR, diHMM and Segway software on
  simulated data, CD8 T-cell CUT&RUN and ENCODE ChIP-seq. See [Benchmarks](docs/benchmarks.md),
  which reports the cases where ChromState is not the best tool.

### Documentation

- This site, with a page for every subcommand and an API reference.

The full list is in the [changelog](CHANGELOG.md).

## Install

ChromState needs Python 3.9 or later.

```bash
pip install "chromstate[all] @ git+https://github.com/helenhuangmath/ChromState.git"
```

See [Install ChromState](docs/INSTALL.md) for conda, optional dependencies and the R
interface.

## Usage

Run the simulated demo (about 15 s):

```bash
chromstate demo -o demo_out
```

Fit a 12-state model to your own data:

```bash
chromstate fit --manifest cellmarkfiletable.txt --chrom-sizes mm10.chrom.sizes \
               -k 12 --blacklist mm10.blacklist.bed -o fit_out
```

Quantify every gene, compare two conditions and link to expression:

```bash
chromstate run --manifest cellmarkfiletable.txt --chrom-sizes mm10.chrom.sizes \
               --gtf genes.gtf -k 12 --groups "Naive=n1,n2;Exhausted=e1,e2" \
               --expression rna_tpm.tsv -o run_out
```

Predict expression from the per-gene state table:

```bash
chromstate predict --genes run_out/genes/gene_quantification.tsv.gz \
                   --expression rna_tpm.tsv -o predict_out
```

| Subcommand | Description |
|---|---|
| [`fit`](docs/fit.md) | Fit a model and write states, scores and figures. |
| [`selectk`](docs/selectk.md) | Choose the number of states by held-out likelihood. |
| [`run`](docs/run.md) | Fit at 200 bp, score every 10 bp, summarise, compare conditions, link genes. |
| [`predict`](docs/predict.md) | Predict expression from per-gene state tables and rank states. |
| [`compare`](docs/compare.md) | Benchmark ChromState against competing models on simulated data. |
| [`demo`](docs/demo.md) | Run the full simulated demo end to end. |
| [`export-chromhmm`](docs/export-chromhmm.md) | Write ChromHMM-format binarised input. |

## Contribute

Questions, bug reports and feature requests go to the
[issue tracker](https://github.com/helenhuangmath/ChromState/issues).

## Citation

If you use ChromState, please cite the software (see `CITATION.cff` in the repository):

> Huang H. ChromState: quantitative chromatin state analysis. Version 1.1.0 (2026).
> https://github.com/helenhuangmath/ChromState

## Other useful links

- [ChromHMM](https://compbio.mit.edu/ChromHMM/)
- [Segway](https://segway.hoffmanlab.org/)
- [EpiCSeg](https://github.com/lamortenera/epicseg)
- [Roadmap Epigenomics chromatin state annotations](https://egg2.wustl.edu/roadmap/web_portal/chr_state_learning.html)
- [ENCODE portal](https://www.encodeproject.org/)

```{toctree}
:maxdepth: 2
:hidden:

docs/INSTALL
docs/quickstart
docs/algorithm
docs/subcommands_index
docs/fileformats_index
docs/tutorial
docs/benchmarks
docs/qa
CHANGELOG
docs/api/index
```
