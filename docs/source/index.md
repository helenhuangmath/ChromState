# ChromState

Quantitative chromatin state analysis for ChIP-seq, CUT&RUN, CUT&Tag, and ATAC-seq.

[Release 1.1](https://github.com/helenhuangmath/ChromState/releases/tag/v1.1) ·
[Installation](docs/INSTALL.md) · [Quick start](docs/quickstart.md) ·
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

## Start here

| Goal | Documentation |
|---|---|
| Install Python or set up the R interface | [Installation](docs/INSTALL.md) |
| Fit a model and inspect a result | [Quick start](docs/quickstart.md) |
| Run a worked analysis | [Tutorials](docs/tutorial.md) |
| Look up command flags or table columns | [Command reference](docs/subcommands_index.md) · [File formats](docs/fileformats_index.md) |
| Use ChromState from Python | [API reference](docs/api/index.md) |
| Compare methods and datasets | [Benchmarks](docs/benchmarks.md) |

### Try the demo

With ChromState installed, run the simulated workflow to generate a segmentation,
quantification tables, and plots.

```bash
chromstate demo -o demo_out
```

## What’s in 1.1

- Poisson, zero-inflated negative-binomial, and negative multinomial emission models.
- Gene-level state importance through [`chromstate fit`](docs/fit.md),
  the [Python API](docs/api/index.md), and the R interface.
- Expanded model benchmarks and validation on ENCODE ChIP-seq and CD8 T-cell CUT&RUN.

## Support

Report bugs or request features in the
[GitHub issue tracker](https://github.com/helenhuangmath/ChromState/issues).

## Citation

If you use ChromState, please cite the software (see `CITATION.cff` in the repository):

> Huang H. ChromState: quantitative chromatin state analysis. Version 1.1.0 (2026).
> https://github.com/helenhuangmath/ChromState

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
