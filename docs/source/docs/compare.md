# compare

Benchmark ChromState against reimplementations of competing models (ChromHMM-like, Segway-like, Gaussian mixture, k-means) on simulated data with known states.

## Example

```bash
chromstate compare --n-bins 200000 -o benchmark_out
```

## Output

A table with, for every model, the agreement with the true states (adjusted Rand
index), detection of bivalent bins, detection of perturbed bins and calibration error,
plus the figures.

To compare with the published software (ChromHMM, EpiCSeg, normR, diHMM, Segway) on
simulated or real data, use the harness in `tools/toolbench`; see
[Benchmarks](benchmarks.md).

## Options

```{literalinclude} ../_generated/compare.txt
:language: text
```
