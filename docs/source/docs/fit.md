# fit

Fit a chromatin state model to the samples in a manifest, segment the genome, and write the states, the per-bin quantification and the figures.

## Example

```bash
chromstate fit --manifest cellmarkfiletable.txt --chrom-sizes mm10.chrom.sizes \
               -k 12 --norm median_ratio --blacklist mm10.blacklist.bed -o fit_out
```

With states defined in advance:

```bash
chromstate fit --manifest cellmarkfiletable.txt --chrom-sizes mm10.chrom.sizes \
               -k 12 --states default --anchor seed -o fit_out
```

## Output

| File | Content |
|---|---|
| `emissions.tsv`, `transitions.tsv` | Fitted emission means (reads per bin) and transition matrix |
| `states.tsv` | State mnemonic, colour, the rule that named it, hard and soft coverage |
| `occupancy.tsv` | Sample × state occupancy (mean posterior) |
| `<sample>_segments.bed` | ChromHMM-style dense BED of the Viterbi path, in Roadmap colours |
| `quantification.tsv.gz` | One row per bin and sample with every per-bin quantity |
| `summary.tsv` | Per-sample summary of the quantification table |
| `figures/*.pdf` | Emissions, transitions, coverage, browser track, confidence, bivalency, attribution, state flow, occupancy change |

## Notes

- `--emission auto` picks the negative-binomial model for reads and a Gaussian or
  Bernoulli model for coverage tracks and peaks.
- `--restarts` sets the minimum number of EM starts.
- Use [`selectk`](selectk.md) first if you do not know how many states to fit.

## Options

```{literalinclude} ../_generated/fit.txt
:language: text
```
