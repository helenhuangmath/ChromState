# Input files

## Manifest

A tab-separated table with one line per sample and mark, the same as ChromHMM's
`cellmarkfiletable`:

```text
sample <TAB> mark <TAB> file [<TAB> control [<TAB> format]]
```

The control (input or IgG) is optional. The format is taken from the fifth column or
from the file extension.

## Signal files

| Format | Extension | Read as | Default model |
|---|---|---|---|
| Aligned reads | `.bam` | One count per fragment midpoint (paired-end) or shifted 5′ end (single-end); MAPQ ≥ 30; duplicates removed | Negative binomial |
| Aligned reads as BED | `.bed`, `.tagAlign` | As for BAM | Negative binomial |
| Coverage track | `.bw`, `.bigWig` | Mean coverage per bin | Gaussian on log scale |
| Called peaks | `narrowPeak`, `broadPeak`, BED3/6 | Peak score per bin | Gaussian or Bernoulli |

**Give raw reads when you can.** The negative-binomial likelihood describes sampling
noise in counts. A normalised or smoothed track does not have that noise, so ChromState
refuses to fit the count model to coverage tracks and peaks and uses a Gaussian or
Bernoulli model for them.

## Other inputs

| Option | File |
|---|---|
| `--chrom-sizes` | Two columns: chromosome, length |
| `--blacklist` | BED of regions to exclude |
| `--spikein` | `sample <TAB> mark <TAB> spikein_reads`, for `--norm spikein` |
| `--gtf` / `--genes-bed` | Gene annotation as GTF/GFF (gene records) or BED6 |
| `--groups` | `"A=s1,s2;B=s3,s4"` or a `sample <TAB> condition` file |
| `--expression` | Genes × samples table with a `gene_id` column |
| `--lfc` | `gene_id <TAB> log2fc`, or a DESeq2 / edgeR result table |
| `--states` | State definitions: a TSV, or lines such as `TssA: H3K4me3+ H3K27ac+ H3K27me3-` |

## Experimental design

Tests between conditions need biological replicates. With one sample per condition
ChromState reports effect sizes only.
