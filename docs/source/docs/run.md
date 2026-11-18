# run

The full pipeline in one command: fit at 200 bp, score every 10 bp, summarise the states, quantify genes, compare conditions and link to expression.

## Examples

Gene-level mode (default) tiles each gene ± 50 kb:

```bash
chromstate run --manifest cellmarkfiletable.txt --chrom-sizes mm10.chrom.sizes \
               --gtf genes.gtf --mode gene -k 10 \
               --groups "TN=tn1,tn2;TEX=tex1,tex2" --expression rna_tpm.tsv -o run_out
```

Genome-wide mode, keeping one chromosome in memory at a time:

```bash
chromstate run --manifest cellmarkfiletable.txt --chrom-sizes mm10.chrom.sizes \
               --gtf genes.gtf --mode genome --stream -k 10 -o run_out
```

With an RNA contrast instead of an expression matrix:

```bash
chromstate run ... --groups "TN=tn1,tn2;TEX=tex1,tex2" --lfc deseq2.tsv --contrast TN,TEX
```

## Main settings

| Setting | Default | Meaning |
|---|---|---|
| `--resolution` | 10 bp | Width of the scored bins; `--fit-bin-size` must be a multiple |
| `--fit-bin-size` | 200 bp | Width of the bins the model is fitted at |
| `--promoter-upstream`, `--promoter-downstream` | 1,000 bp each | The promoter is the TSS plus these flanks |
| `--flank` | 50,000 bp | The regulatory window is the gene ± this; the window minus the promoter is the distal space |
| `--mode` | `gene` | `gene` tiles the regulatory windows; `genome` tiles every chromosome |
| `-k`, `--fit-states` | 10, none | Fit `--fit-states` states and merge to `-k` by emission similarity |
| `--groups` | none | Conditions and their replicates; turns on the comparison |
| `--fit-fraction` | all | Fit on a block subsample of the genome, then decode all of it |

## Output

| Folder | Content |
|---|---|
| `model/` | `emissions.tsv`, `transitions.tsv`, `states.tsv`, and `collapsed_members.tsv` when states were merged |
| `scores/` | `state_scores.tsv.gz` (0–100 per state and bin), `<sample>_segments.bed`, bedGraph tracks with `--tracks` |
| `summary/` | Base pairs and percentage per state per sample and per condition, condition differences, base-pair flow between conditions, state tests |
| `genes/` | `gene_quantification.tsv.gz`, dominant state per promoter and distal window, pattern and category counts, candidate elements |
| `differential/` | Region × state tests, per-state summary, ANOVA for three or more conditions (needs `--groups` with replicates) |
| `expression/` | State importance, representation scores, promoter / distal / body ranking, enhancer importance (needs `--expression` or `--lfc`) |
| `figures/` | Emissions, coverage, alluvial plots by region and by gene, volcano and gain/loss plots |
| `REPORT.md` | What was run and where each result is |

See [Output files](output_files.md) for the columns of the gene table.

## Options

```{literalinclude} ../_generated/run.txt
:language: text
```
