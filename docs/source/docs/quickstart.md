# Quick start

## 1. Describe the samples

ChromState reads ChromHMM's `cellmarkfiletable` unchanged: one line per sample and
mark, tab-separated, with an optional control in the fourth column.

```text
CD8_naive       H3K4me3    naive_K4me3.bam    naive_IgG.bam
CD8_naive       H3K27me3   naive_K27me3.bam   naive_IgG.bam
CD8_exhausted   H3K4me3    exh_K4me3.bam      exh_IgG.bam
CD8_exhausted   H3K27me3   exh_K27me3.bam     exh_IgG.bam
```

Give raw reads (BAM or BED) when you have them. See [Input files](input_formats.md).

## 2. Choose the number of states

```bash
chromstate selectk --manifest cellmarkfiletable.txt --chrom-sizes mm10.chrom.sizes \
                   --candidates 6,8,10,12,15 -o selectk_out
```

## 3. Fit and segment

```bash
chromstate fit --manifest cellmarkfiletable.txt --chrom-sizes mm10.chrom.sizes \
               -k 12 --blacklist mm10.blacklist.bed -o fit_out
```

This writes the emission and transition tables, one segment BED per sample in Roadmap
colours, and the per-bin quantification. See [Output files](output_files.md).

## 4. Quantify genes and compare conditions

```bash
chromstate run --manifest cellmarkfiletable.txt --chrom-sizes mm10.chrom.sizes \
               --gtf genes.gtf -k 12 \
               --groups "Naive=n1,n2;Exhausted=e1,e2" -o run_out
```

`run_out/genes/gene_quantification.tsv.gz` has one row per gene and sample with the
share of every state in the promoter, the gene body and the distal window.
`run_out/differential/` holds the tests between conditions and `run_out/figures/` the
alluvial plots.

## 5. Link to expression

```bash
chromstate predict --genes run_out/genes/gene_quantification.tsv.gz \
                   --expression rna_tpm.tsv -o predict_out
```

## The same in Python

```python
import chromstate as cs
from chromstate.genome import BinSet, Genome

binset = BinSet.from_genome(Genome.from_file("mm10.chrom.sizes"), bin_size=200)
ds = cs.build_dataset(cs.read_manifest("cellmarkfiletable.txt"), binset,
                      blacklist=cs.read_bed_regions("mm10.blacklist.bed"))
cs.normalize(ds, method="median_ratio")
seg = cs.fit(ds, n_states=12)
table = cs.quantify(seg)                                  # per-bin quantities

genes  = cs.FeatureSet.from_gtf("genes.gtf")
gq     = cs.gene_quantification(seg, genes)               # one row per gene and sample
report = cs.differential_report(seg, {"Naive": ["n1", "n2"], "Exhausted": ["e1", "e2"]})
res    = cs.predict_expression(gq, expression=log_expr)   # log_expr: genes x samples
```

Function names are listed in the [API reference](api/index.md).
