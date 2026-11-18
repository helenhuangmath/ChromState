# Common questions

## How is ChromState different from ChromHMM?

ChromHMM turns each mark into present/absent per bin and reports one state per bin.
ChromState models the read counts and reports the probability of every state per bin,
so a promoter can be 70% active and 30% bivalent. It also writes ChromHMM-style segment
BED files, so existing browser sessions and downstream scripts keep working.

## Can I use the same input as ChromHMM?

Yes. The manifest is ChromHMM's `cellmarkfiletable`. To run ChromHMM itself on the same
bins, use [`chromstate export-chromhmm`](export-chromhmm.md).

## Which file type should I give?

BAM or BED reads. Coverage tracks and peaks work, but they are fitted with a Gaussian
or Bernoulli model because they are not counts. See [Input files](input_formats.md).

## How many states should I fit?

Run [`chromstate selectk`](selectk.md) and take the smallest number after which the
held-out likelihood stops improving. The benchmarks here used 15 states for four marks
and 18 for six.

## How do I get states per gene?

`chromstate run --gtf genes.gtf` writes `genes/gene_quantification.tsv.gz`, with the
share of every state in each gene's promoter, body and distal window. In Python use
`cs.gene_quantification(seg, features)`, or `cs.quantify_regions(seg, bed)` for any set
of regions.

## Do I need replicates?

For p-values, yes: at least two per condition. Without replicates ChromState reports
effect sizes and cannot separate a condition effect from sample-to-sample variation.

## Can it compare more than two conditions?

Yes. List them all in `--groups`. Pairwise region × state tests are written per
contrast, an ANOVA covers all conditions, and the alluvial plots show every condition
in the order given.

## It is slow on the whole genome. What can I do?

- Install Numba (`pip install "chromstate[fast]"`).
- Set `CHROMSTATE_THREADS` to the number of cores.
- Use `--mode gene`, which tiles only genes and their flanks.
- Use `--fit-fraction 0.25` to fit on a quarter of the genome and decode all of it.
  The result agrees with the full fit at an adjusted Rand index of about 0.74, so check
  that the states you care about are stable.
- In genome mode, `--stream` keeps one chromosome in memory at a time.

## The state names look wrong

Names are assigned from the emission pattern by rule. Look at `emissions.tsv` and
`states.tsv`, which records the rule used. To control the vocabulary, pass a definition
table with `--states`.

## Two runs gave different states

EM depends on its starting point. Fix `--seed`, raise `--restarts`, and compare runs
with `cs.match_states`.

## Is there an R interface?

Yes, `chromstatR`, which calls the Python package through `reticulate`. See
[Install](INSTALL.md).
