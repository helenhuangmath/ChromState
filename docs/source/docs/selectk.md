# selectk

Choose the number of states. Each candidate is fitted and scored by the likelihood of held-out data.

## Example

```bash
chromstate selectk --manifest cellmarkfiletable.txt --chrom-sizes mm10.chrom.sizes \
                   --candidates 6,8,10,12,15 -o selectk_out
```

## Output

`model_selection.tsv` has one row per candidate with its held-out log-likelihood per bin
(`heldout_loglik_per_bin`), and `model_selection.pdf` shows the curve. The command prints
the candidate with the highest held-out likelihood.
Pick the smallest number of states after which the curve flattens; more states always
fit a little better, and states beyond the plateau are usually splits of the quiescent
background.

## Options

```{literalinclude} ../_generated/selectk.txt
:language: text
```
