# export-chromhmm

Write the binarised input files that the ChromHMM software reads, from the same manifest, so both tools can be run on identical bins.

## Example

```bash
chromstate export-chromhmm --manifest cellmarkfiletable.txt \
                           --chrom-sizes mm10.chrom.sizes -o binarized
java -jar ChromHMM.jar LearnModel binarized chromhmm_out 12 mm10
```

## Options

```{literalinclude} ../_generated/export-chromhmm.txt
:language: text
```
