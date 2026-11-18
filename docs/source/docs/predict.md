# predict

Predict gene expression from the per-gene state table written by `chromstate run`, and rank the states by how much of the prediction depends on them.

## Examples

Expression level:

```bash
chromstate predict --genes run_out/genes/gene_quantification.tsv.gz \
                   --expression rna_tpm.tsv -o predict_out
```

Expression change between two conditions:

```bash
chromstate predict --genes run_out/genes/gene_quantification.tsv.gz \
                   --lfc deseq2.tsv --contrast TN,TEX \
                   --groups "TN=tn1,tn2;TEX=tex1,tex2" -o predict_out
```

## Output

| File | Content |
|---|---|
| `gene_predictions.tsv` | Held-out prediction for every gene; folds are grouped by chromosome |
| `state_importance.tsv` | Per state: accuracy of its own model, and the held-out R² lost when it is permuted |
| `representation_scores.tsv` | Quantitative states against state share, hard label and raw signal |
| `representation_contrasts.tsv` | Paired differences between those representations |
| `part_ranking.tsv` | Promoter, distal and body columns alone and combined |
| `summary.tsv`, `REPORT.md`, `figures/` | Summary, written report and figures |

## Notes

- `--expression` expects genes × samples with a `gene_id` column. Values are
  log-transformed unless `--expression-is-log` is given.
- `--lfc` accepts two columns (`gene_id`, `log2fc`) or a DESeq2 or edgeR table.
- `--model` is `hgb` (gradient-boosted trees, default), `xgboost` or `ridge`.

## Options

```{literalinclude} ../_generated/predict.txt
:language: text
```
