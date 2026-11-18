# Output files

Tables are tab-separated; large ones are gzip-compressed. Figures are vector PDFs.

## Segmentation

`<sample>_segments.bed` is a dense BED9 file of the most likely state path, readable by
genome browsers, with the Roadmap colour of each state in the `itemRgb` column. It has
the same layout as ChromHMM's `*_dense.bed`.

## Model tables

| File | Rows × columns | Values |
|---|---|---|
| `emissions.tsv` | state × mark | Mean reads per bin |
| `transitions.tsv` | state × state | Transition probability per bin |
| `states.tsv` | state | Mnemonic, colour, naming rule, hard and soft coverage |
| `occupancy.tsv` | sample × state | Mean posterior, sums to 1 per sample |

## Per-bin quantification

`quantification.tsv.gz` (from `fit`) has one row per bin and sample: the hard state,
the posterior of every state, entropy, bivalency and mark attribution.
`scores/state_scores.tsv.gz` (from `run`) has `score_<state>` columns, integers from 0
to 100 that sum to 100 in every bin.

## Per-gene quantification

`genes/gene_quantification.tsv.gz` has one row per gene and sample. Column prefixes
give the part of the gene:

| Prefix | Part |
|---|---|
| `P_` | Promoter (TSS ± 1 kb by default) |
| `E_` | Distal space (regulatory window minus the promoter) |
| `B_` | Gene body |

and for each state:

| Column | Meaning |
|---|---|
| `occ_<state>` | Share of the part in the state (posterior mass), 0–1 |
| `int_<state>` | Intensity: signal in the bins of that state |
| `dose_<state>` | Share × intensity |
| `entropy` | How mixed the states are |
| `biv_score`, `biv_magnitude`, `biv_balance` | Bivalency from the H3K4me3 and H3K27me3 signal (when both marks are present) |
| `sig_<mark>` | Mean log(1 + normalised signal) of each mark |

## Differential tables

| File | Content |
|---|---|
| `region_state_tests.tsv` | One row per region, state and contrast: difference in share, moderated t, p, FDR |
| `regions_<A>_vs_<B>.tsv` | One row per region: total change in composition, whether it exceeds the noise floor |
| `state_summary.tsv` | Per state: number of regions gaining and losing it |
| `region_state_anova.tsv` | Moderated F-test across three or more conditions |
| `genome_state_tests.tsv` | Change in each state's genome share with a bootstrap interval |
