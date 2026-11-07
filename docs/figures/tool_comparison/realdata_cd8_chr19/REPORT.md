# Tool comparison on real data

Dataset: tmp/realbench/cd8_chr19/counts.tsv — 921,474 bin-rows, 4 marks (H3K4me3, H3K27ac, H3K27me3, H3K9me3), samples Arm, Cl13, Naive, chromosomes chr19.

6 tool(s) produced a segmentation: ChromState, ChromHMM, EpiCSeg, normR, diHMM, Segway.

## Truth-free metrics

| method | n_states | bin_coverage | n_segments | mean_segment_kb | states_used | occupancy_entropy | mean_max_posterior | agreement_ari | runtime_s | peak_rss_mb |
|---|---|---|---|---|---|---|---|---|---|---|
| ChromState (negative-binomial hidden Markov model) | 15 | 1.000 | 54241.000 | 3.398 | 15.000 | 0.923 | 0.930 | — | 24.916 | — |
| ChromHMM (real, LearnModel) | 15 | 1.000 | 79070.000 | 2.331 | 15.000 | 0.510 | 0.513 | — | 64.856 | 1526.911 |
| EpiCSeg (real, negative multinomial hidden Markov model) | 15 | 1.000 | 48360.000 | 3.811 | 15.000 | 0.887 | 0.940 | 0.620 | 51.246 | 827.052 |
| normR (real, per-mark enrichment; combinatorial patterns) | 16 | 1.000 | 153388.000 | 1.201 | 16.000 | 0.486 | — | 0.221 | 22.009 | 1042.895 |
| diHMM (real, hierarchical hidden Markov model; nucleosome level) | 15 | 1.000 | 68057.000 | 2.708 | 13.000 | 0.269 | — | 0.054 | 3501.241 | 6548.722 |
| Segway (real, train + annotate) | 15 | 1.000 | 64001.000 | 2.880 | 15.000 | 0.940 | — | — | 3497.606 | 4666.556 |

## Promoter recovery (cross-validated AUPRC)

Per bin, the score is its state's promoter rate learned on the other half of the 1 Mb blocks. Chance level is the positive rate.

| tool | method | tss_auprc | tss_baseline | active_tss_auprc | active_tss_baseline | active_tss_auprc_Arm | active_tss_auprc_Cl13 | active_tss_auprc_Naive |
|---|---|---|---|---|---|---|---|---|
| chromstate | ChromState | 0.205 | 0.022 | 0.238 | 0.007 | 0.217 | 0.253 | 0.244 |
| chromhmm | ChromHMM | 0.171 | 0.022 | 0.189 | 0.007 | 0.174 | 0.195 | 0.197 |
| epicseg | EpiCSeg | 0.233 | 0.022 | 0.251 | 0.007 | 0.237 | 0.267 | 0.248 |
| normr | normR | 0.143 | 0.022 | 0.143 | 0.007 | 0.133 | 0.162 | 0.135 |
| dihmm | diHMM | 0.161 | 0.022 | 0.183 | 0.007 | 0.169 | 0.189 | 0.190 |
| segway | Segway | 0.149 | 0.022 | 0.168 | 0.007 | 0.158 | 0.187 | 0.159 |

## Expression from promoter states (held-out Spearman)

Gradient boosting on the promoter's (TSS +/- 1,000 bp) state composition, 5-fold by 1 Mb block. Change is Cl13 minus Naive. 'Raw mark signal' is the no-segmentation baseline: a segmentation that scores below it has thrown away information the marks carried.

| method | n_features | level_rho | level_lo | level_hi | change_rho | change_lo | change_hi |
|---|---|---|---|---|---|---|---|
| ChromState | 15 | 0.579 | 0.544 | 0.618 | 0.360 | 0.273 | 0.456 |
| ChromHMM | 15 | 0.564 | 0.522 | 0.602 | 0.231 | 0.146 | 0.310 |
| EpiCSeg | 15 | 0.559 | 0.516 | 0.594 | 0.295 | 0.208 | 0.381 |
| normR | 16 | 0.593 | 0.557 | 0.623 | 0.229 | 0.136 | 0.315 |
| diHMM | 15 | 0.578 | 0.538 | 0.618 | 0.228 | 0.140 | 0.308 |
| Segway | 15 | 0.614 | 0.578 | 0.651 | 0.121 | 0.024 | 0.219 |
| Raw mark signal (no states) | 4 | 0.602 | 0.566 | 0.639 | 0.341 | 0.248 | 0.430 |

## Agreement between tools (adjusted Rand index, bins both called)

| tool | ChromState | ChromHMM | EpiCSeg | normR | diHMM | Segway |
|---|---|---|---|---|---|---|
| ChromState | 1.000 | 0.125 | 0.620 | 0.221 | 0.054 | 0.478 |
| ChromHMM | 0.125 | 1.000 | 0.153 | 0.435 | 0.513 | 0.100 |
| EpiCSeg | 0.620 | 0.153 | 1.000 | 0.226 | 0.066 | 0.444 |
| normR | 0.221 | 0.435 | 0.226 | 1.000 | 0.253 | 0.198 |
| diHMM | 0.054 | 0.513 | 0.066 | 0.253 | 1.000 | 0.048 |
| Segway | 0.478 | 0.100 | 0.444 | 0.198 | 0.048 | 1.000 |

## Genome share by state class

| method | sample | active | bivalent | heterochromatin | polycomb | quiescent | weak |
|---|---|---|---|---|---|---|---|
| ChromHMM | Arm | 0.028 | 0.021 | 0.118 | 0.108 | 0.647 | 0.077 |
| ChromHMM | Cl13 | 0.030 | 0.014 | 0.104 | 0.094 | 0.685 | 0.073 |
| ChromHMM | Naive | 0.028 | 0.017 | 0.146 | 0.100 | 0.642 | 0.067 |
| ChromState | Arm | 0.035 | 0.006 | 0.149 | 0.184 | 0.571 | 0.056 |
| ChromState | Cl13 | 0.029 | 0.005 | 0.110 | 0.138 | 0.679 | 0.039 |
| ChromState | Naive | 0.040 | 0.007 | 0.191 | 0.156 | 0.539 | 0.067 |
| EpiCSeg | Arm | 0.055 | 0.007 | 0.144 | 0.165 | 0.515 | 0.115 |
| EpiCSeg | Cl13 | 0.054 | 0.006 | 0.132 | 0.162 | 0.555 | 0.091 |
| EpiCSeg | Naive | 0.060 | 0.006 | 0.153 | 0.167 | 0.534 | 0.080 |
| Segway | Arm | 0.031 | 0.043 | 0.143 | 0.099 | 0.453 | 0.231 |
| Segway | Cl13 | 0.024 | 0.022 | 0.104 | 0.072 | 0.553 | 0.225 |
| Segway | Naive | 0.033 | 0.037 | 0.187 | 0.075 | 0.426 | 0.242 |
| diHMM | Arm | 0.040 | 0.008 | 0.037 | 0.106 | 0.807 | 0.002 |
| diHMM | Cl13 | 0.037 | 0.005 | 0.032 | 0.093 | 0.830 | 0.002 |
| diHMM | Naive | 0.038 | 0.006 | 0.046 | 0.098 | 0.811 | 0.002 |
| normR | Arm | 0.055 | 0.042 | 0.142 | 0.290 | 0.472 | 0.000 |
| normR | Cl13 | 0.047 | 0.018 | 0.103 | 0.267 | 0.565 | 0.000 |
| normR | Naive | 0.062 | 0.030 | 0.191 | 0.255 | 0.462 | 0.000 |

## Figures

- `figures/state_profiles.pdf`
- `figures/state_class_fractions.pdf`
- `figures/segment_lengths.pdf`
- `figures/state_overlap.pdf`
- `figures/tool_agreement.pdf`
- `figures/promoter_recovery.pdf`
- `figures/expression_prediction.pdf`
- `figures/locus_Cd274_Pdcd1lg2_Jak2.pdf`
- `figures/locus_Fas.pdf`
- `figures/locus_Cd5_Cd6.pdf`

## How to read these

- State colours blend each state's own mark enrichment (red: active marks, blue: H3K27me3, violet: H3K9me3, magenta: both active and H3K27me3), so tools are drawn in one language without matching their states.
- No row is a score against truth. Each asks how well the tool's states, used naively, answer a question about genes the tool never saw.
- Agreement between tools is consistency, not correctness.
