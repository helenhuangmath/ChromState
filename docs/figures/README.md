# Example figures

These PDF examples are included so the tutorials can be browsed without downloading
datasets or running the analyses. Raw data, model files, and generated testing outputs
are excluded from the repository. The original generated figures remain in the ignored
`demo_out/figures/` and `analysis/scireports2025/out/figures/` directories.

## Simulation

Run the [simulation tutorial](../tutorials/01_simulation.md) to generate these examples.

- [01_emissions](simulation/01_emissions.pdf)
- [02_transitions](simulation/02_transitions.pdf)
- [03_coverage](simulation/03_coverage.pdf)
- [04_track](simulation/04_track.pdf)
- [05_confidence](simulation/05_confidence.pdf)
- [06_bivalency](simulation/06_bivalency.pdf)
- [07_attribution](simulation/07_attribution.pdf)
- [08_state_flow](simulation/08_state_flow.pdf)
- [09_occupancy_change](simulation/09_occupancy_change.pdf)
- [10_method_comparison](simulation/10_method_comparison.pdf)
- [11_binarization_loss](simulation/11_binarization_loss.pdf)
- [12_normalization](simulation/12_normalization.pdf)
- [13_calibration](simulation/13_calibration.pdf)

## CD8 T-cell application

See the [application tutorial](../tutorials/02_cd8_tcell_application.md) and
[results and figure legends](../../analysis/scireports2025/RESULTS.md).

- [Fig1_model](scireports2025/Fig1_model.pdf)
- [Fig2_prediction](scireports2025/Fig2_prediction.pdf)
- [Fig3_bivalency](scireports2025/Fig3_bivalency.pdf)
- [Fig4_cross_lag](scireports2025/Fig4_cross_lag.pdf)
- [Fig5_importance](scireports2025/Fig5_importance.pdf)
- [Fig6_marker_genes](scireports2025/Fig6_marker_genes.pdf)
- [Fig7_tracks_Pdcd1](scireports2025/Fig7_tracks_Pdcd1.pdf)
- [Fig7_tracks_Tcf7](scireports2025/Fig7_tracks_Tcf7.pdf)
- [Fig7_tracks_Tigit](scireports2025/Fig7_tracks_Tigit.pdf)
- [Fig7_tracks_Tox](scireports2025/Fig7_tracks_Tox.pdf)
- [Fig8_poised_genes](scireports2025/Fig8_poised_genes.pdf)
- [Fig9_chromatin_programs](scireports2025/Fig9_chromatin_programs.pdf)

## Expression prediction (Tutorial 3)

`expression_prediction/` holds the `chromstate predict` figures from the CD8 T-cell run on
chr17-19 (`k12_level_*`: unsupervised 12-state fit, expression level; `k12_change_*` and
`defs_change_*`: Cl13-vs-Naive change for the unsupervised and the defined-state fit).
`state_importance` = unique held-out R² per state; `representations` = quantitative vs
occupancy vs hard labels vs raw signal; `predicted_vs_observed` = held-out predictions;
`part_ranking` = promoter vs body. See [docs/tutorials/03_expression_prediction.md](../tutorials/03_expression_prediction.md).

## Statistical figures of the differential report

`statistics/` holds the output of `tools/make_stats_figures.py` (`sbatch/diag.sbatch`)
on a simulated three-condition experiment with three replicates each (ReprPC -> TssA in
half of the Polycomb bins of the two treated conditions): `volcano_*` (omnibus per
region, with the region-level noise floor), `state_volcano_*` (per state),
`region_state_heatmap_*`, `state_gain_loss`, `pvalue_calibration_*` (observed vs
label-permuted), `replicate_concordance_*`, `change_ecdf`, `anova_volcano`, and the
benchmark `perturbation_pr_curves` (nine tools on the simulated perturbation).
