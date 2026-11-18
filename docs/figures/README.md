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

Each panel, or row of panels that shares an axis, is its own PDF with 7 pt type; states
are listed from active transcription start site to quiescent. Fig6, Fig7 and Fig9 are
single figures.

- [Fig1_a_mark_levels](scireports2025/Fig1_a_mark_levels.pdf)
- [Fig1_b_state_share](scireports2025/Fig1_b_state_share.pdf)
- [Fig1_c_model_selection](scireports2025/Fig1_c_model_selection.pdf)
- [Fig1_d_expression_by_state](scireports2025/Fig1_d_expression_by_state.pdf)
- [Fig2_a-c_prediction](scireports2025/Fig2_a-c_prediction.pdf)
- [Fig2_d_single_marks](scireports2025/Fig2_d_single_marks.pdf)
- [Fig3_a_promoter_class](scireports2025/Fig3_a_promoter_class.pdf)
- [Fig3_b_later_expression](scireports2025/Fig3_b_later_expression.pdf)
- [Fig4_a_marks](scireports2025/Fig4_a_marks.pdf)
- [Fig4_b_promoter_states](scireports2025/Fig4_b_promoter_states.pdf)
- [Fig5_a-c_importance](scireports2025/Fig5_a-c_importance.pdf)
- [Fig6_marker_genes](scireports2025/Fig6_marker_genes.pdf)
- [Fig7_tracks_Pdcd1](scireports2025/Fig7_tracks_Pdcd1.pdf)
- [Fig7_tracks_Tcf7](scireports2025/Fig7_tracks_Tcf7.pdf)
- [Fig7_tracks_Tigit](scireports2025/Fig7_tracks_Tigit.pdf)
- [Fig7_tracks_Tox](scireports2025/Fig7_tracks_Tox.pdf)
- [Fig8_a_poised_vs_silent](scireports2025/Fig8_a_poised_vs_silent.pdf)
- [Fig8_b_chromatin_vs_rna](scireports2025/Fig8_b_chromatin_vs_rna.pdf)
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
half of the Polycomb bins under `drug`, and in half of those under the weaker `drugB`):
`volcano_*` (per region, with the noise floor), `state_volcano_*` (per state),
`region_state_heatmap_*` (change in share), `state_gain_loss`, `pvalue_calibration_*`
(observed against label-permuted), `replicate_concordance_*`, `change_ecdf`,
`anova_volcano`, and the benchmark `perturbation_pr_curves`.

Regions are 10-bin tiles and the simulated posteriors are close to 0 or 1, so shares fall
on multiples of 10% and the volcano plots show steps; real data does not. Figures carry
a title, axes and legend only; explanations belong in the legend
(`viz.SHOW_SUBTITLES = True` restores the subtitles).

## Paper figures (Nature format)

`paper/` holds the five paper figures drawn by `tools/make_paper_figures.py` from saved
run tables (no refitting), with their legends in [paper/LEGENDS.md](paper/LEGENDS.md).
Each panel, or row of panels that shares an axis, is its own file (`viz.save_panels`):
6-7 pt type with 8 pt bold panel letters, vector PDF with editable text and a 600 dpi
PNG copy; states are listed from active transcription start site to quiescent. The
earlier composite pages are kept unchanged in [paper/v1.0/](paper/v1.0/) and
[paper/v1.1/](paper/v1.1/).

- Figure 1, nine methods against simulated truth: [a-c_state_recovery](paper/Fig1_a-c_state_recovery.pdf), [d-e_perturbation_calibration](paper/Fig1_d-e_perturbation_calibration.pdf), [f_run_time](paper/Fig1_f_run_time.pdf)
- Figure 2, six tools on CD8 T-cell CUT&RUN and ENCODE ChIP-seq: [a-c_cd8_tcells](paper/Fig2_a-c_cd8_tcells.pdf), [d-f_encode](paper/Fig2_d-f_encode.pdf), [g-h_tool_agreement](paper/Fig2_g-h_tool_agreement.pdf), [i_roadmap_agreement](paper/Fig2_i_roadmap_agreement.pdf)
- Figure 3, replicated CD8 T-cell comparison with a null contrast: [a-d_states_and_changes](paper/Fig3_a-d_states_and_changes.pdf), [e_change_per_tile](paper/Fig3_e_change_per_tile.pdf), [f_confident_changes](paper/Fig3_f_confident_changes.pdf)
- Figure 4, expression prediction and state importance: [a_predicted_vs_measured](paper/Fig4_a_predicted_vs_measured.pdf), [b-c_representations](paper/Fig4_b-c_representations.pdf), [d_importance_cd8](paper/Fig4_d_importance_cd8.pdf), [e_importance_encode](paper/Fig4_e_importance_encode.pdf), [f_gene_part](paper/Fig4_f_gene_part.pdf)
- Figure 5, Roadmap, STARR-seq, CRISPRi and calibration checks: [a-c_states_and_reporter_activity](paper/Fig5_a-c_states_and_reporter_activity.pdf), [d_segmentation_agreement](paper/Fig5_d_segmentation_agreement.pdf), [e_crispri_enhancers](paper/Fig5_e_crispri_enhancers.pdf), [f_posterior_calibration](paper/Fig5_f_posterior_calibration.pdf)
