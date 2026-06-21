# Revision 2026 Nested-CV Workflow Package

This folder contains the revised analysis workflow and generated outputs prepared for the BioEnergy Research revision.

Stable release tag: `revision-2026-nested-cv-v1`

Stable folder URL:
`https://github.com/Fashdan/BMP-Lignin-Moisture-Interaction/tree/revision-2026-nested-cv-v1/revision_2026_nested_cv`

It supplements the original workflow with:

- repeated nested five-fold Random Forest evaluation;
- target and predictor-sensitivity scenarios;
- regenerated GridSearchCV candidate results and selected hyperparameters;
- two-stage BMP/VS-to-BMP/DM sensitivity analysis;
- cross-model SHAP ranking summaries;
- empirical out-of-fold error intervals;
- leave-one-family-out domain-shift stress test;
- row-level out-of-fold predictions and residuals.

The source dataset in `../data/table_complete.csv` was not modified. The row-level supplementary dataset in `predictions/final_row_level_supplementary_dataset_design.csv` is derived from that source table and labels out-of-fold predictions separately from apparent full-data fitted predictions.

## Main Scripts

- `scripts/run_stage2_analysis.py`: nested CV, sensitivity scenarios, SHAP summaries, residual diagnostics, and row-level outputs.
- `scripts/run_two_stage_sensitivity.py`: nested BMP/VS out-of-fold predictions followed by deterministic conversion to BMP/DM using measured VS/DM.

## Main Outputs

- `validated_result_tables/`: manuscript-ready validated result tables.
- `tables/nested_gridsearch_candidate_results_by_outer.csv`: regenerated GridSearchCV candidate results, 216 candidates per outer search.
- `tables/nested_cv_selected_hyperparameters.csv`: selected RF hyperparameters by repeated outer fold.
- `predictions/final_row_level_supplementary_dataset_design.csv`: row-level supplement design with source identifiers, observed targets, fold assignments, out-of-fold predictions, residuals, and apparent full-data fitted predictions.
- `supplementary_data/Supplementary_Dataset_Row_Level_OOF_Predictions.csv`: final row-level supplementary dataset prepared for manuscript submission.
- `supplementary_data/Supplementary_Dataset_Fold_Level_OOF_Assignments.csv`: fold-level repeated out-of-fold predictions and assignment details.
- `supplementary_data/Supplementary_Dataset_Data_Dictionary.csv`: column definitions, units, scenario labels and residual sign conventions.
- `supplementary_data/Supplementary_Dataset_README.md`: concise data-package interpretation note.
- `figures/`: nested out-of-fold observed-vs-predicted residual figures for the sensitivity scenarios.

## Status

Prepared for the revised manuscript package on 2026-06-20 and tagged as `revision-2026-nested-cv-v1`.
