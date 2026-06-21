# Revision Release Manifest

Tag: `revision-2026-nested-cv-v1`

Folder: `revision_2026_nested_cv/`

Purpose: reproducible workflow and machine-readable outputs supporting the BioEnergy Research revision of the BMP explainable-ML manuscript.

## Contents

- `scripts/`: scripts for the revised nested-CV workflow and the two-stage BMP per VS-to-BMP per DM sensitivity analysis.
- `tables/`: complete nested GridSearchCV candidate results, selected hyperparameters, and two-stage summaries.
- `validated_result_tables/`: manuscript-facing validated result tables generated during the revision audit.
- `predictions/`: row-level and fold-level nested-CV predictions and two-stage prediction outputs.
- `supplementary_data/`: final row-level supplementary dataset, fold-assignment file, data dictionary, and README prepared for manuscript submission.
- `figures/`: generated diagnostic figures for nested-CV observed-versus-predicted and residual checks.

## Source Data

The original source data in `../data/table_complete.csv` are not modified by this revision package. All files in this folder are derived analysis scripts or outputs.

## Primary Submission Files

- `supplementary_data/Supplementary_Dataset_Row_Level_OOF_Predictions.csv`
- `supplementary_data/Supplementary_Dataset_Fold_Level_OOF_Assignments.csv`
- `supplementary_data/Supplementary_Dataset_Data_Dictionary.csv`
- `supplementary_data/Supplementary_Dataset_README.md`

## Validation Boundary

Out-of-fold predictions in this package are internal cross-validation predictions. They are not external validation predictions. Full-data fitted predictions, where present, are labelled as apparent fitted values.
