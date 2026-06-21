# Supplementary Dataset Notes

Source dataset: Lallement, A.; Peyrelasse, C.; Lagnet, C.; Barakat, A.; Schraauwers, B.; Maunas, S.; Monlau, F. A Detailed Database of the Chemical Properties and Methane Potential of Biomasses Covering a Large Range of Common Agricultural Biogas Plant Feedstocks. Waste 2023, 1(1), 195-227. DOI: 10.3390/waste1010014.

The row-level CSV contains one eligible feedstock row per record. Nested-CV columns are repeated out-of-fold predictions: a row-level prediction is averaged only from folds in which that row was excluded from model training.

Residual columns use observed minus prediction. Positive residuals indicate underprediction; negative residuals indicate overprediction.

Columns labelled apparent_full_data are fitted on all eligible rows and are provided only as apparent fitted values. They are not validation predictions.

Fold-level assignments and individual repeated out-of-fold predictions are provided in Supplementary_Dataset_Fold_Level_OOF_Assignments.csv.

Column definitions, units, scenario labels and residual sign conventions are provided in Supplementary_Dataset_Data_Dictionary.csv.
