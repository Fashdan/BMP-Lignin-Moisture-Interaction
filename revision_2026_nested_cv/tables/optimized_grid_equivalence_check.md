# Optimized Grid Evaluator Equivalence Check

Purpose: validate the prefix-forest grid evaluator against the native sklearn `GridSearchCV` results already checkpointed for `A_BMP_VS_all_features`, outer fold 1.

Configuration compared:

- Same recovered RF grid: 216 candidates.
- Same inner CV: `KFold(n_splits=5, shuffle=True, random_state=42)`.
- Same scoring: `r2`.
- Same refit criterion: highest mean inner-CV R2, first candidate in grid order for ties.

Result:

| Check | Value |
|---|---:|
| Native GridSearchCV candidate rows | 216 |
| Optimized evaluator matched rows | 216 |
| Native best params | `max_depth=10`, `max_features=sqrt`, `min_samples_leaf=1`, `min_samples_split=2`, `n_estimators=50` |
| Optimized best params | `max_depth=10`, `max_features=sqrt`, `min_samples_leaf=1`, `min_samples_split=2`, `n_estimators=50` |
| Native best mean inner R2 | 0.3627294654497495 |
| Optimized best mean inner R2 | 0.36272946544974954 |
| Maximum absolute candidate-score delta | 5.551115123125783e-17 |
| Mean absolute candidate-score delta | 2.3900634557902677e-17 |

Conclusion: the optimized evaluator reproduces the native GridSearchCV candidate scores and selected parameters for the checked fold to numerical precision. The optimization works because a 150-tree Random Forest with fixed `random_state` contains the same first 50 and first 100 trees that the corresponding 50- and 100-tree candidates would fit for the same non-`n_estimators` hyperparameters.
