# Final Evaluation — notebook `05_model_tuning.ipynb`

## Final model
**Random Forest Regressor** (lecture 4), saved as the full Pipeline (imputers, scaler and one-hot encoder, then the
model) in `models/best_model.joblib`.

| Best Parameters | Value |
|---|---|
| `n_estimators` | 100 |
| `max_depth` | 12 |
| `min_samples_leaf` | 200 |
| `random_state` | 42 |

## How it was chosen (no test data involved)
1. All **13** models from notebook 04 were tuned with `GridSearchCV`. The search used the same `TimeSeriesSplit(5)` on
   the same 30k training sample, scored with `neg_root_mean_squared_error`. The search spaces are in `src/model_zoo.py`.
2. The winner is the **lowest CV RMSE**. The models added in the second round (ANN, gradient-descent linear
   regression, LightGBM, HistGradientBoosting) could only replace the first-round winner if they beat it by **more than
   one fold standard deviation**. None did: Random Forest still has the lowest CV RMSE.
3. The winning pipeline was refit on **all 245,801 training rows**.
4. The test set (61,451 rides, the last 20% in time) was used only for the final evaluation below, never for selection
   or tuning. Notebook 05 was run again after the second round of models; because the selected model and its parameters
   did not change, the re-run reproduced exactly the same test numbers.

## Before vs. after tuning (CV RMSE, lower is better)

| Model | Group | Before | After | ± std | Improvement | After R² | Best Parameters |
|---|---|---|---|---|---|---|---|
| **Random Forest** | lecture | 0.14355 | **0.13168** | 0.00659 | 8.3% | **0.0291** | max_depth=12, min_samples_leaf=200, n_estimators=100 |
| HistGradientBoosting | newer | 0.13413 | 0.13181 | 0.00655 | 1.7% | 0.0273 | learning_rate=0.03, max_depth=3, max_iter=200, min_samples_leaf=200 |
| XGBoost | newer | 0.15307 | 0.13189 | 0.00648 | 13.8% | 0.0259 | learning_rate=0.03, max_depth=3, min_child_weight=50, n_estimators=200 |
| Decision Tree | lecture | 0.21464 | 0.13195 | 0.00665 | 38.5% | 0.0252 | max_depth=8, min_samples_leaf=500 |
| LightGBM | newer | 0.13401 | 0.13200 | 0.00663 | 1.5% | 0.0244 | learning_rate=0.03, min_child_samples=200, n_estimators=200, num_leaves=15 |
| Lasso | extra | 0.13212 | 0.13212 | 0.00652 | 0% | 0.0226 | alpha=0.001 (default already best) |
| Linear Regression (Gradient Descent) | lecture | 0.15398 | 0.13212 | 0.00653 | 14.2% | 0.0226 | alpha=0.001, penalty=l1, learning_rate=adaptive |
| Ridge | extra | 0.13756 | 0.13286 | 0.00772 | 3.4% | 0.0122 | alpha=1000 |
| KNN | extra | 0.14681 | 0.13314 | 0.00662 | 9.3% | 0.0074 | n_neighbors=100, weights=uniform |
| ANN (MLP) | lecture | 0.25888 | 0.13366 | 0.00653 | 48.4% | -0.0004 | hidden_layer_sizes=(64, 32), alpha=100, solver=sgd |
| SVR | lecture | 0.14435 | 0.13544 | 0.00700 | 6.2% | -0.0270 | C=0.1, epsilon=0.01, kernel=rbf |
| Linear Regression | lecture | 0.13778 | 0.13778 | 0.01678 | 0% | -0.0630 | fit_intercept=True (nothing to tune) |
| Polynomial Regression (deg 2) | extra | 0.66496 | 0.15995 | 0.05787 | 76.0% | -0.5336 | Ridge alpha=1000 |
| *Baseline (mean)* | – | 0.13366 | – | 0.00652 | – | 0.0000 | – |

Full table, including per-fold RMSE and fit times: `reports/tuning_comparison.csv`.

## Why Random Forest — honest justification
**The top five models are statistically tied.** Random Forest, HistGradientBoosting, XGBoost, Decision Tree and
LightGBM differ by at most **0.0003 RMSE**, while the fold-to-fold std is about **0.0066**. That gap is 20 times
smaller than the noise between folds (`reports/model_tie_analysis.csv`). No model in this group is clearly better.
Random Forest is chosen because:

| Criterion | Random Forest | Comment |
|---|---|---|
| CV RMSE (mean ± std) | **0.13168 ± 0.00659** | lowest mean of all 13 |
| CV R² | **0.0291** | highest of all 13 |
| Fold-by-fold vs. the challengers | HGB and XGB beat it in 2 of 5 folds, DT and LightGBM in 1 of 5 | no challenger wins most folds |
| Stability | std 0.0066, the same as every other top model | no instability (unlike SGD/MLP with defaults) |
| Training time | slowest of the top 5 (15 s per fold, about 2.5 min on the full data) | acceptable for a one-time fit |
| Interpretability | feature importances, averaged trees | easier to explain than boosting; studied in lecture 4 |
| Robustness | averaging 100 trees with leaves of 200 or more rides | less sensitive to noise than one tree |

If simplicity mattered more, the tuned **Decision Tree** (depth 8) would be an equally valid choice at almost the same
accuracy. All of this uses CV only. The test set played no part in the choice.

## Test-set results (final evaluation only)

| | MSE | RMSE | MAE | R² |
|---|---|---|---|---|
| **Random Forest — all test rides** | **0.01829** | **0.13524** | **0.05609** | **0.0430** |
| Baseline (mean) — all test rides | 0.01911 | 0.13825 | 0.05896 | -0.0001 |
| Random Forest — surged rides only (4,289) | 0.23977 | 0.48966 | 0.40875 | – |
| Baseline — surged rides only | 0.26100 | 0.51089 | 0.43107 | – |

- The model beats the baseline on every metric: −2.2% RMSE overall, −4.2% RMSE on surged rides.
- It predicts a higher surge for rides that really surged (mean 1.053) than for those that did not (1.030), and exactly
  **1.00 for Shared**, which never surges.

### CV vs. test consistency (from notebook 06)

| | RMSE | ± std | R² |
|---|---|---|---|
| CV on 30k training sample | 0.13168 | 0.00659 | 0.0291 |
| CV on full training set (245k) | 0.13263 | 0.00257 | 0.0367 |
| Test set (final evaluation only) | 0.13524 | – | 0.0430 |

The three estimates agree, and R² goes up slightly with more data. The 30k tuning sample did not distort the choice,
and the model does not overfit. The test RMSE is higher only because the test days have a higher surge rate
(6.98% vs 6.79%) and so more variance: the baseline RMSE also rises from 0.1337 (CV) to 0.1383 (test).

## Why R² is low — honest explanation
1. **The target is mostly noise with respect to the available inputs.** About 93% of rides are exactly 1.0. Surge
   is set from live driver supply and rider demand, and the dataset has neither. Hour, weekday and weather barely
   change the surge rate (about 7% in every hour and on every weekday; notebook 03).
2. **We checked this is not a bug: surge comes in episodes that the inputs cannot explain** (notebook 06, section 5b,
   `reports/surge_persistence.csv`). If the previous ride in the same pickup area surged, the chance that the next one
   surges is 44.5% (4.1% otherwise); for the same area *and* product it is 9.7% (6.6% otherwise). Part of this is a
   data-collection effect, because rides of the same area and minute come from one API snapshot. But the episodes are not
   tied to the inputs available at request time: the surge rate is 5.9%–8.0% across hours and 6.6%–6.9% across weekdays.
   The previous ride's surge would predict well, but it is the target itself in another row, so using it as a feature
   would be leakage and it is not used.
3. **We refused the easy fix.** `price` already contains the surge (price = base × surge). Using it would give a high
   R², but that is leakage.
4. **More data barely helps.** The learning curve (notebook 06, `figures/06_learning_curve_importance.png`) flattens
   just below the baseline, and full-train CV (245k rows) improves R² only from 0.029 to 0.037.
5. **What the model does learn:** `distance` (importance 0.29), `name = Shared` (0.19) and pickup area (Back Bay 0.10,
   North End, North Station, …). Weather and hour add very little.

## Balancing, outliers and other checks
See `reports/extra_experiments.md`. Short version:
- **Balancing** (sample weights or oversampling) lowers the error on surged rides but raises the overall RMSE and
  biases every prediction upward. The final model is therefore unbalanced, and a balanced alternative is saved in
  `models/alt_balanced_model.joblib`.
- **Winsorizing** changes CV RMSE by at most 0.00002, so it has no effect and is not used.

## Delivery check
```python
import joblib
model = joblib.load("models/best_model.joblib")
prediction = model.predict(new_data)   # new_data: DataFrame with the 14 columns of feature_schema.md
```
Verified at the end of notebook 05. The loaded file predicts raw rows with no outside preprocessing, its predictions
match the in-memory model, and `model.feature_names_in_` equals the training column order. Notebook 06 also shows it
predicts when some inputs are missing.
