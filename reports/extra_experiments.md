# Extra Experiments — notebook `06_extra_experiments.ipynb`

Every result here comes from **cross-validation on training data only**: `TimeSeriesSplit(5)` on the same 30k
training sample as notebooks 04/05. The test set is not used. All numbers are copied from the notebook run.

## 1. Imbalance — sample weights and oversampling
93.2% of rides have surge = 1.0. SMOTE is a classification method, so for a continuous target we tested:
- **Sample weights:** surged rides count ×3 or ×10 in the loss.
- **Random oversampling:** surged rides are duplicated until they make up 15% or 30% of the training rows.

Each strategy changes **only the training part of each fold** (`src/imbalance.py`). The validation part keeps the
real ~6.8% surge rate.

| Model | Strategy | RMSE (± std) | RMSE surged only | R² | Mean prediction, non-surged (truth = 1.0) | Mean prediction, surged |
|---|---|---|---|---|---|---|
| Random Forest | **none** | **0.13168 ± 0.00659** | 0.48391 | **0.0291** | **1.0310** | 1.0487 |
| Random Forest | weight ×3 | 0.14482 ± 0.00611 | 0.42791 | -0.1762 | 1.0765 | 1.1177 |
| Random Forest | weight ×10 | 0.20792 ± 0.00573 | 0.34367 | -1.4290 | 1.1613 | 1.2376 |
| Random Forest | oversample to 15% | 0.13917 ± 0.00539 | 0.44160 | -0.0858 | 1.0637 | 1.1000 |
| Random Forest | oversample to 30% | 0.17212 ± 0.00562 | 0.38110 | -0.6659 | 1.1160 | 1.1801 |
| Lasso | none | 0.13212 ± 0.00652 | 0.48982 | 0.0226 | 1.0318 | 1.0422 |
| Lasso | weight ×3 | 0.14728 ± 0.01342 | 0.42750 | -0.2140 | 1.0824 | 1.1178 |
| Lasso | weight ×10 | 0.22055 ± 0.03901 | 0.34600 | -1.7562 | 1.1655 | 1.2427 |
| Lasso | oversample to 15% | 0.13944 ± 0.00814 | 0.44470 | -0.0885 | 1.0672 | 1.0958 |
| Lasso | oversample to 30% | 0.17251 ± 0.01261 | 0.38573 | -0.6709 | 1.1163 | 1.1729 |

**Trade-off:** every balancing strategy lowers the error on surged rides (by up to 29%, 0.484 → 0.344), but it raises
the overall RMSE, drops R² below zero, and biases the predictions upward. Non-surged rides, which are 93% of all rides,
are predicted at 1.06–1.16 instead of about 1.03. The models cannot tell surged rides apart from the features, so
balancing does not teach them *which* rides surge. It only shifts every prediction up.

**Decision:** the final model is **not balanced**, because the project metric is RMSE over all rides and real rides
keep the natural surge rate. **A ready alternative** is saved in `models/alt_balanced_model.joblib`. It is the same
tuned Random Forest trained on the full training set with **weight ×3**: the lowest surged-only RMSE among the strategies
whose overall RMSE stays within +10% of the unbalanced model. It has the same 14 inputs, so it can be swapped into
`app.py` if the instructor asks for a balanced model. It was not evaluated on the test set, so the test set stays reserved for the final model.

## 2. Outliers — winsorizing inside the Pipeline
`Winsorizer` (`src/outliers.py`) clips the **continuous** features (distance and weather) to percentiles **learned from
the training fold**. It sits between the imputer and the scaler. The calendar columns (`hour`, `day_of_week`,
`is_weekend`) are not measurements, so they are never clipped. No rows are removed, and the target is never clipped.

| Model | Winsorize | CV RMSE | ± std | R² | Change vs none |
|---|---|---|---|---|---|
| Lasso | none | 0.132117 | 0.006524 | 0.02261 | – |
| Lasso | 1% / 99% | 0.132102 | 0.006502 | 0.02282 | -0.000015 |
| Lasso | 5% / 95% | 0.132097 | 0.006493 | 0.02290 | -0.000020 |
| Random Forest | none | 0.131683 | 0.006589 | 0.02908 | – |
| Random Forest | 1% / 99% | 0.131683 | 0.006588 | 0.02908 | 0.000000 |
| Random Forest | 5% / 95% | 0.131665 | 0.006593 | 0.02934 | -0.000018 |

**Conclusion:** the largest change is 0.00002 RMSE, more than 300 times smaller than the fold std. Tree splits do not
depend on extreme values, and the linear model barely moves. Winsorizing **does not help**, so the final model keeps
the original values. This also confirms that the IQR-flagged values (zero-inflated rain, capped visibility) were not
harmful errors.

## 3. Why Random Forest — statistical tie analysis
`reports/model_tie_analysis.csv`. "Fold diff" is the per-fold RMSE minus Random Forest's RMSE on the same fold.

| Model | Group | CV RMSE | ± std | R² | Mean fold diff vs RF | Folds better than RF | Fit time / fold (s) |
|---|---|---|---|---|---|---|---|
| **Random Forest** | lecture | **0.13168** | 0.00659 | **0.0291** | 0 | – | 15.09 |
| HistGradientBoosting | newer | 0.13181 | 0.00655 | 0.0273 | +0.00012 | 2 / 5 | 2.04 |
| XGBoost | newer | 0.13189 | 0.00648 | 0.0259 | +0.00021 | 2 / 5 | 1.53 |
| Decision Tree | lecture | 0.13195 | 0.00665 | 0.0252 | +0.00026 | 1 / 5 | 0.47 |
| LightGBM | newer | 0.13200 | 0.00663 | 0.0244 | +0.00032 | 1 / 5 | 1.21 |
| Lasso | extra | 0.13212 | 0.00652 | 0.0226 | +0.00043 | 0 / 5 | 0.41 |
| Linear Regression (GD) | lecture | 0.13212 | 0.00653 | 0.0226 | +0.00044 | 0 / 5 | 1.05 |

The top five are within 0.0003 RMSE of each other, while the fold std is 0.0066, so they are **statistically tied**.
Random Forest has the lowest mean RMSE and the highest R², and no challenger beats it in most folds. It was kept by
the rule "replace only if a new model wins by more than one fold std". The full justification table is in
`final_evaluation.md`.

## 4. Bonus B1 — classic trees vs. newer versions (tuned, CV)

| Type | Model | Before tuning | After tuning | ± std | R² | Fit time / fold (s) |
|---|---|---|---|---|---|---|
| classic (lecture 4) | Random Forest | 0.14355 | 0.13168 | 0.00659 | 0.0291 | 15.09 |
| classic (lecture 4) | Decision Tree | 0.21464 | 0.13195 | 0.00665 | 0.0252 | 0.47 |
| newer (boosting) | HistGradientBoosting | 0.13413 | 0.13181 | 0.00655 | 0.0273 | 2.04 |
| newer (boosting) | XGBoost | 0.15307 | 0.13189 | 0.00648 | 0.0259 | 1.53 |
| newer (boosting) | LightGBM | 0.13401 | 0.13200 | 0.00663 | 0.0244 | 1.21 |

The newer gradient-boosting versions:
- are much better **with default settings** (LightGBM 0.1340 and HGB 0.1341, vs. 0.1436 for RF and 0.2146 for DT),
- train **7–12× faster** than Random Forest,
- after tuning, tie with Random Forest.

On this noisy target the upgrade buys speed and robustness to defaults, not accuracy.

## 4b. ANN — SGD vs. Adam and regularisation (lecture 10)
MLP (64-32), same CV and sample:

| Solver | alpha (L2) | CV RMSE | ± std | R² |
|---|---|---|---|---|
| sgd | 0.01 | 0.30272 | 0.31713 | -8.461 |
| sgd | 1 | 0.20688 | 0.15123 | -2.312 |
| sgd | **100** | **0.13366** | 0.00653 | -0.0004 |
| adam | 0.01 | 0.45958 | 0.57767 | -25.590 |
| adam | 1 | 0.26622 | 0.25918 | -5.825 |
| adam | 100 | 0.13413 | 0.00728 | -0.0070 |

- With weak regularisation, the network overfits the noise badly, and Adam overfits even more than SGD.
- With `alpha = 100` it becomes stable, but it only reaches the **mean-baseline level** (R² ≈ 0).
- **Honest conclusion:** the ANN was trained and tuned as required, but it does not find signal beyond the mean on this
  data. The tree models do slightly better because they isolate the "Shared" and pickup-area effects with hard splits.

## 5. Learning curve and feature importance
`figures/06_learning_curve_importance.png`. With `TimeSeriesSplit`, `learning_curve` can only grow to the smallest
training fold (5,000 rows):

| Training rows | Train RMSE | Validation RMSE |
|---|---|---|
| 500 | 0.13594 | 0.13396 |
| 1,250 | 0.14085 | 0.13386 |
| 2,500 | 0.14122 | 0.13301 |
| 3,750 | 0.13069 | 0.13289 |
| 5,000 | 0.12998 | 0.13180 |

Validation RMSE improves only slowly. Train and validation errors stay close together, so the model is **not**
overfitting: the limit is the information in the features, not the model. Train RMSE is sometimes higher than
validation RMSE because early time periods are noisier. Top importances: `distance` (0.29), `name = Shared` (0.19),
`source = Back Bay` (0.10).

## 5b. Why R² is low — surge persistence (notebook 06, `reports/surge_persistence.csv`)
| Condition | P(surge) if previous ride did not surge | P(surge) if previous ride surged |
|---|---|---|
| previous ride, same pickup area | 4.1% | 44.5% |
| previous ride, same pickup area and product | 6.6% | 9.7% |

Surge comes in episodes (part of this is the API-snapshot effect: same area and minute). The episodes are not explained
by the inputs available at request time: the surge rate is 5.9%–8.0% across hours and 6.6%–6.9% across weekdays
(overall 6.8%). The previous ride's surge would predict well, but it is the target itself in another row, so using it as
a feature would be leakage. This is the main reason R² stays low.

## 6. Missing values — imputers inside the saved Pipeline
6 missing values were injected into a **copy** of 5 training rows (distance, temperature + humidity, weather summary,
pickup area + wind speed). The saved `best_model.joblib` predicted all 5 rows without error, using the medians and modes
it learned on the training set (for example distance → 2.15 mi, temperature → 40.92 °F). The real data was not changed.

## 7. Tuning on a 30k sample — is it representative?
Fit time per fold on the 30k sample (notebook 04): SVR 50.0 s, Random Forest 19.2 s, ANN 7.5 s. A full grid on 245k
rows would multiply this by roughly 8× (and far more for SVR, which is O(n²)) for every grid point and fold.

| | RMSE | ± std | R² | Mean baseline RMSE |
|---|---|---|---|---|
| CV on 30k training sample | 0.13168 | 0.00659 | 0.0291 | 0.13366 |
| CV on **full** training set (245k) | 0.13263 | 0.00257 | 0.0367 | – |
| Test set (notebook 05, final evaluation) | 0.13524 | – | 0.0430 | 0.13825 |

The three estimates agree within normal variation, and R² rises slightly with more data. The sample did not
distort the result.
