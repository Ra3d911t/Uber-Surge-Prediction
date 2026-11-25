# Model Comparison — notebook `04_model_training.ipynb`

## Setup (the same for every model)
- **Features:** the 14 columns in `feature_schema.md`, passed through the same `make_preprocessor` inside each `Pipeline`.
- **Data:** a fixed 30,000-row sample of the training set (`random_state=42`, time order kept). SVR, KNN, MLP and
  Polynomial Regression are too slow for 5-fold CV on all 245k rows (fit-time column below). The test set was **not**
  used. Notebook 06 checks that the sample gives the same picture as the full training set.
- **CV:** `TimeSeriesSplit(n_splits=5)`, because the data is time ordered.
- **Metrics:** RMSE (main), MSE, R². Mean and standard deviation over the 5 folds.
- **Settings:** library defaults (tuning is in notebook 05). All models are defined once in `src/model_zoo.py`.
- **Baseline:** `DummyRegressor(mean)` always predicts the average surge. A model is only useful if it beats this.

## Which models and why
| Group | Models | Course link |
|---|---|---|
| **Lecture models** | Linear Regression (closed form), **Linear Regression trained by Gradient Descent** (`SGDRegressor`), Decision Tree, Random Forest, SVR, **ANN** (`MLPRegressor`) | L2 Linear Regression + Gradient Descent, L4 DT & RF, L6 SVM, L9 ANN, L10 optimisation |
| Extra classic models | Ridge, Lasso, Polynomial Regression (deg 2), KNN | not clearly in the lecture list; kept as extra comparisons |
| Newer versions (bonus B1) | XGBoost, LightGBM, HistGradientBoosting | gradient-boosted upgrades of the lecture tree models |

**Not applicable:** Logistic Regression and Naive Bayes are classification models, and K-means is clustering. None of
them predicts a continuous value, so they are not used for this regression target.

## Results (default hyper-parameters)

| Rank | Model | Group | Mean RMSE | Std RMSE | Mean MSE | Mean R² | vs baseline | Fit time / fold (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | Lasso (α = 0.001) | extra | **0.13212** | 0.00652 | 0.01750 | **0.0226** | **+1.2%** | 0.24 |
| 2 | Baseline (predict mean) | – | 0.13366 | 0.00652 | 0.01791 | -0.0003 | 0 | 0.28 |
| 3 | LightGBM | newer | 0.13401 | 0.00758 | 0.01802 | -0.0050 | -0.3% | 0.67 |
| 4 | HistGradientBoosting | newer | 0.13413 | 0.00796 | 0.01805 | -0.0066 | -0.4% | 1.44 |
| 5 | Ridge | extra | 0.13756 | 0.01635 | 0.01919 | -0.0594 | -2.9% | 0.23 |
| 6 | Linear Regression | lecture | 0.13778 | 0.01678 | 0.01926 | -0.0630 | -3.1% | 0.25 |
| 7 | Random Forest | lecture | 0.14355 | 0.00840 | 0.02068 | -0.1546 | -7.4% | 19.21 |
| 8 | SVR (RBF) | lecture | 0.14435 | 0.00274 | 0.02084 | -0.1718 | -8.0% | 50.00 |
| 9 | KNN (k = 5) | extra | 0.14681 | 0.00748 | 0.02161 | -0.2087 | -9.8% | 0.22 |
| 10 | XGBoost | newer | 0.15307 | 0.01613 | 0.02369 | -0.3100 | -14.5% | 2.30 |
| 11 | Linear Regression (Gradient Descent) | lecture | 0.15398 | 0.04816 | 0.02603 | -0.3909 | -15.2% | 0.28 |
| 12 | Decision Tree | lecture | 0.21464 | 0.00928 | 0.04616 | -1.6126 | -60.6% | 0.47 |
| 13 | ANN (MLP) | lecture | 0.25888 | 0.13586 | 0.08548 | -3.4559 | -93.7% | 7.53 |
| 14 | Polynomial Regression (deg 2) | extra | 0.66496 | 0.76521 | 1.02772 | -49.37 | -397% | 2.72 |

All 14 models trained without errors. Plot: `reports/figures/04_model_comparison.png`.

## Interpretation
- **Best default model: Lasso.** It is the only model that beats the baseline. Its L1 penalty shrinks most one-hot and
  weather coefficients to zero and keeps the few real signals (product = Shared, pickup area, distance).
- **Flexible models overfit with default settings.** A fully grown Decision Tree, unlimited-depth Random Forest,
  KNN with k = 5, XGBoost and an unregularised MLP all memorise noise. This is a sign that surge is mostly noise with
  respect to our features.
- **The gradient-descent models are unstable with defaults.** SGDRegressor (std 0.048) and the MLP (std 0.136) have
  very large fold-to-fold variation. They need a tuned learning rate, solver and regularisation (lecture 10).
- Polynomial Regression creates more than 1,200 interaction terms and is unstable on some folds.
- **Pros and cons:**
  - Linear family: fast and interpretable, but cannot capture interactions.
  - Trees and ensembles: capture area × product effects, but need limits on depth and leaf size.
  - KNN: simple, but distance in one-hot space is weak and prediction is slow.
  - SVR: robust, but O(n²) training cost.
  - ANN: most flexible, but sensitive to the optimiser and regularisation.
  - Boosting (newer): strong after tuning, but overfits with some defaults.

## Candidates for tuning (notebook 05)
Default settings are the main reason most models lose to the baseline, so **all 13 models are tuned** with
regularising grids, and the final choice is made on the tuned CV RMSE.
