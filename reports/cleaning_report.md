# Cleaning Report — notebook `02_data_cleaning.ipynb`

Input: `data/raw/uber_raw.csv` (693,071 × 57). Output: `data/processed/uber_cleaned.csv` (**307,252 × 25**), sorted by time.

| Check | Before | After |
|---|---|---|
| Total Rows | 693,071 | 307,252 |
| Columns | 57 | 25 |
| Missing Values | 55,095 (all in `price`, all Uber Taxi) | 0 |
| Exact Duplicates | 0 | 0 |
| Duplicates ignoring `id` | 156 | 0 |
| Invalid Records (distance ≤ 0, surge ≤ 0, bad date, out-of-area coordinates) | 0 | 0 |

## Missing values — before any filtering
| Column | Missing | % of rows | Which rows |
|---|---|---|---|
| `price` | 55,095 | 7.95% | **every one is an Uber Taxi quote**. The API returned no price for taxis (Uber: 14.29% missing; Lyft: 0%) |
| all other 56 columns | 0 | 0% | – |

How they are handled:
- **Not imputed.** `price` is **leakage** (price = base fare × surge), so it is never a model feature. Filling it in
  would only invent values for a column we do not use.
- **The rows are dropped** in step 1 below, because they are Uber rows and Uber's surge target is not real. All
  Uber rows are removed, not only the 55,095 with a missing price.
- **The model can still handle missing feature values.** The Pipeline contains `SimpleImputer`s (median for numeric,
  most frequent for categorical) fitted on the training data only. Notebook 06, section 6, injects 6 missing values
  (distance, temperature, humidity, weather summary, pickup area, wind speed) into a **copy** of 5 training rows, and
  the saved Pipeline still returns predictions. A unit test (`tests/test_modules.py`) checks the same thing.

## Decisions and reasons

| # | Step | Rows | Reason |
|---|---|---|---|
| 1 | Keep **Lyft** rows only | 693,071 → 307,408 | Every Uber row has `surge_multiplier = 1.0`, so the target was never really recorded for Uber. Rows without a real target cannot be used for supervised learning. This also removes all 55,095 missing prices. |
| 2 | Drop missing target | 0 removed | No missing surge values in the Lyft rows. |
| 3 | Drop exact duplicates | 0 removed | None found. |
| 4 | Drop duplicates ignoring `id` | 156 removed | Same quote (same time, route, product, price, distance and weather) stored twice under different ids. Rows with the same time and route but a different distance or price are separate quotes and were **kept**. |
| 5 | Fix data types | n/a | `datetime` converted to datetime and renamed to `pickup_datetime` (0 invalid values). Numeric columns checked. Spaces stripped from `short_summary` (`' Mostly Cloudy '` → `'Mostly Cloudy'`). |
| 6 | Validate impossible values | 0 removed | Checked distance > 0, surge > 0, humidity and cloud cover in [0, 1], coordinates inside the Boston area, dates inside the collection period, and source ≠ destination. |
| 7 | Outliers | **0 removed** | Evidence below. |
| 8 | Drop redundant columns | 57 → 25 columns | Constant columns (`timezone`, `cab_type`), 1:1 copies (`product_id`, `visibility.1`, `icon`), epoch-time helper columns, daily weather aggregates, and per-area coordinates. |

## Outliers — IQR evidence (`reports/outlier_iqr_report.csv`, Lyft rows)

| Column | Flagged by 1.5×IQR | % | Why these are not errors |
|---|---|---|---|
| `surge_multiplier` (target) | 20,975 | 6.83% | Q1 = Q3 = 1.0, so **every surged ride** is flagged. These rides are the signal we predict. **Rows are never dropped by target value.** |
| `precipIntensity` | 66,473 | 21.64% | **Zero-inflated:** 78.4% of rows have no rain, so Q1 = Q3 = 0 and every rainy hour is flagged. Real rain readings (max 0.145 in/hr). |
| `visibility` | 60,378 | 19.65% | **Capped at 10 miles** by the weather API and concentrated near the cap (24.2% exactly 10). Fog and rain hours fall below the fence. Real readings (min 0.72 mi). |
| `temperature` | 16,010 | 5.21% | Coldest (18.9 °F) and mildest (57.2 °F) hours. Normal for Boston in Nov–Dec. |
| `price` | 4,092 | 1.33% | Luxury products and surged rides. Not a feature (leakage). |
| `distance` | 102 | 0.03% | Longest trips, up to 6.33 mi. A realistic city ride. |
| `humidity`, `windSpeed`, `cloudCover` | 0 | 0% | – |

**Extra check — winsorizing inside the Pipeline (notebook 06, section 2).** We tested clipping the continuous features
(distance and weather, not the calendar columns) at the 1/99 and 5/95 percentiles of each **training fold** (the
`Winsorizer` in `src/outliers.py`, placed between the imputer and the scaler). The change in CV RMSE was at most **0.00002** for both Lasso and the tuned Random Forest,
which is more than 300 times smaller than the fold std (about 0.0066). **Conclusion:** the flagged values do not hurt the models, so
removing or clipping them is not needed. The final model does not winsorize.

## Imbalance
93.2% of rides have surge = 1.0. This is regression, so SMOTE does not apply. Notebook 06, section 1, tests real
regression strategies (sample weights ×3 / ×10, oversampling surged rides to 15% / 30%) inside the training folds
only. The results and the decision are in `reports/extra_experiments.md`.

## Notes for the next steps
- **`price` is kept only for EDA.** It already includes the surge, so it is a **leakage** column and must never be a feature.
