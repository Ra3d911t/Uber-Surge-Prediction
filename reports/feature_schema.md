# Feature Schema

The model (`models/best_model.joblib`) takes a pandas DataFrame with **exactly these 14 columns, in this order**.
The same list is in `src/features.py` as `FEATURE_COLUMNS`.

Target: `surge_multiplier` (float, observed values 1.0–3.0).

| # | Column | Meaning | Type | Expected values (training data) | Source |
|---|---|---|---|---|---|
| 1 | `source` | Pickup area | Categorical | Back Bay, Beacon Hill, Boston University, Fenway, Financial District, Haymarket Square, North End, North Station, Northeastern University, South Station, Theatre District, West End | raw |
| 2 | `destination` | Drop-off area | Categorical | same 12 areas (must differ from `source`) | raw |
| 3 | `name` | Lyft product | Categorical | Shared, Lyft, Lyft XL, Lux, Lux Black, Lux Black XL | raw |
| 4 | `short_summary` | Weather description | Categorical | Clear, Partly Cloudy, Mostly Cloudy, Overcast, Foggy, Possible Drizzle, Drizzle, Light Rain, Rain | raw (spaces stripped) |
| 5 | `distance` | Trip distance (miles) | Numeric | 0.39 – 6.33 (median 2.15) | raw |
| 6 | `hour` | Pickup hour | Numeric (int) | 0 – 23 | **derived** from `pickup_datetime` |
| 7 | `day_of_week` | Pickup weekday, Monday = 0 | Numeric (int) | 0 – 6 | **derived** |
| 8 | `is_weekend` | 1 if Saturday or Sunday | Numeric (0/1) | 0, 1 | **derived** |
| 9 | `temperature` | Temperature (°F) | Numeric | 18.9 – 57.2 | raw |
| 10 | `precipIntensity` | Rain intensity (in/hr) | Numeric | 0 – 0.145 | raw |
| 11 | `humidity` | Relative humidity (0–1) | Numeric | 0.44 – 0.96 | raw |
| 12 | `windSpeed` | Wind speed (mph) | Numeric | 0.45 – 12.6 | raw |
| 13 | `visibility` | Visibility (miles) | Numeric | 0.72 – 10 | raw |
| 14 | `cloudCover` | Cloud cover (0–1) | Numeric | 0 – 1 | raw |

## Derived features
`hour`, `day_of_week` and `is_weekend` are made by `src.features.build_features()` from a `pickup_datetime` column.
**The GUI must call the same function.** It must not re-implement the logic.

```python
from src.features import build_features, select_features
row = pd.DataFrame([{"pickup_datetime": "2018-12-17 18:30", "source": "Back Bay", ...}])
X = select_features(build_features(row))
model.predict(X)
```

## Preprocessing inside the saved Pipeline
- Numeric columns: median `SimpleImputer`, then `StandardScaler`.
- Categorical columns: most-frequent `SimpleImputer`, then `OneHotEncoder(handle_unknown="ignore")`.
- The pipeline is fitted on the training data only and is stored inside `best_model.joblib`.
  The GUI sends **raw** values and does no scaling or encoding itself.

## Excluded columns and why

| Column | Reason |
|---|---|
| `price` | **Leakage.** The quoted price already includes the surge. |
| `id` | Identifier |
| `pickup_datetime` | Replaced by `hour`, `day_of_week`, `is_weekend` |
| `day`, `month` | Only 3 weeks of data (Nov–Dec 2018), so it would not generalise to other dates |
| `apparentTemperature`, `dewPoint` | Highly correlated with `temperature` |
| `precipProbability` | Correlated with `precipIntensity` |
| `windGust` | Correlated with `windSpeed` |
| `pressure`, `windBearing`, `uvIndex` | No relationship with surge in EDA; dropped to keep the GUI short |
| All columns dropped in cleaning | See `cleaning_report.md` |

## What the GUI user enters
Pickup area, drop-off area, Lyft product, trip distance, pickup date and time, and the weather
(summary, temperature, rain intensity, humidity, wind speed, visibility, cloud cover).
