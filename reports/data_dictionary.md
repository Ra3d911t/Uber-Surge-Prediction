# Data Dictionary — `data/raw/uber_raw.csv`

Source file: Kaggle *Uber and Lyft Dataset Boston, MA* (`rideshare_kaggle.csv`), stored unchanged.
**693,071 rows × 57 columns**. Each row is one price quote that the ride-hailing API returned for one product
(the author made repeated requests for each route).

The project uses only the **307,408 Lyft rows** (see `dataset_sources.md` for why). The filtering happens in notebook 02.

**Role legend**
- **Target**: the value we predict.
- **Feature (candidate)**: may be used as model input. The final list is in `feature_schema.md`.
- **Identifier**: identifies a row and is not used for learning.
- **Leakage**: only known after surge pricing is applied, so it must never be a feature.
- **Redundant**: duplicates other columns or is constant for the project's rows.

## Trip & time columns

| Column | Meaning | Type | Example | Role |
|---|---|---|---|---|
| `id` | Unique id of the quote | string | `424553bb-7174-…` | Identifier |
| `timestamp` | Unix epoch time of the request (seconds) | float | `1544952607.89` | Redundant (same as `datetime`) |
| `hour` | Hour of the request (0–23) | int | `9` | Redundant (re-derived from `datetime`) |
| `day` | Day of the month | int | `16` | Redundant |
| `month` | Month (only 11 and 12 occur) | int | `12` | Redundant |
| `datetime` | Request date & time (local time) | string → datetime | `2018-12-16 09:30:07` | Feature source (hour, weekday) |
| `timezone` | Time zone, always `America/New_York` | string | `America/New_York` | Redundant (constant) |
| `source` | Pickup area in Boston (12 values) | category | `Haymarket Square` | Feature |
| `destination` | Drop-off area in Boston (12 values) | category | `North Station` | Feature |
| `cab_type` | Company: `Uber` / `Lyft` | category | `Lyft` | Filter (Lyft only), then constant |
| `product_id` | Product code | category | `lyft_line` | Redundant (1:1 with `name`) |
| `name` | Product name (`Shared`, `Lyft`, `Lyft XL`, `Lux`, `Lux Black`, `Lux Black XL`) | category | `Shared` | Feature |
| `price` | Quoted price in USD, **already includes surge** | float | `5.0` | **Leakage** |
| `distance` | Trip distance (miles) | float | `0.44` | Feature |
| `surge_multiplier` | Surge multiplier applied to the base fare | float | `1.0`, `1.25`, `2.0` | **Target** |
| `latitude` | Latitude of the weather point for the pickup area | float | `42.2148` | Redundant (one value per `source`) |
| `longitude` | Longitude of the weather point | float | `-71.033` | Redundant |

## Weather columns
The dataset author joined weather to each row by area and hour. All values are imperial units
(°F, inches/hour, mph, miles).

| Column | Meaning | Type | Example | Role |
|---|---|---|---|---|
| `temperature` | Temperature (°F) | float | `42.34` | Feature |
| `apparentTemperature` | Feels-like temperature (°F) | float | `37.12` | Redundant (highly correlated with `temperature`) |
| `short_summary` | Short weather text | category | `Mostly Cloudy` | Feature |
| `long_summary` | Daily weather sentence | string | `Rain throughout the day.` | Redundant (daily, free text) |
| `precipIntensity` | Rain intensity (in/hr) | float | `0.0` | Feature |
| `precipProbability` | Probability of rain (0–1) | float | `0.0` | Candidate (correlated with intensity) |
| `humidity` | Relative humidity (0–1) | float | `0.68` | Feature |
| `windSpeed` | Wind speed (mph) | float | `8.66` | Feature |
| `windGust` | Wind gust (mph) | float | `9.17` | Redundant (correlated with `windSpeed`) |
| `windGustTime` | Epoch time of the max gust | int | `1545015600` | Redundant |
| `visibility` | Visibility (miles) | float | `10.0` | Feature |
| `visibility.1` | Exact duplicate of `visibility` | float | `10.0` | Redundant |
| `temperatureHigh` / `temperatureLow` | Daily high / low (°F) | float | `43.68` | Redundant (daily) |
| `temperatureHighTime` / `temperatureLowTime` | Epoch time of the daily high / low | int | `1544968800` | Redundant |
| `apparentTemperatureHigh` / `…Low` | Daily feels-like high / low | float | `37.95` | Redundant |
| `apparentTemperatureHighTime` / `…LowTime` | Epoch times of the above | int | | Redundant |
| `icon` | Weather icon name | category | `partly-cloudy-night` | Redundant (same info as `short_summary`) |
| `dewPoint` | Dew point (°F) | float | `32.7` | Candidate |
| `pressure` | Air pressure (hPa) | float | `1021.98` | Candidate |
| `windBearing` | Wind direction (degrees) | int | `57` | Candidate |
| `cloudCover` | Cloud cover (0–1) | float | `0.72` | Feature |
| `uvIndex` | UV index | int | `0` | Candidate |
| `ozone` | Ozone (Dobson units) | float | `303.8` | Redundant |
| `sunriseTime` / `sunsetTime` | Epoch times | int | | Redundant |
| `moonPhase` | Moon phase (0–1) | float | `0.3` | Redundant |
| `precipIntensityMax` | Daily max rain intensity | float | `0.1276` | Redundant (daily) |
| `uvIndexTime` | Epoch time of the max UV | int | | Redundant |
| `temperatureMin` / `temperatureMax` | Daily min / max (°F) | float | `39.89` | Redundant (daily) |
| `temperatureMinTime` / `temperatureMaxTime` | Epoch times | int | | Redundant |
| `apparentTemperatureMin` / `…Max` | Daily feels-like min / max | float | | Redundant |
| `apparentTemperatureMinTime` / `…MaxTime` | Epoch times | int | | Redundant |

## Columns the project wanted but the dataset does not have
`Traffic`, `Available Drivers` and `Ride Requests` are not in the data. They were **not** invented or joined
from another source.
