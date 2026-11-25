# Uber/Lyft Surge Multiplier Prediction

Machine Learning course project (regression), Helwan National University, Fall 2026/2027.
The project predicts the **surge multiplier** of a ride-hailing trip in Boston from the trip, time and weather.

- **Data:** Kaggle *Uber and Lyft Dataset Boston, MA*. The project uses the **Lyft rows only** (307,252 trips after
  cleaning), because Uber's surge is always 1.0 in this dataset. See `reports/dataset_sources.md`.
- **Final model:** tuned Random Forest inside a full preprocessing Pipeline (`models/best_model.joblib`).
- **Test results:** RMSE 0.1352 vs 0.1383 for the mean baseline. R² 0.043. Why R² is low is explained in
  `reports/final_evaluation.md`.

## Project structure
```
data/raw/uber_raw.csv                  original Kaggle file (Google Drive, git-ignored)
data/processed/                        uber_cleaned.csv, uber_features.csv (Drive), X/y train/test (git)
notebooks/01_dataset_research.ipynb    person 2: dataset research
notebooks/02_data_cleaning.ipynb       person 3: cleaning
notebooks/03_eda_preprocessing.ipynb   person 4: EDA, features, split
notebooks/04_model_training.ipynb      person 5: all models + CV comparison
notebooks/05_model_tuning.ipynb        person 6: GridSearchCV + final evaluation
notebooks/06_extra_experiments.ipynb   imbalance, winsorizing, model tie analysis, ANN solvers, learning curve (CV only)
src/features.py                        feature engineering (shared with the GUI)
src/preprocessing.py                   make_preprocessor(): unfitted ColumnTransformer (optional winsorizing)
src/modeling.py                        shared CV settings
src/model_zoo.py                       all 14 models + GridSearchCV search spaces
src/imbalance.py                       sample weights / oversampling applied to training folds only
src/outliers.py                        IQR report + Winsorizer (fit on training fold)
models/best_model.joblib               final Pipeline (preprocessing + model)
models/alt_balanced_model.joblib       ready balanced alternative (same inputs)
app.py                                 person 7: Uber-style Streamlit GUI (map, moving cars, ride options)
src/map_data.py, src/map_view.py       GUI: area coordinates, distance/fare tables, Leaflet map
src/predict.py                         GUI: surge prediction for every ride type
scripts/build_gui_tables.py            builds models/gui_tables.json (distance per area pair, fare per product)
reports/                               data dictionary, cleaning, comparison, tuning, final evaluation, feature schema,
                                       extra_experiments.md
tests/test_app.py                      model + GUI sanity tests
tests/test_modules.py                  pytest unit tests for src modules
```

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```
Download `rideshare_kaggle.csv` from Kaggle (or the team Drive) and save it as `data/raw/uber_raw.csv`.

## Run
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_dataset_research.ipynb   # ...then 02 to 05 in order
streamlit run app.py
python -m tests.test_app
python -m pytest -q tests/test_modules.py
```
Notebooks must be run from the project folder. On Colab, clone the repo first: each notebook finds the project root
by itself. The train/test CSVs and the model are in the repo, so notebooks 04, 05 and the app run without the large
raw data.

## Key rules
No leakage: `price` is never a feature. All preprocessing is fitted inside the Pipeline on training data only.
The split is time-ordered. The test set is used only for the final evaluation at the end of notebook 05, never for selection or tuning. Details are in `CLAUDE.md`.

## GUI (Uber-style)
`streamlit run app.py` shows a dark map of Boston with the 12 pickup/drop-off areas, the route, a surge halo
(green / amber / red) and moving cars. Pick a pickup, a drop-off, a time and the weather; the app predicts the surge for
every Lyft ride type with the saved model and shows an **estimated** fare (typical fare for that distance x predicted
surge; `price` is still never a model feature). Distances come from the real median distance of each area pair.
The map needs internet (Leaflet + free OpenStreetMap tiles, Esri as fallback, no API key); cars and route are illustrative, area positions are approximate.
