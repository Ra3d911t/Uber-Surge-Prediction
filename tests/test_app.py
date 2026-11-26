"""GUI + model sanity tests. Run from the project root:  python -m tests.test_app"""
import datetime as dt

import joblib
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.features import FEATURE_COLUMNS
from src.map_data import trip_distance
from src.predict import predict_surges

DEFAULT_WEATHER = {"short_summary": "Overcast", "temperature": 40.0, "precipIntensity": 0.0, "humidity": 0.72,
                   "windSpeed": 5.0, "visibility": 10.0, "cloudCover": 0.8}
WHEN = dt.datetime(2018, 12, 17, 18, 0)        # the GUI defaults


def test_saved_model_predicts_raw_rows():
    model = joblib.load("models/best_model.joblib")
    X_test = pd.read_csv("data/processed/X_test.csv")
    pred = model.predict(X_test.head(100))
    assert list(model.feature_names_in_) == FEATURE_COLUMNS
    assert len(pred) == 100 and ((pred >= 1.0) & (pred <= 3.0)).all()


def run_gui(pickup="Back Bay", dropoff="South Station", product="Lyft"):
    at = AppTest.from_file("../app.py", default_timeout=90).run()
    next(w for w in at.selectbox if w.label == "Pickup").set_value(pickup)
    next(w for w in at.selectbox if w.label == "Drop-off").set_value(dropoff)
    at.run()
    if pickup != dropoff:
        next(w for w in at.radio if w.label == "Choose a ride").set_value(product)
        at.run()
    return at


CASES = [("Back Bay", "South Station", "Lyft"), ("North End", "Fenway", "Lux Black XL"), ("Fenway", "West End", "Shared")]

if __name__ == "__main__":
    test_saved_model_predicts_raw_rows()
    print("model sanity: OK")
    model = joblib.load("models/best_model.joblib")
    for pickup, dropoff, product in CASES:
        at = run_gui(pickup, dropoff, product)
        assert not at.exception, at.exception
        shown = at.metric[0].value
        expected = predict_surges(model, pickup, dropoff, trip_distance(pickup, dropoff), WHEN, DEFAULT_WEATHER)[product]
        assert shown == f"{expected:.2f}x", (shown, expected)
        print(f"{pickup} -> {dropoff} ({product}): shown {shown} == model {expected:.2f}x  OK")
    at = run_gui("Back Bay", "Back Bay")
    assert not at.exception and "must be different" in at.error[0].value
    print("same pickup/drop-off ->", at.error[0].value)
