"""Prediction helper shared by the GUI and its tests (same feature code as training)."""
import datetime as dt

import numpy as np
import pandas as pd

from src.features import PRODUCTS, build_features, select_features


def predict_surges(model, source: str, destination: str, distance: float, when: dt.datetime, weather: dict) -> dict:
    """Predicted surge multiplier for every Lyft product, for one trip."""
    rows = [{"pickup_datetime": when, "source": source, "destination": destination, "name": p,
             "distance": distance, **weather} for p in PRODUCTS]
    X = select_features(build_features(pd.DataFrame(rows)))
    pred = np.clip(model.predict(X), 1.0, 3.0)
    return {p: float(v) for p, v in zip(PRODUCTS, pred)}
