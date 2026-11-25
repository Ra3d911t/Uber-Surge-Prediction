"""Feature engineering shared by training, evaluation and the GUI.

Every consumer (notebooks and app.py) must call `build_features` and then `select_features`,
so the model always sees the same columns, built the same way.
"""
import pandas as pd

TARGET = "surge_multiplier"

CATEGORICAL_FEATURES = ["source", "destination", "name", "short_summary"]
NUMERIC_FEATURES = [
    "distance",
    "hour",
    "day_of_week",
    "is_weekend",
    "temperature",
    "precipIntensity",
    "humidity",
    "windSpeed",
    "visibility",
    "cloudCover",
]
# Final column order expected by models/best_model.joblib
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Known before the trip is requested? No -> never use as a feature.
LEAKAGE_COLUMNS = ["price"]

# Allowed category values (from the cleaned Lyft data). The GUI uses these for its dropdowns.
LOCATIONS = [
    "Back Bay", "Beacon Hill", "Boston University", "Fenway", "Financial District",
    "Haymarket Square", "North End", "North Station", "Northeastern University",
    "South Station", "Theatre District", "West End",
]
PRODUCTS = ["Shared", "Lyft", "Lyft XL", "Lux", "Lux Black", "Lux Black XL"]
WEATHER_SUMMARIES = [
    "Clear", "Partly Cloudy", "Mostly Cloudy", "Overcast", "Foggy",
    "Possible Drizzle", "Drizzle", "Light Rain", "Rain",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive time features from `pickup_datetime`. Returns a copy."""
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.day_of_week        # Monday = 0
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    if "short_summary" in df:
        df["short_summary"] = df["short_summary"].astype(str).str.strip()
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the model input columns, in the trained order."""
    return df[FEATURE_COLUMNS]
