"""Unfitted preprocessing used inside every model Pipeline.

The preprocessor is fitted as part of the Pipeline, so in cross-validation each fold learns the
imputation values, scaling and categories from its own training part only (no leakage).
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Calendar features are not measurements, so they can never be "outliers" and are never winsorized.
CALENDAR_FEATURES = ["hour", "day_of_week", "is_weekend"]


def make_preprocessor(X, winsorize=None):
    """Return an unfitted ColumnTransformer for the columns present in X.

    winsorize: None (default, used by the final model) or (lower, upper) quantiles, e.g. (0.01, 0.99).
    When set, only the continuous numeric columns (distance and weather) are clipped, at percentiles learned
    from the training fold only. Calendar columns (hour, day_of_week, is_weekend) are left untouched.
    """
    numeric_features = [c for c in NUMERIC_FEATURES if c in X.columns]
    categorical_features = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    transformers = []

    if numeric_features and winsorize is None:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    elif numeric_features:
        from src.outliers import Winsorizer
        continuous = [c for c in numeric_features if c not in CALENDAR_FEATURES]
        calendar = [c for c in numeric_features if c in CALENDAR_FEATURES]
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("winsorizer", Winsorizer(*winsorize)),
            ("scaler", StandardScaler()),
        ]), continuous))
        if calendar:
            transformers.append(("calendar", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), calendar))

    if categorical_features:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")
