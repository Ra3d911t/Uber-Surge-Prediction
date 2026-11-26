"""Outlier tools: an IQR report and a Winsorizer that learns its clip limits from the training fold only."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each numeric column to [q_low, q_high] percentiles learned in `fit`.

    Used inside the Pipeline, so in cross-validation the limits come from the training fold only.
    """

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low_ = np.nanquantile(X, self.lower, axis=0)
        self.high_ = np.nanquantile(X, self.upper, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, ["low_", "high_"])
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features) if input_features is not None else None


def iqr_report(df, columns, k=1.5):
    """Count values outside [Q1 - k*IQR, Q3 + k*IQR] per column."""
    rows = []
    for c in columns:
        s = df[c].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        flagged = ((s < lo) | (s > hi)).sum()
        rows.append({
            "column": c, "Q1": q1, "Q3": q3, "IQR": iqr, "lower_fence": lo, "upper_fence": hi,
            "flagged": int(flagged), "flagged_%": flagged / len(s) * 100,
            "min": s.min(), "max": s.max(),
            "share_of_zeros_%": (s == 0).mean() * 100, "share_at_max_%": (s == s.max()).mean() * 100,
        })
    return pd.DataFrame(rows).set_index("column")
