"""Imbalance strategies for a regression target where ~93% of rides have surge = 1.0.

SMOTE is for classes; for a continuous target we use
- sample weights: surged rides (y > 1) count `factor` times in the loss,
- random oversampling: duplicate surged rides until they make up `target_share` of the training rows.

Both are applied ONLY to the training part of each CV fold (`cv_balance_eval`); validation folds keep
the real distribution, so there is no leakage.
"""
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score

SURGE_THRESHOLD = 1.0


def surge_weights(y, factor):
    """Weight `factor` for surged rides, 1 for the rest."""
    y = np.asarray(y)
    return np.where(y > SURGE_THRESHOLD, float(factor), 1.0)


def oversample(X, y, target_share, random_state=42):
    """Duplicate random surged rows until they are `target_share` of all rows. Returns new X, y."""
    y = pd.Series(np.asarray(y), index=X.index)
    surged = y.index[y > SURGE_THRESHOLD]
    n_total, n_surged = len(y), len(surged)
    if n_surged == 0 or n_surged / n_total >= target_share:
        return X, y
    # solve (n_surged + k) / (n_total + k) = share  ->  k = (share*n_total - n_surged) / (1 - share)
    k = int(np.ceil((target_share * n_total - n_surged) / (1 - target_share)))
    rng = np.random.default_rng(random_state)
    extra = rng.choice(surged, size=k, replace=True)
    idx = np.concatenate([X.index.to_numpy(), extra])
    return X.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True)


def apply_strategy(X, y, strategy):
    """strategy: ("none",) | ("weight", factor) | ("oversample", share). Returns X, y, fit_params."""
    kind = strategy[0]
    if kind == "none":
        return X, y, {}
    if kind == "weight":
        return X, y, {"model__sample_weight": surge_weights(y, strategy[1])}
    if kind == "oversample":
        Xo, yo = oversample(X, y, strategy[1])
        return Xo, yo, {}
    raise ValueError(f"unknown strategy {strategy}")


def fold_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    s = y_true > SURGE_THRESHOLD
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "RMSE surged only": np.sqrt(mean_squared_error(y_true[s], y_pred[s])) if s.any() else np.nan,
        "R2": r2_score(y_true, y_pred),
        "Mean pred (non-surged)": y_pred[~s].mean(),
        "Mean pred (surged)": y_pred[s].mean() if s.any() else np.nan,
    }


def cv_balance_eval(pipeline, X, y, cv, strategy):
    """Cross-validate `pipeline` with a balancing strategy applied to each TRAIN fold only."""
    rows = []
    for fold, (tr, va) in enumerate(cv.split(X)):
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        X_tr, y_tr, fit_params = apply_strategy(X_tr, y_tr, strategy)
        model = clone(pipeline)
        start = time.time()
        model.fit(X_tr, y_tr, **fit_params)
        m = fold_metrics(y.iloc[va], model.predict(X.iloc[va]))
        m.update(fold=fold, fit_time=time.time() - start, train_rows=len(y_tr),
                 train_surged_share=float((np.asarray(y_tr) > SURGE_THRESHOLD).mean()))
        rows.append(m)
    return pd.DataFrame(rows)
