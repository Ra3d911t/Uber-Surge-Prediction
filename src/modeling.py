"""Shared modelling settings so notebooks 04 (comparison) and 05 (tuning) use the exact same CV setup."""
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

RANDOM_STATE = 42
CV_SAMPLE_SIZE = 30_000   # SVR/KNN/Polynomial are too slow for 5-fold CV on all 245k rows
N_SPLITS = 5


def load_split(path="data/processed"):
    X_train = pd.read_csv(f"{path}/X_train.csv")
    X_test = pd.read_csv(f"{path}/X_test.csv")
    y_train = pd.read_csv(f"{path}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{path}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def cv_sample(X_train, y_train, n=CV_SAMPLE_SIZE):
    """A fixed random sample of the TRAINING set that keeps time order (X_train is saved oldest -> newest)."""
    idx = X_train.sample(n=min(n, len(X_train)), random_state=RANDOM_STATE).index.sort_values()
    return X_train.loc[idx].reset_index(drop=True), y_train.loc[idx].reset_index(drop=True)


def make_cv():
    """The CV splitter agreed for the project: data is time ordered, so TimeSeriesSplit."""
    return TimeSeriesSplit(n_splits=N_SPLITS)


SCORING = {
    "RMSE": "neg_root_mean_squared_error",
    "MSE": "neg_mean_squared_error",
    "R2": "r2",
}
