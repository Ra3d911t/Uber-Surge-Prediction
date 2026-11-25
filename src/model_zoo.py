"""Single registry of every regression model in the project: default settings (notebook 04)
and GridSearchCV search spaces (notebook 05), so both notebooks always use the same models.

`course` says where the model sits relative to the course lectures:
- "lecture":  studied in the lectures (2 Linear Regression + Gradient Descent, 4 DT & RF, 6 SVM, 9 ANN)
- "extra":    classic regression model not clearly in the lecture list (kept as extra comparison)
- "newer":    newer version of a studied model (bonus B1): gradient-boosted trees upgrade DT / RF
- "baseline": reference only
"""
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.modeling import RANDOM_STATE

RS = RANDOM_STATE

# name: (course group, lecture/upgrade note, default estimator)
MODELS = {
    "Baseline (predict mean)": ("baseline", "reference", DummyRegressor(strategy="mean")),
    "Linear Regression": ("lecture", "L2 (closed form)", LinearRegression()),
    "Linear Regression (Gradient Descent)": ("lecture", "L2 + L10 (SGD)",
        SGDRegressor(random_state=RS, max_iter=2000, tol=1e-4)),
    "Decision Tree": ("lecture", "L4", DecisionTreeRegressor(random_state=RS)),
    "Random Forest": ("lecture", "L4", RandomForestRegressor(random_state=RS, n_jobs=1)),
    "SVR": ("lecture", "L6 (SVM for regression)", SVR()),
    "ANN (MLP)": ("lecture", "L9 + L10", MLPRegressor(random_state=RS, max_iter=300, early_stopping=True)),
    "Ridge": ("extra", "regularised linear", Ridge(random_state=RS)),
    "Lasso": ("extra", "regularised linear", Lasso(alpha=0.001, random_state=RS)),
    "Polynomial Regression (deg 2)": ("extra", "linear on poly terms", Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("lin", Ridge(alpha=1.0)),           # small ridge penalty keeps the 1,200+ poly terms numerically stable
    ])),
    "KNN": ("extra", "instance based", KNeighborsRegressor()),
    "XGBoost (bonus)": ("newer", "upgrade of DT/RF (boosting)",
        XGBRegressor(random_state=RS, n_jobs=1, verbosity=0)),
    "LightGBM (bonus)": ("newer", "upgrade of DT/RF (leaf-wise boosting)",
        LGBMRegressor(random_state=RS, n_jobs=1, verbose=-1)),
    "HistGradientBoosting (bonus)": ("newer", "upgrade of DT/RF (histogram boosting)",
        HistGradientBoostingRegressor(random_state=RS)),
}

SEARCH_SPACES = {
    "Linear Regression": {"model__fit_intercept": [True, False]},
    "Linear Regression (Gradient Descent)": {
        "model__alpha": [1e-5, 1e-3, 1e-1],
        "model__penalty": ["l2", "l1"],
        "model__learning_rate": ["invscaling", "adaptive"],
    },
    "Decision Tree": {
        "model__max_depth": [3, 5, 8, 12],
        "model__min_samples_leaf": [20, 100, 500],
    },
    "Random Forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [6, 12, None],
        "model__min_samples_leaf": [10, 50, 200],
    },
    "SVR": {
        "model__kernel": ["rbf"],
        "model__C": [0.1, 1, 10],
        "model__epsilon": [0.01, 0.1],
    },
    "ANN (MLP)": {
        "model__hidden_layer_sizes": [(32,), (64, 32)],
        "model__alpha": [1.0, 10.0, 100.0],          # strong L2 - the target is noisy
        "model__solver": ["sgd", "adam"],            # lecture 10: optimisation algorithms
    },
    "Ridge": {"model__alpha": [0.1, 1, 10, 100, 1000]},
    "Lasso": {"model__alpha": [1e-4, 1e-3, 1e-2, 1e-1]},
    "Polynomial Regression (deg 2)": {"model__lin__alpha": [1, 10, 100, 1000]},
    "KNN": {
        "model__n_neighbors": [15, 50, 100, 200],
        "model__weights": ["uniform", "distance"],
    },
    "XGBoost (bonus)": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.03, 0.1],
        "model__min_child_weight": [1, 50],
    },
    "LightGBM (bonus)": {
        "model__n_estimators": [200, 400],
        "model__num_leaves": [15, 31],
        "model__learning_rate": [0.03, 0.1],
        "model__min_child_samples": [20, 200],
    },
    "HistGradientBoosting (bonus)": {
        "model__max_iter": [200, 400],
        "model__max_depth": [3, 6],
        "model__learning_rate": [0.03, 0.1],
        "model__min_samples_leaf": [20, 200],
    },
}


def course_group(name):
    return MODELS[name][0]


def course_note(name):
    return MODELS[name][1]
