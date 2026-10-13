"""The models of the project: building blocks used by notebooks 04 and 05, and the three selected
models retrained by `python -m opensky_eta.train` with the hyperparameters of `config.py`."""
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline

from opensky_eta.config import (
    BOOSTING_PARAMS, CATEGORICAL_FEATURES, LABEL, LINEAR_MAE_ALPHA, MIN_AIRPORT_FLIGHTS, N_POINTS,
    NUMERIC_FEATURES, PHASE_TO_SEGMENT, RANDOM_STATE, SECONDS_PER_HOUR, SEGMENT_MAE_ALPHAS, SEGMENTS,
)
from opensky_eta.preprocessing import make_preprocessor, make_tree_preprocessor

MAE_LOSS = {"loss": "epsilon_insensitive", "epsilon": 0.0}  # epsilon-insensitive with epsilon = 0 is |r|
MODEL_NAMES = ["Linear, MAE loss", "Per segment, MAE loss", "Gradient boosting, MAE loss"]


# ----------------------------------------------------------------------------------------------
# Linear models trained by SGD (notebook 04): fit them on the remaining time in hours
# ----------------------------------------------------------------------------------------------
def make_sgd_regressor(alpha, **loss_params):
    """Linear model with an L2 penalty, trained by SGD with the given loss (fit it on a target in hours)."""
    return SGDRegressor(penalty="l2", alpha=alpha, max_iter=50, tol=1e-4, average=True,
                        random_state=RANDOM_STATE, **loss_params)


def make_sgd_pipeline(alpha, **loss_params):
    """Preprocessing + linear model trained by SGD with the given loss."""
    return Pipeline([("preprocess", make_preprocessor()),
                     ("model", make_sgd_regressor(alpha, **loss_params))])


def make_mae_regressor(alpha):
    """Linear model with an MAE loss and an L2 penalty (used per segment in part 2)."""
    return make_sgd_regressor(alpha, **MAE_LOSS)


def make_segment_pipeline(segment, alpha):
    """Model of one flight segment (strategy B of notebook 04): own preprocessor, airport threshold in flights."""
    return Pipeline([
        ("preprocess", make_preprocessor(min_airport_rows=MIN_AIRPORT_FLIGHTS * N_POINTS[segment])),
        ("model", make_mae_regressor(alpha)),
    ])


def predict_per_segment(segment_pipelines, X, segment):
    """Prediction (seconds) of the per-segment model: each row goes to the pipeline of its segment."""
    segment = np.asarray(segment)
    prediction = np.full(len(X), np.nan)
    for name, pipeline in segment_pipelines.items():
        rows = segment == name
        prediction[rows] = pipeline.predict(X[rows]) * SECONDS_PER_HOUR
    assert not np.isnan(prediction).any(), "every row must belong to a segment"
    return prediction


# ----------------------------------------------------------------------------------------------
# Gradient boosting (notebook 05)
# ----------------------------------------------------------------------------------------------
def make_boosting_pipeline(loss, max_leaf_nodes, min_samples_leaf=20, learning_rate=0.1, max_iter=50,
                           categorical_features=CATEGORICAL_FEATURES, warm_start=False):
    """Tree preprocessing + gradient boosting with the given loss.

    The built-in early stopping is disabled: it would hold out *random rows* of the training set,
    i.e. points of flights whose other points are used for training. With `warm_start=True`,
    increasing `max_iter` and fitting again adds trees to the existing ones.
    """
    return Pipeline([
        ("preprocess", make_tree_preprocessor(categorical_features=categorical_features)),
        ("model", HistGradientBoostingRegressor(
            loss=loss, learning_rate=learning_rate, max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
            categorical_features=categorical_features, early_stopping=False, warm_start=warm_start,
            random_state=RANDOM_STATE,
        )),
    ])


# ----------------------------------------------------------------------------------------------
# The three selected models
# ----------------------------------------------------------------------------------------------
def train_selected_models(features):
    """Trains the three selected models on the training flights of `features.parquet`.

    Returns {name: fitted model}; the per-segment model is a dictionary {segment: pipeline}.
    """
    train = features[features["split"] == "train"]
    X, y = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train[LABEL]
    segment = train["phase_of_flight"].map(PHASE_TO_SEGMENT).to_numpy()
    models, start = {}, time.time()

    def done(name):
        print(f"[{(time.time() - start) / 60:5.1f} min] {name} trained", flush=True)

    models["Linear, MAE loss"] = make_sgd_pipeline(LINEAR_MAE_ALPHA, **MAE_LOSS).fit(X, y / SECONDS_PER_HOUR)
    done("Linear, MAE loss")
    models["Per segment, MAE loss"] = {
        name: make_segment_pipeline(name, SEGMENT_MAE_ALPHAS[name]).fit(X[segment == name], y[segment == name] / SECONDS_PER_HOUR)
        for name in SEGMENTS
    }
    done("Per segment, MAE loss")
    models["Gradient boosting, MAE loss"] = make_boosting_pipeline(**BOOSTING_PARAMS).fit(X, y)
    done("Gradient boosting, MAE loss")
    return models


def predict_selected_models(models, frame):
    """Predictions (seconds) of the three selected models on the rows of `frame`."""
    X = frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    return {
        "Linear, MAE loss": models["Linear, MAE loss"].predict(X) * SECONDS_PER_HOUR,
        "Per segment, MAE loss": predict_per_segment(models["Per segment, MAE loss"], X,
                                                     frame["phase_of_flight"].map(PHASE_TO_SEGMENT)),
        "Gradient boosting, MAE loss": models["Gradient boosting, MAE loss"].predict(X),
    }
