"""Fitted preprocessing of the features: wrapped in scikit-learn objects, fitted on the training rows only."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from opensky_eta.config import CATEGORICAL_FEATURES, MIN_AIRPORT_FLIGHTS, NUMERIC_FEATURES, POINTS_PER_FLIGHT


def make_preprocessor(numeric_features=NUMERIC_FEATURES, categorical_features=CATEGORICAL_FEATURES,
                      min_airport_rows=MIN_AIRPORT_FLIGHTS * POINTS_PER_FLIGHT):
    """Preprocessing of the linear models (notebook 03): scale the numeric features,
    one-hot encode the airports seen in at least `min_airport_rows` training rows
    (rarer ones -> one "infrequent" column)."""
    transformers = [("numeric", StandardScaler(), list(numeric_features))]
    if categorical_features:
        encoder = OneHotEncoder(
            min_frequency=min_airport_rows,  # in rows: 60 rows per flight over the whole flight
            handle_unknown="infrequent_if_exist",
        )
        transformers.append(("airports", encoder, list(categorical_features)))
    return ColumnTransformer(transformers)


def make_tree_preprocessor(numeric_features=NUMERIC_FEATURES, categorical_features=CATEGORICAL_FEATURES,
                           min_airport_rows=MIN_AIRPORT_FLIGHTS * POINTS_PER_FLIGHT):
    """Preprocessing of the tree models: numeric features unchanged (a split threshold does not
    depend on the scale), airports encoded as integer codes for the native categorical support
    of HistGradientBoosting.

    Same rule as the one-hot encoder for the frequent airports: rarer ones share one
    "infrequent" code. An airport never seen in training gets NaN, which the trees treat as
    a missing value. The output is a DataFrame that keeps the column names.
    """
    transformers = [("numeric", "passthrough", list(numeric_features))]
    if categorical_features:
        encoder = OrdinalEncoder(
            min_frequency=min_airport_rows,
            handle_unknown="use_encoded_value", unknown_value=np.nan,
        )
        transformers.append(("airports", encoder, list(categorical_features)))
    return ColumnTransformer(transformers, verbose_feature_names_out=False).set_output(transform="pandas")
