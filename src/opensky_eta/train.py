"""Training of the three selected models: `python -m opensky_eta.train`.

Reads `features.parquet`, retrains the selected models with the hyperparameters of `config.py`
(chosen in notebooks 04 and 05: no search here), saves them to `data/models/`, then prints and
writes the comparison on the validation flights (as `python -m opensky_eta.evaluate`).
"""
import joblib
import pandas as pd

from opensky_eta.config import (
    BOOSTING_MODEL_PATH, FEATURES_PATH, LINEAR_MODEL_PATH, MODELS_DIR, PROJECT_DIR, SEGMENT_MODELS_PATH,
)
from opensky_eta.evaluate import report
from opensky_eta.models import train_selected_models


def main():
    features = pd.read_parquet(FEATURES_PATH)
    models = train_selected_models(features)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, path in [("Linear, MAE loss", LINEAR_MODEL_PATH), ("Per segment, MAE loss", SEGMENT_MODELS_PATH),
                       ("Gradient boosting, MAE loss", BOOSTING_MODEL_PATH)]:
        joblib.dump(models[name], path)
        print(f"Saved {path.relative_to(PROJECT_DIR)}")

    report(models, features)


if __name__ == "__main__":
    main()
