# Predicting the remaining flight time from ADS-B data

How long until this aircraft lands? This project predicts the **remaining flight time** of commercial flights at any moment of their flight, from the state vectors broadcast by the aircraft (position, altitude, speed, heading…) and collected by the [OpenSky Network](https://opensky-network.org).

It covers the whole chain on one full day of real air traffic (2026-03-01): exploring the data, building a clean dataset from ~700 million raw state vectors, engineering features, and comparing linear models and gradient boosting with a single, fixed evaluation protocol.

> **Status: v0.1, work in progress.** All the results below are measured on the validation flights; the test flights are kept untouched for the final evaluation (see the [next steps](#next-steps)).

## Results

Mean absolute error (MAE) on the validation set (5,141 flights, 308,460 state vectors), in minutes:

| Model | MAE | 95 % interval | Median error | 95th percentile |
|---|---|---|---|---|
| Constant (median of the training set) | 44.0 | | 23.5 | 149.0 |
| Naive physical estimate: distance / current speed | 16.1 | [15.3, 16.9] | 8.7 | 42.5 |
| Linear model, MAE loss | 8.1 | [7.5, 8.8] | 3.0 | 16.0 |
| Linear models, one per flight phase¹ | 7.5 | [6.9, 8.2] | 2.6 | 15.1 |
| **Gradient boosting, MAE loss** | **5.2** | **[4.7, 5.6]** | **1.9** | **11.8** |

¹ Assumes that the phase of flight (climb, cruise, descent) is known at any time.

- **Gradient boosting divides the error of the naive physical estimate by three**, and makes about 30 % less error than the best linear model (−2.4 minutes, with a 95 % interval of [−2.9, −1.9]). It also improves the 95th percentile and every phase of flight.
- **It learns the phase of flight by itself**: given the phase as an extra input, it does not improve, while the linear models need one model per phase to benefit from it.
- **The largest errors come from the data, not from the models:** 4 % of the flights have an impossible label (several flights of the same aircraft merged into one record, or a wrong arrival airport), and they make more than a third of the error of every model (see [known limitations](#known-limitations)).

The full comparison is in [`reports/validation_results.md`](reports/validation_results.md).

## Approach

The notebooks, in order:

| Notebook | Content |
|---|---|
| [01 — Exploring the OpenSky API](notebooks/01_data_exploration.ipynb) | What a state vector, a flight and a track are; first definition of the label; why the REST API is not enough to build a dataset |
| [02 — Data processing](notebooks/02_data_processing.ipynb) | Matching ~700 M raw state vectors to their flights, cleaning the altitude errors (spikes, offset segments, impossible values), selecting the commercial flights, detecting the flight phases, sampling 60 points per flight (15 climb / 30 cruise / 15 descent): 2.1 M state vectors of 35,000 flights |
| [03 — Feature engineering](notebooks/03_feature_engineering.ipynb) | Airport coordinates, train / validation / test split **by flight**, 22 numeric features (distance and naive time to arrival, progress along the route, heading towards the destination, altitudes, speeds…), PCA |
| [04 — Linear models](notebooks/04_linear_models.ipynb) | Choice of the metrics before any experiment; Ridge vs Huber vs MAE loss; one model per flight phase; analysis of the worst-predicted flights |
| [05 — Gradient boosting](notebooks/05_gradient_boosting.ipynb) | `HistGradientBoostingRegressor`, tuned with an early stopping on the validation flights; permutation importance; the phase of flight as a feature; comparison with the linear models |

A few methodological choices that run through the project:

- **The evaluation protocol is fixed before the experiments.** The MAE decides (it reads as "minutes of error, on average"); the MAE of each flight phase and the 95th percentile of the error are guardrails; the median is reported.
- **Everything is split by flight.** The 60 points of a flight are strongly correlated, so the train / validation / test split, the confidence intervals (paired bootstrap over the flights) and the early stopping of the gradient boosting all work on whole flights. Scikit-learn's built-in early stopping, which holds out random rows, is deliberately not used.
- **The MAE loss rather than the squared error.** A few impossible labels would drive a squared loss; the absolute loss is robust to them, and wins for both the linear models and the gradient boosting.

## Known limitations

- **Impossible labels.** About 4 % of the validation flights have a label that cannot be right: the same departure and arrival airport, or an average speed far from that of the flights of the same route length (typically several flights of the same aircraft merged into one OpenSky record). They make 36 to 41 % of the error of every model; without them, the MAE of the gradient boosting would be 3.45 minutes instead of 5.15. They are only flagged after the fact (notebooks 04 and 05), not yet removed from the dataset.
- **A single day of data.** The validation flights flew on the same day as the training flights (same winds, same weather, same congestion): the errors measure the performance on new flights of the same day. On another day, they will probably be higher.
- **The per-phase linear model assumes that the phase of flight is known at any time** (for live flights, it could come from the flight plan). The gradient boosting does not need this assumption.
- **The altitude cleaning works hour by hour** (see the design note below): its rolling window is cut at the hour boundaries, which affects less than 1 % of the points.

## Next steps

1. **Rebuild the dataset flight by flight** instead of hour by hour, and **remove the impossible labels** at this stage.
2. Retrain and re-select the models on the cleaned data, then **evaluate the final models once on the test flights**.

## Reproducing the project

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .  # makes the shared code of src/opensky_eta importable from the notebooks
```

Developed with Python 3.14; the pinned versions of `requirements.txt` need Python 3.12 or later.

### Data

The OpenSky historical database is queried through Trino, but its access is reserved to researchers and academics. This project therefore uses the **[public datasets](https://opensky-network.org/data/scientific#d11)** that OpenSky publishes on its website, which are exports of this database: the dataset *Complete One Day Trino Tables Snapshot (March 2026)*, i.e. one UTC day (2026-03-01) of the tables `state_vectors_data4` (one row per aircraft and per second, ~12 GB) and `flights_data4`. Download them and store the parquet files in `data/raw/state_vectors/` and `data/raw/flights/`. The data is not part of this repository.

### Running the pipeline

The essential steps can be run without the notebooks, with three commands:

| Command | What it does |
|---|---|
| `python -m opensky_eta.dataset` | raw files → cleaned state vectors (`data/processed/clean/`) → `ml_dataset.parquet` → `features.parquet` |
| `python -m opensky_eta.train` | retrains the three selected models with the hyperparameters of `config.py`, saves them to `data/models/`, prints the comparison on the validation flights |
| `python -m opensky_eta.evaluate` | loads the saved models and writes the comparison on the validation flights to `reports/validation_results.md` |

The commands use the same functions and write the same files as the notebooks (the dataset command reproduces the files of notebooks 02 and 03 byte for byte). The hyperparameters are not searched again by `train`: they were chosen in notebooks 04 and 05.

**Design note: processing by hour.** The raw data comes as 24 hourly files, and the dataset is built hour file by hour file, as in notebook 02. Only the cleaned state vectors are kept on disk; the other intermediate files are temporary. Rebuilding the dataset by flight (whole flights, all hours together) would remove the hour-boundary limit of the cleaning and simplify the processing: it is the next step of the project.

### Credentials

Notebook 01 queries the live [OpenSky REST API](https://openskynetwork.github.io/opensky-api/rest.html). Anonymous requests are enough for `/states/all`; the `/flights` and `/tracks` endpoints need an API client: log in on [opensky-network.org](https://opensky-network.org), open the *Account* page and create an API client. Its client id and secret are read from a local `.env` file, ignored by git (see `.env.example`):

```bash
OPENSKY_CLIENT_ID=...
OPENSKY_CLIENT_SECRET=...
```

The API only accepts the OAuth2 *client credentials* flow: the client id and secret are exchanged for an access token (valid 30 minutes) at `https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token`, then sent as a `Bearer` header.

Since notebook 01 works on live data, running it again gives different numbers from the saved run.

## Project structure

```
├── notebooks/            the analysis, 01 to 05
├── src/opensky_eta/      the code shared by the notebooks and the commands
│   ├── config.py         paths, parameters, selected hyperparameters
│   ├── dataset.py        dataset pipeline (python -m opensky_eta.dataset)
│   ├── models.py         model constructors and the selected models
│   ├── train.py          python -m opensky_eta.train
│   ├── evaluate.py       python -m opensky_eta.evaluate
│   └── ...               cleaning, features, preprocessing, evaluation metrics, plots
├── reports/              comparison tables written by the commands
└── data/                 raw and processed data, saved models (not versioned)
```

## Credits

The data comes from the [OpenSky Network](https://opensky-network.org), used under its [terms of use](https://opensky-network.org/about/terms-of-use) (research and non-commercial purposes). As requested by OpenSky, work using its data cites:

> Matthias Schäfer, Martin Strohmeier, Vincent Lenders, Ivan Martinovic and Matthias Wilhelm. *Bringing Up OpenSky: A Large-scale ADS-B Sensor Network for Research*. In Proceedings of the 13th IEEE/ACM International Symposium on Information Processing in Sensor Networks (IPSN), pages 83-94, April 2014.

The airport coordinates come from [OurAirports](https://ourairports.com/data/) (public domain).

Claude Code was used as a coding assistant during the development of this project, helping with implementation, debugging, and code organization.
