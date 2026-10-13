"""Paths, units, identifiers and parameters shared by the notebooks.

The parameters of each processing step are justified in the notebook that uses them
(02: cleaning, IFR selection, phases, sampling; 03: split and features).
"""
from pathlib import Path

# ------------------------------------------------------------------
# Paths (absolute, so that the code works from any working directory)
# ------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"

RAW_STATE_VECTORS_DIR = DATA_DIR / "raw" / "state_vectors"
RAW_FLIGHTS_DIR = DATA_DIR / "raw" / "flights"

PROCESSED_DIR = DATA_DIR / "processed"
MATCHED_DIR = PROCESSED_DIR / "matched"
CLEAN_DIR = PROCESSED_DIR / "clean"
IFR_DIR = PROCESSED_DIR / "ifr"
PHASES_DIR = PROCESSED_DIR / "phases"
CRUISE_INFO_PATH = PROCESSED_DIR / "cruise_info.parquet"
ML_DIR = PROCESSED_DIR / "ml"
ML_DATASET_PATH = ML_DIR / "ml_dataset.parquet"

AIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
AIRPORTS_PATH = DATA_DIR / "external" / "airports.csv"
FEATURES_DIR = PROCESSED_DIR / "features"
FEATURES_PATH = FEATURES_DIR / "features.parquet"
# Validation predictions of the reference models (saved by notebook 04, compared in notebook 05)
LINEAR_PREDICTIONS_PATH = PROCESSED_DIR / "predictions" / "validation_linear.parquet"
# Selected models, saved for the final evaluation on the test set
MODELS_DIR = DATA_DIR / "models"
LINEAR_MODEL_PATH = MODELS_DIR / "linear_mae.joblib"                # notebook 04, part 1
SEGMENT_MODELS_PATH = MODELS_DIR / "linear_mae_per_segment.joblib"  # notebook 04, part 2
BOOSTING_MODEL_PATH = MODELS_DIR / "gradient_boosting.joblib"       # notebook 05
REPORTS_DIR = PROJECT_DIR / "reports"  # comparison tables written by `python -m opensky_eta.evaluate`

BATCH_SIZE = 500_000  # rows per batch when streaming a parquet file

# ------------------------------------------------------------------
# Units and identifiers
# ------------------------------------------------------------------
M_TO_FT = 1 / 0.3048
MS_TO_KT = 1.94384
MS_TO_FT_MIN = M_TO_FT * 60
KT_TO_KMH = 1.852
EARTH_RADIUS_KM = 6371.0
SECONDS_PER_HOUR = 3600

# A flight is identified by its aircraft and its time window
FLIGHT_KEY = ["icao24", "firstSeen", "lastSeen"]
LABEL = "remaining_time"  # seconds

RANDOM_STATE = 42

# ------------------------------------------------------------------
# Notebook 02, section 3: altitude cleaning
# ------------------------------------------------------------------
ROLLING_WINDOW = 31              # points (~30 s at one point per second)
MAX_DEVIATION_FT = 1_000         # max distance to the rolling median
MAX_BARO_GEO_GAP_FT = 5_000      # max |baroAltitude - geoAltitude|
MIN_JUMP_FT = 1_000              # smaller steps are measurement noise
MAX_APPARENT_VZ_FT_MIN = 7_500   # faster apparent vertical speeds are impossible
MAX_JUMP_MISMATCH = 0.5          # the exit jump must cancel the entry jump
MAX_SEGMENT_DURATION_S = 5 * 60  # longer "offset segments" are real flight
SEGMENT_RESIDUAL_FT = 5_000      # deviation that confirms an offset segment
ALTITUDE_CEILING_FT = 52_000     # no airliner or business jet flies higher
MIN_ALTITUDE_FT = -1_500         # the lowest airport is ~1,300 ft below sea level
MAX_SPEED_KT = 750               # record airliner ground speeds are ~700 kt
MAX_VERTICAL_RATE_FT_MIN = 10_000  # airliners rarely exceed 6,000 ft/min

# ------------------------------------------------------------------
# Notebook 02, section 4: IFR flight selection
# ------------------------------------------------------------------
MIN_MAX_ALTITUDE_FT = 10_000
MIN_CRUISE_SPEED_KT = 200
MIN_DURATION_MIN = 30
MAX_DURATION_H = 20              # the longest scheduled flights last ~19 h

# ------------------------------------------------------------------
# Notebook 02, section 5: flight phases
# ------------------------------------------------------------------
LEVEL_VZ_FT_MIN = 200            # |vertical rate| below this = level flight
ALTITUDE_BIN_FT = 500
CRUISE_BAND_FT = 500             # cruise band = cruise level +/- this
SHORT_CRUISE_S = 10 * 60
PHASES = ["climb", "est_climb", "cruise", "short cruise", "descent", "est_descent"]

# ------------------------------------------------------------------
# Notebook 02, section 6: sampling (points per segment; each segment groups two phases)
# ------------------------------------------------------------------
N_POINTS = {"pre": 15, "cruise": 30, "post": 15}
SEGMENTS = list(N_POINTS)
POINTS_PER_FLIGHT = sum(N_POINTS.values())  # 60
PHASE_TO_SEGMENT = {"climb": "pre", "est_climb": "pre", "cruise": "cruise",
                    "short cruise": "cruise", "descent": "post", "est_descent": "post"}
MAX_CRUISE_GAP_FT = 5_000        # max (max altitude - cruise level)

# ------------------------------------------------------------------
# Notebook 03: split and features
# ------------------------------------------------------------------
TEST_SHARE = 0.15
VALIDATION_SHARE = 0.15
MIN_AIRPORT_FLIGHTS = 50  # rarer airports are grouped into one "infrequent" category
MIN_SPEED_KT = 100        # speed floor of the naive time to arrival (avoids dividing by ~0)
MAX_NAIVE_TIME_H = 24     # cap of the naive time to arrival (no flight lasts more than 20 h)

NUMERIC_FEATURES = [
    "distance_to_arrival_km", "naive_time_to_arrival_s", "distance_from_departure_km",
    "route_length_km", "route_progress", "heading_error_cos", "heading_error_sin",
    "heading_sin", "heading_cos", "hour_sin", "hour_cos", "altitude_ft", "geo_altitude_ft",
    "height_above_arrival_ft", "speed_kt", "vertical_rate_ft_min",
    "lat", "lon", "departure_lat", "departure_lon", "arrival_lat", "arrival_lon",
]
CATEGORICAL_FEATURES = ["estDepartureAirport", "estArrivalAirport"]

# ------------------------------------------------------------------
# Selected models, with the hyperparameters tuned in notebooks 04 and 05
# (retrained as they are by `python -m opensky_eta.train`, no search)
# ------------------------------------------------------------------
LINEAR_MAE_ALPHA = 1e-5                                         # notebook 04, part 1
SEGMENT_MAE_ALPHAS = {"pre": 1e-7, "cruise": 1e-9, "post": 1e-8}  # notebook 04, part 2 (strategy B)
BOOSTING_PARAMS = {"loss": "absolute_error", "learning_rate": 0.1, "max_leaf_nodes": 63,
                   "min_samples_leaf": 20, "max_iter": 1500}      # notebook 05
