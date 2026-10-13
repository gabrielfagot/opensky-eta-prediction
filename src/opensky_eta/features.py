"""Airport coordinates, train / validation / test split and stateless features (notebook 03)."""
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import GroupShuffleSplit

from opensky_eta.config import (
    AIRPORTS_PATH, AIRPORTS_URL, EARTH_RADIUS_KM, FLIGHT_KEY, KT_TO_KMH, M_TO_FT,
    MAX_NAIVE_TIME_H, MIN_SPEED_KT, MS_TO_FT_MIN, MS_TO_KT, RANDOM_STATE, TEST_SHARE,
    VALIDATION_SHARE,
)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points given in degrees."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def initial_bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing (0-360°, clockwise from north) from point 1 to point 2."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    x = np.sin(lon2 - lon1) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return np.degrees(np.arctan2(x, y)) % 360


def load_airport_lookup(path=AIRPORTS_PATH, url=AIRPORTS_URL):
    """Latitude, longitude and elevation of every airport code (OurAirports, downloaded once).

    A code is looked up in the `ident` column first, then in the `gps_code` column.
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)

    airports = pd.read_csv(
        path,
        usecols=["ident", "gps_code", "latitude_deg", "longitude_deg", "elevation_ft"],
    )
    by_ident = airports.set_index("ident")
    by_gps_code = airports.dropna(subset=["gps_code"]).drop_duplicates("gps_code").set_index("gps_code")
    return pd.concat(
        [by_ident, by_gps_code[~by_gps_code.index.isin(by_ident.index)]]
    )[["latitude_deg", "longitude_deg", "elevation_ft"]]


def add_airport_coordinates(df, airport_lookup):
    """Adds the coordinates and elevation of the departure and arrival airports (NaN if unknown)."""
    for role, column in [("departure", "estDepartureAirport"), ("arrival", "estArrivalAirport")]:
        coordinates = airport_lookup.reindex(df[column]).to_numpy()
        df[f"{role}_lat"] = coordinates[:, 0]
        df[f"{role}_lon"] = coordinates[:, 1]
        df[f"{role}_elevation_ft"] = coordinates[:, 2]
    return df


def split_by_flight(df, test_share=TEST_SHARE, validation_share=VALIDATION_SHARE,
                    random_state=RANDOM_STATE):
    """"train" / "validation" / "test" for every row; all the rows of a flight get the same split."""
    groups = df.groupby(FLIGHT_KEY, sort=False).ngroup().to_numpy()

    outer = GroupShuffleSplit(n_splits=1, test_size=test_share, random_state=random_state)
    train_val_index, test_index = next(outer.split(df, groups=groups))
    inner = GroupShuffleSplit(
        n_splits=1, test_size=validation_share / (1 - test_share), random_state=random_state
    )
    train_index, val_index = next(inner.split(train_val_index, groups=groups[train_val_index]))

    split = np.empty(len(df), dtype=object)
    split[test_index] = "test"
    split[train_val_index[train_index]] = "train"
    split[train_val_index[val_index]] = "validation"
    return pd.Categorical(split, categories=["train", "validation", "test"])


def add_stateless_features(df):
    """Adds the features that learn nothing from the data (computed row by row)."""
    df["distance_to_arrival_km"] = haversine_km(df["lat"], df["lon"], df["arrival_lat"], df["arrival_lon"])
    df["distance_from_departure_km"] = haversine_km(df["departure_lat"], df["departure_lon"], df["lat"], df["lon"])
    df["route_length_km"] = haversine_km(
        df["departure_lat"], df["departure_lon"], df["arrival_lat"], df["arrival_lon"]
    )
    travelled_and_left = df["distance_from_departure_km"] + df["distance_to_arrival_km"]
    df["route_progress"] = (df["distance_from_departure_km"] / travelled_and_left).where(travelled_and_left > 0, 0.0)

    bearing_to_arrival = initial_bearing_deg(df["lat"], df["lon"], df["arrival_lat"], df["arrival_lon"])
    heading_error = np.radians((df["heading"] - bearing_to_arrival + 180) % 360 - 180)
    df["heading_error_cos"] = np.cos(heading_error)
    df["heading_error_sin"] = np.sin(heading_error)

    df["heading_sin"] = np.sin(np.radians(df["heading"]))
    df["heading_cos"] = np.cos(np.radians(df["heading"]))
    hour = (df["time"] % 86_400) / 3_600
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    df["altitude_ft"] = df["baroAltitude"] * M_TO_FT
    df["geo_altitude_ft"] = df["geoAltitude"] * M_TO_FT
    df["speed_kt"] = df["velocity"] * MS_TO_KT
    df["vertical_rate_ft_min"] = df["vertRate"] * MS_TO_FT_MIN
    df["height_above_arrival_ft"] = df["altitude_ft"] - df["arrival_elevation_ft"]

    # Naive time to arrival: remaining distance flown at the current ground speed.
    # The speed is floored so that a (rare) near-zero speed does not explode the ratio,
    # and the result is capped: early in the climb, the speed is still low and the
    # estimate can exceed 24 h, which is meaningless (no flight lasts more than 20 h).
    speed_kmh = df["speed_kt"].clip(lower=MIN_SPEED_KT) * KT_TO_KMH
    df["naive_time_to_arrival_s"] = (
        df["distance_to_arrival_km"] / speed_kmh * 3600
    ).clip(upper=MAX_NAIVE_TIME_H * 3600)
    return df
