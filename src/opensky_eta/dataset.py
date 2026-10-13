"""Building the ML dataset: from the raw OpenSky files to `features.parquet`.

Each function is one step of notebooks 02 and 03, which call them and display their results.
`python -m opensky_eta.dataset` runs them all in a row. The processing is done hour file by hour
file, as in notebook 02; only the cleaned state vectors (`clean/`) are kept on disk, the other
intermediate files are written to a temporary folder and deleted as soon as they are no longer needed.
"""
import shutil
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from opensky_eta.cleaning import compute_flight_stats, offset_segments, rolling_median_outliers
from opensky_eta.config import (
    ALTITUDE_BIN_FT, ALTITUDE_CEILING_FT, BATCH_SIZE, CATEGORICAL_FEATURES, CLEAN_DIR, CRUISE_BAND_FT,
    CRUISE_INFO_PATH, FEATURES_PATH, FLIGHT_KEY, LABEL, LEVEL_VZ_FT_MIN, M_TO_FT, MAX_BARO_GEO_GAP_FT,
    MAX_CRUISE_GAP_FT, MAX_DURATION_H, MAX_SPEED_KT, MAX_VERTICAL_RATE_FT_MIN, MIN_ALTITUDE_FT,
    MIN_CRUISE_SPEED_KT, MIN_DURATION_MIN, MIN_MAX_ALTITUDE_FT, ML_DATASET_PATH, MS_TO_FT_MIN, MS_TO_KT,
    N_POINTS, NUMERIC_FEATURES, PHASE_TO_SEGMENT, PHASES, POINTS_PER_FLIGHT, PROCESSED_DIR, PROJECT_DIR,
    RAW_FLIGHTS_DIR, RAW_STATE_VECTORS_DIR, SEGMENTS, SHORT_CRUISE_S,
)
from opensky_eta.features import (
    add_airport_coordinates, add_stateless_features, load_airport_lookup, split_by_flight,
)
from opensky_eta.storage import file_time_range, list_parquet_files, to_utc, write_filtered

# Columns of the state vectors kept by the matching
MATCHED_COLUMNS = ["time", "icao24", "lat", "lon", "velocity", "heading", "vertRate",
                   "baroAltitude", "geoAltitude", "onGround"]
# Columns of features.parquet
FEATURE_TABLE_COLUMNS = (FLIGHT_KEY + ["time", "split", LABEL, "phase_of_flight"]
                         + NUMERIC_FEATURES + CATEGORICAL_FEATURES)


# ----------------------------------------------------------------------------------------------
# Notebook 02, sections 1 and 2: flights and matching
# ----------------------------------------------------------------------------------------------
def load_flights(directory=RAW_FLIGHTS_DIR):
    """The raw flights table (one row per flight)."""
    return pd.concat([pd.read_parquet(path) for path in sorted(directory.glob("*.parquet"))],
                     ignore_index=True)


def make_flights_lookup(flights):
    """Flights with a known `firstSeen` and both estimated airports, sorted by aircraft and time."""
    valid_flights = flights.dropna(subset=["firstSeen", "estDepartureAirport", "estArrivalAirport"])
    return (
        valid_flights[FLIGHT_KEY + ["estDepartureAirport", "estArrivalAirport"]]
        .sort_values(["icao24", "firstSeen"])
        .reset_index(drop=True)
    )


def match_file(path, flights_lookup, destination):
    """Airborne state vectors of one hourly raw file, linked to their flight, with a first label.

    Writes `destination` (if any row is matched) and returns a summary row.
    """
    hour_start, hour_end = file_time_range(path)
    candidates = flights_lookup[
        (flights_lookup["firstSeen"] <= hour_end) & (flights_lookup["lastSeen"] >= hour_start)
    ]
    candidate_aircraft = candidates["icao24"].unique()

    writer, n_read, n_matched = None, 0, 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_SIZE, columns=MATCHED_COLUMNS):
        df = batch.to_pandas()
        n_read += len(df)
        df = df[df["icao24"].isin(candidate_aircraft)].dropna()
        df = df[~df["onGround"].astype(bool)].drop(columns="onGround")

        df = df.merge(candidates, on="icao24")
        df = df[(df["time"] >= df["firstSeen"]) & (df["time"] <= df["lastSeen"])]
        if df.empty:
            continue
        df["remaining_time"] = df["lastSeen"] - df["time"]

        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(destination, table.schema)
        writer.write_table(table)
        n_matched += len(df)
    if writer is not None:
        writer.close()
    return {"hour": to_utc(hour_start), "candidate flights": len(candidates),
            "rows read": n_read, "rows matched": n_matched}


def match_state_vectors(flights_lookup, raw_dir=RAW_STATE_VECTORS_DIR, matched_dir=PROCESSED_DIR / "matched",
                        n_files=None):
    """Matching of every hourly raw file (notebook 02, section 2): writes `matched_dir`."""
    matched_dir.mkdir(parents=True, exist_ok=True)
    summary = [match_file(path, flights_lookup, matched_dir / path.name)
               for path in list_parquet_files(raw_dir, n_files)]
    return pd.DataFrame(summary).sort_values("hour")


# ----------------------------------------------------------------------------------------------
# Notebook 02, section 3: cleaning
# ----------------------------------------------------------------------------------------------
def clean_file(path, destination):
    """Removes the altitude outliers and the physically impossible values of one matched file.

    The rows are ordered by flight and time through an index (the table itself is never sorted,
    to keep the memory low). Writes `destination` and returns a summary row.
    """
    table = pq.read_table(
        path, columns=FLIGHT_KEY + ["time", "baroAltitude", "geoAltitude", "vertRate", "velocity"]
    )
    n_rows = table.num_rows

    # Order of the rows by flight, then time
    icao24_code = table["icao24"].combine_chunks().dictionary_encode().indices.to_numpy()
    first_seen = table["firstSeen"].to_numpy()
    last_seen = table["lastSeen"].to_numpy()
    time_s = table["time"].to_numpy().astype(np.float64)
    order = np.lexsort((time_s, last_seen, first_seen, icao24_code))
    icao24_code, first_seen, last_seen, time_s = (
        array[order] for array in (icao24_code, first_seen, last_seen, time_s)
    )
    flight_start = np.r_[True, (icao24_code[1:] != icao24_code[:-1])
                         | (first_seen[1:] != first_seen[:-1])
                         | (last_seen[1:] != last_seen[:-1])]
    flight_id = np.cumsum(flight_start)

    baro_ft = table["baroAltitude"].to_numpy()[order] * M_TO_FT
    geo_ft = table["geoAltitude"].to_numpy()[order] * M_TO_FT
    vz_ft_min = table["vertRate"].to_numpy()[order] * MS_TO_FT_MIN
    speed_kt = table["velocity"].to_numpy()[order] * MS_TO_KT
    del table

    # 4. Physically impossible values (checked first: they must not bias the next steps)
    is_impossible = (
        (speed_kt > MAX_SPEED_KT)
        | (np.abs(vz_ft_min) > MAX_VERTICAL_RATE_FT_MIN)
        | (np.minimum(baro_ft, geo_ft) < MIN_ALTITUDE_FT)
        | (np.maximum(baro_ft, geo_ft) > ALTITUDE_CEILING_FT)
    )

    # 1. Spikes and 2. inconsistencies
    is_spike = rolling_median_outliers(baro_ft, flight_id) | rolling_median_outliers(geo_ft, flight_id)
    is_inconsistent = np.abs(baro_ft - geo_ft) > MAX_BARO_GEO_GAP_FT

    # 3. Offset segments, detected on the remaining points
    kept = np.flatnonzero(~(is_spike | is_inconsistent | is_impossible))
    kept_start = np.r_[True, flight_id[kept][1:] != flight_id[kept][:-1]]
    in_segment = np.zeros(n_rows, dtype=bool)
    for altitude_ft in (baro_ft, geo_ft):
        segment = offset_segments(altitude_ft[kept], time_s[kept], vz_ft_min[kept], kept_start)
        in_segment[kept[segment]] = True

    is_outlier = is_spike | is_inconsistent | is_impossible | in_segment
    keep = np.empty(n_rows, dtype=bool)
    keep[order] = ~is_outlier
    write_filtered(path, destination, keep)

    return {"file": path.name[11:19], "rows": n_rows, "spikes": is_spike.sum(),
            "inconsistent": is_inconsistent.sum(), "impossible": is_impossible.sum(),
            "offset segments": in_segment.sum(), "removed": is_outlier.sum()}


def clean_state_vectors(matched_dir=PROCESSED_DIR / "matched", clean_dir=CLEAN_DIR, n_files=None):
    """Cleaning of every matched file (notebook 02, section 3): writes `clean_dir`."""
    clean_dir.mkdir(parents=True, exist_ok=True)
    return pd.DataFrame([clean_file(path, clean_dir / path.name)
                         for path in list_parquet_files(matched_dir, n_files)])


# ----------------------------------------------------------------------------------------------
# Notebook 02, section 4: selection of the commercial (IFR) flights
# ----------------------------------------------------------------------------------------------
def ifr_criteria(flight_stats):
    """For each flight, the selection criteria it fails (True = fails)."""
    return pd.DataFrame({
        f"max altitude < {MIN_MAX_ALTITUDE_FT:,} ft":
            flight_stats["max_altitude_ft"] < MIN_MAX_ALTITUDE_FT,
        f"cruise speed < {MIN_CRUISE_SPEED_KT} kt":
            flight_stats["cruise_speed_kt"] < MIN_CRUISE_SPEED_KT,
        f"observed duration < {MIN_DURATION_MIN} min":
            flight_stats["duration_min"] < MIN_DURATION_MIN,
        f"observed duration > {MAX_DURATION_H} h":
            flight_stats["duration_min"] > MAX_DURATION_H * 60,
    })


def write_ifr_flights(ifr_flights, clean_dir=CLEAN_DIR, ifr_dir=PROCESSED_DIR / "ifr", n_files=None):
    """Keeps the state vectors of the selected flights: writes `ifr_dir`, returns (rows before, rows after)."""
    ifr_dir.mkdir(parents=True, exist_ok=True)
    n_before = n_after = 0
    for path in list_parquet_files(clean_dir, n_files):
        writer = None
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_SIZE):
            df = batch.to_pandas()
            n_before += len(df)
            df = df.merge(ifr_flights, on=FLIGHT_KEY)
            if df.empty:
                continue
            n_after += len(df)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(ifr_dir / path.name, table.schema)
            writer.write_table(table)
        if writer is not None:
            writer.close()
    return n_before, n_after


# ----------------------------------------------------------------------------------------------
# Notebook 02, section 5: flight phases
# ----------------------------------------------------------------------------------------------
def label_phases(ifr_dir=PROCESSED_DIR / "ifr", phases_dir=PROCESSED_DIR / "phases",
                 cruise_info_path=CRUISE_INFO_PATH, n_files=None):
    """Cruise level, cruise window and phase of every state vector, in three passes.

    Writes `cruise_info_path` and `phases_dir`; returns (cruise info, phase counts per flight,
    number of flights).
    """
    phases_dir.mkdir(parents=True, exist_ok=True)
    ifr_files = list_parquet_files(ifr_dir, n_files)

    # Pass 1: cruise level = most populated 500 ft bin of level-flight altitudes
    bin_parts, max_parts = [], []
    for path in ifr_files:
        file_bins, file_max = [], []
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=BATCH_SIZE, columns=FLIGHT_KEY + ["baroAltitude", "vertRate"]
        ):
            df = batch.to_pandas()
            file_max.append(df.groupby(FLIGHT_KEY, sort=False)["baroAltitude"].max())
            level = df[(df["vertRate"] * MS_TO_FT_MIN).abs() < LEVEL_VZ_FT_MIN]
            altitude_bin = np.floor(
                (level["baroAltitude"] * M_TO_FT + ALTITUDE_BIN_FT / 2) / ALTITUDE_BIN_FT
            ).astype(int)
            file_bins.append(
                level.assign(altitude_bin=altitude_bin)
                .groupby(FLIGHT_KEY + ["altitude_bin"], sort=False).size()
            )
        bin_parts.append(pd.concat(file_bins).groupby(level=FLIGHT_KEY + ["altitude_bin"]).sum())
        max_parts.append(pd.concat(file_max).groupby(level=FLIGHT_KEY).max())

    bin_counts = pd.concat(bin_parts).groupby(level=FLIGHT_KEY + ["altitude_bin"]).sum()
    max_altitude_ft = pd.concat(max_parts).groupby(level=FLIGHT_KEY).max() * M_TO_FT

    cruise_info = (
        bin_counts.rename("count").reset_index()
        .sort_values("count", ascending=False)
        .drop_duplicates(FLIGHT_KEY)
        .set_index(FLIGHT_KEY)
    )
    cruise_info["cruise_altitude_ft"] = cruise_info["altitude_bin"] * ALTITUDE_BIN_FT
    cruise_info["cruise_lower_ft"] = cruise_info["cruise_altitude_ft"] - CRUISE_BAND_FT
    cruise_info["cruise_upper_ft"] = cruise_info["cruise_altitude_ft"] + CRUISE_BAND_FT
    cruise_info = cruise_info[["cruise_altitude_ft", "cruise_lower_ft", "cruise_upper_ft"]]

    # Pass 2: cruise window = first and last times inside the cruise band
    band = cruise_info[["cruise_lower_ft", "cruise_upper_ft"]].reset_index()
    window_parts = []
    for path in ifr_files:
        file_windows = []
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=BATCH_SIZE, columns=FLIGHT_KEY + ["time", "baroAltitude"]
        ):
            df = batch.to_pandas().merge(band, on=FLIGHT_KEY)
            altitude_ft = df["baroAltitude"] * M_TO_FT
            in_band = (altitude_ft >= df["cruise_lower_ft"]) & (altitude_ft <= df["cruise_upper_ft"])
            file_windows.append(df[in_band].groupby(FLIGHT_KEY, sort=False)["time"].agg(["min", "max"]))
        window_parts.append(pd.concat(file_windows).groupby(level=FLIGHT_KEY).agg({"min": "min", "max": "max"}))
    window = pd.concat(window_parts).groupby(level=FLIGHT_KEY).agg({"min": "min", "max": "max"})

    cruise_info["first_cruise_time"] = window["min"]
    cruise_info["last_cruise_time"] = window["max"]
    cruise_info["cruise_duration_s"] = cruise_info["last_cruise_time"] - cruise_info["first_cruise_time"]
    cruise_info["cruise_type"] = np.where(
        cruise_info["cruise_duration_s"] > SHORT_CRUISE_S, "cruise", "short cruise"
    )
    cruise_info["max_altitude_ft"] = max_altitude_ft
    cruise_info = cruise_info.reset_index()
    cruise_info.to_parquet(cruise_info_path, index=False)

    # Pass 3: label every state vector and write the phases files
    window_columns = FLIGHT_KEY + ["first_cruise_time", "last_cruise_time", "cruise_type"]
    phase_count_parts = []
    for path in ifr_files:
        writer = None
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_SIZE):
            df = batch.to_pandas().merge(cruise_info[window_columns], on=FLIGHT_KEY, how="left")
            vz_ft_min = df["vertRate"] * MS_TO_FT_MIN
            has_cruise = df["first_cruise_time"].notna()
            before = has_cruise & (df["time"] < df["first_cruise_time"])
            after = has_cruise & (df["time"] > df["last_cruise_time"])
            within = has_cruise & ~before & ~after

            phase = pd.Series(pd.NA, index=df.index, dtype="string")
            phase[within] = df.loc[within, "cruise_type"]
            phase[before & (vz_ft_min > LEVEL_VZ_FT_MIN)] = "climb"
            phase[before & (vz_ft_min <= LEVEL_VZ_FT_MIN)] = "est_climb"
            phase[after & (vz_ft_min < -LEVEL_VZ_FT_MIN)] = "descent"
            phase[after & (vz_ft_min >= -LEVEL_VZ_FT_MIN)] = "est_descent"
            df = df.drop(columns=window_columns[3:]).assign(phase_of_flight=phase)

            phase_count_parts.append(df.groupby(FLIGHT_KEY + ["phase_of_flight"], sort=False).size())
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(phases_dir / path.name, table.schema)
            writer.write_table(table)
        if writer is not None:
            writer.close()

    phase_counts = (
        pd.concat(phase_count_parts).groupby(level=FLIGHT_KEY + ["phase_of_flight"]).sum()
        .unstack(fill_value=0).reindex(columns=PHASES, fill_value=0)
    )
    return cruise_info, phase_counts, len(max_altitude_ft)


# ----------------------------------------------------------------------------------------------
# Notebook 02, section 6: sampling of 15 / 30 / 15 state vectors per flight
# ----------------------------------------------------------------------------------------------
def sample_ml_dataset(phases_dir=PROCESSED_DIR / "phases", cruise_info_path=CRUISE_INFO_PATH,
                      ml_dataset_path=ML_DATASET_PATH, n_files=None):
    """Selects the eligible flights and samples their state vectors by rank, in three passes.

    Writes `ml_dataset_path`; returns the dataset and the numbers displayed by notebook 02.
    """
    segment_code_of = {segment: code for code, segment in enumerate(SEGMENTS)}
    phase_to_code = {phase: segment_code_of[segment] for phase, segment in PHASE_TO_SEGMENT.items()}
    n_segments = len(SEGMENTS)
    rank_base = 10_000_000  # integer key = group * rank_base + rank

    ml_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    phase_files = sorted(list_parquet_files(phases_dir, n_files), key=file_time_range)

    # Pass A: points and time span of each segment, last airborne time of each flight
    segment_parts, arrival_parts = [], []
    for path in phase_files:
        file_segments, file_arrivals = [], []
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=BATCH_SIZE, columns=FLIGHT_KEY + ["time", "phase_of_flight"]
        ):
            df = batch.to_pandas()
            file_arrivals.append(df.groupby(FLIGHT_KEY, sort=False)["time"].max())
            df["segment"] = df["phase_of_flight"].map(PHASE_TO_SEGMENT)
            df = df.dropna(subset=["segment"])
            file_segments.append(
                df.groupby(FLIGHT_KEY + ["segment"], sort=False)["time"].agg(["min", "max", "size"])
            )
        arrival_parts.append(pd.concat(file_arrivals).groupby(level=FLIGHT_KEY).max())
        segment_parts.append(
            pd.concat(file_segments).groupby(level=FLIGHT_KEY + ["segment"])
            .agg({"min": "min", "max": "max", "size": "sum"})
        )

    last_airborne_time = (
        pd.concat(arrival_parts).groupby(level=FLIGHT_KEY).max().rename("last_airborne_time")
    )
    segment_stats = (
        pd.concat(segment_parts).groupby(level=FLIGHT_KEY + ["segment"])
        .agg({"min": "min", "max": "max", "size": "sum"})
    )
    segment_counts = (
        segment_stats["size"].unstack("segment", fill_value=0).reindex(columns=SEGMENTS, fill_value=0)
    )
    # Share of the seconds of a segment that actually have a state vector
    segment_coverage = (
        (segment_stats["size"] / (segment_stats["max"] - segment_stats["min"] + 1))
        .unstack("segment").reindex(columns=SEGMENTS)
    )

    # Eligibility
    enough_points = (segment_counts >= pd.Series(N_POINTS)).all(axis=1)
    flight_cruise = pd.read_parquet(cruise_info_path).set_index(FLIGHT_KEY).reindex(segment_counts.index)
    consistent_cruise = (
        flight_cruise["max_altitude_ft"] - flight_cruise["cruise_altitude_ft"] <= MAX_CRUISE_GAP_FT
    )
    eligible = enough_points & consistent_cruise

    flights = segment_counts.index[eligible].to_frame(index=False)
    flights["flight_id"] = np.arange(len(flights))
    n_groups = len(flights) * n_segments  # group = flight_id * n_segments + segment code

    # Target ranks of every (flight, segment) group
    group_counts = np.zeros(n_groups, dtype=np.int64)
    target_keys = []
    for segment, n_points in N_POINTS.items():
        counts = segment_counts.loc[eligible, segment].to_numpy()
        group = flights["flight_id"].to_numpy() * n_segments + segment_code_of[segment]
        group_counts[group] = counts
        ranks = np.floor((np.arange(n_points) + 0.5)[None, :] / n_points * counts[:, None])
        target_keys.append((group[:, None] * rank_base + ranks.astype(np.int64)).ravel())
    target_keys = np.sort(np.concatenate(target_keys))

    # Pass B: rank of every point in its segment (files in chronological order)
    points_seen = np.zeros(n_groups, dtype=np.int64)
    selected_parts = []
    for path in phase_files:
        file_parts = []
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=BATCH_SIZE, columns=FLIGHT_KEY + ["time", "phase_of_flight"]
        ):
            df = batch.to_pandas()
            df["segment_code"] = df["phase_of_flight"].map(phase_to_code)
            df = df.dropna(subset=["segment_code"]).merge(flights, on=FLIGHT_KEY)
            file_parts.append(pd.DataFrame({
                "group": df["flight_id"].to_numpy() * n_segments + df["segment_code"].to_numpy(dtype=np.int64),
                "time": df["time"].to_numpy(dtype=np.int64),
            }))
        file_points = pd.concat(file_parts, ignore_index=True).sort_values(["group", "time"])
        group = file_points["group"].to_numpy()
        rank = points_seen[group] + file_points.groupby("group").cumcount().to_numpy()
        points_seen += np.bincount(group, minlength=n_groups)
        selected_parts.append(file_points[np.isin(group * rank_base + rank, target_keys)])

    assert np.array_equal(points_seen, group_counts), "passes A and B did not see the same points"
    selected = pd.concat(selected_parts, ignore_index=True)
    selected = pd.DataFrame({"flight_id": selected["group"] // n_segments, "time": selected["time"]})

    # Pass C: extract the full rows of the selected state vectors
    flights = flights.merge(last_airborne_time.reset_index(), on=FLIGHT_KEY)
    extracted = []
    for path in phase_files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_SIZE):
            df = batch.to_pandas().merge(flights, on=FLIGHT_KEY)
            df["time"] = df["time"].astype(np.int64)
            df = df.merge(selected, on=["flight_id", "time"])
            if not df.empty:
                extracted.append(df)

    df_ml = (
        pd.concat(extracted, ignore_index=True)
        .drop(columns="flight_id")
        .rename(columns={"remaining_time": "remaining_time_lastseen"})
        .sort_values(FLIGHT_KEY + ["time"])
        .reset_index(drop=True)
    )
    df_ml["remaining_time"] = df_ml["last_airborne_time"] - df_ml["time"]
    df_ml.to_parquet(ml_dataset_path, index=False)

    report = {
        "eligibility": pd.Series({
            "flights with phases": len(segment_counts),
            "enough points in every segment": enough_points.sum(),
            "... and consistent cruise level": eligible.sum(),
        }, name="flights"),
        "share with coverage gaps": (segment_coverage[eligible] < 0.9).any(axis=1).mean(),
    }
    return df_ml, report


def check_ml_dataset(df_ml):
    """Consistency checks of the ML dataset (all must be 0, except the counts)."""
    per_segment = (
        df_ml.assign(segment=df_ml["phase_of_flight"].map(PHASE_TO_SEGMENT))
        .groupby(FLIGHT_KEY + ["segment"]).size().unstack(fill_value=0).reindex(columns=SEGMENTS)
    )
    return pd.Series({
        "flights": len(per_segment),
        "state vectors": len(df_ml),
        "flights with exactly 15 / 30 / 15 points": (per_segment == pd.Series(N_POINTS)).all(axis=1).sum(),
        "duplicated state vectors": df_ml.duplicated(FLIGHT_KEY + ["time"]).sum(),
        "missing values": df_ml.isna().sum().sum(),
        "negative labels": (df_ml["remaining_time"] < 0).sum(),
        "values out of physical bounds": (
            (df_ml["velocity"] * MS_TO_KT > MAX_SPEED_KT)
            | (df_ml["vertRate"].abs() * MS_TO_FT_MIN > MAX_VERTICAL_RATE_FT_MIN)
            | (df_ml[["baroAltitude", "geoAltitude"]].min(axis=1) * M_TO_FT < MIN_ALTITUDE_FT)
            | (df_ml[["baroAltitude", "geoAltitude"]].max(axis=1) * M_TO_FT > ALTITUDE_CEILING_FT)
        ).sum(),
        f"|baro - geo| > {MAX_BARO_GEO_GAP_FT:,} ft": (
            (df_ml["baroAltitude"] - df_ml["geoAltitude"]).abs() * M_TO_FT > MAX_BARO_GEO_GAP_FT
        ).sum(),
    }, name="value")


# ----------------------------------------------------------------------------------------------
# Notebook 03: airports, split and features
# ----------------------------------------------------------------------------------------------
def add_airports_and_drop_unknown(df, airport_lookup):
    """Adds the airport coordinates, drops the flights with an unknown airport, sets a missing
    arrival elevation to 0 ft. Returns the new table and the numbers displayed by notebook 03."""
    df = add_airport_coordinates(df, airport_lookup)
    missing = df["departure_lat"].isna() | df["arrival_lat"].isna()
    missing_codes = pd.concat([
        df.loc[df["departure_lat"].isna(), "estDepartureAirport"],
        df.loc[df["arrival_lat"].isna(), "estArrivalAirport"],
    ]).value_counts() // POINTS_PER_FLIGHT
    report = {"flights without coordinates": missing.sum() // POINTS_PER_FLIGHT,
              "share of rows without coordinates": missing.mean(), "unknown codes": missing_codes}

    df = df[~missing].reset_index(drop=True)
    report["share of rows without arrival elevation"] = df["arrival_elevation_ft"].isna().mean()
    df[["departure_elevation_ft", "arrival_elevation_ft"]] = (
        df[["departure_elevation_ft", "arrival_elevation_ft"]].fillna(0)
    )
    return df, report


def build_features(ml_dataset_path=ML_DATASET_PATH, features_path=FEATURES_PATH):
    """Notebook 03 without its explorations: writes `features_path` and returns the table."""
    df = pd.read_parquet(ml_dataset_path)
    df, _ = add_airports_and_drop_unknown(df, load_airport_lookup())
    df["split"] = split_by_flight(df)
    df = add_stateless_features(df)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    df[FEATURE_TABLE_COLUMNS].to_parquet(features_path, index=False)
    return df[FEATURE_TABLE_COLUMNS]


# ----------------------------------------------------------------------------------------------
# The whole chain
# ----------------------------------------------------------------------------------------------
def build_dataset():
    """Raw OpenSky files -> clean/ (kept) -> ml_dataset.parquet -> features.parquet.

    The other intermediate files are written to a temporary folder, deleted as soon as possible.
    """
    tmp_dir = PROCESSED_DIR / "tmp_build"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    start = time.time()

    def step(message):
        print(f"[{(time.time() - start) / 60:5.1f} min] {message}", flush=True)

    step("1/6 matching and cleaning the state vectors, hour file by hour file")
    flights_lookup = make_flights_lookup(load_flights())
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    removed = rows = 0
    for path in list_parquet_files(RAW_STATE_VECTORS_DIR):
        matched_path = tmp_dir / path.name
        match_file(path, flights_lookup, matched_path)
        summary = clean_file(matched_path, CLEAN_DIR / path.name)
        matched_path.unlink()
        removed, rows = removed + summary["removed"], rows + summary["rows"]
    step(f"    {rows:,} matched state vectors, {removed:,} removed by the cleaning ({removed / rows:.2%})")

    step("2/6 selecting the commercial (IFR) flights")
    flight_stats = compute_flight_stats(CLEAN_DIR)
    is_ifr = ~ifr_criteria(flight_stats).any(axis=1)
    n_before, n_after = write_ifr_flights(flight_stats.loc[is_ifr, FLIGHT_KEY], ifr_dir=tmp_dir / "ifr")
    step(f"    {is_ifr.sum():,} of {len(is_ifr):,} flights kept, {n_after:,} of {n_before:,} state vectors")

    step("3/6 flight phases")
    cruise_info, _, n_flights = label_phases(ifr_dir=tmp_dir / "ifr", phases_dir=tmp_dir / "phases")
    shutil.rmtree(tmp_dir / "ifr")
    step(f"    cruise identified for {len(cruise_info):,} of {n_flights:,} flights")

    step("4/6 sampling 15 / 30 / 15 state vectors per flight")
    df_ml, _ = sample_ml_dataset(phases_dir=tmp_dir / "phases")
    shutil.rmtree(tmp_dir)
    checks = check_ml_dataset(df_ml)
    step(f"    {checks['state vectors']:,} state vectors of {checks['flights']:,} flights -> "
         f"{ML_DATASET_PATH.relative_to(PROJECT_DIR)}")

    step("5/6 airports, split by flight and features")
    features = build_features()
    step(f"    {len(features):,} rows, {len(FEATURE_TABLE_COLUMNS)} columns -> {FEATURES_PATH.relative_to(PROJECT_DIR)}")
    step("6/6 done")


if __name__ == "__main__":
    build_dataset()
