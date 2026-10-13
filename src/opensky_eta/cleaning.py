"""Flight statistics and altitude cleaning (notebook 02, sections 2 to 4)."""
import numpy as np
import pandas as pd

from opensky_eta.config import (
    FLIGHT_KEY, M_TO_FT, MAX_APPARENT_VZ_FT_MIN, MAX_DEVIATION_FT, MAX_JUMP_MISMATCH,
    MAX_SEGMENT_DURATION_S, MIN_JUMP_FT, MS_TO_KT, ROLLING_WINDOW, SEGMENT_RESIDUAL_FT,
)
from opensky_eta.storage import list_parquet_files


def compute_flight_stats(directory, n_files=None):
    """One row per flight: max altitude, cruise speed, observed duration, points.

    The cruise speed is the max velocity observed at the flight's max
    (barometric) altitude. A flight can span several hourly files, so
    per-file aggregates are merged at the end.
    """
    parts = []
    for path in list_parquet_files(directory, n_files):
        df = pd.read_parquet(
            path, columns=FLIGHT_KEY + ["time", "baroAltitude", "velocity"]
        )
        grouped = df.groupby(FLIGHT_KEY, sort=False)
        at_max_altitude = df["baroAltitude"] == grouped["baroAltitude"].transform("max")
        stats = grouped.agg(
            max_altitude=("baroAltitude", "max"),
            first_time=("time", "min"),
            last_time=("time", "max"),
            n_points=("time", "size"),
        )
        stats["cruise_speed"] = (
            df[at_max_altitude].groupby(FLIGHT_KEY, sort=False)["velocity"].max()
        )
        parts.append(stats.reset_index())

    stats = pd.concat(parts, ignore_index=True)
    global_max = stats.groupby(FLIGHT_KEY)["max_altitude"].transform("max")
    flight_stats = stats.groupby(FLIGHT_KEY).agg(
        max_altitude=("max_altitude", "max"),
        first_time=("first_time", "min"),
        last_time=("last_time", "max"),
        n_points=("n_points", "sum"),
    )
    flight_stats["cruise_speed"] = (
        stats[stats["max_altitude"] == global_max]
        .groupby(FLIGHT_KEY)["cruise_speed"].max()
    )
    flight_stats["max_altitude_ft"] = flight_stats["max_altitude"] * M_TO_FT
    flight_stats["cruise_speed_kt"] = flight_stats["cruise_speed"] * MS_TO_KT
    flight_stats["duration_min"] = (
        flight_stats["last_time"] - flight_stats["first_time"]
    ) / 60
    return flight_stats.reset_index()


def rolling_median_outliers(altitude_ft, flight_id):
    """Points farther than MAX_DEVIATION_FT from the rolling median of their flight.

    The arrays must be sorted by flight, then time.
    """
    median = (
        pd.Series(altitude_ft).groupby(flight_id)
        .rolling(ROLLING_WINDOW, center=True, min_periods=1).median()
        .to_numpy()
    )
    return np.abs(altitude_ft - median) > MAX_DEVIATION_FT


def offset_segments(altitude_ft, time_s, vz_ft_min, flight_start):
    """Points inside short offset segments (arrays sorted by flight, then time)."""
    step_ft = np.diff(altitude_ft, prepend=np.nan)
    step_ft[flight_start] = np.nan
    apparent_vz = step_ft / np.diff(time_s, prepend=np.nan) * 60

    # Altitude change expected from the reported vertical rate since the last
    # altitude update: a catch-up after a stale value is not a jump.
    changed = flight_start | (step_ft != 0)
    last_update = pd.Series(np.where(changed, time_s, np.nan)).ffill().to_numpy()
    expected_step_ft = vz_ft_min * (time_s - np.r_[np.nan, last_update[:-1]]) / 60

    is_jump = (
        (np.abs(step_ft) > MIN_JUMP_FT)
        & (np.abs(apparent_vz) > MAX_APPARENT_VZ_FT_MIN)
        & (np.abs(step_ft - expected_step_ft) > MIN_JUMP_FT)
    )
    jumps = np.flatnonzero(is_jump)
    flight_id = np.cumsum(flight_start)

    in_segment = np.zeros(len(altitude_ft), dtype=bool)
    for start, end in zip(jumps[:-1], jumps[1:]):
        if (
            flight_id[start] != flight_id[end]
            or end <= start + 1
            or np.sign(step_ft[start]) == np.sign(step_ft[end])
            or abs(step_ft[start] + step_ft[end]) > MAX_JUMP_MISMATCH * abs(step_ft[start])
            or time_s[end] - time_s[start] > MAX_SEGMENT_DURATION_S
        ):
            continue
        # Straight line between the last point before and the first point after
        before = start - 1
        slope = (altitude_ft[end] - altitude_ft[before]) / (time_s[end] - time_s[before])
        expected_ft = altitude_ft[before] + slope * (time_s[start:end] - time_s[before])
        if np.max(np.abs(altitude_ft[start:end] - expected_ft)) > SEGMENT_RESIDUAL_FT:
            in_segment[start:end] = True
    return in_segment
