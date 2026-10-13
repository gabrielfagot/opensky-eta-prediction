"""Evaluation metrics, chosen once in notebook 04 and used for every model.

Decision metric: the MAE. Guardrails: the MAE of each flight segment and the 95th
percentile of the absolute error. The median is the typical error. The errors by
remaining time are information only.
"""
import numpy as np
import pandas as pd

from opensky_eta.config import FLIGHT_KEY, KT_TO_KMH, LABEL, RANDOM_STATE, SEGMENTS

# Remaining-time groups (information only)
REMAINING_BINS_MIN = [0, 15, 30, 60, 120, 240, 480, 1440]
REMAINING_LABELS = ["< 15 min", "15-30 min", "30-60 min", "1-2 h", "2-4 h", "4-8 h", "> 8 h"]

# A label is suspicious when the implied average speed of the flight is below 0.6 or above
# 1.6 times the typical speed of the flights of the same route length (about the 1st and
# 99th percentiles of the ratio).
SUSPICIOUS_SPEED_RATIO = (0.6, 1.6)


def evaluate(y_true, y_pred, segment):
    """The evaluation metrics, in minutes (y_true and y_pred in seconds)."""
    absolute_error = np.abs(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)) / 60
    segment = np.asarray(segment)
    metrics = {
        "MAE (min)": absolute_error.mean(),
        "median AE (min)": np.median(absolute_error),
        "P95 AE (min)": np.percentile(absolute_error, 95),
    }
    for name in SEGMENTS:
        metrics[f"MAE {name} (min)"] = absolute_error[segment == name].mean()
    return metrics


def remaining_time_report(y_true, predictions):
    """Information only: MAE (min) and relative error (%) by remaining time, for several models.

    Relative error of a group = MAE / mean true remaining time of the group. (Averaging
    |error| / remaining time point by point would explode for points about to land.)
    """
    y_true = np.asarray(y_true, dtype=float)
    remaining = pd.cut(y_true / 60, bins=REMAINING_BINS_MIN, labels=REMAINING_LABELS,
                       include_lowest=True)
    in_group = {label: np.asarray(remaining == label) for label in REMAINING_LABELS}
    mae, relative = {}, {}
    for name, y_pred in predictions.items():
        absolute_error = np.abs(np.asarray(y_pred, dtype=float) - y_true)
        mae[name] = {label: absolute_error[rows].mean() / 60 for label, rows in in_group.items()}
        relative[name] = {label: 100 * absolute_error[rows].sum() / y_true[rows].sum()
                          for label, rows in in_group.items()}
    mae_table = pd.DataFrame(mae)
    mae_table.insert(0, "share of rows (%)", {label: 100 * rows.mean() for label, rows in in_group.items()})
    return mae_table, pd.DataFrame(relative)


def bootstrap_comparison(y_true, predictions, flight_id, reference, n_resamples=1000):
    """MAE of each model with a 95 % confidence interval, and its difference with a reference.

    Flights (not rows) are resampled with replacement, because the points of a flight are
    correlated. The comparison is *paired*: every model is evaluated on the same resampled
    flights, so the difficulty of each flight cancels out in the differences.
    """
    y_true = np.asarray(y_true, dtype=float)
    n_flights = flight_id.max() + 1
    points_per_flight = np.bincount(flight_id, minlength=n_flights)
    error_per_flight = {  # sum of the absolute errors of each flight, in minutes
        name: np.bincount(flight_id, weights=np.abs(np.asarray(pred, dtype=float) - y_true) / 60,
                          minlength=n_flights)
        for name, pred in predictions.items()
    }
    # counts[r, f] = number of times flight f is drawn in resample r
    rng = np.random.default_rng(RANDOM_STATE)
    draws = rng.integers(0, n_flights, size=(n_resamples, n_flights))
    counts = np.stack([np.bincount(draw, minlength=n_flights) for draw in draws])
    resampled_points = counts @ points_per_flight
    resampled_mae = {name: counts @ errors / resampled_points for name, errors in error_per_flight.items()}

    reference_mae = error_per_flight[reference].sum() / points_per_flight.sum()
    rows = {}
    for name, errors in error_per_flight.items():
        mae = errors.sum() / points_per_flight.sum()
        difference = resampled_mae[name] - resampled_mae[reference]
        rows[name] = {
            "MAE (min)": round(mae, 2),
            "MAE 95 % CI": f"[{np.percentile(resampled_mae[name], 2.5):.2f}, {np.percentile(resampled_mae[name], 97.5):.2f}]",
            "difference vs reference (min)": round(mae - reference_mae, 2),
            "difference 95 % CI": f"[{np.percentile(difference, 2.5):+.2f}, {np.percentile(difference, 97.5):+.2f}]",
        }
    return pd.DataFrame(rows).T


def flight_errors(frame, predictions=None):
    """One row per flight: route, remaining time at its first sampled point, implied average
    speed and, for each model, its MAE (min) and mean signed error (min) on the flight.

    The remaining time at the first sampled point (just after the take-off) is close to the
    duration of the flight, so route length / this time is the average speed that the label
    implies. A negative signed error means that the model predicts less time than the label.
    """
    frame = frame.reset_index(drop=True)
    flight = frame.groupby(FLIGHT_KEY, sort=False)
    flights = flight.agg(
        departure=("estDepartureAirport", "first"),
        arrival=("estArrivalAirport", "first"),
        route_length_km=("route_length_km", "first"),
        first_remaining_h=(LABEL, "max"),
    )
    flights["first_remaining_h"] /= 3600
    flights["implied_speed_kt"] = flights["route_length_km"] / KT_TO_KMH / flights["first_remaining_h"]
    for name, prediction in (predictions or {}).items():
        error_min = pd.Series((np.asarray(prediction, dtype=float) - frame[LABEL].to_numpy()) / 60)
        grouped = error_min.groupby([frame[key] for key in FLIGHT_KEY], sort=False)
        flights[f"{name}: MAE (min)"] = grouped.apply(lambda e: e.abs().mean())
        flights[f"{name}: mean signed error (min)"] = grouped.mean()
    return flights


def flag_suspicious_labels(flights, reference_flights, n_bins=10, speed_ratio=SUSPICIOUS_SPEED_RATIO):
    """Flights whose label looks impossible (outputs of `flight_errors`).

    - same departure and arrival airport: the distance to the arrival does not describe the flight;
    - implied average speed below / above `speed_ratio` times the typical speed of the flights of
      the same route length (median of `reference_flights`, the training flights, by deciles of
      route length): the label is far too long (e.g. several flights merged into one OpenSky
      record) or far too short (e.g. a wrong arrival airport).
    """
    def different_airports(table):
        return table["departure"] != table["arrival"]

    reference = reference_flights[different_airports(reference_flights)]
    edges = np.quantile(reference["route_length_km"], np.linspace(0, 1, n_bins + 1))
    bin_of = lambda lengths: np.clip(np.searchsorted(edges, lengths, side="right") - 1, 0, n_bins - 1)
    typical = reference.groupby(bin_of(reference["route_length_km"]))["implied_speed_kt"].median()

    ratio = flights["implied_speed_kt"] / typical.reindex(bin_of(flights["route_length_km"])).to_numpy()
    same_airport = ~different_airports(flights)
    flags = pd.DataFrame({
        "same departure and arrival airport": same_airport,
        f"implied speed < {speed_ratio[0]} x typical": ~same_airport & (ratio < speed_ratio[0]),
        f"implied speed > {speed_ratio[1]} x typical": ~same_airport & (ratio > speed_ratio[1]),
    })
    flags["suspicious label"] = flags.any(axis=1)
    return flags, ratio.rename("speed / typical speed")


def error_concentration(flight_mae, shares=(0.01, 0.05, 0.10)):
    """Share of the total error made on the worst flights, and the whole cumulative curve.

    Every flight has the same number of rows, so the total error is proportional to the sum
    of the MAE of the flights.
    """
    ordered = np.sort(np.asarray(flight_mae, dtype=float))[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    summary = pd.Series({f"worst {share:.0%} of the flights": cumulative[int(np.ceil(share * len(ordered))) - 1]
                         for share in shares})
    return summary, cumulative


def suspicious_label_report(flags, flight_mae):
    """For each flag of `flag_suspicious_labels`: the flights concerned and, for each model
    (`flight_mae`: {model: MAE of each flight}), their share of its 1 % worst flights and of its total error.
    """
    rows = {}
    for flag, flagged in flags.items():
        row = {"flights": int(flagged.sum()), "share of the flights": flagged.mean()}
        for name, mae in flight_mae.items():
            worst = mae.nlargest(int(np.ceil(0.01 * len(mae)))).index
            row[f"{name}: share of its worst 1 %"] = flagged.loc[worst].mean()
            row[f"{name}: share of its total error"] = mae[flagged].sum() / mae.sum()
        rows[flag] = row
    return pd.DataFrame(rows).T
