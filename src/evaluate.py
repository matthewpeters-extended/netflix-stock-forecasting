"""Forecast evaluation.

The metric set is chosen so that a model cannot look good by accident.

RMSE on a trending price series is easy to make small: predict yesterday's price
and the error is just one day of noise. So RMSE is reported, because it is the
number every tutorial quotes, but it is never the number we judge by.

Directional accuracy is the honest test. A coin flip scores 50 percent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def directional_accuracy(
    actual: np.ndarray,
    predicted: np.ndarray,
    previous: np.ndarray,
    tol: float = 1e-4,
) -> float:
    """Percentage of days where the predicted direction of change is correct.

    `previous` is the last observed price before each prediction, which is what
    makes this a fair test: the model must commit to up or down relative to
    something it already knew.

    `tol` is a relative threshold. A predicted move smaller than `tol` times the
    previous price is treated as no position rather than as a directional call,
    which matters because a model can predict a change of 1e-12 and technically
    have a sign. Scoring those would report the direction of floating point
    noise. Returns NaN when the model takes no real position anywhere, which is
    the correct answer for the naive forecast: it predicts no change, so it
    never makes a directional claim to be right or wrong about.
    """
    actual, predicted, previous = map(np.asarray, (actual, predicted, previous))
    predicted_move = predicted - previous
    actual_move = actual - previous

    takes_position = np.abs(predicted_move) > tol * np.abs(previous)
    if not takes_position.any():
        return float("nan")

    correct = np.sign(predicted_move[takes_position]) == np.sign(actual_move[takes_position])
    return float(correct.mean() * 100)


def evaluate(
    name: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    previous: np.ndarray,
) -> dict:
    """Full metric row for one model on one test set."""
    actual, predicted, previous = map(np.asarray, (actual, predicted, previous))

    actual_returns = (actual - previous) / previous
    predicted_returns = (predicted - previous) / previous

    return {
        "model": name,
        "RMSE": round(rmse(actual, predicted), 4),
        "MAE": round(mae(actual, predicted), 4),
        "MAPE %": round(mape(actual, predicted), 3),
        "Directional %": round(directional_accuracy(actual, predicted, previous), 2),
        "RMSE on returns": round(rmse(actual_returns, predicted_returns), 6),
    }


def results_table(rows: list[dict]) -> pd.DataFrame:
    """Assemble metric rows into the project's single comparison table."""
    return pd.DataFrame(rows).set_index("model")


def chronological_split(
    n: int, train_frac: float = 0.70, val_frac: float = 0.15
) -> tuple[slice, slice, slice]:
    """Index slices for a time ordered train/validation/test split.

    Never shuffled. Shuffling a time series lets the model train on the future
    and test on the past, which produces excellent and entirely fictional scores.
    """
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


def lag_correlation(actual: np.ndarray, predicted: np.ndarray, max_lag: int = 5) -> pd.DataFrame:
    """Where does the prediction best line up with the actual series?

    A model that has genuinely learned something correlates best at lag 0. A
    model that has learned to copy its most recent input correlates best at
    lag 1, which is the signature of a dressed up naive forecast.
    """
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    rows = []
    for lag in range(0, max_lag + 1):
        if lag == 0:
            a, p = actual, predicted
        else:
            a, p = actual[lag:], predicted[:-lag]
        rows.append({"lag_days": lag, "correlation": round(float(np.corrcoef(a, p)[0, 1]), 6)})
    return pd.DataFrame(rows)
