"""Baseline forecasters.

Every one of these is a handful of lines. That is the point: if a neural network
cannot beat them, the neural network has not earned its complexity.

All follow the same contract. Given the full price series and the test index
positions, return a one step ahead prediction for each test date, using only
information available on the day before.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def naive(close: pd.Series, test_idx: np.ndarray) -> np.ndarray:
    """Tomorrow equals today.

    Under a random walk this is the theoretically optimal forecast, which is why
    it is the bar every other model has to clear.
    """
    return close.values[test_idx - 1]


def drift(close: pd.Series, test_idx: np.ndarray, train_end: int) -> np.ndarray:
    """Naive plus the average daily change measured on the training set only."""
    train = close.values[:train_end]
    daily_drift = (train[-1] - train[0]) / (len(train) - 1)
    return close.values[test_idx - 1] + daily_drift


def sma(close: pd.Series, test_idx: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average of the previous `window` closes."""
    rolled = close.rolling(window).mean().shift(1)
    return rolled.values[test_idx]


def ema(close: pd.Series, test_idx: np.ndarray, span: int = 5) -> np.ndarray:
    """Exponentially weighted moving average, weighting recent days more."""
    rolled = close.ewm(span=span, adjust=False).mean().shift(1)
    return rolled.values[test_idx]


def tune_window(
    close: pd.Series,
    val_idx: np.ndarray,
    fn,
    candidates: tuple[int, ...] = (2, 3, 5, 10, 20, 50),
) -> tuple[int, pd.DataFrame]:
    """Pick the window that minimises validation RMSE.

    Tuned on validation, never on test. This is the discipline that stops a
    baseline from being quietly fitted to the data it is later scored on.
    """
    from ..evaluate import rmse

    actual = close.values[val_idx]
    rows = []
    for c in candidates:
        pred = fn(close, val_idx, c)
        ok = ~np.isnan(pred)
        rows.append({"parameter": c, "val_RMSE": round(rmse(actual[ok], pred[ok]), 4)})
    table = pd.DataFrame(rows)
    best = int(table.loc[table["val_RMSE"].idxmin(), "parameter"])
    return best, table
