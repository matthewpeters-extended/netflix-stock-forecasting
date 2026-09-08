"""Sliding window construction for sequence models, with scaler discipline.

This module exists because of one specific bug that the conventional tutorial
implementation of this project contains, and that propagates into most copies of
it.

That code calls `scaler.fit_transform(dataset)` on the **entire** price series
and only afterwards splits into train and test. Fitting a MinMaxScaler computes
the minimum and maximum of whatever it is given, so the scaler has already seen
the highest and lowest prices of the test period before training begins. Every
training example is then scaled using information from the future.

The effect is subtle rather than catastrophic, which is what makes it dangerous:
the model does not obviously cheat, it simply operates in a normalised space it
could not have known at training time, and the reported error is optimistic by
an amount nobody measures.

Here the scaler is fitted on training data only. `tests/test_windowing.py`
asserts it, so the bug cannot quietly return.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass
class Sequences:
    """Windowed data ready for a sequence model, plus what is needed to score it."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: MinMaxScaler
    test_actual: np.ndarray      # unscaled true values for the test period
    test_previous: np.ndarray    # unscaled value on the day before each prediction
    test_dates: pd.DatetimeIndex

    def shapes(self) -> dict:
        return {
            "X_train": self.X_train.shape,
            "X_val": self.X_val.shape,
            "X_test": self.X_test.shape,
        }


def _windows_for(scaled: np.ndarray, positions: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) where X[i] is the `window` scaled values before positions[i].

    An empty `positions` array returns correctly shaped empty arrays rather than
    raising, so a caller can legitimately ask for a zero length split.
    """
    if len(positions) == 0:
        return np.empty((0, window, 1)), np.empty((0,))
    X = np.stack([scaled[p - window : p, 0] for p in positions])
    y = scaled[positions, 0]
    return X[..., np.newaxis], y


def build_sequences(
    series: pd.Series,
    train_end: int,
    val_end: int,
    window: int = 60,
) -> Sequences:
    """Window a price series into train/validation/test sets without leakage.

    The scaler is fitted on `series[:train_end]` and nothing else.

    Windows for validation and test are allowed to reach back across a split
    boundary into earlier real observations. That is not leakage: predicting the
    first test day using the sixty genuine trading days before it is exactly what
    a forecaster would do in production. Leakage would be the reverse, letting
    later data influence earlier fitting, which is what the scaler rule prevents.
    """
    values = series.values.astype(float).reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(values[:train_end])          # TRAIN ONLY. This is the whole point.
    scaled = scaler.transform(values)

    train_pos = np.arange(window, train_end)
    val_pos = np.arange(train_end, val_end)
    test_pos = np.arange(val_end, len(values))

    X_train, y_train = _windows_for(scaled, train_pos, window)
    X_val, y_val = _windows_for(scaled, val_pos, window)
    X_test, y_test = _windows_for(scaled, test_pos, window)

    return Sequences(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        scaler=scaler,
        test_actual=values[test_pos, 0],
        test_previous=values[test_pos - 1, 0],
        test_dates=series.index[test_pos],
    )


def inverse(scaler: MinMaxScaler, scaled_values: np.ndarray) -> np.ndarray:
    """Map scaled predictions back to prices."""
    return scaler.inverse_transform(np.asarray(scaled_values).reshape(-1, 1))[:, 0]
