"""The test that matters.

The conventional implementation of this project fits its scaler on the full
series before splitting, so future minima and maxima leak into training. These
tests assert that cannot happen here.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.windowing import build_sequences  # noqa: E402

WINDOW = 10


@pytest.fixture
def series() -> pd.Series:
    """A series whose test period contains values far outside the training range.

    If the scaler were fitted on everything, that extreme tail would compress the
    training data and the difference would be measurable.
    """
    dates = pd.bdate_range("2020-01-01", periods=200)
    values = np.linspace(10.0, 30.0, 200)
    values[170:] += 500.0          # a huge excursion, test period only
    return pd.Series(values, index=dates, name="Close")


def test_scaler_is_fitted_on_training_data_only(series):
    seq = build_sequences(series, train_end=140, val_end=170, window=WINDOW)
    train_values = series.values[:140]

    assert np.isclose(seq.scaler.data_min_[0], train_values.min())
    assert np.isclose(seq.scaler.data_max_[0], train_values.max())

    # The full series maximum must NOT be what the scaler learned.
    assert not np.isclose(seq.scaler.data_max_[0], series.values.max())


def test_training_windows_are_unchanged_by_future_data(series):
    """Truncating everything after the training set must not alter training inputs.

    This is the direct behavioural test for leakage: if future values can change
    a training example, the future is influencing the past.
    """
    full = build_sequences(series, train_end=140, val_end=170, window=WINDOW)
    truncated = build_sequences(series.iloc[:140], train_end=140, val_end=140, window=WINDOW)

    np.testing.assert_allclose(full.X_train, truncated.X_train)
    np.testing.assert_allclose(full.y_train, truncated.y_train)


def test_scaled_training_range_is_exactly_zero_to_one(series):
    seq = build_sequences(series, train_end=140, val_end=170, window=WINDOW)
    assert seq.y_train.min() >= 0.0
    assert seq.y_train.max() <= 1.0


def test_test_values_may_exceed_one(series):
    """A correct train only scaler produces out of range test values, and should.

    Values above 1.0 are evidence the scaler never saw the test period. The
    leaky version would clamp everything neatly into [0, 1], which looks tidier
    and is wrong.
    """
    seq = build_sequences(series, train_end=140, val_end=170, window=WINDOW)
    assert seq.y_test.max() > 1.0


def test_target_alignment(series):
    """y[i] must be the true value at the predicted position, and previous is the day before."""
    seq = build_sequences(series, train_end=140, val_end=170, window=WINDOW)
    np.testing.assert_allclose(seq.test_actual, series.values[170:])
    np.testing.assert_allclose(seq.test_previous, series.values[169:-1])


def test_no_nan_anywhere(series):
    seq = build_sequences(series, train_end=140, val_end=170, window=WINDOW)
    for name, array in [("X_train", seq.X_train), ("X_val", seq.X_val), ("X_test", seq.X_test)]:
        assert not np.isnan(array).any(), f"{name} contains NaN"
