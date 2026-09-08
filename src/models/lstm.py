"""LSTM sequence model.

The architecture is deliberately identical to the conventional implementation of
this project: two stacked LSTM layers of 128 and 64 units, then dense layers of
25 and 1, optimised with Adam against mean squared error.

Keeping the architecture fixed is what makes the comparison fair. If this model
fails to beat a naive forecast, nobody can say it was because the network was
built badly. What changes here is the *training and evaluation protocol*, not
the network:

- the scaler is fitted on training data only (see `src.windowing`)
- there is a real validation split, held out chronologically
- training runs to early stopping on validation loss rather than a fixed epoch
- batch size is 32 rather than 1, so training is both faster and less noisy
"""

from __future__ import annotations

import numpy as np


def set_seeds(seed: int = 7) -> None:
    """Make a run reproducible as far as the framework allows."""
    import random

    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def build(input_shape: tuple[int, int], units: tuple[int, int] = (128, 64)):
    """Build the two layer LSTM used throughout this project."""
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(units[0], return_sequences=True),
            layers.LSTM(units[1], return_sequences=False),
            layers.Dense(25),
            layers.Dense(1),
        ],
        name="lstm_forecaster",
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 10,
    verbose: int = 0,
):
    """Fit with early stopping on validation loss.

    `restore_best_weights=True` matters: without it the returned model is
    whatever the last epoch produced, which is typically several epochs past the
    best one.

    `shuffle=False` keeps batches in time order. Shuffling the windows themselves
    would not leak the future, since each window already carries its own history,
    but keeping the order makes the training curve interpretable.
    """
    from tensorflow import keras

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=verbose,
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        shuffle=False,
        verbose=verbose,
    )
    return history
