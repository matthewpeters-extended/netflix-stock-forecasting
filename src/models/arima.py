"""ARIMA fitting and rolling one step ahead forecasting.

Two things this module is careful about.

First, order selection happens on training data by AIC, not on the test set.

Second, the test forecast is genuinely one step ahead and rolling. After each
prediction the true observation is appended to the model's history before the
next prediction is made, so the model always knows everything up to yesterday
and nothing about tomorrow. Forecasting the whole test period in one call
instead would let errors compound and would not answer the question we are
asking.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def fit_grid(
    train: pd.Series,
    orders: list[tuple[int, int, int]],
) -> pd.DataFrame:
    """Fit each candidate order on the training data and rank by AIC."""
    rows = []
    values = pd.Series(np.asarray(train, dtype=float))  # plain index, no date frequency warnings

    for order in orders:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ARIMA(values, order=order).fit()
            rows.append(
                {
                    "order": str(order),
                    "AIC": round(res.aic, 2),
                    "BIC": round(res.bic, 2),
                    "converged": bool(res.mle_retvals.get("converged", True))
                    if hasattr(res, "mle_retvals")
                    else True,
                }
            )
        except Exception as exc:  # a non invertible order is a result, not a crash
            rows.append({"order": str(order), "AIC": np.nan, "BIC": np.nan,
                         "converged": False, "error": type(exc).__name__})

    return pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)


def rolling_forecast(
    series: pd.Series,
    order: tuple[int, int, int],
    n_test: int,
) -> np.ndarray:
    """One step ahead predictions for the final `n_test` observations.

    The model is fitted once on everything before the test period. Each true
    observation is then appended without refitting, which is standard practice:
    it keeps the parameters fixed while the state updates, and it is what makes
    441 sequential forecasts tractable.
    """
    values = pd.Series(np.asarray(series, dtype=float))
    split = len(values) - n_test

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(values.iloc[:split], order=order).fit()

        predictions = np.empty(n_test)
        for i in range(n_test):
            predictions[i] = res.forecast(steps=1).iloc[0]
            res = res.append(values.iloc[split + i : split + i + 1], refit=False)

    return predictions
