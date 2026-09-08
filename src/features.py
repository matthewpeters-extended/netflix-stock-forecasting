"""Feature engineering for daily price series.

Kept deliberately small and pure: every function takes a Series or DataFrame and
returns a new one. Nothing mutates its input, so notebook cells can be re-run in
any order without silently corrupting state — a real hazard in pandas 3, where
chained assignment on a slice no longer does what older tutorials assume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252  # NYSE sessions in a typical year, used to annualize


def daily_returns(close: pd.Series) -> pd.Series:
    """Simple percentage return: (P_t - P_{t-1}) / P_{t-1}."""
    return close.pct_change().rename("daily_return")


def log_returns(close: pd.Series) -> pd.Series:
    """Log return: ln(P_t / P_{t-1}).

    Preferred for modeling because log returns add across time, and because
    their distribution is better behaved than that of the price level.
    """
    return np.log(close / close.shift(1)).rename("log_return")


def add_moving_averages(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (10, 20, 50),
    column: str = "Close",
) -> pd.DataFrame:
    """Return a copy of `df` with an `MA_{n}` column per window."""
    out = df.copy()
    for window in windows:
        out[f"MA_{window}"] = out[column].rolling(window).mean()
    return out


def rolling_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """Annualized rolling standard deviation of returns.

    A 21-day window is roughly one trading month.
    """
    return (returns.rolling(window).std() * np.sqrt(TRADING_DAYS)).rename(
        f"volatility_{window}d"
    )


def return_summary(returns: pd.Series) -> dict:
    """Distribution statistics for a return series.

    `excess_kurtosis` is the number that matters here. A normal distribution has
    excess kurtosis of 0. Anything materially above that means fat tails: extreme
    days happen far more often than a normal distribution would allow.
    """
    clean = returns.dropna()
    return {
        "observations": len(clean),
        "mean_daily": clean.mean(),
        "std_daily": clean.std(),
        "annualized_return": clean.mean() * TRADING_DAYS,
        "annualized_volatility": clean.std() * np.sqrt(TRADING_DAYS),
        "skew": clean.skew(),
        "excess_kurtosis": clean.kurt(),
        "worst_day": clean.min(),
        "best_day": clean.max(),
    }


def tail_counts(returns: pd.Series, sigmas: tuple[float, ...] = (3.0, 4.0, 5.0)) -> pd.DataFrame:
    """Observed versus normal-implied counts of extreme days.

    The clearest single piece of evidence that daily returns are not normal,
    and therefore that any model assuming normality understates risk.
    """
    from scipy import stats

    clean = returns.dropna()
    n, sd = len(clean), clean.std()

    rows = []
    for k in sigmas:
        observed = int((clean.abs() > k * sd).sum())
        expected = 2 * stats.norm.sf(k) * n
        rows.append(
            {
                "threshold": f"|return| > {k:.0f} sigma",
                "observed_days": observed,
                "normal_expects": round(expected, 2),
                "times_more_common": round(observed / expected, 1) if expected else np.nan,
            }
        )
    return pd.DataFrame(rows)
