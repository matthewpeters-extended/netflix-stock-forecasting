"""Statistical tests for time series structure.

Three tests, each answering a different question:

- **ADF** — does the series have a unit root? Null hypothesis: it does
  (non-stationary). A small p-value means we reject that and call it stationary.
- **KPSS** — is the series stationary around a level? Null hypothesis: it *is*
  (the opposite of ADF). A small p-value means we reject stationarity.
- **Ljung-Box** — is there autocorrelation in the first k lags? Null hypothesis:
  none. A small p-value means there is.

Running ADF and KPSS together matters. They have opposite nulls, so agreement is
much stronger evidence than either alone, and disagreement is informative rather
than merely confusing.
"""

from __future__ import annotations

import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss


def adf_test(series: pd.Series, name: str = "", alpha: float = 0.05) -> dict:
    """Augmented Dickey-Fuller. Null: a unit root is present (non-stationary)."""
    clean = series.dropna()
    # result_object=False pins the current tuple return; statsmodels 0.16 changes
    # the default. Being explicit keeps this working across that upgrade.
    stat, pvalue, lags, nobs, crit, _ = adfuller(
        clean, autolag="AIC", result_object=False
    )
    return {
        "series": name or series.name or "series",
        "test": "ADF",
        "null_hypothesis": "has unit root (non-stationary)",
        "statistic": round(stat, 4),
        "p_value": round(pvalue, 6),
        "lags_used": lags,
        "crit_5pct": round(crit["5%"], 4),
        "reject_null": bool(pvalue < alpha),
        "conclusion": "stationary" if pvalue < alpha else "NON-stationary",
    }


def kpss_test(series: pd.Series, name: str = "", alpha: float = 0.05) -> dict:
    """KPSS. Null: the series IS stationary around a constant level."""
    import warnings

    clean = series.dropna()
    with warnings.catch_warnings():
        # statsmodels warns when the p-value is outside its lookup table; that
        # simply means the result is more extreme than the table's range.
        warnings.simplefilter("ignore")
        stat, pvalue, lags, crit = kpss(clean, regression="c", nlags="auto")
    return {
        "series": name or series.name or "series",
        "test": "KPSS",
        "null_hypothesis": "is stationary",
        "statistic": round(stat, 4),
        "p_value": round(pvalue, 6),
        "lags_used": lags,
        "crit_5pct": round(crit["5%"], 4),
        "reject_null": bool(pvalue < alpha),
        "conclusion": "NON-stationary" if pvalue < alpha else "stationary",
    }


def stationarity_report(series_map: dict[str, pd.Series]) -> pd.DataFrame:
    """Run both tests over several series and tabulate the verdicts."""
    rows = []
    for name, series in series_map.items():
        adf, kp = adf_test(series, name), kpss_test(series, name)
        rows.append(
            {
                "series": name,
                "ADF p": adf["p_value"],
                "ADF says": adf["conclusion"],
                "KPSS p": kp["p_value"],
                "KPSS says": kp["conclusion"],
                "agree": adf["conclusion"] == kp["conclusion"],
            }
        )
    return pd.DataFrame(rows)


def ljung_box(series: pd.Series, lags: tuple[int, ...] = (1, 5, 10, 21)) -> pd.DataFrame:
    """Ljung-Box test for autocorrelation up to each lag. Null: no autocorrelation."""
    clean = series.dropna()
    result = acorr_ljungbox(clean, lags=list(lags), return_df=True)
    result = result.rename(columns={"lb_stat": "statistic", "lb_pvalue": "p_value"})
    result["autocorrelated"] = result["p_value"] < 0.05
    result.index.name = "lag"
    return result.round(6)
