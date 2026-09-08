"""Download and cache daily price bars from Yahoo Finance.

Replaces the `pandas_datareader` + `yf.pdr_override()` pattern used by the 2021
reference notebook. `pdr_override()` no longer exists in yfinance, so that code
cannot run at all today.

Two current yfinance behaviours this module pins down explicitly, because
relying on their defaults is exactly what breaks old notebooks:

1. `auto_adjust` now defaults to True, so there is no `Adj Close` column. We
   keep auto-adjustment on, so `Close` is already adjusted for splits
   and dividends, which is what we want for return calculations, and we say so
   rather than leaving it implicit.
2. `multi_level_index` now defaults to True, so even a single ticker comes back
   with MultiIndex columns. We flatten to plain columns.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from . import config


def _flatten(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Reduce a yfinance frame to plain OHLCV columns with a DatetimeIndex."""
    if isinstance(df.columns, pd.MultiIndex):
        # Columns look like ("Close", "NFLX"); drop the ticker level.
        levels = [df.columns.get_level_values(i) for i in range(df.columns.nlevels)]
        ticker_level = next(
            (i for i, lv in enumerate(levels) if ticker in set(lv)), None
        )
        if ticker_level is not None:
            df = df.xs(ticker, axis=1, level=ticker_level)
        else:
            df.columns = df.columns.get_level_values(0)

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[keep].sort_index()


def download_prices(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch daily bars for one ticker straight from Yahoo Finance."""
    start = start or config.START
    end = end if end is not None else config.END

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,       # Close is split/dividend adjusted
        multi_level_index=False,
        progress=False,
        actions=False,
    )
    if raw is None or raw.empty:
        raise ValueError(
            f"Yahoo Finance returned no rows for {ticker!r} "
            f"between {start} and {end or 'today'}."
        )
    return _flatten(raw, ticker)


def load_prices(
    ticker: str = config.TICKER,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return daily bars for `ticker`, using the on-disk cache when possible.

    The cache means repeated notebook runs do not hammer Yahoo Finance, and it
    keeps results stable while you are iterating. Pass `refresh=True` to force a
    fresh download.
    """
    config.ensure_dirs()
    path = config.DATA_RAW / f"{ticker}.csv"

    if path.exists() and not refresh:
        df = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
        return df.sort_index()

    df = download_prices(ticker, start=start, end=end)
    df.to_csv(path)
    return df


def load_many(
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load several tickers as {ticker: DataFrame}.

    An explicit dict, rather than the reference notebook's `globals()[stock] = ...`
    trick, which is fragile and invisible to linters and type checkers.
    """
    tickers = tickers or [config.TICKER, *config.PEERS]
    return {t: load_prices(t, start=start, end=end, refresh=refresh) for t in tickers}


def closing_prices(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One DataFrame of adjusted closes, one column per ticker, dates aligned."""
    return pd.DataFrame({t: df["Close"] for t, df in frames.items()}).sort_index()


def validate_prices(df: pd.DataFrame, name: str = "") -> dict:
    """Sanity-check a price frame and return a report.

    Deliberately returns findings instead of raising: gaps in a market calendar
    are normal (weekends, holidays) and need judgement, not an exception.
    """
    label = name or "series"
    close = df["Close"]

    gaps = df.index.to_series().diff().dt.days
    report = {
        "ticker": label,
        "rows": len(df),
        "start": df.index.min().date().isoformat(),
        "end": df.index.max().date().isoformat(),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_dates": int(df.index.duplicated().sum()),
        "non_positive_prices": int((close <= 0).sum()),
        "is_monotonic_dates": bool(df.index.is_monotonic_increasing),
        "largest_gap_days": int(gaps.max()) if len(gaps.dropna()) else 0,
    }
    report["ok"] = (
        report["missing_values"] == 0
        and report["duplicate_dates"] == 0
        and report["non_positive_prices"] == 0
        and report["is_monotonic_dates"]
    )
    return report


if __name__ == "__main__":
    frame = load_prices()
    print(frame.tail())
    print()
    for key, value in validate_prices(frame, config.TICKER).items():
        print(f"{key:>22}: {value}")
