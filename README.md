# Netflix (NFLX) Stock Price Forecasting

**Does an LSTM actually beat a naive "tomorrow = today" baseline at predicting Netflix's daily closing price?**

This project builds both, evaluates them honestly under walk-forward validation, and reports the
answer — including if the answer is unflattering to the deep learning model.

**Status:** Planning and scaffolding complete. Modeling not yet started.
See [`WALKTHROUGH.md`](WALKTHROUGH.md) for the step-by-step build log and [`PLAN.md`](PLAN.md)
for the full execution plan.

---

## Why this project is framed the way it is

Search GitHub for "LSTM stock prediction" and you get thousands of repositories that do the same
thing: feed 60 days of closing prices into an LSTM, predict day 61, plot a line that hugs the
actual price almost perfectly, report a small RMSE, and stop.

That result is an illusion, and the reason is worth stating precisely.

Daily equity prices behave close to a random walk. When you train a network to minimize squared
error on the *price level*, the best available answer is approximately "yesterday's price." The
model learns to copy its most recent input with a one-day lag. On a chart this looks like
near-perfect prediction. In reality it has learned nothing that a one-line baseline doesn't
already know.

ProjectPro's own write-up concedes the point — machine learning techniques "remain unreliable for
real-world market prediction." So rather than reproduce the illusion, this project measures it:

- Build the LSTM properly, using the same architecture the popular tutorial uses, so the
  comparison is fair.
- Build the baselines it is never compared against — naive persistence, drift, SMA, EMA, ARIMA.
- Evaluate everything on identical chronological splits with rolling-origin validation.
- Report **directional accuracy** (was the sign of the predicted move right?) alongside RMSE.
  A coin flip scores 50%.

If the LSTM does not win, that is the finding, and it gets written up as the finding.

---

## The four defects this project fixes

The most-referenced Kaggle notebook on this topic (Fares Sayah, 899k views) contains four
methodological problems that propagate into most derivative projects. Each becomes a section of
the final write-up:

| # | Defect | Fix |
|---|---|---|
| 1 | **Data leakage.** `scaler.fit_transform()` is called on the entire series *before* the train/test split, so the scaler sees future minimums and maximums. | Fit the scaler on training data only; transform the test set. Enforced by a unit test. |
| 2 | **No baseline.** An RMSE is reported with nothing to compare it against. | Naive, drift, SMA, EMA, and ARIMA baselines on identical splits. |
| 3 | **Price-space evaluation only.** RMSE on a trending price series rewards lag, not skill. | Add MAE, MAPE, directional accuracy, RMSE on returns, and prediction-vs-actual lag correlation. |
| 4 | **One arbitrary split.** `epochs=1`, `batch_size=1`, no validation set, no early stopping. | Chronological 70/15/15 split, early stopping on validation loss, walk-forward evaluation. |

The notebook also no longer runs as written: `yf.pdr_override()` has been removed from `yfinance`,
and `yfinance` now defaults to `auto_adjust=True`, so there is no `Adj Close` column unless you
ask for one. Both are handled in `src/data.py`.

---

## Results

Pending. Populated by `scripts/run_experiment.py` in the final session.

| Model | RMSE | MAE | MAPE | Directional accuracy |
|---|---|---|---|---|
| Naive (persistence) | — | — | — | — |
| Drift | — | — | — | — |
| SMA | — | — | — | — |
| EMA | — | — | — | — |
| ARIMA | — | — | — | — |
| LSTM | — | — | — | — |

Coin-flip directional accuracy is 50%. Anything in the 50–53% range is noise, not signal.

**Pre-registered expectation**, recorded before any model was run: the LSTM will land within a few
percent of the naive baseline on RMSE and near 50% on directional accuracy.

---

## Data

- **Ticker:** NFLX, daily bars from Yahoo Finance via `yfinance`
- **Peers** (for the correlation and risk sections): DIS, SPY, AMZN, WBD
- **Range:** 2015-01-01 to present — roughly ten years, deliberately spanning the 2022
  subscriber-loss crash and the streaming-wars regime shift
- Adjusted closes; business-day index, not reindexed to calendar days

Raw data is cached to `data/raw/` and gitignored. `src/data.py` regenerates it from scratch.

---

## Repository layout

```
netflix-stock-forecasting/
├── README.md               <- you are here
├── WALKTHROUGH.md          <- first-person, step-by-step build log
├── PLAN.md                 <- full execution plan, session by session
├── requirements.txt
├── data/
│   ├── raw/                <- cached yfinance pulls (gitignored)
│   └── processed/          <- feature matrices (gitignored)
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda_market_analysis.ipynb
│   ├── 03_stationarity_and_features.ipynb
│   ├── 04_baselines_and_arima.ipynb
│   ├── 05_lstm.ipynb
│   └── 06_results.ipynb
├── src/                    <- importable, tested code; notebooks call into this
│   ├── config.py           <- ticker, date range, window size, split ratios
│   ├── data.py             <- download + on-disk cache
│   ├── features.py         <- moving averages, returns, volatility, lags, calendar features
│   ├── windowing.py        <- leak-free sequence builder and scaler discipline
│   ├── evaluate.py         <- RMSE, MAE, MAPE, directional accuracy, walk-forward
│   ├── plots.py
│   └── models/
│       ├── baselines.py    <- naive, drift, SMA, EMA
│       ├── arima.py
│       └── lstm.py
├── scripts/run_experiment.py   <- one command reproduces every number above
├── reports/figures/            <- committed PNGs so this page renders on GitHub
├── tests/                      <- pytest; the leakage test is the important one
└── docs/
    ├── SETUP.md            <- machine setup: Homebrew, Python, venv, GitHub CLI
    └── sources.md          <- annotated bibliography
```

The `src/` package exists on purpose. Notebooks alone read as coursework; a tested, importable
module with a single reproducible entry point reads as engineering.

---

## Method

1. **Collect** NFLX and peer daily bars, cache locally, validate for gaps and duplicates.
2. **Explore** — closing price and volume history, 10/20/50-day moving averages, daily return
   distribution and its fat tails, peer correlation on returns *and* prices, risk-versus-return
   scatter.
3. **Establish stationarity.** Run the Augmented Dickey-Fuller test on raw closes (expected: fails
   to reject — non-stationary) and on log returns (expected: rejects — stationary). This contrast
   is the analytical core of the project: it is *why* the naive baseline is so hard to beat.
4. **Engineer features** — lags, rolling mean and standard deviation, realized volatility,
   cyclical day-of-week encoding. Read candidate ARIMA orders off the ACF/PACF plots.
5. **Baseline** — naive, drift, SMA, EMA, then ARIMA and auto-ARIMA.
6. **Model** — `LSTM(128) → LSTM(64) → Dense(25) → Dense(1)`, Adam and MSE, 60-day window, with
   train-only scaling, early stopping, and ablations on returns and multivariate inputs.
7. **Compare** every model on one table, and diagnose the LSTM's lag against the naive baseline
   directly.

---

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_experiment.py
```

Setting up a Mac from scratch — Homebrew, Python 3.12, the GitHub CLI, the virtual environment —
is covered step by step in [`docs/SETUP.md`](docs/SETUP.md).

---

## Limitations

- Single ticker. Nothing here is validated cross-sectionally.
- No transaction costs, slippage, market impact, or liquidity constraints are modeled, so
  directional accuracy is not the same as profitability.
- Daily bars only. Any genuine short-horizon signal is likely to live at higher frequency or in
  data that is not price history.
- No news, sentiment, or fundamental data — which is precisely why exogenous shocks such as the
  2022 subscriber-loss crash are unpredictable to every model here.
- Not investment advice. This is a methodology exercise.

---

## Sources

Four references, merged into one pipeline rather than treated as separate projects. Full
annotations in [`docs/sources.md`](docs/sources.md).

- ProjectPro — [Stock Price Prediction Using Machine Learning](https://www.projectpro.io/article/stock-price-prediction-using-machine-learning-project/571) — problem framing, SMA/EMA baselines, RMSE and MAPE, the limitations argument
- Interview Query — [16 Best Fintech Machine Learning Projects](https://www.interviewquery.com/p/fintech-machine-learning-projects), project #2 "Predicting Netflix Stock Prices" — the project specification
- Fares Sayah — [Stock Market Analysis + Prediction using LSTM](https://www.kaggle.com/code/faressayah/stock-market-analysis-prediction-using-lstm) — the EDA template and the LSTM recipe being tested
- andreshg — [TimeSeries Analysis: A Complete Guide](https://www.kaggle.com/code/andreshg/timeseries-analysis-a-complete-guide/notebook) — ADF, differencing, decomposition, ACF/PACF, ARIMA
