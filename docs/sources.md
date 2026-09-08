# Annotated sources

All four references were reviewed on 2026-09-08. Notes on what each one actually contains and
how it is used here live in `PLAN.md` §1; this file is the citable bibliography for the README.

## 1. ProjectPro — Stock Price Prediction Using Machine Learning
<https://www.projectpro.io/article/stock-price-prediction-using-machine-learning-project/571>

Article (no runnable repo). Uses Netflix data from MarketWatch, Mar 2019 – Mar 2022, with
OHLC + Volume. Covers SMA and EMA as statistical baselines, then single- and multi-layer LSTM
architectures (forget / input / output gates). Reports **RMSE** and **MAPE**.

Its most useful contribution is the limitations section: models "could not follow the trends
disrupted by the COVID-19 pandemic," and it is "nearly impossible to anticipate a piece of news
that will shatter or boost the stock market." We cite this directly in the README's Limitations
section.

## 2. Interview Query — 16 Best Fintech Machine Learning Projects, project #2
<https://www.interviewquery.com/p/fintech-machine-learning-projects>

Project #2, "Predicting Netflix Stock Prices," listed under *Beginner*.
Skills: time series analysis, sequence modeling. Tools: Python, LSTM/RNN (Keras).
Dataset: Netflix stock price data from Yahoo! Finance.
Its own "extra resources" list points at sources 3 and 4 below — which is why all four are
merged into a single pipeline rather than treated as separate projects.

## 3. Kaggle — Fares Sayah, "Stock Market Analysis + Prediction using LSTM"
<https://www.kaggle.com/code/faressayah/stock-market-analysis-prediction-using-lstm>

Apache-2.0. Structured around six questions: price change over time, average daily return,
moving averages, inter-stock correlation, value at risk, and predicting AAPL's close with an LSTM.

Modeling recipe: `Close` only → `MinMaxScaler(0,1)` → 60-day sliding window → 95/5 split →
`LSTM(128, return_sequences=True)` → `LSTM(64)` → `Dense(25)` → `Dense(1)`, Adam + MSE,
`batch_size=1`, `epochs=1`. Reports RMSE.

**Four defects we fix** (leakage in scaling, no baseline, price-space-only evaluation,
single arbitrary split) — see `PLAN.md` §1. It also uses `yf.pdr_override()`, which has been
removed from `yfinance`, so the notebook no longer runs as written.

## 4. Kaggle — andreshg, "TimeSeries Analysis: A Complete Guide"
<https://www.kaggle.com/code/andreshg/timeseries-analysis-a-complete-guide/notebook>

Table of contents: data visualization; preprocessing (chronological order and equidistant
timestamps, missing values, smoothing/resampling, stationarity via **Augmented Dickey-Fuller**,
transforming, differencing); feature engineering (cyclical encoding, decomposition, lags);
EDA (autocorrelation); modeling (Prophet, ARIMA, auto-ARIMA, LSTM, multivariate Prophet).

Demonstrated on a groundwater-depth dataset. We port the method to NFLX — in particular the
ADF-on-price vs. ADF-on-returns contrast, which is the analytical core of this project.
