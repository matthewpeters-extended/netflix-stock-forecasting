# Walkthrough — How I Built This

A first-person, step-by-step account of the project. Part 1 covers the work that is done.
Part 2 covers the steps still ahead, written the same way, so anyone reading this repo can follow
along or reproduce it.

**Where things stand:** Part 1 is complete. Part 2 has not been run yet — every step in it is
marked as pending, and nothing in this document reports a result I have not actually produced.

---

# Part 1 — What I have done so far

## Step 1: I picked a project, then questioned the framing

I started from Interview Query's list of fintech machine learning projects and chose project #2,
"Predicting Netflix Stock Prices." The specification is short: predict NFLX prices from historical
Yahoo Finance data using an LSTM or RNN in Keras.

Before writing any code I went looking at how other people had done it, and that changed what I
wanted to build. The most-referenced implementation is a Kaggle notebook by Fares Sayah with
roughly 900,000 views, and its structure is repeated across thousands of GitHub repositories: take
60 days of closing prices, scale them, push them through a two-layer LSTM, predict the next day,
plot it, report RMSE, done. The chart always looks fantastic — the predicted line sits almost on
top of the actual price.

I did not trust it, for a specific reason. Daily equity prices behave close to a random walk. If I
train a network to minimize squared error on the *price level*, the lowest-error answer available
to it is roughly "output yesterday's price." A model that has learned to copy its last input with a
one-day lag will produce exactly that beautiful chart and exactly that small RMSE, while knowing
nothing useful.

So I decided the project would not be "build an LSTM that predicts Netflix." It would be:

> Does the LSTM actually beat a naive "tomorrow = today" baseline — and if not, can I show
> precisely why it only looks like it does?

That reframing is the whole reason this repo is worth reading. It also means I have to be willing
to publish a negative result, so I wrote down my expectation in advance (README, "pre-registered
expectation") before running anything.

## Step 2: I read all four sources and worked out what each one was actually for

I had four links and initially assumed they were four alternative tutorials. They are not. Read
together, they decompose cleanly into one pipeline, and I wrote up the details in
[`docs/sources.md`](docs/sources.md).

- **ProjectPro's article** gave me the problem framing, the simple statistical baselines (SMA and
  EMA), and the metrics to report (RMSE and MAPE). Its most useful section is the one on
  limitations — it states plainly that these models could not follow the trends disrupted by
  COVID, and that ML techniques remain unreliable for real market prediction. That is the argument
  my project is designed to test rather than repeat.
- **Interview Query** gave me the specification: NFLX, Yahoo Finance, Keras. Notably, its own
  "extra resources" section links to the other two Kaggle notebooks, which confirmed my read that
  these are one project's worth of material and not four.
- **Fares Sayah's notebook** gave me two things: an exploratory-analysis template built around six
  questions (price over time, daily returns, moving averages, correlation between stocks, value at
  risk, and prediction), and the exact LSTM recipe I intend to put on trial.
- **andreshg's time series guide** gave me the rigor the other three are missing — stationarity
  testing with the Augmented Dickey-Fuller test, transforming and differencing, seasonal
  decomposition, ACF and PACF, ARIMA and auto-ARIMA. It demonstrates all of this on groundwater
  data, so my job is to port the method to NFLX.

The Kaggle pages do not render their notebook bodies in plain HTML, so I read the executed
notebooks directly from the result frames rather than relying on the summary pages. That is worth
mentioning because it is how I found the next two things.

## Step 3: I catalogued the defects in the tutorial I am reproducing

Reading the actual notebook source rather than a description of it turned up four methodological
problems. These are now the backbone of the project, because fixing each one is a concrete,
explainable piece of work:

1. **Data leakage.** The notebook calls `scaler.fit_transform(dataset)` on the entire series and
   only afterwards splits into train and test. The scaler has therefore already seen the future
   maximum and minimum of the series. The fix is to fit on training data only. I am going to write
   a unit test that enforces it, because "I found a leak" is a claim, and "here is the test that
   fails if the leak comes back" is evidence.
2. **No baseline at all.** An RMSE is reported in isolation. There is nothing to say whether it is
   good.
3. **Evaluation in price space only.** RMSE on a trending price series systematically rewards a
   lagging prediction. I will add MAE, MAPE, directional accuracy, RMSE computed on returns, and a
   lag-correlation diagnostic between prediction and actual.
4. **A single arbitrary split.** `epochs=1`, `batch_size=1`, no validation set, no early stopping,
   no walk-forward testing.

## Step 4: I found that the reference notebook no longer runs

It was published four years ago and two APIs have moved underneath it:

- It calls `yf.pdr_override()`, which has since been **removed** from `yfinance`. The
  `pandas_datareader` bridge is not needed at all any more; `yf.download()` and
  `yf.Ticker(...).history()` work directly.
- `yfinance` now defaults to `auto_adjust=True`, so there is **no `Adj Close` column** unless you
  explicitly ask for unadjusted data. Every line in the notebook that references `Adj Close` fails.

There is also a fragility issue: the notebook assigns downloaded frames into `globals()` in a loop,
and multi-ticker downloads now come back as a MultiIndex column frame. I am using an explicit
dictionary of ticker to DataFrame instead.

I am recording all of this because "modernized a deprecated data pipeline" is a real piece of work
and it is invisible unless I write it down.

## Step 5: I checked what my machine can actually run

Before planning sessions I checked the environment, and it is bare:

- macOS 26.6.2 on Apple Silicon (`arm64`)
- System Python **3.9.6** only, with pip 21.2.4 — no Homebrew, no conda, no pyenv, no uv
- `pandas` and `numpy` are not installed; there is no Jupyter
- `git` is present; the GitHub CLI (`gh`) is not

Two consequences I planned around. First, Python 3.9 sits at the edge of TensorFlow's support
window, so the environment step includes an explicit import check and two documented fallbacks:
pin TensorFlow to `>=2.16,<2.18`, or install Python 3.12 from python.org, which needs no Homebrew.
Second, without `gh` I will create the GitHub repository through the web interface and add the
remote by hand.

I would rather discover this now than halfway through training a model.

## Step 6: I scaffolded the repository

I laid out the structure in [`PLAN.md`](PLAN.md) §3 and created it: `src/` for importable and
tested code, `notebooks/` for the six-notebook narrative, `data/raw` and `data/processed`
gitignored, `reports/figures/` for committed PNGs so the README renders on GitHub, `tests/`,
`scripts/run_experiment.py` as the single reproduction entry point, and `docs/sources.md`.

The `src/` package is deliberate. Notebooks on their own read as coursework. A tested module that
the notebooks import from, plus one command that regenerates every number in the README, reads as
engineering — and it is the cheapest credibility upgrade available to me.

I also wrote the full six-session plan, the metric definitions, the scope guardrails (no intraday
data, no P&L backtest, no transformers, no sentiment analysis — each of those is a *different*
project), and draft résumé bullets with the numbers left blank.

---

# Part 2 — The steps ahead

Not yet run. Each step lists what I will do and what "done" means, so progress is checkable rather
than vibes-based.

## Step 7: Set up the environment — pending

```bash
cd ~/projects/netflix-stock-forecasting
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python -c "import tensorflow as tf, sys; print(sys.version.split()[0], tf.__version__)"
```

The full machine setup — Homebrew, Python 3.12, the GitHub CLI, git identity — is written up in
[`docs/SETUP.md`](docs/SETUP.md), because my machine started with nothing but Apple's system
Python 3.9.

That last line is the real test. If TensorFlow will not import, I pin it or swap the LSTM to
PyTorch. I have given myself until the end of the first session to decide, so it cannot block the
exploratory work.

**Done when:** the version check prints without error, and `git init` plus a first commit is pushed
to a public GitHub repository.

## Step 8: Collect and validate the data — pending

I will write `src/config.py` (ticker, date range, 60-day window, split ratios) and `src/data.py`,
which downloads through `yfinance`, caches to `data/raw/NFLX.csv`, and never re-hits the network if
the cache exists. Then I pull NFLX plus DIS, SPY, AMZN and WBD for the correlation work, and check
for business-day continuity, duplicate dates, NaNs, and non-positive prices.

I am using 2015 to present — about ten years — deliberately spanning the 2022 subscriber-loss crash
and the streaming-wars regime shift, and I will note that regime break explicitly rather than
pretend the series is homogeneous.

**Done when:** `from src.data import load_prices; load_prices("NFLX").tail()` works from a clean
shell.

## Step 9: Exploratory analysis — pending

Porting the Fares Sayah template to NFLX: closing price and volume history; 10, 20 and 50-day
moving averages overlaid; the daily return series and its histogram, where I want to point out the
fat tails against a normal fit; a correlation heatmap across the peer set on returns *and* on
prices, explaining why the price correlation is the misleading one; and the risk-versus-return
scatter of mean return against standard deviation.

I will annotate the 2022 crash — Netflix lost roughly a third of its value in a day on a subscriber
miss — as the concrete instance of ProjectPro's warning that news shocks break these models. No
model in this repo sees news, so no model in this repo could have seen that coming.

**Done when:** at least six figures are saved to `reports/figures/` and committed, each with a
written paragraph of interpretation.

## Step 10: Establish stationarity — pending

This is the analytical core, ported from the andreshg guide. I run the Augmented Dickey-Fuller test
twice:

- On raw closing prices, where I expect it to **fail to reject** the null — the series is
  non-stationary, it has a unit root, it wanders.
- On log returns, where I expect it to **reject** — returns are stationary.

That contrast is the mathematical statement of why the naive baseline is so hard to beat, and it is
the thing I want to be able to explain out loud in an interview. Then log transform and first
differencing, seasonal decomposition into trend, seasonal and residual components, and ACF and PACF
plots on the returns to read off candidate ARIMA orders. Finally `src/features.py`: lags, rolling
mean and standard deviation, realized volatility, and cyclical sine/cosine day-of-week encoding.

**Done when:** I can state in one sentence, with actual numbers, whether NFLX daily returns are
autocorrelated.

## Step 11: Build the baselines — pending

`src/models/baselines.py` gets naive persistence, drift, SMA and EMA. `src/evaluate.py` gets RMSE,
MAE, MAPE, directional accuracy, and a rolling-origin `walk_forward()` evaluator. Then ARIMA
through `statsmodels` using the orders I read off the ACF/PACF, followed by `auto_arima` so I can
compare what the automatic search picks against what I chose by eye.

I am locking the results-table format at this step. Every later model appends a row to the same
table on the same splits — that is what makes the comparison honest.

**Done when:** a committed results table with naive, drift, SMA, EMA and ARIMA rows filled in.

## Step 12: Build the LSTM — pending

`src/windowing.py` builds the sliding windows, and the scaler fits on training data only.
`tests/test_windowing.py` asserts that no test-set statistic can influence a training row — the one
test in this repo that genuinely matters.

Then the same architecture as the notebook I am testing, so the comparison is fair:
`LSTM(128, return_sequences=True) → LSTM(64) → Dense(25) → Dense(1)`, Adam, MSE. Trained properly
this time: chronological 70/15/15 split, batch size 32, up to 100 epochs with early stopping on
validation loss and best weights restored. I will plot training against validation loss.

The diagnostic I most want is this: overlay the LSTM prediction, the actual price, and the naive
baseline on one set of axes, then compute the cross-correlation between prediction and actual and
find where it peaks. If the peak sits at one day, the model has learned persistence and nothing
else, and I will be able to show that rather than assert it.

Then two ablations — the LSTM trained on returns instead of prices, and a multivariate version with
volume, volatility and peer prices as extra inputs.

**Done when:** the results table is complete and I can explain every row in it.

## Step 13: Write it up and publish — pending

`scripts/run_experiment.py` regenerates every number end to end. I fill in the README's headline
result and results table, write the two or three paragraphs explaining the finding, standardize the
figures, freeze the dependency versions, run my own quickstart commands in a *fresh* virtual
environment to prove they work, and push with repository topics set.

Then I finalize the résumé bullets in [`PLAN.md`](PLAN.md) §7 with the real numbers.

**Done when:** a stranger can clone the repository, run three commands, and reproduce the table.

---

## What I want this project to demonstrate

Not that I can call `model.fit()`. That is the easy part and everyone has it.

What I want it to show is that I read the reference implementation closely enough to find a data
leak in it, that I know why a low RMSE on a price series is a weak claim, that I know how to test
whether a series is stationary and what follows from the answer, that I compare against baselines
before believing a model, and that I will publish a result that does not flatter my model if that
is what the evidence says.
