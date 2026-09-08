# Plan of Execution: Netflix (NFLX) Stock Price Forecasting

**Owner:** Matthew Peters
**Created:** 2026-09-08
**Status:** Complete. All six sessions delivered; see [`README.md`](README.md) for results and
[`WALKTHROUGH.md`](WALKTHROUGH.md) for the build account.
**Estimated effort:** 12 to 16 hours across 6 working sessions

> **Outcome.** No model beat the naive forecast. AIC independently selected ARIMA(0,1,0), which is
> the naive forecast. An apparent 55.15% directional edge (p = 0.018) proved to be seed variance:
> the five seed mean was 51.51% (p = 0.28). Fixing the scaler leak changed reported RMSE from 13.44
> to 29.97, revealing an extrapolation failure the leak had concealed.

---

## 0. What this project is

Interview Query's *Fintech ML Projects* list, project **#2, Predicting Netflix Stock Prices**:

> Learn the fundamentals of time series forecasting in finance by predicting stock prices
> using historical data and deep learning models.
> **Skills:** time series analysis, sequence modeling · **Tools:** Python, LSTM/RNN (Keras)
> **Dataset:** Netflix stock price data from Yahoo Finance

The three other links are the *method* references, not separate projects. They are merged into
one pipeline (see §1).

### The framing decision that makes this resume-worthy

Nearly every version of this project on GitHub does the same thing: feed 60 days of closing
prices into an LSTM, predict day 61, plot a line that hugs the actual price, report a small
RMSE, and declare success. **That result is an illusion.** An LSTM trained this way learns
to output approximately "yesterday's price," because on daily equity data that is very close
to the optimal least-squares answer. It looks brilliant on a price chart and is worthless.

So this project is deliberately built as an **honest evaluation** of that claim:

> *Does an LSTM on NFLX daily prices actually beat a naive "tomorrow = today" baseline?*

We build the LSTM properly, and we build the baselines, and we report both. If the LSTM
does not win, **we say so and explain why**. That is the finding, and it is a much stronger
interview story than a pretty chart. ProjectPro's own article concedes the point:
ML techniques "remain unreliable for real-world market prediction."

**Interview talking point:** *"I reproduced the standard LSTM stock-prediction project, then
showed its headline metric was an artifact of evaluating a near-random-walk series in price
space rather than return space. I re-ran it against naive and ARIMA baselines with
walk-forward validation and reported directional accuracy instead of RMSE."*

---

## 1. Source synthesis: what each link contributes

| Source | What we take from it | What we deliberately change |
|---|---|---|
| **ProjectPro**, *Stock Price Prediction Using ML* | Problem framing; SMA/EMA baselines; RMSE + MAPE as metrics; the explicit limitations section (COVID/news shocks break these models) | It stays at the article level, with no runnable repo. We supply the code. |
| **Interview Query #2**, *Predicting Netflix Stock Prices* | The ticker (**NFLX**), the Yahoo Finance data source, and the deliverable spec (LSTM/RNN in Keras) | none |
| **Kaggle**, *Stock Market Analysis + Prediction using LSTM* | The EDA template (closing price, volume, 10/20/50-day MAs, daily returns, return distribution, correlation heatmap, risk-vs-return scatter) and the LSTM recipe (60-day window, MinMaxScaler, `LSTM(128) → LSTM(64) → Dense(25) → Dense(1)`, Adam + MSE) | Four real defects, fixed in §4. See below. |
| **Kaggle**, *TimeSeries Analysis: A Complete Guide* | The rigorous time-series spine: missing/equidistant timestamps, resampling, **ADF stationarity test**, transforming & differencing, cyclical encoding, seasonal decomposition, lag features, **ACF/PACF**, ARIMA / auto-ARIMA / Prophet | Its notebook is on groundwater data, so we port the method to NFLX |

### The four defects we inherit from the Kaggle LSTM notebook and fix

These are the substance of the project. Each one becomes a section in the README.

1. **Data leakage in scaling.** The notebook calls `scaler.fit_transform(dataset)` on the
   *entire* series before splitting. The scaler therefore sees future max/min values.
   → *Fix:* `fit` on train only, `transform` test.
2. **No baseline.** RMSE is reported with nothing to compare it against.
   → *Fix:* naive/persistence, drift, SMA, EMA, and ARIMA baselines, all on identical splits.
3. **Evaluation in price space only.** RMSE on a trending price series rewards lag, not skill.
   → *Fix:* also report MAE, MAPE, **directional accuracy**, and metrics on *returns*.
4. **Single split, one epoch, `batch_size=1`.** No validation set, no early stopping, no
   walk-forward test, and a training config that is essentially arbitrary.
   → *Fix:* chronological train/val/test, early stopping, and rolling-origin (walk-forward)
   evaluation.

### Dead API calls in the 2021 notebook (expect to hit these)

The Kaggle notebook is 4 years old and will not run as written:

- `yf.pdr_override()` was **removed** from `yfinance`, and `pandas_datareader` is no longer needed.
  Use `yf.download(...)` / `yf.Ticker("NFLX").history(...)` directly.
- `yfinance` now defaults to `auto_adjust=True`, so there is **no `Adj Close` column** unless you
  pass `auto_adjust=False`. Decide once and document it (we use adjusted closes).
- `yf.download` on multiple tickers returns a **MultiIndex column frame**; the notebook's
  `globals()[stock] = ...` pattern is fragile, so we use an explicit `dict[str, DataFrame]`.

---

## 2. Environment: what we need before writing any code

Current machine state (checked 2026-09-08):

- macOS 26.6.2, Apple Silicon (`arm64`)
- **Only** system Python **3.9.6** at `/usr/bin/python3`, with pip 21.2.4
- No Homebrew, no conda, no pyenv, no uv, no Jupyter
- `pandas`, `numpy` **not installed**
- `git` present; **`gh` (GitHub CLI) not installed**

### Step 0. Set up the environment (do this first, ~20 min)

```bash
cd /Users/matthew/projects/netflix-stock-forecasting
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Then verify TensorFlow actually imported on Python 3.9 / arm64:

```bash
python -c "import tensorflow as tf, sys; print(sys.version.split()[0], tf.__version__)"
```

**If that fails:** Python 3.9 is at the edge of TensorFlow's support window. Install
Python 3.12 from the official installer at <https://www.python.org/downloads/macos/>
(a `.pkg`; it will ask for your Mac password, and no Homebrew is required), then rebuild the venv with
`/usr/local/bin/python3.12 -m venv .venv`. Everything else in `requirements.txt` is unpinned
enough to work on either.

**Fallback if TF stays broken:** swap the LSTM to PyTorch (`pip install torch`). The plan does
not otherwise depend on Keras. Decide by end of Session 1, and do not let it block the EDA work.

### Step 1. git + GitHub

```bash
git init && git add -A && git commit -m "Scaffold NFLX forecasting project"
```

`gh` is not installed, so create the repo through the GitHub web UI and then:

```bash
git remote add origin https://github.com/<your-username>/netflix-stock-forecasting.git
git branch -M main && git push -u origin main
```

**Push early and push often.** A repo with 15 commits telling a story reads far better to a
hiring manager than one commit called "initial commit."

---

## 3. Repository layout

```
netflix-stock-forecasting/
├── README.md               <- the deliverable recruiters actually read (see §6)
├── PLAN.md                 <- this file
├── requirements.txt
├── .gitignore
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
├── src/                    <- importable, testable code; notebooks call into this
│   ├── config.py           <- ticker, date range, window size, split ratios
│   ├── data.py             <- download + on-disk cache
│   ├── features.py         <- MAs, returns, volatility, lags, calendar features
│   ├── windowing.py        <- leak-free sequence builder + scaler discipline
│   ├── evaluate.py         <- RMSE, MAE, MAPE, directional accuracy, walk-forward
│   ├── plots.py
│   └── models/
│       ├── baselines.py    <- naive, drift, SMA, EMA
│       ├── arima.py
│       └── lstm.py
├── scripts/run_experiment.py   <- one command reproduces every number in the README
├── reports/figures/            <- PNGs committed, so the README renders on GitHub
├── tests/                      <- pytest; the leakage test is the important one
└── docs/sources.md             <- annotated bibliography
```

**Why `src/` and not just notebooks:** notebooks alone signal "student." A package with tests
and a single reproducible entry point signals "engineer." This is the cheapest possible
credibility upgrade and it is the reason for the split.

---

## 4. Execution phases

### Session 1. Environment + data (2h)
- [ ] Run Step 0 and Step 1 above; confirm TensorFlow imports
- [ ] `src/config.py`: `TICKER="NFLX"`, `START="2015-01-01"`, `END="today"`, `WINDOW=60`
- [ ] `src/data.py`: `load_prices(ticker)` → downloads via `yfinance`, caches to
      `data/raw/NFLX.csv`, returns a tidy DataFrame. Never re-hit the network if cached.
- [ ] `notebooks/01`: pull NFLX + peers (`DIS`, `SPY`, and optionally `AMZN`, `WBD`), since the peers
      power the correlation section
- [ ] Sanity checks: business-day continuity, no NaNs, no duplicate dates, no zero/negative prices
- [ ] Commit
- **Done when:** `python -c "from src.data import load_prices; print(load_prices('NFLX').tail())"` works

### Session 2. EDA (2.5h), porting the reference EDA template
- [ ] Closing price history + volume
- [ ] 10 / 20 / 50-day moving averages overlay
- [ ] Daily returns series + histogram; note the fat tails vs. a normal fit
- [ ] Correlation heatmap of NFLX vs. peers (returns **and** prices), explaining why the price
      correlation is the misleading one)
- [ ] Risk-vs-return scatter (mean return vs. std dev) across the peer set
- [ ] Annotate the 2022 subscriber-loss crash (~-35% in a day) as the concrete example of
      ProjectPro's "news shocks break these models" warning
- [ ] Save every figure to `reports/figures/`
- **Done when:** 6+ committed PNGs and a written paragraph per chart

### Session 3. Time-series rigor (2.5h), porting the time series guide
- [ ] Chronological order + equidistant timestamp check; explain why markets are closed weekends
      and why we do **not** reindex to calendar days
- [ ] **ADF test on raw close** → expect *fails to reject* (non-stationary)
- [ ] **ADF test on log returns** → expect *rejects* (stationary). This single contrast is the
      analytical core of the whole project, because it is *why* the naive baseline is so hard to beat
- [ ] Log transform + first differencing
- [ ] Seasonal decomposition (trend / seasonal / residual)
- [ ] **ACF / PACF** plots on returns → read off candidate ARIMA (p, d, q)
- [ ] `src/features.py`: lag features, rolling mean/std, realized volatility, day-of-week
      (cyclical sin/cos encoding)
- **Done when:** you can state, in one sentence with numbers, whether NFLX daily returns are
  autocorrelated

### Session 4. Baselines + classical models (2.5h)
- [ ] `src/models/baselines.py`: naive (persistence), drift, SMA(k), EMA(α)
- [ ] `src/evaluate.py`: RMSE, MAE, MAPE, **directional accuracy**, and a `walk_forward()`
      rolling-origin evaluator
- [ ] ARIMA via `statsmodels` using the (p,d,q) from ACF/PACF; then `pmdarima.auto_arima` and
      compare the chosen orders
- [ ] *(Optional, if time)* Prophet. ProjectPro and Interview Query both mention it; it is a
      good "I know when the interpretable tool is enough" note, but it is **not** required
- [ ] Lock the results table format now; every later model appends a row to it
- **Done when:** a committed results table with naive/drift/SMA/EMA/ARIMA rows

### Session 5. The LSTM (3h), the headline deliverable
- [ ] `src/windowing.py`: sliding-window builder. **Scaler fits on train only.**
- [ ] `tests/test_windowing.py`: assert no test-set statistic can influence a training row.
      This is the one test worth writing and it is worth mentioning in interviews.
- [ ] `src/models/lstm.py`: `LSTM(128, return_sequences=True) → LSTM(64) → Dense(25) → Dense(1)`,
      Adam + MSE, the same architecture as the reference implementation, so the comparison is fair
- [ ] Train properly: chronological 70/15/15 train/val/test, `batch_size=32`, up to 100 epochs,
      `EarlyStopping(patience=10, restore_best_weights=True)`
- [ ] Plot train vs. val loss
- [ ] Predict, inverse-transform, add to the results table
- [ ] **The key diagnostic:** overlay the LSTM prediction against the naive baseline on the same
      axes and compute the lag between prediction and actual. Show it visually.
- [ ] Ablation: LSTM on *returns* instead of prices; LSTM with multivariate inputs
      (volume + volatility + peers)
- **Done when:** the results table is complete and you can explain each row

### Session 6. Write-up + publish (2.5h)
- [ ] `scripts/run_experiment.py` reproduces every number end to end
- [ ] Write `README.md` in full (§6)
- [ ] Final figures at consistent size/DPI
- [ ] `pip freeze > requirements-lock.txt`
- [ ] Push, add repo topics (`machine-learning`, `time-series`, `lstm`, `fintech`, `python`)
- [ ] Add the résumé bullets (§7)

---

## 5. Metrics we report (and why)

| Metric | Why it is in the table |
|---|---|
| RMSE | Because every tutorial reports it. We include it *so we can show why it misleads* |
| MAE | Less outlier-sensitive than RMSE |
| MAPE | Scale-free; ProjectPro's second metric |
| **Directional accuracy** | % of days the sign of the predicted change is right. This is the one a trading desk cares about. **Coin flip = 50%.** Anything at 50–53% is noise |
| RMSE on returns | Strips out the trend that flatters price-space RMSE |
| Lag correlation | `argmax` of cross-correlation between prediction and actual. If it is 1 day, the model has learned persistence and nothing else |

**Pre-registered expectation:** the LSTM lands within a few percent of the naive baseline on
RMSE and near 50% directional accuracy. Write that prediction down *before* running the model,
then report what happened. Stating a hypothesis in advance and reporting an unflattering result
is exactly the behavior that reads as scientific maturity.

---

## 6. README plan: the part that gets you the interview

Recruiters and hiring managers read the README and maybe two figures. Structure:

1. **One-line summary** + the honest headline result, stated up front
2. **Screenshot**: the best single figure (prediction vs. actual vs. naive baseline)
3. **Results table**: every model, every metric, bolded winner
4. **The finding**: two or three paragraphs on why the LSTM's low RMSE is not skill; ADF evidence that
   returns are stationary and near-unpredictable; what would actually be needed to do better
   (alternative data, higher frequency, cross-sectional signals)
5. **What I built**: repo tour, why `src/` + tests exist
6. **Methodology**: data, splits, leakage prevention, walk-forward validation
7. **Reproduce it**: exact commands, verified by running them in a clean venv
8. **Limitations**: no transaction costs, no slippage, no survivorship-bias handling,
   single ticker, not investment advice
9. **Sources**: credit all four references explicitly

Rules: no emoji-stuffed headers, no "🚀 Awesome"; every claim backed by a number in the table;
figures committed as PNG so they render on GitHub; and **run your own reproduce commands in a
fresh venv before you push**. A broken quickstart is the fastest way to lose a reader.

---

## 7. Résumé bullets (final, with measured numbers)

> **Netflix Stock Forecasting: Time Series & Deep Learning** · Python, TensorFlow/Keras,
> statsmodels, scikit-learn, pandas ·
> github.com/matthewpeters-extended/netflix-stock-forecasting
>
> - Built an end to end forecasting pipeline over 2,936 sessions of NFLX daily data, benchmarking
>   LSTM sequence models against naive, moving average and ARIMA baselines on identical
>   chronological splits, packaged as a tested module with a single command reproduction script.
> - Identified and fixed data leakage in the standard approach, where a MinMaxScaler is fitted
>   before the train/test split; quantified its effect at a 2.2x reduction in reported RMSE (29.97
>   to 13.44) and added regression tests enforcing train only fitting.
> - Demonstrated that an apparent 55.15% directional accuracy (one sided binomial p = 0.018) was
>   seed variance rather than signal: across five random seeds the mean fell to 51.51% (p = 0.28),
>   beating the naive baseline in one run of five. Published the negative result.
> - Established non stationarity through agreeing ADF and KPSS tests, then showed returns are
>   serially uncorrelated (Ljung-Box p = 0.85 at lag 1) while absolute returns are strongly
>   autocorrelated past 60 lags, identifying volatility rather than direction as the tractable
>   forecasting target.

**Shorter two line version, if space is tight:**

> **Netflix Stock Forecasting** · Python, TensorFlow/Keras, statsmodels · Benchmarked LSTM models
> against naive and ARIMA baselines on 11 years of NFLX daily data. Found and fixed a data leak in
> the standard approach (2.2x RMSE effect, regression tested), and showed an apparent 55% directional
> edge was seed variance, not signal (five seed mean 51.5%, p = 0.28).

## 8. Open questions, as resolved

1. **Date range.** Settled on 2015 to present, 2,936 sessions. The 2022 crash and the streaming
   wars regime shift are both inside the sample and are called out explicitly rather than avoided.
2. **Prediction target.** Both, as planned. Next day close was the headline; predicting log returns
   instead became the fix for the extrapolation failure and is reported alongside.
3. **Prophet.** Cut. ARIMA and the LSTM answered the question, and AIC selecting (0,1,0) made a
   third forecasting family redundant.
4. **Repo name and visibility.** Public, at
   github.com/matthewpeters-extended/netflix-stock-forecasting.

### Added during the build, not in the original plan

- `src/diagnostics.py` for ADF, KPSS and Ljung-Box, since stationarity testing grew past what
  belonged in a notebook.
- The **seed sweep**, which was not planned and turned out to be the finding.
- The **deliberate leaky rerun**, to measure the bug rather than only fix it.

## 9. Scope guardrails

**In scope:** univariate + light multivariate daily forecasting, classical vs. deep comparison,
honest evaluation, clean packaging, a README.

**Out of scope (do not let these creep in):** intraday/tick data, backtesting a trading strategy
with P&L, transformers, reinforcement learning, sentiment/NLP on news, live deployment or a
dashboard. Each is a *separate* resume project. Finishing this one in six sessions beats
half-finishing something bigger.
