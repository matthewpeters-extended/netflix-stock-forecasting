# Walkthrough: How I Built This

A step by step account of the project, in the order I actually did it, with the reasoning behind
each design decision. [`README.md`](README.md) covers what I found; this page covers how the thing
is put together and why.

---

## Step 1: I chose the problem, then changed the question

The brief was short: predict Netflix stock prices from Yahoo Finance history using an LSTM in
Keras.

Before writing code I looked at how this problem is usually solved, and that changed what I wanted
to build. The standard implementation is consistent wherever it appears: take 60 days of closing
prices, scale them into [0, 1], push them through two stacked LSTM layers, predict the next day,
plot the result, report RMSE, stop. The chart always looks superb, with the predicted line sitting
almost exactly on the actual price.

I did not trust it, for a specific reason. Daily equity prices behave close to a random walk. If I
train a network to minimise squared error on the *price level*, the lowest error answer available
is approximately "output yesterday's price". A model that has learned to copy its most recent input
with a one day lag produces exactly that beautiful chart and exactly that small RMSE, while having
learned nothing usable.

So I reframed the project around a question that has a falsifiable answer:

> Does the LSTM actually beat a naive "tomorrow equals today" forecast, and if not, can I show
> precisely why it only appears to?

That reframing is what the repository is organised around. It also commits me to publishing an
unflattering result if that is what the evidence says, so I wrote my expectation down in advance
before running anything.

---

## Step 2: I designed the architecture before writing the first module

Two decisions here shaped everything after.

**Logic lives in `src/`, not in notebooks.** Notebooks are excellent for narrative and terrible for
reuse: state is invisible, cells run out of order, and nothing can be tested. So every function
lives in an importable package and the notebooks call into it. The same code path runs in the
notebooks, in `pytest`, and in `scripts/run_experiment.py`, which means the story the notebooks
tell cannot drift away from the numbers the script produces.

**Every tunable lives in `src/config.py`.** Ticker, date range, peer list, window size, split
fractions, random seed. If a number affects a result, it is in that file and nowhere else. This
sounds fussy on a project this size and pays for itself the moment you want to know whether a
result depends on a choice you made three weeks earlier.

The layout that fell out of those two decisions:

```
src/
├── config.py       <- every tunable
├── data.py         <- acquisition, caching, validation
├── features.py     <- returns, moving averages, volatility, lags, calendar encoding
├── windowing.py    <- sequence construction and scaler discipline
├── diagnostics.py  <- ADF, KPSS, Ljung-Box
├── evaluate.py     <- metrics, splits, lag correlation
├── plots.py        <- shared figure style and saving
└── models/
    ├── baselines.py
    ├── arima.py
    └── lstm.py
```

The module boundary I care about most is `windowing.py` being separate from `models/lstm.py`. Data
preparation is where the leakage bug lives, so isolating it means it can be tested on its own,
without training anything.

---

## Step 3: I set up the machine, and hit the first real constraint

My Mac had Xcode Command Line Tools and `git`, and nothing else relevant. System Python 3.9.6 with
pip 21, no Homebrew, no conda, no pandas, no Jupyter.

Python 3.9 sits at the edge of TensorFlow's support window, so rather than discover that halfway
through I installed Homebrew, then Python 3.12, and built the project a virtual environment of its
own. Then I verified TensorFlow imported *before* writing any modelling code, with two documented
fallbacks ready if it had not. It imported cleanly, so neither was needed.

I also checked what was actually installed rather than assuming, which turned out to matter:
pandas 3.0 and numpy 2.5 are both recent majors. pandas 3.0 makes copy on write the default, so the
`frame['new_col'] = values` pattern on a slice, which older tutorials use freely, now errors or
silently does nothing rather than warning. Knowing that in advance saved a confusing debugging
session later.

The whole process is written up in [`docs/SETUP.md`](docs/SETUP.md).

---

## Step 4: I built the data layer, and found the reference code no longer runs

`src/data.py` downloads through `yfinance`, caches to `data/raw/`, and validates.

Two current library behaviours are pinned down explicitly rather than left to defaults, because
relying on those defaults is what breaks older code:

- **`auto_adjust` now defaults to `True`**, so there is no `Adj Close` column. Every line in the
  older approach that references it fails. I keep auto adjustment on, because `Close` is then
  already corrected for splits and dividends, which is what return calculations need.
- **`multi_level_index` now defaults to `True`**, so even a single ticker returns MultiIndex
  columns. I flatten explicitly.

I also replaced the `globals()[ticker] = frame` loop that older code uses with a plain
`dict[str, DataFrame]`. Assigning into `globals()` is invisible to linters, type checkers and
readers, and it breaks the moment the download shape changes.

Separately, `yf.pdr_override()` has been removed from `yfinance` entirely, so the
`pandas_datareader` bridge that older implementations depend on cannot work at all any more.
`yf.download()` is called directly instead.

**Validation returns findings rather than raising.** Gaps in a market calendar are normal, since
markets close at weekends and holidays, so an exception would be the wrong response. The function
reports row counts, missing values, duplicate dates, monotonicity and the largest gap, and lets the
caller judge. I deliberately do **not** reindex to a continuous calendar: filling weekends with
interpolated prices would invent trading that never happened and would damp the volatility I later
measure.

One thing I did not anticipate: if you run the pipeline while the US market is open, the final row
is a **partial session**, with a close that is just the last trade so far and a volume a fraction
of a full day. Training on it would teach the model from an incomplete bar. The notebook detects
this by comparing the last bar's volume against the median and drops it, printing what it did
rather than doing it silently.

---

## Step 5: I did the exploratory analysis, and one result reframed the risk discussion

Standard market EDA: price and volume history, 10/20/50 day moving averages, daily return series
and distribution, peer correlation, risk versus return.

Two things I did differently.

**I plotted the return distribution twice**, once on a linear axis and once on a log axis. On
linear axes a fitted normal looks fine. On log axes the observed data extends far past where the
fitted curve has gone to zero. Then I counted the extreme days: 5 sigma moves occurred 10 times
where a normal distribution predicts 0.002. Putting a number on it turns "returns have fat tails",
which everyone says, into something specific enough to act on.

**I plotted correlation of returns and correlation of price levels side by side.** The price level
version reports high correlations between series that need not move together at all, because both
trend and correlation picks up the shared trend. That is spurious correlation between
non-stationary series, and showing the size of the distortion is more convincing than asserting it.
It also sets up the stationarity work directly.

I also flagged a data quality issue rather than working around it quietly: all five tickers report
history back to 2015, but Warner Bros. Discovery did not exist until 2022, and Yahoo backfills the
ticker with its predecessor. I quantified how much the correlation changes if you use only the
period since the company existed, and left both numbers visible.

---

## Step 6: I established stationarity, which is where the project's argument comes from

This is the analytical core, and everything downstream depends on it.

**I ran two tests with opposite null hypotheses.** ADF assumes a unit root and asks you to reject
it; KPSS assumes stationarity and asks you to reject that. Running only one leaves you with a
result that could be a power problem. Running both means agreement is strong evidence and
disagreement is informative. They agreed on all three series: prices and log prices non-stationary,
log returns stationary.

That result is the argument the whole project rests on. A unit root means the series is a random
walk with drift:

$$P_t = P_{t-1} + \mu + \varepsilon_t$$

If that is the process, the best possible forecast of tomorrow given everything knowable today is
$P_t + \mu$, which is the naive baseline. Not *a* sensible baseline, the theoretically optimal one.
So any model claiming to beat it is claiming the random walk does not hold, and that needs real out
of sample evidence rather than a chart.

**Then I tested autocorrelation twice**, on returns and on absolute returns. Returns showed
essentially none (Ljung-Box p = 0.849 at lag 1). Absolute returns showed strong, persistent
structure (p = 0.000 at every lag, significant past sixty). Direction is unpredictable, magnitude
is not. That contrast is the single most useful thing the analysis produced, and I would not have
found it by testing returns alone.

**I ran seasonal decomposition and reported that it found nothing.** The seasonal component came to
2.89% of the observed standard deviation. Time series guides demonstrate decomposition on data with
genuine calendar seasonality; a liquid large cap equity has none, because a repeatable monthly price
cycle would be arbitraged away. Reporting a null result is a finding, not a skipped step.

**Then I built the feature matrix**, with one rule: every feature uses only information available
strictly before the day being predicted. Rolling windows and lags look backwards, the target is
shifted forwards. I asserted the alignment in the notebook rather than trusting it, including a
check that a rolling statistic at time *t* is unchanged when all rows after *t* are deleted.

---

## Step 7: I built the baselines before touching a neural network

This ordering is deliberate. Building baselines *after* your main model invites you to accept
whatever they show, because by then you have an answer you like.

Four baselines, each a few lines: naive persistence, drift, SMA, EMA. Plus ARIMA, with the order
chosen by AIC over a grid rather than assumed.

Three design decisions worth naming:

**I tuned the SMA and EMA windows on validation, never on test.** Choosing a parameter by test
performance is fitting to the data you are about to be scored on. The tuning result then became a
finding in its own right: validation error rose monotonically with window length, so the shortest
window won, and its limit is the naive forecast again.

**ARIMA forecasts are rolling and one step ahead.** After each prediction the true observation is
appended to the model's history before the next prediction is made, so the model always knows
everything up to yesterday and nothing about tomorrow. Forecasting the whole test period in a
single call would let errors compound and would not be comparable to the baselines.

**Directional accuracy returns NaN when a model takes no position.** The naive forecast predicts no
change, so it never makes a directional claim, and scoring it would be inventing a number. I also
added a relative tolerance, because a model can predict a change of 1e-12 and technically have a
sign; without the tolerance you end up reporting the direction of floating point noise. ARIMA(0,1,0)
did exactly that in my first run and produced a meaningless 25%.

AIC selected **ARIMA(0,1,0)**, which is the random walk, which is the naive forecast. Classical
model selection, given a real menu of alternatives, concluded there was nothing to model.

---

## Step 8: I built the LSTM, and isolated the leakage bug in a tested module

The architecture is deliberately identical to the conventional implementation:
`LSTM(128) -> LSTM(64) -> Dense(25) -> Dense(1)`, Adam, mean squared error, 60 day window. Keeping
it fixed means nobody can attribute the outcome to a badly built network. What I changed is the
protocol, not the model:

| Conventional | Here | Why |
|---|---|---|
| Scaler fitted on the full series | Fitted on training data only | The scaler otherwise knows the test period's minimum and maximum before training |
| No validation set | Chronological 70/15/15 | Nothing to early stop on, and no way to tune without touching test |
| `epochs=1` | Up to 100 with early stopping | One epoch is an arbitrary stopping point |
| `batch_size=1` | 32 | Faster, and less gradient noise |
| Single split | Splits fixed, plus a seed sweep | One run is one draw from a distribution |

**The leakage fix lives in `src/windowing.py` and is enforced by `tests/test_windowing.py`.** Six
tests, but two carry the weight:

- The scaler's learned minimum and maximum must equal the *training* minimum and maximum, and must
  not equal the full series maximum.
- Truncating the series to the training set must leave every training window byte identical. If
  future values can change a training example, the future is influencing the past.

There is a third test that looks strange and is the most diagnostic of the set: it asserts that
scaled test values are **allowed to exceed 1.0**. A correct train only scaler produces out of range
test values whenever the test period moves outside the training range. The leaky version compresses
everything neatly into [0, 1], which looks tidier and is precisely the bug. Tidy output is the
symptom.

Writing the fixture also mattered: the synthetic series jumps by 500 in its test region, so if the
scaler were fitted on everything the difference would be large and unmistakable rather than subtle.

---

## Step 9: I ran the experiments in an order that could prove me wrong

Four runs, deliberately sequenced.

**First, the honest version.** Train only scaling, price levels. It failed badly: RMSE about 30
against a baseline of 2.13, under-predicting by an average of $28 a day. The diagnosis was visible
in one number I had printed earlier, the scaled test target reaching 2.0. NFLX topped out near $69
in training and reached $134 in test, so the network was being asked to extrapolate outside
everything it had seen, in a space bounded to [0, 1]. That is a real limitation of the standard
framing, not a coding error.

**Second, the leaky version**, run deliberately, with the same architecture, seed and splits and
one line changed. RMSE fell from 30 to 13. The leak roughly halves the reported error. It does not
manufacture a good result here, but it hides the extrapolation failure behind a number that looks
mediocre rather than catastrophic, and on a series with a smaller regime shift that would be enough
to turn a failure into an apparent success.

**Third, the reframed version.** The extrapolation problem has a principled fix that the
stationarity work had already pointed at: predict log returns, which are stationary and centred
near zero, then convert back with $\hat{P}_{t+1} = P_t e^{\hat{r}_{t+1}}$. Same architecture,
different target. RMSE came level with the baseline and directional accuracy hit 55.15%, p = 0.018.

That is where I could have stopped, and it is where the conventional implementation does stop.

**Fourth, the seed sweep.** A network's initial weights are random, so training once gives one draw
from a distribution, and reporting it as the result is the same error as reporting one coin toss as
evidence of bias. Five seeds gave a mean of 51.51%, p = 0.28, beating the baseline in one run out of
five. The 55% was the lucky draw.

I designed the sequence so that step four could overturn step three. It did.

---

## Step 10: I made it reproducible

`scripts/run_experiment.py` regenerates every number in the README from a clean checkout. It takes
`--seeds` so a reader can set it to 1 and watch the apparent directional edge reappear, which is
the project's whole point made executable, and `--skip-lstm` for a three second run of the
baselines and ARIMA alone.

The notebooks import from `src/`, the script imports from `src/`, and the tests import from `src/`.
There is one implementation of everything.

Figures are committed as PNGs so the README renders on GitHub without anyone running a cell.
Cached price data is gitignored, because `src/data.py` regenerates it and committing data you can
download is noise.

---

## What I would do differently

**I would have run the seed sweep earlier.** I built it as a robustness check at the end and it
turned out to be the finding. Anything with random initialisation should be run several times
before you believe a single number, and I now think of that as part of the first evaluation rather
than a final audit.

**I would separate the leakage demonstration from the leakage fix sooner.** I wrote the correct
version first and reconstructed the buggy one afterwards to measure it. Building both from the
start would have made the comparison cleaner and cost nothing.

**The volatility result deserves its own project.** Absolute returns being strongly autocorrelated
out past sixty lags is the most actionable thing in the analysis, and a GARCH model on this exact
data is a tractable problem with a real answer. I kept it out of scope to finish this one, which I
think was right, but it is the obvious next build.

---

## What this project is meant to demonstrate

Not that I can call `model.fit()`. That part is four lines and everyone has it.

That I read a reference implementation closely enough to find a data leak in it, and then wrote the
test that stops the leak coming back. That I know why a low RMSE on a trending series is a weak
claim, and what to measure instead. That I test for stationarity before choosing a model family,
and know what follows from the answer. That I build baselines before the thing I am hoping will
win. And that when a result looked significant, I tried to break it before believing it, and
published the version that survived.
