"""Reproduce every number in the README with one command.

    python scripts/run_experiment.py

The notebooks tell the story; this script proves the story is reproducible. It
fits every model in the results table on the same chronological splits and
prints the comparison.

Use --seeds to change how many random seeds the log return LSTM is trained
across. The default of 5 is what the README reports. Setting it to 1 reproduces
the single run result that looks like a directional edge, which is the point the
project is making.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

from src import config, evaluate as ev
from src.data import load_prices
from src.models import arima
from src.models import baselines as bl
from src.windowing import build_sequences, inverse

ARIMA_ORDERS = [(0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 2), (5, 1, 0)]
DEFAULT_SEEDS = [7, 17, 42, 123, 2024]


def main(seeds: list[int], skip_lstm: bool) -> None:
    started = time.time()

    close = load_prices(config.TICKER).iloc[:-1]["Close"]
    n = len(close)
    train_sl, val_sl, test_sl = ev.chronological_split(
        n, config.TRAIN_FRAC, config.VAL_FRAC
    )
    test_idx = np.arange(test_sl.start, test_sl.stop)
    actual = close.values[test_idx]
    previous = close.values[test_idx - 1]
    n_test = len(test_idx)

    print(f"{config.TICKER}: {n} sessions, "
          f"{close.index[0].date()} to {close.index[-1].date()}")
    print(f"split: {train_sl.stop} train / {val_sl.stop - val_sl.start} val / {n_test} test\n")

    rows = []

    # ---- baselines -------------------------------------------------------
    val_idx = np.arange(val_sl.start, val_sl.stop)
    best_sma, _ = bl.tune_window(close, val_idx, bl.sma)
    best_ema, _ = bl.tune_window(close, val_idx, bl.ema)
    print(f"tuned on validation: SMA window {best_sma}, EMA span {best_ema}")

    rows.append(ev.evaluate("Naive (persistence)", actual, bl.naive(close, test_idx), previous))
    rows.append(ev.evaluate("Drift", actual, bl.drift(close, test_idx, train_sl.stop), previous))
    rows.append(ev.evaluate(f"SMA({best_sma})", actual, bl.sma(close, test_idx, best_sma), previous))
    rows.append(ev.evaluate(f"EMA({best_ema})", actual, bl.ema(close, test_idx, best_ema), previous))

    # ---- ARIMA -----------------------------------------------------------
    grid = arima.fit_grid(close.iloc[: val_sl.stop], ARIMA_ORDERS)
    chosen = grid.loc[0, "order"]
    print(f"ARIMA order chosen by AIC: {chosen}")
    for order in [(0, 1, 0), (1, 1, 1)]:
        pred = arima.rolling_forecast(close, order, n_test)
        rows.append(ev.evaluate(f"ARIMA{order}", actual, pred, previous))

    if not skip_lstm:
        from src.models import lstm

        # ---- LSTM on price levels, honest scaling ------------------------
        seq = build_sequences(close, train_sl.stop, val_sl.stop, window=config.WINDOW)
        lstm.set_seeds(config.RANDOM_SEED)
        model = lstm.build((config.WINDOW, 1))
        lstm.train(model, seq.X_train, seq.y_train, seq.X_val, seq.y_val)
        pred = inverse(seq.scaler, model.predict(seq.X_test, verbose=0).ravel())
        rows.append(ev.evaluate("LSTM (price, train only scaler)", actual, pred, previous))
        print("trained: LSTM on price levels, train only scaler")

        # ---- LSTM on price levels, leaky scaling -------------------------
        values = close.values.astype(float).reshape(-1, 1)
        leaky = MinMaxScaler().fit(values)          # the bug, reproduced deliberately
        scaled = leaky.transform(values)
        W = config.WINDOW

        def windows(pos):
            X = np.stack([scaled[p - W:p, 0] for p in pos])
            return X[..., np.newaxis], scaled[pos, 0]

        Xtr, ytr = windows(np.arange(W, train_sl.stop))
        Xva, yva = windows(np.arange(train_sl.stop, val_sl.stop))
        Xte, _ = windows(np.arange(val_sl.stop, n))
        lstm.set_seeds(config.RANDOM_SEED)
        leaky_model = lstm.build((W, 1))
        lstm.train(leaky_model, Xtr, ytr, Xva, yva)
        pred = leaky.inverse_transform(leaky_model.predict(Xte, verbose=0)).ravel()
        rows.append(ev.evaluate("LSTM (leaky scaler)", actual, pred, previous))
        print("trained: LSTM on price levels, leaky scaler")

        # ---- LSTM on log returns, across seeds ---------------------------
        log_returns = np.log(close / close.shift(1)).dropna()
        rv = log_returns.values.astype(float)
        tr_end, va_end = train_sl.stop - 1, val_sl.stop - 1
        mu, sd = rv[:tr_end].mean(), rv[:tr_end].std()
        z = (rv - mu) / sd

        def rwindows(pos):
            X = np.stack([z[p - W:p] for p in pos])
            return X[..., np.newaxis], z[pos]

        Xtr, ytr = rwindows(np.arange(W, tr_end))
        Xva, yva = rwindows(np.arange(tr_end, va_end))
        Xte, _ = rwindows(np.arange(va_end, len(rv)))

        seed_rows = []
        for seed in seeds:
            lstm.set_seeds(seed)
            m = lstm.build((W, 1))
            lstm.train(m, Xtr, ytr, Xva, yva)
            p = previous * np.exp(m.predict(Xte, verbose=0).ravel() * sd + mu)
            r = ev.evaluate(f"seed {seed}", actual, p, previous)
            seed_rows.append(r)
            print(f"trained: LSTM on log returns, seed {seed} "
                  f"(RMSE {r['RMSE']:.4f}, directional {r['Directional %']:.2f}%)")

        seeds_df = pd.DataFrame(seed_rows)
        rows.append({
            "model": f"LSTM (log returns, {len(seeds)} seed mean)",
            "RMSE": round(float(seeds_df["RMSE"].mean()), 4),
            "MAE": round(float(seeds_df["MAE"].mean()), 4),
            "MAPE %": round(float(seeds_df["MAPE %"].mean()), 3),
            "Directional %": round(float(seeds_df["Directional %"].mean()), 2),
            "RMSE on returns": round(float(seeds_df["RMSE on returns"].mean()), 6),
        })

    # ---- report ----------------------------------------------------------
    table = ev.results_table(rows)
    naive_rmse = table.loc["Naive (persistence)", "RMSE"]
    table["RMSE vs naive"] = (table["RMSE"] / naive_rmse).round(3)
    table = table.sort_values("RMSE")

    print("\n" + "=" * 92)
    print(table.to_string())
    print("=" * 92)

    se = np.sqrt(0.25 / n_test) * 100
    print(f"\ncoin flip standard error over {n_test} days: +/-{se:.2f}%")

    if not skip_lstm and len(seeds) > 1:
        d = seeds_df["Directional %"]
        k = int(round(d.mean() / 100 * n_test))
        p = stats.binomtest(k, n_test, 0.5, alternative="greater").pvalue
        print(f"log return LSTM directional accuracy across {len(seeds)} seeds: "
              f"mean {d.mean():.2f}%, std {d.std():.2f}, range {d.min():.2f} to {d.max():.2f}")
        print(f"one sided binomial p for the mean: {p:.4f}")
        beat = int((seeds_df['RMSE'] < naive_rmse).sum())
        print(f"beat the naive baseline in {beat} of {len(seeds)} runs")

    config.ensure_dirs()
    out = config.DATA_PROCESSED / "results_all.csv"
    table.to_csv(out)
    print(f"\nsaved -> {out.relative_to(config.PROJECT_ROOT)}")
    print(f"total runtime: {time.time() - started:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS),
                        help="how many random seeds to train the log return LSTM across")
    parser.add_argument("--skip-lstm", action="store_true",
                        help="baselines and ARIMA only, runs in seconds")
    args = parser.parse_args()
    main(DEFAULT_SEEDS[: max(1, args.seeds)], args.skip_lstm)
