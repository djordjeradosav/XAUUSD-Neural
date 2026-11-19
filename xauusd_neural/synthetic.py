"""Generate a synthetic dataset in the same file formats the notebooks expect.

The real data (MT5 / Investing.com / FRED exports) is not redistributed. This lets anyone run
the notebooks and tests end to end; the numbers it produces are meaningless (random walk).

    python -m xauusd_neural.synthetic --out dataset
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _ohlc_from_path(index, close, rng, wick):
    open_ = np.r_[close[0], close[:-1]]
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, wick, len(close))))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, wick, len(close))))
    return pd.DataFrame({"Open": open_, "High": hi, "Low": lo, "Close": close}, index=index)


def _random_walk(n, start, vol, rng, drift=0.0):
    return start * np.exp(np.cumsum(rng.normal(drift, vol, n)))


def _write_mt5(df, path):
    out = df.copy()
    out["Volume"] = 1000
    out.index = out.index.strftime("%Y-%m-%d %H:%M:%S")
    out.round(3).to_csv(path, sep="\t", header=False)


def _write_investing(s, path, rng):
    s = s.sort_index(ascending=False)
    df = pd.DataFrame({
        "Date": s.index.strftime("%m/%d/%Y"),
        "Price": [f"{v:,.2f}" for v in s.values],
        "Open": [f"{v:,.2f}" for v in s.values],
        "High": [f"{v:,.2f}" for v in s.values * 1.005],
        "Low": [f"{v:,.2f}" for v in s.values * 0.995],
        "Vol.": "",
        "Change %": [f"{v:.2%}" for v in rng.normal(0, 0.01, len(s))],
    })
    df.to_csv(path, index=False, quoting=1)


def _write_fred(s, name, path):
    pd.DataFrame({"observation_date": s.index.strftime("%Y-%m-%d"), name: s.round(4).values}).to_csv(
        path, index=False)


def make_synthetic_dataset(out_dir, start="2016-01-01", end="2026-11-05", seed=0):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    hours = pd.date_range(start, end, freq="h")
    hours = hours[hours.dayofweek < 5]
    gold_1h = _ohlc_from_path(hours, _random_walk(len(hours), 1060.0, 0.0025, rng, 2e-5), rng, 0.001)
    gold_1d = gold_1h.resample("D").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    _write_mt5(gold_1h, out / "XAUUSD60.csv")
    _write_mt5(gold_1d, out / "XAUUSD1440.csv")

    btc_days = pd.date_range("2017-05-23", end, freq="D")
    btc = _ohlc_from_path(btc_days, _random_walk(len(btc_days), 2000.0, 0.03, rng), rng, 0.01)
    _write_mt5(btc, out / "BTCUSD1440.csv")

    bdays = pd.bdate_range(start, end)
    _write_investing(pd.Series(_random_walk(len(bdays), 98.0, 0.004, rng), bdays),
                     out / "US_Dollar_Index_Historical_Data.csv", rng)
    _write_investing(pd.Series(np.clip(_random_walk(len(bdays), 18.0, 0.05, rng), 9, 80), bdays),
                     out / "CBOE_Volatility_Index_Historical_Data.csv", rng)
    _write_investing(pd.Series(_random_walk(len(bdays), 37.0, 0.02, rng), bdays),
                     out / "Crude_Oil_WTI_Futures_Historical_Data.csv", rng)
    _write_fred(pd.Series(np.clip(2.2 + np.cumsum(rng.normal(0, 0.04, len(bdays))), 0.3, 6), bdays),
                "DGS10", out / "US10YN.csv")

    months = pd.date_range(start, end, freq="MS")
    _write_fred(pd.Series(_random_walk(len(months), 237.0, 0.002, rng, 0.002), months), "CPIAUCSL", out / "CPI.csv")
    _write_fred(pd.Series(_random_walk(len(months), 185.0, 0.004, rng, 0.002), months), "PPIACO", out / "PPI.csv")
    _write_fred(pd.Series(143000 + np.cumsum(rng.normal(150, 200, len(months))), months), "PAYEMS",
                out / "NFP.csv")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-11-05")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print("Synthetic dataset written to", make_synthetic_dataset(a.out, a.start, a.end, a.seed).resolve())


if __name__ == "__main__":
    main()
