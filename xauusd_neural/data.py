"""Loaders for the CSV exports used by the notebooks (MT5, Investing.com, FRED)."""

import re
from pathlib import Path

import pandas as pd


def find_file(data_dir, key):
    """Return the first CSV in ``data_dir`` whose name contains ``key``.

    Matching ignores case, spaces, ``-`` and ``_``.
    """
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    for p in sorted(Path(data_dir).glob("*.csv")):
        if norm(key) in norm(p.name):
            return p
    raise FileNotFoundError(f"No CSV containing '{key}' in {data_dir}")


def load_mt5(path):
    """MT5 export: tab-separated, no header: Date Open High Low Close [Volume ...]."""
    df = pd.read_csv(path, sep="\t", header=None, usecols=range(5),
                     names=["Date", "Open", "High", "Low", "Close"])
    df["Date"] = pd.to_datetime(df["Date"], format="mixed")
    df = df.set_index("Date").sort_index()
    return df[~df.index.duplicated(keep="last")]


def load_investing(path):
    """Investing.com export -> daily close series (``Price`` column)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().strip('"') for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    s = df.set_index("Date")["Price"].sort_index()
    if s.dtype == object:
        s = s.str.replace(",", "").astype(float)
    return s[~s.index.duplicated(keep="last")].astype(float)


def load_fred(path, monthly=False):
    """FRED export (``observation_date,<SERIES>``). ``monthly=True`` -> PeriodIndex."""
    df = pd.read_csv(path)
    df.columns = ["Date", "Value"]
    df["Date"] = pd.to_datetime(df["Date"])
    s = df.set_index("Date")["Value"].sort_index().astype(float)
    if monthly:
        s.index = s.index.to_period("M")
    return s
