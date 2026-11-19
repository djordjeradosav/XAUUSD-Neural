import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def hourly_ohlc():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2020-01-01", periods=24 * 40, freq="h")
    close = 1500 * np.exp(np.cumsum(rng.normal(0, 0.002, len(idx))))
    open_ = np.r_[close[0], close[:-1]]
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 5e-4, len(idx))))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 5e-4, len(idx))))
    return pd.DataFrame({"Open": open_, "High": hi, "Low": lo, "Close": close}, index=idx)


def make_bars(rows, start="2024-01-01"):
    """rows: list of (open, high, low, close)."""
    idx = pd.date_range(start, periods=len(rows), freq="h")
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx, dtype=float)
