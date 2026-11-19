import numpy as np
import pandas as pd
import pytest

from conftest import make_bars
from xauusd_neural import triple_barrier

FLAT = (100, 100.5, 99.5, 100)


def label(rows, horizon=3, R=1.0, rr=2.0):
    bars = make_bars(rows)
    atr = pd.Series(R, index=bars.index)
    return triple_barrier(bars, atr, sl_mult=1.0, rr=rr, horizon=horizon)


def test_long_hits_tp_first():
    # entry = open of bar 1 = 100, TP = 102, SL = 99
    out = label([FLAT, (100, 100.5, 99.5, 100), (100, 102.5, 99.8, 102), FLAT, FLAT])
    row = out.iloc[0]
    assert row["long_win"] and not row["short_win"]
    assert row["long_res"] == 2.0
    assert row["label"] == 2
    assert row["long_exit"] == 2


def test_short_hits_tp_first():
    out = label([FLAT, (100, 100.5, 99.5, 100), (100, 100.2, 97.5, 98), FLAT, FLAT])
    row = out.iloc[0]
    assert row["short_win"] and not row["long_win"]
    assert row["short_res"] == 2.0
    assert row["label"] == 1


def test_same_bar_sl_and_tp_counts_as_sl():
    out = label([FLAT, (100, 102.5, 98.5, 100), FLAT, FLAT, FLAT])
    row = out.iloc[0]
    assert not row["long_win"] and row["long_res"] == -1.0
    assert not row["short_win"] and row["short_res"] == -1.0
    assert row["label"] == 0


def test_timeout_exits_at_last_close():
    out = label([FLAT, (100, 100.5, 99.5, 100.2), (100.2, 100.6, 99.6, 100.4), (100.4, 100.8, 99.9, 100.6), FLAT])
    row = out.iloc[0]
    assert row["label"] == 0
    assert row["long_res"] == pytest.approx(0.6)
    assert row["short_res"] == pytest.approx(-0.6)
    assert row["long_exit"] == 3


def test_only_rows_with_full_window_and_nan_risk_unlabeled():
    bars = make_bars([FLAT] * 10)
    atr = pd.Series([np.nan] + [1.0] * 9, index=bars.index)
    out = triple_barrier(bars, atr, 1.0, 2.0, horizon=4)
    assert len(out) == 6
    assert np.isnan(out["label"].iloc[0])
    assert out["label"].iloc[1:].notna().all()


def test_labels_depend_only_on_future_window(hourly_ohlc):
    """Changing bars after t+horizon must not change the label of bar t."""
    H = 12
    atr = (hourly_ohlc["High"] - hourly_ohlc["Low"]).rolling(14).mean()
    base = triple_barrier(hourly_ohlc, atr, 1.0, 2.0, H)
    t = 300
    shocked = hourly_ohlc.copy()
    shocked.iloc[t + H + 1:] *= 1.5
    after = triple_barrier(shocked, atr, 1.0, 2.0, H)
    cols = ["label", "long_res", "short_res", "long_exit", "short_exit"]
    pd.testing.assert_frame_equal(base.iloc[: t + 1][cols], after.iloc[: t + 1][cols])
