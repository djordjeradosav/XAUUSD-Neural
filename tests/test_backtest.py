import numpy as np
import pandas as pd
import pytest

from xauusd_neural import make_signals, rr_stats, run_rr_backtest, trend_state


def test_make_signals():
    p = np.array([[0.8, 0.1, 0.1],   # below thresholds
                  [0.2, 0.1, 0.7],   # long
                  [0.2, 0.7, 0.1],   # short
                  [0.0, 0.5, 0.5]])  # tie -> none
    assert make_signals(p, 0.5, 0.5).tolist() == [0, 1, -1, 0]


def _lab(index, res, exit_pos, R=10.0):
    return pd.DataFrame({"R": R, "long_res": res, "short_res": [-r for r in res],
                         "long_exit": exit_pos, "short_exit": exit_pos}, index=index)


def test_backtest_one_position_at_a_time_and_spread():
    idx = pd.date_range("2024-01-01", periods=6, freq="h")
    lab = _lab(idx, [2.0, -1.0, 2.0, -1.0, 2.0, 0.5], [3, 2, 3, 4, 5, 5])
    sig = pd.Series([1, 1, 1, -1, 1, 0], index=idx)
    tr = run_rr_backtest(sig, lab, idx, spread=1.0)
    # trade at 0 is open until bar 3 -> signals at 1 and 2 are skipped; 3 (short) then 4 (long)
    assert list(tr.index) == [idx[0], idx[3], idx[4]]
    assert tr["side"].tolist() == ["long", "short", "long"]
    assert tr["R_gross"].tolist() == [2.0, 1.0, 2.0]
    assert tr["R_net"].tolist() == pytest.approx([1.9, 0.9, 1.9])


def test_rr_stats():
    idx = pd.date_range("2024-01-01", periods=4, freq="h")
    tr = pd.DataFrame({"side": ["long", "short", "long", "long"], "R_gross": [2.0, -1.0, -1.0, 0.5],
                       "R_net": [2.0, -1.0, -1.0, 0.5]}, index=idx)
    s = rr_stats(tr, rr=2.0)
    assert s["trades"] == 4
    assert s["TP hit"] == 0.25 and s["SL hit"] == 0.5 and s["timeout"] == 0.25
    assert s["total R (net)"] == pytest.approx(0.5)
    assert s["profit factor"] == pytest.approx(2.5 / 2.0)
    assert s["max drawdown (R)"] == pytest.approx(-2.0)
    assert rr_stats(tr.iloc[:0]) == {"trades": 0}


def test_trend_state():
    close = pd.Series([100.0] * 5 + [130.0] * 5 + [90.0] * 5)
    t = trend_state(close, lookback=3, pips=250, ema_len=0, pip_size=0.10)
    assert t.iloc[6] == 1 and t.iloc[11] == -1 and t.iloc[14] == 0
    t_ema = trend_state(close, lookback=3, pips=250, ema_len=50, pip_size=0.10)
    assert t_ema.iloc[11] == -1
    up_below_ema = pd.Series([200.0] * 60 + [100.0] * 5 + [130.0] * 3)
    assert trend_state(up_below_ema, 3, 250, 50).iloc[-1] == 0
