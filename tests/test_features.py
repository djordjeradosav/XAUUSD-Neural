import numpy as np
import pandas as pd

from xauusd_neural import (attach_daily, attach_monthly, daily_series_features, intraday_features,
                           monthly_features, rsi, technical_features)


def test_technical_features_are_causal(hourly_ohlc):
    """Features at bar t computed on data[:t+1] equal features computed on the full history."""
    full = technical_features(hourly_ohlc)
    t = 600
    part = technical_features(hourly_ohlc.iloc[: t + 1])
    pd.testing.assert_frame_equal(full.iloc[: t + 1], part)


def test_intraday_features_are_causal(hourly_ohlc):
    full = intraday_features(hourly_ohlc)
    t = 500
    part = intraday_features(hourly_ohlc.iloc[: t + 1])
    pd.testing.assert_frame_equal(full.iloc[: t + 1], part)


def test_technical_features_are_scale_free(hourly_ohlc):
    a = technical_features(hourly_ohlc)
    b = technical_features(hourly_ohlc * 3.0)
    pd.testing.assert_frame_equal(a, b, rtol=1e-6)


def test_rsi_bounds():
    zigzag = np.tile([1.0, -0.5], 25)
    s = pd.Series(100 + np.cumsum(np.r_[zigzag, -zigzag]))
    r = rsi(s, 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    assert r.iloc[48] > 60 and r.iloc[-1] < 40


def test_attach_daily_uses_previous_day_only():
    days = pd.date_range("2024-01-01", "2024-01-10", freq="D")
    daily = pd.DataFrame({"v": np.arange(len(days), dtype=float)}, index=days)
    bars = pd.date_range("2024-01-03 00:00", "2024-01-03 23:00", freq="h")
    out = attach_daily(bars, daily)
    assert (out["v"] == daily.loc["2024-01-02", "v"]).all()


def test_attach_daily_forward_fills_weekends():
    bdays = pd.bdate_range("2024-01-01", "2024-01-12")
    daily = pd.DataFrame({"v": np.arange(len(bdays), dtype=float)}, index=bdays)
    monday = pd.date_range("2024-01-08 00:00", periods=3, freq="h")
    out = attach_daily(monday, daily)
    assert (out["v"] == daily.loc["2024-01-05", "v"]).all()


def test_monthly_features_are_lagged_two_months():
    months = pd.period_range("2023-01", "2023-12", freq="M")
    s = pd.Series(np.arange(100, 112, dtype=float), index=months)
    feat = monthly_features(s, "cpi", pd.Period("2024-02", "M"))
    bars = pd.date_range("2023-06-15", periods=2, freq="h")
    out = attach_monthly(bars, feat)
    expected = s.diff().loc[pd.Period("2023-04", "M")]
    assert (out["cpi_diff"] == expected).all()


def test_daily_series_features_columns():
    s = pd.Series(np.linspace(1, 2, 40), index=pd.bdate_range("2024-01-01", periods=40))
    f = daily_series_features(s, "dxy")
    assert list(f.columns) == ["dxy_ret1d", "dxy_ret5d", "dxy_z20", "dxy_sma20_ratio"]
