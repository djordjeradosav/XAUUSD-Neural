"""Technical, intraday and macro features. Every feature at bar ``t`` uses data up to ``t`` only."""

import numpy as np
import pandas as pd

DEFAULT_TA = dict(
    rsi_len=16,
    macd=(12, 26, 9),
    ema_lens=(20, 50, 100),
    sma_lens=(50, 200),
    boll_len=20,
    atr_len=16,
)


def rsi(close, n):
    """Wilder RSI."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def true_range(x):
    c = x["Close"]
    return pd.concat([x["High"] - x["Low"], (x["High"] - c.shift(1)).abs(),
                      (x["Low"] - c.shift(1)).abs()], axis=1).max(axis=1)


def technical_features(x, ta=None):
    """Scale-free technical features (ratios / returns), so the model does not learn the price level."""
    ta = {**DEFAULT_TA, **(ta or {})}
    f = pd.DataFrame(index=x.index)
    c = x["Close"]

    f["RSI"] = rsi(c, ta["rsi_len"])

    fast, slow, sig = ta["macd"]
    macd_line = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    f["MACD_line"] = macd_line / c
    f["MACD_hist"] = (macd_line - macd_line.ewm(span=sig, adjust=False).mean()) / c

    emas = {n: c.ewm(span=n, adjust=False).mean() for n in ta["ema_lens"]}
    for n, e in emas.items():
        f[f"EMA{n}_ratio"] = c / e - 1
    ns = sorted(emas)
    f[f"EMA{ns[0]}_vs_EMA{ns[-1]}"] = emas[ns[0]] / emas[ns[-1]] - 1

    smas = {n: c.rolling(n).mean() for n in ta["sma_lens"]}
    for n, s in smas.items():
        f[f"SMA{n}_ratio"] = c / s - 1
    ns = sorted(smas)
    f[f"SMA{ns[0]}_vs_SMA{ns[-1]}"] = smas[ns[0]] / smas[ns[-1]] - 1

    mid, sd = c.rolling(ta["boll_len"]).mean(), c.rolling(ta["boll_len"]).std()
    f["BOLL_pos"] = (c - mid) / (2 * sd.replace(0, np.nan))

    f["ATR_norm"] = true_range(x).rolling(ta["atr_len"]).mean() / c

    for h in (1, 3, 6, 12, 24):
        f[f"ret_{h}"] = c.pct_change(h)
    r1 = c.pct_change()
    for w in (12, 24, 72):
        f[f"vol_{w}"] = r1.rolling(w).std()
    f["dow"] = x.index.dayofweek
    return f


def intraday_features(x):
    """Position within the current day + hour/session encodings (1h data only)."""
    f = pd.DataFrame(index=x.index)
    day = pd.Index(x.index.date)
    c = x["Close"]
    hi = x["High"].groupby(day).cummax()
    lo = x["Low"].groupby(day).cummin()
    f["day_range_pos"] = (c - lo) / (hi - lo).replace(0, np.nan)
    f["dist_day_high"] = (hi - c) / c
    f["dist_day_low"] = (c - lo) / c
    daily_close = c.groupby(day).last()
    daily_close.index = pd.to_datetime(daily_close.index)
    f["gap_vs_prev_close"] = c / x.index.normalize().map(daily_close.shift(1)).values - 1
    h = x.index.hour
    f["hour_sin"] = np.sin(2 * np.pi * h / 24)
    f["hour_cos"] = np.cos(2 * np.pi * h / 24)
    f["sess_asia"] = ((h >= 0) & (h <= 7)).astype(int)
    f["sess_london"] = ((h >= 8) & (h <= 12)).astype(int)
    f["sess_ny"] = ((h >= 13) & (h <= 20)).astype(int)
    return f


def daily_series_features(s, prefix):
    d = pd.DataFrame(index=s.index)
    d[f"{prefix}_ret1d"] = s.pct_change()
    d[f"{prefix}_ret5d"] = s.pct_change(5)
    m, sd = s.rolling(20).mean(), s.rolling(20).std()
    d[f"{prefix}_z20"] = (s - m) / sd.replace(0, np.nan)
    d[f"{prefix}_sma20_ratio"] = s / m - 1
    return d


def attach_daily(bar_index, feat):
    """Align daily features to intraday bars without look-ahead.

    A daily value for day D is only known at the end of D, so it is forward-filled onto
    calendar days and shifted by one day: a bar on date D sees days <= D-1.
    """
    end = max(feat.index.max(), bar_index.max().normalize())
    cal = feat.reindex(pd.date_range(feat.index.min(), end, freq="D")).ffill().shift(1)
    out = cal.reindex(bar_index.normalize())
    out.index = bar_index
    return out


def monthly_features(s, prefix, end):
    """MoM change + difference, lagged 2 months (release in M+1, conservatively M+2)."""
    s = s.interpolate(limit_area="inside")
    d = pd.DataFrame({f"{prefix}_mom": s.pct_change(), f"{prefix}_diff": s.diff()})
    d = d.reindex(pd.period_range(d.index.min(), max(d.index.max(), end), freq="M"))
    return d.shift(2)


def attach_monthly(bar_index, feat):
    out = feat.reindex(bar_index.to_period("M"))
    out.index = bar_index
    return out
