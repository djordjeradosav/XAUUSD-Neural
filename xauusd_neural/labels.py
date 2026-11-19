"""Triple-barrier labels for fixed risk:reward trades."""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def triple_barrier(ohlc, atr, sl_mult, rr, horizon):
    """Simulate a hypothetical long and short trade for every bar ``t``.

    * entry = open of bar ``t+1``; ``R = sl_mult * atr[t]``; SL = entry -/+ 1R, TP = entry +/- rr*R
    * if one bar touches both SL and TP, SL is assumed (conservative)
    * if neither is hit within ``horizon`` bars, exit at the close of bar ``t+horizon``

    Returns one row per bar ``t`` that has a full window, with ``{side}_win``, ``{side}_res``
    (result in R, gross), ``{side}_exit`` (positional index of the exit bar) and
    ``label``: 2 = long hits TP first, 1 = short hits TP first, 0 = neither.
    """
    o, h, l, c = (ohlc[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
    n = len(ohlc)
    m = n - horizon
    Hw = sliding_window_view(h, horizon)[1:m + 1]
    Lw = sliding_window_view(l, horizon)[1:m + 1]
    c_last = c[horizon:horizon + m]
    e = o[1:m + 1]
    R = np.asarray(atr, dtype=float)[:m] * sl_mult
    big = horizon + 1
    out = pd.DataFrame(index=ohlc.index[:m])
    out["entry"], out["R"] = e, R
    for side, d in (("long", 1), ("short", -1)):
        tp, sl = e + d * rr * R, e - d * R
        hit_tp = (Hw >= tp[:, None]) if d == 1 else (Lw <= tp[:, None])
        hit_sl = (Lw <= sl[:, None]) if d == 1 else (Hw >= sl[:, None])
        i_tp = np.where(hit_tp.any(1), hit_tp.argmax(1), big)
        i_sl = np.where(hit_sl.any(1), hit_sl.argmax(1), big)
        win = i_tp < i_sl
        loss = (i_sl < big) & (i_sl <= i_tp)
        with np.errstate(divide="ignore", invalid="ignore"):
            res = np.where(win, rr, np.where(loss, -1.0, d * (c_last - e) / R))
        ex = np.where(win, i_tp, np.where(loss, i_sl, horizon - 1))
        out[f"{side}_win"] = win
        out[f"{side}_res"] = res
        out[f"{side}_exit"] = np.arange(m) + 1 + ex
    out["label"] = np.select([out["long_win"], out["short_win"]], [2, 1], default=0).astype(float)
    out.loc[out["R"].isna(), "label"] = np.nan
    return out
