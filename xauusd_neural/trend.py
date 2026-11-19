"""Rule-based trend direction: price move of +/- ``pips`` over ``lookback`` bars, confirmed by an EMA."""

import pandas as pd


def trend_state(close, lookback, pips, ema_len, pip_size=0.10):
    """+1 = LONG, -1 = SHORT, 0 = neutral. ``ema_len=0`` disables the EMA confirmation."""
    mv = (close - close.shift(lookback)) / pip_size
    t = pd.Series(0, index=close.index)
    t[mv >= pips] = 1
    t[mv <= -pips] = -1
    if ema_len:
        ema = close.ewm(span=ema_len, adjust=False).mean()
        t[(t == 1) & (close < ema)] = 0
        t[(t == -1) & (close > ema)] = 0
    return t
