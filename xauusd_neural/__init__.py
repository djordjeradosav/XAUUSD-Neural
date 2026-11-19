"""Reusable, tested core of the XAUUSD LSTM notebooks.

The functions mirror the ones defined inline in ``notebooks/02_lstm_rr12_1h_full.ipynb``
so they can be unit-tested (including look-ahead checks) and reused outside Jupyter.
"""

from .backtest import make_signals, rr_stats, run_rr_backtest
from .data import find_file, load_fred, load_investing, load_mt5
from .features import (
    attach_daily,
    attach_monthly,
    daily_series_features,
    intraday_features,
    monthly_features,
    rsi,
    technical_features,
    true_range,
)
from .labels import triple_barrier
from .trend import trend_state

__all__ = [
    "attach_daily",
    "attach_monthly",
    "daily_series_features",
    "find_file",
    "intraday_features",
    "load_fred",
    "load_investing",
    "load_mt5",
    "make_signals",
    "monthly_features",
    "rr_stats",
    "rsi",
    "run_rr_backtest",
    "technical_features",
    "trend_state",
    "triple_barrier",
    "true_range",
]
