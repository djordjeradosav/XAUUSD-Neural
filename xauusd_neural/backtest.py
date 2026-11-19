"""Signal generation and a one-position-at-a-time R-multiple backtest."""

import numpy as np
import pandas as pd


def make_signals(proba, thr_long, thr_short):
    """3-class probabilities (0 = none, 1 = short TP, 2 = long TP) -> +1 / -1 / 0."""
    sig = np.zeros(len(proba), dtype=int)
    sig[(proba[:, 2] >= thr_long) & (proba[:, 2] > proba[:, 1])] = 1
    sig[(proba[:, 1] >= thr_short) & (proba[:, 1] > proba[:, 2])] = -1
    return sig


def run_rr_backtest(sig, lab, gold_index, spread=0.25):
    """Backtest signals using pre-computed ``triple_barrier`` outcomes.

    ``sig`` is a Series indexed by decision time with values 1 / -1 / 0. The trade opens on the
    next bar; while a trade is open new signals are skipped. ``spread`` (USD/oz round trip) is
    converted to R and subtracted.
    """
    pos = gold_index.get_indexer(sig.index)
    rows, busy = [], -1
    for tm, s, p in zip(sig.index, sig.values, pos):
        if s == 0 or p < busy:
            continue
        side = "long" if s == 1 else "short"
        res, R = float(lab.at[tm, f"{side}_res"]), float(lab.at[tm, "R"])
        rows.append((tm, side, res, res - spread / R))
        busy = int(lab.at[tm, f"{side}_exit"])
    return pd.DataFrame(rows, columns=["time", "side", "R_gross", "R_net"]).set_index("time")


def rr_stats(tr, rr=2.0, risk_pct=1.0):
    """Summary statistics of a trade list produced by ``run_rr_backtest``."""
    if tr.empty:
        return {"trades": 0}
    eq = tr["R_net"].cumsum()
    wins, losses = tr["R_net"][tr["R_net"] > 0].sum(), -tr["R_net"][tr["R_net"] < 0].sum()
    tp = tr["R_gross"] >= rr - 1e-9
    sl = tr["R_gross"] <= -1 + 1e-9
    return {
        "trades": len(tr),
        "long / short": f"{(tr['side'] == 'long').sum()} / {(tr['side'] == 'short').sum()}",
        "TP hit": tp.mean(),
        "SL hit": sl.mean(),
        "timeout": (~tp & ~sl).mean(),
        "mean R (net)": tr["R_net"].mean(),
        "total R (net)": tr["R_net"].sum(),
        "profit factor": wins / losses if losses > 0 else np.nan,
        "max drawdown (R)": (eq - eq.cummax()).min(),
        f"return at {risk_pct:g}% risk": (1 + risk_pct / 100 * tr["R_net"]).prod() - 1,
    }
