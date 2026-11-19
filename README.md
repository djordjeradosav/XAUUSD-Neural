# XAUUSD-Neural

LSTM neural network and gradient-boosted trees for gold (XAUUSD, 1h) trades with a fixed **1:2 risk:reward**, built for my bachelor's thesis (*diplomski rad*). Notebook text and comments are in Serbian.

For every hourly candle the model outputs the probability of three outcomes (triple-barrier labels):

| class | meaning |
|---|---|
| 0 | no trade: neither long nor short reaches +2R before −1R |
| 1 | **short** hits TP (+2R) before SL (−1R) |
| 2 | **long** hits TP (+2R) before SL (−1R) |

Trade rules: enter at the open of the next candle, `R = SL_ATR_MULT × ATR14`, SL = 1R, TP = 2R, and a time limit of `HORIZON` candles (then exit at the close). If a candle touches both SL and TP, the trade counts as SL. **Train: 2016–2025, test: 2026.**

## Notebooks

| notebook | contents |
|---|---|
| `diplomski_XAUUSD_LSTM_RR12_1h (1) 3.ipynb` | **Main notebook.** Features, labels, LSTM, trees and ensemble, trend rule, meta-model, walk-forward. |
| `diplomski_XAUUSD_LSTM_RR12_simple.ipynb` | Compact version of the same pipeline (`SL = 1 × ATR`, `HORIZON = 24`, 24 back-candles). |
| `diplomski_XAUUSD_LSTM_RR12_1h (1) 2.ipynb` | Earlier experiment with **RR = 3** (title still says 1:2), `SL = 1 × ATR`, no walk-forward. |

## Pipeline (main notebook)

1. **Data**: MT5 exports of XAUUSD (1h and 1d) and BTCUSD (1d); Investing.com DXY, VIX and WTI; FRED US10Y, CPI, PPI and NFP.
2. **Features (54)**: scale-free technicals (RSI, MACD, EMA/SMA ratios, Bollinger position, ATR, returns, volatility), intraday position and session, the previous daily gold candle, and daily macro data shifted by one day. Monthly macro data is lagged two months, so it is known at decision time.
3. **Labels**: vectorised triple barrier for long and short on every candle.
4. **Models**: a 3-class LSTM (43 candles × 54 features) with class weights and early stopping, plus `HistGradientBoostingClassifier` and their average (ensemble).
5. **Signals**: LONG if `P(2) ≥ threshold` and `P(2) > P(1)`, SHORT analogously. Thresholds are percentiles of **validation** probabilities. One position at a time, with a $0.25 spread converted to R.
6. **Trend rule and meta-model**: a ±N-pips move over a lookback window plus EMA confirmation sets the direction. Its parameters are chosen on 2016–2025 only. A secondary model (trees, LSTM, ensemble) decides whether to take each trend setup.
7. **Walk-forward**: five expanding folds, validated on 2021–2025.

## Results (test = 2026-01-04 … 2026-11-02, from the saved outputs of the main notebook)

3-class LSTM: test accuracy 72.7% vs. 80.2% for always predicting the majority class. Balanced accuracy is 35.5% (chance ≈ 33%), and the one-vs-rest AUC is 0.51–0.58. **The LSTM cannot predict direction on its own.** Trend-following plus a meta-model that filters setups does better:

| strategy (1:2, net of spread) | trades | TP hit | mean R / trade | total R | profit factor | max DD (R) |
|---|---|---|---|---|---|---|
| LSTM signals | 89 | 12.4% | −0.010 | −0.9 | 0.98 | −13.2 |
| Trees signals | 60 | 13.3% | 0.041 | 2.4 | 1.08 | −10.6 |
| Trend rule only | 141 | 13.5% | 0.109 | 15.4 | 1.25 | −7.5 |
| Trend + meta LSTM | 83 | 19.3% | 0.302 | 25.1 | 1.73 | −4.5 |
| Trend + meta ensemble | 64 | 20.3% | 0.279 | 17.8 | 1.61 | −5.0 |
| Baseline: always long | 171 | 7.6% | −0.079 | −13.6 | 0.85 | −29.1 |
| Baseline: always short | 155 | 12.3% | 0.115 | 17.8 | 1.25 | −12.9 |

The meta-model AUC is 0.63 on test for the meta LSTM, against 0.69 on validation.

### Limitations

- **One short test window.** About 10 months and 60–170 trades per strategy, so ±0.2 R/trade is within noise. The best variant (meta LSTM, +25.1 R) was picked after seeing several variants on the same test set. Its edge over the "always short" baseline (+17.8 R) is not statistically established.
- **Walk-forward thresholds are in-sample.** `optimize_thresholds_strict` picks the thresholds and reports `mean_R` on the same validation fold.
- **Costs.** Only a fixed $0.25 spread is included. Swap, slippage and commission are not.

## Running

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib joblib jupyter
jupyter lab "diplomski_XAUUSD_LSTM_RR12_1h (1) 3.ipynb"
```

Put the CSV exports into `./dataset` (or set `XAU_DATA_DIR`). The data files are matched by name, case-insensitive:
- **MT5** (tab-separated, no header): `XAUUSD60.csv`, `XAUUSD1440.csv`, `BTCUSD1440*.csv`.
- **Investing.com**: `*US_Dollar_Index*`, `*Volatility_Index*`, `*WTI*`.
- **FRED**: `*US10YN*`, `*CPI*`, `*PPI*`, `*NFP*`.

Market data is not included in the repository.

Disclaimer: research and educational code, not financial advice.
