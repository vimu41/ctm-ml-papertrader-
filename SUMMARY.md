# CTM ML v3 paper trading - live forward test (risk 1 ATR, target 1.5 ATR)

Started 2026-11-10 04:10 UTC · last run 2026-11-27 18:07 UTC

| Metric | Value |
|---|---|
| Paper balance | $9,981.89 (start $10,000) |
| Closed trades | 3 |
| Win rate | 33% |
| Average R per trade | -0.241 (v3 backtest: about +0.03, win 47%) |
| Total R | -0.72 |
| Worst drawdown | -2.17 R |
| Open trades | 0 |

## Last 20 closed trades

| Symbol | Side | Entry (UTC) | Exit (UTC) | Reason | R | $ |
|---|---|---|---|---|---|---|
| AUDJPY | BUY | 2026-11-23 01:00 | 2026-11-23 13:00 | SL | -1.07 | -26.76 |
| AUDUSD | BUY | 2026-11-23 04:00 | 2026-11-23 13:00 | SL | -1.104 | -27.59 |
| XAGUSD | BUY | 2026-11-23 03:00 | 2026-11-23 06:00 | TP | 1.45 | 36.24 |

Full journal: `data/trades.csv` · every signal: `data/signals.csv`. Paper trades on Deriv public prices, spread deducted; no real or demo orders are placed.
