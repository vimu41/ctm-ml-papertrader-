# CTM ML paper trading - live forward test

Started 2026-11-10 04:10 UTC · last run 2026-11-24 22:09 UTC

| Metric | Value |
|---|---|
| Paper balance | $10,014.15 (start $10,000) |
| Closed trades | 5 |
| Win rate | 40% |
| Average R per trade | +0.115 (backtest: about +0.12) |
| Total R | +0.57 |
| Worst drawdown | -3.28 R |
| Open trades | 0 |

## Last 20 closed trades

| Symbol | Side | Entry (UTC) | Exit (UTC) | Reason | R | $ |
|---|---|---|---|---|---|---|
| AUDUSD | BUY | 2026-11-23 16:00 | 2026-11-24 15:00 | TP | 1.9 | 47.36 |
| XAGUSD | BUY | 2026-11-23 03:00 | 2026-11-23 14:00 | TP | 1.95 | 48.61 |
| AUDJPY | BUY | 2026-11-23 01:00 | 2026-11-23 13:00 | SL | -1.07 | -26.76 |
| AUDUSD | BUY | 2026-11-23 03:00 | 2026-11-23 13:00 | SL | -1.103 | -27.5 |
| AUDUSD | BUY | 2026-11-23 01:00 | 2026-11-23 03:00 | SL | -1.102 | -27.56 |

Full journal: `data/trades.csv` · every signal: `data/signals.csv`. Paper trades on Deriv public prices, spread deducted; no real or demo orders are placed.
