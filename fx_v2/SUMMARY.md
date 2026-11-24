# CTM ML paper trading - live forward test

Started 2026-11-10 04:10 UTC · last run 2026-11-24 10:07 UTC

| Metric | Value |
|---|---|
| Paper balance | $9,966.79 (start $10,000) |
| Closed trades | 4 |
| Win rate | 25% |
| Average R per trade | -0.331 (backtest: about +0.12) |
| Total R | -1.33 |
| Worst drawdown | -3.28 R |
| Open trades | 1 |

## Open trades

| Symbol | Side | Entry (UTC) | Entry | SL | TP | Pred |
|---|---|---|---|---|---|---|
| AUDUSD | BUY | 2026-11-23 16:00 | 0.66211 | 0.66060 | 0.66512 | 0.145 |

## Last 20 closed trades

| Symbol | Side | Entry (UTC) | Exit (UTC) | Reason | R | $ |
|---|---|---|---|---|---|---|
| XAGUSD | BUY | 2026-11-23 03:00 | 2026-11-23 14:00 | TP | 1.95 | 48.61 |
| AUDJPY | BUY | 2026-11-23 01:00 | 2026-11-23 13:00 | SL | -1.07 | -26.76 |
| AUDUSD | BUY | 2026-11-23 03:00 | 2026-11-23 13:00 | SL | -1.103 | -27.5 |
| AUDUSD | BUY | 2026-11-23 01:00 | 2026-11-23 03:00 | SL | -1.102 | -27.56 |

Full journal: `data/trades.csv` · every signal: `data/signals.csv`. Paper trades on Deriv public prices, spread deducted; no real or demo orders are placed.
