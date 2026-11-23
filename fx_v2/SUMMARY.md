# CTM ML paper trading - live forward test

Started 2026-11-10 04:10 UTC · last run 2026-11-23 13:07 UTC

| Metric | Value |
|---|---|
| Paper balance | $9,918.18 (start $10,000) |
| Closed trades | 3 |
| Win rate | 0% |
| Average R per trade | -1.092 (backtest: about +0.12) |
| Total R | -3.28 |
| Worst drawdown | -3.28 R |
| Open trades | 1 |

## Open trades

| Symbol | Side | Entry (UTC) | Entry | SL | TP | Pred |
|---|---|---|---|---|---|---|
| XAGUSD | BUY | 2026-11-23 03:00 | 41.5734 | 40.87622 | 42.96776 | 0.101 |

## Last 20 closed trades

| Symbol | Side | Entry (UTC) | Exit (UTC) | Reason | R | $ |
|---|---|---|---|---|---|---|
| AUDJPY | BUY | 2026-11-23 01:00 | 2026-11-23 13:00 | SL | -1.07 | -26.76 |
| AUDUSD | BUY | 2026-11-23 03:00 | 2026-11-23 13:00 | SL | -1.103 | -27.5 |
| AUDUSD | BUY | 2026-11-23 01:00 | 2026-11-23 03:00 | SL | -1.102 | -27.56 |

Full journal: `data/trades.csv` · every signal: `data/signals.csv`. Paper trades on Deriv public prices, spread deducted; no real or demo orders are placed.
