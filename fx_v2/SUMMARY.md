# CTM ML paper trading - live forward test

Started 2026-11-10 04:10 UTC · last run 2026-11-23 03:07 UTC

| Metric | Value |
|---|---|
| Paper balance | $9,972.44 (start $10,000) |
| Closed trades | 1 |
| Win rate | 0% |
| Average R per trade | -1.102 (backtest: about +0.12) |
| Total R | -1.10 |
| Worst drawdown | -1.10 R |
| Open trades | 3 |

## Open trades

| Symbol | Side | Entry (UTC) | Entry | SL | TP | Pred |
|---|---|---|---|---|---|---|
| AUDJPY | BUY | 2026-11-23 01:00 | 105.631 | 105.27493 | 106.34314 | 0.116 |
| AUDUSD | BUY | 2026-11-23 03:00 | 0.66353 | 0.66207 | 0.66645 | 0.125 |
| XAGUSD | BUY | 2026-11-23 03:00 | 41.5734 | 40.87622 | 42.96776 | 0.101 |

## Last 20 closed trades

| Symbol | Side | Entry (UTC) | Exit (UTC) | Reason | R | $ |
|---|---|---|---|---|---|---|
| AUDUSD | BUY | 2026-11-23 01:00 | 2026-11-23 03:00 | SL | -1.102 | -27.56 |

Full journal: `data/trades.csv` · every signal: `data/signals.csv`. Paper trades on Deriv public prices, spread deducted; no real or demo orders are placed.
