# CTM ML v3 paper trading - live forward test (risk 1 ATR, target 1.5 ATR)

Started 2026-11-10 04:10 UTC · last run 2026-11-23 10:07 UTC

| Metric | Value |
|---|---|
| Paper balance | $10,036.24 (start $10,000) |
| Closed trades | 1 |
| Win rate | 100% |
| Average R per trade | +1.450 (v3 backtest: about +0.03, win 47%) |
| Total R | +1.45 |
| Worst drawdown | 0.00 R |
| Open trades | 2 |

## Open trades

| Symbol | Side | Entry (UTC) | Entry | SL | TP | Pred |
|---|---|---|---|---|---|---|
| AUDJPY | BUY | 2026-11-23 01:00 | 105.631 | 105.27493 | 106.16511 | 0.469 |
| AUDUSD | BUY | 2026-11-23 04:00 | 0.66352 | 0.66207 | 0.66569 | 0.466 |

## Last 20 closed trades

| Symbol | Side | Entry (UTC) | Exit (UTC) | Reason | R | $ |
|---|---|---|---|---|---|---|
| XAGUSD | BUY | 2026-11-23 03:00 | 2026-11-23 06:00 | TP | 1.45 | 36.24 |

Full journal: `data/trades.csv` · every signal: `data/signals.csv`. Paper trades on Deriv public prices, spread deducted; no real or demo orders are placed.
