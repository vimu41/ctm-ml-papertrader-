# CTM ML paper trader

Hourly forward test of the CTM ML v2 model (LightGBM, forex + metals, H1), run by GitHub Actions.
No money and no broker account involved: trades are "paper" trades on Deriv's public price feed.

- Forex v3 rules (from 2026-11-19): LightGBM win-probability model; trade when P is in the top 0.5% of the last 14 days, max 10 new trades a week, SL 1×ATR14, TP 1.5×ATR14 ($2 risk : $3 target), close after 48 H1 bars, no trades from 20:00-22:00 UTC bars. v2 keeps running in fx_v2/ for comparison.
- Paper balance $10,000, risk 0.25% per trade, realistic Deriv spread deducted from every trade.
- Results: see **SUMMARY.md**, full journal `data/trades.csv`, every signal `data/signals.csv`.
