# CTM ML paper trader

Hourly forward test of the CTM ML v2 model (LightGBM, forex + metals, H1), run by GitHub Actions.
No money and no broker account involved: trades are "paper" trades on Deriv's public price feed.

- Rules: predicted R > 0.10 → enter next bar open, SL 1×ATR14, TP 2×ATR14, close after 48 H1 bars. Max 6 open, one per symbol. No new trades from the 20:00-22:00 UTC bars (rollover spreads).
- Paper balance $10,000, risk 0.25% per trade, realistic Deriv spread deducted from every trade.
- Results: see **SUMMARY.md**, full journal `data/trades.csv`, every signal `data/signals.csv`.
