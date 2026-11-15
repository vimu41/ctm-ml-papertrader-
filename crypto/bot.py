"""CTM crypto paper trader (EXPERIMENTAL). Runs hourly on GitHub Actions, 7 days a week.

Rules (the best walk-forward setup, long only so it fits a spot account):
  - after each closed 1h bar, predict R for a long trade; if > THRESHOLD and no trade open on that coin: buy at next bar open
  - SL = 2 x ATR14, TP = 4 x ATR14, close after 48 bars if neither is hit
  - R minus 0.2% round-trip cost (Binance fees + slippage); $ = R x 0.25% of the paper balance
Also saves an order-book / large-trade ("whale") snapshot per coin each run, to backtest later.
Writes crypto/data/{trades.csv, signals.csv, whales.csv, state.json} and crypto/SUMMARY.md."""
import csv, datetime as dt, json, os, sys, time, urllib.request
os.environ.setdefault('TF', '1h')
import numpy as np, pandas as pd, lightgbm as lgb
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import research as R

DATA = os.path.join(HERE, 'data')
COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'LTCUSDT']
THRESHOLD, SLM, RR, HOLD, MAX_OPEN, COST = 0.20, 2.0, 2.0, 48, 5, 0.002
START_BALANCE, RISK_PCT = 10000.0, 0.25
API = ['https://data-api.binance.vision', 'https://api.binance.com']
TRADE_COLS = ['symbol', 'side', 'signal_bar_utc', 'entry_time_utc', 'entry', 'SL', 'TP', 'exit_time_utc', 'exit', 'exit_reason',
              'bars_held', 'R_gross', 'cost_R', 'R_net', 'risk_usd', 'pnl_usd', 'balance_after', 'pred']


def utc(t): return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime('%Y-%m-%d %H:%M')


def get(path):
    err = None
    for base in API:
        try:
            return json.loads(urllib.request.urlopen(base + path, timeout=30).read())
        except Exception as e:
            err = e
    raise RuntimeError(f'{path}: {err}')


def klines(sym):
    k = get(f'/api/v3/klines?symbol={sym}&interval=1h&limit=1000')
    return pd.DataFrame({'time': [int(r[0]) // 1000 for r in k], 'open': [float(r[1]) for r in k], 'high': [float(r[2]) for r in k],
                         'low': [float(r[3]) for r in k], 'close': [float(r[4]) for r in k], 'volume': [float(r[5]) for r in k],
                         'quote_volume': [float(r[7]) for r in k], 'trades': [float(r[8]) for r in k], 'taker_buy_base': [float(r[9]) for r in k]})


def whale_snapshot(sym, now):
    try:
        d = get(f'/api/v3/depth?symbol={sym}&limit=500'); a = get(f'/api/v3/aggTrades?symbol={sym}&limit=1000')
    except Exception as e:
        print('whale', sym, e); return None
    bids = [(float(p), float(q)) for p, q in d['bids']]; asks = [(float(p), float(q)) for p, q in d['asks']]
    mid = (bids[0][0] + asks[0][0]) / 2
    near = lambda side, pct: sum(p * q for p, q in side if abs(p / mid - 1) <= pct)
    bwall = max(bids, key=lambda x: x[0] * x[1]); awall = max(asks, key=lambda x: x[0] * x[1])
    big = [t for t in a if float(t['p']) * float(t['q']) >= 100000]
    big_buy = sum(float(t['p']) * float(t['q']) for t in big if not t['m']); big_sell = sum(float(t['p']) * float(t['q']) for t in big if t['m'])
    span_min = (a[-1]['T'] - a[0]['T']) / 60000 if len(a) > 1 else 0
    return {'run_utc': utc(now), 'symbol': sym, 'mid': round(mid, 6), 'spread_bp': round((asks[0][0] - bids[0][0]) / mid * 1e4, 2),
            'bid_usd_1pct': round(near(bids, 0.01)), 'ask_usd_1pct': round(near(asks, 0.01)),
            'imbalance_1pct': round((near(bids, 0.01) - near(asks, 0.01)) / max(near(bids, 0.01) + near(asks, 0.01), 1), 3),
            'bid_wall_price': bwall[0], 'bid_wall_usd': round(bwall[0] * bwall[1]), 'ask_wall_price': awall[0], 'ask_wall_usd': round(awall[0] * awall[1]),
            'big_trades': len(big), 'big_buy_usd': round(big_buy), 'big_sell_usd': round(big_sell), 'trades_window_min': round(span_min, 1)}


def append_csv(name, cols, row):
    p = os.path.join(DATA, name); new = not os.path.exists(p)
    with open(p, 'a', newline='') as f:
        w = csv.writer(f)
        if new: w.writerow(cols)
        w.writerow([row[c] for c in cols])


def manage(sym, pos, H, cutoff, state):
    bars = H[(H.time >= pos['entry_t']) & (H.time + 3600 <= cutoff)]
    for k, b in enumerate(bars.itertuples(index=False), 1):
        hit = None
        if b.low <= pos['sl']: hit = ('SL', pos['sl'], -1.0)
        elif b.high >= pos['tp']: hit = ('TP', pos['tp'], RR)
        elif k >= HOLD: hit = ('TIME_48H', b.close, (b.close - pos['entry']) / pos['risk_px'])
        if hit:
            reason, px, r = hit
            cost_r = COST * pos['entry'] / pos['risk_px']; rn = r - cost_r; pnl = rn * pos['risk_usd']
            state['balance'] = round(state['balance'] + pnl, 2)
            append_csv('trades.csv', TRADE_COLS, {
                'symbol': sym, 'side': 'BUY', 'signal_bar_utc': utc(pos['signal_t']), 'entry_time_utc': utc(pos['entry_t']), 'entry': pos['entry'],
                'SL': pos['sl'], 'TP': pos['tp'], 'exit_time_utc': utc(int(b.time) + 3600), 'exit': px, 'exit_reason': reason, 'bars_held': k,
                'R_gross': round(r, 3), 'cost_R': round(cost_r, 3), 'R_net': round(rn, 3), 'risk_usd': pos['risk_usd'], 'pnl_usd': round(pnl, 2),
                'balance_after': state['balance'], 'pred': pos['pred']})
            print(f'CLOSE {sym} {reason} R {rn:+.2f}')
            return True
    return False


def run(now):
    os.makedirs(DATA, exist_ok=True)
    sp = os.path.join(DATA, 'state.json')
    state = json.load(open(sp)) if os.path.exists(sp) else {'balance': START_BALANCE, 'open': {}, 'last_bar': {}, 'started': utc(now)}
    model = lgb.Booster(model_file=os.path.join(HERE, 'model_long.txt')); feats = json.load(open(os.path.join(HERE, 'model_feats.json')))
    raw = {s: klines(s) for s in COINS}
    btc = R.btc_context(raw['BTCUSDT'])
    for s in COINS:
        H = raw[s]; closed_mask = H.time + 3600 <= now
        F = R.features(H.copy(), btc).replace([np.inf, -np.inf], np.nan)
        Hc = H[closed_mask].reset_index(drop=True); Fc = F[closed_mask.values].reset_index(drop=True)
        atr = pd.concat([Hc.high - Hc.low, (Hc.high - Hc.close.shift()).abs(), (Hc.low - Hc.close.shift()).abs()], axis=1).max(axis=1).rolling(14).mean()
        last = state['last_bar'].get(s, 0)
        todo = [i for i in range(len(Hc)) if Hc.time[i] > last][-48:]
        if not last: todo = todo[-1:]
        preds = model.predict(Fc.loc[todo, feats]) if todo else []
        for i, p in zip(todo, preds):
            t = int(Hc.time[i]); state['last_bar'][s] = t
            if s in state['open'] and manage(s, state['open'][s], H, t + 3600, state):
                del state['open'][s]
            action = 'none'
            if p > THRESHOLD:
                nxt = H[H.time == t + 3600]
                if s in state['open']: action = 'skip_has_pos'
                elif len(state['open']) >= MAX_OPEN: action = 'skip_max_open'
                elif not len(nxt): action = 'skip_no_next_bar'
                else:
                    e = float(nxt.open.iloc[0]); rp = float(atr[i]) * SLM
                    state['open'][s] = {'signal_t': t, 'entry_t': t + 3600, 'entry': e, 'sl': round(e - rp, 8), 'tp': round(e + RR * rp, 8),
                                        'risk_px': rp, 'pred': round(float(p), 3), 'risk_usd': round(state['balance'] * RISK_PCT / 100, 2)}
                    action = 'OPEN BUY'; print(f'OPEN {s} @ {e} pred {p:.3f}')
            append_csv('signals.csv', ['run_utc', 'symbol', 'bar_utc', 'pred_long', 'action'],
                       {'run_utc': utc(now), 'symbol': s, 'bar_utc': utc(t), 'pred_long': round(float(p), 4), 'action': action})
        if s in state['open'] and manage(s, state['open'][s], H, now, state):
            del state['open'][s]
        w = whale_snapshot(s, now)
        if w: append_csv('whales.csv', list(w.keys()), w)
        time.sleep(0.3)
    state['last_run'] = utc(now)
    json.dump(state, open(sp, 'w'), indent=1)
    write_summary(state)


def write_summary(state):
    p = os.path.join(DATA, 'trades.csv'); rows = list(csv.DictReader(open(p))) if os.path.exists(p) else []
    Rn = [float(r['R_net']) for r in rows]; n = len(Rn)
    eq = pk = dd = 0.0
    for r in Rn: eq += r; pk = max(pk, eq); dd = min(dd, eq - pk)
    L = ['# CTM crypto paper trading (experimental)', '', f'Started {state["started"]} UTC · last run {state.get("last_run", "-")} UTC', '',
         '| Metric | Value |', '|---|---|', f'| Paper balance | ${state["balance"]:,.2f} (start ${START_BALANCE:,.0f}) |', f'| Closed trades | {n} |',
         f'| Win rate | {sum(r > 0 for r in Rn) / n * 100:.0f}% |' if n else '| Win rate | - |',
         f'| Average R per trade | {sum(Rn) / n:+.3f} (backtest: about +0.07) |' if n else '| Average R per trade | - |',
         f'| Total R | {sum(Rn):+.2f} |', f'| Worst drawdown | {dd:.2f} R |', f'| Open trades | {len(state["open"])} |', '']
    if state['open']:
        L += ['## Open trades', '', '| Coin | Entry (UTC) | Entry | SL | TP | Pred |', '|---|---|---|---|---|---|']
        L += [f'| {s} | {utc(o["entry_t"])} | {o["entry"]} | {o["sl"]} | {o["tp"]} | {o["pred"]} |' for s, o in state['open'].items()]
    open(os.path.join(HERE, 'SUMMARY.md'), 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    run(time.time())
