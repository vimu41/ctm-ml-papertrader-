"""CTM ML v2 paper trader. Runs once per hour (GitHub Actions).

Same rules as the backtest and the MT5 EA (CTM_ML_EA):
  - after each closed H1 bar, predict R for a long and a short
  - if the better side > THRESHOLD and no trade is open on that symbol: enter at the open of the next bar
  - SL = 1 x ATR14, TP = 2 x ATR14, close at the close of the 48th bar if neither is hit
  - result in R minus a realistic Deriv spread; $ result = R x (RISK_PCT of the paper balance)
Writes data/trades.csv (one row per closed trade), data/signals.csv, data/state.json and SUMMARY.md.
"""
import csv, datetime as dt, json, os, sys, time
import engine

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')
SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'EURJPY', 'GBPJPY', 'EURGBP',
           'XAUUSD', 'XAGUSD', 'AUDJPY', 'EURAUD', 'GBPAUD', 'CADJPY']
THRESHOLD, RR, HOLD, MAX_OPEN = 0.10, 2.0, 48, 6
START_BALANCE, RISK_PCT = 10000.0, 0.25
REAL_COST = {'EURUSD': 0.00012, 'GBPUSD': 0.00015, 'USDJPY': 0.015, 'AUDUSD': 0.00015, 'USDCAD': 0.0002,
             'USDCHF': 0.0002, 'EURJPY': 0.025, 'GBPJPY': 0.035, 'EURGBP': 0.0002, 'AUDJPY': 0.025,
             'EURAUD': 0.0003, 'GBPAUD': 0.0004, 'CADJPY': 0.03, 'XAUUSD': 0.35, 'XAGUSD': 0.035}
WS_URLS = ['wss://ws.derivws.com/websockets/v3?app_id=1089', 'wss://api.derivws.com/trading/v1/options/ws/public']
TRADE_COLS = ['symbol', 'side', 'signal_bar_utc', 'entry_time_utc', 'entry', 'SL', 'TP', 'exit_time_utc', 'exit',
              'exit_reason', 'bars_held', 'R_gross', 'cost_R', 'R_net', 'risk_usd', 'pnl_usd', 'balance_after', 'pred']


def utc(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime('%Y-%m-%d %H:%M')


# ---------------------------------------------------------------- data
def fetch_all(symbols):
    import websocket                                   # pip install websocket-client
    last_err = None
    for url in WS_URLS:
        try:
            ws = websocket.create_connection(url, timeout=30)
            out = {}
            for s in symbols:
                out[s] = {}
                for g, cnt in ((3600, 1000), (86400, 300)):
                    ws.send(json.dumps({'ticks_history': 'frx' + s, 'style': 'candles', 'granularity': g,
                                        'count': cnt, 'end': 'latest'}))
                    m = json.loads(ws.recv())
                    if m.get('error'):
                        print(f'{s} {g}: {m["error"].get("message")}'); out[s][g] = None; continue
                    out[s][g] = [{'t': int(c['epoch']), 'o': float(c['open']), 'h': float(c['high']),
                                  'l': float(c['low']), 'c': float(c['close'])} for c in m.get('candles', [])]
                    time.sleep(0.4)
            ws.close()
            return out
        except Exception as e:                         # try the next endpoint
            last_err = e; print('feed error', url, e)
    raise SystemExit(f'no data feed: {last_err}')


# ---------------------------------------------------------------- state
def load_state():
    p = os.path.join(DATA, 'state.json')
    if os.path.exists(p):
        return json.load(open(p))
    return {'balance': START_BALANCE, 'open': {}, 'last_bar': {}, 'started': utc(time.time())}


def append_csv(name, cols, row):
    p = os.path.join(DATA, name); new = not os.path.exists(p)
    with open(p, 'a', newline='') as f:
        w = csv.writer(f)
        if new: w.writerow(cols)
        w.writerow([row[c] for c in cols])


# ---------------------------------------------------------------- trade management
def manage(sym, pos, H1, now, state):
    """walk closed bars from the entry bar; SL checked before TP (same as the backtest)."""
    bars = [b for b in H1 if b['t'] >= pos['entry_t'] and b['t'] + 3600 <= now]
    long_ = pos['side'] == 'BUY'
    for k, b in enumerate(bars, 1):
        hit = None
        if long_:
            if b['l'] <= pos['sl']: hit = ('SL', pos['sl'], -1.0)
            elif b['h'] >= pos['tp']: hit = ('TP', pos['tp'], RR)
        else:
            if b['h'] >= pos['sl']: hit = ('SL', pos['sl'], -1.0)
            elif b['l'] <= pos['tp']: hit = ('TP', pos['tp'], RR)
        if hit is None and k >= HOLD:
            px = b['c']; r = (px - pos['entry']) / pos['atr'] * (1 if long_ else -1)
            hit = ('TIME_48H', px, r)
        if hit:
            reason, px, r = hit
            cost_r = REAL_COST[sym] / pos['atr']; rn = r - cost_r
            risk = pos['risk_usd']; pnl = rn * risk
            state['balance'] = round(state['balance'] + pnl, 2)
            append_csv('trades.csv', TRADE_COLS, {
                'symbol': sym, 'side': pos['side'], 'signal_bar_utc': utc(pos['signal_t']), 'entry_time_utc': utc(pos['entry_t']),
                'entry': pos['entry'], 'SL': pos['sl'], 'TP': pos['tp'], 'exit_time_utc': utc(b['t'] + 3600), 'exit': px,
                'exit_reason': reason, 'bars_held': k, 'R_gross': round(r, 3), 'cost_R': round(cost_r, 3), 'R_net': round(rn, 3),
                'risk_usd': round(risk, 2), 'pnl_usd': round(pnl, 2), 'balance_after': state['balance'], 'pred': pos['pred']})
            print(f'CLOSE {sym} {pos["side"]} {reason} R {rn:+.2f} ${pnl:+.2f}')
            return True
    return False


def run(feed, now):
    os.makedirs(DATA, exist_ok=True)
    model = json.load(open(os.path.join(ROOT, 'model.json')))
    state = load_state()
    # 1) close finished trades
    for sym in list(state['open']):
        H1 = (feed.get(sym) or {}).get(3600)
        if H1 and manage(sym, state['open'][sym], H1, now, state):
            del state['open'][sym]
    # 2) new signals on the latest closed bar
    for sym in SYMBOLS:
        d = feed.get(sym) or {}
        H1, D1 = d.get(3600), d.get(86400)
        if not H1 or not D1:
            continue
        closed = [b for b in H1 if b['t'] + 3600 <= now][-1000:]
        if not closed or closed[-1]['t'] <= state['last_bar'].get(sym, 0):
            continue
        state['last_bar'][sym] = closed[-1]['t']
        if now - closed[-1]['t'] > 3 * 3600:          # market closed / stale feed
            continue
        meta = model['meta'][sym]
        F = engine.features(closed, D1, meta['point'], meta['spr'])
        if F is None:
            continue
        pl, ps = engine.predict(model['L'], F['x']), engine.predict(model['S'], F['x'])
        best = max(pl, ps); side = 'BUY' if pl >= ps else 'SELL'
        action = 'none'
        if best > THRESHOLD:
            nxt = [b for b in H1 if b['t'] == F['t'] + 3600]
            if sym in state['open']: action = 'skip_has_pos'
            elif len(state['open']) >= MAX_OPEN: action = 'skip_max_open'
            elif not nxt: action = 'skip_no_next_bar'
            else:
                e = nxt[0]['o']; a = F['atr']
                sl, tp = (e - a, e + RR * a) if side == 'BUY' else (e + a, e - RR * a)
                sl, tp = round(sl, 6), round(tp, 6)
                state['open'][sym] = {'side': side, 'signal_t': F['t'], 'entry_t': nxt[0]['t'], 'entry': e, 'sl': sl, 'tp': tp,
                                      'atr': a, 'pred': round(best, 3), 'risk_usd': round(state['balance'] * RISK_PCT / 100, 2)}
                action = 'OPEN ' + side
                print(f'OPEN {sym} {side} @ {e} SL {sl:.5f} TP {tp:.5f} pred {best:.3f}')
                # a trade can already be finished if the run was late
                if manage(sym, state['open'][sym], H1, now, state):
                    del state['open'][sym]
        append_csv('signals.csv', ['run_utc', 'symbol', 'bar_utc', 'pred_long', 'pred_short', 'action'],
                   {'run_utc': utc(now), 'symbol': sym, 'bar_utc': utc(F['t']), 'pred_long': round(pl, 4),
                    'pred_short': round(ps, 4), 'action': action})
    state['last_run'] = utc(now)
    json.dump(state, open(os.path.join(DATA, 'state.json'), 'w'), indent=1)
    write_summary(state)


def write_summary(state):
    p = os.path.join(DATA, 'trades.csv')
    rows = list(csv.DictReader(open(p))) if os.path.exists(p) else []
    R = [float(r['R_net']) for r in rows]
    n = len(R); wins = sum(1 for r in R if r > 0)
    eq = peak = dd = 0.0
    for r in R:
        eq += r; peak = max(peak, eq); dd = min(dd, eq - peak)
    lines = ['# CTM ML paper trading - live forward test', '',
             f'Started {state["started"]} UTC · last run {state.get("last_run", "-")} UTC', '',
             '| Metric | Value |', '|---|---|',
             f'| Paper balance | ${state["balance"]:,.2f} (start ${START_BALANCE:,.0f}) |',
             f'| Closed trades | {n} |',
             f'| Win rate | {wins / n * 100:.0f}% |' if n else '| Win rate | - |',
             f'| Average R per trade | {sum(R) / n:+.3f} (backtest: about +0.12) |' if n else '| Average R per trade | - |',
             f'| Total R | {sum(R):+.2f} |', f'| Worst drawdown | {dd:.2f} R |',
             f'| Open trades | {len(state["open"])} |', '']
    if state['open']:
        lines += ['## Open trades', '', '| Symbol | Side | Entry (UTC) | Entry | SL | TP | Pred |', '|---|---|---|---|---|---|---|']
        for s, o in state['open'].items():
            lines.append(f'| {s} | {o["side"]} | {utc(o["entry_t"])} | {o["entry"]} | {o["sl"]:.5f} | {o["tp"]:.5f} | {o["pred"]} |')
        lines.append('')
    if rows:
        lines += ['## Last 20 closed trades', '', '| Symbol | Side | Entry (UTC) | Exit (UTC) | Reason | R | $ |', '|---|---|---|---|---|---|---|']
        for r in rows[-20:][::-1]:
            lines.append(f'| {r["symbol"]} | {r["side"]} | {r["entry_time_utc"]} | {r["exit_time_utc"]} | {r["exit_reason"]} | {r["R_net"]} | {r["pnl_usd"]} |')
    lines += ['', 'Full journal: `data/trades.csv` · every signal: `data/signals.csv`. Paper trades on Deriv public prices, spread deducted; no real or demo orders are placed.']
    open(os.path.join(ROOT, 'SUMMARY.md'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    now = time.time()
    if dt.datetime.fromtimestamp(now, dt.timezone.utc).weekday() == 5:      # Saturday: forex closed
        print('weekend, nothing to do'); sys.exit(0)
    run(fetch_all(SYMBOLS), now)
