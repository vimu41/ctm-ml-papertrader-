"""Deriv CV 01 paper trader — ICT OTE rule on Deriv volatility indices. Runs once per hour, every day.

Rule (the only one of 5 technical rules that beat random entries in the Nov 2025 - Nov 2026 backtest:
360 trades, win 46.7%, +0.14R after costs; break-even at 1.5:1 is ~41%; may still be luck, hence this forward test):
  - swing = highest high / lowest low of the last 48 closed H1 bars; up swing if the high came after the low
  - BUY  when the last closed bar sits in the 62-79% retracement (OTE) of an UP swing and closes above the
         previous bar's high (confirmation); SELL is the mirror (down swing, close below previous bar's low)
  - entry = next bar's open, SL = 1 x ATR14, TP = 1.5 x ATR14, close at the 48th bar if neither is hit
  - one trade per index at a time, risk 0.25% of the paper balance, cost 0.03R per trade
Writes deriv/data/trades.csv, signals.csv, state.json and deriv/SUMMARY.md. Paper only; no orders are sent.
"""
import csv, datetime as dt, json, os, time

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')
SYMBOLS = ['R_10', 'R_25', 'R_50', 'R_75', 'R_100', '1HZ10V', '1HZ25V', '1HZ50V', '1HZ75V', '1HZ100V']
NAMES = {'R_10': 'Volatility 10', 'R_25': 'Volatility 25', 'R_50': 'Volatility 50', 'R_75': 'Volatility 75',
         'R_100': 'Volatility 100', '1HZ10V': 'Volatility 10 (1s)', '1HZ25V': 'Volatility 25 (1s)',
         '1HZ50V': 'Volatility 50 (1s)', '1HZ75V': 'Volatility 75 (1s)', '1HZ100V': 'Volatility 100 (1s)'}
RR, HOLD, SW, COST_R = 1.5, 48, 48, 0.03
OTE_LO, OTE_HI = 0.62, 0.79
START_BALANCE, RISK_PCT = 10000.0, 0.25
WS_URLS = ['wss://ws.derivws.com/websockets/v3?app_id=1089', 'wss://api.derivws.com/trading/v1/options/ws/public']
TRADE_COLS = ['symbol', 'side', 'signal_bar_utc', 'entry_time_utc', 'entry', 'SL', 'TP', 'exit_time_utc', 'exit',
              'exit_reason', 'bars_held', 'R_gross', 'cost_R', 'R_net', 'risk_usd', 'pnl_usd', 'balance_after', 'pred']
SIG_COLS = ['run_utc', 'symbol', 'bar_utc', 'pred_long', 'pred_short', 'swing', 'retr', 'action']


def utc(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime('%Y-%m-%d %H:%M')


def fetch_all():
    import websocket
    last = None
    for url in WS_URLS:
        try:
            ws = websocket.create_connection(url, timeout=30); out = {}
            for s in SYMBOLS:
                ws.send(json.dumps({'ticks_history': s, 'style': 'candles', 'granularity': 3600, 'count': 300, 'end': 'latest'}))
                m = json.loads(ws.recv())
                if m.get('error'):
                    print(s, m['error'].get('message')); out[s] = None; time.sleep(2); continue
                out[s] = [{'t': int(c['epoch']), 'o': float(c['open']), 'h': float(c['high']), 'l': float(c['low']),
                           'c': float(c['close'])} for c in m.get('candles', [])]
                time.sleep(0.5)
            ws.close(); return out
        except Exception as e:
            last = e; print('feed error', url, e)
    raise SystemExit(f'no data feed: {last}')


def signal(B, i):
    """OTE rule on closed bar i. Returns (side or None, info dict)."""
    if i < max(SW, 15): return None, None
    tr = [max(B[k]['h'] - B[k]['l'], abs(B[k]['h'] - B[k - 1]['c']), abs(B[k]['l'] - B[k - 1]['c'])) for k in range(i - 13, i + 1)]
    atr = sum(tr) / 14
    win = B[i - SW + 1:i + 1]
    hh = max(b['h'] for b in win); ll = min(b['l'] for b in win)
    ih = next(k for k, b in enumerate(win) if b['h'] == hh); il = next(k for k, b in enumerate(win) if b['l'] == ll)
    up = ih > il; rng = hh - ll
    if not (rng > 0 and atr > 0): return None, None
    c = B[i]['c']; retr = (hh - c) / rng if up else (c - ll) / rng
    info = {'atr': atr, 'up': up, 'retr': retr, 'hh': hh, 'll': ll}
    in_ote = OTE_LO <= retr <= OTE_HI
    if up and in_ote and c > B[i - 1]['h']: return 'BUY', info
    if (not up) and in_ote and c < B[i - 1]['l']: return 'SELL', info
    return None, info


def load_state():
    p = os.path.join(DATA, 'state.json')
    return json.load(open(p)) if os.path.exists(p) else {'balance': START_BALANCE, 'open': {}, 'last_bar': {}, 'started': utc(time.time())}


def append_csv(name, cols, row):
    p = os.path.join(DATA, name); new = not os.path.exists(p)
    with open(p, 'a', newline='') as f:
        w = csv.writer(f)
        if new: w.writerow(cols)
        w.writerow([row[c] for c in cols])


def manage(sym, pos, B, now, state):
    bars = [b for b in B if b['t'] >= pos['entry_t'] and b['t'] + 3600 <= now]
    lg = pos['side'] == 'BUY'
    for k, b in enumerate(bars, 1):
        hit = None
        if lg:
            if b['l'] <= pos['sl']: hit = ('SL', pos['sl'], -1.0)
            elif b['h'] >= pos['tp']: hit = ('TP', pos['tp'], RR)
        else:
            if b['h'] >= pos['sl']: hit = ('SL', pos['sl'], -1.0)
            elif b['l'] <= pos['tp']: hit = ('TP', pos['tp'], RR)
        if hit is None and k >= HOLD:
            hit = ('TIME_48H', b['c'], (b['c'] - pos['entry']) / pos['atr'] * (1 if lg else -1))
        if hit:
            reason, px, r = hit; rn = r - COST_R; pnl = rn * pos['risk_usd']
            state['balance'] = round(state['balance'] + pnl, 2)
            append_csv('trades.csv', TRADE_COLS, {
                'symbol': sym, 'side': pos['side'], 'signal_bar_utc': utc(pos['signal_t']), 'entry_time_utc': utc(pos['entry_t']),
                'entry': pos['entry'], 'SL': pos['sl'], 'TP': pos['tp'], 'exit_time_utc': utc(b['t'] + 3600), 'exit': px,
                'exit_reason': reason, 'bars_held': k, 'R_gross': round(r, 3), 'cost_R': COST_R, 'R_net': round(rn, 3),
                'risk_usd': pos['risk_usd'], 'pnl_usd': round(pnl, 2), 'balance_after': state['balance'], 'pred': pos['pred']})
            print(f'CLOSE {sym} {pos["side"]} {reason} R {rn:+.2f} ${pnl:+.2f}')
            return True
    return False


def run(feed, now):
    os.makedirs(DATA, exist_ok=True); state = load_state()
    for sym in SYMBOLS:
        B = feed.get(sym)
        if not B: continue
        closed = [b for b in B if b['t'] + 3600 <= now]
        last = state['last_bar'].get(sym, 0)
        todo = [k for k, b in enumerate(closed) if b['t'] > last][-48:]
        if not last: todo = todo[-1:]
        for k in todo:
            bar = closed[k]; state['last_bar'][sym] = bar['t']
            if sym in state['open'] and manage(sym, state['open'][sym], B, bar['t'] + 3600, state):
                del state['open'][sym]
            side, info = signal(closed, k)
            if info is None: continue
            action = 'none'
            if side:
                nxt = [b for b in B if b['t'] == bar['t'] + 3600]
                if sym in state['open']: action = 'skip_has_pos'
                elif not nxt: action = 'skip_no_next_bar'
                else:
                    e = nxt[0]['o']; a = info['atr']
                    sl, tp = (e - a, e + RR * a) if side == 'BUY' else (e + a, e - RR * a)
                    state['open'][sym] = {'side': side, 'signal_t': bar['t'], 'entry_t': nxt[0]['t'], 'entry': e,
                                          'sl': round(sl, 4), 'tp': round(tp, 4), 'atr': a, 'pred': round(info['retr'], 3),
                                          'risk_usd': round(state['balance'] * RISK_PCT / 100, 2)}
                    action = 'OPEN ' + side; print(f'OPEN {sym} {side} @ {e} SL {sl:.4f} TP {tp:.4f} retr {info["retr"]:.2f}')
            r = round(info['retr'], 4)
            append_csv('signals.csv', SIG_COLS, {'run_utc': utc(now), 'symbol': sym, 'bar_utc': utc(bar['t']),
                       'pred_long': r if info['up'] else 0, 'pred_short': r if not info['up'] else 0,
                       'swing': 'up' if info['up'] else 'down', 'retr': r, 'action': action})
        if sym in state['open'] and manage(sym, state['open'][sym], B, now, state):
            del state['open'][sym]
    state['last_run'] = utc(now)
    json.dump(state, open(os.path.join(DATA, 'state.json'), 'w'), indent=1)
    summary(state)


def summary(state):
    p = os.path.join(DATA, 'trades.csv')
    rows = list(csv.DictReader(open(p, newline=''))) if os.path.exists(p) else []
    R = [float(r['R_net']) for r in rows]; n = len(R)
    L = ['# Deriv CV 01 — ICT OTE paper bot (volatility indices, risk 1 ATR : target 1.5 ATR)', '',
         f'Started {state["started"]} UTC · last run {state.get("last_run", "-")} UTC', '',
         '| Metric | Value |', '|---|---|', f'| Paper balance | ${state["balance"]:,.2f} |', f'| Closed trades | {n} |',
         f'| Win rate | {sum(r > 0 for r in R) / n * 100:.0f}% (backtest 46.7%, break-even ~41%) |' if n else '| Win rate | - |',
         f'| Average R | {sum(R) / n:+.3f} (backtest +0.14) |' if n else '| Average R | - |', f'| Open trades | {len(state["open"])} |', '']
    for r in rows[-20:][::-1]:
        L.append(f'- {r["symbol"]} {r["side"]} {r["entry_time_utc"]} → {r["exit_reason"]} {r["R_net"]}R ${r["pnl_usd"]}')
    open(os.path.join(ROOT, 'SUMMARY.md'), 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    run(fetch_all(), time.time())
