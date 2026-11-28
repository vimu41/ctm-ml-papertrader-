"""Download H1 candle history for Deriv synthetic (volatility) indices, 2021 -> now.
Runs on GitHub Actions (the Deriv websocket is reachable there). Output: deriv/history/<SYM>_H1.csv.gz"""
import sys, csv, gzip, json, os, time, websocket
GRAN = int(os.environ.get('GRAN', '3600')); DAYS = int(os.environ.get('DAYS', '0'))
SYMS = sys.argv[1:] or ['R_10', 'R_25', 'R_50', 'R_75', 'R_100', '1HZ10V', '1HZ25V', '1HZ50V', '1HZ75V', '1HZ100V']
URLS = ['wss://ws.derivws.com/websockets/v3?app_id=1089', 'wss://api.derivws.com/trading/v1/options/ws/public']
START = int(time.time()) - DAYS * 86400 if os.environ.get('DAYS') else 1609459200  # default 2021-01-01
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'history'); os.makedirs(OUT, exist_ok=True)

def conn():
    for u in URLS:
        try: return websocket.create_connection(u, timeout=60)
        except Exception as e: print('connect fail', u, e)
    raise SystemExit('no deriv connection')

TF = {60: 'M1', 300: 'M5', 900: 'M15', 3600: 'H1'}[GRAN]
ws = conn()
for s in SYMS:
    rows, end = {}, 'latest'
    for _ in range(120):
        try:
            ws.send(json.dumps({'ticks_history': s, 'style': 'candles', 'granularity': GRAN, 'count': 5000, 'end': end}))
            m = json.loads(ws.recv())
        except Exception as e:
            print(s, 'retry', e); time.sleep(3); ws = conn(); continue
        if m.get('error'):
            msg = m['error'].get('message', '')
            if 'rate limit' in msg.lower(): print(s, 'rate limit, waiting 65s', flush=True); time.sleep(65); continue
            print(s, msg); break
        c = m.get('candles') or []
        if not c: break
        for k in c: rows[int(k['epoch'])] = (k['open'], k['high'], k['low'], k['close'])
        first = int(c[0]['epoch'])
        if first <= START or len(c) < 10: break
        end = first - 1; time.sleep(2)
    if not rows: continue
    with gzip.open(f'{OUT}/{s}_{TF}.csv.gz', 'wt', newline='') as f:
        w = csv.writer(f); w.writerow(['time', 'open', 'high', 'low', 'close'])
        for t in sorted(rows):
            if t >= START: w.writerow([t, *rows[t]])
    print(s, len(rows), 'bars from', time.strftime('%Y-%m-%d', time.gmtime(min(rows))), flush=True)
