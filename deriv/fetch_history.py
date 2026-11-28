"""Download H1 candle history for Deriv synthetic (volatility) indices, 2021 -> now.
Runs on GitHub Actions (the Deriv websocket is reachable there). Output: deriv/history/<SYM>_H1.csv.gz"""
import csv, gzip, json, os, time, websocket
SYMS = ['R_10', 'R_25', 'R_50', 'R_75', 'R_100', '1HZ10V', '1HZ25V', '1HZ50V', '1HZ75V', '1HZ100V']
URLS = ['wss://ws.derivws.com/websockets/v3?app_id=1089', 'wss://api.derivws.com/trading/v1/options/ws/public']
START = 1609459200  # 2021-01-01
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'history'); os.makedirs(OUT, exist_ok=True)

def conn():
    for u in URLS:
        try: return websocket.create_connection(u, timeout=60)
        except Exception as e: print('connect fail', u, e)
    raise SystemExit('no deriv connection')

ws = conn()
for s in SYMS:
    rows, end = {}, 'latest'
    for _ in range(40):
        try:
            ws.send(json.dumps({'ticks_history': s, 'style': 'candles', 'granularity': 3600, 'count': 5000, 'end': end}))
            m = json.loads(ws.recv())
        except Exception as e:
            print(s, 'retry', e); time.sleep(3); ws = conn(); continue
        if m.get('error'): print(s, m['error'].get('message')); break
        c = m.get('candles') or []
        if not c: break
        for k in c: rows[int(k['epoch'])] = (k['open'], k['high'], k['low'], k['close'])
        first = int(c[0]['epoch'])
        if first <= START or len(c) < 10: break
        end = first - 1; time.sleep(0.3)
    if not rows: continue
    with gzip.open(f'{OUT}/{s}_H1.csv.gz', 'wt', newline='') as f:
        w = csv.writer(f); w.writerow(['time', 'open', 'high', 'low', 'close'])
        for t in sorted(rows):
            if t >= START: w.writerow([t, *rows[t]])
    print(s, len(rows), 'bars from', time.strftime('%Y-%m-%d', time.gmtime(min(rows))), flush=True)
