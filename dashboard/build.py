"""Collect both paper bots' journals into site/data.json for the Netlify dashboard."""
import csv, json, os, time, datetime as dt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rows(p):
    return list(csv.DictReader(open(p, newline=''))) if os.path.exists(p) else []


def bot(name, folder, backtest_r, threshold, two_sided):
    st = json.load(open(os.path.join(folder, 'state.json'))) if os.path.exists(os.path.join(folder, 'state.json')) else {}
    trades = rows(os.path.join(folder, 'trades.csv')); sig = rows(os.path.join(folder, 'signals.csv'))
    latest = {}
    for r in sig: latest[r['symbol']] = r                      # last evaluated bar per symbol
    lat = []
    for s, r in sorted(latest.items()):
        pl = float(r['pred_long']); ps = float(r['pred_short']) if two_sided and r.get('pred_short') not in (None, '') else None
        lat.append({'sym': s, 'bar': r['bar_utc'], 'long': pl, 'short': ps, 'action': r['action']})
    since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=7)).strftime('%Y-%m-%d %H:%M')
    week = [r for r in sig if r['bar_utc'] >= since]
    top = []
    for r in week:
        v = max(float(r['pred_long']), float(r['pred_short']) if two_sided and r.get('pred_short') else -9)
        top.append((v, r['symbol'], r['bar_utc'], r['action']))
    top = sorted(top, reverse=True)[:8]
    eq = []; c = 0.0
    for t in trades:
        c += float(t['R_net']); eq.append({'t': t['exit_time_utc'], 'r': round(c, 3), 'bal': float(t['balance_after'])})
    return {'name': name, 'balance': st.get('balance', 10000), 'started': st.get('started'), 'last_run': st.get('last_run'),
            'open': [dict(sym=k, **v) for k, v in st.get('open', {}).items()], 'trades': trades[-50:][::-1], 'n_trades': len(trades),
            'equity': eq, 'latest': lat, 'top_week': [{'pred': round(a, 3), 'sym': b, 'bar': c_, 'action': d} for a, b, c_, d in top],
            'bars_week': len(week), 'backtest_r': backtest_r, 'threshold': threshold}


whales = rows(os.path.join(ROOT, 'crypto', 'data', 'whales.csv'))
lw = {}
for w in whales: lw[w['symbol']] = w
out = {'built_utc': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M'),
       'forex': bot('Forex ML (H1, 15 pairs)', os.path.join(ROOT, 'data'), 0.12, 0.10, True),
       'crypto': bot('Crypto ML (1h, 10 coins, long only)', os.path.join(ROOT, 'crypto', 'data'), 0.07, 0.20, False),
       'whales': list(lw.values()), 'whale_history': whales[-240:]}
os.makedirs(os.path.join(ROOT, 'site'), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, 'site', 'data.json'), 'w'), separators=(',', ':'))
print('site/data.json written')
