"""Boom/Crash system test (M1 data, last 45 days, 6 indices).
For BOOM (drifts down, spikes up) — CRASH is the exact mirror:
  A) Drift trade with the trend = SELL Boom. Variants: random times (baseline) vs only when M15 + H1 trend is down
     and price pulled back to M15 EMA20 (trend + pullback). SL above the last 30-min high + 0.5 ATR(M15), TP 1.5x SL, 4h max.
  B) Spike catch = BUY Boom. Variants: random times (baseline) vs only at support (POI): touch of the 4h swing low,
     or a bullish M15 FVG / demand. Exit on the first spike (bar close), SL below the level, or give up after 30 min (small loss).
All results in index points (gross, before spread) and in R (R = distance to stop). The break-even spread is the
average gross points per trade: if your MT5 spread is bigger than that, the system loses."""
import glob, os, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(7)

def ema(x, n): return pd.Series(x).ewm(span=n, adjust=False).mean().values

def run(sym):
    d = pd.read_csv(f'{H}/history/{sym}_M1.csv.gz').sort_values('time').reset_index(drop=True)
    boom = sym.startswith('BOOM'); sg = 1 if boom else -1      # spike direction
    o, h, l, c = (d[k].values for k in ('open', 'high', 'low', 'close')); n = len(c)
    body = (c - o) * sg; med = pd.Series(np.abs(c - o)).rolling(500, min_periods=50).median().values
    spike = body > 8 * med                                             # spike bar in the spike direction
    # higher timeframes built from M1 (only closed data used: shift by one HTF bar)
    t = d.time.values
    def htf(mins):
        g = t // (mins * 60); df = pd.DataFrame({'g': g, 'h': h, 'l': l, 'c': c})
        a = df.groupby('g').agg(h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
        e50, e200, e20 = ema(a.c.values, 50), ema(a.c.values, 200), ema(a.c.values, 20)
        tr = (a.h - a.l).rolling(14).mean().values
        prev = pd.DataFrame({'e50': e50, 'e200': e200, 'e20': e20, 'atr': tr, 'hi': a.h.values, 'lo': a.l.values}, index=a.index).shift(1)
        return prev.reindex(g).values
    M15, H1 = htf(15), htf(60)
    e50_15, e200_15, e20_15, atr15 = M15[:, 0], M15[:, 1], M15[:, 2], M15[:, 3]
    e50_h1, e200_h1 = H1[:, 0], H1[:, 1]
    hi30 = pd.Series(h).rolling(30).max().values; lo30 = pd.Series(l).rolling(30).min().values
    lo240 = pd.Series(l).shift(1).rolling(240).min().values; hi240 = pd.Series(h).shift(1).rolling(240).max().values
    # drift-direction trend (for Boom: down) on M15 and H1
    trend = ((e50_15 - e200_15) * -sg > 0) & ((e50_h1 - e200_h1) * -sg > 0)
    pull = (h >= e20_15) if boom else (l <= e20_15)                    # pulled back to M15 EMA20 against the drift
    conf = ((c < o) if boom else (c > o)) & ~spike                     # M1 candle back with the drift
    out = {}

    def drift(entries, name):
        res, busy = [], -1
        for i in entries:
            if i <= busy or i + 1 >= n or np.isnan(atr15[i]): continue
            e = c[i]; sl = (max(hi30[i], e) + 0.5 * atr15[i]) if boom else (min(lo30[i], e) - 0.5 * atr15[i]); risk = abs(sl - e)
            if risk <= 0: continue
            tp = e - 1.5 * risk * sg; r = None
            for j in range(i + 1, min(n, i + 240)):
                if (h[j] >= sl) if boom else (l[j] <= sl):
                    # a spike jumps over the stop in one tick: MT5 fills at the price after the jump (≈ bar extreme)
                    fill = (h[j] if boom else l[j]) if spike[j] else sl
                    r = -abs(fill - e); break
                if (l[j] <= tp) if boom else (h[j] >= tp): r = 1.5 * risk; break
            if r is None: j = min(n - 1, i + 240); r = (e - c[j]) * sg
            res.append((r, r / risk)); busy = j
        out[name] = res

    def catch(entries, name, lvl):
        res, busy = [], -1
        for i in entries:
            if i <= busy or i + 1 >= n or np.isnan(med[i]): continue
            e = c[i]; stop = (lvl[i] - 3 * med[i]) if boom else (lvl[i] + 3 * med[i])
            if (stop >= e) if boom else (stop <= e): stop = e - 3 * med[i] * sg
            risk = abs(e - stop); r = None
            for j in range(i + 1, min(n, i + 31)):
                if (l[j] <= stop) if boom else (h[j] >= stop): r = -risk; break
                if spike[j]: r = (c[j] - e) * sg; break
            if r is None: j = min(n - 1, i + 30); r = (c[j] - e) * sg
            res.append((r, r / risk)); busy = j
        out[name] = res

    idx = np.arange(300, n)
    drift(rng.choice(idx, 600, replace=False).tolist() and sorted(rng.choice(idx, 600, replace=False)), 'A0 drift, random times')
    drift(idx[trend[idx] & pull[idx] & conf[idx]], 'A1 drift, trend M15+H1 + pullback')
    drift(idx[(~trend[idx]) & pull[idx] & conf[idx]], 'A2 drift, AGAINST trend + pullback')
    catch(sorted(rng.choice(idx, 600, replace=False)), 'B0 spike catch, random times', c - 0 * c)
    sup = lo240 if boom else hi240
    touch = (l <= sup) & (c > sup) if boom else (h >= sup) & (c < sup)
    catch(idx[touch[idx]], 'B1 spike catch at 4h swing low (POI)', sup)
    fvg = (l > np.roll(h, 2)) if boom else (h < np.roll(l, 2))
    return out

def show(rows, label):
    if not rows: print(f'{label:42s} trades    0'); return
    a = np.array(rows); p, R = a[:, 0], a[:, 1]
    print(f'{label:42s} trades {len(a):5d} win {np.mean(p > 0) * 100:5.1f}% avgR {R.mean():+.3f} t {R.mean() / R.std() * np.sqrt(len(R)):+5.1f}  gross pts/trade {p.mean():+8.3f}')

allr = {}
for fn in sorted(glob.glob(f'{H}/history/*_M1.csv.gz')):
    sym = os.path.basename(fn).split('_M1')[0]
    if not sym.startswith(('BOOM', 'CRASH')): continue
    res = run(sym); print(f'\n== {sym}')
    for k, v in res.items(): show(v, k); allr.setdefault(k, []).extend([(r[0], r[1]) for r in v])
print('\n== ALL 6 indices together (R adds up across indices; points do not)')
for k, v in allr.items():
    a = np.array(v); R = a[:, 1]
    print(f'{k:42s} trades {len(a):5d} win {np.mean(R > 0) * 100:5.1f}% avgR {R.mean():+.3f} t {R.mean() / R.std() * np.sqrt(len(R)):+5.1f}  worst {R.min():+.1f}R')
