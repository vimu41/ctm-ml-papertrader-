"""Deriv CV 01 rule (ICT OTE + confirmation, SL 1 ATR, TP 1.5 ATR, 48h) run unchanged on Boom/Crash H1 (Nov 2025-Nov 2026).
Two stop-fill assumptions, because a spike can jump over the stop inside a bar:
  optimistic  = filled exactly at the SL
  realistic   = if the bar that hits the SL is a spike bar on the stop side, filled at that bar's extreme"""
import glob, os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import bot as DB
RR, HOLD, COST = 1.5, 48, 0.03
def sim(h, realistic):
    B = [{'t': r.time, 'o': r.open, 'h': r.high, 'l': r.low, 'c': r.close} for r in h.itertuples()]
    o, hi, lo, c = h.open.values, h.high.values, h.low.values, h.close.values; n = len(c)
    rngbar = hi - lo; med = pd.Series(rngbar).rolling(200, min_periods=20).median().values
    out, busy = [], -1
    for i in range(60, n - 1):
        if i <= busy: continue
        s, info = DB.signal(B, i)
        if not s: continue
        e, a, sd = o[i + 1], info['atr'], (1 if s == 'BUY' else -1)
        sl, tp = e - sd * a, e + sd * RR * a; r = None
        for j in range(i + 1, min(n, i + 1 + HOLD)):
            hit_sl = lo[j] <= sl if sd > 0 else hi[j] >= sl; hit_tp = hi[j] >= tp if sd > 0 else lo[j] <= tp
            if hit_sl:
                fill = sl
                if realistic and rngbar[j] > 3 * med[j]: fill = lo[j] if sd > 0 else hi[j]   # spike jumped the stop
                r = (fill - e) * sd / a; break
            if hit_tp: r = RR; break
        if r is None: j = min(n - 1, i + HOLD); r = (c[j] - e) * sd / a
        out.append((s, r - COST)); busy = j
    return out
print(f'{"index":10s} {"trades":>6s} {"BUY/SELL":>9s}  {"optimistic fill":>26s}  {"realistic (spike) fill":>32s}')
tot = {False: [], True: []}
for fn in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'history', '*_H1.csv.gz'))):
    sym = os.path.basename(fn).split('_H1')[0]
    if not sym.startswith(('BOOM', 'CRASH')): continue
    h = pd.read_csv(fn).sort_values('time').reset_index(drop=True); line = f'{sym:10s}'
    for real in (False, True):
        T = sim(h, real); R = np.array([r for _, r in T]); tot[real] += list(R)
        if not real: line += f' {len(T):6d} {sum(s == "BUY" for s, _ in T):4d}/{sum(s == "SELL" for s, _ in T):<4d}'
        line += f'  win {np.mean(R > 0) * 100:4.1f}% avgR {R.mean():+.3f} worst {R.min():+5.1f}R'
    print(line)
for real in (False, True):
    R = np.array(tot[real]); print(f'ALL {"realistic " if real else "optimistic"}: trades {len(R)} win {np.mean(R > 0) * 100:.1f}% avgR {R.mean():+.3f} t {R.mean() / R.std() * np.sqrt(len(R)):+.1f}')
