"""Forex v3: does a 'one trade per currency' rule help? Same walk-forward predictions (2021-2026) and same live
selection (top 0.5% of last 14 days, max 10/week, no overlapping trade per pair), plus currency rules:
  base      : current bot (one open trade per pair only)
  per-ccy   : skip a signal if ANY currency of the pair is already in an open trade (e.g. AUDUSD open -> skip AUDJPY)
  same-dir  : skip only if it adds exposure in the SAME direction (long AUDUSD + long AUDJPY = double long AUD -> skip;
              long AUDUSD + short AUDJPY allowed)
Open trades are counted as open for their full 48h window (exit times are not stored; conservative)."""
import numpy as np, pandas as pd
from collections import deque
P = pd.read_pickle('v3_smc_oos_base_v3.pkl')
ccy = lambda s: (s[:3], s[3:])

def run(minp, mode, q=0.995, cap=10):
    S = P.sort_values('time'); win = deque(); thr = None; out = []; busy = {}; wk = {}; openx = []   # (until, sym, {ccy: sign})
    for t, g in S.groupby('time', sort=True):
        while win and win[0][0] < t - 14 * 86400: win.popleft()
        if len(win) > 50: thr = np.quantile(np.concatenate([a for _, a in win]), q)
        win.append((t, g.p.values))
        if thr is None: continue
        openx = [o for o in openx if o[0] > t]
        for r in g[(g.p >= thr) & (g.tp_pips >= minp)].sort_values('p', ascending=False).itertuples():
            if busy.get(r.sym, -1) > t or wk.get(r.week, 0) >= cap: continue
            b, qc = ccy(r.sym); expo = {b: r.side, qc: -r.side}
            if mode == 'per-ccy' and any(c in o[2] for o in openx for c in expo): continue
            if mode == 'same-dir' and any(o[2].get(c) == sgn for o in openx for c, sgn in expo.items()): continue
            out.append((t, r.year, r.r, r.w, r.sym)); busy[r.sym] = t + 48 * 3600; wk[r.week] = wk.get(r.week, 0) + 1
            openx.append((t + 48 * 3600, r.sym, expo))
    return pd.DataFrame(out, columns=['t', 'year', 'r', 'w', 'sym'])

def dd(r):
    eq = np.cumsum(r); return (eq - np.maximum.accumulate(np.r_[0, eq])[1:]).min()

def cluster_losses(T):    # number of times 2+ trades lost within the same 4 hours
    L = T[T.r < 0].t.values; return int(np.sum(np.diff(np.sort(L)) <= 4 * 3600))

print(f'{"targets":10s} {"rule":9s} {"trades":>6s} {"/wk":>4s} {"win%":>6s} {"avgR":>7s} {"t":>5s} {"totalR":>7s} {"maxDD R":>8s} {"clustered losses":>17s}  per-year avgR')
for minp in (0, 50):
    for mode in ('base', 'per-ccy', 'same-dir'):
        T = run(minp, mode); r = T.r.values
        yr = T.groupby('year').r.mean().round(2).to_dict()
        print(f'{"all" if not minp else "50+ pips":10s} {mode:9s} {len(T):6d} {len(T) / 306:4.1f} {T.w.mean() * 100:6.1f} {r.mean():+7.3f} {r.mean() / r.std() * np.sqrt(len(r)):5.1f} {r.sum():+7.1f} {dd(r):8.1f} {cluster_losses(T):17d}  {yr}')
    print()
