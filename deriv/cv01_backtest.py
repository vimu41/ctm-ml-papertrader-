"""Deriv CV 01 — can technical analysis find an edge on Deriv volatility indices?
Tests rule strategies (Moving averages, Ichimoku, SMC/ICT sweep+FVG+discount, combo) and an ML model
(LightGBM on MA/Ichimoku/RSI/range + SMC features), all with SL 1 ATR / TP 1.5 ATR / max 48 H1 bars.
Break-even win rate at 1.5:1 is 40% before costs. Walk-forward for ML; rules need no training."""
import glob, gzip, os, sys, numpy as np, pandas as pd, lightgbm as lgb
from collections import deque
sys.path.insert(0, '/home/claude/ml'); from v3_smc import smc_features
H = os.path.dirname(os.path.abspath(__file__)); RR, HOLD, COST = 1.5, 48, 0.03

def labels(h, atr):
    hi, lo, c = h.high.values, h.low.values, h.close.values; n = len(c); a = atr.values
    out = np.full((n, 2), np.nan)                     # R for long, short (gross)
    for i in range(n - 1):
        if not a[i] > 0: continue
        for k, sd in enumerate((1, -1)):
            sl, tp, r = c[i] - sd * a[i], c[i] + sd * RR * a[i], None
            for j in range(i + 1, min(n, i + 1 + HOLD)):
                hit_sl = lo[j] <= sl if sd > 0 else hi[j] >= sl; hit_tp = hi[j] >= tp if sd > 0 else lo[j] <= tp
                if hit_sl: r = -1.0; break
                if hit_tp: r = RR; break
            if r is None and i + HOLD < n: r = sd * (c[min(n - 1, i + HOLD)] - c[i]) / a[i]
            if r is not None: out[i, k] = r
    return out

def feats(h):
    c, hi, lo = h.close, h.high, h.low
    tr = pd.concat([hi - lo, (hi - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1); atr = tr.rolling(14).mean()
    f = pd.DataFrame(index=h.index)
    for r in (1, 3, 6, 12, 24, 48, 120): f[f'ret{r}'] = (c - c.shift(r)) / atr
    f['volr'] = atr / tr.rolling(100).mean()
    for p in (20, 50, 200): e = c.ewm(span=p, adjust=False).mean(); f[f'e{p}'] = (c - e) / atr; f[f'es{p}'] = (e - e.shift(5)) / atr
    mid = lambda n: (hi.rolling(n).max() + lo.rolling(n).min()) / 2
    tk, kj = mid(9), mid(26); sa = ((tk + kj) / 2).shift(26); sb = mid(52).shift(26)
    f['cloud'] = (c - (sa + sb) / 2) / atr; f['cloudw'] = (sa - sb) / atr; f['tkkj'] = (tk - kj) / atr
    d = c.diff(); up = d.clip(lower=0).rolling(14).mean(); dn = (-d.clip(upper=0)).rolling(14).mean(); f['rsi'] = 100 - 100 / (1 + up / dn)
    for p in (20, 100): f[f'rng{p}'] = (c - lo.rolling(p).min()) / (hi.rolling(p).max() - lo.rolling(p).min())
    return f, atr, tk, kj, sa, sb

def rules(h, f, tk, kj, sa, sb, S):
    c = h.close; e20, e50, e200 = (c.ewm(span=p, adjust=False).mean() for p in (20, 50, 200))
    up_tr, dn_tr = (e50 > e200) & (c > e200), (e50 < e200) & (c < e200)
    x_up = (c > e20) & (c.shift() <= e20.shift()); x_dn = (c < e20) & (c.shift() >= e20.shift())
    above, below = c > np.maximum(sa, sb), c < np.minimum(sa, sb)
    tkx_up = (tk > kj) & (tk.shift() <= kj.shift()); tkx_dn = (tk < kj) & (tk.shift() >= kj.shift())
    hi, lo = h.high, h.low
    pl48, ph48 = lo.shift(1).rolling(48).min(), hi.shift(1).rolling(48).max()
    swept_l = ((lo < pl48) & (c > pl48)).astype(int).rolling(6).max() > 0      # sell-side liquidity grabbed in last 6 bars
    swept_h = ((hi > ph48) & (c < ph48)).astype(int).rolling(6).max() > 0
    bfvg, sfvg = lo > hi.shift(2), hi < lo.shift(2)                              # fresh fair value gap = displacement
    mss_l, mss_s = c > hi.shift(1).rolling(5).max(), c < lo.shift(1).rolling(5).min()   # market structure shift
    smc_l, smc_s = swept_l & bfvg & mss_l, swept_h & sfvg & mss_s
    conf_l, conf_s = c > hi.shift(1), c < lo.shift(1)
    ote_l = (S.ote_long > 0) & conf_l; ote_s = (S.ote_short > 0) & conf_s
    return {
        'MA trend + EMA20 pullback cross': (up_tr & x_up, dn_tr & x_dn),
        'Ichimoku: above cloud + TK cross': (above & tkx_up, below & tkx_dn),
        'SMC/ICT: sweep + FVG + MSS': (smc_l, smc_s),
        'ICT OTE 62-79% + bar confirmation': (ote_l, ote_s),
        'Combo: MA trend + cloud + OTE': (up_tr & above & ote_l, dn_tr & below & ote_s),
    }

def take(sig_l, sig_s, R):          # one trade at a time per symbol
    out, busy = [], -1
    for i in np.where(sig_l.values | sig_s.values)[0]:
        if i <= busy: continue
        k = 0 if sig_l.values[i] else 1; r = R[i, k]
        if np.isnan(r): continue
        out.append(r - COST); busy = i + HOLD
    return out

def stat(rs):
    rs = np.array(rs)
    if len(rs) < 5: return f'trades {len(rs):5d}'
    return f'trades {len(rs):5d}  win {np.mean(rs > 0) * 100:4.1f}%  avgR {rs.mean():+.3f}  t {rs.mean() / rs.std() * np.sqrt(len(rs)):+5.1f}'

D = []; res = {}
for fn in sorted(glob.glob(f'{H}/history/*_H1.csv.gz')):
    sym = os.path.basename(fn).split('_H1')[0]
    h = pd.read_csv(fn).sort_values('time').reset_index(drop=True)
    if len(h) < 1500: continue
    d1 = h.assign(day=h.time // 86400).groupby('day').agg(time=('time', 'first'), open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last')).reset_index(drop=True)
    d1['time'] = (d1.time // 86400) * 86400
    f, atr, tk, kj, sa, sb = feats(h); S = smc_features(h, d1); R = labels(h, atr)
    for name, (l, s) in rules(h, f, tk, kj, sa, sb, S).items(): res.setdefault(name, []).extend(take(l.fillna(False), s.fillna(False), R))
    rnd = np.random.default_rng(1); m = rnd.random(len(h)) < 0.01
    res.setdefault('Random entries (baseline)', []).extend(take(pd.Series(m & (rnd.random(len(h)) < .5)), pd.Series(m), R))
    X = pd.concat([f, S.drop(columns=[]).add_prefix('smc_')], axis=1)
    X['sym'] = sym; X['time'] = h.time; X['rL'] = R[:, 0]; X['rS'] = R[:, 1]; X['bar'] = np.arange(len(h))
    D.append(X.iloc[250:]); print(sym, len(h), 'bars', pd.to_datetime(h.time.iloc[0], unit='s').date(), '->', pd.to_datetime(h.time.iloc[-1], unit='s').date(), flush=True)

print('\n=== Rule strategies (all indices, SL 1 ATR / TP 1.5 ATR, cost 0.03R) — break-even win ≈ 41%')
for k, v in res.items(): print(f'{k:36s} {stat(v)}')

# ML walk-forward: train on everything before each month, test on that month; causal top-0.5% threshold
D = pd.concat(D, ignore_index=True).dropna(subset=['rL', 'rS']); fc = [c for c in D.columns if c not in ('sym', 'time', 'rL', 'rS', 'bar')]
D['wL'] = (D.rL > 0).astype(int); D['wS'] = (D.rS > 0).astype(int); D['mon'] = pd.to_datetime(D.time, unit='s').dt.to_period('M')
mons = sorted(D.mon.unique()); P = []
params = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=300, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=1)
for m in mons[len(mons) // 3:]:
    te = D[D.mon == m]; tr = D[D.time < te.time.min() - HOLD * 3600]
    if len(tr) < 5000: continue
    r = te[['sym', 'time', 'bar', 'rL', 'rS']].copy()
    for sd in 'LS': r['p' + sd] = lgb.train(params, lgb.Dataset(tr[fc], tr['w' + sd]), 200).predict(te[fc])
    P.append(r)
P = pd.concat(P); P['p'] = P[['pL', 'pS']].max(axis=1); P['r'] = np.where(P.pL >= P.pS, P.rL, P.rS) - COST
from sklearn.metrics import roc_auc_score
print('\n=== ML model (LightGBM, MA + Ichimoku + RSI + range + SMC features), walk-forward by month')
print('AUC long', round(roc_auc_score((P.rL > 0), P.pL), 3), ' AUC short', round(roc_auc_score((P.rS > 0), P.pS), 3), ' (0.50 = no skill)')
S_ = P.sort_values('time'); win, out, busy, thr = deque(), [], {}, None
for t, g in S_.groupby('time'):
    while win and win[0][0] < t - 14 * 86400: win.popleft()
    if len(win) > 50: thr = np.quantile(np.concatenate([a for _, a in win]), 0.995)
    win.append((t, g.p.values))
    if thr is None: continue
    for x in g[g.p >= thr].itertuples():
        if busy.get(x.sym, -1) >= x.bar: continue
        out.append(x.r); busy[x.sym] = x.bar + HOLD
print(f'{"ML top 0.5% (same rule as forex v3)":36s} {stat(out)}')
for q in (0.9, 0.97):
    sel = P[P.p >= P.p.quantile(q)]; print(f'{"ML top " + str(round((1-q)*100)) + "% (no overlap filter)":36s} {stat(sel.r.values)}')
