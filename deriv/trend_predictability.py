"""Can ANY method predict the next trend on Deriv synthetics? Boom/Crash treated as ordinary price charts (spikes included
in price, no spike logic), plus the 10 volatility indices for comparison. H1 data, Nov 2025 - Nov 2026.
Target: direction of price over the next 1h and next 4h. A method has an edge only if its hit rate is clearly above 50%
(t-stat well above 2) and it holds in both halves of the year."""
import glob, os, sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '/home/claude/ml'); from v3_smc import smc_features
H = os.path.dirname(os.path.abspath(__file__))

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def build(h):
    c, hi, lo = h.close, h.high, h.low
    tr = pd.concat([hi - lo, (hi - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1); atr = tr.rolling(14).mean()
    t = pd.to_datetime(h.time, unit='s'); X = pd.DataFrame(index=h.index)
    X['dow'] = t.dt.dayofweek; X['hour'] = t.dt.hour
    for r in (1, 3, 6, 12, 24, 48): X[f'mom{r}'] = (c - c.shift(r)) / atr
    for p in (20, 50, 200): X[f'ema{p}'] = (c - ema(c, p)) / atr
    X['ema50_200'] = (ema(c, 50) - ema(c, 200)) / atr
    d = c.diff(); X['rsi'] = 100 - 100 / (1 + d.clip(lower=0).rolling(14).mean() / (-d.clip(upper=0)).rolling(14).mean())
    mid = lambda n: (hi.rolling(n).max() + lo.rolling(n).min()) / 2
    tk, kj = mid(9), mid(26); sa = ((tk + kj) / 2).shift(26); sb = mid(52).shift(26)
    X['cloud'] = (c - (sa + sb) / 2) / atr; X['tkkj'] = (tk - kj) / atr
    # structure: break of structure = close beyond the previous 24-bar high/low
    X['bos'] = np.where(c > hi.shift(1).rolling(24).max(), 1, np.where(c < lo.shift(1).rolling(24).min(), -1, 0))
    d1 = h.assign(day=h.time // 86400).groupby('day').agg(time=('time', 'first'), open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last')).reset_index(drop=True)
    d1['time'] = (d1.time // 86400) * 86400
    S = smc_features(h, d1); X = pd.concat([X, S.add_prefix('smc_')], axis=1)
    for k in (1, 4): X[f'y{k}'] = np.sign(c.shift(-k) - c)
    X['time'] = h.time
    return X

def hit(sig, y):          # sig in {-1,0,+1}; hit rate where sig != 0
    m = (sig != 0) & (y != 0) & np.isfinite(sig) & np.isfinite(y)
    n = int(m.sum());
    if n < 30: return n, np.nan, np.nan
    p = (np.sign(sig[m]) == y[m]).mean(); return n, p * 100, (p - 0.5) / np.sqrt(0.25 / n)

RULES = {
    'Time: weekday (best-day rule, in-sample)': None, 'Time: hour of day (best-hour rule, in-sample)': None,
    'Momentum 24h (follow the last day)': lambda X: np.sign(X.mom24),
    'Momentum 3h (follow the last 3 bars)': lambda X: np.sign(X.mom3),
    'Mean reversion 3h (fade the last 3 bars)': lambda X: -np.sign(X.mom3),
    'EMA trend (50 vs 200)': lambda X: np.sign(X.ema50_200),
    'Price vs EMA20': lambda X: np.sign(X.ema20),
    'RSI (above 50 = up)': lambda X: np.sign(X.rsi - 50),
    'RSI extremes (fade >70 / <30)': lambda X: np.where(X.rsi > 70, -1, np.where(X.rsi < 30, 1, 0)),
    'Ichimoku (side of cloud)': lambda X: np.sign(X.cloud),
    'Ichimoku Tenkan vs Kijun': lambda X: np.sign(X.tkkj),
    'SMC break of structure (follow BOS)': lambda X: X.bos.astype(float),
    'SMC liquidity sweep (fade the sweep)': lambda X: np.where(X.smc_sweep_high > 0, -1, np.where(X.smc_sweep_low > 0, 1, 0)),
    'ICT OTE (trade back with the swing)': lambda X: np.where(X.smc_ote_long > 0, 1, np.where(X.smc_ote_short > 0, -1, 0)),
    'FVG (follow the newest gap)': lambda X: np.sign(X.smc_bull_fvg_20 - X.smc_bear_fvg_20),
}

def evaluate(group, files):
    D = []
    for fn in files:
        sym = os.path.basename(fn).split('_H1')[0]; h = pd.read_csv(fn).sort_values('time').reset_index(drop=True)
        X = build(h).iloc[250:-5]; X['sym'] = sym; D.append(X)
    D = pd.concat(D, ignore_index=True); half = D.time >= D.time.median()
    print(f'\n######## {group}: {D.sym.nunique()} indices, {len(D):,} H1 bars')
    print(f'{"method":46s} {"next 1h":>22s} {"next 4h":>22s}   {"4h: 1st half / 2nd half":>24s}')
    for name, f in RULES.items():
        if f is None:
            key = 'dow' if 'weekday' in name else 'hour'
            # direction per weekday/hour learned on the 1st half, tested on the 2nd half (fair test)
            res = []
            for k in (1, 4):
                tr = D[~half].groupby(key)[f'y{k}'].mean(); sig = np.sign(D[key].map(tr)).values
                res.append(hit(sig[half.values], D[f'y{k}'].values[half.values]))
            print(f'{name:46s} ' + ' '.join(f'{p:5.1f}% (t{t:+5.1f}, n{n:6d})' for n, p, t in res) + '   (learned 1st half, tested 2nd)')
            continue
        sig = np.asarray(f(D), dtype=float); out = [hit(sig, D[f'y{k}'].values) for k in (1, 4)]
        h1 = hit(sig[~half.values], D.y4.values[~half.values]); h2 = hit(sig[half.values], D.y4.values[half.values])
        print(f'{name:46s} ' + ' '.join(f'{p:5.1f}% (t{t:+5.1f}, n{n:6d})' for n, p, t in out) + f'   {h1[1]:5.1f}% / {h2[1]:5.1f}%')
    # ML: everything together, trained on months before, tested month by month
    fc = [c for c in D.columns if c not in ('y1', 'y4', 'time', 'sym')]; D['mon'] = pd.to_datetime(D.time, unit='s').dt.to_period('M')
    P = []
    for m in sorted(D.mon.unique())[4:]:
        te = D[D.mon == m]; tr = D[(D.time < te.time.min() - 4 * 3600) & (D.y4 != 0)]
        mdl = lgb.train(dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=300, feature_fraction=0.7, verbose=-1, seed=1),
                        lgb.Dataset(tr[fc], (tr.y4 > 0).astype(int)), 200)
        P.append(pd.DataFrame({'p': mdl.predict(te[fc]), 'y': te.y4.values}))
    P = pd.concat(P); P = P[P.y != 0]
    auc = roc_auc_score(P.y > 0, P.p); top = P[(P.p - 0.5).abs() >= (P.p - 0.5).abs().quantile(0.9)]
    n, p, t = hit(np.sign(top.p - 0.5).values, top.y.values)
    print(f'{"ML (all of the above, walk-forward), next 4h":46s} AUC {auc:.3f} (0.500 = no skill) · most confident 10%: {p:.1f}% hit (t{t:+.1f}, n{n})')

evaluate('BOOM / CRASH (as plain charts)', sorted(glob.glob(f'{H}/history/BOOM*_H1.csv.gz') + glob.glob(f'{H}/history/CRASH*_H1.csv.gz')))
evaluate('VOLATILITY indices', sorted(glob.glob(f'{H}/history/R_*_H1.csv.gz') + glob.glob(f'{H}/history/1HZ*_H1.csv.gz')))
