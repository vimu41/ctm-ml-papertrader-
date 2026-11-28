"""How often does pure randomness produce a result as good as CV 01's backtest?
Simulate 10 random-walk H1 series (no edge possible), run the same 5 rules, keep the best one (as we did).
Repeat many times. CV 01 real backtest: best rule avgR +0.137, t 2.1 over 360 trades."""
import os, sys, numpy as np, pandas as pd
from multiprocessing import Pool
H = os.path.dirname(os.path.abspath(__file__))
ns = {'__file__': os.path.join(H, 'cv01_backtest.py')}
exec(open(os.path.join(H, 'cv01_backtest.py')).read().split("D = []; res = {}")[0], ns)

def one(seed):
    rng = np.random.default_rng(seed); res = {}
    for k in range(10):
        n, sub = 8784, 30
        steps = rng.normal(0, 0.004 / np.sqrt(sub), n * sub)
        p = 1000 * np.exp(np.cumsum(steps)).reshape(n, sub)
        h = pd.DataFrame({'time': np.arange(n) * 3600 + 1_700_000_000, 'open': np.r_[1000, p[:-1, -1]], 'high': p.max(1), 'low': p.min(1), 'close': p[:, -1]})
        h['high'] = np.maximum(h.high, h.open); h['low'] = np.minimum(h.low, h.open)
        d1 = h.assign(day=h.time // 86400).groupby('day').agg(time=('time', 'first'), open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last')).reset_index(drop=True)
        d1['time'] = (d1.time // 86400) * 86400
        f, atr, tk, kj, sa, sb = ns['feats'](h); S = ns['smc_features'](h, d1); R = ns['labels'](h, atr)
        for name, (l, s) in ns['rules'](h, f, tk, kj, sa, sb, S).items():
            res.setdefault(name, []).extend(ns['take'](l.fillna(False), s.fillna(False), R))
    best = max(((np.mean(v), np.mean(v) / np.std(v) * np.sqrt(len(v)), len(v), k) for k, v in res.items() if len(v) > 30))
    return best

if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    with Pool(os.cpu_count()) as P: out = P.map(one, range(N))
    a = np.array([[m, t, n] for m, t, n, _ in out])
    print(f'{N} random worlds · best-of-5 rule: median avgR {np.median(a[:, 0]):+.3f}, median t {np.median(a[:, 1]):+.2f}')
    print(f'share of random worlds where the best rule had avgR >= +0.137: {np.mean(a[:, 0] >= 0.137) * 100:.0f}%')
    print(f'share where best rule had t >= 2.1: {np.mean(a[:, 1] >= 2.1) * 100:.0f}%')
    print('which rule won:', pd.Series([k for *_, k in out]).value_counts().to_dict())
