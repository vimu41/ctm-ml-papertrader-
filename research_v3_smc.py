"""Does adding SMC-style features (Fib/OTE, liquidity, sweeps, order blocks/POI, FVG) improve the v3 model?
Same labels, same walk-forward (2021-2026), same real-time selection rule as the live bot:
adaptive top-0.5% threshold over the last 14 days, max 10 trades/week, one per symbol, no rollover hours.
Compares: base v3 features vs base + SMC features, all targets and >= 50-pip targets."""
import os, numpy as np, pandas as pd, lightgbm as lgb
from collections import deque
os.environ['CTM_TF'] = 'H1'
import ctm_ml_v2 as M
HERE = os.path.dirname(os.path.abspath(__file__))
SW = 48            # swing window (H1 bars)


def smc_features(h, d1):
    o, hi, lo, c = h.open, h.high, h.low, h.close
    tr = pd.concat([hi - lo, (hi - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean(); f = pd.DataFrame(index=h.index)
    # --- Fib / OTE on the last SW-bar swing
    hh = hi.rolling(SW).max(); ll = lo.rolling(SW).min(); rng = (hh - ll).replace(0, np.nan)
    ih = hi.rolling(SW).apply(np.argmax, raw=True); il = lo.rolling(SW).apply(np.argmin, raw=True)
    up = ih > il                                             # high came after low -> up swing, retracing down
    retr = np.where(up, (hh - c) / rng, (c - ll) / rng)      # 0 = at the swing end, 1 = full retracement
    f['swing_up'] = up.astype(float); f['retr'] = retr; f['swing_atr'] = rng / atr
    f['ote_long'] = (up & (retr >= 0.62) & (retr <= 0.79)).astype(float)
    f['ote_short'] = ((~up) & (retr >= 0.62) & (retr <= 0.79)).astype(float)
    # --- liquidity: previous day high/low (only fully closed days), equal highs/lows, sweeps
    dd = pd.DataFrame({'time': d1.time + 86400, 'pdh': d1.high, 'pdl': d1.low}).sort_values('time')
    m = pd.merge_asof(h[['time']].reset_index(), dd, on='time', direction='backward').set_index('index')
    f['pdh_dist'] = (m.pdh.values - c) / atr; f['pdl_dist'] = (c - m.pdl.values) / atr
    tol = 0.15 * atr
    f['eq_highs'] = sum(((hh - hi.shift(k)).abs() < tol).astype(float) for k in range(1, SW))   # highs sitting at the top
    f['eq_lows'] = sum(((lo.shift(k) - ll).abs() < tol).astype(float) for k in range(1, SW))
    f['sweep_pdh'] = ((hi > m.pdh.values) & (c < m.pdh.values)).rolling(3).max()
    f['sweep_pdl'] = ((lo < m.pdl.values) & (c > m.pdl.values)).rolling(3).max()
    prev_hh = hi.shift(1).rolling(SW).max(); prev_ll = lo.shift(1).rolling(SW).min()
    f['sweep_high'] = ((hi > prev_hh) & (c < prev_hh)).rolling(3).max()
    f['sweep_low'] = ((lo < prev_ll) & (c > prev_ll)).rolling(3).max()
    # --- order blocks (POI): last opposite candle before a displacement of > 1.5 ATR within 3 bars
    disp_up = (c.shift(-3) - c) > 1.5 * atr       # NOTE: uses future bars -> shift back so it is only known 3 bars later
    disp_dn = (c - c.shift(-3)) > 1.5 * atr
    bull_ob = ((c < o) & disp_up).shift(3).fillna(False).astype(bool)   # known 3 bars after the OB candle
    bear_ob = ((c > o) & disp_dn).shift(3).fillna(False).astype(bool)
    bull_lvl = pd.Series(np.where(bull_ob, h.high.shift(3), np.nan), index=h.index).ffill(limit=100)
    bear_lvl = pd.Series(np.where(bear_ob, h.low.shift(3), np.nan), index=h.index).ffill(limit=100)
    f['bull_ob_dist'] = (c - bull_lvl) / atr; f['bear_ob_dist'] = (bear_lvl - c) / atr
    # --- fair value gaps in the last 20 bars (3-candle gaps), nearest distance
    bfvg = lo > hi.shift(2); sfvg = hi < lo.shift(2)
    f['bull_fvg_20'] = bfvg.rolling(20).sum(); f['bear_fvg_20'] = sfvg.rolling(20).sum()
    bfl = pd.Series(np.where(bfvg, hi.shift(2), np.nan), index=h.index).ffill(limit=20)
    sfl = pd.Series(np.where(sfvg, lo.shift(2), np.nan), index=h.index).ffill(limit=20)
    f['bull_fvg_dist'] = (c - bfl) / atr; f['bear_fvg_dist'] = (sfl - c) / atr
    return f.replace([np.inf, -np.inf], np.nan)


def select_causal(P, minp=0, q=0.995, cap=10):
    S = P.sort_values('time'); win = deque(); out = []; busy = {}; wk = {}; thr = None
    for t, g in S.groupby('time', sort=True):
        while win and win[0][0] < t - 14 * 86400: win.popleft()
        if len(win) > 50: thr = np.quantile(np.concatenate([a for _, a in win]), q)
        win.append((t, g.p.values))
        if thr is None: continue
        for r in g[(g.p >= thr) & (g.tp_pips >= minp)].itertuples():
            if busy.get(r.sym, -1) >= r.bar or wk.get(r.week, 0) >= cap: continue
            out.append((r.year, r.r, r.w)); busy[r.sym] = r.bar + 48; wk[r.week] = wk.get(r.week, 0) + 1
    return pd.DataFrame(out, columns=['year', 'r', 'w'])


def main():
    d = pd.read_pickle(os.path.join(HERE, 'v3_data.pkl'))
    pipv = lambda s: 0.01 if 'JPY' in s else 0.1 if s == 'XAUUSD' else 0.01 if s == 'XAGUSD' else 0.0001
    parts = []
    for s, g in d.groupby('sym'):
        h = pd.read_csv(f'{M.DATA}/{s}_H1.csv'); d1 = pd.read_csv(f'{M.DATA}/{s}_D1.csv')
        F = smc_features(h, d1)
        c = h.close; tr = pd.concat([h.high - h.low, (h.high - c.shift()).abs(), (h.low - c.shift()).abs()], axis=1).max(axis=1)
        F['tp_pips'] = 1.5 * tr.rolling(14).mean() / pipv(s)
        F = F.iloc[g.bar.values].set_index(g.index); parts.append(F); print('smc', s, flush=True)
    d = d.join(pd.concat(parts))
    base = [c for c in d.columns if c not in ('wL', 'wS', 'rL', 'rS', 'sym', 'time', 'bar', 'year', 'hour', 'tp_pips')
            and c not in parts[0].columns]
    smc = [c for c in parts[0].columns if c != 'tp_pips']
    params = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=500, feature_fraction=0.7,
                  bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=1)
    for name, feats in (('base v3', base), ('v3 + SMC', base + smc)):
        res = []; imp = None
        for Y in range(2021, 2027):
            te = d[d.year == Y]; tr_ = d[d.time < te.time.min() - 48 * 3600]
            r = te[['sym', 'time', 'bar', 'year', 'wL', 'wS', 'rL', 'rS', 'tp_pips']].copy()
            for side in 'LS':
                m = lgb.train(params, lgb.Dataset(tr_[feats], tr_['w' + side]), num_boost_round=300)
                r['p' + side] = m.predict(te[feats])
                if Y == 2026 and side == 'L' and name != 'base v3':
                    imp = pd.Series(m.feature_importance('gain'), feats).sort_values(ascending=False)
            res.append(r)
        P = pd.concat(res); P['p'] = np.maximum(P.pL, P.pS); P['side'] = np.where(P.pL >= P.pS, 1, -1)
        P['r'] = np.where(P.side > 0, P.rL, P.rS); P['w'] = np.where(P.side > 0, P.wL, P.wS)
        P['week'] = pd.to_datetime(P.time, unit='s').dt.to_period('W')
        P.to_pickle(os.path.join(HERE, f"v3_smc_oos_{name.replace(' ', '_').replace('+', 'p')}.pkl"))
        for minp in (0, 50):
            T = select_causal(P, minp)
            yr = T.groupby('year').r.mean().round(2).to_dict()
            print(f'{name:9s} min {minp:2d} pips: trades {len(T):4d} ({len(T)/306:.1f}/wk) win {T.w.mean()*100:4.1f}% avgR {T.r.mean():+.3f} '
                  f't {T.r.mean()/T.r.std()*np.sqrt(len(T)):4.1f} {yr}', flush=True)
        if imp is not None:
            rank = {f: i + 1 for i, f in enumerate(imp.index)}
            print('SMC feature importance rank (of %d):' % len(feats), {f: rank[f] for f in smc})


if __name__ == '__main__':
    main()
