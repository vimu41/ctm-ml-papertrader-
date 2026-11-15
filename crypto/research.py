"""CTM crypto ML research: walk-forward LightGBM on Binance 1h (or 4h) bars, 10 coins pooled.

Features = the forex model's trend/momentum set + crypto order-flow and volume-profile features:
  taker-buy ratio (1/6/24 bars), cumulative volume delta (24/72), volume z-score and ratios, trade-count ratio,
  rolling volume profile (POC distance, value-area high/low position, inside-value-area flag), BTC context.
Label (same as forex): enter next bar open, SL = SLM x ATR14, TP = RR x SL, close after HOLD bars.
Result in R minus a percentage cost (fees + slippage, round trip).
Usage: python research.py [TF=1h|4h] [COST=0.002] [SLM=1.0]"""
import glob, os, sys
import numpy as np, pandas as pd, lightgbm as lgb
from numpy.lib.stride_tricks import sliding_window_view as swv

HERE = os.path.dirname(os.path.abspath(__file__))
TF = os.environ.get('TF', '1h'); COST = float(os.environ.get('COST', '0.002')); SLM = float(os.environ.get('SLM', '1.0'))
RR = 2.0
HOLD = 48 if TF == '1h' else 30
BARSEC = 3600 if TF == '1h' else 14400
VP_N, VP_BINS = (168, 24) if TF == '1h' else (126, 24)     # volume profile over ~1 week (1h) / ~3 weeks (4h)


def load(path):
    h = pd.read_csv(path).astype(float)
    h['time'] = h.time.astype(np.int64)
    h = h.drop_duplicates('time').sort_values('time').reset_index(drop=True)
    if TF == '4h':
        g = h.time // 14400
        h = h.groupby(g).agg(time=('time', 'first'), open=('open', 'first'), high=('high', 'max'), low=('low', 'min'),
                             close=('close', 'last'), volume=('volume', 'sum'), quote_volume=('quote_volume', 'sum'),
                             trades=('trades', 'sum'), taker_buy_base=('taker_buy_base', 'sum')).reset_index(drop=True)
    return h


def ema(x, n): return x.ewm(span=n, adjust=False).mean()


def volume_profile(h):
    tp = ((h.high + h.low + h.close) / 3).values; v = h.volume.values; n = len(h)
    out = np.full((n, 4), np.nan)
    if n <= VP_N: return out
    P = swv(tp, VP_N); V = swv(v, VP_N)                     # rows: window ending at i = VP_N-1 ...
    lo = P.min(1, keepdims=True); hi = P.max(1, keepdims=True); rng = np.where(hi - lo > 0, hi - lo, np.nan)
    b = np.clip(((P - lo) / rng * VP_BINS).astype(float), 0, VP_BINS - 1e-9)
    b = np.nan_to_num(b).astype(int)
    H = np.zeros((P.shape[0], VP_BINS))
    np.add.at(H, (np.repeat(np.arange(P.shape[0]), VP_N), b.ravel()), V.ravel())
    centers = lo + (np.arange(VP_BINS) + 0.5) / VP_BINS * (hi - lo)
    poc = centers[np.arange(len(H)), H.argmax(1)]
    order = np.argsort(-H, 1); Hs = np.take_along_axis(H, order, 1)
    cum = np.cumsum(Hs, 1) / np.maximum(Hs.sum(1, keepdims=True), 1e-12)
    inc = np.concatenate([np.ones((len(H), 1), bool), cum[:, :-1] < 0.7], 1)
    cs = np.take_along_axis(np.broadcast_to(centers, H.shape), order, 1)
    vah = np.where(inc, cs, -np.inf).max(1); val = np.where(inc, cs, np.inf).min(1)
    out[VP_N - 1:, 0] = poc; out[VP_N - 1:, 1] = vah; out[VP_N - 1:, 2] = val
    return out


def features(h, btc):
    f = pd.DataFrame(index=h.index); c, hi, lo = h.close, h.high, h.low
    tr = pd.concat([hi - lo, (hi - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean(); h['atr'] = atr
    for n in (1, 3, 6, 12, 24, 48, 120): f[f'ret{n}'] = (c - c.shift(n)) / atr
    f['vol_ratio'] = atr / tr.rolling(100).mean(); f['atr_pct'] = atr / c
    for n in (20, 50, 200):
        e = ema(c, n); f[f'dist_ema{n}'] = (c - e) / atr; f[f'slope_ema{n}'] = (e - e.shift(5)) / atr
    tk = (hi.rolling(9).max() + lo.rolling(9).min()) / 2; kj = (hi.rolling(26).max() + lo.rolling(26).min()) / 2
    sa = ((tk + kj) / 2).shift(26); sb = ((hi.rolling(52).max() + lo.rolling(52).min()) / 2).shift(26)
    f['cloud_pos'] = (c - (sa + sb) / 2) / atr; f['tk_kj'] = (tk - kj) / atr
    d = c.diff(); up = d.clip(lower=0).rolling(14).mean(); dn = (-d.clip(upper=0)).rolling(14).mean()
    f['rsi'] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    for n in (20, 100):
        rh, rl = hi.rolling(n).max(), lo.rolling(n).min(); f[f'rangepos{n}'] = (c - rl) / (rh - rl).replace(0, np.nan)
    t = pd.to_datetime(h.time, unit='s')
    f['hour_sin'] = np.sin(2 * np.pi * t.dt.hour / 24); f['hour_cos'] = np.cos(2 * np.pi * t.dt.hour / 24); f['dow'] = t.dt.dayofweek
    # --- order flow
    vol = h.volume.replace(0, np.nan); tb = h.taker_buy_base
    for n in (1, 6, 24): f[f'taker_buy{n}'] = tb.rolling(n).sum() / vol.rolling(n).sum() - 0.5
    delta = 2 * tb - h.volume
    for n in (24, 72): f[f'cvd{n}'] = delta.rolling(n).sum() / vol.rolling(n).sum()
    lv = np.log(vol)
    f['vol_z'] = (lv - lv.rolling(168).mean()) / lv.rolling(168).std()
    f['vol_24_168'] = vol.rolling(24).mean() / vol.rolling(168).mean()
    f['trades_24_168'] = h.trades.rolling(24).mean() / h.trades.rolling(168).mean()
    f['avg_trade_size'] = np.log((vol / h.trades.replace(0, np.nan)) / (vol / h.trades.replace(0, np.nan)).rolling(168).mean())
    # --- volume profile
    vp = volume_profile(h)
    f['poc_dist'] = (c - vp[:, 0]) / atr; f['vah_dist'] = (c - vp[:, 1]) / atr; f['val_dist'] = (c - vp[:, 2]) / atr
    f['in_va'] = ((c <= vp[:, 1]) & (c >= vp[:, 2])).astype(float)
    # --- BTC context (same timestamp, already closed)
    if btc is not None:
        b = btc[['time', 'b_ret24', 'b_ret6', 'b_dist200']]
        m = h[['time']].merge(b, on='time', how='left')
        for col in ('b_ret24', 'b_ret6', 'b_dist200'): f[col] = m[col].values
    return f


def labels(h):
    o, hi, lo, c, atr = h.open.values, h.high.values, h.low.values, h.close.values, h.atr.values
    n = len(h); L = np.full(n, np.nan); S = np.full(n, np.nan); LX = np.zeros(n, int); SX = np.zeros(n, int)
    for i in range(n - HOLD - 2):
        a = atr[i] * SLM
        if not np.isfinite(a) or a <= 0: continue
        e = o[i + 1]; cr = COST * e / a
        sl, tp = e - a, e + RR * a; r = None
        for j in range(i + 1, i + 1 + HOLD):
            if lo[j] <= sl: r = -1.0; LX[i] = j; break
            if hi[j] >= tp: r = RR; LX[i] = j; break
        if r is None: r = (c[i + HOLD] - e) / a; LX[i] = i + HOLD
        L[i] = r - cr
        sl, tp = e + a, e - RR * a; r = None
        for j in range(i + 1, i + 1 + HOLD):
            if hi[j] >= sl: r = -1.0; SX[i] = j; break
            if lo[j] <= tp: r = RR; SX[i] = j; break
        if r is None: r = (e - c[i + HOLD]) / a; SX[i] = i + HOLD
        S[i] = r - cr
    return L, S, LX, SX


def main():
    paths = sorted(glob.glob(os.path.join(HERE, 'history', '*_1h.csv.gz')))
    raw = {os.path.basename(p).split('_')[0]: load(p) for p in paths}
    b = raw['BTCUSDT'].copy(); bc = b.close
    btr = pd.concat([b.high - b.low, (b.high - bc.shift()).abs(), (b.low - bc.shift()).abs()], axis=1).max(axis=1).rolling(14).mean()
    b['b_ret24'] = (bc - bc.shift(24)) / btr; b['b_ret6'] = (bc - bc.shift(6)) / btr; b['b_dist200'] = (bc - ema(bc, 200)) / btr
    frames = []
    for s, h in raw.items():
        f = features(h, b); L, S, LX, SX = labels(h)
        f['yL'], f['yS'], f['LX'], f['SX'] = L, S, LX, SX
        f['sym'] = s; f['time'] = h.time.values; f['bar'] = np.arange(len(h)); f['year'] = pd.to_datetime(h.time, unit='s').dt.year.values
        f['cost_r'] = COST * h.close / (h.atr * SLM)
        frames.append(f); print(s, len(f), flush=True)
    data = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna(subset=['yL', 'yS'])
    feats = [c for c in data.columns if c not in ('yL', 'yS', 'LX', 'SX', 'sym', 'time', 'bar', 'year', 'cost_r')]
    params = dict(objective='regression', learning_rate=0.03, num_leaves=15, min_data_in_leaf=500, feature_fraction=0.7,
                  bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=1)
    preds = []
    for Y in range(2021, 2027):
        te = data[data.year == Y]
        if not len(te): continue
        tr = data[data.time < te.time.min() - HOLD * BARSEC]
        res = te[['sym', 'time', 'bar', 'year', 'yL', 'yS', 'LX', 'SX', 'cost_r']].copy()
        for side in 'LS':
            m = lgb.train(params, lgb.Dataset(tr[feats], tr['y' + side]), num_boost_round=300)
            res['p' + side] = m.predict(te[feats])
            if Y == 2026 and side == 'L':
                imp = pd.Series(m.feature_importance('gain'), feats).sort_values(ascending=False)
        preds.append(res); print('year', Y, 'train', len(tr), 'test', len(te), flush=True)
    P = pd.concat(preds); tag = f'{TF}_c{COST}_s{SLM}'
    P.to_pickle(os.path.join(HERE, f'oos_{tag}.pkl'))
    print('\nbaseline mean R every bar: long %.3f short %.3f, median cost %.3fR' % (P.yL.mean(), P.yS.mean(), P.cost_r.median()))
    print('top features:', ', '.join(imp.index[:12]))
    for thr in (0.0, 0.1, 0.2, 0.3):
        for mode in ('both', 'long'):
            T = []
            for s, g in P.groupby('sym'):
                busy = -1
                for bar, pl, ps, yl, ys, lx, sx, y, t in g.sort_values('bar')[['bar', 'pL', 'pS', 'yL', 'yS', 'LX', 'SX', 'year', 'time']].itertuples(index=False):
                    if bar <= busy: continue
                    if mode == 'long': ps = -9
                    if max(pl, ps) <= thr: continue
                    if pl >= ps: T.append((y, s, t, yl)); busy = lx
                    else: T.append((y, s, t, ys)); busy = sx
            T = pd.DataFrame(T, columns=['year', 'sym', 'time', 'r'])
            if not len(T): continue
            mo = T.groupby(pd.to_datetime(T.time, unit='s').dt.to_period('M')).r.sum()
            yr = T.groupby('year').r.mean().round(2).to_dict(); sy = (T.groupby('sym').r.mean() > 0).sum()
            print(f'thr {thr} {mode:4s}: n {len(T):5d} avgR {T.r.mean():+.3f} t {T.r.mean() / T.r.std() * np.sqrt(len(T)):4.1f} '
                  f'months+ {(mo > 0).mean() * 100:3.0f}% coins+ {sy}/10 per-year {yr}')


if __name__ == '__main__':
    main()
