"""Open-source reconstruction of the FGI (Fibonacci Golden Indicator) strategies taught on the FGI Vision channel,
backtested honestly: FIXED stop loss, NO averaging down, one trade per symbol, spread costs, 1.5:1 target.
FGI itself is closed (invite-only), so each piece is our best open equivalent of what the videos describe:
  FGI trend line   = SuperTrend on close, ATR(10) x 2.618 (the 'Fibonacci multiplication value' in its settings)
  Entry (all)      = wait for the signal candle to CLOSE, then a stop order at its high (buy) / low (sell), valid 3 bars
  Stop             = beyond the signal candle's low/high (min 0.5 ATR) · Target = 1.5 x risk · max 48 bars
Strategies:
  S1 Trend follow (Part 05)   : H1 + H4 FGI trend up, a pullback candle (red candle touching EMA20) -> buy its break
  S2 Extreme zones (Part 06)  : price reaches the extreme band (EMA50 -/+ 2.618 ATR) and a reversal candle closes -> trade its break
  S3 SR letters (Part 04)     : confirmed swing low 'S' (5-bar fractal) -> buy the break of that candle's high (mirror for 'R')
  S4 FGI signal + H4 (15m vid): FGI trend flips on H1 (B/S letter) while H4 trend agrees -> trade the break of the flip candle
  S0 Baseline                 : same entry/stop/target on random candles (what 'no skill' looks like)"""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, '/home/claude/ml'); import ctm_ml_v2 as M
RR, HOLD, VALID = 1.5, 48, 3

def supertrend(h, n=10, mult=2.618):
    hl2 = (h.high + h.low) / 2; c = h.close.values
    tr = pd.concat([h.high - h.low, (h.high - h.close.shift()).abs(), (h.low - h.close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean().values
    up, dn = (hl2 - mult * atr).values, (hl2 + mult * atr).values
    fu, fd, d = up.copy(), dn.copy(), np.ones(len(c))
    for i in range(1, len(c)):
        fu[i] = max(up[i], fu[i - 1]) if c[i - 1] > fu[i - 1] else up[i]
        fd[i] = min(dn[i], fd[i - 1]) if c[i - 1] < fd[i - 1] else dn[i]
        d[i] = 1 if c[i] > fd[i - 1] else -1 if c[i] < fu[i - 1] else d[i - 1]
    return d, np.where(d > 0, fu, fd)

def h4_trend(h):
    g = h.time // 14400; a = h.groupby(g).agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last')).reset_index(drop=True)
    d, _ = supertrend(a); s = pd.Series(d, index=np.unique(g)).shift(1)          # only CLOSED H4 bars
    return s.reindex(g).values

def simulate(h, sig, side, cost, atr, LH, LL, min_risk):
    o, hi, lo, c = h.open.values, h.high.values, h.low.values, h.close.values; n = len(c); out = []; busy = -1
    for s in np.where(sig)[0]:
        if s <= busy or s + VALID + 2 >= n or not atr[s] > 0: continue
        sd = side[s]; lvl = LH[s] if sd > 0 else LL[s]
        stop = (LL[s] if sd > 0 else LH[s]); risk_min = min_risk * atr[s]
        j0 = None
        for j in range(s + 1, s + 1 + VALID):                 # pending stop order
            if (hi[j] >= lvl) if sd > 0 else (lo[j] <= lvl): j0 = j; break
        if j0 is None: continue
        e = max(lvl, o[j0]) if sd > 0 else min(lvl, o[j0])
        risk = max(abs(e - stop), risk_min); sl = e - sd * risk; tp = e + sd * RR * risk; r = None
        # entry candle: the order of its high/low is unknown, so judge it only by its close
        if (c[j0] <= sl) if sd > 0 else (c[j0] >= sl): r = -1.0; j = j0
        elif (c[j0] >= tp) if sd > 0 else (c[j0] <= tp): r = RR; j = j0
        else:
            for j in range(j0 + 1, min(n, j0 + HOLD)):
                if (lo[j] <= sl) if sd > 0 else (hi[j] >= sl): r = -1.0; break
                if (hi[j] >= tp) if sd > 0 else (lo[j] <= tp): r = RR; break
        if r is None: j = min(n - 1, j0 + HOLD); r = sd * (c[j] - e) / risk
        out.append((h.time.values[s], r - cost[s] / risk, r)); busy = j
    return out

def run_symbol(path):
    sym = os.path.basename(path).replace('_H1.csv', ''); h = pd.read_csv(path)
    h = h[h.time >= 1420070400].reset_index(drop=True)                      # 2015+
    c, o, hi, lo = h.close, h.open, h.high, h.low
    cost = M.REAL.get(sym, 0.0) * (c / c.iloc[-1]).values          # spread scales with price (matters for BTC/ETH since 2015)
    tr = pd.concat([hi - lo, (hi - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1); atr = tr.rolling(14).mean().values
    d1, st = supertrend(h); d4 = h4_trend(h); e20 = c.ewm(span=20, adjust=False).mean(); e50 = c.ewm(span=50, adjust=False).mean()
    hour = pd.to_datetime(h.time, unit='s').dt.hour.values; ok_hour = ~np.isin(hour, (20, 21, 22))   # avoid rollover spreads
    bull, bear = (c > o).values, (c < o).values
    R = {}
    # S1 trend follow: both trends up, red pullback candle touching EMA20
    l1 = (d1 > 0) & (d4 > 0) & bear & (lo.values <= e20.values); s1 = (d1 < 0) & (d4 < 0) & bull & (hi.values >= e20.values)
    # S2 extreme zones: beyond EMA50 -/+ 2.618 ATR, reversal candle
    lz, uz = e50.values - 2.618 * atr, e50.values + 2.618 * atr
    l2 = (lo.values <= lz) & bull; s2 = (hi.values >= uz) & bear
    # S3 SR letters: 5-bar fractal low/high confirmed 2 bars later (signal candle = the fractal candle, known at i)
    fl = (lo.shift(2) < lo.shift(3)) & (lo.shift(2) < lo.shift(4)) & (lo.shift(2) < lo.shift(1)) & (lo.shift(2) < lo)
    fh = (hi.shift(2) > hi.shift(3)) & (hi.shift(2) > hi.shift(4)) & (hi.shift(2) > hi.shift(1)) & (hi.shift(2) > hi)
    l3, s3 = fl.fillna(False).values, fh.fillna(False).values
    # S4 FGI flip (B/S letter) with H4 agreement
    flip_up = (d1 > 0) & (np.r_[0, d1[:-1]] < 0); flip_dn = (d1 < 0) & (np.r_[0, d1[:-1]] > 0)
    l4, s4 = flip_up & (d4 > 0), flip_dn & (d4 < 0)
    rng = np.random.default_rng(len(h)); rnd = rng.random(len(h)) < 0.01; rs = rng.random(len(h)) < 0.5
    LH, LL = hi.values, lo.values; FH, FL = hi.shift(2).values, lo.shift(2).values     # S3 uses the fractal candle (2 bars back)
    for name, L, S in (('S1 Trend follow (H1+H4 trend, pullback)', l1, s1), ('S2 Extreme zones (2.618 ATR band)', l2, s2),
                       ('S3 SR letters (swing S/R break)', l3, s3), ('S3b SR letters + H4 trend', l3 & (d4 > 0), s3 & (d4 < 0)),
                       ('S4 FGI flip (B/S) + H4 trend', l4, s4), ('S0 Random candles (baseline)', rnd & rs, rnd & ~rs)):
        sig = (L | S) & ok_hour & np.isfinite(atr); side = np.where(L, 1, -1)
        lh, ll = (FH, FL) if name.startswith('S3') else (LH, LL)
        for mr in (0.5, 1.0):
            R[f'{name} | stop>={mr} ATR'] = simulate(h, sig, side, cost, atr, lh, ll, mr)
    return sym, R

def stat(rows):
    if len(rows) < 20: return f'trades {len(rows):5d}'
    r = np.array([x for _, x, _ in rows]); g = np.array([x for _, _, x in rows]); y = pd.to_datetime([t for t, _, _ in rows], unit='s').year
    half = np.array(y) >= 2021
    return (f'trades {len(r):5d} ({len(r) / 600:4.1f}/wk) win {np.mean(r > 0) * 100:5.1f}% avgR {r.mean():+.3f} t {r.mean() / r.std() * np.sqrt(len(r)):+5.1f}'
            f' | gross {g.mean():+.3f} | 15-20 {r[~half].mean():+.3f} · 21-26 {r[half].mean():+.3f}')

if __name__ == '__main__':
    groups = {'FOREX (15 pairs)': [], 'GOLD/SILVER': [], 'CRYPTO (BTC, ETH)': []}
    agg = {}
    for p in sorted(glob.glob(f'{M.DATA}/*_H1.csv')):
        sym, R = run_symbol(p)
        g = 'GOLD/SILVER' if sym.startswith('XA') else 'CRYPTO (BTC, ETH)' if sym in ('BTCUSD', 'ETHUSD') else 'FOREX (15 pairs)'
        for k, v in R.items(): agg.setdefault(g, {}).setdefault(k, []).extend(v)
        print('done', sym, flush=True)
    print('\nFGI-style strategies, H1, 2015-2026, fixed SL, no averaging, spread included, target 1.5 x risk (break-even win = 40%)')
    for g, d in agg.items():
        print(f'\n== {g}')
        for k, v in d.items(): print(f'{k:58s} {stat(v)}')
