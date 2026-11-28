"""CV 01 backtest trades re-priced with swap and spread.
Swap per night = position value x annual swap rate / 360 (Deriv's synthetic formula). In R units that is
(rate/360) / (ATR/price): small for V75/V100, large for low-volatility V10/V25. Nights = 00:00 UTC crossings held."""
import glob, os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import bot as DB
H = os.path.dirname(os.path.abspath(__file__))
rows = []
for fn in sorted(glob.glob(f'{H}/history/*_H1.csv.gz')):
    sym = os.path.basename(fn).split('_H1')[0]
    if sym not in DB.SYMBOLS: continue
    h = pd.read_csv(fn).sort_values('time').reset_index(drop=True); B = [{'t': r.time, 'o': r.open, 'h': r.high, 'l': r.low, 'c': r.close} for r in h.itertuples()]
    hi, lo, c, t = h.high.values, h.low.values, h.close.values, h.time.values; n = len(c); busy = -1
    for i in range(60, n - 1):
        if i <= busy: continue
        s, info = DB.signal(B, i)
        if not s: continue
        e, a, sd = c[i], info['atr'], 1 if s == 'BUY' else -1; sl, tp = e - sd * a, e + sd * 1.5 * a; r = None
        for j in range(i + 1, min(n, i + 49)):
            if (lo[j] <= sl) if sd > 0 else (hi[j] >= sl): r = -1.0; break
            if (hi[j] >= tp) if sd > 0 else (lo[j] <= tp): r = 1.5; break
        if r is None: j = min(n - 1, i + 48); r = (c[j] - e) * sd / a
        nights = int((t[j] + 3600) // 86400 - t[i] // 86400)
        rows.append(dict(sym=sym, r=r, nights=nights, atr_pct=a / e * 100, hours=j - i)); busy = j
T = pd.DataFrame(rows)
print(f'trades {len(T)}, avg hours held {T.hours.mean():.1f}, avg nights {T.nights.mean():.2f}, gross avgR {T.r.mean():+.3f}\n')
print(f'{"index":9s} {"n":>4s} {"ATR % of price":>15s} {"gross R":>8s} | net R with annual swap rate:  {"0%":>7s} {"-5%":>7s} {"-7.5%":>7s} {"-15%":>7s}')
for sym, g in T.groupby('sym', sort=False):
    line = f'{sym:9s} {len(g):4d} {g.atr_pct.mean():14.3f}% {g.r.mean():+8.3f} |' + ' ' * 30
    for rate in (0, 5, 7.5, 15):
        sw = g.nights * (rate / 360) / g.atr_pct            # swap in R
        line += f' {(g.r - 0.03 - sw).mean():+7.3f}'
    print(line)
for rate in (0, 5, 7.5, 15):
    sw = T.nights * (rate / 360) / T.atr_pct; net = T.r - 0.03 - sw
    print(f'ALL, swap {rate:>4}%/yr: net avgR {net.mean():+.3f}  t {net.mean() / net.std() * np.sqrt(len(net)):+.1f}')
sd = (T.r - 0.03).std(); print(f'\nstd of R per trade {sd:.2f} -> trades needed to confirm +0.137R at t=2: {int((2 * sd / 0.137) ** 2)} '
      f'(~{(2 * sd / 0.137) ** 2 / (len(T) / 52):.0f} weeks at {len(T) / 52:.1f} trades/week)')
