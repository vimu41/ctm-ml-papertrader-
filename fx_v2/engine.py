"""CTM ML v2 feature engine + LightGBM tree evaluation in pure Python.
Same maths as ctm_ml.py (research), CTM_ML_Features.mqh (MT5 EA) and the dashboard JS.
Bars are dicts {t,o,h,l,c}, oldest first; the last element is the last CLOSED bar."""
import math

NAN = float('nan')


def _mean(v, i, n):
    if i - n + 1 < 0:
        return NAN
    return sum(v[i - n + 1:i + 1]) / n


def _hmax(B, i, n):
    return max(b['h'] for b in B[i - n + 1:i + 1])


def _lmin(B, i, n):
    return min(b['l'] for b in B[i - n + 1:i + 1])


def _ema_pair(c, n, upto):
    a = 2.0 / (n + 1)
    e = c[0]; p5 = e
    for i in range(1, upto + 1):
        e = a * c[i] + (1 - a) * e
        if i == upto - 5:
            p5 = e
    return e, p5


def _tr(B):
    out = []
    for k, b in enumerate(B):
        if k == 0:
            out.append(b['h'] - b['l'])
        else:
            pc = B[k - 1]['c']
            out.append(max(b['h'] - b['l'], abs(b['h'] - pc), abs(b['l'] - pc)))
    return out


def features(H, D, point, spr_pts):
    n = len(H)
    if n < 400:
        return None
    i = n - 1
    x = [NAN] * 29
    c = [b['c'] for b in H]
    tr = _tr(H)
    atr, atr100 = _mean(tr, i, 14), _mean(tr, i, 100)
    if not atr > 0:
        return None
    for k, r in enumerate((1, 3, 6, 12, 24, 48, 120)):
        x[k] = (c[i] - c[i - r]) / atr
    x[7] = atr / atr100
    for k, p in enumerate((20, 50, 200)):
        e, p5 = _ema_pair(c, p, i)
        x[8 + 2 * k] = (c[i] - e) / atr
        x[9 + 2 * k] = (e - p5) / atr
    tk = (_hmax(H, i, 9) + _lmin(H, i, 9)) / 2
    kj = (_hmax(H, i, 26) + _lmin(H, i, 26)) / 2
    j = i - 26
    sa = ((_hmax(H, j, 9) + _lmin(H, j, 9)) / 2 + (_hmax(H, j, 26) + _lmin(H, j, 26)) / 2) / 2
    sb = (_hmax(H, j, 52) + _lmin(H, j, 52)) / 2
    x[14] = (c[i] - (sa + sb) / 2) / atr
    x[15] = (sa - sb) / atr
    x[16] = (tk - kj) / atr
    up = dn = 0.0
    for k in range(i - 13, i + 1):
        d = c[k] - c[k - 1]
        if d > 0: up += d
        else: dn -= d
    up /= 14; dn /= 14
    x[17] = NAN if dn == 0 else 100 - 100 / (1 + up / dn)
    for k, p in enumerate((20, 100)):
        hh, ll = _hmax(H, i, p), _lmin(H, i, p)
        x[18 + k] = NAN if hh - ll == 0 else (c[i] - ll) / (hh - ll)
    import datetime as _dt
    dt = _dt.datetime.fromtimestamp(H[i]['t'], _dt.timezone.utc)
    x[20] = math.sin(2 * math.pi * dt.hour / 24)
    x[21] = math.cos(2 * math.pi * dt.hour / 24)
    x[22] = dt.weekday()                         # Monday = 0
    x[23] = spr_pts * point / atr
    if D:
        q = -1
        for k in range(len(D) - 1, -1, -1):
            if D[k]['t'] + 86400 <= H[i]['t']:
                q = k; break
        if q >= 205:
            Dd = D[:q + 1]; dc = [b['c'] for b in Dd]; dtr = _tr(Dd); datr = _mean(dtr, q, 14)
            if datr > 0:
                x[24] = (dc[q] - dc[q - 1]) / datr
                x[25] = (dc[q] - dc[q - 5]) / datr
                x[26] = (dc[q] - dc[q - 20]) / datr
                x[27] = (dc[q] - _ema_pair(dc, 50, q)[0]) / datr
                x[28] = (dc[q] - _ema_pair(dc, 200, q)[0]) / datr
    return {'x': x, 'atr': atr, 't': H[i]['t'], 'close': c[i]}


def predict(M, x):
    f, t, l, r, m, v = M['f'], M['t'], M['l'], M['r'], M['m'], M['v']
    s = 0.0
    for nd in M['root']:
        while nd >= 0:
            val = x[f[nd]]
            if val != val:                                   # NaN
                left = (0 <= t[nd]) if m[nd] == 0 else (m[nd] == 1)
            else:
                left = val <= t[nd]
            nd = l[nd] if left else r[nd]
        s += v[-nd - 1]
    return s
