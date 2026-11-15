"""Download Binance spot 1h klines (monthly zips from data.binance.vision) and save one gzipped CSV per symbol.
Columns: time(open, s), open, high, low, close, volume, quote_volume, trades, taker_buy_base, taker_buy_quote."""
import csv, gzip, io, os, sys, urllib.request, zipfile, datetime as dt

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'LTCUSDT']
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'history')
BASE = 'https://data.binance.vision/data/spot/monthly/klines/{s}/1h/{s}-1h-{y}-{m:02d}.zip'


def months(start=(2019, 1)):
    y, m = start; today = dt.date.today()
    while (y, m) < (today.year, today.month):
        yield y, m
        m += 1
        if m == 13: y, m = y + 1, 1


def main():
    os.makedirs(OUT, exist_ok=True)
    for s in SYMBOLS:
        rows = []
        for y, m in months():
            try:
                data = urllib.request.urlopen(BASE.format(s=s, y=y, m=m), timeout=60).read()
            except Exception:
                continue                                  # symbol not listed yet that month
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for line in z.read(z.namelist()[0]).decode().splitlines():
                    p = line.split(',')
                    if not p[0].isdigit(): continue
                    t = int(p[0]); t = t // 1000000 if t > 10**14 else t // 1000   # 2025+ files use microseconds
                    rows.append([t, p[1], p[2], p[3], p[4], p[5], p[7], p[8], p[9], p[10]])
        rows.sort(key=lambda r: r[0])
        with gzip.open(os.path.join(OUT, f'{s}_1h.csv.gz'), 'wt', newline='') as f:
            w = csv.writer(f)
            w.writerow(['time', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote'])
            w.writerows(rows)
        print(s, len(rows), 'bars', rows[0][0] if rows else '-', flush=True)


if __name__ == '__main__':
    main()
