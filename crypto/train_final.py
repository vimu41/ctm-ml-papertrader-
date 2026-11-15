"""Train the live crypto model (long side only) on all history: 1h bars, SL 2 x ATR, TP 4 x ATR, 48-bar hold, 0.2% cost."""
import os, glob, json
os.environ.setdefault('TF', '1h'); os.environ.setdefault('SLM', '2.0'); os.environ.setdefault('COST', '0.002')
import numpy as np, pandas as pd, lightgbm as lgb
import research as R

raw = {os.path.basename(p).split('_')[0]: R.load(p) for p in sorted(glob.glob(os.path.join(R.HERE, 'history', '*_1h.csv.gz')))}
b = R.btc_context(raw['BTCUSDT'])
frames = []
for s, h in raw.items():
    f = R.features(h, b); L, S, LX, SX = R.labels(h); f['yL'] = L; frames.append(f)
data = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna(subset=['yL'])
feats = [c for c in data.columns if c != 'yL']
params = dict(objective='regression', learning_rate=0.03, num_leaves=15, min_data_in_leaf=500, feature_fraction=0.7,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=1)
m = lgb.train(params, lgb.Dataset(data[feats], data['yL']), num_boost_round=300)
m.save_model(os.path.join(R.HERE, 'model_long.txt'))
json.dump(feats, open(os.path.join(R.HERE, 'model_feats.json'), 'w'))
print('trained on', len(data), 'rows,', len(feats), 'features')
