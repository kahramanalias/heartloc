# -*- coding: utf-8 -*-
"""Supplementary Figure S3: sensitivity to block size (5, 10, 20 spatial blocks), predictor set A.
Gradient boosting on set A and the coordinates-only random forest, ten repetitions. Checkpointed."""
import os, sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, 'kod/revision')
from common import veri, koord, bolucu
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
OUT = 'sonuclar/revision/out'; os.makedirs(OUT, exist_ok=True)
d, FA, FB = veri(); xy, km = koord(d); y = d.kalp_olum_hizi.values.astype(float); n = len(y)
VAR = np.var(y, ddof=1)
XA = d[FA].apply(pd.to_numeric, errors='coerce').values
N_REP = int(os.environ.get('N_REP', 10)); SEEDS = [42 + 1000*s for s in range(N_REP)]
def model(ad, seed):
    if ad == 'GB':
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                             l2_regularization=1.0, early_stopping=True, random_state=seed)
    return RandomForestRegressor(200, max_features=1.0, min_samples_leaf=3, random_state=seed, n_jobs=-1)
X = {'GB': XA, 'coords_RF': xy}
CK = f'{OUT}/b11_blocksize.csv'; cv = pd.read_csv(CK) if os.path.exists(CK) else pd.DataFrame()
t0 = time.time()
for seed in SEEDS:
    for k in [5, 10, 20]:
        spl, grp = bolucu('Spatial', d, xy, seed, k=k); folds = list(spl.split(xy, y, grp))
        for ad in X:
            if len(cv) and ((cv.arm == ad) & (cv.k == k) & (cv.tohum == seed)).any(): continue
            oof = np.zeros(n)
            for tr, te in folds: oof[te] = model(ad, seed).fit(X[ad][tr], y[tr]).predict(X[ad][te])
            rec = dict(arm=ad, k=k, tohum=seed, n_blok=int(np.mean(np.bincount(grp))), R2_havuz=1 - np.mean((y - oof)**2) / VAR)
            cv = pd.concat([cv, pd.DataFrame([rec])], ignore_index=True); cv.to_csv(CK, index=False)
            print(f"{seed} k={k:<3}{ad:<10} R2={rec['R2_havuz']:.3f} ({time.time()-t0:.0f}s)", flush=True)
print(cv.groupby(['arm', 'k']).R2_havuz.agg(['mean', 'std', 'count']).round(3))
# figure
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'DejaVu Sans'})
g = cv.groupby(['arm', 'k']).R2_havuz.agg(['mean', 'std']).reset_index()
fig, ax = plt.subplots(figsize=(5.2, 3.2))
for ad, lab, col in [('GB', 'Gradient boosting', '#c44e52'), ('coords_RF', 'Coordinates only', '#8c8c8c')]:
    q = g[g.arm == ad].sort_values('k'); x = n / q.k.values
    ax.errorbar(x, q['mean'], yerr=q['std'], marker='o', color=col, capsize=3, label=lab)
ax.invert_xaxis(); ax.set_xlabel('Counties per spatial block'); ax.set_ylabel('Pooled out-of-fold $R^2$')
ax.grid(alpha=.2); ax.legend(frameon=False, loc='center right')
os.makedirs('sonuclar/revision/fig', exist_ok=True)
for ext in ['png', 'pdf']: fig.savefig(f'sonuclar/revision/fig/FigureS3_block_size.{ext}', dpi=600, bbox_inches='tight')
print('BITTI')
