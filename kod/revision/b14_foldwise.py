# -*- coding: utf-8 -*-
"""Supplementary Figure S5: fold-wise R² against within-block outcome SD, spatial blocking, ten repetitions,
gradient boosting (set A) and coordinates-only random forest; also the fold-wise mean R² quoted in the Results."""
import os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, 'kod/revision')
from common import veri, koord, bolucu
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
OUT = 'sonuclar/revision/out'
d, FA, FB = veri(); xy, km = koord(d); y = d.kalp_olum_hizi.values.astype(float); n = len(y); VAR = np.var(y, ddof=1)
XA = d[FA].apply(pd.to_numeric, errors='coerce').values
def model(ad, seed):
    if ad == 'GB': return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                                        l2_regularization=1.0, early_stopping=True, random_state=seed)
    return RandomForestRegressor(200, max_features=1.0, min_samples_leaf=3, random_state=seed, n_jobs=-1)
X = {'GB': XA, 'coords_RF': xy}; rows = []
for seed in [42 + 1000*s for s in range(10)]:
    spl, grp = bolucu('Spatial', d, xy, seed)
    for ad in X:
        oof = np.zeros(n)
        for b, (tr, te) in enumerate(spl.split(xy, y, grp)):
            oof[te] = model(ad, seed).fit(X[ad][tr], y[tr]).predict(X[ad][te])
            rows.append(dict(tohum=seed, arm=ad, blok=b, sd_blok=y[te].std(ddof=1),
                             R2_kat=1 - np.sum((y[te]-oof[te])**2) / np.sum((y[te]-y[te].mean())**2)))
        rows.append(dict(tohum=seed, arm=ad, blok=-1, sd_blok=np.nan, R2_kat=1 - np.mean((y-oof)**2) / VAR))
D = pd.DataFrame(rows); D.to_csv(f'{OUT}/b14_foldwise.csv', index=False)
f = D[D.blok >= 0].groupby(['arm', 'tohum']).R2_kat.mean().groupby('arm').mean()
p = D[D.blok < 0].groupby('arm').R2_kat.mean()
print('fold-wise mean R2:', f.round(5).to_dict(), '| pooled R2:', p.round(5).to_dict())
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'DejaVu Sans'})
fig, ax = plt.subplots(figsize=(5.6, 3.6)); q = D[D.blok >= 0]
for ad, lab, c in [('GB', 'Gradient boosting', '#c44e52'), ('coords_RF', 'Coordinates only', '#8c8c8c')]:
    s = q[q.arm == ad]; ax.scatter(s.sd_blok, s.R2_kat, s=14, alpha=.6, color=c, label=lab, linewidths=0)
ax.axhline(0, color='k', lw=.7); ax.axvline(np.sqrt(VAR), color='k', ls=':', lw=.9)
ax.text(np.sqrt(VAR), ax.get_ylim()[0], ' overall SD', fontsize=7, va='bottom')
ax.set_xlabel('Within-block outcome SD'); ax.set_ylabel('Fold-wise $R^2$'); ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.2)
for ext in ['png', 'pdf']: fig.savefig(f'sonuclar/revision/fig/FigureS5_foldwise.{ext}', dpi=600, bbox_inches='tight')
print('BITTI')
