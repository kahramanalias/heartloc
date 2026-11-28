# -*- coding: utf-8 -*-
"""Supplementary Table S12b: pure-noise scenario. Independent Gaussian noise (no signal) is smoothed with the
five-nearest-neighbour smoother (w = 0, 0.25, 0.5, 0.75) and fitted with the coordinates-only random forest
under random and spatial validation, three repetitions (same seeds as b7_smooth.py)."""
import os, sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, 'kod/revision')
from common import veri, koord, bolucu
from sklearn.ensemble import RandomForestRegressor
from scipy.spatial import cKDTree
OUT = 'sonuclar/revision/out'
d, FA, FB = veri(); xy, km = koord(d); n = len(d)
y0 = np.random.default_rng(2025).normal(size=n)                      # pure noise, no signal
nn5 = cKDTree(km).query(km, k=6)[1][:, 1:]
SC = {f'w{w}': (1-w)*y0 + w*y0[nn5].mean(1) for w in [0.0, 0.25, 0.5, 0.75]}
CK = f'{OUT}/b12_purenoise.csv'; cv = pd.read_csv(CK) if os.path.exists(CK) else pd.DataFrame()
for seed in [42, 1042, 2042]:
    for rej in ['Random', 'Spatial']:
        spl, grp = bolucu(rej, d, xy, seed); folds = list(spl.split(xy, y0, grp))
        for sc, y in SC.items():
            if len(cv) and ((cv.senaryo == sc) & (cv.rejim == rej) & (cv.tohum == seed)).any(): continue
            oof = np.zeros(n)
            for tr, te in folds:
                oof[te] = RandomForestRegressor(200, max_features=1.0, min_samples_leaf=3, random_state=seed,
                                                n_jobs=-1).fit(xy[tr], y[tr]).predict(xy[te])
            rec = dict(senaryo=sc, rejim=rej, tohum=seed, R2=1 - np.mean((y - oof)**2) / np.var(y, ddof=1))
            cv = pd.concat([cv, pd.DataFrame([rec])], ignore_index=True); cv.to_csv(CK, index=False)
print(cv.pivot_table(index='senaryo', columns='rejim', values='R2').round(3)); print('BITTI')
