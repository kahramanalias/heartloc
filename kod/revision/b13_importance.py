# -*- coding: utf-8 -*-
"""Supplementary Table S6 (set B) and Supplementary Figure S6 (set A): permutation importance of gradient boosting,
measured on each held-out spatial block of the first repetition (seed 42), 10 permutations per block."""
import os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, 'kod/revision')
from common import veri, koord, bolucu
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
OUT = 'sonuclar/revision/out'
d, FA, FB = veri(); xy, km = koord(d); y = d.kalp_olum_hizi.values.astype(float)
spl, grp = bolucu('Spatial', d, xy, 42); folds = list(spl.split(xy, y, grp))
res = []
for setad, F in [('A', FA), ('B', FB)]:
    X = d[F].apply(pd.to_numeric, errors='coerce').values; imp = np.zeros((len(folds), len(F)))
    for k, (tr, te) in enumerate(folds):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                          l2_regularization=1.0, early_stopping=True, random_state=42).fit(X[tr], y[tr])
        imp[k] = permutation_importance(m, X[te], y[te], n_repeats=10, random_state=42, scoring='r2', n_jobs=-1).importances_mean
    res.append(pd.DataFrame({'set': setad, 'degisken': F, 'onem_ort': imp.mean(0), 'onem_sd': imp.std(0)}))
R = pd.concat(res).sort_values(['set', 'onem_ort'], ascending=[True, False]); R.to_csv(f'{OUT}/b13_importance.csv', index=False)
for s in 'AB': print(s); print(R[R.set == s].head(12).round(4).to_string(index=False))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'DejaVu Sans'})
AD = {'MEDHHINC15': 'Median household income', 'SNAPSPTH17': 'SNAP outlets per 1,000', 'FOODINSEC_12_14': 'Food insecurity 2012–14 (%)',
      'FOODINSEC_15_17': 'Food insecurity 2015–17 (%)', 'PCT_HISP10': 'Hispanic population (%)', 'PC_FFRSALES12': 'Fast-food expenditure per capita',
      'PCT_NHASIAN10': 'Asian population (%)', 'PCT_18YOUNGER10': 'Population under 18 (%)', 'PCT_NHNA10': 'Native American population (%)',
      'POVRATE15': 'Poverty rate (%)', 'PCT_NHBLACK10': 'Black population (%)', 'RECFACPTH16': 'Recreation facilities per 1,000',
      'FSRPTH16': 'Full-service restaurants per 1,000', 'SUPERCPTH16': 'Supercentres per 1,000', 'GROCPTH16': 'Grocery stores per 1,000',
      'FFRPTH16': 'Fast-food restaurants per 1,000', 'CONVSPTH16': 'Convenience stores per 1,000', 'SPECSPTH16': 'Specialised food stores per 1,000',
      'PCT_65OLDER10': 'Population aged 65+ (%)', 'CHILDPOVRATE15': 'Child poverty rate (%)', 'PCT_LACCESS_POP15': 'Low access to stores (%)',
      'PCT_LACCESS_LOWI15': 'Low income and low access (%)', 'PERPOV10': 'Persistent poverty', 'METRO13': 'Metropolitan county',
      'POPLOSS10': 'Population-loss county', 'log_pestisit_kg': 'Pesticide applied (log10 kg)', 'pestisit_bilesik_sayisi': 'Pesticide compounds (count)'}
q = R[R.set == 'A'].head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(5.6, 4.2))
ax.barh([AD.get(v, v) for v in q.degisken], q.onem_ort, xerr=q.onem_sd, color='#55828B', ecolor='k', capsize=2)
ax.axvline(0, color='k', lw=.6); ax.set_xlabel('Decrease in $R^2$ (permutation importance)'); ax.grid(axis='x', alpha=.2)
for ext in ['png', 'pdf']: fig.savefig(f'sonuclar/revision/fig/FigureS6_importance_setA.{ext}', dpi=600, bbox_inches='tight')
print('BITTI')
