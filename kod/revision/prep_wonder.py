# -*- coding: utf-8 -*-
"""Builds veri/wonder_ilce.csv from a CDC WONDER export and the Census age file.
Usage: python kod/revision/prep_wonder.py <wonder_export.txt> <cc-est2024-agesex-all.csv>
WONDER query: Underlying Cause of Death 2018-2024 Single Race; Group by County; Ten-Year Age Groups 35-44..85+;
Years 2018-2022; ICD-10 113 Cause List '#Diseases of heart (I00-I09,I11,I13,I20-I51)'; Show Zero/Suppressed."""
import sys, io, re, numpy as np, pandas as pd
wsrc, csrc = sys.argv[1], sys.argv[2]
L = open(wsrc, encoding='latin-1').read().split('\n')
body = [l for l in L[:L.index('"---"')] if l.strip()]
w = pd.read_csv(io.StringIO('\n'.join(body)), sep='\t', dtype=str)
w = w[w['County Code'].notna()].copy()
w['fips'] = w['County Code'].str.zfill(5).replace({'46113': '46102'})   # Oglala Lakota
w['olum'] = pd.to_numeric(w.Deaths, errors='coerce')                     # 'Suppressed' -> NaN
c = pd.read_csv(csrc, encoding='latin-1')
age = {'35_44': ['AGE3539_TOT','AGE4044_TOT'], '45_54': ['AGE4549_TOT','AGE5054_TOT'], '55_64': ['AGE5559_TOT','AGE6064_TOT'],
       '65_74': ['AGE6569_TOT','AGE7074_TOT'], '75_84': ['AGE7579_TOT','AGE8084_TOT'], '85p': ['AGE85PLUS_TOT']}
cols = [x for v in age.values() for x in v]
c = c[c.YEAR.isin([2, 3, 4])].copy()          # 1 July 2020, 2021, 2022
c['fips'] = c.STATE.astype(str).str.zfill(2) + c.COUNTY.astype(str).str.zfill(3)
p = c.groupby('fips')[cols].mean().round(0)
p['nufus_35arti'] = p[cols].sum(axis=1)
# NCHS Health, United States (data.cdc.gov 7aq9-prdf), heart disease, age-specific crude rates, 2018..2022
R = {'35_44': [25.5,25.2,28.9,29.4,28.1], '45_54': [77.4,76.2,84.6,84.9,79.9], '55_64': [191.7,190.4,208.8,208.7,203.7],
     '65_74': [392.4,388.8,414.2,416.6,415.7], '75_84': [1008.3,991.2,1017.2,1044.2,1016.6], '85p': [3844.8,3798.3,3822.1,4078.7,3835.8]}
p['beklenen'] = 5 * sum(p[v].sum(axis=1) * np.mean(R[k]) / 1e5 for k, v in age.items())
p.reset_index().to_csv('veri/nufus_duzeltilmis.csv', index=False)
m = w[['fips','County','olum']].merge(p[['nufus_35arti','beklenen']].reset_index(), on='fips', how='inner')
an = pd.read_csv('veri/analiz_tablosu_setB.csv', dtype={'fips': str}); an['fips'] = an.fips.str.zfill(5)
m = m[m.fips.isin(an.fips)].copy()            # contiguous-US analysis counties
ok = m.olum.notna()
print('suppressed:', (~ok).sum(), '| observed/expected:', round(m.olum[ok].sum()/m.beklenen[ok].sum(), 3))
m['SMR'] = m.olum / m.beklenen
m['ham_hiz'] = m.olum / (5 * m.nufus_35arti) * 1e5
m['yas_std'] = m.SMR * (m.olum[ok].sum() / (5 * m.nufus_35arti[ok].sum()) * 1e5)
m[ok].to_csv('veri/wonder_ilce.csv', index=False)
