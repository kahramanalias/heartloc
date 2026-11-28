# -*- coding: utf-8 -*-
"""Ortak veri hazırlama ve blok/rejim tanımları (revizyon analizleri)."""
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, GroupKFold

VERI = "veri/analiz_tablosu_setB.csv"
PLACES = ['sigortasiz_pay','asiri_alkol_orani','yillik_kontrol_orani','sigara_orani',
          'depresyon_yayginligi','sosyal_destek_yoklugu','gida_guvencesizligi','gida_yardimi',
          'kotu_saglik_algisi','sosyal_izolasyon','fiziksel_hareketsizlik','sik_ruhsal_sikinti',
          'yetersiz_uyku']
_FA_CACHE = []
ID = ['fips','kalp_olum_hizi','lat','lon','eyalet','ilce_ad','blok']

def veri():
    d = pd.read_csv(VERI)
    d['fips'] = d['fips'].astype(str).str.zfill(5)
    d = d.reset_index(drop=True)
    featB = [c for c in d.columns if c not in ID]
    featA = [c for c in featB if c not in PLACES]
    _FA_CACHE.clear(); _FA_CACHE.extend(featA)
    return d, featA, featB

def koord(d):
    """Kabaca eş-alan düzlem (km)."""
    lat = d.lat.values; lon = d.lon.values
    xy = np.c_[lon*np.cos(np.deg2rad(lat.mean())), lat]
    return xy, xy*111.0

def bloklar(xy, seed, k=10):
    """Döndürülmüş eşit-sayılı ızgara."""
    nx, ny = {5:(5,1), 10:(5,2), 20:(5,4)}[k]
    a = np.random.default_rng(seed).uniform(0, np.pi)
    R = np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    P = xy @ R.T
    b = np.empty(len(P), int)
    s = pd.qcut(P[:,0], nx, labels=False, duplicates='drop')
    for u in np.unique(s):
        m = s == u
        b[m] = u*ny + (pd.qcut(P[m,1], ny, labels=False, duplicates='drop') if ny>1 else 0)
    return b

def eyalet_bloklari(d, seed, k=10):
    """R2.7: eyaletleri katlamalara TOHUMA GÖRE RASTGELE dağıt (eskiden sabitti)."""
    ey = pd.factorize(d.eyalet.astype(str))[0]
    n_ey = ey.max()+1
    rng = np.random.default_rng(seed)
    atama = rng.permutation(n_ey) % k
    return atama[ey]

def cevre_bloklari(d, seed, k=10):
    """R2.6: özellik uzayında k-means (set A, standartlaştırılmış) ile çevresel bloklar."""
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    featB = [c for c in d.columns if c not in ID and c in _FA_CACHE]
    X = SimpleImputer(strategy='median').fit_transform(d[featB].apply(pd.to_numeric, errors='coerce'))
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    return KMeans(k, n_init=10, random_state=seed).fit_predict(X)

def bolucu(rejim, d, xy, seed, k=10):
    if rejim == 'Random':
        return KFold(k, shuffle=True, random_state=seed), None
    if rejim == 'Spatial':
        return GroupKFold(k), bloklar(xy, seed, k)
    if rejim == 'State':
        return GroupKFold(k), eyalet_bloklari(d, seed, k)
    if rejim == 'Env':
        return GroupKFold(k), cevre_bloklari(d, seed, k)
    raise ValueError(rejim)

def r2_havuz(y, oof, var_toplam):
    g = ~np.isnan(oof)
    return 1 - np.mean((y[g]-oof[g])**2)/var_toplam
