# -*- coding: utf-8 -*-
"""Stage 1 — tüm model kolları × 3 rejim × tohumlar. Kontrol noktalı."""
import os, sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "kod/revision")
from common import veri, koord, bolucu, r2_havuz
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.interpolate import RBFInterpolator

OUT = "sonuclar/revision/out"
CK  = None
d, FA, FB = veri()
SONUC = os.environ.get('SONUC', 'ham_hiz')          # WONDER düzleştirilmemiş sonuç  [R2.4]
TAG   = os.environ.get('TAG', 'w_ham')
w = pd.read_csv("veri/wonder_ilce.csv", dtype={'fips': str})
d = d.merge(w[['fips', SONUC]], on='fips', how='inner')
d = d[d[SONUC].notna()].reset_index(drop=True)
xy, xykm = koord(d)
y = d[SONUC].values.astype(float)
print("n =", len(d), "| sonuç:", SONUC)
n = len(d); VAR = float(np.var(y, ddof=1))

# nonsense (anlamsız) yordayıcılar: mekânsal otokorelasyonlu rastgele alanlar  [R2.1]
def sahte_alanlar(seed=12345, k=5, uzunluk=3.0):
    rng = np.random.default_rng(seed)
    merkez = rng.uniform(xy.min(0), xy.max(0), size=(60, 2))
    F = []
    for j in range(k):
        w = rng.normal(size=60)
        D2 = ((xy[:, None, :] - merkez[None, :, :])**2).sum(-1)
        F.append(np.exp(-D2/(2*uzunluk**2)) @ w)
    F.append(pd.factorize(np.sort(d.ilce_ad.astype(str).values))[0][
        np.argsort(np.argsort(d.ilce_ad.astype(str).values))].astype(float))  # alfabetik sıra
    return np.c_[tuple(F)]
SAHTE = sahte_alanlar()

def X_of(ad):
    A = d[FA].apply(pd.to_numeric, errors='coerce').values.astype(float)
    B = d[FB].apply(pd.to_numeric, errors='coerce').values.astype(float)
    return {
        'A_preds':        A,
        'A_preds_coords': np.c_[A, xy],
        'B_preds':        B,
        'B_preds_coords': np.c_[B, xy],
        'coords_RF':      xy,
        'coords_TPS':     xy,
        'coords_GP':      xy,
        'coords_kNN':     xy,
        'nonsense':       SAHTE,
    }[ad]

def rf(seed, p):
    return Pipeline([("i", SimpleImputer(strategy="median")),
                     ("m", RandomForestRegressor(n_estimators=200,
                          max_features=0.33 if p > 2 else 1.0,
                          min_samples_leaf=3, random_state=seed, n_jobs=-1))])

def tahmin(ad, Xtr, ytr, Xte, seed):
    if ad == 'coords_TPS':                       # ince plaka spline (düzgün trend yüzeyi)
        f = RBFInterpolator(Xtr, ytr, kernel='thin_plate_spline', smoothing=len(ytr)*1.0)
        return f(Xte)
    if ad == 'coords_GP':                        # sabit çekirdekli kriging benzeri GP
        k = ConstantKernel(np.var(ytr)) * Matern(length_scale=2.0, nu=1.5) \
            + WhiteKernel(noise_level=np.var(ytr)*0.3)
        g = GaussianProcessRegressor(kernel=k, optimizer=None, normalize_y=True)
        g.fit(Xtr, ytr); return g.predict(Xte)
    if ad == 'coords_kNN':                       # sonucun kNN mekânsal gecikmesi
        m = KNeighborsRegressor(n_neighbors=10, weights='distance').fit(Xtr, ytr)
        return m.predict(Xte)
    m = rf(seed, Xtr.shape[1]).fit(Xtr, ytr)
    return m.predict(Xte)

ARMS   = ['A_preds','B_preds','coords_RF','coords_TPS','coords_GP','coords_kNN','nonsense']
REJIM  = ['Random','Spatial','State']
N_REP  = int(os.environ.get('N_REP', 10))
SEEDS  = [42 + 1000*s for s in range(N_REP)]

CK = f"{OUT}/a2_{TAG}.csv"
gorev = [(a, r, s) for s in SEEDS for r in REJIM for a in ARMS]
cv = pd.read_csv(CK) if os.path.exists(CK) else pd.DataFrame()
t0 = time.time(); yeni = 0
for i, (ad, rejim, seed) in enumerate(gorev):
    if len(cv) and ((cv.arm == ad) & (cv.rejim == rejim) & (cv.tohum == seed)).any():
        continue
    X = X_of(ad)
    spl, grp = bolucu(rejim, d, xy, seed)
    oof = np.full(n, np.nan); blok_id = np.full(n, -1)
    for f, (tr, te) in enumerate(spl.split(X, y, grp)):
        oof[te] = tahmin(ad, X[tr], y[tr], X[te], seed); blok_id[te] = f
    g = ~np.isnan(oof); res = y[g] - oof[g]
    # havuzlanmış ve blok-içi (merkezlenmiş) R²  [R2.5]
    ic_ss = 0.0; ic_tss = 0.0; blok_rmse = []
    for f in np.unique(blok_id[blok_id >= 0]):
        m = blok_id == f
        ic_ss  += np.sum((y[m]-oof[m])**2)
        ic_tss += np.sum((y[m]-y[m].mean())**2)
        blok_rmse.append(float(np.sqrt(np.mean((y[m]-oof[m])**2))))
    cv = pd.concat([cv, pd.DataFrame([{
        'arm': ad, 'rejim': rejim, 'tohum': seed,
        'R2_havuz': 1-np.mean(res**2)/VAR,
        'R2_blok_ici': 1-ic_ss/ic_tss,
        'RMSE': float(np.sqrt(np.mean(res**2))),
        'MAE': float(np.mean(np.abs(res))),
        'blok_RMSE_ort': float(np.mean(blok_rmse)),
        'blok_RMSE_sd': float(np.std(blok_rmse, ddof=1)),
    }])], ignore_index=True)
    cv.to_csv(CK, index=False)
    if seed == SEEDS[0]:
        np.save(f"{OUT}/{TAG}_oof_{ad}_{rejim}.npy", oof)
        np.save(f"{OUT}/{TAG}_blok_{ad}_{rejim}.npy", blok_id)
    yeni += 1
    print(f"[{i+1}/{len(gorev)}] {rejim:<8}{ad:<16}t={seed:<5} "
          f"R2h={cv.R2_havuz.iloc[-1]:.3f} R2ic={cv.R2_blok_ici.iloc[-1]:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)
print("BITTI", yeni, "yeni kayit", time.time()-t0)
