# -*- coding: utf-8 -*-
"""R2.4: gözlenen sonuçtan BAĞIMSIZ bilinen bir sinyale üç farklı düzleştirici uygula."""
import os, sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0,'kod/revision')
from common import veri, koord, bolucu
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from scipy.spatial import cKDTree
OUT='sonuclar/revision/out'
d,FA,FB=veri(); xy,km=koord(d); n=len(d)
XA=SimpleImputer(strategy='median').fit_transform(d[FA].apply(pd.to_numeric,errors='coerce'))
Z=(XA-XA.mean(0))/XA.std(0)
p=pd.read_csv('veri/nufus_duzeltilmis.csv',dtype={'fips':str})
pop=d[['fips']].merge(p[['fips','nufus_35arti']],on='fips',how='left').nufus_35arti.values
pop=np.where(np.isnan(pop),np.nanmedian(pop),pop)
rng=np.random.default_rng(2024)
beta=rng.normal(size=Z.shape[1]); s=Z@beta; s=(s-s.mean())/s.std()     # bilinen sinyal, gözlenen sonuçla ilişkisiz katsayılar
eps=rng.normal(scale=np.sqrt(0.6667),size=n)                             # bağımsız gürültü -> gerçek R2 ~ 0.60
y0=s+eps
nn5=cKDTree(km).query(km,k=6)[1][:,1:]
D,I=cKDTree(km).query(km,k=200)
def kernel(v,h):
    W=np.exp(-0.5*(D/h)**2); return (W*v[I]).sum(1)/W.sum(1)
SC={'none':y0}
for w in [0.25,0.5,0.75]: SC[f'kNN5_w{w}']=(1-w)*y0+w*y0[nn5].mean(1)
for h in [25,50,100]: SC[f'Gauss_{h}km']=kernel(y0,h)
lam=1/(1+pop/np.median(pop))                                             # küçük ilçe daha çok büzülür (EB benzeri)
SC['EB_like_100km']=(1-lam)*y0+lam*kernel(y0,100)
def fit(ad,seed,Xtr,ytr,Xte):
    if ad=='coords_kNN': return KNeighborsRegressor(10,weights='distance').fit(Xtr,ytr).predict(Xte)
    pp=Xtr.shape[1]
    return Pipeline([("i",SimpleImputer(strategy="median")),("m",RandomForestRegressor(200,max_features=0.33 if pp>2 else 1.0,
        min_samples_leaf=3,random_state=seed,n_jobs=-1))]).fit(Xtr,ytr).predict(Xte)
X={'A_preds':XA,'coords_RF':xy,'coords_kNN':xy}
CK=f"{OUT}/b7_smooth.csv"; cv=pd.read_csv(CK) if os.path.exists(CK) else pd.DataFrame()
t0=time.time()
for seed in [42,1042,2042]:
  for rej in ['Random','Spatial']:
    spl,grp=bolucu(rej,d,xy,seed); folds=list(spl.split(xy,y0,grp))
    for sc,y in SC.items():
      for ad in X:
        if len(cv) and ((cv.senaryo==sc)&(cv.rejim==rej)&(cv.arm==ad)&(cv.tohum==seed)).any(): continue
        oof=np.zeros(n)
        for tr,te in folds: oof[te]=fit(ad,seed,X[ad][tr],y[tr],X[ad][te])
        rec=dict(senaryo=sc,rejim=rej,arm=ad,tohum=seed,R2_duz=1-np.mean((y-oof)**2)/np.var(y,ddof=1),
                 R2_gercek=1-np.mean((y0-oof)**2)/np.var(y0,ddof=1),R2_sinyal=np.corrcoef(oof,s)[0,1]**2,
                 cor_smooth_raw=np.corrcoef(y,y0)[0,1])
        cv=pd.concat([cv,pd.DataFrame([rec])],ignore_index=True); cv.to_csv(CK,index=False)
        print(f"{seed} {rej:<8}{sc:<15}{ad:<10} R2={rec['R2_duz']:.3f} ({time.time()-t0:.0f}s)",flush=True)
print("BITTI")
