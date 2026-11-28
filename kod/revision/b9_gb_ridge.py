# -*- coding: utf-8 -*-
"""Tablo 1 için GB ve Ridge, set A, 3 rejim × 10 tohum (eyalet katlamaları artık tohuma göre değişiyor)."""
import os, sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0,'kod/revision')
from common import veri, koord, bolucu
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
OUT='sonuclar/revision/out'
SONUC=os.environ.get('SONUC','kalp_olum_hizi'); TAG=os.environ.get('TAG','')
d,FA,FB=veri()
if SONUC!='kalp_olum_hizi':
    w=pd.read_csv("veri/wonder_ilce.csv",dtype={'fips':str})
    d=d.merge(w[['fips',SONUC]],on='fips',how='inner'); d=d[d[SONUC].notna()].reset_index(drop=True)
xy,_=koord(d); y=d[SONUC].values.astype(float); n=len(y); VAR=np.var(y,ddof=1)
XA=d[FA].apply(pd.to_numeric,errors='coerce').values
def mk(ad,seed):
    if ad=='Ridge': return Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("m",RidgeCV(alphas=np.logspace(-3,4,40)))])
    return Pipeline([("m",HistGradientBoostingRegressor(max_iter=400,learning_rate=0.06,max_leaf_nodes=31,l2_regularization=1.0,early_stopping=True,random_state=seed))])
CK=f"{OUT}/b9_gbridge{TAG}.csv"; cv=pd.read_csv(CK) if os.path.exists(CK) else pd.DataFrame(); t0=time.time()
for seed in [42+1000*s for s in range(10)]:
  for rej in ['Random','Spatial','State']:
    spl,grp=bolucu(rej,d,xy,seed); folds=list(spl.split(XA,y,grp))
    for ad in ['GB','Ridge']:
        if len(cv) and ((cv.arm==ad)&(cv.rejim==rej)&(cv.tohum==seed)).any(): continue
        oof=np.zeros(n); fid=np.zeros(n,int); fw=[]
        for f,(tr,te) in enumerate(folds):
            oof[te]=mk(ad,seed).fit(XA[tr],y[tr]).predict(XA[te]); fid[te]=f
            fw.append(1-np.sum((y[te]-oof[te])**2)/np.sum((y[te]-y[te].mean())**2))
        ic=sum(np.sum((y[fid==f]-oof[fid==f])**2) for f in range(10))/sum(np.sum((y[fid==f]-y[fid==f].mean())**2) for f in range(10))
        res=y-oof
        rec=dict(arm=ad,rejim=rej,tohum=seed,R2_havuz=1-np.mean(res**2)/VAR,R2_blok_ici=1-ic,R2_foldwise=np.mean(fw),
                 RMSE=np.sqrt(np.mean(res**2)),MAE=np.mean(np.abs(res)))
        cv=pd.concat([cv,pd.DataFrame([rec])],ignore_index=True); cv.to_csv(CK,index=False)
        if seed==42: np.save(f"{OUT}/{TAG}oof_{ad}_{rej}.npy",oof)
        print(f"{seed} {rej:<8}{ad:<6} R2={rec['R2_havuz']:.3f} ({time.time()-t0:.0f}s)",flush=True)
print("BITTI")
