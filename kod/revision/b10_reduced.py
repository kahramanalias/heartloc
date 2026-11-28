# -*- coding: utf-8 -*-
"""Eyalet düzeyinde sabit üç girdiyi (PC_FFRSALES12, FOODINSEC_12_14, FOODINSEC_15_17) çıkararak set A RF."""
import os,sys,time,warnings,numpy as np,pandas as pd; warnings.filterwarnings("ignore")
sys.path.insert(0,'kod/revision'); from common import veri,koord,bolucu
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
OUT=os.environ.get('OUTDIR','sonuclar/revision/out')
d,FA,FB=veri(); xy,_=koord(d); y=d.kalp_olum_hizi.values; n=len(y); V=np.var(y,ddof=1)
EY=['PC_FFRSALES12','FOODINSEC_12_14','FOODINSEC_15_17']; FR=[c for c in FA if c not in EY]
X=d[FR].apply(pd.to_numeric,errors='coerce').values
CK=f"{OUT}/b10_reduced.csv"; cv=pd.read_csv(CK) if os.path.exists(CK) else pd.DataFrame(); t0=time.time()
for seed in [42+1000*s for s in range(10)]:
  for rej in ['Random','Spatial','State']:
    if len(cv) and ((cv.rejim==rej)&(cv.tohum==seed)).any(): continue
    spl,grp=bolucu(rej,d,xy,seed); oof=np.zeros(n); fid=np.zeros(n,int)
    for f,(tr,te) in enumerate(spl.split(X,y,grp)):
        m=Pipeline([("i",SimpleImputer(strategy="median")),("m",RandomForestRegressor(200,max_features=0.33,min_samples_leaf=3,random_state=seed,n_jobs=-1))])
        oof[te]=m.fit(X[tr],y[tr]).predict(X[te]); fid[te]=f
    ic=1-sum(np.sum((y[fid==f]-oof[fid==f])**2) for f in range(10))/sum(np.sum((y[fid==f]-y[fid==f].mean())**2) for f in range(10))
    cv=pd.concat([cv,pd.DataFrame([dict(arm="A_reduced",rejim=rej,tohum=seed,R2_havuz=1-np.mean((y-oof)**2)/V,R2_blok_ici=ic,RMSE=np.sqrt(np.mean((y-oof)**2)),MAE=np.mean(np.abs(y-oof)))])],ignore_index=True); cv.to_csv(CK,index=False)
    print(seed,rej,round(cv.R2_havuz.iloc[-1],3),f"({time.time()-t0:.0f}s)",flush=True)
