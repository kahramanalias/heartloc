# -*- coding: utf-8 -*-
"""R2.5: nüfus ağırlıklı fit ve değerlendirme (duyarlılık)."""
import os, sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0,'kod/revision')
from common import veri, koord, bolucu
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
OUT='sonuclar/revision/out'
SONUC=os.environ.get('SONUC','kalp_olum_hizi'); TAG=os.environ.get('TAG','atlas')
d,FA,FB=veri()
if SONUC!='kalp_olum_hizi':
    w=pd.read_csv("veri/wonder_ilce.csv",dtype={'fips':str})
    d=d.merge(w[['fips',SONUC]],on='fips',how='inner'); d=d[d[SONUC].notna()].reset_index(drop=True)
p=pd.read_csv('veri/nufus_duzeltilmis.csv',dtype={'fips':str})
d=d.merge(p[['fips','nufus_35arti']],on='fips',how='left')
d['nufus_35arti']=d.nufus_35arti.fillna(d.nufus_35arti.median())
xy,km=koord(d); y=d[SONUC].values.astype(float); n=len(y); W=d.nufus_35arti.values
def wr2(y,o,w): mu=np.average(y,weights=w); return 1-np.sum(w*(y-o)**2)/np.sum(w*(y-mu)**2)
X={'A_preds':d[FA].apply(pd.to_numeric,errors='coerce').values,'B_preds':d[FB].apply(pd.to_numeric,errors='coerce').values,'coords_RF':xy}
CK=f"{OUT}/b8_{TAG}.csv"; cv=pd.read_csv(CK) if os.path.exists(CK) else pd.DataFrame(); t0=time.time()
for seed in [42+1000*s for s in range(int(os.environ.get('N_REP',5)))]:
  for rej in ['Random','Spatial','State']:
    spl,grp=bolucu(rej,d,xy,seed); folds=list(spl.split(xy,y,grp))
    for ad,Xa in X.items():
      for fitw in ['unweighted','pop_weighted']:
        if len(cv) and ((cv.arm==ad)&(cv.rejim==rej)&(cv.fit==fitw)&(cv.tohum==seed)).any(): continue
        oof=np.zeros(n)
        for tr,te in folds:
            m=Pipeline([("i",SimpleImputer(strategy="median")),("m",RandomForestRegressor(200,max_features=0.33 if Xa.shape[1]>2 else 1.0,min_samples_leaf=3,random_state=seed,n_jobs=-1))])
            kw={'m__sample_weight':W[tr]} if fitw=='pop_weighted' else {}
            oof[te]=m.fit(Xa[tr],y[tr],**kw).predict(Xa[te])
        rec=dict(arm=ad,rejim=rej,fit=fitw,tohum=seed,R2=1-np.mean((y-oof)**2)/np.var(y,ddof=1),R2_w=wr2(y,oof,W))
        cv=pd.concat([cv,pd.DataFrame([rec])],ignore_index=True); cv.to_csv(CK,index=False)
        print(f"{seed} {rej:<8}{ad:<10}{fitw:<13} R2={rec['R2']:.3f} R2w={rec['R2_w']:.3f} ({time.time()-t0:.0f}s)",flush=True)
print("BITTI")
