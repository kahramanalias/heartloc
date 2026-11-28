# -*- coding: utf-8 -*-
"""R1.2/R2.4/R2.7: tamponlu mekânsal CV, daha geniş tamponlar, tutarlı tekrarlar, eşleşmiş rastgele-çıkarma kontrolü."""
import os, sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0,'kod/revision')
from common import veri, koord, bolucu
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from scipy.spatial import cKDTree
OUT='sonuclar/revision/out'
SONUC=os.environ.get('SONUC','kalp_olum_hizi'); TAG=os.environ.get('TAG','atlas')
d,FA,FB=veri()
if SONUC!='kalp_olum_hizi':
    w=pd.read_csv("veri/wonder_ilce.csv",dtype={'fips':str})
    d=d.merge(w[['fips',SONUC]],on='fips',how='inner'); d=d[d[SONUC].notna()].reset_index(drop=True)
xy,km=koord(d); y=d[SONUC].values.astype(float); n=len(y); VAR=np.var(y,ddof=1)
XA=d[FA].apply(pd.to_numeric,errors='coerce').values
def model(ad,seed,Xtr,ytr,Xte):
    if ad=='coords_kNN': return KNeighborsRegressor(10,weights='distance').fit(Xtr,ytr).predict(Xte)
    p=Xtr.shape[1]
    m=Pipeline([("i",SimpleImputer(strategy="median")),("m",RandomForestRegressor(200,max_features=0.33 if p>2 else 1.0,
        min_samples_leaf=3,random_state=seed,n_jobs=-1))]).fit(Xtr,ytr)
    return m.predict(Xte)
X={'A_preds':XA,'coords_RF':xy,'coords_kNN':xy}
TAMPON=[0,25,50,100,200,300,500]
N_REP=int(os.environ.get('N_REP',5)); SEEDS=[42+1000*s for s in range(N_REP)]
CK=f"{OUT}/b5_{TAG}.csv"; cv=pd.read_csv(CK) if os.path.exists(CK) else pd.DataFrame()
t0=time.time()
for seed in SEEDS:
    spl,grp=bolucu('Spatial',d,xy,seed); folds=list(spl.split(xy,y,grp))
    # her katlama için: tampon içi eğitim ilçeleri
    near={}
    for f,(tr,te) in enumerate(folds):
        dist=cKDTree(km[te]).query(km[tr],k=1)[0]
        for b in TAMPON: near[(f,b)]=dist<=b if b>0 else np.zeros(len(tr),bool)
    for b in TAMPON:
        for tur in (['buffer'] if b==0 else ['buffer','control']):
            for ad in X:
                if len(cv) and ((cv.arm==ad)&(cv.tampon==b)&(cv.tur==tur)&(cv.tohum==seed)).any(): continue
                oof=np.zeros(n); cikan=[]; ntr=[]
                for f,(tr,te) in enumerate(folds):
                    m=near[(f,b)]
                    if tur=='control':
                        keep=np.ones(len(tr),bool); keep[np.random.default_rng(seed+f+b).choice(len(tr),m.sum(),replace=False)]=False
                    else: keep=~m
                    cikan.append((~keep).sum()); ntr.append(keep.sum())
                    oof[te]=model(ad,seed,X[ad][tr[keep]],y[tr[keep]],X[ad][te])
                rec=dict(arm=ad,tampon=b,tur=tur,tohum=seed,R2=1-np.mean((y-oof)**2)/VAR,
                         RMSE=np.sqrt(np.mean((y-oof)**2)),cikan_ort=np.mean(cikan),egitim_ort=np.mean(ntr))
                cv=pd.concat([cv,pd.DataFrame([rec])],ignore_index=True); cv.to_csv(CK,index=False)
                print(f"{seed} {b:>3}km {tur:<7}{ad:<10} R2={rec['R2']:.3f} çıkan={rec['cikan_ort']:.0f} ({time.time()-t0:.0f}s)",flush=True)
print("BITTI",time.time()-t0)
