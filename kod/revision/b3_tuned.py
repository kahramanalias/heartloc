# -*- coding: utf-8 -*-
"""R2.2: koordinat tabanlarını rejime uygun iç CV ile ayarla (iç içe). Kontrol noktalı."""
import os, sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "kod/revision")
from common import veri, koord, bolucu, bloklar
from sklearn.model_selection import KFold, GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.interpolate import RBFInterpolator
OUT="sonuclar/revision/out"
SONUC=os.environ.get('SONUC','kalp_olum_hizi'); TAG=os.environ.get('TAG','atlas')
d,FA,FB=veri()
if SONUC!='kalp_olum_hizi':
    w=pd.read_csv("veri/wonder_ilce.csv",dtype={'fips':str})
    d=d.merge(w[['fips',SONUC]],on='fips',how='inner'); d=d[d[SONUC].notna()].reset_index(drop=True)
xy,_=koord(d); y=d[SONUC].values.astype(float); n=len(y); VAR=np.var(y,ddof=1)
GRID={'TPS':[0.001,0.01,0.1,1.0],'GP':[0.5,1.0,2.0,4.0],'kNN':[5,10,25,50]}
def fit_pred(m,p,Xtr,ytr,Xte):
    if m=='TPS': return RBFInterpolator(Xtr,ytr,kernel='thin_plate_spline',smoothing=len(ytr)*p)(Xte)
    if m=='GP':
        k=ConstantKernel(np.var(ytr))*Matern(length_scale=p,nu=1.5)+WhiteKernel(np.var(ytr)*0.3)
        return GaussianProcessRegressor(k,optimizer=None,normalize_y=True).fit(Xtr,ytr).predict(Xte)
    return KNeighborsRegressor(n_neighbors=p,weights='distance').fit(Xtr,ytr).predict(Xte)
def ic_bolme(rejim,tr,seed):
    if rejim=='Random': return list(KFold(3,shuffle=True,random_state=seed).split(tr))
    if rejim=='Spatial': g=bloklar(xy[tr],seed+7,5) % 3
    else: g=pd.factorize(d.eyalet.values[tr])[0]
    return list(GroupKFold(3).split(tr,groups=g))
N_REP=int(os.environ.get('N_REP',10)); SEEDS=[42+1000*s for s in range(N_REP)]
CK=f"{OUT}/b3_{TAG}.csv"; cv=pd.read_csv(CK) if os.path.exists(CK) else pd.DataFrame()
t0=time.time()
for seed in SEEDS:
  for rejim in ['Random','Spatial','State']:
    spl,grp=bolucu(rejim,d,xy,seed); folds=list(spl.split(xy,y,grp))
    for m in ['kNN','TPS','GP']:
        if len(cv) and ((cv.model==m)&(cv.rejim==rejim)&(cv.tohum==seed)).any(): continue
        oof=np.zeros(n); secim=[]; oracle={p:np.zeros(n) for p in GRID[m]}
        for tr,te in folds:
            sc=[]
            for p in GRID[m]:
                e=0.0
                for itr,ite in ic_bolme(rejim,tr,seed):
                    e+=np.sum((y[tr][ite]-fit_pred(m,p,xy[tr][itr],y[tr][itr],xy[tr][ite]))**2)
                sc.append(e)
            for p in GRID[m]: oracle[p][te]=fit_pred(m,p,xy[tr],y[tr],xy[te])
            pb=GRID[m][int(np.argmin(sc))]; secim.append(pb); oof[te]=oracle[pb][te]
        rec={'model':m,'rejim':rejim,'tohum':seed,'R2_ayarli':1-np.mean((y-oof)**2)/VAR,
             'secim':'|'.join(map(str,secim))}
        for p in GRID[m]: rec[f'R2_p{p}']=1-np.mean((y-oracle[p])**2)/VAR
        rec['R2_oracle']=max(rec[f'R2_p{p}'] for p in GRID[m])
        cv=pd.concat([cv,pd.DataFrame([rec])],ignore_index=True); cv.to_csv(CK,index=False)
        if seed==SEEDS[0]: np.save(f"{OUT}/b3_{TAG}_oof_{m}_{rejim}.npy",oof)
        print(f"{seed} {rejim:<8}{m:<4} ayarli={rec['R2_ayarli']:.3f} oracle={rec['R2_oracle']:.3f} [{rec['secim']}] ({time.time()-t0:.0f}s)",flush=True)
print("BITTI",time.time()-t0)
