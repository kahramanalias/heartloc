# -*- coding: utf-8 -*-
"""R1.3/R2.6: dissimilarity index (Meyer & Pebesma 2021), AOA, blok bazlı tablo, LISA."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'kod/revision'); from common import veri, koord, bolucu, bloklar
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
OUT='sonuclar/revision/out'
d,FA,FB=veri(); xy,km=koord(d); y=d.kalp_olum_hizi.values; n=len(y); VAR=np.var(y,ddof=1)
XA=d[FA].apply(pd.to_numeric,errors='coerce').values
seed=42
kayit=[]
for rejim in ['Spatial','Random','State']:
    spl,grp=bolucu(rejim,d,xy,seed); folds=list(spl.split(XA,y,grp))
    fold_id=np.zeros(n,int)
    for f,(tr,te) in enumerate(folds): fold_id[te]=f
    if rejim=='Spatial':                      # blok numarası Şekil 2 ile aynı kimlikten
        from common import bloklar
        bid=bloklar(xy,seed,10); fmap={f:int(np.bincount(bid[te]).argmax()) for f,(tr,te) in enumerate(folds)}
        folds=[folds[f] for f in sorted(fmap,key=lambda f:fmap[f])]
        for f,(tr,te) in enumerate(folds): fold_id[te]=f
    DI=np.zeros(n); THR=np.zeros(n); GEO=np.zeros(n)
    for f,(tr,te) in enumerate(folds):
        imp=SimpleImputer(strategy='median').fit(XA[tr]); Xtr=imp.transform(XA[tr]); Xte=imp.transform(XA[te])
        mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd==0]=1
        rf=RandomForestRegressor(200,max_features=0.33,min_samples_leaf=3,random_state=seed,n_jobs=-1).fit(Xtr,y[tr])
        wgt=rf.feature_importances_
        Ztr=(Xtr-mu)/sd*wgt; Zte=(Xte-mu)/sd*wgt
        rng=np.random.default_rng(f); sub=rng.choice(len(tr),min(1500,len(tr)),replace=False)
        dbar=pdist(Ztr[sub]).mean()
        DI[te]=cKDTree(Ztr).query(Zte,k=1)[0]/dbar
        # eğitim DI: her eğitim noktasının kendi katlaması dışındaki eğitim noktalarına uzaklığı
        ftr=fold_id[tr]; ditr=np.zeros(len(tr))
        for g in np.unique(ftr):
            m=ftr==g; ditr[m]=cKDTree(Ztr[~m]).query(Ztr[m],k=1)[0]/dbar
        q1,q3=np.percentile(ditr,[25,75]); THR[te]=q3+1.5*(q3-q1)
        GEO[te]=cKDTree(km[tr]).query(km[te],k=1)[0]
    oA=np.load(f'{OUT}/oof_A_preds_{rejim}.npy'); oC=np.load(f'{OUT}/oof_coords_RF_{rejim}.npy')
    kayit.append(pd.DataFrame({'fips':d.fips,'rejim':rejim,'fold':fold_id,'DI':DI,'AOA_esik':THR,
        'AOA_ici':DI<=THR,'geo_km':GEO,'y':y,'pred_A':oA,'pred_C':oC}))
    print(rejim,'medyan DI',np.median(DI).round(3),'AOA dışı %',(100*(DI>THR).mean()).round(1),'geo medyan',np.median(GEO).round(0),flush=True)
K=pd.concat(kayit); K.to_csv(f'{OUT}/b4_aoa_ilce.csv',index=False)

# --- LISA (yerel Moran), 8-NN satır-standart, 999 permütasyon
z=(y-y.mean())/y.std(); nn=cKDTree(km).query(km,k=9)[1][:,1:]
lag=z[nn].mean(1); Ii=z*lag
rng=np.random.default_rng(0); cnt=np.zeros(n)
for _ in range(999):
    zp=z[rng.permutation(n)]
    # koşullu permütasyon yaklaşık: komşuları rastgele başka ilçelerle değiştir
    cnt+=np.abs(z*zp[nn].mean(1))>=np.abs(Ii)
p=(cnt+1)/1000
q=np.where(p<0.05,np.select([(z>0)&(lag>0),(z<0)&(lag<0),(z>0)&(lag<0),(z<0)&(lag>0)],['HH','LL','HL','LH'],'ns'),'ns')
pd.DataFrame({'fips':d.fips,'lisa':q,'lisa_p':p}).to_csv(f'{OUT}/b4_lisa.csv',index=False)
print('LISA:',pd.Series(q).value_counts().to_dict())

# --- blok bazlı tablo (mekânsal, tohum 42)
S=K[K.rejim=='Spatial'].copy(); S['lisa']=q
rows=[]
for f,g in S.groupby('fold'):
    P=km[g.index.values] if False else km[S.index[S.fold==f]]
    eA=g.y-g.pred_A; eC=g.y-g.pred_C
    rows.append(dict(block=f+1,n=len(g),ext_EW_km=np.ptp(P[:,0]),ext_NS_km=np.ptp(P[:,1]),
        lon_c=d.lon.values[g.index].mean(),lat_c=d.lat.values[g.index].mean(),
        mean_y=g.y.mean(),sd_y=g.y.std(),
        RMSE_A=np.sqrt(np.mean(eA**2)),RMSE_C=np.sqrt(np.mean(eC**2)),
        bias_A=-eA.mean(),bias_C=-eC.mean(),
        R2w_A=1-np.sum(eA**2)/np.sum((g.y-g.y.mean())**2),
        R2w_C=1-np.sum(eC**2)/np.sum((g.y-g.y.mean())**2),
        DI_med=g.DI.median(),out_AOA_pct=100*(~g.AOA_ici).mean(),geo_med_km=g.geo_km.median(),
        HH_pct=100*(g.lisa=='HH').mean(),LL_pct=100*(g.lisa=='LL').mean()))
B=pd.DataFrame(rows); B['gap_RMSE']=B.RMSE_C-B.RMSE_A
pd.set_option('display.width',250); print(B.round(2).to_string())
B.to_csv(f'{OUT}/b4_blok_tablo.csv',index=False)
# hata ~ DI ve AOA içi/dışı
for rej in ['Spatial','Random','State']:
    g=K[K.rejim==rej]
    for lab,m in [('AOA içi',g.AOA_ici),('AOA dışı',~g.AOA_ici)]:
        if m.sum()<5: print(rej,lab,'n=',m.sum()); continue
        r2a=1-np.mean((g.y[m]-g.pred_A[m])**2)/VAR; r2c=1-np.mean((g.y[m]-g.pred_C[m])**2)/VAR
        print(f"{rej} {lab}: n={m.sum()} MAE_A={np.mean(np.abs(g.y[m]-g.pred_A[m])):.1f} MAE_C={np.mean(np.abs(g.y[m]-g.pred_C[m])):.1f} R2_A={r2a:.3f} R2_C={r2c:.3f}")
