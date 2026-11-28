# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, os, sys
sys.path.insert(0,'kod/revision'); from common import veri, koord, bolucu
O='sonuclar/revision/out'; T='sonuclar/revision/tab'; rng=np.random.default_rng(20261125)
def ci(x,B=10000):
    x=np.asarray(x,float); m=x.mean()
    bs=rng.choice(x,(B,len(x))).mean(1); return m,np.percentile(bs,2.5),np.percentile(bs,97.5)
def f(m,l,u,p=3): return f"{m:+.{p}f} ({l:+.{p}f} to {u:+.{p}f})"
a=pd.read_csv(f'{O}/a1_metrik.csv'); g=pd.read_csv(f'{O}/b9_gbridge.csv')
al=pd.concat([a,g]); P=al.pivot_table(index='tohum',columns=['arm','rejim'],values='R2_havuz')
PI=al.pivot_table(index='tohum',columns=['arm','rejim'],values='R2_blok_ici')
R=['Random','Spatial','State']
# ---- Tablo 1
rows=[]
for r in R:
    for arm,lab in [('GB','Gradient boosting'),('A_preds','Random forest'),('Ridge','Ridge'),('coords_RF','Coordinates only (RF)')]:
        s=al[(al.arm==arm)&(al.rejim==r)]
        rows.append(dict(Validation=r,Model=lab,R2=f"{s.R2_havuz.mean():.3f} ± {s.R2_havuz.std():.3f}",
            R2_within=f"{s.R2_blok_ici.mean():.3f}",RMSE=f"{s.RMSE.mean():.1f} ± {s.RMSE.std():.1f}",MAE=f"{s.MAE.mean():.1f}"))
t1=pd.DataFrame(rows); t1.to_csv(f'{T}/T1.csv',index=False); print(t1.to_string())
# ---- Tablo 2: artışlar, sabit tabanlar 10 tohum
rows=[]
for base,lab in [('coords_RF','Random forest'),('coords_kNN','k-nearest neighbours (k=10)'),('coords_GP','Gaussian process (Matérn)'),('coords_TPS','Thin-plate spline'),('nonsense','Nonsense predictors (RF)')]:
    inc={r:(P[('A_preds',r)]-P[(base,r)]).values for r in R}
    rec=dict(Baseline=lab)
    for r in R: rec[f'base_{r}']=f"{P[(base,r)].mean():.3f}"; rec[f'inc_{r}']=f(*ci(inc[r]))
    rec['diff_Spatial_minus_Random']=f(*ci(inc['Spatial']-inc['Random']))
    rec['diff_State_minus_Random']=f(*ci(inc['State']-inc['Random']))
    rat=inc['Spatial']/inc['Random']
    rec['ratio']=f"{inc['Spatial'].mean()/inc['Random'].mean():.1f}" if inc['Random'].mean()>0.02 else 'n.d.'
    rows.append(rec)
t2=pd.DataFrame(rows); t2.to_csv(f'{T}/T2_increments.csv',index=False); print(t2.T.to_string())
# ---- ayarlı tabanlar (3 tohum, eşleşmiş)
b=pd.read_csv(f'{O}/b3_atlas.csv'); sd=b.tohum.unique()
rows=[]
for m,lab in [('kNN','k-nearest neighbours'),('GP','Gaussian process'),('TPS','Thin-plate spline')]:
    rec=dict(Baseline=lab+' (tuned)')
    for r in R:
        q=b[(b.model==m)&(b.rejim==r)].set_index('tohum').loc[sd]
        A=P.loc[sd,('A_preds',r)].values
        rec[f'tuned_{r}']=f"{q.R2_ayarli.mean():.3f}"; rec[f'inc_tuned_{r}']=f"{(A-q.R2_ayarli.values).mean():+.3f}"
        rec[f'oracle_{r}']=f"{q.R2_oracle.mean():.3f}"; rec[f'inc_oracle_{r}']=f"{(A-q.R2_oracle.values).mean():+.3f}"
    rows.append(rec)
t2b=pd.DataFrame(rows); t2b.to_csv(f'{T}/T2b_tuned.csv',index=False); print(t2b.T.to_string())
# grid duyarlılığı
gc=[c for c in b.columns if c.startswith('R2_p')]
gs=b.groupby(['model','rejim'])[gc].mean().round(3); gs.to_csv(f'{T}/S_grid.csv'); print(gs)
# ---- Set B - Set A ve ayrıştırma
rows=[]
for r in R:
    dBA=(P[('B_preds',r)]-P[('A_preds',r)]).values
    incA=(P[('A_preds',r)]-P[('coords_RF',r)]).values; incB=(P[('B_preds',r)]-P[('coords_RF',r)]).values
    rows.append(dict(Validation=r,A=f"{P[('A_preds',r)].mean():.3f}",B=f"{P[('B_preds',r)].mean():.3f}",
        coords=f"{P[('coords_RF',r)].mean():.3f}",A_coords=f"{P[('A_preds_coords',r)].mean():.3f}",B_coords=f"{P[('B_preds_coords',r)].mean():.3f}",
        incA=f(*ci(incA)),incB=f(*ci(incB)),B_minus_A=f(*ci(dBA)),
        coords_added_to_A=f(*ci((P[('A_preds_coords',r)]-P[('A_preds',r)]).values)),
        coords_added_to_B=f(*ci((P[('B_preds_coords',r)]-P[('B_preds',r)]).values)),
        A_within=f"{PI[('A_preds',r)].mean():.3f}",B_within=f"{PI[('B_preds',r)].mean():.3f}",C_within=f"{PI[('coords_RF',r)].mean():.3f}"))
t3=pd.DataFrame(rows); t3.to_csv(f'{T}/T3_setB_decomp.csv',index=False); print(t3.T.to_string())
# ---- blok/eyalet bootstrap (tohum 42 OOF)
d,FA,FB=veri(); xy,_=koord(d); y=d.kalp_olum_hizi.values; V=np.var(y,ddof=1)
rows=[]
for r in ['Spatial','State']:
    oA=np.load(f'{O}/oof_A_preds_{r}.npy'); oC=np.load(f'{O}/oof_coords_RF_{r}.npy')
    grp=np.load(f'{O}/blok_A_preds_{r}.npy') if r=='Spatial' else pd.factorize(d.eyalet)[0]
    u=np.unique(grp); idx={k:np.where(grp==k)[0] for k in u}
    res=[]
    for _ in range(5000):
        s=rng.choice(u,len(u)); ii=np.concatenate([idx[k] for k in s]); yy=y[ii]; vv=np.var(yy,ddof=1)
        a_=1-np.mean((yy-oA[ii])**2)/vv; c_=1-np.mean((yy-oC[ii])**2)/vv; res.append((a_,c_,a_-c_))
    res=np.array(res); lo,hi=np.percentile(res,[2.5,97.5],axis=0)
    rows.append(dict(Validation=r,unit='block' if r=='Spatial' else 'state',n_units=len(u),
        A=f"{1-np.mean((y-oA)**2)/V:.3f} ({lo[0]:.3f}–{hi[0]:.3f})",C=f"{1-np.mean((y-oC)**2)/V:.3f} ({lo[1]:.3f}–{hi[1]:.3f})",
        inc=f"{(np.mean((y-oC)**2)-np.mean((y-oA)**2))/V:+.3f} ({lo[2]:+.3f} to {hi[2]:+.3f})"))
t4=pd.DataFrame(rows); t4.to_csv(f'{T}/S_blockboot.csv',index=False); print(t4.to_string())
