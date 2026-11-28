# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, json, sys
sys.path.insert(0,'kod/revision')
rng=np.random.default_rng(20261125)
_MEMO={}
def ci(x,p=3,sign=True):
    x=np.asarray(x,float); bs=rng.choice(x,(10000,len(x))).mean(1)   # rng her çağrıda ilerler (diğer aralıklar değişmez)
    key=(tuple(np.round(x,10)),p,sign)
    if key in _MEMO: return _MEMO[key]                                  # aynı veri -> aynı aralık (Tablo 2 = Tablo 4)
    l,u=np.percentile(bs,[2.5,97.5])
    def f(v, q=p):
        s=f"{v:+.{q}f}" if sign else f"{v:.{q}f}"
        return f(v, q+1) if float(s)==0 and v!=0 and q<p+2 else s   # e.g. -0.0004 instead of -0.000
    out=f"{f(x.mean())} ({f(l)} to {f(u)})"; _MEMO[key]=out; return out
def m(x,p=3): return f"{np.mean(x):.{p}f}"
T={}
a=pd.read_csv('sonuclar/revision/out/a1_metrik.csv'); g=pd.read_csv('sonuclar/revision/out/b9_gbridge.csv'); b3=pd.read_csv('sonuclar/revision/out/b3_atlas.csv'); rd=pd.read_csv('sonuclar/revision/out/b10_reduced.csv')
al=pd.concat([a,g,rd]); P=al.pivot_table(index='tohum',columns=['arm','rejim'],values='R2_havuz')
PI=al.pivot_table(index='tohum',columns=['arm','rejim'],values='R2_blok_ici')
RL={'Random':'Random 10-fold','Spatial':'Spatial blocks','State':'State hold-out'}
R=['Random','Spatial','State']
# Table 1
rows=[]
for r in R:
    for arm,lab in [('GB','Gradient boosting'),('A_preds','Random forest'),('Ridge','Ridge'),('A_reduced','Random forest, no state-level inputs'),('coords_RF','Coordinates only (RF)')]:
        s=al[(al.arm==arm)&(al.rejim==r)]
        rows.append([RL[r],lab,f"{s.R2_havuz.mean():.3f} ± {s.R2_havuz.std():.3f}",f"{s.R2_blok_ici.mean():.3f}",f"{s.RMSE.mean():.1f} ± {s.RMSE.std():.1f}",f"{s.MAE.mean():.1f}"])
T['T1']=rows
# Table 2: baselines (tuned for kNN/GP/TPS)
Tu={mm:b3[b3.model==mm].pivot_table(index='tohum',columns='rejim',values='R2_ayarli').loc[P.index] for mm in ['kNN','GP','TPS']}
base={'Random forest (coordinates)':{r:P[('coords_RF',r)].values for r in R},
      'k-nearest neighbours, tuned':{r:Tu['kNN'][r].values for r in R},
      'Gaussian process, tuned':{r:Tu['GP'][r].values for r in R},
      'Thin-plate spline, tuned':{r:Tu['TPS'][r].values for r in R},
      'Nonsense predictors (RF)':{r:P[('nonsense',r)].values for r in R}}
rows=[]
for lab,v in base.items():
    inc={r:P[('A_preds',r)].values-v[r] for r in R}
    for r in R:
        d='—' if r=='Random' else ci(inc[r]-inc['Random'])
        rows.append([lab if r=='Random' else '',RL[r],m(v[r]),ci(inc[r]),d])
T['T2']=rows
T['T2_pred']={r:m(P[('A_preds',r)]) for r in R}
# Table 3 buffered atlas
def buf(tag):
    c=pd.read_csv(f'sonuclar/revision/out/b5_{tag}.csv'); out=[]
    for bb in sorted(c.tampon.unique()):
        q=c[c.tampon==bb]; get=lambda arm,t: q[(q.arm==arm)&(q.tur==t)].R2.mean()
        pv=q[q.tur=='buffer'].pivot_table(index='tohum',columns='arm',values='R2')
        f=lambda arm: f"{get(arm,'buffer'):.3f}"+("" if bb==0 else f" ({get(arm,'control'):.3f})")
        out.append(['None' if bb==0 else f"{bb} km",f"{q.cikan_ort.mean():.0f}",f('A_preds'),f('coords_RF'),f('coords_kNN'),ci(pv['A_preds']-pv['coords_RF'],sign=False)])
    return out
T['T3']=buf('atlas'); T['S11']=buf('w_std'); T['S11b']=buf('w_ham')
# Table 4 smoothing robustness
def outc(tag,gb,lab,tuned=False):
    w=pd.read_csv(f'sonuclar/revision/out/{tag}.csv')
    if gb: w=pd.concat([w,pd.read_csv(f'sonuclar/revision/out/{gb}.csv')])
    Q=w.pivot_table(index='tohum',columns=['arm','rejim'],values='R2_havuz'); rows=[]
    for r in R:
        if tuned:
            bl={'kNN':Tu['kNN'][r].values,'GP':Tu['GP'][r].values,'TPS':Tu['TPS'][r].values,'RF':Q[('coords_RF',r)].values}
        else:
            bl={'kNN':Q[('coords_kNN',r)].values,'GP':Q[('coords_GP',r)].values,'TPS':Q[('coords_TPS',r)].values,'RF':Q[('coords_RF',r)].values}
        best=max(bl,key=lambda k:np.mean(bl[k])); A=Q[('A_preds',r)].values
        incs=[np.mean(A-bl[k]) for k in bl]
        rows.append([lab if r=='Random' else '',RL[r],m(A),m(bl['RF']),f"{np.mean(bl[best]):.3f} ({best})",ci(A-bl['RF']),f"{min(incs):+.3f} to {max(incs):+.3f}"])
    return rows
T['T4']=outc('a1_metrik',None,'Published, smoothed (2022–2024; n = 3,099)',tuned=True)+outc('a2_w_std','b9_gbridge_w_std','WONDER, age-standardised (2018–2022; n = 3,077)')+outc('a2_w_ham',None,'WONDER, crude (2018–2022; n = 3,077)')
# S2 paired reductions
rows=[]
for tgt in ['Spatial','State']:
    for arm,lab in [('GB','Gradient boosting'),('A_preds','Random forest'),('Ridge','Ridge'),('A_reduced','Random forest, no state-level inputs'),('coords_RF','Coordinates only')]:
        rows.append([f"Random → {'Spatial' if tgt=='Spatial' else 'State'}",lab,m(P[(arm,'Random')]),m(P[(arm,tgt)]),ci(P[(arm,'Random')]-P[(arm,tgt)],sign=False)])
T['S2']=rows
# S3 set B / decomposition
rows=[]
for r in R:
    rows.append([RL[r],m(P[('coords_RF',r)]),m(P[('A_preds',r)]),m(P[('A_preds_coords',r)]),m(P[('B_preds',r)]),m(P[('B_preds_coords',r)]),
                 ci(P[('B_preds',r)]-P[('A_preds',r)]),ci(P[('A_preds_coords',r)]-P[('A_preds',r)])])
T['S3']=rows
T['S3w']={r:[m(PI[('A_preds',r)]),m(PI[('B_preds',r)])] for r in R}
# S4 Moran
M=pd.read_csv('sonuclar/revision/tab/S_moran.csv'); T['S4']=[[s,v,f"{i:.3f}",f"{p:.3f}"] for s,v,i,p in M.values]
# S6 correlogram
V=pd.read_csv('sonuclar/revision/out/b1_variogram.csv'); T['S6']=[[r.variable,f"{r.corr_25km:.2f}",f"{r.corr_200km:.2f}",f"{r.corr_below_0p1_km:.0f}",f"{r.range_sph_km:.0f}",f"{r.range_exp_km:.0f}"] for r in V.itertuples()]
S=pd.read_csv('sonuclar/revision/out/b1_blok_boyut_rotated.csv'); T['S6b']={k:f"{S[k].median():.0f} ({S[k].quantile(.25):.0f}–{S[k].quantile(.75):.0f}; range {S[k].min():.0f}–{S[k].max():.0f})" for k in ['ext_a','ext_b','min_ext']}
# S7 tuned grid
G={'kNN':[5,10,25,50],'GP':[0.5,1.0,2.0,4.0],'TPS':[0.001,0.01,0.1,1.0]}
rows=[]
for mm,lab in [('kNN','k-nearest neighbours (k)'),('GP','Gaussian process (length scale, °)'),('TPS','Thin-plate spline (penalty × n)')]:
    q=b3[b3.model==mm]
    for r in R:
        qq=q[q.rejim==r]
        rows.append([lab if r=='Random' else '',RL[r]]+[f"{qq[f'R2_p{p}'].mean():.3f}" for p in G[mm]]+[f"{qq.R2_ayarli.mean():.3f}",
           ', '.join(f"{k}: {v}" for k,v in pd.Series('|'.join(qq.secim).split('|')).value_counts().items())])
T['S7']=rows
T['S7w']={r:[m(PI[(k,r)]) for k in ['coords_RF','coords_kNN','coords_GP','coords_TPS','nonsense']] for r in R}
# S8 per block
B=pd.read_csv('sonuclar/revision/out/b4_blok_tablo.csv')
T['S8']=[[str(r.block),f"{r.lon_c:.0f}, {r.lat_c:.0f}",f"{r.ext_EW_km:.0f} × {r.ext_NS_km:.0f}",f"{r.mean_y:.0f} ({r.sd_y:.0f})",f"{r.RMSE_A:.1f}",f"{r.RMSE_C:.1f}",f"{r.RMSE_C-r.RMSE_A:+.1f}",f"{r.R2w_A:.2f}",f"{r.R2w_C:.2f}",f"{r.DI_med:.2f}",f"{r.out_AOA_pct:.1f}",f"{r.HH_pct:.0f} / {r.LL_pct:.0f}"] for r in B.itertuples()]
K=pd.read_csv('sonuclar/revision/out/b4_aoa_ilce.csv'); VAR=np.var(K[K.rejim=='Spatial'].y,ddof=1); rows=[]
for r in R:
    q=K[K.rejim==r]
    for lab,mk in [('Inside AOA',q.AOA_ici),('Outside AOA',~q.AOA_ici)]:
        rows.append([RL[r] if lab=='Inside AOA' else '',lab,f"{mk.sum()} ({100*mk.mean():.1f} %)",f"{1-np.mean((q.y[mk]-q.pred_A[mk])**2)/VAR:.3f}",f"{1-np.mean((q.y[mk]-q.pred_C[mk])**2)/VAR:.3f}",f"{np.mean(np.abs(q.y[mk]-q.pred_A[mk])):.1f}",f"{np.mean(np.abs(q.y[mk]-q.pred_C[mk])):.1f}"])
T['S8b']=rows
# S9 env
e=pd.read_csv('sonuclar/revision/out/b6_env.csv'); E=e.pivot_table(index='tohum',columns='arm',values='R2_havuz'); EI=e.pivot_table(index='tohum',columns='arm',values='R2_blok_ici')
T['S9']=[[lab,m(E[k]),m(EI[k]),('—' if k in('A_preds','B_preds') else ci(E['A_preds']-E[k]))] for k,lab in [('A_preds','Predictor set A (RF)'),('B_preds','Predictor set B (RF)'),('coords_RF','Coordinates, RF'),('coords_kNN','Coordinates, kNN (k = 10)'),('coords_GP','Coordinates, GP (fixed)'),('coords_TPS','Coordinates, TPS (fixed)'),('nonsense','Nonsense predictors')]]
# S10 WONDER all arms
rows=[]
for tag,gb,lab in [('a2_w_std','b9_gbridge_w_std','Age-standardised'),('a2_w_ham',None,'Crude')]:
    w=pd.read_csv(f'sonuclar/revision/out/{tag}.csv')
    if gb: w=pd.concat([w,pd.read_csv(f'sonuclar/revision/out/{gb}.csv')])
    for k,kl in [('GB','Gradient boosting'),('A_preds','Random forest, set A'),('Ridge','Ridge'),('B_preds','Random forest, set B'),('coords_RF','Coordinates, RF'),('coords_kNN','Coordinates, kNN'),('coords_GP','Coordinates, GP'),('coords_TPS','Coordinates, TPS'),('nonsense','Nonsense predictors')]:
        q=w[w.arm==k]
        if len(q)==0: continue
        rows.append([lab if k in('GB','A_preds') and (k=='GB' or gb is None) else '',kl]+[f"{q[q.rejim==r].R2_havuz.mean():.3f} ({q[q.rejim==r].R2_blok_ici.mean():.3f})" for r in R])
T['S10']=rows
# S12 smoothing sim
c=pd.read_csv('sonuclar/revision/out/b7_smooth.csv')
order=[('none','None'),('kNN5_w0.25','5-NN, w = 0.25'),('kNN5_w0.5','5-NN, w = 0.5'),('kNN5_w0.75','5-NN, w = 0.75'),('Gauss_25km','Gaussian, 25 km'),('Gauss_50km','Gaussian, 50 km'),('Gauss_100km','Gaussian, 100 km'),('EB_like_100km','Population-dependent, 100 km')]
rows=[]
for k,lab in order:
    q=c[c.senaryo==k]; f=lambda r,a: q[(q.rejim==r)&(q.arm==a)].R2_duz.mean()
    rows.append([lab,f"{q.cor_smooth_raw.mean():.2f}",f"{f('Random','A_preds'):.3f}",f"{f('Random','coords_RF'):.3f}",f"{f('Spatial','A_preds'):.3f}",f"{f('Spatial','coords_RF'):.3f}",f"{f('Random','A_preds')-f('Random','coords_RF'):+.3f}",f"{f('Spatial','A_preds')-f('Spatial','coords_RF'):+.3f}"])
T['S12']=rows
# S13 weighted
rows=[]
for tag,lab in [('atlas','Published, smoothed'),('w_std','WONDER, age-standardised')]:
    w=pd.read_csv(f'sonuclar/revision/out/b8_{tag}.csv')
    for fit,fl in [('unweighted','Unweighted'),('pop_weighted','Population-weighted')]:
        for k,kl in [('A_preds','Set A'),('B_preds','Set B'),('coords_RF','Coordinates (RF)')]:
            q=w[(w.fit==fit)&(w.arm==k)]
            rows.append([lab if (fit=='unweighted' and k=='A_preds') else '',fl if k=='A_preds' else '',kl]+[f"{q[q.rejim==r].R2.mean():.3f} / {q[q.rejim==r].R2_w.mean():.3f}" for r in R])
T['S13']=rows
# S14 block bootstrap
bb=pd.read_csv('sonuclar/revision/tab/S_blockboot.csv'); T["S14"]=[[RL[r.Validation],("47 states and DC" if r.unit=="state" else f"{r.unit} ({r.n_units})"),r.A,r.C,r.inc] for r in bb.itertuples()]
import sys as _s; _s.path.insert(0,'.'); from common import veri as _v
_d,_FA,_FB=_v()
DESC={'pestisit_bilesik_sayisi':('Distinct agricultural pesticide compounds applied (count)','USGS','2019','County'),
'log_pestisit_kg':('Agricultural pesticide applied, log10(1 + kg active ingredient)','USGS','2019','County'),
'PCT_LACCESS_POP15':('Population with low access to stores (%)','USDA Atlas','2015','County'),
'PCT_LACCESS_LOWI15':('Low income and low access to stores (%)','USDA Atlas','2015','County'),
'GROCPTH16':('Grocery stores per 1,000 population','USDA Atlas','2016','County'),
'SUPERCPTH16':('Supercentres and club stores per 1,000 population','USDA Atlas','2016','County'),
'CONVSPTH16':('Convenience stores per 1,000 population','USDA Atlas','2016','County'),
'SPECSPTH16':('Specialised food stores per 1,000 population','USDA Atlas','2016','County'),
'SNAPSPTH17':('SNAP-authorised stores per 1,000 population','USDA Atlas','2017','County'),
'FFRPTH16':('Fast-food restaurants per 1,000 population','USDA Atlas','2016','County'),
'FSRPTH16':('Full-service restaurants per 1,000 population','USDA Atlas','2016','County'),
'PC_FFRSALES12':('Expenditure per capita, fast food','USDA Atlas','2012','State'),
'FOODINSEC_12_14':('Household food insecurity (%, three-year average)','USDA Atlas','2012–2014','State'),
'FOODINSEC_15_17':('Household food insecurity (%, three-year average)','USDA Atlas','2015–2017','State'),
'RECFACPTH16':('Recreation and fitness facilities per 1,000 population','USDA Atlas','2016','County'),
'PCT_NHBLACK10':('Non-Hispanic Black population (%)','USDA Atlas','2010','County'),
'PCT_HISP10':('Hispanic population (%)','USDA Atlas','2010','County'),
'PCT_NHASIAN10':('Non-Hispanic Asian population (%)','USDA Atlas','2010','County'),
'PCT_NHNA10':('Non-Hispanic American Indian or Alaska Native population (%)','USDA Atlas','2010','County'),
'PCT_65OLDER10':('Population aged 65 and over (%)','USDA Atlas','2010','County'),
'PCT_18YOUNGER10':('Population under 18 (%)','USDA Atlas','2010','County'),
'MEDHHINC15':('Median household income (USD)','USDA Atlas','2015','County'),
'POVRATE15':('Poverty rate (%)','USDA Atlas','2015','County'),
'PERPOV10':('Persistent-poverty county (0/1)','USDA Atlas','2010','County'),
'CHILDPOVRATE15':('Child poverty rate (%)','USDA Atlas','2015','County'),
'METRO13':('Metropolitan county (0/1)','USDA Atlas','2013','County'),
'POPLOSS10':('Population-loss county (0/1)','USDA Atlas','2010','County'),
'sigara_orani':('Current smoking (%)','CDC PLACES','2024 release','County (modelled)'),
'fiziksel_hareketsizlik':('No leisure-time physical activity (%)','CDC PLACES','2024 release','County (modelled)'),
'asiri_alkol_orani':('Binge drinking (%)','CDC PLACES','2024 release','County (modelled)'),
'yetersiz_uyku':('Short sleep duration (%)','CDC PLACES','2024 release','County (modelled)'),
'sik_ruhsal_sikinti':('Frequent mental distress (%)','CDC PLACES','2024 release','County (modelled)'),
'depresyon_yayginligi':('Depression (%)','CDC PLACES','2024 release','County (modelled)'),
'sosyal_izolasyon':('Feeling socially isolated (%)','CDC PLACES','2024 release','County (modelled)'),
'sosyal_destek_yoklugu':('Lack of social and emotional support (%)','CDC PLACES','2024 release','County (modelled)'),
'gida_guvencesizligi':('Food insecurity in the past 12 months (%)','CDC PLACES','2024 release','County (modelled)'),
'gida_yardimi':('Received food stamps in the past 12 months (%)','CDC PLACES','2024 release','County (modelled)'),
'yillik_kontrol_orani':('Routine check-up in the past year (%)','CDC PLACES','2024 release','County (modelled)'),
'sigortasiz_pay':('Current lack of health insurance, ages 18–64 (%)','CDC PLACES','2024 release','County (modelled)'),
'kotu_saglik_algisi':('Fair or poor self-rated health (%)','CDC PLACES','2024 release','County (modelled)')}
rows=[]
for c in _FB:
    de,so,yr,lv=DESC[c]
    rows.append(['A' if c in _FA else 'B',de,so,yr,lv,str(int(pd.to_numeric(_d[c],errors='coerce').isna().sum()))])
T['S1b']=rows
PR=pd.concat([a,rd]).pivot_table(index='tohum',columns=['arm','rejim'],values='R2_havuz')
_ir=(PR[('A_reduced','Random')]-PR[('coords_RF','Random')]).values; _is=(PR[('A_reduced','Spatial')]-PR[('coords_RF','Spatial')]).values
T['S3_reduced_dd']=ci(_is-_ir)
pd.DataFrame(T['T1'],columns=['Validation','Model','Pooled R2','Within-block R2','RMSE','MAE']).to_csv('sonuclar/revision/tab/T1.csv',index=False)
json.dump(T,open('sonuclar/revision/tab/tables.json','w'),ensure_ascii=False,indent=1)
for k,v in T.items():
    print('==',k); 
    if isinstance(v,list): [print('   ',x) for x in v[:20]]
    else: print('   ',v)
