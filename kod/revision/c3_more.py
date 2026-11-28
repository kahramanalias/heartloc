import numpy as np, pandas as pd
rng=np.random.default_rng(7)
def ci(x):
    x=np.asarray(x,float); bs=rng.choice(x,(10000,len(x))).mean(1); return f"{x.mean():+.3f} ({np.percentile(bs,2.5):+.3f} to {np.percentile(bs,97.5):+.3f})"
R=['Random','Spatial','State']
# WONDER std
for tag,gb in [('w_std','_w_std'),('w_ham',None)]:
    a=pd.read_csv(f'sonuclar/revision/out/a2_{tag}.csv')
    if gb: a=pd.concat([a,pd.read_csv(f'sonuclar/revision/out/b9_gbridge{gb}.csv')])
    print(f"\n=== WONDER {tag}: seeds {a.tohum.nunique()}")
    print(a.groupby(['arm','rejim']).agg(R2=('R2_havuz','mean'),sd=('R2_havuz','std'),ic=('R2_blok_ici','mean'),RMSE=('RMSE','mean')).round(3).unstack(1).to_string())
    P=a.pivot_table(index='tohum',columns=['arm','rejim'],values='R2_havuz')
    for base in ['coords_RF','coords_kNN','coords_GP','coords_TPS','nonsense']:
        inc={r:(P[('A_preds',r)]-P[(base,r)]).values for r in R}
        print(f"{base:<11} R {ci(inc['Random'])} | S {ci(inc['Spatial'])} | St {ci(inc['State'])} | S-R {ci(inc['Spatial']-inc['Random'])} | ratio {inc['Spatial'].mean()/inc['Random'].mean():.1f}")
    for r in R: print('B-A',r,ci(P[('B_preds',r)]-P[('A_preds',r)]))
# tampon
for tag in ['atlas','w_std','w_ham']:
    c=pd.read_csv(f'sonuclar/revision/out/b5_{tag}.csv'); print(f"\n=== buffer {tag} seeds {c.tohum.nunique()}")
    t=c.pivot_table(index=['tampon'],columns=['tur','arm'],values='R2',aggfunc='mean')
    out=pd.DataFrame({'removed':c[c.tur=='buffer'].groupby('tampon').cikan_ort.mean().round(0),
       'A_buf':t[('buffer','A_preds')],'A_ctl':t.get(('control','A_preds')),'C_buf':t[('buffer','coords_RF')],'C_ctl':t.get(('control','coords_RF')),
       'K_buf':t[('buffer','coords_kNN')],'K_ctl':t.get(('control','coords_kNN'))})
    out['gap_RF']=out.A_buf-out.C_buf; out['gap_kNN']=out.A_buf-out.K_buf
    # gap CI over seeds
    pv=c[c.tur=='buffer'].pivot_table(index='tohum',columns=['tampon','arm'],values='R2')
    out['gap_RF_CI']=[ci(pv[(b,'A_preds')]-pv[(b,'coords_RF')]) for b in out.index]
    print(out.round(3).to_string())
# env
e=pd.read_csv('sonuclar/revision/out/b6_env.csv'); print("\n=== env seeds",e.tohum.nunique()); P=e.pivot_table(index='tohum',columns='arm',values='R2_havuz')
print(e.groupby('arm')[['R2_havuz','R2_blok_ici']].mean().round(3))
for b in ['coords_RF','coords_kNN','coords_GP','coords_TPS','nonsense']: print('inc',b,ci(P['A_preds']-P[b]))
print('B-A',ci(P['B_preds']-P['A_preds']))
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'kod/revision'); from common import veri,cevre_bloklari
d,FA,FB=veri(); _S=sorted(e.tohum.unique()); _B=[np.bincount(cevre_bloklari(d,int(s))) for s in _S]
print('env block sizes per seed (min, max):', [(int(s),int(b.min()),int(b.max())) for s,b in zip(_S,_B)], '| overall', min(b.min() for b in _B), '-', max(b.max() for b in _B))
# weighted
for tag in ['atlas','w_std']:
    w=pd.read_csv(f'sonuclar/revision/out/b8_{tag}.csv'); print(f"\n=== weighted {tag}")
    print(w.groupby(['fit','arm','rejim'])[['R2','R2_w']].mean().round(3).unstack(2).to_string())
    P=w.pivot_table(index='tohum',columns=['fit','arm','rejim'],values='R2_w')
    for fit in ['unweighted','pop_weighted']:
        print(fit,'wInc R',ci(P[(fit,'A_preds','Random')]-P[(fit,'coords_RF','Random')]),'S',ci(P[(fit,'A_preds','Spatial')]-P[(fit,'coords_RF','Spatial')]),'St',ci(P[(fit,'A_preds','State')]-P[(fit,'coords_RF','State')]))
