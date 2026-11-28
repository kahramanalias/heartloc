import sys, numpy as np, pandas as pd
sys.path.insert(0,'kod/revision'); from common import veri, koord
from scipy.spatial import cKDTree
d,FA,FB=veri(); xy,km=koord(d); y=d.kalp_olum_hizi.values; n=len(y)
nn=cKDTree(km).query(km,k=9)[1][:,1:]
def moran(v,perm=999,seed=0):
    z=v-v.mean(); I=(n/np.sum(z**2))*np.sum(z*z[nn].mean(1))/n*n/n
    I=np.sum(z*z[nn].mean(1))/np.sum(z**2)
    rng=np.random.default_rng(seed); c=0
    for _ in range(perm):
        zp=z[rng.permutation(n)]; c+=abs(np.sum(zp*zp[nn].mean(1))/np.sum(zp**2))>=abs(I)
    return I,(c+1)/(perm+1)
rows=[('Raw outcome','—',*moran(y))]
for r in ['Random','Spatial','State']:
    for arm,lab in [('GB','Gradient boosting'),('A_preds','Random forest'),('Ridge','Ridge'),('coords_RF','Coordinates only')]:
        o=np.load(f'sonuclar/revision/out/oof_{arm}_{r}.npy'); rows.append((lab+' residuals',r,*moran(y-o)))
M=pd.DataFrame(rows,columns=['Series','Validation','I','p']); print(M.round(3).to_string()); M.to_csv('sonuclar/revision/tab/S_moran.csv',index=False)
