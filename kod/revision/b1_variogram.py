import sys, numpy as np, pandas as pd
sys.path.insert(0,'kod/revision'); from common import veri, koord, bloklar
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform
d,FA,FB=veri(); xy,km=koord(d); n=len(d)
D=pdist(km)  # km
w=pd.read_csv('veri/wonder_ilce.csv',dtype={'fips':str})[['fips','ham_hiz','yas_std']]
d=d.merge(w,on='fips',how='left')
oofR=np.load('sonuclar/revision/out/oof_A_preds_Random.npy'); oofS=np.load('sonuclar/revision/out/oof_A_preds_Spatial.npy')
V={'Outcome (Atlas, smoothed)':d.kalp_olum_hizi.values,
   'Outcome (WONDER, crude)':d.ham_hiz.values,
   'Outcome (WONDER, age-standardised)':d.yas_std.values,
   'Physical inactivity':d.fiziksel_hareketsizlik.values,
   'Smoking':d.sigara_orani.values,
   'Median household income':d.MEDHHINC15.values,
   'Population aged 65+':d.PCT_65OLDER10.values,
   'Residual, predictor model (random CV)':d.kalp_olum_hizi.values-oofR,
   'Residual, predictor model (spatial CV)':d.kalp_olum_hizi.values-oofS}
bins=np.r_[0,25,50,75,100,150,200,300,400,500,600,800,1000,1250,1500,2000]
mid=(bins[:-1]+bins[1:])/2
def sph(h,c0,c,a): return np.where(h<a,c0+c*(1.5*h/a-0.5*(h/a)**3),c0+c)
def expo(h,c0,c,a): return c0+c*(1-np.exp(-3*h/a))
iu=np.triu_indices(n,1)
rows=[];curves={}
for k,v in V.items():
    z=(v-np.nanmean(v))/np.nanstd(v); ok=~np.isnan(z)
    zi=z[iu[0]]; zj=z[iu[1]]; g=~(np.isnan(zi)|np.isnan(zj))
    sv=0.5*(zi-zj)**2; pr=zi*zj
    idx=np.digitize(D,bins)-1
    gam=[];mor=[];npair=[]
    for b in range(len(mid)):
        m=(idx==b)&g
        gam.append(sv[m].mean()); a_=np.r_[zi[m],zj[m]]; b_=np.r_[zj[m],zi[m]]; mor.append(np.corrcoef(a_,b_)[0,1]); npair.append(m.sum())
    gam=np.array(gam); curves[k]=(gam,np.array(mor))
    m=mid<=1500
    try:
        p,_=curve_fit(expo,mid[m],gam[m],p0=[0.2,0.8,500],bounds=([0,0,10],[2,3,5000]),maxfev=20000)
        ps,_=curve_fit(sph,mid[m],gam[m],p0=[0.2,0.8,500],bounds=([0,0,10],[2,3,5000]),maxfev=20000)
    except Exception as e: p=ps=[np.nan]*3
    # korelogramın ilk kez 0.1 altına indiği mesafe
    mm=np.array(mor); cross=mid[np.argmax(mm<0.1)] if (mm<0.1).any() else np.nan
    rows.append(dict(variable=k,npair_first=npair[0],nugget_exp=p[0],sill_exp=p[0]+p[1],range_exp_km=p[2],
                     nugget_sph=ps[0],range_sph_km=ps[2],corr_below_0p1_km=cross,
                     corr_25km=mm[0],corr_200km=mm[np.searchsorted(mid,200)]))
R=pd.DataFrame(rows); pd.set_option('display.width',250)
print(R.round(3).to_string())
R.to_csv('sonuclar/revision/out/b1_variogram.csv',index=False)
np.save('sonuclar/revision/out/b1_curves.npy',{'mid':mid,'curves':curves},allow_pickle=True)
# blok boyutu: tohum başına blokların eşdeğer çapı
sz=[]
for s in [42+1000*i for i in range(10)]:
    b=bloklar(xy,s,10)
    for u in np.unique(b):
        P=km[b==u]; ext=np.ptp(P,0); sz.append(dict(seed=s,block=u,n=(b==u).sum(),ext_x=ext[0],ext_y=ext[1],
              min_ext=ext.min(),diag=np.hypot(*ext)))
S=pd.DataFrame(sz); S.to_csv('sonuclar/revision/out/b1_blok_boyut.csv',index=False)
print("\nblok boyutu km (10 tohum x 10 blok):\n",S[['n','ext_x','ext_y','min_ext','diag']].describe().round(0))
# en yakın komşu uzaklığı
DD=squareform(D); np.fill_diagonal(DD,np.inf); print("NN km medyan",np.median(DD.min(1)).round(1))

# --- Revizyon 2: blok genişliği, blokların kurulduğu döndürülmüş eksenlerde
def _rot_ext(k):
    out=[]
    for s in [42+1000*i for i in range(10)]:
        a=np.random.default_rng(s).uniform(0,np.pi); R_=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        P_=km@R_.T; b_=bloklar(xy,s,k)
        for u in np.unique(b_):
            e=np.ptp(P_[b_==u],0); out.append((s,u,e[0],e[1],e.min()))
    return pd.DataFrame(out,columns=['seed','block','ext_a','ext_b','min_ext'])
_rot_ext(10).to_csv('sonuclar/revision/out/b1_blok_boyut_rotated.csv',index=False)
_rot_ext(20).to_csv('sonuclar/revision/out/b1_blok_boyut_rotated_20.csv',index=False)
