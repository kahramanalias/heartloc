# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, sys, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,'kod/revision'); from common import veri, koord, bloklar
from scipy.spatial import cKDTree
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':600,'font.family':'DejaVu Sans'})
O='sonuclar/revision/out'; F='sonuclar/revision/fig'
d,FA,FB=veri(); xy,km=koord(d); y=d.kalp_olum_hizi.values
RED='#c44e52'; GREY='#8c8c8c'; BLUE='#4c72b0'
def save(fig,name): fig.savefig(f'{F}/{name}.png',bbox_inches='tight'); fig.savefig(f'{F}/{name}.pdf',bbox_inches='tight'); plt.close(fig)
# ---------- Figure 2: bloklar, tohum 42 (blok tablosuyla aynı numaralama)
b=bloklar(xy,42,10)
fig,ax=plt.subplots(figsize=(7.2,3.6)); cm=plt.get_cmap('tab10')
for k in range(10):
    m=b==k; ax.scatter(d.lon[m],d.lat[m],s=1.6,color=cm(k),label=f'Block {k+1} (n={m.sum()})')
ax.set_xlabel('Longitude (°)'); ax.set_ylabel('Latitude (°)'); ax.grid(alpha=.2)
ax.set_aspect(1/np.cos(np.deg2rad(d.lat.mean())))
for k in range(10):
    m=b==k; ax.text(np.median(d.lon[m]),np.median(d.lat[m]),str(k+1),fontsize=8,fontweight='bold',ha='center',va='center',bbox=dict(boxstyle='round,pad=0.15',fc='white',ec='none',alpha=.8))
ax.legend(ncol=1,fontsize=6.5,frameon=False,markerscale=3,loc='center left',bbox_to_anchor=(1.01,0.5)); save(fig,'Figure2_spatial_blocks')
# ---------- Figure 3: rejimler
a=pd.concat([pd.read_csv(f'{O}/a1_metrik.csv'),pd.read_csv(f'{O}/b9_gbridge.csv')])
arms=[('Ridge','Ridge'),('A_preds','Random\nforest'),('GB','Gradient\nboosting'),('coords_RF','Coordinates\nonly')]
fig,ax=plt.subplots(figsize=(6.3,3.6)); w=.26
for j,(r,c,lab) in enumerate([('Random',BLUE,'Random 10-fold'),('Spatial',RED,'Spatial blocks'),('State','#8c8c8c','State hold-out')]):
    mu=[a[(a.arm==k)&(a.rejim==r)].R2_havuz.mean() for k,_ in arms]; sd=[a[(a.arm==k)&(a.rejim==r)].R2_havuz.std() for k,_ in arms]
    ax.bar(np.arange(4)+(j-1)*w,mu,w,yerr=sd,capsize=3,color=c,label=lab,edgecolor='white',linewidth=.5)
ax.set_xticks(range(4)); ax.set_xticklabels([l for _,l in arms]); ax.set_ylabel('Pooled out-of-fold $R^2$')
ax.legend(ncol=3,frameon=False,fontsize=7,loc='upper right'); ax.grid(axis='y',alpha=.2); ax.set_ylim(0,.7); save(fig,'Figure3_validation_regimes')
# ---------- Figure 4: (a) coğrafi uzaklık, (b) özellik uzayı farklılığı
K=pd.read_csv(f'{O}/b4_aoa_ilce.csv',dtype={'fips':str})
S=K[K.rejim=='Spatial'].reset_index(drop=True); Rn=K[K.rejim=='Random'].reset_index(drop=True)
gS=np.load(f'{O}/oof_GB_Spatial.npy'); gR=np.load(f'{O}/oof_GB_Random.npy')
cS=np.load(f'{O}/oof_coords_RF_Spatial.npy'); cR=np.load(f'{O}/oof_coords_RF_Random.npy')
bins=[0,25,50,100,200,400,1e9]; labs=['0–25','25–50','50–100','100–200','200–400','>400']
band=pd.cut(S.geo_km,bins,labels=False,include_lowest=True).values
q=np.quantile(S.DI,np.linspace(0,1,7)); dband=np.clip(np.searchsorted(q,S.DI,side='right')-1,0,5)
qR=np.quantile(Rn.DI,np.linspace(0,1,7)); dbandR=np.clip(np.searchsorted(qR,Rn.DI,side='right')-1,0,5)
out=[]
fig,axs=plt.subplots(1,2,figsize=(7.2,3.2),sharey=True)
for ax,bb,bbR,xl,xt in [(axs[0],band,band,'Distance to nearest training county\nunder spatial blocking (km)',labs),
                        (axs[1],dband,dbandR,'Dissimilarity index (sextile)',[f'Q{i+1}' for i in range(6)])]:
    for pred,predR,lab,col,mk in [(gS,gR,'Gradient boosting',RED,'o'),(cS,cR,'Coordinates only',GREY,'s')]:
        e=np.abs(y-pred); eR=np.abs(y-predR)
        mS=[e[bb==i].mean() for i in range(6)]; mR=[eR[bbR==i].mean() for i in range(6)]
        ax.plot(range(6),mS,'-',marker=mk,color=col,label=f'{lab}, spatial CV')
        ax.plot(range(6),mR,'--',marker=mk,color=col,alpha=.45,label=f'{lab}, random CV')
        out.append(pd.DataFrame({'panel':xl,'model':lab,'band':xt,'MAE_spatial':mS,'MAE_random':mR,'n':[int((bb==i).sum()) for i in range(6)]}))
    if ax is axs[0]: xt=[f'{l}\nn={int((bb==i).sum()):,}' for i,l in enumerate(xt)]
    ax.set_xticks(range(6)); ax.set_xticklabels(xt,fontsize=6.5 if ax is axs[0] else 7); ax.set_xlabel(xl); ax.grid(alpha=.2)
axs[0].set_ylabel('Mean absolute error (deaths per 100,000)'); axs[0].legend(fontsize=6,frameon=False)
axs[0].set_title('a',loc='left',fontweight='bold'); axs[1].set_title('b',loc='left',fontweight='bold')
save(fig,'Figure4_distance_decay'); pd.concat(out).to_csv(f'{O}/d1_fig4_values.csv',index=False)
print(pd.concat(out).round(1).to_string())
print('DI sextile edges spatial:',np.round(q,3),' random:',np.round(qR,3))
# ---------- S: korelogram
C=np.load(f'{O}/b1_curves.npy',allow_pickle=True).item(); mid=C['mid']
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),sharey=True)
st={'Outcome (Atlas, smoothed)':('k','-'),'Outcome (WONDER, crude)':('k',':'),'Outcome (WONDER, age-standardised)':('k','--'),'Physical inactivity':(BLUE,'-'),'Smoking':(BLUE,'--'),
    'Median household income':('#55a868','-'),'Population aged 65+':('#55a868','--'),
    'Residual, predictor model (random CV)':(RED,'--'),'Residual, predictor model (spatial CV)':(RED,'-')}
for k,(gam,mor) in C['curves'].items():
    ax=axs[1] if k.startswith('Residual') or k.startswith('Outcome') else axs[0]
    c,ls=st[k]; ax.plot(mid,mor,ls,color=c,marker='.',ms=3,label=k)
for ax in axs:
    ax.axhline(0.1,color='grey',lw=.6,ls=':'); ax.axhline(0,color='grey',lw=.5); ax.set_xscale('log'); ax.set_xlabel('Distance band midpoint (km)'); ax.grid(alpha=.2); ax.legend(fontsize=6,frameon=False,loc='lower left') if ax is axs[0] else ax.legend(fontsize=6,frameon=False,loc='upper center',bbox_to_anchor=(0.5,-0.2),ncol=2)
axs[0].set_ylabel('Correlation between county pairs'); axs[0].set_title('a  Predictors',loc='left',fontweight='bold'); axs[1].set_title('b  Outcome and residuals',loc='left',fontweight='bold')
save(fig,'FigureS1_correlogram')
# ---------- S: on blok yapılandırması
fig,axs=plt.subplots(2,5,figsize=(7.2,2.9))
for i,ax in enumerate(axs.flat):
    bb=bloklar(xy,42+1000*i,10)
    for k in range(10): m=bb==k; ax.scatter(d.lon[m],d.lat[m],s=.15,color=cm(k))
    ax.set_title(f'Repetition {i+1}',fontsize=7); ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
save(fig,'FigureS2_block_configurations')
# ---------- S: LISA + blok sınırları
L=pd.read_csv(f'{O}/b4_lisa.csv',dtype={'fips':str})
fig,axs=plt.subplots(1,2,figsize=(7.2,2.6))
col={'HH':RED,'LL':BLUE,'HL':'#e7a1a3','LH':'#a6bddb','ns':'#dddddd'}
for lab in ['ns','LL','HH','HL','LH']:
    m=L.lisa.values==lab; axs[0].scatter(d.lon[m],d.lat[m],s=1,color=col[lab],label={'ns':'Not significant','HH':'High–high','LL':'Low–low','HL':'High–low','LH':'Low–high'}[lab])
axs[0].legend(fontsize=6,frameon=False,markerscale=4,loc='upper left',bbox_to_anchor=(0.0,-0.02),ncol=3); axs[0].set_title('a  Local Moran clusters (p < 0.05)',loc='left',fontsize=8,fontweight='bold')
B=pd.read_csv(f'{O}/b4_blok_tablo.csv')
x=np.arange(10); axs[1].bar(x-.2,B.HH_pct,.4,color=RED,label='High–high'); axs[1].bar(x+.2,B.LL_pct,.4,color=BLUE,label='Low–low')
axs[1].set_xticks(x); axs[1].set_xticklabels([str(i) for i in B.block]); axs[1].set_xlabel('Block (Figure 2)'); axs[1].set_ylabel('% of block counties'); axs[1].legend(fontsize=6,frameon=False)
axs[1].set_title('b  Cluster membership by block',loc='left',fontsize=8,fontweight='bold')
for ax in [axs[0]]: ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
save(fig,'FigureS4_lisa_blocks')
print('ok')
