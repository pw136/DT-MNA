from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Patch, Rectangle, PathPatch
from matplotlib.path import Path as MPath
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results' / 'figure_data'
NEW = ROOT / 'results' / 'csf_strengthening_v2'
FIG = ROOT / 'results' / 'figures'
COL = {'SG-global': '#A2ABB1', 'SG-time': '#555F69', 'G-time': '#5379AF', 'MG-time': '#349F92',
       'C2-time': '#D45E78', 'G-het': '#AF842D', 'H': '#DDE5EA'}
LAB = {'SG-global': r'$SG_{\mathrm{c}}$', 'SG-time': r'$SG_{\mathrm{t}}$', 'G-time': r'$G(t)$',
       'MG-time': r'$MG(t)$', 'C2-time': r'$C_2(t)$', 'G-het': r'$G_{\mathrm{het}}(t)$'}
INK = '#293842'
GRAY = '#6F7C86'
NU = ['#3D8097', '#C87962']
EVIDENCE = {}
HEADERS = []
plt.rcParams.update({'font.family': 'Arial', 'mathtext.fontset': 'stix', 'font.size': 8.8,
                     'axes.labelsize': 8.6, 'axes.titlesize': 9.4, 'xtick.labelsize': 7.5,
                     'ytick.labelsize': 7.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': .65, 'xtick.major.width': .6, 'ytick.major.width': .6,
                     'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': GRAY, 'ytick.color': GRAY,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})

def save_fig(f, name):
    FIG.mkdir(parents=True, exist_ok=True)
    f.savefig(FIG / f'{name}.pdf', bbox_inches='tight', pad_inches=.055,
              metadata={'Title': f'Results Figure {name}', 'Creator': 'Reproducible Matplotlib figure'})
    f.savefig(FIG / f'{name}.png', dpi=280, bbox_inches='tight', pad_inches=.055)
    plt.close(f)

def header(f,x,letter,title,y=.957):
 l=f.text(x,y,letter+'.',weight='bold',fontsize=11,va='baseline')
 t=f.text(x+.023,y,title,fontsize=10.2,va='baseline')
 f.canvas.draw();a=l.get_transform().transform(l.get_position())[1];b=t.get_transform().transform(t.get_position())[1]
 assert abs(a-b)<1e-8
 HEADERS.append(dict(figure=int(f.get_label()),panel=title,label=letter+'.',baseline_difference_px=float(abs(a-b)),dot_suffix=True))
def tidy(a,axis='x'):
 a.set_axisbelow(True);a.grid(axis=axis,color='#E8EDF0',lw=.6);a.tick_params(length=3)
def band_path(x0,x1,y0,y1,z0,z1):
 mx=(x0+x1)/2
 return MPath([(x0,y0),(mx,y0),(mx,z0),(x1,z0),(x1,z1),(mx,z1),(mx,y1),(x0,y1),(x0,y0)],[MPath.MOVETO]+[MPath.CURVE4]*3+[MPath.LINETO]+[MPath.CURVE4]*3+[MPath.CLOSEPOLY])
def figure7():
 h=pd.read_csv(NEW/'fit_metrics.csv');h=h[h.case.str.startswith('hetero')].copy();d=pd.read_csv(NEW/'decisions.csv')
 models=list(LAB);names=[f'synthetic_{i:02d}' for i in range(8)]+['Email-Enron','High-School','Primary-School','NDC-classes-full']
 short=[f'S{i+1:02d}' for i in range(8)]+['Enron','High-School','Primary-School','NDC-classes']
 f=plt.figure(figsize=(9.6,4.45));f.set_label('7')
 a=f.add_axes([.075,.15,.245,.735]);rail=f.add_axes([.332,.15,.056,.735],sharey=a)
 header(f,.012,'a','Accuracy and parameterization')
 for j,m in enumerate(models):
  y=5-j
  if j%2==0:a.axhspan(y-.48,y+.48,color='#F5F8FA',zorder=0)
  for nu,sgn in [(1,1),(2,-1)]:
   z=h[(h.model==m)&(h.nu==nu)].sort_values('context');vals=np.log10(z.test_rmse.to_numpy());xx=np.linspace(vals.min(),vals.max(),160);density=gaussian_kde(vals,bw_method=.42)(xx);width=density/density.max()*.32
   a.fill_between(xx,y,y+sgn*width,color=COL[m],alpha=.43 if nu==1 else .15,lw=0,zorder=2)
   a.plot(xx,y+sgn*width,color=COL[m],lw=.85,ls='-' if nu==1 else '--',zorder=3)
   q=np.quantile(vals,[.25,.5,.75]);a.plot(q[[0,2]],[y+sgn*.08]*2,c=COL[m],lw=2.2,zorder=4);a.scatter(q[1],y+sgn*.08,c='white',edgecolor=COL[m],s=15,lw=.9,zorder=5)
   for k,row in enumerate(z.itertuples()):
    isreal=row.domain!='synthetic';a.scatter(np.log10(row.test_rmse),y+sgn*(.37+(k%2)*.055),s=9,marker='^' if isreal else 'o',c=COL[m],edgecolor='white',lw=.25,zorder=6)
  counts=np.unique(h[h.model==m].parameters);lo,hi=counts.min(),counts.max();rail.plot([np.log10(lo),np.log10(hi)],[y,y],c=COL[m],lw=3.3,solid_capstyle='round');rail.scatter(np.log10(counts),np.full(len(counts),y),s=8,c=COL[m],edgecolor='white',lw=.35,zorder=4)
  rail.text(1.50,y+.25,str(int(lo)) if lo==hi else f'{int(lo)}–{int(hi)}',ha='center',va='bottom',fontsize=7.0,color=COL[m])
 a.set(xlim=(-3.0,-.53),ylim=(-.65,6.3),yticks=range(6),yticklabels=[LAB[m] for m in models[::-1]],xticks=[-3,-2,-1],xticklabels=[r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$'],xlabel='Held-out probability RMSE')
 a.spines['left'].set_visible(False);a.tick_params(axis='y',length=0,pad=7);tidy(a)
 a.legend(handles=[Line2D([],[],c=GRAY,lw=1.4,label=r'$\nu=1$'),Line2D([],[],c=GRAY,lw=1.4,ls='--',label=r'$\nu=2$')],ncol=2,loc='upper left',bbox_to_anchor=(.0,.995),frameon=False,fontsize=8,handlelength=1.4,columnspacing=1.0)
 rail.set(xlim=(-.1,3.12),xticks=[0,3],xticklabels=['1',r'$10^3$']);rail.tick_params(axis='y',left=False,labelleft=False);rail.spines['left'].set_visible(False);rail.text(1.5,5.97,'Parameters',ha='center',fontsize=7.6);rail.set_xlabel('Count',fontsize=7.7)
 a.text(.015,.865,'12 networks / mechanism',transform=a.transAxes,fontsize=7.0,color=GRAY)
 b=f.add_axes([.456,.15,.155,.735]);header(f,.425,'b','Paired RMSE ratio')
 p=h.pivot(index=['context','nu'],columns='model',values='test_rmse').reset_index();p['ratio']=p['G-het']/p['C2-time'];p['network']=p.context.str.replace('hetero_','',regex=False).str.replace(r'_nu[12]$','',regex=True);mat=p.pivot(index='network',columns='nu',values='ratio').loc[names]
 for nu,x,side,c in [(1,0,-1,NU[0]),(2,1,1,NU[1])]:
  vals=mat[nu].to_numpy();yy=np.linspace(vals.min(),vals.max(),140);dens=gaussian_kde(vals,bw_method=.45)(yy);wid=dens/dens.max()*.27
  b.fill_betweenx(yy,x,x+side*wid,facecolor=c,alpha=.24,lw=0);b.plot(x+side*wid,yy,c=c,lw=.8)
  q=np.quantile(vals,[.25,.5,.75]);b.plot([x+side*.33]*2,q[[0,2]],c=c,lw=2.6);b.scatter(x+side*.33,q[1],s=18,c='white',edgecolor=c,lw=1,zorder=5)
 for j,name in enumerate(names):
  pair=mat.loc[name].values;b.plot([0,1],pair,c='#AABBC5',lw=.8,alpha=.62,zorder=2)
  b.scatter([0,1],pair,s=23 if j>=8 else 17,marker='^' if j>=8 else 'o',facecolor=NU,edgecolor='white',lw=.5,zorder=5)
 b.set(xlim=(-.55,1.55),ylim=(0,.86),xticks=[0,1],xticklabels=[r'$\nu=1$',r'$\nu=2$'],yticks=[0,.2,.4,.6],ylabel=r'$\mathrm{RMSE}_{G_{\mathrm{het}}}/\mathrm{RMSE}_{C_2}$');tidy(b,'y')
 b.spines['bottom'].set_visible(False);b.tick_params(axis='x',length=0,pad=8)
 b.legend(handles=[Line2D([],[],marker='o',ls='',c=GRAY,ms=3.5,label='Synthetic'),Line2D([],[],marker='^',ls='',c=GRAY,ms=4,label='Real structure')],loc='upper left',bbox_to_anchor=(.015,.995),frameon=False,fontsize=7.0,labelspacing=.4,handletextpad=.4)
 b.text(.50,.78,'24/24 ratios < 1',transform=b.transAxes,ha='center',fontsize=7.0,color=GRAY)
 c=f.add_axes([.665,.075,.33,.825]);header(f,.655,'c','Budget-dependent selection')
 c.set_aspect('equal');c.set(xlim=(-2.22,2.22),ylim=(-2.60,2.18));c.axis('off');records=[]
 radii={1:[.62,.77,.92,1.07],2:[1.32,1.47,1.62,1.77]};budgets=[.02,.05,.10,.20];start=66;step=26
 for j,name in enumerate(names):
  hi=start-j*step;lo=hi-step+1.25;theta=np.deg2rad((hi+lo)/2)
  for nu in (1,2):
   for q,r in zip(budgets,radii[nu]):
    row=d[(d.context==f'hetero_{name}_nu{nu}')&d.expanded&(d.budget==q)].iloc[0]
    c.add_patch(Wedge((0,0),r+.135,lo,hi,width=.135,fc=COL['G-het'] if row.chosen=='G-het' else COL['H'],ec='white',lw=.45))
    if row.unsafe:c.plot((r+.067)*np.cos(theta),(r+.067)*np.sin(theta),'x',c='#A52E50',ms=4.4,mew=1.0)
    records.append(dict(network=name,nu=nu,budget=q,chosen=row.chosen,unsafe=bool(row.unsafe)))
  radius=2.01;angle=(hi+lo)/2;rot=angle-90
  if angle< -90:rot+=180
  c.text(radius*np.cos(theta),radius*np.sin(theta),short[j],ha='center',va='center',rotation=rot,rotation_mode='anchor',fontsize=7.0,color=INK if j>=8 else GRAY)

 for nu in (1,2):
  for q,r in zip(budgets,radii[nu]):c.text(.13,r+.064,f'{q:.2f}',ha='center',va='center',fontsize=6.7,color=GRAY)
 c.text(0,.19,'29 / 96',ha='center',fontsize=15,color=COL['G-het'],weight='bold');c.text(0,-.03,r'$G_{\mathrm{het}}(t)$',ha='center',fontsize=11);c.text(0,-.23,'accepted',ha='center',fontsize=7.7,color=GRAY)
 c.text(-.25,.96,r'$\nu=1$',ha='center',va='center',fontsize=7.2,color=INK);c.text(.13,2.075,r'$\varepsilon$',ha='center',fontsize=9);c.text(-.25,1.62,r'$\nu=2$',ha='center',fontsize=7.2,color=INK)
 c.legend(handles=[Patch(fc=COL['H'],label=r'$H(t)$'),Patch(fc=COL['G-het'],label=r'$G_{\mathrm{het}}(t)$'),Line2D([],[],c='#A52E50',marker='x',ms=4,ls='',label='Exceeded')],loc='lower center',bbox_to_anchor=(.5,.015),ncol=3,frameon=False,fontsize=7.0,handlelength=1,columnspacing=.8,handletextpad=.4)
 assert len(h)==144 and len(mat)==12 and len(records)==96 and sum(r['unsafe'] for r in records)==2 and sum(r['chosen']=='G-het' for r in records)==29
 EVIDENCE['figure7']={'observations':len(h),'paired_ratios':p[['network','nu','ratio']].to_dict('records'),'decision_cells':records,'distribution_display':'Gaussian KDE on log10(RMSE), bandwidth factor 0.42, clipped to observed ranges, each half scaled to equal maximum width; raw markers and empirical quartiles shown. Panel b KDE bandwidth 0.45 on untransformed ratios. Categorical vertical offsets only.','original_selection_all_H':bool((d[~d.expanded].chosen=='H').all())}
 save_fig(f,'images7')
def figure8():
 d=pd.read_csv(DATA/'budget_choices.csv');s=pd.read_csv(DATA/'equal_budget_audit_summary.csv');raw=pd.read_csv(DATA/'equal_budget_audit.csv')
 f=plt.figure(figsize=(9.6,4.20));f.set_label('8');header(f,.012,'a','Representation pathways');header(f,.515,'b','Network-level audit profiles')
 a=f.add_axes([.062,.155,.415,.715]);cols={'SG':'#9CA9B2','G':COL['G-time'],'MG':COL['MG-time'],'SC2':COL['C2-time']};labels={'SG':r'$SG$','G':r'$G(t)$','MG':r'$MG(t)$','SC2':r'$C_2(t)$'}
 budgets=sorted(d.budget.unique());strengths=sorted(d.synergy.unique());lookup=d.pivot(index='synergy',columns='budget',values='selected');positions={};height=.59;gap=.23
 for j,q in enumerate(budgets):
  y=0
  for m in cols:
   ss=[v for v in strengths if lookup.loc[v,q]==m]
   for v in ss:positions[j,v]=(y,y+height,m);y+=height
   if ss:y+=gap
 for j in range(5):
  for strength in strengths:
   y0,y1,m0=positions[j,strength];z0,z1,m1=positions[j+1,strength];path=band_path(j+.055,j+1-.055,y0+.015,y1-.015,z0+.015,z1-.015);patch=PathPatch(path,fc='none',ec='none');a.add_patch(patch)
   left=np.array(to_rgb(cols[m0]));right=np.array(to_rgb(cols[m1]));rgba=np.ones((2,96,4));rgba[:,:,:3]=np.linspace(left,right,96)[None,:,:];rgba[:,:,3]=.37
   im=a.imshow(rgba,extent=(j+.055,j+.945,min(y0,z0),max(y1,z1)),origin='lower',aspect='auto',zorder=1);im.set_clip_path(patch)
 for j,q in enumerate(budgets):
  for strength in strengths:
   y0,y1,m=positions[j,strength];a.add_patch(Rectangle((j-.055,y0),.11,y1-y0,fc=cols[m],ec='white',lw=.55,zorder=4))
  a.text(j,-.29,f'{q:g}',ha='center',va='top',fontsize=7.8)
 for strength in strengths:
  y0,y1,m=positions[0,strength];a.text(-.17,(y0+y1)/2,f'{strength:g}',ha='right',va='center',fontsize=7.2,color=GRAY)
 for m in cols:
  spans=[positions[5,v][:2] for v in strengths if positions[5,v][2]==m]
  if spans:a.text(5.13,np.mean([min(y[0] for y in spans),max(y[1] for y in spans)]),labels[m]+f'  ×{len(spans)}',fontsize=8.3,ha='left',va='center',color=cols[m])
 a.text(-.18,5.12,'Synergy',ha='right',fontsize=7.6);a.text(2.5,-.80,r'Node-error budget $\varepsilon$',ha='center',fontsize=8.6)
 a.set(xlim=(-.50,6.05),ylim=(-.92,5.87));a.axis('off')
 a.legend(handles=[Patch(fc=c,label=labels[m]) for m,c in cols.items()],loc='upper center',bbox_to_anchor=(.5,.995),ncol=4,frameon=False,fontsize=8,handlelength=1.0,columnspacing=1.0)
 a.text(2.35,4.99,'7 strengths · 6 budgets · 42 observed choices',ha='center',fontsize=7.4,color=GRAY)
 methods=[('uniform','Uniform',COL['G-time']),('density_stratified','Stratified',COL['MG-time']),('targeted','Targeted',COL['C2-time'])];evidence=[];network_rows=[]
 axes=[]
 for k,q in enumerate((4,12,24)):
  b=f.add_axes([.566+k*.141,.155,.125,.715]);axes.append(b)
  b.set(xlim=(-1.25,21.5),ylim=(-.62,4.32),xticks=[0,10,20],yticks=[]);b.spines['left'].set_visible(False);tidy(b,'x');b.text(.50,.965,f'{q} queries',transform=b.transAxes,ha='center',fontsize=9.0,weight='bold')
  for j,(m,label,color) in enumerate(methods):
   center=2.65-j*1.14
   sub=raw[(raw.budget_group==q)&(raw.method==m)];nets=sub.groupby('network').unsafe.agg(['sum','size']);assert len(nets)==24 and (nets['size']==36).all()
   vals=100*nets['sum']/nets['size'];counts=np.bincount(nets['sum'],minlength=16)[:16];assert counts.sum()==24 and nets['sum'].max()<=7;xs=np.arange(16)*100/36
   b.axhline(center-.02,c='#DCE4E9',lw=.6,zorder=1)
   b.bar(xs,counts/24*.95,width=2.15,bottom=center,color=color,alpha=.34,edgecolor=color,lw=.4,zorder=3)
   for net,row in nets.iterrows():network_rows.append(dict(queries=q,method=m,network=int(net),unsafe=int(row['sum']),decisions=int(row['size']),unsafe_rate=float(row['sum']/row['size'])))
   r=s[(s.q==q)&(s.method==m)].iloc[0];v=r.unsafe_rate*100;lo=r.ci_low*100;hi=r.ci_high*100
   assert int(nets['sum'].sum())==int(r.unsafe) and np.isclose(vals.mean(),v)
   b.errorbar(v,center-.19,xerr=[[v-lo],[hi-v]],fmt='o',color=color,ms=4,mec='white',mew=.5,elinewidth=1.2,capsize=2.2,zorder=5)
   b.text(20.8,center+.46,f'{int(r.unsafe)}/864',ha='right',va='center',fontsize=7.0,color=color)
   if k==0:b.text(-.24,(center+.14+.62)/4.94,label,transform=b.transAxes,ha='right',va='center',fontsize=7.6,color=color)
   evidence.append(dict(queries=q,method=m,unsafe=int(r.unsafe),decisions=int(r.decisions),unsafe_rate=float(r.unsafe_rate),ci_low=float(r.ci_low),ci_high=float(r.ci_high)))
  b.text(.5,-.13,'Exceeded (%)',transform=b.transAxes,ha='center',fontsize=7.6)

 f.text(.77,.765,'No extra queries: 245/864 (28.36%)',ha='center',fontsize=7.0,color=GRAY)
 EVIDENCE['figure8']={'alluvial_choices':d[['synergy','budget','selected']].to_dict('records'),'intervals':evidence,'network_profiles':network_rows,'profile_counts':24,'decisions_per_network':36,'histogram_x_values':'Exact per-network observed unsafe rates; bins are the discrete counts 0..15 out of 36. Bar heights use one common scale: 0.95 times fraction of the 24 networks.','ribbon_semantics':'One equal-width ribbon per measured synergy strength, only connecting the six tested budget columns. Colors identify endpoint selections, not probabilities or a continuous fitted boundary.','confidence_intervals':'Archived network-cluster bootstrap intervals, unchanged and not recomputed.'}
 assert len(network_rows)==216 and len(evidence)==9 and len(d)==42
 save_fig(f,'images8')

def main():
    figure7()
    figure8()
    (DATA / 'figure_evidence.json').write_text(json.dumps(EVIDENCE, indent=2), encoding='utf-8')
    print('redesign figures done', flush=True)

if __name__ == '__main__':
    main()
