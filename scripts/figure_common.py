
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch,FancyArrowPatch,Circle
from matplotlib.collections import PathCollection,PolyCollection
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.transforms import Bbox
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'results/figure_data';FIG=ROOT/'results/figures'
SAVE_NAMES={1:'images1',3:'images3_base',4:'images5',5:'images6',7:'images9'}
plt.rcParams.update({'font.family':'Arial','mathtext.fontset':'stix','font.size':8.2,'axes.labelsize':8.5,'axes.titlesize':8.7,'legend.fontsize':7.2,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,'lines.linewidth':1.4,'savefig.facecolor':'white'})
COL={'SG-global':'#A2ABB1','SG-time':'#555F69','G-time':'#5379AF','MG-time':'#349F92','C2-time':'#D45E78','H':'#27323D','causal':'#8664A6'}
LAB={'SG-global':r'$SG_{\mathrm{c}}$','SG-time':r'$SG_{\mathrm{t}}$','G-time':r'$G(t)$','MG-time':r'$MG(t)$','C2-time':r'$C_2(t)$','H':r'$H(t)$'}
MODELS=list(COL)[:5];LAYOUT=[];HEADERS=[]

def data(n):return pd.read_csv(DATA/(n+'.csv'))

def title(a,l,t):
 head=a.set_title(t,loc='left',y=1.025,pad=7,fontweight='normal',va='baseline')
 badge=a.text(-.15,1.025,l+'.',transform=head.get_transform(),weight='bold',fontsize=10.5,va='baseline',ha='left')
 a._panel_header=(head,badge)

def clean(a,axis='y'):
 a.grid(axis=axis,lw=.45,color='#E5E8EA');a.set_axisbelow(True)

def hits(a,bb):
 hit=[]
 for line in a.lines:
  if not line.get_visible() or line.get_gid()=='decor':continue
  path=line.get_transform().transform_path(line.get_path())
  if len(path.vertices) and path.intersects_bbox(bb,filled=False):hit.append('line')
 for coll in a.collections:
  if coll.get_gid()=='decor':continue
  if isinstance(coll,PathCollection):
   xy=coll.get_offset_transform().transform(np.asarray(coll.get_offsets()))
   if len(xy) and np.any((xy[:,0]>bb.x0-2)&(xy[:,0]<bb.x1+2)&(xy[:,1]>bb.y0-2)&(xy[:,1]<bb.y1+2)):hit.append('points')
  elif isinstance(coll,PolyCollection):
   for path in coll.get_paths():
    if coll.get_transform().transform_path(path).intersects_bbox(bb,filled=True):hit.append('band');break
 for p in a.patches:
  if p.get_gid()=='data' and p.get_window_extent().overlaps(bb):hit.append('bar')
 for t in a.texts:
  if t.get_gid()!='decor' and t.get_window_extent().overlaps(bb):hit.append('text')
 return hit

def legend(a,handles=None,labels=None,ncol=1,size=7.1,prefer='upper right'):
 kw={'ncol':ncol,'fontsize':size,'loc':'upper left','bbox_to_anchor':(.02,.98),'borderaxespad':0,'borderpad':.35,'labelspacing':.25,'handlelength':1.35,'handletextpad':.4,'columnspacing':.8,'frameon':True,'framealpha':.96,'facecolor':'white','edgecolor':'none'}
 le=a.legend(handles=handles,**kw) if labels is None else a.legend(handles=handles,labels=labels,**kw) if handles is not None else a.legend(**kw)
 a.figure.canvas.draw();renderer=a.figure.canvas.get_renderer();area=a.get_window_extent();bb=le.get_window_extent(renderer);w=bb.width/area.width;h=bb.height/area.height
 if w>.97 or h>.97:raise ValueError('Legend is larger than axes')
 left,right=.018,1-w-.018;top,bottom=.982,h+.018
 initial={'upper right':(right,top),'upper left':(left,top),'lower right':(right,bottom),'lower left':(left,bottom),'center':((1-w)/2,(1+h)/2)}[prefer]
 candidates=[initial,(right,top),(left,top),(right,bottom),(left,bottom)]
 candidates += [(x,y) for y in np.linspace(top,bottom,17) for x in np.linspace(left,right,13)]
 best=None
 for x,y in candidates:
  le.set_bbox_to_anchor((x,y),transform=a.transAxes);bb=le.get_window_extent(renderer).padded(1.5)
  overlap=hits(a,bb)
  if best is None or len(overlap)<best[0]:best=(len(overlap),x,y,overlap)
  if not overlap:break
 le.set_bbox_to_anchor((best[1],best[2]),transform=a.transAxes);le.set_gid('inside_legend');a.figure.canvas.draw()
 bb=le.get_window_extent(renderer);inside=area.contains(bb.x0,bb.y0) and area.contains(bb.x1,bb.y1)
 LAYOUT.append({'figure':a.figure.number,'panel':a.get_title(loc='left'),'inside_axes':bool(inside),'curve_or_value_overlaps':best[3]})
 return le
def save(f,n):
 f.canvas.draw()
 for a in f.axes:
  if hasattr(a,'_panel_header'):
   head,badge=a._panel_header;hy=head.get_transform().transform(head.get_position())[1];by=badge.get_transform().transform(badge.get_position())[1]
   HEADERS.append({'figure':n,'panel':head.get_text(),'label':badge.get_text(),'baseline_difference_px':float(abs(hy-by)),'dot_suffix':badge.get_text().endswith('.')})
   assert abs(hy-by)<1e-9
  le=a.get_legend()
  if le is not None:
   record=next(r for r in reversed(LAYOUT) if r['panel']==a.get_title(loc='left'));record['figure']=n;record['curve_or_value_overlaps']=hits(a,le.get_window_extent().padded(1.5))
 f.savefig(FIG/f'{SAVE_NAMES[n]}.pdf',bbox_inches='tight',pad_inches=.045,metadata={'Title':f'Journal Results Figure {n}','Creator':'Verified observations; Matplotlib'})
 f.savefig(FIG/f'{SAVE_NAMES[n]}.png',dpi=240,bbox_inches='tight',pad_inches=.045)
 plt.close(f)

def figure1():
 d=data('order_curves');p=data('order_permutations');w=data('heterogeneous_graph');stats=json.loads((DATA/'statistics.json').read_text())
 f,ax=plt.subplots(1,3,figsize=(8.2,3.25),gridspec_kw={'width_ratios':[1.23,1,1.02]});f.subplots_adjust(left=.075,right=.985,bottom=.19,top=.87,wspace=.39)
 a=ax[0];a.fill_between(d.time,d.shuffle_p05,d.shuffle_p95,color='#ADC6CF',alpha=.34,lw=0)
 for k,lab,c,ls in [('original_density','Original',COL['H'],'-'),('reverse_density','Reverse',COL['C2-time'],'--'),('shuffle_density_mean','Shuffle','#91B6C1','-'),('mean_kernel_static_density','Mean kernel','#B5A46C',':'),('static_union_density','Union',COL['MG-time'],'-.')]:a.plot(d.time,d[k],color=c,ls=ls,label=lab,lw=1.3)
 a.set(xlabel='Snapshot $t$',ylabel='Expected infected fraction',xlim=(-.15,6.2),ylim=(0,1.05),xticks=[0,2,4,6],yticks=[0,.5,1]);clean(a);title(a,'a','Temporal ordering');legend(a,ncol=3,size=6.5,prefer='lower right')
 a=ax[1];counts,bins,patches=a.hist(p.endpoint_tv,bins=np.linspace(0,.8,17),color='#96C4BB',edgecolor='white',linewidth=.65)
 for patch in patches:patch.set_gid('data')
 for key,lab,c in [('reverse','Reverse',COL['C2-time']),('mean_kernel_static','Mean kernel','#B5A46C'),('static_union','Static union',COL['MG-time'])]:
  value=stats['temporal_order']['endpoint_tv'][key];a.plot([value,value],[0,counts.max()*1.10],ls='--',c=c,lw=1.1,label=lab)
 a.set(xlabel='Endpoint distribution TV',ylabel='Number of event orders',xlim=(0,.8),ylim=(0,counts.max()*1.65),xticks=[0,.2,.4,.6,.8]);clean(a);title(a,'b','Distributional discrepancy');legend(a,size=6.9,prefer='upper left')
 a=ax[2]
 for k,b in enumerate(sorted(w.beta.unique())):
  g=w[w.beta==b].sort_values(['kind','network']);j=np.linspace(-.065,.065,len(g))
  for off,col,c in [(-.11,'max_node_error',COL['MG-time']),(.11,'shared_beta_naive_error',COL['C2-time'])]:
   vals=g[col].to_numpy();bp=a.boxplot(vals,positions=[k+off],widths=.16,manage_ticks=False,patch_artist=True,showfliers=False,whis=(0,100),boxprops={'facecolor':c,'alpha':.2,'edgecolor':c,'linewidth':.8},medianprops={'color':c,'linewidth':1.2},whiskerprops={'color':c,'linewidth':.8},capprops={'color':c,'linewidth':.8})
   a.scatter(k+off+j,vals,c=c,s=5,alpha=.55,zorder=3)
 a.set(yscale='log',ylim=(8e-17,3),xlim=(-.5,2.5),xticks=[0,1,2],xticklabels=['.01','.05','.10'],xlabel=r'$\beta^{MG}$',ylabel='Maximum probability error');a.set_yticks([1e-15,1e-10,1e-5,1]);clean(a);title(a,'c','Multiplicity-preserving map')
 h=[Line2D([],[],color=COL['MG-time'],marker='o',ls='',ms=3,label='Edge-specific rate'),Line2D([],[],color=COL['C2-time'],marker='o',ls='',ms=3,label='Shared rate')];legend(a,handles=h,size=6.9,prefer='center')

 net=json.loads((ROOT/'results/paper_core_extension/certificate/case_000/network.json').read_text());mult=sum(0 in g and 1 in g for g in net['snapshots']['0']);assert mult==3
 inset=a.inset_axes([.065,.21,.87,.30]);inset.set(xlim=(0,1),ylim=(0,1));inset.axis('off')
 for rad in [-.28,0,.28]:inset.add_patch(FancyArrowPatch((.08,.68),(.32,.68),connectionstyle=f'arc3,rad={rad}',arrowstyle='-',color=COL['MG-time'],lw=.9))
 inset.plot([.7,.94],[.68,.68],color=COL['C2-time'],lw=1.3)
 for x in [.08,.32,.7,.94]:inset.add_patch(Circle((x,.68),.035,fc='white',ec=COL['H'],lw=.7))
 inset.annotate('',xy=(.63,.68),xytext=(.41,.68),arrowprops={'arrowstyle':'->','lw':.8,'color':COL['H']});inset.text(.20,.35,'$w=3$',ha='center',fontsize=7)
 inset.text(.52,.04,r'$\beta^G_{ij}=1-(1-\beta^{MG})^3$',ha='center',fontsize=7.5)

 le=a.get_legend();lb=le.get_window_extent();ib=inset.get_window_extent()
 if lb.overlaps(ib):
  le.remove();LAYOUT.pop();legend(a,handles=h,size=6.9,prefer='upper left')
 save(f,1)
def figure4():
 d=data('synthetic_independent_tests');f,ax=plt.subplots(1,3,figsize=(7.8,2.85),sharey=True);f.subplots_adjust(left=.10,right=.985,bottom=.20,top=.87,wspace=.24)
 for a,mech,tag in zip(ax,['pairwise','synergy','conflict'],'abc'):
  for k,m in enumerate(MODELS):
   g=d[(d.mechanism==mech)&(d.model==m)];pos=4-k
   a.boxplot(g.rmse,positions=[pos],orientation='horizontal',widths=.44,manage_ticks=False,patch_artist=True,showfliers=False,whis=(0,100),boxprops={'fc':COL[m],'ec':COL[m],'alpha':.16,'lw':.8},medianprops={'color':COL[m],'lw':1.4},whiskerprops={'color':COL[m],'lw':.7},capprops={'color':COL[m],'lw':.7})
   for rep in range(5):
    pair=g[g.replicate==rep].set_index('test_role');yy=pos+(rep-2)*.065;x=pair.loc['test-id','rmse'];z=pair.loc['test-shift','rmse'];a.plot([x,z],[yy,yy],color=COL[m],alpha=.38,lw=.6);a.scatter(x,yy,s=13,facecolor='white',edgecolor=COL[m],lw=.7,zorder=3);a.scatter(z,yy,s=12,c=COL[m],marker='^',lw=0,zorder=4)
  a.set(xscale='log',xlim=(1e-9,.45),ylim=(-.6,5.5),yticks=[4,3,2,1,0],yticklabels=[LAB[m] for m in MODELS],xticks=[1e-8,1e-5,1e-2],xlabel='Independent-test RMSE');clean(a,'x');title(a,tag,mech.capitalize())
 ax[2].set(xlim=(.015,.3),xticks=[.02,.05,.1,.2],xticklabels=['.02','.05','.10','.20']);ax[2].minorticks_off()
 h=[Line2D([],[],marker='o',mfc='white',mec=COL['H'],ls='',ms=3,label='Matched'),Line2D([],[],marker='^',color=COL['H'],ls='',ms=3,label='Shifted')];legend(ax[1],handles=h,ncol=2,size=7,prefer='upper right');save(f,4)
def figure5():
 d=data('real_timeblock_tests');f,ax=plt.subplots(2,2,figsize=(7.4,4.25));f.subplots_adjust(left=.09,right=.985,bottom=.12,top=.92,wspace=.26,hspace=.54)
 for a,name,tag in zip(ax.flat,['Email-Enron','High-School','NDC-classes-full','Primary-School'],'abcd'):
  sub=d[d.dataset==name]
  for m in MODELS:
   g=sub[sub.model==m];s=g.groupby('events').rmse.agg(['mean','min','max']);a.fill_between(s.index,s['min'],s['max'],color=COL[m],alpha=.11,lw=0);a.plot(s.index,s['mean'],marker='o',ms=2.7,c=COL[m],ls='--' if m=='SG-global' else '-',label=LAB[m]);
   for block,mk in [(0,'o'),(1,'s'),(2,'^')]:
    obs=g[g.block==block];a.scatter(obs.events,obs.rmse,marker=mk,color=COL[m],s=8,alpha=.52,lw=0)
  ticks=sorted(sub.events.unique());a.set(xscale='log',xticks=ticks,xticklabels=[f'{v:,}' for v in ticks],ylim=(0,sub.rmse.max()*1.48),xlabel='Events per window',ylabel='Test RMSE');a.minorticks_off();clean(a);title(a,tag,name);legend(a,ncol=2,size=7.1,prefer='upper left')
 save(f,5)
def phase_fractions():
 d=data('dynamic_processes');rows=[];model_order=['G-time','MG-time','C2-time','H']
 for ctx,g in d.groupby('context'):
  domain=g.domain.iloc[0];folder=ROOT/('results/results_revision_supplement/dynamic' if domain=='synthetic' else 'results/paper_core_extension/dynamic')/ctx
  seq=np.array(json.loads((folder/'selection.json').read_text())['sequences']['Causal-checked']);T=len(seq)
  for phase in range(6):
   vv=seq[np.minimum(5,6*np.arange(T)//T)==phase]
   for k,m in enumerate(model_order):rows.append({'context':ctx,'domain':domain,'phase':phase+1,'model':m,'fraction':float(np.mean(vv==k))})
 df=pd.DataFrame(rows);df.to_csv(DATA/'phase_representation_fractions.csv',index=False,encoding='utf-8-sig');return df
def figure7():
 d=data('dynamic_processes');phase=phase_fractions();z=np.load(ROOT/'results/results_revision_supplement/dynamic/synthetic_0/trajectories.npz');f,ax=plt.subplots(1,3,figsize=(7.8,3.0),gridspec_kw={'width_ratios':[1.13,.94,1.0]});f.subplots_adjust(left=.065,right=.985,bottom=.20,top=.87,wspace=.49)
 a=ax[0];source=z['source_states'].mean(2);a.fill_between(np.arange(25),np.quantile(source,.05,axis=0),np.quantile(source,.95,axis=0),color='#BDCCD0',alpha=.38,lw=0)
 for k,lab,c,ls in [('source_states',r'$H(t)$',COL['H'],'-'),('Causal-checked_states','Selected',COL['causal'],'--'),('Fixed-MG-time_states',r'Fixed $MG$',COL['MG-time'],'-.'),('Fixed-C2-time_states',r'Fixed $C_2$',COL['C2-time'],':')]:a.plot(z[k].mean(axis=(0,2)),color=c,ls=ls,label=lab)
 a.set(xlim=(0,24),ylim=(0,1.04),xticks=[0,8,16,24],yticks=[0,.5,1],xlabel='Snapshot $t$',ylabel='Mean infected fraction');clean(a);title(a,'a','Paired propagation');legend(a,ncol=2,size=6.7,prefer='lower right')
 a=ax[1];model_order=['G-time','MG-time','C2-time','H'];mean=phase.groupby(['phase','model']).fraction.mean().unstack().reindex(columns=model_order);left=np.zeros(6)
 for m in model_order:
  bars=a.barh(np.arange(6)[::-1],mean[m],left=left,color=COL[m],height=.62,edgecolor='white',lw=.6,label=LAB[m]);left+=mean[m]
  for b in bars:b.set_gid('data')
 a.set(xlim=(0,1),ylim=(-.6,7.0),xticks=[0,.5,1],xticklabels=['0','50','100'],yticks=range(6),yticklabels=['Conflict','Pairwise','Synergy','Conflict','Synergy','Pairwise'],xlabel='Selected snapshots (%)');title(a,'b','Representation composition');legend(a,ncol=2,size=7,prefer='upper right')
 a=ax[2];methods=[('Fixed-G-time',r'$G(t)$',COL['G-time']),('Fixed-MG-time',r'$MG(t)$',COL['MG-time']),('Fixed-C2-time',r'$C_2(t)$',COL['C2-time']),('Causal-checked','Selected',COL['causal']),('Fixed-H',r'$H(t)$',COL['H'])]
 for i,(m,lab,c) in enumerate(methods):
  for domain,mk in [('synthetic','o'),('real_chronological','^')]:
   g=d[(d.method==m)&(d.domain==domain)].sort_values('context');yy=(4-i)+np.linspace(-.17,.17,len(g));a.scatter(g.paired_density_mae,yy,c=c,marker=mk,s=9,alpha=.55,lw=0)
  avg=d[d.method==m].paired_density_mae.mean();a.plot([avg,avg],[4-i-.3,4-i+.3],c=c,lw=1.4)
 a.set_xscale('symlog',linthresh=1e-5);a.set(xlim=(-1e-6,.09),ylim=(-.6,5.55),xticks=[0,1e-4,1e-2],xticklabels=['0',r'$10^{-4}$',r'$10^{-2}$'],yticks=[4,3,2,1,0],yticklabels=[x[1] for x in methods],xlabel='Paired density MAE');clean(a,'x');title(a,'c','Accuracy across networks')
 h=[Line2D([],[],marker='o',color=COL['H'],ls='',ms=3,label='Synthetic'),Line2D([],[],marker='^',color=COL['H'],ls='',ms=3,label='Real structure')];legend(a,handles=h,size=6.7,prefer='upper right');save(f,7)
