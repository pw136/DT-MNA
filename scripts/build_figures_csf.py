from copy import deepcopy
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (ArrayObject, ContentStream, DecodedStreamObject,
                           FloatObject, NameObject, NumberObject, TextStringObject)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import figure_common as fc
from figure_common import COL, LAB, DATA

NEW = ROOT / 'results' / 'csf_strengthening_v2'
FIG = ROOT / 'results' / 'figures'
COL['G-het'] = '#AF842D'
LAB['G-het'] = r'$G_{\mathrm{het}}(t)$'

def save_fig(f, name):
    FIG.mkdir(parents=True, exist_ok=True)
    f.canvas.draw()
    f.savefig(FIG / f'{name}.pdf', bbox_inches='tight', pad_inches=.045,
              metadata={'Title': name, 'Creator': 'Verified observations; Matplotlib'})
    f.savefig(FIG / f'{name}.png', dpi=240, bbox_inches='tight', pad_inches=.045)
    plt.close(f)

def stats():
 e=pd.read_csv(NEW/'endpoints.csv');h=pd.read_csv(NEW/'fit_metrics.csv');h=h[h.case.str.startswith('hetero')];d=pd.read_csv(NEW/'decisions.csv');s=pd.read_csv(NEW/'sis_endpoints.csv')
 summary=dict(exact=e[e.horizon==256].groupby(['strength','dwell','model'])[['tv','bound','source_survival','model_survival']].mean().reset_index().to_dict('records'),hetero=h.groupby(['nu','model'])[['test_rmse','test_worst','parameters']].mean().reset_index().to_dict('records'),large_sis=s[s.horizon==256].groupby(['dwell','model'])[['density_mae','source_survival','model_survival','path_match']].mean().reset_index().to_dict('records'),decisions=d.groupby(['nu','expanded','chosen']).size().rename('count').reset_index().to_dict('records'),unsafe=d.groupby('expanded').unsafe.sum().to_dict(),parameter_ranges=h.groupby('model').parameters.agg(['min','max']).reset_index().to_dict('records'))
 p=h.pivot(index=['context','nu'],columns='model',values='test_rmse');summary['hetero_vs_c2_wins']=int((p['G-het']<p['C2-time']).sum());summary['hetero_vs_all_shared_wins']=int((p['G-het']<p.drop(columns='G-het').min(axis=1)).sum())
 rng=np.random.RandomState(98124);cis=[]
 for nu in (1.,2.):
  v=p.xs(nu,level='nu');delta=(v['C2-time']-v['G-het']).values;means=delta[rng.randint(0,len(delta),(10000,len(delta)))].mean(axis=1)
  cis.append(dict(nu=nu,mean_rmse_difference=float(delta.mean()),lower=float(np.quantile(means,.025)),upper=float(np.quantile(means,.975)),networks=len(delta)))
 summary['paired_bootstrap']=cis
 selected_failed=[];starts=0;fails=0
 for f in NEW.glob('*/fits.json'):
  for m,r in json.loads(f.read_text()).items():
   if not r['options'][r['selected_index']]['best']['success']:selected_failed.append([f.parent.name,m])
   for o in r['options']:
    for z in o['starts']:starts+=1;fails+=not z['success']
 summary['optimizer']={'starts':starts,'failed_starts':fails,'failed_selected':selected_failed}
 summary['counts']={'exact_source_settings':30,'exact_temporal_processes':120,'exact_endpoints_by_model':len(e),'parameterization_networks':12,'parameterization_contexts':24,'selection_decisions_per_set':96,'sis_network_schedule_combinations':32,'paired_paths_per_candidate':16000}
 (ROOT/'results/figure_data/new_statistics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'wins':summary['hetero_vs_all_shared_wins'],'ci':cis},ensure_ascii=False))
def figure3():

 d=fc.data('sis_certificates');s=fc.data('certificate_selections');ss=fc.data('certificate_summary')
 f,ax=plt.subplots(1,3,figsize=(8.35,3.05),gridspec_kw={'width_ratios':[1.05,1.18,.68]});f.subplots_adjust(left=.065,right=.985,bottom=.20,top=.86,wspace=.64)
 a=ax[0]
 for col,lab,c,m in [('worst_row_bound','Worst row','#9BB5AA','^'),('contraction_bound','Contraction','#B0B8BD','x'),('distribution_bound','Distribution',COL['C2-time'],'o')]:a.scatter(d.actual_tv,d[col],c=c,marker=m,s=9,alpha=.65,lw=.45,label=lab)
 a.plot([0,1],[0,1],'--',c='#808890',lw=.7);a.set(xlim=(-.02,1.03),ylim=(-.02,1.4),xlabel=r'Endpoint TV $E_T^M$',ylabel='Upper bound',yticks=[0,.5,1],xticks=[0,.5,1]);fc.clean(a);fc.title(a,'a','Finite-time bounds');fc.legend(a,ncol=2,size=6.7,prefer='upper left')
 a=ax[1];rules=[('distribution','Distribution'),('worst','Worst / contraction'),('node_validation','Node diagnostic'),('endpoint_oracle','Endpoint reference')]
 for i,(rule,label) in enumerate(rules):
  g=s[s.rule==rule];counts=[int((g.simplified&~g.unsafe).sum()),int(g.unsafe.sum()),int((~g.simplified).sum())];left=0
  for n,c in zip(counts,[COL['MG-time'],COL['C2-time'],'#DDE2E5']):
   bars=a.barh(3-i,n,left=left,height=.50,color=c,edgecolor='white',lw=.5)
   for b in bars:b.set_gid('data')
   if n>10:a.text(left+n/2,3-i,str(n),ha='center',va='center',fontsize=6.8,color='white' if c!='#DDE2E5' else COL['H'])
   elif n>0:a.text(left+n/2,3-i-.36,str(n),ha='center',va='center',fontsize=6.2,color=COL['H'])
   left+=n
 a.set(yticks=[3,2,1,0],yticklabels=[r[1] for r in rules],xlim=(0,182),ylim=(-.65,5.25),xticks=[0,90,180],xlabel='Decisions (180 total)');a.tick_params(axis='y',labelsize=6.7);fc.title(a,'b','Budget outcomes');fc.clean(a,'x');fc.legend(a,handles=[Patch(fc=COL['MG-time'],label='Safe simplification'),Patch(fc=COL['C2-time'],label='Exceeded'),Patch(fc='#DDE2E5',label='Retain H')],ncol=1,size=6.5,prefer='upper left')
 a=ax[2]
 lookup=json.loads((DATA/'statistics.json').read_text())['certificate_summary'];costs={r['rule']:r['cost'] for r in lookup}
 for i,(r,l) in enumerate(rules):a.barh(3-i,costs[r],height=.5,color=[COL['C2-time'],'#9BB5AA','#A4B1BC',COL['MG-time']][i]);a.text(costs[r]+.13,3-i,f'{costs[r]:.3f}',va='center',fontsize=6.7)
 a.set(xlim=(0,7.3),ylim=(-.65,5.25),yticks=[3,2,1,0],yticklabels=[],xlabel='Mean score',xticks=[0,3,6]);fc.title(a,'c','Complexity');fc.clean(a,'x');save_fig(f,'images3_base')
def figure4():
 e=pd.read_csv(NEW/'endpoints.csv');ser=pd.read_csv(NEW/'series.csv');large=pd.read_csv(NEW/'sis_endpoints.csv');last=e[e.horizon==256];dcolors=['#8AAEBB','#5379AF','#705E91','#B27B45']
 f,ax=plt.subplots(2,3,figsize=(8.5,6.5));f.subplots_adjust(left=.075,right=.98,bottom=.09,top=.93,wspace=.40,hspace=.64)
 a=ax[0,0]
 def schedule(dwell,horizon):
  return np.tile(np.tile(np.repeat(np.arange(8),dwell),8//dwell),horizon//64)
 mat=np.array([schedule(d,64) for d in (1,2,4,8)])
 a.imshow(mat,aspect='auto',interpolation='nearest',cmap=ListedColormap(['#D7E5ED','#8EAFC1','#426E8E','#9EBFB1','#518778','#E6C98D','#B78E46','#965F66']),extent=(-.5,63.5,3.5,-.5))
 a.set(yticks=range(4),yticklabels=[1,2,4,8],xticks=[0,16,32,48,64],xlabel='Step within a 64-step cycle',ylabel=r'Dwell length $\ell$');fc.title(a,'a','Temporal persistence')

 a=ax[0,1]
 for k,dw in enumerate((1,2,4,8)):
  arr=[]
  for rep in range(6):
   z=np.load(NEW/f'exact_{rep:02d}_0.24/exact.npz');mu=z[f'mu_d{dw}_H'][-1];arr.append(np.bincount(z['states'].sum(axis=1),weights=mu,minlength=8))
  y=np.mean(arr,axis=0);xx=np.arange(8);offset=3-k;a.fill_between(xx,offset,offset+y*.9,step='mid',color=dcolors[k],alpha=.38);a.step(xx,offset+y*.9,where='mid',color=dcolors[k],lw=1.1)
 a.set(xlim=(-.5,7.5),ylim=(-.05,4.0),yticks=[3,2,1,0],yticklabels=[1,2,4,8],xticks=[0,2,4,6],xlabel='Number of infected nodes',ylabel=r'Dwell length $\ell$');a.text(.98,.98,r'$\lambda_\star=0.24,\ T=256$'+'\nHeight = 0.9 x probability',transform=a.transAxes,ha='right',va='top',fontsize=6.5);fc.title(a,'b','Terminal state distributions')
 a=ax[0,2];pivot=last.groupby(['strength','dwell','model']).tv.mean().unstack('model');delta=(pivot['C2-time']-pivot['G-het']).unstack('dwell');lim=max(abs(delta.to_numpy()).max(),.13)
 im=a.imshow(delta,origin='lower',cmap='PuOr_r',norm=TwoSlopeNorm(vcenter=0,vmin=-lim,vmax=lim),interpolation='nearest',aspect='auto')
 for iy,row in enumerate(delta.to_numpy()):
  for ix,v in enumerate(row):a.text(ix,iy,('0.000' if abs(v)<.0005 else f'{v:+.3f}'),ha='center',va='center',fontsize=6.4,color='white' if abs(v)>.09 else COL['H'])
 a.set(xticks=range(4),xticklabels=[1,2,4,8],yticks=range(5),yticklabels=['.06','.12','.24','.40','.64'],xlabel=r'Dwell length $\ell$',ylabel=r'Source strength $\lambda_\star$');fc.title(a,'c','Relative process error')
 cb=f.colorbar(im,ax=a,orientation='horizontal',fraction=.09,pad=.27,aspect=24);cb.set_label(r'$E_T^{C_2}-E_T^{G_{\mathrm{het}}}$',size=7);cb.ax.tick_params(labelsize=6)
 a=ax[1,0]
 for dw,c in zip((1,2,4,8),dcolors):
  g=last[(last.dwell==dw)&(last.model=='C2-time')].groupby('strength').source_survival.mean();a.plot(g.index,g,'o-',color=c,ms=3,label=fr'$\ell={dw}$')
 a.set(xlim=(.035,.67),ylim=(-.03,1.35),xticks=[.06,.24,.4,.64],yticks=[0,.5,1],xlabel=r'Source strength $\lambda_\star$',ylabel=r'Finite-horizon survival at $T=256$');fc.clean(a);fc.title(a,'d','Source survival');fc.legend(a,ncol=2,size=6.7,prefer='upper left')
 a=ax[1,1]
 for m in ('C2-time','G-het'):
  for dw,ls in [(1,'-'),(8,'--')]:
   g=ser[(ser.model==m)&(ser.strength==.24)&(ser.dwell==dw)].groupby('horizon').tv.agg(['mean','std']);a.plot(g.index,g['mean'],color=COL[m],ls=ls,label=LAB[m]+fr', $\ell={dw}$')
 a.set(xlim=(0,256),ylim=(0,.5),xticks=[0,64,128,256],yticks=[0,.2,.4],xlabel='Horizon $T$',ylabel='Exact endpoint TV');fc.clean(a);fc.title(a,'e',r'Error evolution ($\lambda_\star=0.24$)');fc.legend(a,ncol=1,size=6.5,prefer='upper left')
 a=ax[1,2]
 for iy,m in enumerate(('G-time','MG-time','C2-time','G-het')):
  g=large[(large.model==m)&(large.horizon==256)].pivot(index='replicate',columns='dwell',values='density_mae')
  for rep,row in g.iterrows():
   y=3-iy+(rep-3.5)*.055;a.plot([row[1],row[8]],[y,y],color=COL[m],alpha=.35,lw=.75);a.scatter(row[1],y,s=15,facecolor='white',edgecolor=COL[m],lw=.65,zorder=3);a.scatter(row[8],y,s=15,color=COL[m],lw=0,zorder=3)
 a.set(xlim=(.005,.060),ylim=(-.5,4.55),yticks=[3,2,1,0],yticklabels=[LAB[m] for m in ('G-time','MG-time','C2-time','G-het')],xticks=[.01,.03,.05],xlabel='Paired density MAE');fc.clean(a,'x');fc.title(a,'f','Paired SIS trajectories');fc.legend(a,handles=[Line2D([],[],marker='o',mfc='white',mec=COL['H'],ls='',label=r'$\ell=1$',ms=3),Line2D([],[],marker='o',mfc=COL['H'],mec=COL['H'],ls='',label=r'$\ell=8$',ms=3)],ncol=2,size=6.7,prefer='upper left');save_fig(f,'images4')

def complete_y_axis():
    src = FIG / 'images3_base.pdf'
    dst = FIG / 'images3.pdf'
    reader = PdfReader(src)
    page = reader.pages[0]
    ops = ContentStream(page.get_contents(), reader).operations
    labels = ['Distribution', 'Worst / contraction', 'Node diagnostic', 'Endpoint reference']
    selected = {}
    for i, (args, op) in enumerate(ops):
        if op != b'TJ':
            continue
        text = ''.join(str(x) for x in args[0] if isinstance(x, TextStringObject))
        if text not in labels:
            continue
        block = ops[i - 5:i + 3]
        assert [x[1] for x in block] == [b'q', b'cm', b'BT', b'Tf', b'Td', b'TJ', b'ET', b'Q']
        if 180 < float(block[1][0][4]) < 240 and 40 < float(block[1][0][5]) < 125:
            selected[text] = deepcopy(block)
    assert list(selected) == labels
    b_x = next(float(a[0]) for a, o in ops if o == b'm' and 255 < float(a[0]) < 256)
    c_x = next(float(a[0]) for a, o in ops if o == b'm' and 495 < float(a[0]) < 496)
    dx = c_x - b_x
    added = ContentStream(None, reader)
    added.operations = [([], b'q'), ([NumberObject(0)], b'g'), ([NumberObject(0)], b'G'),
                        ([NumberObject(1), NumberObject(0), NumberObject(0), NumberObject(1), FloatObject(dx), NumberObject(0)], b'cm')]
    for label in labels:
        added.operations.extend(selected[label])
    added.operations.append(([], b'Q'))
    writer = PdfWriter(clone_from=reader)
    p = writer.pages[0]
    old_contents = p.raw_get('/Contents')
    contents = ArrayObject(list(old_contents.get_object())) if isinstance(old_contents.get_object(), ArrayObject) else ArrayObject([old_contents])
    stream = DecodedStreamObject()
    stream.set_data(added.get_data())
    contents.append(writer._add_object(stream))
    p[NameObject('/Contents')] = contents
    with dst.open('wb') as f:
        writer.write(f)
    src.unlink()

def main():
    stats()
    figure3()
    complete_y_axis()
    figure4()
    print('csf figures done', flush=True)

if __name__ == '__main__':
    main()
