
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v]='1'
import argparse, hashlib, itertools, json, math, platform, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack, block_diag
from scipy.optimize import minimize
from .exp_results_revision import ProbabilityEngine, disjoint_panels, risk, metrics, save_network, write_json, write_csv
from .exp_innovation23_supplementary import _random_hypergraph
from .conversion_validation_data import load_real_temporal
from .exp_paper_core import kernel
from ..dynamics import HypergraphSIParameters
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/csf_strengthening_v2'
MODELS=('SG-global','SG-time','G-time','MG-time','C2-time','G-het')
SMALL=('G-time','MG-time','C2-time','G-het')

def source_parameters(network,strength,nu):
    rates={(t,len(g),chi):strength/(len(g)-1)**nu for t in network.time_points for g in network.active_hyperedges(t) for chi in range(1,len(g))}
    return HypergraphSIParameters(rates,nu)

def schedule(dwell,horizon,bank=8,repetitions=8):
    if repetitions%dwell or horizon%(bank*repetitions):raise ValueError('Incomplete cycle')
    cycle=np.tile(np.repeat(np.arange(bank),dwell),repetitions//dwell)
    return np.tile(cycle,horizon//len(cycle))

class Design:

    def __init__(self,engine,model):
        self.engine,self.model=engine,model
        self.edges={};self.widths=[];self.fast={}
        for t in engine.points:
            if model=='G-het':
                a=engine.adjacency['G',t].tocoo()
                e=np.array([(i,j) for i,j in zip(a.row,a.col) if i<j],dtype=int).reshape(-1,2)
                self.edges[t]=e;self.widths.append(len(e))
            else:self.widths.append(2 if model=='C2-time' else 1)
        self.offsets=np.r_[0,np.cumsum(self.widths)]
        self.count=1 if model=='SG-global' else int(self.offsets[-1])
        if model=='G-het':self.payload=sum(e.nbytes for e in self.edges.values())
        elif model=='C2-time':self.payload=sum(c.nbytes+8 for cells in engine.cells.values() for c in cells)
        else:
            name=model.split('-')[0];tt=engine.points[:1] if name=='SG' else engine.points
            self.payload=sum(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes for a in [engine.adjacency[name,t] for t in tt])
    def local(self,states,t):
        e=self.engine;n=e.n;states=np.asarray(states);sus=states==0
        if self.model!='G-het':return csr_matrix(e.features(states,t,self.model)[sus])
        rows=np.full(states.shape,-1,dtype=int);rows[sus]=np.arange(sus.sum())
        rr=[];cc=[]
        for col,(i,j) in enumerate(self.edges[t]):
            for dest,other in ((i,j),(j,i)):
                valid=sus[:,dest]&(states[:,other]==1)
                rr.extend(rows[valid,dest]);cc.extend([col]*int(valid.sum()))
        return csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(sus.sum(),len(self.edges[t])))
    def panel(self,panel):
        blocks=[self.local(panel[t],t) for t in self.engine.points]
        return vstack(blocks,format='csr') if self.model=='SG-global' else block_diag(blocks,format='csr')
    def prepare_fast(self,t):
        e=self.engine;n=e.n;m=self.model
        if m=='G-het':self.fast[t]=None;return
        if m!='C2-time':self.fast[t]=(e.adjacency[m.split('-')[0],t],None,None);return
        a=np.zeros((n,n));pairs={};entries=[]
        for c in e.cells[t]:
            for i in c:
                others=[j for j in c if j!=i]
                for j in others:a[i,j]+=1
                if len(others)==2:
                    pair=tuple(sorted(others));idx=pairs.setdefault(pair,len(pairs));entries.append((idx,i))
        ij=np.array(list(pairs),dtype=int).reshape(-1,2)
        q=csr_matrix((np.ones(len(entries)),tuple(zip(*entries))),shape=(len(pairs),n)) if entries else csr_matrix((0,n))
        self.fast[t]=(csr_matrix(a),ij,q)
    def predict(self,states,t,h):
        e=self.engine;m=self.model;pos=e.points.index(t);lo,hi=self.offsets[pos:pos+2]
        coeff=h[:1] if m=='SG-global' else h[lo:hi]
        if m=='G-het':
            ed=self.edges[t];ii=np.r_[ed[:,0],ed[:,1]];jj=np.r_[ed[:,1],ed[:,0]]
            a=csr_matrix((np.r_[coeff,coeff],(ii,jj)),shape=(e.n,e.n));hazard=a.dot(states.T).T
        else:
            if t not in self.fast:self.prepare_fast(t)
            a,ij,q=self.fast[t];hazard=a.dot(states.T).T*coeff[0]
            if m=='C2-time' and len(ij):hazard+=q.T.dot((states[:,ij[:,0]]*states[:,ij[:,1]]).T).T*coeff[1]
        return -np.expm1(-hazard)*(1-states)

def target(engine,panel):return np.concatenate([engine.source(panel[t],t)[panel[t]==0] for t in engine.points])
def loss_gradient(h,X,y,alpha,eta):
    survival=np.exp(-X.dot(h));residual=1-survival-y;w=2*residual/len(y);loss=float(np.mean(residual**2))
    if eta:
        k=max(1,int(np.ceil((1-alpha)*len(y))));idx=np.argpartition(abs(residual),len(y)-k)[-k:];tail=abs(residual[idx]).mean()
        loss+=eta*tail**2;w[idx]+=2*eta*tail*np.sign(residual[idx])/k
    return loss,np.asarray(X.T.dot(w*survival)).ravel()

def fit(X,y,V,yv,cfg,etas):
    options=[];starttime=time.perf_counter()
    if X.shape[1]==0:return np.zeros(0),dict(options=[],selected_index=0,seconds=0,all_starts_success=True)
    for eta in etas:
        starts=[]
        for beta in cfg['optimizer_starts']:
            r=minimize(loss_gradient,np.full(X.shape[1],-np.log1p(-beta)),args=(X,y,cfg['alpha'],eta),method='L-BFGS-B',jac=True,bounds=[(0,-np.log1p(-cfg['fit_probability_max']))]*X.shape[1],options=dict(maxiter=cfg['optimizer_maxiter'],maxfun=cfg['optimizer_maxfun'],ftol=1e-12,gtol=1e-9,maxls=30))
            starts.append(dict(h=r.x,objective=float(r.fun),success=bool(r.success),message=str(r.message),nfev=int(r.nfev),nit=int(r.nit),initial_probability=beta))
        best=min(starts,key=lambda z:z['objective']);vr=risk(yv,-np.expm1(-V.dot(best['h'])),cfg['alpha'],cfg['validation_tail_weight'])
        options.append(dict(eta=eta,validation_risk=vr,best=best,starts=starts))
    chosen=int(np.argmin([o['validation_risk'] for o in options]));h=options[chosen]['best']['h']
    return h,dict(options=options,selected_index=chosen,seconds=time.perf_counter()-starttime,all_starts_success=all(s['success'] for o in options for s in o['starts']))

def setup_network(kind,rep,cfg):
    p=cfg['exact'] if kind=='exact' else cfg['parameterization']
    if isinstance(rep,str):return load_real_temporal(rep,p['real_max_events'],cfg['bank_size'],p['max_hyperedge_size'])[0]
    return _random_hypergraph(p['nodes'],cfg['bank_size'],cfg['seed']+(0 if kind=='exact' else 100000)+rep,events_per_snapshot=p['events_per_snapshot'],persistence=0,overlap=.45,duplicate_rate=.15)

def fitted_context(engine,panels,models,cfg,etas,directory):
    ys={s:target(engine,p) for s,p in panels.items()};designs={};fits={};records={};rows=[];arrays={f'y_{s}':y for s,y in ys.items()}
    for m in models:
        d=Design(engine,m);designs[m]=d;xx={s:d.panel(p) for s,p in panels.items()}
        h,r=fit(xx['train'],ys['train'],xx['validation'],ys['validation'],cfg,etas);fits[m]=h;records[m]=r
        row=dict(model=m,parameters=d.count,numeric_input_bytes=d.payload,fit_seconds=r['seconds'],all_starts_success=r['all_starts_success'])
        for split,X in xx.items():
            pred=-np.expm1(-X.dot(h));arrays[f'{m}_{split}']=pred
            row.update({f'{split}_{k}':v for k,v in metrics(ys[split],pred).items()})
        rows.append(row)
    write_json(directory/'fits.json',records);np.savez_compressed(directory/'probabilities.npz',**arrays)
    np.savez_compressed(directory/'parameters.npz',**fits);save_network(directory/'network',engine,panels)
    return designs,fits,rows

def run_exact(rep,strength,cfg,out):
    ident=f'exact_{rep:02d}_{strength:.2f}';directory=out/ident
    if (directory/'complete.json').exists():return ident
    directory.mkdir(parents=True,exist_ok=True);p=cfg['exact'];net=setup_network('exact',rep,cfg);e=ProbabilityEngine(net,source_parameters(net,strength,p['nu']))
    panels=disjoint_panels(e.n,e.points,{s:p[s+'_states'] for s in ('train','validation','test')},cfg['seed']+200000+rep)
    designs,fits,fitrows=fitted_context(e,panels,SMALL,cfg,[cfg['tail_weight']],directory)
    states=np.array(list(itertools.product((0,1),repeat=e.n)),dtype=np.int8);ks={'H':np.array([kernel(states,e.source(states,t),cfg['recovery']) for t in e.points])}
    for m in SMALL:ks[m]=np.array([kernel(states,designs[m].predict(states,t,fits[m]),cfg['recovery']) for t in e.points])
    for K in ks.values():assert np.allclose(K.sum(axis=2),1,atol=1e-12) and np.min(K)>=0
    rowtv={m:.5*abs(ks['H']-ks[m]).sum(axis=2) for m in SMALL}
    mu0=(states.sum(axis=1)==p['initial_infected']).astype(float);mu0/=mu0.sum();density=states.mean(axis=1)
    rows=[];series=[];raw={'states':states,'mu0':mu0,**{f'kernel_{m}':v for m,v in ks.items()}}
    for dwell in cfg['dwell_lengths']:
        seq=schedule(dwell,max(cfg['horizons']),cfg['bank_size'],cfg['bank_repetitions']);mus={m:mu0.copy() for m in ks};bounds={m:0. for m in SMALL};history={m:[mu0.copy()] for m in ks}
        for step,pos in enumerate(seq,1):
            for m in SMALL:bounds[m]=min(1.,bounds[m]+mus['H']@rowtv[m][pos])
            mus={m:mus[m]@ks[m][pos] for m in ks}
            for m in ks:history[m].append(mus[m].copy())
            for m in SMALL:
                tv=.5*abs(mus['H']-mus[m]).sum();assert tv<=bounds[m]+1e-10
                r=dict(replicate=rep,strength=strength,nu=p['nu'],dwell=dwell,horizon=step,model=m,tv=tv,bound=bounds[m],source_survival=1-mus['H'][0],model_survival=1-mus[m][0],source_density=mus['H']@density,model_density=mus[m]@density)
                if step in cfg['horizons']:rows.append(r)
                if step==1 or step%4==0:series.append(r)
        for m in ks:raw[f'mu_d{dwell}_{m}']=np.array(history[m])
    write_csv(directory/'endpoints.csv',rows);write_csv(directory/'series.csv',series);write_csv(directory/'fit_metrics.csv',fitrows);np.savez_compressed(directory/'exact.npz',**raw)
    write_json(directory/'complete.json',dict(id=ident,config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()))
    return ident

def choose(rows,budget,expanded):
    allowed=[r for r in rows if (expanded or r['model']!='G-het') and max(r['validation_worst'],r['audit_worst'])<=budget]
    if not allowed:return None
    return min(allowed,key=lambda r:(r['parameters'],r['numeric_input_bytes'],MODELS.index(r['model'])))

def simulate(e,designs,fits,rep,cfg,directory):
    n=e.n;runs=cfg['large_sis']['runs'];T=max(cfg['horizons']);models=('H',)+SMALL;rows=[];series=[];raw={}

    seed=cfg['seed']+500000+rep;rng=np.random.RandomState(seed)
    initial=(rng.rand(runs,n)<cfg['large_sis']['initial_fraction']).astype(np.int8);uniform=rng.rand(T,runs,n)
    for dwell in cfg['dwell_lengths']:
        xx={m:initial.copy() for m in models};hist={m:[initial.copy()] for m in models};matches={m:np.ones(runs,dtype=bool) for m in SMALL};maes={m:np.zeros(runs) for m in SMALL}
        for step,pos in enumerate(schedule(dwell,T),1):
            t=e.points[pos]
            for m in models:
                p=e.source(xx[m],t) if m=='H' else designs[m].predict(xx[m],t,fits[m]);xx[m]=(uniform[step-1]<np.where(xx[m]==1,1-cfg['recovery'],p)).astype(np.int8);hist[m].append(xx[m].copy())
            for m in SMALL:
                matches[m]&=(xx[m]==xx['H']).all(axis=1);maes[m]+=abs(xx[m].mean(axis=1)-xx['H'].mean(axis=1))
                r=dict(replicate=rep,dwell=dwell,horizon=step,model=m,source_survival=(xx['H'].sum(axis=1)>0).mean(),model_survival=(xx[m].sum(axis=1)>0).mean(),source_density=xx['H'].mean(),model_density=xx[m].mean(),density_mae=(maes[m]/step).mean(),path_match=matches[m].mean(),runs=runs)
                if step in cfg['horizons']:rows.append(r)
                if step%4==0:series.append(r)
        for m in models:raw[f'packed_d{dwell}_{m}']=np.packbits(np.array(hist[m]),axis=-1)
    raw.update(seed=seed,nodes=n,runs=runs,horizon=T);np.savez_compressed(directory/'sis_paths.npz',**raw);write_csv(directory/'sis_endpoints.csv',rows);write_csv(directory/'sis_series.csv',series)

def run_parameterization(rep,nu,cfg,out):
    label=f'synthetic_{rep:02d}' if isinstance(rep,int) else rep;ident=f'hetero_{label}_nu{nu:.0f}';directory=out/ident
    if (directory/'complete.json').exists():return ident
    directory.mkdir(parents=True,exist_ok=True);p=cfg['parameterization'];net=setup_network('large',rep,cfg);e=ProbabilityEngine(net,source_parameters(net,p['strength'],nu))
    seed=cfg['seed']+300000+(rep if isinstance(rep,int) else 100+p['real_datasets'].index(rep))
    panels=disjoint_panels(e.n,e.points,{s:p[s+'_states'] for s in ('train','validation','audit','test')},seed)
    designs,fits,rows=fitted_context(e,panels,MODELS,cfg,cfg['tail_weight_grid'],directory)
    for r in rows:r.update(context=ident,domain='synthetic' if isinstance(rep,int) else rep,replicate=rep,nu=nu,nodes=e.n)
    decisions=[]
    for b in p['budgets']:
        for expanded in (False,True):
            r=choose(rows,b,expanded);decisions.append(dict(context=ident,nu=nu,budget=b,expanded=expanded,chosen='H' if r is None else r['model'],test_worst=0 if r is None else r['test_worst'],unsafe=False if r is None else r['test_worst']>b,parameters=None if r is None else r['parameters']))
    write_csv(directory/'fit_metrics.csv',rows);write_csv(directory/'decisions.csv',decisions)
    if isinstance(rep,int) and nu==cfg['large_sis']['nu']:simulate(e,designs,fits,rep,cfg,directory)
    write_json(directory/'complete.json',dict(id=ident,config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()))
    return ident

def aggregate(out):
    for filename in ('endpoints','series','fit_metrics','decisions','sis_endpoints','sis_series'):
        files=sorted(out.glob('*/'+filename+'.csv'))
        if files:pd.concat([pd.read_csv(f).assign(case=f.parent.name) for f in files],ignore_index=True).to_csv(out/(filename+'.csv'),index=False,encoding='utf-8-sig')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    cfg=json.loads((ROOT/'configs/csf_strengthening.json').read_text(encoding='utf-8-sig'));out=ROOT/'tmp/csf_smoke' if args.smoke else OUT;out.mkdir(parents=True,exist_ok=True)
    if args.smoke:
        cfg['exact']['replicates']=1;cfg['exact']['strengths']=[.24];cfg['large_sis']['runs']=20;cfg['optimizer_maxiter']=100;cfg['optimizer_maxfun']=200
    manifest=out/'protocol.json'
    if not manifest.exists():write_json(manifest,dict(frozen_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),config=cfg,config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    jobs=[('exact',r,s) for r in range(cfg['exact']['replicates']) for s in cfg['exact']['strengths']]
    jobs+=[('hetero',r,nu) for r in (list(range(cfg['parameterization']['synthetic_replicates']))+cfg['parameterization']['real_datasets']) for nu in cfg['parameterization']['nu_values']]
    if args.smoke:jobs=[('exact',0,.24),('hetero',0,2.)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_exact if kind=='exact' else run_parameterization,r,v,cfg,out):(kind,r,v) for kind,r,v in jobs}
        for future in as_completed(futures):print('completed',future.result(),flush=True)
    aggregate(out);print('ALL COMPLETE',out,flush=True)
if __name__=='__main__':main()
