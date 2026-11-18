
from pathlib import Path
import sys, json, math, itertools, time, csv, hashlib, argparse, platform
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.sparse import csr_matrix
from ..temporal_structures import TemporalHypergraph
from ..dynamics import HypergraphSIParameters, infection_probabilities
from ..theory import all_binary_states, build_temporal_c3_lambda_rates, C3ProjectionRoute
from ..approximate.certificates import build_error_certificate
from .exp_results_revision import (ProbabilityEngine, source_parameters, fit_model,
    disjoint_panels, samples, predict_samples, metrics, risk, save_network, write_json,
    write_csv, dynamic_context, INDEPENDENT, FAIR_MODELS)
from .exp_innovation23_supplementary import _random_hypergraph
from .innovation2_core import build_conflicting_lambda_rates
from .conversion_validation_data import REAL_PROFILES
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'configs/paper_core_extension.json'
OUT=ROOT/'results/paper_core_extension'
MODELS=('G-time','MG-time','C2-time','H')
COST=np.array([1.,2.,4.5,10.])

def config(): return json.loads(CONFIG.read_text(encoding='utf-8-sig'))
def complete(p): return p.exists()
def save_records(path, value): write_json(path,value)
def make_source(net, mechanism, variant=0):
    if mechanism!='conflict': return source_parameters(net,mechanism)
    rates,_=build_conflicting_lambda_rates(net,2.,.045,(.01,.05)[variant%2],.009)
    return HypergraphSIParameters(rates,2.)
def fit_candidates(engine, panel, cfg, model_names=MODELS[:-1]):
    train=samples(engine,panel)
    fitted,logs={},[]
    for model in model_names:
        fitted[model],record=fit_model(train,model,cfg['tail_weight'],cfg,len(engine.points))
        logs.append({'model':model,**record})
    return fitted,logs

def kernel(states, probabilities, recovery):
    one=np.where(states==1,1.-recovery,probabilities)
    K=np.prod(np.where(states[None,:,:]==1,one[:,None,:],1.-one[:,None,:]),axis=2)
    if not np.allclose(K.sum(1),1.,atol=1e-12): raise AssertionError('Nonstochastic kernel')
    return K

def certificate_case(spec, directory):
    cfg=config(); cid,n,T,mechanism,rep=spec
    folder=Path(directory)/'certificate'/f'case_{cid:03d}';folder.mkdir(parents=True,exist_ok=True)
    result=folder/'result.json'
    if complete(result): return json.loads(result.read_text(encoding='utf-8'))
    seed=cfg['seed']+1009*cid;started=time.perf_counter()
    net=_random_hypergraph(n,T,seed,events_per_snapshot=max(3,n//2),persistence=.2+.2*rep,overlap=.3+.2*rep,duplicate_rate=.15+.1*rep)
    engine=ProbabilityEngine(net,make_source(net,mechanism,rep))
    panel=disjoint_panels(n,engine.points,{'train':12,'validation':8},seed+11)
    fitted,logs=fit_candidates(engine,panel['train'],cfg)
    save_network(folder/'network',engine,panel);write_json(folder/'fits.json',logs)
    states=np.array(all_binary_states(n),dtype=np.int8);mu=(states.sum(1)==1).astype(float);mu/=mu.sum()
    recovery=(.1,.2)[rep%2]
    source_prob=np.array([engine.source(states,t) for t in engine.points])
    source_K=np.array([kernel(states,p,recovery) for p in source_prob]); arrays={'states':states,'mu0':mu,'source_probabilities':source_prob,'source_kernels':source_K}
    rows=[]
    for model in MODELS:
        if model=='H':
            rows.append({'model':model,'cost':10.,'actual_tv':0.,'distribution_bound':0.,'worst_bound':0.,'contraction_bound':0.,'validation_error':0.,'certificate_seconds':0.});continue
        predicted=np.array([engine.predict(states,t,model,fitted[model]) for t in engine.points])
        Ks=np.array([kernel(states,p,recovery) for p in predicted]);tick=time.perf_counter()
        cert=build_error_certificate(mu,source_K,Ks)
        valmax=max(float(np.abs(engine.source(panel['validation'][t],t)-engine.predict(panel['validation'][t],t,model,fitted[model])).max()) for t in engine.points)
        rows.append({'model':model,'cost':float(COST[MODELS.index(model)]),'actual_tv':cert.final_actual_error,
            'distribution_bound':cert.final_expected_bound,'worst_bound':cert.final_basic_bound,'contraction_bound':cert.final_contraction_bound,
            'validation_error':valmax,'certificate_seconds':time.perf_counter()-tick})
        arrays[model+'_kernels']=Ks;arrays[model+'_probabilities']=predicted
        if not cert.covered_by_basic or not cert.covered_by_expected or not cert.covered_by_contraction: raise AssertionError('Invalid certificate')
    selections=[]
    for budget in cfg['certificate']['budgets']:
        oracle=next(r for r in rows if r['actual_tv']<=budget+1e-12)
        for rule,key in [('distribution','distribution_bound'),('worst','worst_bound'),('contraction','contraction_bound'),('node_validation','validation_error'),('endpoint_oracle','actual_tv')]:
            chosen=next(r for r in rows if r[key]<=budget+1e-12)
            selections.append({'case':cid,'mechanism':mechanism,'nodes':n,'horizon':T,'budget':budget,'rule':rule,
                'selected':chosen['model'],'cost':chosen['cost'],'actual_tv':chosen['actual_tv'],'screen_value':chosen[key],
                'unsafe':chosen['actual_tv']>budget+1e-12,'simplified':chosen['model']!='H','oracle':oracle['model'],
                'oracle_simplified':oracle['model']!='H','cost_gap':chosen['cost']-oracle['cost']})
    np.savez_compressed(folder/'kernels.npz',**arrays)
    data={'case':cid,'seed':seed,'nodes':n,'horizon':T,'mechanism':mechanism,'replicate':rep,'recovery':recovery,'rows':rows,'selections':selections,'seconds':time.perf_counter()-started}
    save_records(result,data);return data

def risk_vectors(engine,states,t):
    g=engine.features(states,t,'G-time')[:,:,0]; mg=engine.features(states,t,'MG-time')[:,:,0]
    c2=engine.features(states,t,'C2-time')[:,:,1]
    return {'density':states.mean(1),'synergy':c2.max(1),'repeat':(mg-g).max(1)}
def random_order(ids,rng): return np.asarray(ids)[rng.permutation(len(ids))]
def stratified_order(ids,states,rng):
    buckets=[list(random_order(np.asarray(ids)[states[np.asarray(ids)].sum(1)==k],rng)) for k in range(1,states.shape[1])]
    ordered=[]
    while any(buckets):
        for bucket in buckets:
            if bucket:ordered.append(bucket.pop())
    return np.array(ordered,dtype=int)

def audit_case(spec,directory):
    cfg=config(); netid,mechanism,bias=spec
    folder=Path(directory)/'audit'/f'n{netid:02d}_{mechanism}_{bias}';folder.mkdir(parents=True,exist_ok=True)
    result=folder/'result.json'
    if complete(result):return json.loads(result.read_text(encoding='utf-8'))
    settings=cfg['audit'];n=settings['nodes'];T=settings['snapshots'];seed=cfg['seed']+200000+1009*netid
    net=_random_hypergraph(n,T,seed,events_per_snapshot=5+netid%3,persistence=.15+.2*(netid%4),overlap=.2+.15*(netid%5),duplicate_rate=.1+.1*(netid%3))
    engine=ProbabilityEngine(net,make_source(net,mechanism,netid%2))
    states=np.array([x for x in all_binary_states(n) if 0<sum(x)<n],dtype=np.int8)
    rng=np.random.RandomState(seed+17+13*settings['biases'].index(bias));train={};valid={};orders={};indices={}
    for t in engine.points:
        risks=risk_vectors(engine,states,t);allids=np.arange(len(states));tie=rng.random_sample(len(states))
        if bias=='uniform':order=random_order(allids,rng)
        else:
            key={'low_density':'density','low_synergy':'synergy','low_repeat':'repeat'}[bias]
            order=np.lexsort((tie,risks[key]))
        base=order[:settings['train_states']+settings['validation_states']].copy();rng.shuffle(base)
        tr=base[:settings['train_states']];va=base[settings['train_states']:]
        train[t]=states[tr];valid[t]=states[va];remain=np.setdiff1d(allids,base)

        score=risks['synergy']/(risks['synergy'].max()+1)+risks['repeat']/(risks['repeat'].max()+1)
        targeted=remain[np.lexsort((rng.random_sample(len(remain)),-score[remain]))]
        orders[t]={'uniform':random_order(remain,rng),'density_stratified':stratified_order(remain,states,rng),'targeted':targeted}
        indices[t]={'train':tr.tolist(),'validation':va.tolist(),**{k:v[:max(settings['query_budgets'])].tolist() for k,v in orders[t].items()}}
    fitted,logs=fit_candidates(engine,train,cfg);save_network(folder/'network',engine,{'train':train,'validation':valid})
    write_json(folder/'fits.json',logs);write_json(folder/'queries.json',indices)
    valerr=np.zeros(len(MODELS));testerr=np.zeros(len(MODELS));stateerr=np.zeros((T,len(MODELS),len(states)))
    for t in engine.points:
        source=engine.source(states,t)
        for k,model in enumerate(MODELS[:-1]):
            stateerr[t,k]=np.max(np.abs(engine.predict(states,t,model,fitted[model])-source),axis=1)
            valerr[k]=max(valerr[k],stateerr[t,k,indices[t]['validation']].max())
            testerr[k]=max(testerr[k],stateerr[t,k].max())
    rows=[]
    for eps in settings['error_budgets']:
        oracle=int(np.flatnonzero(testerr<=eps+1e-12)[0]);base=int(np.flatnonzero(valerr<=eps+1e-12)[0])
        for q in settings['query_budgets']:
            for method in ('validation_only',*settings['audit_methods']):
                check=valerr.copy()
                if method!='validation_only':
                    for t in engine.points:check=np.maximum(check,stateerr[t][:,orders[t][method][:q]].max(axis=1))
                chosen=int(np.flatnonzero(check<=eps+1e-12)[0])
                rows.append({'network':netid,'mechanism':mechanism,'bias':bias,'epsilon':eps,'audit_queries_per_snapshot':q if method!='validation_only' else 0,
                    'budget_group':q,'method':method,'selected':MODELS[chosen],'oracle':MODELS[oracle],
                    'cost':float(COST[chosen]),'test_error':float(testerr[chosen]),'unsafe':bool(testerr[chosen]>eps+1e-12),
                    'agreement':chosen==oracle,'overmodel':chosen>oracle,'corrected':bool(testerr[base]>eps+1e-12 and testerr[chosen]<=eps+1e-12),
                    'vector_queries':T*(settings['train_states']+settings['validation_states']+(q if method!='validation_only' else 0))})
    np.savez_compressed(folder/'errors.npz',states=states,state_errors=stateerr,validation_errors=valerr,test_errors=testerr)
    data={'network':netid,'seed':seed,'mechanism':mechanism,'bias':bias,'rows':rows};write_json(result,data);return data

def chronological_events(name,max_size=6):
    prefix=REAL_PROFILES[name]['folder'];folder=ROOT/'data'/prefix
    sizes=np.loadtxt(folder/(prefix+'-nverts.txt'),dtype=np.int64);times=np.loadtxt(folder/(prefix+'-times.txt'),dtype=np.int64)
    members=np.loadtxt(folder/(prefix+'-simplices.txt'),dtype=np.int64)
    if len(sizes)!=len(times) or int(sizes.sum())!=len(members):raise AssertionError('Raw data mismatch')
    offsets=np.r_[0,np.cumsum(sizes)];rows=[]
    for i in np.flatnonzero((sizes>=2)&(sizes<=max_size)):
        rows.append((int(times[i]),int(i),tuple(int(v) for v in members[offsets[i]:offsets[i+1]])))
    rows.sort(key=lambda x:(x[0],x[1]));return rows

def chronological_blocks(rows,count=3):
    boundaries=[0]
    for k in range(1,count):
        i=len(rows)*k//count
        while i<len(rows) and rows[i][0]==rows[i-1][0]:i+=1
        boundaries.append(i)
    boundaries.append(len(rows));return [rows[a:b] for a,b in zip(boundaries,boundaries[1:])]

def event_network(rows,T=12):
    first,last=rows[0][0],rows[-1][0];span=max(1,last-first+1);snaps={t:[] for t in range(T)}
    for timestamp,idx,group in rows:snaps[min(T-1,(timestamp-first)*T//span)].append(group)
    nodes=sorted({v for row in rows for v in row[2]})
    return TemporalHypergraph(nodes,{t:tuple(v) for t,v in snaps.items()})

def prepare_real(directory):
    cfg=config();out=Path(directory);specs=[];manifest=[]
    for name in cfg['real']['datasets']:
        rows=chronological_events(name,cfg['real']['max_hyperedge_size'])
        for b,block in enumerate(chronological_blocks(rows,cfg['real']['chronological_blocks'])):
            for limit in cfg['real']['event_limits']:
                if len(block)<limit:
                    manifest.append({'dataset':name,'block':b,'event_limit':limit,'available':len(block),'status':'insufficient_events'});continue
                chosen=block[:limit];cid=f'{name}_b{b}_e{limit}';folder=out/'real'/cid;folder.mkdir(parents=True,exist_ok=True)
                p=folder/'events.json';write_json(p,chosen)
                record={'context':cid,'dataset':name,'block':b,'events':limit,'available':len(block),'first_timestamp':chosen[0][0],
                    'last_timestamp':chosen[-1][0],'snapshots':12,'event_file':str(p.relative_to(ROOT)),'status':'included'}
                manifest.append(record);specs.append(record)
    write_json(out/'real_sampling_manifest.json',manifest);return specs

def real_case(spec,directory):
    cfg=config();folder=Path(directory)/'real'/spec['context'];result=folder/'result.json'
    if complete(result):return json.loads(result.read_text(encoding='utf-8'))
    events=json.loads((folder/'events.json').read_text(encoding='utf-8'));net=event_network(events,cfg['real']['snapshots'])
    engine=ProbabilityEngine(net,source_parameters(net,cfg['real']['mechanism']))
    seed=cfg['seed']+300000+sum(ord(c) for c in spec['context'])*101
    counts={k:cfg['real'][k+'_states'] for k in ('train','validation','test')}
    panel=disjoint_panels(engine.n,engine.points,counts,seed);save_network(folder/'network',engine,panel)
    data={k:samples(engine,p) for k,p in panel.items()};rows=[];logs=[];preds={'test_target':data['test']['target'],'test_time':data['test']['time']}
    for model in FAIR_MODELS:
        opts=[]
        for eta in cfg['tail_weight_grid']:
            params,log=fit_model(data['train'],model,eta,cfg,len(engine.points))
            score=risk(data['validation']['target'],predict_samples(data['validation'],model,params),cfg['alpha'],cfg['validation_tail_weight'])
            opts.append((score,params,log));logs.append({'model':model,'validation_risk':score,**log})
        score,params,log=min(opts,key=lambda x:x[0]);prediction=predict_samples(data['test'],model,params)
        preds[model]=prediction;write_json(folder/(model+'_parameters.json'),params)
        rows.append({**spec,'nodes':engine.n,'model':model,'selected_eta':log['eta'],'optimizer_success':log['success'],**metrics(data['test']['target'],prediction)})
    write_json(folder/'fits.json',logs);np.savez_compressed(folder/'test_predictions.npz',**preds)
    data={'context':spec['context'],'rows':rows};write_json(result,data);return data

def dynamic_case(spec,directory):
    cfg=config();base=Path(directory)/'real'/spec['context'];events=json.loads((base/'events.json').read_text(encoding='utf-8'))
    net=event_network(events,12);engine=ProbabilityEngine(net,source_parameters(net,'conflict',dynamic=True))
    return dynamic_context('chronological_'+spec['context'],engine,cfg,cfg['seed']+400000+spec['block']*1009+sum(map(ord,spec['dataset'])),Path(directory),domain='real_chronological')

def weighted_case(spec,directory):
    cfg=config();kind,idx,beta=spec;folder=Path(directory)/'weighted'/f'{kind}_{idx}_{beta:.2f}';folder.mkdir(parents=True,exist_ok=True)
    result=folder/'result.json'
    if complete(result):return json.loads(result.read_text(encoding='utf-8'))
    seed=cfg['seed']+500000+int(idx)*101
    if kind=='synthetic':
        n=cfg['weighted']['nodes'][int(idx)%2];net=_random_hypergraph(n,12,seed,duplicate_rate=.15+.1*(int(idx)%3))
    else:
        manifest=json.loads((Path(directory)/'real_sampling_manifest.json').read_text(encoding='utf-8'))
        cases=[x for x in manifest if x.get('status')=='included' and x.get('events')==600];item=cases[int(idx)]
        events=json.loads((ROOT/item['event_file']).read_text(encoding='utf-8'));net=event_network(events,12)
    rates=build_temporal_c3_lambda_rates(net,2,2.,{t:beta for t in net.time_points},exposure_semantics=INDEPENDENT,route=C3ProjectionRoute.DIRECT)
    engine=ProbabilityEngine(net,HypergraphSIParameters(rates,2.));rng=np.random.RandomState(seed)

    hazards={};certain={};edge_records=[]
    for t in engine.points:
        W=engine.adjacency[('MG',t)].tocoo();probs=-np.expm1(W.data*np.log1p(-beta))
        sure=probs>=1.0
        hh=-np.log1p(-probs[~sure])
        hazards[t]=csr_matrix((hh,(W.row[~sure],W.col[~sure])),shape=W.shape)
        certain[t]=csr_matrix((np.ones(int(sure.sum())),(W.row[sure],W.col[sure])),shape=W.shape)
        edge_records.extend({'time':t,'i':int(i),'j':int(j),'multiplicity':int(w),'edge_probability':float(p)} for i,j,w,p in zip(W.row,W.col,W.data,probs))
    def graph_prob(states,t):
        value=hazards[t].dot(states.T).T
        return np.where(certain[t].dot(states.T).T>0,1.,-np.expm1(-value))*(1-states)
    maxerr=0.;naiveerr=0.
    for t in engine.points:
        states=(rng.random_sample((cfg['weighted']['states'],engine.n))<.3).astype(np.int8)
        source=engine.source(states,t);maxerr=max(maxerr,float(np.abs(source-graph_prob(states,t)).max()))
        naive=-np.expm1(engine.features(states,t,'G-time')[:,:,0]*np.log1p(-beta));naiveerr=max(naiveerr,float(np.abs(source-naive).max()))
    runs=cfg['weighted']['runs'];initial=(rng.random_sample((runs,engine.n))<.15).astype(np.int8)
    for i in range(runs):
        if not initial[i].any():initial[i,rng.randint(engine.n)]=1
    uniforms=rng.random_sample((runs,len(engine.points),engine.n));source=initial.copy();graph=initial.copy();matched=np.ones(runs,dtype=bool)
    for k,t in enumerate(engine.points):
        source=np.where(source==1,1,(uniforms[:,k]<engine.source(source,t)).astype(np.int8))
        graph=np.where(graph==1,1,(uniforms[:,k]<graph_prob(graph,t)).astype(np.int8));matched &= np.all(source==graph,axis=1)
    save_network(folder/'network',engine,{})
    write_csv(folder/'edge_probabilities.csv',edge_records)
    np.savez_compressed(folder/'pairing.npz',initial=initial,uniforms=uniforms,source_final=source,graph_final=graph,matched=matched)
    row={'kind':kind,'network':idx,'beta':beta,'nodes':engine.n,'snapshots':len(engine.points),'max_node_error':maxerr,'shared_beta_naive_error':naiveerr,'matched_paths':int(matched.sum()),'runs':runs}
    write_json(result,row);return row

def run_pool(stage,specs,fn,directory,jobs):
    result=[];start=time.perf_counter()
    if jobs==1:
        for k,s in enumerate(specs):result.append(fn(s,directory));print(f'{stage} {k+1}/{len(specs)} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures={executor.submit(fn,s,directory):s for s in specs}
            for k,f in enumerate(as_completed(futures)):
                result.append(f.result());print(f'{stage} {k+1}/{len(specs)} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    write_json(Path(directory)/(stage+'_all.json'),result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['certificate','audit','real','dynamic','weighted','all'],default='all');p.add_argument('--jobs',type=int,default=3);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    cfg=config();out=ROOT/'tmp/paper_core_smoke' if args.smoke else OUT;out.mkdir(parents=True,exist_ok=True)
    write_json(out/'run_manifest.json',{'config':cfg,'config_sha256':hashlib.sha256(CONFIG.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'python':sys.version,'platform':platform.platform()})
    stages=['certificate','audit','real','dynamic','weighted'] if args.stage=='all' else [args.stage]
    specs_real=None
    for stage in stages:
        if stage=='certificate':
            specs=[(i,*s) for i,s in enumerate(itertools.product(cfg['certificate']['nodes'],cfg['certificate']['horizons'],cfg['certificate']['mechanisms'],range(cfg['certificate']['replicates'])))];fn=certificate_case
        elif stage=='audit':specs=list(itertools.product(range(cfg['audit']['networks']),cfg['audit']['mechanisms'],cfg['audit']['biases']));fn=audit_case
        elif stage in ('real','dynamic'):
            specs_real=prepare_real(out) if specs_real is None else specs_real
            specs=specs_real if stage=='real' else [x for x in specs_real if x['events']==600];fn=real_case if stage=='real' else dynamic_case
        else:
            specs_real=prepare_real(out) if specs_real is None else specs_real
            real600=[x for x in specs_real if x['events']==600]
            specs=[(kind,i,b) for kind,count in [('synthetic',cfg['weighted']['synthetic_networks']),('real',len(real600))] for i in range(count) for b in cfg['weighted']['betas']];fn=weighted_case
        if args.smoke:specs=specs[:1]
        run_pool(stage,specs,fn,str(out),args.jobs)
if __name__=='__main__':main()
