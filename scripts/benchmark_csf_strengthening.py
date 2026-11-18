
import json,sys,time,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.experiments.exp_csf_strengthening import *

def main():
 cfg=json.loads((ROOT/'configs/csf_strengthening.json').read_text());rows=[]
 for rep in (0,'High-School'):
  name=f'hetero_synthetic_{rep:02d}_nu2' if isinstance(rep,int) else f'hetero_{rep}_nu2'
  directory=OUT/name;e=ProbabilityEngine(setup_network('large',rep,cfg),source_parameters(setup_network('large',rep,cfg),.4,2.))
  arr=np.load(directory/'network.npz');panels={s:{t:arr[f'{s}_{t}'] for t in e.points} for s in ('train','validation','audit','test')};ys={s:target(e,p) for s,p in panels.items()}
  archived=np.load(directory/'parameters.npz')
  for m in MODELS:
   started=time.perf_counter();d=Design(e,m);xx={s:d.panel(p) for s,p in panels.items()};preprocessing=time.perf_counter()-started
   h,record=fit(xx['train'],ys['train'],xx['validation'],ys['validation'],cfg,cfg['tail_weight_grid'])
   assert np.allclose(h,archived[m],atol=1e-10)
   ast=time.perf_counter();audit=-np.expm1(-xx['audit'].dot(h));worst=float(abs(audit-ys['audit']).max());audit_seconds=time.perf_counter()-ast
   times=[]
   for k in range(12):
    start=time.perf_counter()
    pred=[d.predict(panels['test'][t],t,h) for t in e.points]
    if k:times.append(time.perf_counter()-start)
   rows.append(dict(context=name,nodes=e.n,model=m,parameters=d.count,input_bytes=d.payload,parameter_bytes=h.nbytes,preprocessing_seconds=preprocessing,fit_seconds=record['seconds'],audit_seconds=audit_seconds,inference_seconds=float(np.median(times)),inference_min_seconds=min(times),inference_max_seconds=max(times),state_vectors=sum(len(p) for p in panels['test'].values())))
  times=[]
  for k in range(12):
   start=time.perf_counter();pred=[e.source(panels['test'][t],t) for t in e.points]
   if k:times.append(time.perf_counter()-start)
  rows.append(dict(context=name,nodes=e.n,model='H',parameters=0,input_bytes=sum(g.nbytes+8 for gs in e.groups.values() for g in gs),parameter_bytes=16,preprocessing_seconds=0,fit_seconds=0,audit_seconds=0,inference_seconds=float(np.median(times)),inference_min_seconds=min(times),inference_max_seconds=max(times),state_vectors=sum(len(p) for p in panels['test'].values())))
  print('benchmarked',name,flush=True)
 write_csv(OUT/'runtime.csv',rows)
 write_json(OUT/'runtime_environment.json',dict(cpu=platform.processor(),platform=platform.platform(),numpy=np.__version__,blas_threads=1,inference_repeats=11,warmup_repeats=1,fit_repeats=1,description='Separate sequential process after full experiment completion; no simultaneous experiment workers. Source labels known; fitting times exclude common label generation. Payload excludes Python object overhead and is not RAM. Preprocessing includes candidate input assembly and design matrices.'))
if __name__=='__main__':main()
