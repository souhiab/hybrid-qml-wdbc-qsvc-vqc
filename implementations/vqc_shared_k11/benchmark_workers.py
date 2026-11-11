import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import concurrent.futures,multiprocessing,time,json,importlib.util
from pathlib import Path
import numpy as np
import psutil
Q=Path(__file__).resolve().parent
def work(seed,warm=False):
 spec=importlib.util.spec_from_file_location('candidate',Q/'vqc_engine.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 if warm:time.sleep(1);return os.getpid()
 rng=np.random.default_rng(seed);k=11;states=rng.normal(size=(32,1<<k))+1j*rng.normal(size=(32,1<<k));states/=np.linalg.norm(states,axis=1)[:,None]
 ops=[('ry',q,float(rng.random())) for q in range(k)]+[('cx',q,q-1) for q in range(k-1,0,-1)]+[('ry',q,float(rng.random())) for q in range(k)]
 started=time.perf_counter()
 for repeat in range(10):
  for row in states:m.sampled_binary_probabilities(m.evolve_ansatz(row[None,:].copy(),ops),1101)
 return {'seconds':time.perf_counter()-started,'rss':psutil.Process().memory_info().rss}
if __name__=='__main__':
 results=[]
 for workers in [1,2,3,4]:
  with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
   list(pool.map(work,range(workers),[True]*workers))
   started=time.perf_counter();out=list(pool.map(work,range(8)));elapsed=time.perf_counter()-started
   results.append({'workers':workers,'threads_each':1,'seconds':elapsed,'synthetic_tasks_per_second':8/elapsed,'peak_task_rss':max(r['rss'] for r in out)})
 best=max(results,key=lambda x:x['synthetic_tasks_per_second'])
 report={'results':results,'selected_workers':best['workers'],'available_ram':psutil.virtual_memory().available,'training_performed':False,'limitations':'Synthetic inner-loop throughput; actual full-fit throughput may differ.'}
 (Q/'records/worker_benchmark.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
