import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[n]='1'
os.environ['VQC_CHUNK_ROWS']='1'
import argparse,json,concurrent.futures,multiprocessing
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from fit_worker import fit_one,validate_receipt
from vqc_engine import atomic_json,sha
ROOT=Path(__file__).resolve().parent
def run(budget):
 m=json.loads((ROOT/'manifest.json').read_text());rows=[r for r in m['fits'] if r['budget']==budget];assert len(rows)==48
 assert json.loads((ROOT/'records/optimized_circuit_verification.json').read_text())['passed']
 for file,digest in json.loads((ROOT/'records/source_lock.json').read_text())['files'].items():assert sha(ROOT/file)==digest
 results={}
 for r in rows:
  receipt=validate_receipt(r)
  if receipt and receipt['status']=='completed':results[r['fit_id']]=receipt
 atomic_json(ROOT/'status.json',{'budget':budget,'completed':len(results),'expected':48,'state':'running','skip_verified_completed':True})
 from adaptive_scheduler import run_adaptive
 run_adaptive(ROOT,budget,rows,results,validate_receipt,atomic_json)
 assert len(results)==48
 assert all(validate_receipt(r)['status']=='completed' for r in rows)
 atomic_json(ROOT/f'COMPLETED_{budget}.json',{'verified':48,'budget':budget,'fresh_initializations':True})
 return
 # Keep a single three-seed configuration bank active; bound private memory separately.
 for config in m['configurations']:
  group=[r for r in rows if r['configuration_id']==config['configuration_id'] and r['fit_id'] not in results]
  if not group:continue
  vm=psutil.virtual_memory();reserve=max(1_500_000_000,int(.12*vm.total));bank=455*(2**17)*16
  if vm.available<reserve+bank+512_000_000:raise MemoryError('Insufficient safe RAM for 17-qubit group; resume when available')
  measured=json.loads((ROOT/'records/optimized_worker_benchmark.json').read_text())['selected_workers']
  workers=max(1,min(measured,3,(psutil.cpu_count(logical=False) or 2)-1,int((vm.available-reserve-bank)//512_000_000)))
  atomic_json(ROOT/'records/resources_latest.json',{'workers':workers,'threads':1,'available':vm.available,'reserve':reserve,'shared_bank_upper':bank,'private_allowance':512_000_000})
  for round_id in range(2):
   pending=[r for r in group if r['fit_id'] not in results]
   if not pending:break
   with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
    futures={}
    for r in pending:
     directory=ROOT/'artifacts'/r['artifact_key'];attempts=sorted(directory.glob('attempt_*/attempt.json'))
     failures=0
     for p in attempts:
      a=json.loads(p.read_text())
      if a['status']=='running':a.update(status='interrupted',error='Fresh restart from original initialization; no optimizer restoration');atomic_json(p,a)
      failures+=int(a['status']=='failed' and a.get('technical_failure',False))
     if failures>=2:raise RuntimeError('Technical retry limit reached: '+r['fit_id'])
     futures[pool.submit(fit_one,r,len(attempts)+1,workers)]=r
    for future in concurrent.futures.as_completed(futures):
     r=futures[future];receipt=future.result()
     if receipt['status']=='completed':
      assert validate_receipt(r)['status']=='completed';results[r['fit_id']]=receipt
     elif not receipt.get('technical_failure'):raise RuntimeError(str(receipt))
     pd.DataFrame([{'fit_id':v['fit_id'],'budget':budget,'termination':v['termination'],'nfev':v['nfev'],'runtime_seconds':v['runtime']['total_seconds'],**v['metrics']['validation']} for v in results.values()]).to_csv(ROOT/f'tables/budget_{budget}.csv',index=False)
     atomic_json(ROOT/'status.json',{'budget':budget,'completed':len(results),'expected':48,'workers':workers})
 assert len(results)==48
 assert all(validate_receipt(r)['status']=='completed' for r in rows)
 atomic_json(ROOT/f'COMPLETED_{budget}.json',{'verified':48,'budget':budget,'fresh_initializations':True})
def compare():
 m=json.loads((ROOT/'manifest.json').read_text());base=ROOT;bm=m
 panels={};summaries=[]
 for budget in [100,300,600]:
  rows=[r for r in (bm if budget==100 else m)['fits'] if r['budget']==budget];panel=[];receipts=[]
  assert len(rows)==48
  for r in rows:
   receipt=validate_receipt(r);assert receipt and receipt['status']=='completed';receipts.append(receipt)
   with np.load(Path(r['study_root'])/receipt['attempt_path']/'predictions.npz') as d:panel.append(d['validation_predictions']==d['validation_labels'])
  panels[budget]=np.asarray(panel,dtype=float)
  summaries.append({'budget':budget,'mean_accuracy':float(panels[budget].mean()),'converged':sum(r['termination']=='converged' for r in receipts),'fits':48,'runtime_seconds':sum(r['runtime']['total_seconds'] for r in receipts),'actual_nfev':sum(r['nfev'] for r in receipts),'mean_auc':float(np.mean([r['metrics']['validation']['roc_auc'] for r in receipts]))})
 comparisons=[];draws=np.random.default_rng(20261108).integers(0,91,(10000,91))
 for hi,lo in [(300,100),(600,300),(600,100)]:
  diff=100*(panels[hi]-panels[lo]).mean(axis=0);ci=np.percentile(diff[draws].mean(axis=1),[2.5,97.5])
  comparisons.append({'comparison':f'{hi}-{lo}','change_pp':float(diff.mean()),'ci95_low':float(ci[0]),'ci95_high':float(ci[1]),'bootstrap_seed':20261108,'bootstrap_resamples':10000,'conditional_on_fixed_split_models_panel':True})
 pd.DataFrame(summaries).to_csv(ROOT/'tables/Budget_Summary.csv',index=False);pd.DataFrame(comparisons).to_csv(ROOT/'tables/Paired_Comparisons.csv',index=False)
 atomic_json(ROOT/'records/final_budget_comparison.json',{'summaries':summaries,'paired_comparisons':comparisons,'no_independent_validation_claim':True,'all_144_verified':True})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--budget',type=int,choices=[100,300,600]);p.add_argument('--compare',action='store_true');a=p.parse_args()
 if a.budget:run(a.budget)
 if a.compare:compare()
