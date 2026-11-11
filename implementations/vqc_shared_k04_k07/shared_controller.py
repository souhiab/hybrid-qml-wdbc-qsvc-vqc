import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[n]='1'
import argparse,concurrent.futures,json,multiprocessing,subprocess,datetime
from pathlib import Path
import numpy as np,pandas as pd,psutil
from vqc_engine import atomic_json,sha,clinical
from fit_worker import fit_one,validate_receipt
from writer_support import Writer
ROOT=Path(__file__).resolve().parent
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def check():
 m=json.loads((ROOT/'manifest.json').read_text());assert len(m['fits'])==180 and len(m['configurations'])==4
 for key,value in json.loads((ROOT/'records/package_lock.json').read_text())['files'].items():assert sha(ROOT/key)==value,key
 for row in m['fits']:
  d=Path(row['study_root']);split=json.loads((d/'split.json').read_text());assert len(set(split['pilot_training_indices'])&set(split['validation_indices']))==0
  audit=json.loads((d/f'preprocessing/p{row["p"]}.audit.json').read_text())
  assert audit['fit_rows']==row['training_rows'] and not audit['validation_rows_in_fit'] and not audit['validation_labels_in_fit']
  assert audit['scaler_fit_index_sha256']==audit['rfe_fit_index_sha256']==split['training_index_sha256']
  assert audit['rfe_fit_labels_sha256']==split['training_label_sha256']
 return m
def table_save(name,rows):
 p=ROOT/'tables'/f'{name}.csv';t=p.with_suffix('.tmp');pd.DataFrame(rows).to_csv(t,index=False);t.replace(p)
def snapshot(m,results,attempts,final=False):
 rows=[];summary=[];coverage=[]
 for r in m['fits']:
  v=results.get(r['fit_id']);good=v and v['status']=='completed'
  row={k:r[k] for k in ['fit_id','candidate_id','source_configuration_id','qubits','budget','repeat_id','outer_fold_index','seed_pair_id','initialization_seed','sampler_seed']}
  row.update(status=v['status'] if v else 'pending',termination=v.get('termination') if v else None,nfev=v.get('nfev') if v else None,runtime_seconds=v['runtime']['total_seconds'] if v else None)
  for key in ['accuracy','roc_auc','average_precision','sensitivity','specificity','precision','f1','correct_count','TP','FN','FP','TN']:row[key]=v['metrics']['validation'][key] if good else None
  rows.append(row)
 for c in m['configurations']:
  for repeat in [1,2,3]:
   for fold in range(5):
    group=[r for r in rows if r['candidate_id']==c['candidate_id'] and r['repeat_id']==repeat and r['outer_fold_index']==fold];valid=[r for r in group if r['status']=='completed'];complete=len(valid)==3
    row={'candidate_id':c['candidate_id'],'repeat_id':repeat,'outer_fold_index':fold,'complete':complete,'valid_seed_outcomes':len(valid),'expected':3,'failed_seed_outcomes':sum(r['status']=='failed' for r in group),'pending_seed_outcomes':sum(r['status']=='pending' for r in group)}
    for metric in ['accuracy','roc_auc','average_precision','sensitivity','specificity','precision','f1']:row['mean_'+metric]=float(np.mean([r[metric] for r in valid])) if complete else None
    row['sample_sd_accuracy']=float(np.std([r['accuracy'] for r in valid],ddof=1)) if complete else None
    summary.append(row);coverage.append({k:row[k] for k in ['candidate_id','repeat_id','outer_fold_index','complete','valid_seed_outcomes','expected','failed_seed_outcomes','pending_seed_outcomes']})
 payload={'Grid_Results':rows,'Configuration_Summary':summary,'Repeat_Summary':[],'Coverage':coverage,'Attempts_Errors':[{'fit_id':fid,**a} for fid,items in attempts.items() for a in items],'Protocol':[{'item':k,'value':json.dumps(v) if isinstance(v,(dict,list)) else v} for k,v in m['protocol'].items()],'Provenance':[{'item':name,'value':sha(ROOT/name)} for name in ['manifest.json','candidates.json','folds/outer_inner_assignments.json','records/package_lock.json']]}
 if final:payload['Repeat_Summary']=consolidate_oof(m,results)
 for name,records in payload.items():table_save(name,records)
 completed=sum(r['status']=='completed' for r in rows);failed=sum(r['status']=='failed' for r in rows)
 atomic_json(ROOT/'status.json',{'state':'RUNNING','pid':os.getpid(),'expected_fits':180,'completed':completed,'failed':failed,'pending':180-completed-failed,'updated':now()})
 return payload
def consolidate_oof(m,results):
 predictions=[];repeat_rows=[];aggregates=[]
 for c in m['configurations']:
  pool=[]
  for repeat in [1,2,3]:
   seed_metrics=[]
   for seed in [1,2,3]:
    subset=[r for r in m['fits'] if r['candidate_id']==c['candidate_id'] and r['repeat_id']==repeat and r['seed_pair_id']==seed]
    parts=[]
    for row in subset:
     result=results[row['fit_id']];assert result['status']=='completed'
     with np.load(Path(row['study_root'])/result['attempt_path']/'predictions.npz') as d:
      frame=pd.DataFrame({'candidate_id':c['candidate_id'],'repeat_id':repeat,'outer_fold_index':row['outer_fold_index'],'seed_pair_id':seed,'initialization_seed':row['initialization_seed'],'sampler_seed':row['sampler_seed'],'sample_index':d['validation_indices'],'source_label':d['validation_labels'],'prediction':d['validation_predictions'],'malignant_probability':d['validation_probabilities'][:,0]})
      parts.append(frame);predictions.append(frame)
    part=pd.concat(parts).sort_values('sample_index');assert part['sample_index'].tolist()==list(range(569))
    prob=part['malignant_probability'].to_numpy();metrics=clinical(part['source_label'],part['prediction'],np.c_[prob,1-prob])
    seed_metrics.append(metrics);repeat_rows.append({'candidate_id':c['candidate_id'],'repeat_id':repeat,'seed_pair_id':seed,'aggregation':'pooled OOF, one patient once within repeat/seed',**metrics})
   mean={metric:float(np.mean([d[metric] for d in seed_metrics])) for metric in ['accuracy','roc_auc','average_precision','sensitivity','specificity','precision','f1']}
   repeat_rows.append({'candidate_id':c['candidate_id'],'repeat_id':repeat,'seed_pair_id':'mean-of-3','aggregation':'mean of three seed-specific pooled OOF metrics; predictions not pooled as extra patients',**mean});pool.append(mean)
  r=[results[x['fit_id']] for x in m['fits'] if x['candidate_id']==c['candidate_id']]
  aggregates.append({'candidate_id':c['candidate_id'],'complete_fits':45,'fold_mean_accuracy':float(np.mean([v['metrics']['validation']['accuracy'] for v in r])),'mean_repeat_pooled_accuracy':float(np.mean([x['accuracy'] for x in pool])),'sample_sd_repeat_pooled_accuracy':float(np.std([x['accuracy'] for x in pool],ddof=1)),'mean_repeat_pooled_auc':float(np.mean([x['roc_auc'] for x in pool])),'converged':sum(v['termination']=='converged' for v in r),'budget_exhausted':sum(v['termination']=='budget_exhausted' for v in r),'failed':0})
 table_save('Patient_Predictions',pd.concat(predictions).to_dict('records'));table_save('Candidate_Summaries',aggregates)
 return repeat_rows
def run(m):
 lock=ROOT/'controller.lock'
 if lock.exists() and psutil.pid_exists(json.loads(lock.read_text())['pid']):raise RuntimeError('Shared-fold controller already active')
 if lock.exists():lock.unlink()
 fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,json.dumps({'pid':os.getpid(),'started':now()}).encode());os.close(fd)
 try:
  # One coordinator owns workbook writes; workers only save per-fit evidence.
  marker=ROOT/'records/workbook_authoring_started.json'
  if not marker.exists():
   helper='C:/Users/souha/.codex/plugins/cache/openai-primary-runtime/spreadsheets/26.1105.12137/skills/spreadsheets/container_tools/mark_artifact_operation_started.mjs'
   subprocess.run(['node',helper,'--operation-kind','create','--expected-output-count','1','--output-format','xlsx'],check=True);atomic_json(marker,{'started':now()})
  results={};attempts={}
  for row in m['fits']:
   directory=Path(row['study_root'])/'artifacts'/row['artifact_key'];records=[]
   for p in sorted(directory.glob('attempt_*/attempt.json')):
    a=json.loads(p.read_text())
    if a['status']=='running':a.update(status='interrupted',reason='Restart from original saved initialization; optimizer restoration unsupported');atomic_json(p,a)
    records.append(a)
   attempts[row['fit_id']]=records;receipt=validate_receipt(row)
   if receipt:results[row['fit_id']]=receipt
  vm=psutil.virtual_memory();reserve=max(1_500_000_000,int(.12*vm.total));workers=max(1,min(3,(psutil.cpu_count(logical=False) or 2)-1,int(max(0,vm.available-reserve)//500_000_000)))
  if vm.available<reserve+500_000_000:raise MemoryError('Insufficient safe RAM; preserved fits can resume later')
  atomic_json(ROOT/'records/execution_resources.json',{'workers':workers,'threads_per_worker':1,'available':vm.available,'reserve':reserve,'cpu_only':True})
  writer=Writer();writer.submit(snapshot(m,results,attempts))
  for round_id in range(2):
   pending=[]
   for row in m['fits']:
    result=results.get(row['fit_id']);failures=sum(a.get('status')=='failed' and a.get('technical_failure') for a in attempts[row['fit_id']])
    if result is None or (result['status']=='failed' and result.get('technical_failure') and failures<2):pending.append(row)
   if not pending:break
   with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
    futures={pool.submit(fit_one,row,len(attempts[row['fit_id']])+1,workers):row for row in pending}
    for future in concurrent.futures.as_completed(futures):
     row=futures[future];result=future.result();results[row['fit_id']]=result
     if result.get('attempt_record'):attempts[row['fit_id']].append(result['attempt_record'])
     if result['status']=='completed':assert validate_receipt(row)['status']=='completed'
     writer.submit(snapshot(m,results,attempts));print('SAVED '+row['fit_id']+' '+result['status'],flush=True)
  unresolved=[row['fit_id'] for row in m['fits'] if results.get(row['fit_id'],{}).get('status')!='completed']
  if unresolved:
   writer.finish(snapshot(m,results,attempts));raise RuntimeError('Unresolved fits: '+str(unresolved))
  assert all(validate_receipt(row)['status']=='completed' for row in m['fits'])
  writer.finish(snapshot(m,results,attempts,final=True))
  verification={'verified_at':now(),'fits':180,'candidate_count':4,'outer_partitions':15,'seeds':3,'all_receipts_and_metrics_verified':True,'training_only_preprocessing':True,'workbook_readback_all_passed':True,'fold_manifest_sha256':sha(ROOT/'folds/outer_inner_assignments.json'),'all_fits_converged_claim':False,'all_seed_outcomes_retained':True}
  atomic_json(ROOT/'records/final_verification.json',verification);atomic_json(ROOT/'COMPLETED.json',{'verified_fits':180,'final_verification_sha256':sha(ROOT/'records/final_verification.json')});atomic_json(ROOT/'status.json',{'state':'COMPLETE','completed':180,'failed':0,'pending':0})
 except Exception as e:
  atomic_json(ROOT/'ERROR.json',{'error':str(e),'at':now(),'completed_fits_preserved':True});raise
 finally:
  if lock.exists():lock.unlink()
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');a=parser.parse_args();m=check()
 if a.run:run(m)
 else:print('VERIFIED: four candidates, 15 shared partitions, three seeds, 180 fits. No VQC training launched.')
