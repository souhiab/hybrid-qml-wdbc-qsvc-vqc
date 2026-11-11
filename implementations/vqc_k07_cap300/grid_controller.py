import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
import argparse
import importlib.metadata
import concurrent.futures
import copy
import json
import multiprocessing
import queue
import subprocess
import threading
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from vqc_engine import ROOT,sha,atomic_json
from fit_worker import fit_one,validate_receipt

def check():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    assert len(manifest['fits'])==192 and len(manifest['configurations'])==64
    assert all(r['qubits']==7 and r['budget']==300 and r['training_rows']==455 and r['evaluation_rows']==114 for r in manifest['fits'])
    for name in ['source_lock','input_lock']:
        lock=json.loads((ROOT/f'records/{name}.json').read_text())
        assert all(sha(ROOT/file)==digest for file,digest in lock['files'].items()),name+' mismatch'
    validation=json.loads((ROOT/'records/scheduling_numerical_validation.json').read_text())
    assert validation['passed'] and validation['engine_sha256']==sha(ROOT/'vqc_engine.py')
    environment=json.loads((ROOT/'records/environment.json').read_text())
    assert all(importlib.metadata.version(name)==version for name,version in environment['packages'].items()),'Verified environment package version changed'
    split=json.loads((ROOT/'split.json').read_text())
    for p in range(4):
        audit=json.loads((ROOT/f'preprocessing/p{p}.audit.json').read_text())
        assert audit['fit_rows']==455 and not audit['test_in_fit'] and not audit['test_labels_in_fit']
        assert audit['scaler_fit_index_sha256']==audit['rfe_fit_index_sha256']==split['training_index_sha256']
        with np.load(ROOT/f'preprocessing/p{p}.npz',allow_pickle=False) as d:
            assert d['X_train'].shape==(455,7) and d['X_validation'].shape==(114,7)
            assert d['train_indices'].tolist()==split['pilot_training_indices'] and d['validation_indices'].tolist()==split['validation_indices']
    return manifest

class Writer:
    def __init__(self):
        self.queue=queue.Queue(maxsize=4);self.process=None;self.error=None
        self.thread=threading.Thread(target=self.work,daemon=True);self.thread.start()
    def work(self):
        while True:
            item=self.queue.get()
            try:
                if item is None:return
                command,payload=item
                atomic_json(ROOT/'workbook_support/excel_snapshot.json',payload)
                answer=None
                for attempt in range(3):
                    try:
                        if self.process is None or self.process.poll() is not None:
                            stderr=(ROOT/'records/workbook_stderr.log').open('a',encoding='utf-8')
                            self.process=subprocess.Popen(['node',str(ROOT/'workbook_support/workbook_writer.mjs')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,encoding='utf-8')
                        self.process.stdin.write(command+'\n');self.process.stdin.flush()
                        while True:
                            line=self.process.stdout.readline()
                            if not line:raise OSError('Workbook writer exited')
                            if line.startswith('@@'):
                                answer=json.loads(line[2:])
                                if not answer['ok']:raise OSError(answer['error'])
                                break
                        self.error=None;break
                    except Exception as error:
                        self.error=str(error)
                        if self.process and self.process.poll() is None:self.process.terminate();self.process.wait()
                        self.process=None
                atomic_json(ROOT/'records/workbook_writer_status.json',{'last_command':command,'error':self.error,'receipts_preserved':True})
            finally:self.queue.task_done()
    def submit(self,payload,final=False):self.queue.put(('final' if final else 'sync',copy.deepcopy(payload)))
    def finish(self,payload):
        self.submit(payload,True);self.queue.join()
        if self.error:raise OSError(self.error)
        assert json.loads((ROOT/'records/workbook_readback.json').read_text())['allPassed']

def tables(manifest,results,attempts):
    rows=[];summary=[]
    for row in manifest['fits']:
        result=results.get(row['fit_id']);valid=result and result['status']=='completed'
        out={key:row[key] for key in ['fit_id','configuration_id','scaler','rfe_estimator','encoder','topology','feature_map_reps','ansatz_reps','initialization_seed','sampler_seed','budget']}
        out.update({'status':result['status'] if result else 'pending','termination':result.get('termination') if result else None,'nfev':result.get('nfev') if result else None,'runtime_seconds':result['runtime']['total_seconds'] if result else None,'error':result.get('error') if result else None})
        for metric in ['accuracy','roc_auc','sensitivity','specificity','precision','f1','correct_count','TP','FN','FP','TN']:out[metric]=result['metrics']['validation'][metric] if valid else None
        rows.append(out)
    for config in manifest['configurations']:
        group=[r for r in rows if r['configuration_id']==config['configuration_id']]
        valid=[r for r in group if r['status']=='completed']
        out={'configuration_id':config['configuration_id'],'complete':len(valid)==3,'valid_seed_outcomes':len(valid),'failed_seed_outcomes':sum(r['status']=='failed' for r in group),'pending_seed_outcomes':sum(r['status']=='pending' for r in group)}
        for metric in ['accuracy','roc_auc','sensitivity','specificity','precision','f1']:out['mean_'+metric]=float(np.mean([r[metric] for r in valid])) if len(valid)==3 else None
        out['sample_std_accuracy']=float(np.std([r['accuracy'] for r in valid],ddof=1)) if len(valid)==3 else None
        out['sum_correct_count']=sum(r['correct_count'] for r in valid) if len(valid)==3 else None
        summary.append(out)
    payload={'Grid_Results':rows,'Configuration_Summary':summary,'Coverage':[{k:v for k,v in r.items() if k in ['configuration_id','complete','valid_seed_outcomes','failed_seed_outcomes','pending_seed_outcomes']} for r in summary],'Attempts_Errors':[{'fit_id':fid,**attempt} for fid,records in attempts.items() for attempt in records],'Protocol':[{'item':k,'value':json.dumps(v) if isinstance(v,(list,dict)) else v} for k,v in manifest['protocol'].items()],'Provenance':[{'item':f,'value':sha(ROOT/f)} for f in ['manifest.json','split.json','records/source_lock.json','records/input_lock.json','records/environment.json']]}
    for name,data in payload.items():
        file=ROOT/'tables'/f'{name}.csv';temp=file.with_suffix('.tmp');pd.DataFrame(data).to_csv(temp,index=False);temp.replace(file)
    atomic_json(ROOT/'status.json',{'execution_state':'running','controller_active':True,'qubits':7,'budget':300,'expected_fits':192,'completed':sum(r['status']=='completed' for r in rows),'failed':sum(r['status']=='failed' for r in rows),'pending':sum(r['status']=='pending' for r in rows)})
    return payload

def run(manifest):
    other=ROOT.parent/'wdbc_vqc_execution_20261108/controller.lock'
    if other.exists():
        pid=json.loads(other.read_text())['pid']
        if psutil.pid_exists(pid) and 'sequence_controller.py' in ' '.join(psutil.Process(pid).cmdline()):raise RuntimeError('Current budget pilots are still active. This prepared grid is deliberately not launched alongside them.')
    lock=ROOT/'controller.lock'
    if lock.exists() and psutil.pid_exists(json.loads(lock.read_text())['pid']):raise RuntimeError('Grid controller already running')
    atomic_json(lock,{'pid':os.getpid()})
    try:
        # First creation is deferred until this deliberately invoked future run.
        marker=ROOT/'records/workbook_authoring_started.json'
        if not marker.exists():
            helper=Path('C:/Users/souha/.codex/plugins/cache/openai-primary-runtime/spreadsheets/26.1105.12137/skills/spreadsheets/container_tools/mark_artifact_operation_started.mjs')
            subprocess.run(['node',str(helper),'--operation-kind','create','--expected-output-count','1','--output-format','xlsx'],check=True)
            atomic_json(marker,{'started':True})
        results={};attempts={}
        for row in manifest['fits']:
            path=ROOT/'artifacts'/row['artifact_key'];records=[]
            for file in sorted(path.glob('attempt_*/attempt.json')):
                attempt=json.loads(file.read_text())
                if attempt['status']=='running':attempt.update({'status':'interrupted','error':'Restart from saved original initialization/seeds; exact optimizer resume unavailable'});atomic_json(file,attempt)
                records.append(attempt)
            attempts[row['fit_id']]=records
            receipt=validate_receipt(row)
            if receipt:results[row['fit_id']]=receipt
        vm=psutil.virtual_memory();reserve=max(1_500_000_000,int(.12*vm.total))
        workers=max(1,min(3,max(1,(psutil.cpu_count(logical=False) or 2)-1),int(max(0,vm.available-reserve)//500_000_000)))
        atomic_json(ROOT/'records/execution_settings.json',{'workers':workers,'threads_per_worker':1,'available_ram':vm.available,'ram_reserve':reserve,'cpu_only':True,'worker_memory_allowance_with_safety_margin':500_000_000})
        writer=Writer();payload=tables(manifest,results,attempts);writer.submit(payload)
        for round_number in range(2):
            pending=[]
            for row in manifest['fits']:
                r=results.get(row['fit_id']);failures=sum(a.get('status')=='failed' and a.get('technical_failure') for a in attempts[row['fit_id']])
                if r is None or (r['status']=='failed' and r.get('technical_failure') and failures<2):pending.append(row)
            if not pending:break
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures={pool.submit(fit_one,row,len(attempts[row['fit_id']])+1,workers):row for row in pending}
                for future in concurrent.futures.as_completed(futures):
                    row=futures[future]
                    try:result=future.result()
                    except Exception:
                        result=validate_receipt(row)
                        if not result or result['status']!='completed':raise
                    results[row['fit_id']]=result
                    if result.get('attempt_record'):attempts[row['fit_id']].append(result['attempt_record'])
                    payload=tables(manifest,results,attempts);writer.submit(payload)
                    print('SAVED '+row['fit_id']+' '+result['status'],flush=True)
        payload=tables(manifest,results,attempts);writer.finish(payload)
        verified=[validate_receipt(row) for row in manifest['fits']]
        failed=[row['fit_id'] for row,r in zip(manifest['fits'],verified) if not r or r['status']!='completed']
        assert not failed,'Unresolved outcomes: '+str(failed)
        complete=payload['Configuration_Summary'];best=max(r['sum_correct_count'] for r in complete)
        shortlist=[r for r in complete if r['sum_correct_count']==best]
        pd.DataFrame(shortlist).to_csv(ROOT/'tables/Accuracy_Shortlist.csv',index=False)
        atomic_json(ROOT/'records/final_verification.json',{'completed':192,'failed':0,'configuration_summaries':64,'predictions_and_metrics_verified':True,'workbook_all_sheets_match_tables':True,'budget_exception_retained':True,'all_fits_converged_claim':False})
        atomic_json(ROOT/'COMPLETED.json',{'verified_fits':192,'final_verification_sha256':sha(ROOT/'records/final_verification.json')})
        atomic_json(ROOT/'status.json',{'execution_state':'complete','controller_active':False,'completed':192,'failed':0,'pending':0})
    except Exception as error:
        atomic_json(ROOT/'FAILED.json',{'error_type':type(error).__name__,'error':str(error),'receipts_preserved':True})
        raise
    finally:
        if lock.exists():lock.unlink()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    manifest=check()
    if args.run and not args.check_only:run(manifest)
    else:print('READY: seven-qubit full grid, 64 configurations, 192 fresh full-training fits, cap 300. No training launched.')
