from __future__ import annotations
import argparse
import concurrent.futures
import contextlib
import csv
import datetime
import gc
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from qiskit import qpy
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.utils import algorithm_globals as algorithm_rng
from qiskit_machine_learning.utils import algorithm_globals as ml_rng
from threadpoolctl import threadpool_limits

from vqc_engine import ROOT,VERSION,atomic_json,sha,array_sha,CachedStatevectorQNN,RecordedCOBYLA,clinical
from prepare_vqc import circuits

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')

def append_event(name,payload):
    with (ROOT/'records'/name).open('a',encoding='utf-8') as handle:
        handle.write(json.dumps({'time':now(),**payload},default=str)+'\n');handle.flush()

def validate_receipt(row):
    path=ROOT/'artifacts'/row['fit_id']/'receipt.json'
    if not path.exists():return None
    try:
        receipt=json.loads(path.read_text())
        if receipt['status']!='completed':return receipt if receipt['status']=='failed' else None
        attempt=ROOT/receipt['attempt_path']
        for filename,digest in receipt['file_hashes'].items():
            if sha(attempt/filename)!=digest:raise RuntimeError(f'Hash mismatch: {filename}')
        with np.load(attempt/'predictions.npz',allow_pickle=False) as arrays:
            split=json.loads((ROOT/'source/historical/split_idx.json').read_text())
            assert arrays['test_indices'].tolist()==split['test_idx']
            assert arrays['train_indices'].tolist()==split['train_idx']
            for label in ('train','test'):
                probs=arrays[f'{label}_probabilities'];pred=arrays[f'{label}_predictions']
                assert len(probs)==(455 if label=='train' else 114)
                assert probs.shape[1]==2 and np.isfinite(probs).all()
                np.testing.assert_allclose(probs.sum(axis=1),1,atol=1e-12)
                np.testing.assert_array_equal(pred,np.argmax(probs,axis=1))
                expected=clinical(arrays[f'{label}_labels'],pred,probs)
                for key,value in expected.items():assert abs(value-receipt['metrics'][label][key])<1e-12
        assert receipt['objective_evaluations']<=100
        assert receipt['termination'] in ('converged','budget_exhausted')
        assert receipt['seed_pair_id']==row['seed_pair_id']
        np.testing.assert_array_equal(receipt['initial_parameters'],row['initial_parameters'])
        return receipt
    except Exception as error:
        append_event('checkpoint_validation_errors.jsonl',{'fit_id':row['fit_id'],'error':str(error)})
        raise

class MemoryObserver:
    def __init__(self):
        self.process=psutil.Process();self.peak=self.process.memory_info().rss
        self.stop=threading.Event();self.samples=[]
    def sample(self):
        while not self.stop.wait(.1):
            rss=self.process.memory_info().rss;self.peak=max(self.peak,rss)
    def __enter__(self):
        self.thread=threading.Thread(target=self.sample,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.peak=max(self.peak,self.process.memory_info().rss);self.stop.set();self.thread.join()

def fit_one(row,threads=1,benchmark=False):
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]=str(threads)
    fit_root=ROOT/'artifacts'/row['fit_id'];fit_root.mkdir(exist_ok=True)
    previous=sorted(fit_root.glob('attempt_*'))
    attempt_id=f'attempt_{len(previous)+1:03d}'
    attempt=fit_root/attempt_id;attempt.mkdir()
    prior_attempt=previous[-1].name if previous else None
    started=now();timer=time.perf_counter();cpu=time.process_time()
    atomic_json(attempt/'attempt.json',{'fit_id':row['fit_id'],'attempt_id':attempt_id,'state':'running','pid':os.getpid(),'started':started,'restarted_from':prior_attempt,'restart_uses_original_initial_point':True,'exact_optimizer_resume':False})
    atomic_json(attempt/'initialization.json',{'initial_parameters':row['initial_parameters'],'parameter_order':row['weight_parameter_order'],'initialization_seed':row['initialization_seed'],'sampler_seed':row['sampler_seed']})
    observer=MemoryObserver()
    try:
        with observer,threadpool_limits(limits=threads),warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            np.random.seed(row['initialization_seed']);algorithm_rng.random_seed=row['initialization_seed'];ml_rng.random_seed=row['initialization_seed']
            with np.load(ROOT/f'preprocessing/{row["preprocessing_id"]}.npz',allow_pickle=False) as data:
                Xtrain=data['X_train'];ytrain=data['y_train'];itrain=data['train_idx']
                # Test observations are intentionally loaded only after model.fit.
                names=data['selected_names'].astype(str).tolist()
            fmap,ansatz=circuits(row)
            initial=np.asarray(row['initial_parameters'],dtype=np.float64)
            sampler=StatevectorSampler(default_shots=1024,seed=int(row['sampler_seed']))
            optimizer=RecordedCOBYLA(attempt,budget=100)
            model=VQC(feature_map=fmap,ansatz=ansatz,loss='cross_entropy',optimizer=optimizer,
                sampler=sampler,warm_start=False,initial_point=initial.copy())
            network=CachedStatevectorQNN(fmap,ansatz,sampler,row['sampler_seed'])
            model._neural_network=network
            train_started=time.perf_counter();model.fit(Xtrain,ytrain);fit_seconds=time.perf_counter()-train_started
            if optimizer.raw_status['termination']=='optimizer_failed':raise ArithmeticError('Optimizer termination failure: '+str(optimizer.raw_status))
            if not all(call['rows']==455 and call['input_sha256']==array_sha(Xtrain) for call in network.forward_calls):raise RuntimeError('Non-training inputs reached optimization')
            optimization_forward_calls=len(network.forward_calls)
            # VQC one-hot encoding sorts the original source labels; verify ordering.
            classes=model._target_encoder.categories_[0].astype(int).tolist()
            if classes!=[0,1]:raise RuntimeError(f'Unexpected source-class probability order {classes}')
            final=np.asarray(model.weights,dtype=np.float64)
            if final.shape!=initial.shape or not np.isfinite(final).all():raise ArithmeticError('Invalid final parameters')
            train_prob=np.asarray(model.predict_proba(Xtrain),dtype=np.float64)
            with np.load(ROOT/f'preprocessing/{row["preprocessing_id"]}.npz',allow_pickle=False) as data:
                Xtest=data['X_test'];ytest=data['y_test'];itest=data['test_idx']
            before=len(network.forward_calls)
            test_prob=np.asarray(model.predict_proba(Xtest),dtype=np.float64)
            if len(network.forward_calls)!=before+1:raise RuntimeError('Expected one final test probability call')
            train_pred=np.argmax(train_prob,axis=1).astype(np.int64);test_pred=np.argmax(test_prob,axis=1).astype(np.int64)
            metrics={'train':clinical(ytrain,train_pred,train_prob),'test':clinical(ytest,test_pred,test_prob)}
            np.savez_compressed(attempt/'predictions.npz',train_indices=itrain,test_indices=itest,train_labels=ytrain,test_labels=ytest,train_predictions=train_pred,test_predictions=test_pred,train_probabilities=train_prob,test_probabilities=test_prob,initial_parameters=initial,final_parameters=final)
            pd.DataFrame({'sample_index':itest,'source_label':ytest,'predicted_source_label':test_pred,'malignant_probability':test_prob[:,0],'benign_probability':test_prob[:,1]}).to_csv(attempt/'test_predictions.csv',index=False)
            pd.DataFrame({'sample_index':itrain,'source_label':ytrain,'predicted_source_label':train_pred,'malignant_probability':train_prob[:,0],'benign_probability':train_prob[:,1]}).to_csv(attempt/'train_predictions.csv',index=False)
            prep_audit=json.loads((ROOT/f'preprocessing/{row["preprocessing_id"]}.audit.json').read_text())
            atomic_json(attempt/'preprocessing.json',prep_audit)
            atomic_json(attempt/'fit_audit.json',{'train_index_sha256':array_sha(itrain),'train_label_sha256':array_sha(ytrain),'optimization_input_sha256':array_sha(Xtrain),'optimization_forward_calls':optimization_forward_calls,'test_inputs_loaded_after_fit':True,'test_rows_in_any_fit':False,'test_labels_in_any_fit':False,'final_test_probability_calls':1,'classes':classes,'positive_source_label':0,'tie_rule':'argmax first column; malignant wins tie'})
            atomic_json(attempt/'parameters.json',{'initial':initial,'final':final,'parameter_order':row['weight_parameter_order'],'feature_parameter_order':row['input_parameter_order']})
            shutil.copy2(ROOT/row['circuit_path'],attempt/'circuits.qpy')
            atomic_json(attempt/'configuration.json',{'manifest_row':row,'protocol':json.loads((ROOT/'manifest.json').read_text())['protocol'],'environment_sha256':sha(ROOT/'records/environment.json'),'source_lock_sha256':sha(ROOT/'records/production_source_lock.json') if (ROOT/'records/production_source_lock.json').exists() else sha(ROOT/'records/validation_source_lock.json'),'reconstruction':'Load circuits.qpy (feature map, ansatz, composed circuit), bind parameters.json final vector in recorded order, simulate feature-map state and ansatz, sample 1024 shots using integer sampler seed for each bound circuit; output=basis_state%2'})
            serialization={'supported':True,'path':'model.dill'}
            try:model.save(str(attempt/'model.dill'))
            except Exception as error:serialization={'supported':False,'error':str(error)}
            atomic_json(attempt/'serialization.json',serialization)
            atomic_json(attempt/'warnings.json',[{'category':str(w.category.__name__),'message':str(w.message)} for w in captured])
            history=optimizer.history
            runtime={'started':started,'ended':now(),'total_seconds':time.perf_counter()-timer,'fit_seconds':fit_seconds,'cpu_seconds':time.process_time()-cpu,'peak_rss_bytes':observer.peak,'threads':threads,'pid':os.getpid(),'benchmark':benchmark}
            receipt={**{k:v for k,v in row.items() if k!='circuit_operations'},'status':'completed','attempt_id':attempt_id,'attempt_path':str(attempt.relative_to(ROOT)),'restarted_from':prior_attempt,
                'implementation_version':VERSION,'termination':optimizer.raw_status['termination'],'optimizer_status':optimizer.raw_status,
                'objective_evaluations':len(history),'initial_loss':history[0]['loss'],'last_evaluated_loss':history[-1]['loss'],'minimum_training_loss':min(item['loss'] for item in history),'fitted_objective':optimizer.raw_status['fun'],
                'initial_parameters':initial.tolist(),'final_parameters':final.tolist(),'selected_features':names,'metrics':metrics,'runtime':runtime,'model_serialization':serialization,
                'file_hashes':{p.name:sha(p) for p in attempt.iterdir() if p.is_file() and p.name!='attempt.json'},'verification':'verified'}
            atomic_json(attempt/'receipt.json',receipt)
            atomic_json(attempt/'attempt.json',{'fit_id':row['fit_id'],'attempt_id':attempt_id,'state':'completed','started':started,'ended':now(),'restarted_from':prior_attempt,'termination':receipt['termination']})
            atomic_json(fit_root/'receipt.json',receipt)
            return receipt
    except Exception as error:
        technical=isinstance(error,(OSError,MemoryError,TimeoutError))
        failure={**{k:v for k,v in row.items() if k!='circuit_operations'},'status':'failed','attempt_id':attempt_id,'attempt_path':str(attempt.relative_to(ROOT)),'restarted_from':prior_attempt,'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),'technical_failure':technical,'runtime':{'started':started,'ended':now(),'total_seconds':time.perf_counter()-timer,'peak_rss_bytes':observer.peak,'threads':threads},'objective_evaluations':len(optimizer.history) if 'optimizer' in locals() else 0,'termination':'failed','verification':'failed'}
        atomic_json(attempt/'failure.json',failure);atomic_json(fit_root/'receipt.json',failure)
        atomic_json(attempt/'attempt.json',{'fit_id':row['fit_id'],'attempt_id':attempt_id,'state':'failed','ended':now(),'error':str(error),'technical_failure':technical})
        return failure

def valid_results(manifest):
    return {row['fit_id']:result for row in manifest['fits'] if (result:=validate_receipt(row)) is not None}

def make_snapshot(manifest,results):
    run_rows=[];train_rows=[];grid_rows=[];summaries=[]
    for row in manifest['fits']:
        r=results.get(row['fit_id'])
        item={k:row[k] for k in ['fit_id','configuration_id','seed_pair_id','initialization_seed','sampler_seed','scaler','rfe_estimator','encoder','topology','feature_map_reps','ansatz_reps']}
        item.update({'status':r['status'] if r else 'pending','termination':r['termination'] if r else None,'objective_evaluations':r.get('objective_evaluations') if r else None})
        for key in ['accuracy','correct_count','roc_auc','average_precision','sensitivity','specificity','precision','f1','TP','FN','FP','TN']:
            item[key]=r['metrics']['test'][key] if r and r['status']=='completed' else None
        item.update({'runtime_seconds':r['runtime']['total_seconds'] if r else None,'peak_rss_bytes':r['runtime'].get('peak_rss_bytes') if r else None,'attempt_id':r['attempt_id'] if r else None,'verification':r.get('verification') if r else None,'artifact_path':r['attempt_path'] if r else None,'error':r.get('error') if r else None})
        run_rows.append(item)
        if r:
            train_rows.append({'fit_id':row['fit_id'],'attempt_id':r['attempt_id'],'objective_evaluations':r.get('objective_evaluations'),'termination':r['termination'],'initial_loss':r.get('initial_loss'),'last_evaluated_loss':r.get('last_evaluated_loss'),'minimum_training_loss':r.get('minimum_training_loss'),'fitted_objective':r.get('fitted_objective'),'trace_path':r['attempt_path']+'/training_loss.jsonl','optimizer_status':r.get('optimizer_status',{}).get('message',r.get('error'))})
    for config in manifest['configurations']:
        group=[row for row in run_rows if row['configuration_id']==config['configuration_id']]
        valid=[r for r in group if r['status']=='completed'];failed=sum(r['status']=='failed' for r in group)
        complete=len(valid)==3
        total_correct=sum(r['correct_count'] for r in valid) if complete else None
        item={k:config[k] for k in ['configuration_id','scaler','rfe_estimator','encoder','topology','feature_map_reps','ansatz_reps']}
        item.update({'valid_seed_outcomes':len(valid),'failed_seed_outcomes':failed,'pending_seed_outcomes':3-len(valid)-failed,'complete':complete,'failure_rate':failed/3,'total_correct_across_seeds':total_correct})
        for metric in ['accuracy','roc_auc','sensitivity','specificity','f1']:
            item['mean_'+metric]=(total_correct/342 if metric=='accuracy' else float(np.mean([r[metric] for r in valid]))) if complete else None
        item.update({'sample_std_accuracy':float(np.std([r['accuracy'] for r in valid],ddof=1)) if complete else None,'min_accuracy':min(r['accuracy'] for r in valid) if complete else None,'max_accuracy':max(r['accuracy'] for r in valid) if complete else None,'accuracy_shortlist':False})
        summaries.append(item)
        grid_rows.append({k:v for k,v in item.items() if not k.startswith('mean_') and k not in ('sample_std_accuracy','min_accuracy','max_accuracy','accuracy_shortlist')})
    complete=[r for r in summaries if r['complete']]
    if complete:
        best=max(r['total_correct_across_seeds'] for r in complete)
        for row in summaries:row['accuracy_shortlist']=row['complete'] and row['total_correct_across_seeds']==best
    completed=sum(r['status']=='completed' for r in run_rows);failed=sum(r['status']=='failed' for r in run_rows)
    status={'updated':now(),'expected_fits':192,'expected_configurations':64,'completed':completed,'failed':failed,'pending':192-completed-failed,'complete_configurations':len(complete),'failure_rate':failed/192,'converged':sum(r['termination']=='converged' for r in run_rows),'budget_exhausted':sum(r['termination']=='budget_exhausted' for r in run_rows),'runner_pid':os.getpid()}
    summary=[{'item':key,'value':value} for key,value in status.items()]
    summary.extend([{'item':'primary_ranking','value':manifest['protocol']['primary_ranking']},{'item':'interpretation','value':'Exploratory fixed split; three seeds assess training variability on the same patients'},{'item':'budget','value':'100 objective evaluations; budget-limited valid fits retained'}])
    provenance=[{'item':key,'value':json.dumps(value) if isinstance(value,(dict,list)) else value} for key,value in manifest['protocol'].items()]
    for file in ['manifest.json','source/data/wdbc_arrays.npz','source/historical/split_idx.json','records/environment.json','records/simulator_validation.json','vqc_engine.py','run_vqc.py']:
        provenance.append({'item':file,'value':sha(ROOT/file)})
    payload={'Summary':summary,'Run_Results':run_rows,'Configuration_Summary':summaries,'Grid_Coverage':grid_rows,'Training_Log':train_rows,'Provenance':provenance}
    atomic_json(ROOT/'status.json',status);atomic_json(ROOT/'workbook_support/snapshot.json',payload)
    tables=ROOT/'tables';tables.mkdir(exist_ok=True)
    for name,rows in payload.items():
        if rows:
            temp=tables/f'{name}.csv.tmp';pd.DataFrame(rows).to_csv(temp,index=False);os.replace(temp,tables/f'{name}.csv')
    return payload,status

class WorkbookCoordinator:
    def __init__(self):
        self.process=None
    def sync(self):
        try:
            if self.process is None or self.process.poll() is not None:
                self.stderr=(ROOT/'records/workbook_stderr.log').open('a',encoding='utf-8')
                self.process=subprocess.Popen(['node',str(ROOT/'workbook_support/workbook_writer.mjs')],cwd=ROOT/'workbook_support',stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,encoding='utf-8')
            self.process.stdin.write('sync\n');self.process.stdin.flush()
            while True:
                line=self.process.stdout.readline()
                if not line:raise RuntimeError('Workbook writer exited before synchronization')
                if line.startswith('@@'):
                    ack=json.loads(line[2:])
                    if not ack['ok']:raise RuntimeError(ack['error'])
                    return True
        except Exception as error:
            append_event('workbook_save_errors.jsonl',{'error':str(error),'receipts_preserved':True})
            return False
    def close(self):
        if self.process and self.process.poll() is None:
            self.process.stdin.write('final\n');self.process.stdin.flush()
            while True:
                line=self.process.stdout.readline()
                if not line:break
                if line.startswith('@@'):print(line.strip(),flush=True)
            self.process.wait()

def source_lock(name):
    files=['vqc_engine.py','run_vqc.py','prepare_vqc.py','validate_vqc.py','workbook_support/workbook_writer.mjs']
    atomic_json(ROOT/'records'/name,{'version':VERSION,'locked':now(),'files':{f:sha(ROOT/f) for f in files}})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    manifest=json.loads((ROOT/'manifest.json').read_text());lock=json.loads((ROOT/'records/manifest_lock.json').read_text())
    assert sha(ROOT/'manifest.json')==lock['manifest_sha256']
    assert json.loads((ROOT/'records/simulator_validation.json').read_text())['passed']
    if args.verify_only:
        results=valid_results(manifest);_,status=make_snapshot(manifest,results);print(json.dumps(status));return
    lockpath=ROOT/'runner.lock'
    if lockpath.exists():
        old=json.loads(lockpath.read_text())
        if psutil.pid_exists(old['pid']):raise RuntimeError(f'Runner already active: {old}')
    fd=os.open(lockpath,os.O_CREAT|os.O_EXCL|os.O_WRONLY) if not lockpath.exists() else None
    if fd is not None:
        with os.fdopen(fd,'w') as handle:json.dump({'pid':os.getpid(),'started':now()},handle)
    else:atomic_json(lockpath,{'pid':os.getpid(),'started':now(),'stale_lock_replaced':True})
    workbook=WorkbookCoordinator()
    try:
        results=valid_results(manifest)
        # A surviving 'running' attempt has no restorable COBYLA state: retain it as interrupted.
        for row in manifest['fits']:
            if row['fit_id'] in results:continue
            for path in (ROOT/'artifacts'/row['fit_id']).glob('attempt_*/attempt.json'):
                old=json.loads(path.read_text())
                if old.get('state')=='running':
                    old.update({'state':'interrupted','interruption_recognized':now(),'restart_policy':'new attempt; original initialization and seed pair'});atomic_json(path,old)
        source_lock('validation_source_lock.json')
        make_snapshot(manifest,results);workbook.sync()
        benchmark_results=[]
        for fit_id in manifest['protocol']['benchmark_ids']:
            row=next(row for row in manifest['fits'] if row['fit_id']==fit_id)
            result=results.get(fit_id)
            if not result or result['status']!='completed':
                result=fit_one(row,threads=1,benchmark=True);results[fit_id]=result
                make_snapshot(manifest,results);workbook.sync()
                print(f"BENCHMARK {fit_id} {result['status']} {result['termination']} {result['runtime']['total_seconds']:.2f}s",flush=True)
            if result['status']!='completed':raise RuntimeError('Benchmark failed: '+result.get('error',''))
            benchmark_results.append(result)
        histories=[]
        for result in benchmark_results:
            path=ROOT/result['attempt_path']/'training_loss.jsonl'
            trace=[json.loads(line) for line in path.read_text().splitlines()]
            losses=np.asarray([row['loss'] for row in trace])
            histories.append({'fit_id':result['fit_id'],'objective_evaluations':len(trace),'termination':result['termination'],'first_loss':float(losses[0]),'last_evaluated_loss':float(losses[-1]),'min_loss':float(losses.min()),'first_10_mean':float(losses[:10].mean()),'last_10_mean':float(losses[-10:].mean()),'loss_trace_sha256':sha(path)})
        vm=psutil.virtual_memory();physical=psutil.cpu_count(logical=False) or 1
        peak=max(r['runtime']['peak_rss_bytes'] for r in benchmark_results)
        workers=max(1,min(3,max(1,physical-1),int(max(0,vm.available-1_500_000_000)//max(peak*2,1))))
        # Preserve the fixed budget. Review contains training-only histories, with no test metric.
        review={'reviewed':now(),'training_histories':histories,'decision':'Retain the locked 100-objective-evaluation budget across the full declared grid. Budget-exhausted results are valid budget-limited outcomes; no budget revision or selection based on test results.','production_budget':100,'worker_count':workers,'threads_per_worker':1,'physical_cores':physical,'available_ram':vm.available,'benchmark_peak_rss':peak,'system_ram_reserve':1_500_000_000,'per_worker_memory_factor':2,'reason':'At most three independent one-thread fits, leaving one physical core for coordinator/system; memory margin based on both benchmark observations.'}
        atomic_json(ROOT/'records/benchmark_training_review.json',review)
        source_lock('production_source_lock.json')
        print(f'PRODUCTION FROZEN workers={workers} threads=1 budget=100',flush=True)
        pending=[]
        for row in manifest['fits']:
            result=results.get(row['fit_id'])
            if result is None:pending.append(row)
            elif result['status']=='failed' and result.get('technical_failure') and len(list((ROOT/'artifacts'/row['fit_id']).glob('attempt_*')))<2:pending.append(row)
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(fit_one,row,1,False):row for row in pending}
            for future in concurrent.futures.as_completed(futures):
                row=futures[future]
                try:result=future.result()
                except Exception as error:
                    result={**row,'status':'failed','termination':'failed','error_type':type(error).__name__,'error':str(error),'technical_failure':type(error).__name__=='BrokenProcessPool','attempt_id':'process_failure','attempt_path':f'artifacts/{row["fit_id"]}','runtime':{'total_seconds':0},'verification':'failed'}
                    atomic_json(ROOT/'artifacts'/row['fit_id']/'receipt.json',result)
                results[row['fit_id']]=result
                if result['status']=='failed' and result.get('technical_failure') and len(list((ROOT/'artifacts'/row['fit_id']).glob('attempt_*')))<2:
                    append_event('technical_retries.jsonl',{'fit_id':row['fit_id'],'failed_attempt':result['attempt_id'],'rule':'one retry for technical failure only'})
                    result=fit_one(row,1,False);results[row['fit_id']]=result
                make_snapshot(manifest,results);workbook.sync()
                complete=sum(r['status']=='completed' for r in results.values())
                print(f"{complete}/192 {row['fit_id']} {result['status']} {result['termination']} {result['runtime']['total_seconds']:.2f}s",flush=True)
        verified=valid_results(manifest);_,status=make_snapshot(manifest,verified);workbook.sync()
        atomic_json(ROOT/'records/execution_completion.json',status)
        print('COMPLETE '+json.dumps(status),flush=True)
    finally:
        workbook.close()
        if lockpath.exists() and json.loads(lockpath.read_text()).get('pid')==os.getpid():lockpath.unlink()

if __name__=='__main__':main()
