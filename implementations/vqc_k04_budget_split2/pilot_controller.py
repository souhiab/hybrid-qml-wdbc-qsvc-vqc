from __future__ import annotations
import argparse
import concurrent.futures
import datetime
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from vqc_engine import ROOT,atomic_json,sha
from pilot_fit import fit_one,validate_receipt,now
from pilot_analysis import snapshot,final_analysis,write_report

def event(file,payload):
    with (ROOT/'records'/file).open('a',encoding='utf-8') as handle:handle.write(json.dumps({'time':now(),**payload},default=str)+'\n');handle.flush()

class ResourceMonitor:
    def __enter__(self):
        self.stop=threading.Event();self.samples=[];self.process=psutil.Process();psutil.cpu_percent(interval=None)
        def observe():
            last=time.perf_counter()
            while not self.stop.wait(.25):
                current=time.perf_counter();vm=psutil.virtual_memory()
                self.samples.append({'cpu_percent':psutil.cpu_percent(interval=None),'available_ram':vm.available,'memory_percent':vm.percent,'scheduler_delay_seconds':max(0,current-last-.25)})
                last=current
        self.thread=threading.Thread(target=observe,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join()
    def summary(self):
        if not self.samples:return {'mean_cpu_percent':None,'min_available_ram':psutil.virtual_memory().available,'max_scheduler_delay_seconds':0}
        return {'mean_cpu_percent':float(np.mean([s['cpu_percent'] for s in self.samples])),'peak_cpu_percent':max(s['cpu_percent'] for s in self.samples),'min_available_ram':min(s['available_ram'] for s in self.samples),'peak_memory_percent':max(s['memory_percent'] for s in self.samples),'max_scheduler_delay_seconds':max(s['scheduler_delay_seconds'] for s in self.samples),'samples':len(self.samples)}

class ExcelWriter:
    def __init__(self):self.process=None;self.error_count=0
    def command(self,command):
        for retry in range(3):
            try:
                if self.process is None or self.process.poll() is not None:
                    error_log=(ROOT/'records/workbook_stderr.log').open('a',encoding='utf-8')
                    self.process=subprocess.Popen(['node',str(ROOT/'workbook_support/workbook_writer.mjs')],cwd=ROOT/'workbook_support',stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=error_log,text=True,encoding='utf-8')
                self.process.stdin.write(command+'\n');self.process.stdin.flush()
                while True:
                    line=self.process.stdout.readline()
                    if not line:raise OSError('Workbook writer exited before acknowledgement')
                    if line.startswith('@@'):
                        answer=json.loads(line[2:])
                        if not answer['ok']:raise OSError(answer['error'])
                        return answer['result']
            except Exception as error:
                self.error_count+=1;event('workbook_save_errors.jsonl',{'error':str(error),'retry':retry+1,'fit_receipts_preserved':True})
                if self.process and self.process.poll() is None:self.process.terminate();self.process.wait()
                self.process=None;time.sleep(.5)
        return None
    def finish(self):return self.command('final')

class Controller:
    def __init__(self):
        self.manifest=json.loads((ROOT/'manifest.json').read_text())
        lock=json.loads((ROOT/'records/manifest_lock.json').read_text())
        assert sha(ROOT/'manifest.json')==lock['manifest_sha256']
        assert sha(ROOT/'split.json')==lock['split_sha256']
        assert sha(ROOT/'source/original_training_only.npz')==lock['training_data_sha256']
        assert sha(ROOT/'vqc_engine.py')==lock['engine_sha256']==lock['original_engine_sha256']
        assert json.loads((ROOT/'records/numerical_validation.json').read_text())['passed']
        self.audit=[]
        assert len(self.manifest['fits'])==96 and len(self.manifest['configurations'])==16
        assert self.manifest['protocol']['budgets']==[100,200]
        assert json.loads((ROOT/'split.json').read_text())['random_state']==20261109
        self.results={};self.attempts={};self.attempt_numbers={}
        self.benchmarks=[];self.uncertainty=None;self.writer=ExcelWriter()
        # The only recovery inventory; thereafter receipts and returned attempts are tracked in memory.
        for row in self.manifest['fits']:
            fit_root=ROOT/'artifacts'/row['fit_id']
            attempts=[]
            for file in sorted(fit_root.glob('attempt_*/attempt.json')):
                attempt=json.loads(file.read_text())
                if attempt['status']=='running':
                    attempt.update({'status':'interrupted','ended':now(),'error':'Controller restart detected an incomplete attempt; original initialization will restart under a new attempt ID.'})
                    atomic_json(file,attempt)
                attempts.append(attempt)
            self.attempts[row['fit_id']]=attempts;self.attempt_numbers[row['fit_id']]=len(attempts)
            result=validate_receipt(row)
            if result is not None:self.results[row['fit_id']]=result
        atomic_json(ROOT/'records/startup_inventory.json',{'time':now(),'completed':sum(r['status']=='completed' for r in self.results.values()),'failed':sum(r['status']=='failed' for r in self.results.values()),'pending':96-len(self.results),'attempts':sum(self.attempt_numbers.values())})
        atomic_json(ROOT/'records/startup_confirmation.json',{'started':now(),'pid':os.getpid(),'expected_fits':96,'configurations':16,'budgets':[100,200],'internal_split_seed':20261109,'manifest_sha256':sha(ROOT/'manifest.json'),'manifest_loaded':True,'initialization_successful':True})
        print('STUDY INITIALIZED: split2, 96 fits, budgets 100/200, seed 20261109',flush=True)
        if (ROOT/'records/resource_benchmarks.json').exists():self.benchmarks=json.loads((ROOT/'records/resource_benchmarks.json').read_text())['benchmarks']

    def sync(self):
        self.payload,self.status=snapshot(self.manifest,self.results,self.attempts,self.audit,self.benchmarks,self.uncertainty)
        self.writer.command('sync')

    def pending(self,rows):
        pending=[]
        for row in rows:
            result=self.results.get(row['fit_id'])
            if result is None or (result['status']=='failed' and result.get('technical_failure') and self.attempt_numbers[row['fit_id']]<2):pending.append(row)
        return pending

    def accept(self,row,result):
        self.results[row['fit_id']]=result
        self.attempts[row['fit_id']].append(result['attempt_record'])
        self.sync()
        event('fit_completion.jsonl',{'fit_id':row['fit_id'],'budget':row['budget'],'status':result['status'],'termination':result['termination'],'nfev':result['nfev'],'runtime_seconds':result['runtime']['total_seconds']})
        print(f"saved {len(self.results)}/96 {row['fit_id']} {result['status']} nfev={result['nfev']}",flush=True)

    def run_batch(self,rows,workers,benchmark=False):
        vm=psutil.virtual_memory();reserve=max(1_000_000_000,int(vm.total*.06))
        safe=max(1,min(workers,int(max(0,vm.available-reserve)//400_000_000)))
        if safe<workers:event('resource_reductions.jsonl',{'requested':workers,'selected':safe,'available_ram':vm.available,'reserve':reserve,'reason':'automatic live-memory safety reduction'})
        workers=safe
        pending=self.pending(rows)
        if not pending:return {'workers':workers,'new_fits':0,'resumed':True}
        timer=time.perf_counter();finished=[]
        with ResourceMonitor() as monitor,concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures={}
            for row in pending:
                self.attempt_numbers[row['fit_id']]+=1
                futures[pool.submit(fit_one,row,self.attempt_numbers[row['fit_id']],workers if benchmark else None)]=row
            for future in concurrent.futures.as_completed(futures):
                row=futures[future]
                try:result=future.result()
                except Exception as error:
                    metadata={'fit_id':row['fit_id'],'attempt_id':f'attempt_{self.attempt_numbers[row["fit_id"]]:03d}','status':'failed','started':None,'ended':now(),'technical_failure':True,'error':f'{type(error).__name__}: {error}','exact_optimizer_resume':False}
                    result={**row,'status':'failed','termination':'failed','nfev':0,'attempt_id':metadata['attempt_id'],'attempt_path':f'artifacts/{row["fit_id"]}/{metadata["attempt_id"]}','attempt_record':metadata,'technical_failure':True,'error':metadata['error'],'runtime':{'total_seconds':0,'peak_rss_bytes':0,'started':now(),'ended':now()}}
                    atomic_json(ROOT/'artifacts'/row['fit_id']/'receipt.json',result)
                self.accept(row,result);finished.append(result)
        # One retry for technical failures only; original failure remains in its attempt directory.
        for row in pending:
            result=self.results[row['fit_id']]
            if result['status']=='failed' and result.get('technical_failure') and self.attempt_numbers[row['fit_id']]<2:
                event('technical_retries.jsonl',{'fit_id':row['fit_id'],'prior_attempt':result['attempt_id'],'rule':'one retry for technical failures only'})
                self.attempt_numbers[row['fit_id']]+=1
                result=fit_one(row,self.attempt_numbers[row['fit_id']],workers if benchmark else None)
                self.accept(row,result)
        elapsed=time.perf_counter()-timer
        valid=[r for r in finished if r['status']=='completed']
        starts=[datetime.datetime.fromisoformat(r['runtime']['started']) for r in valid]
        ends=[datetime.datetime.fromisoformat(r['runtime']['ended']) for r in valid]
        fit_span=(max(ends)-min(starts)).total_seconds() if valid else elapsed
        summary={'workers':workers,'threads_per_worker':1,'new_fits':len(finished),'valid_fits':len(valid),'wall_seconds_including_excel':elapsed,'fit_wall_span_seconds':fit_span,'fits_per_second_fit_span':len(valid)/max(1,fit_span),'fits_per_second_including_excel':len(valid)/elapsed,'max_worker_rss_bytes':max((r['runtime']['peak_rss_bytes'] for r in finished),default=0),**monitor.summary()}
        return summary

    def choose_resources(self):
        vm=psutil.virtual_memory()
        reserve=max(1_000_000_000,int(vm.total*.06))
        peak=200_000_000
        workers=max(1,min(3,max(1,(psutil.cpu_count(logical=False) or 2)-1),int(max(0,vm.available-reserve)//(2*peak))))
        settings={'selected_workers':workers,'threads_per_worker':1,'cpu_only':True,'physical_cores':psutil.cpu_count(logical=False),'logical_processors':psutil.cpu_count(),'system_ram_reserve_bytes':reserve,'available_ram_at_freeze':vm.available,'prior_measured_worker_rss_bytes':peak,'memory_margin_factor_per_worker':2,'reason':'Reuse validated three-worker first-pilot execution, reduced only for current CPU/RAM headroom. No extra tuning fits or metric-driven changes.','optimizations':['unchanged cached encoder states','vectorized exact seeded shot counting','invariant preprocessing reuse','incremental sole-writer Excel'],'frozen':now()}
        atomic_json(ROOT/'records/execution_settings.json',settings)
        return settings

    def verify_final(self):
        expected=self.manifest['fits'];max_error=0
        verified={}
        for row in expected:
            result=validate_receipt(row)
            if result is None:raise RuntimeError(f'Missing outcome: {row["fit_id"]}')
            verified[row['fit_id']]=result
        assert len(verified)==96 and len(self.manifest['configurations'])==16
        for config in self.manifest['configurations']:
            assert config['feature_map_index']==(config['p']+config['r'])%4
            for pair in (1,2,3):
                rows=[r for r in expected if r['configuration_id']==config['configuration_id'] and r['seed_pair_id']==pair]
                assert sorted(r['budget'] for r in rows)==[100,200]
                assert all(r['initial_parameters']==rows[0]['initial_parameters'] for r in rows)
        run=pd.read_csv(ROOT/'tables/Run_Results.csv',keep_default_na=False)
        assert len(run)==96 and run.fit_id.nunique()==96
        for row in run.itertuples(index=False):
            result=verified[row.fit_id]
            if result['status']=='completed':
                for metric in ['accuracy','roc_auc','average_precision','sensitivity','specificity','precision','f1']:
                    max_error=max(max_error,abs(float(getattr(row,metric))-result['metrics']['validation'][metric]))
        configs=pd.read_csv(ROOT/'tables/Configuration_Summary.csv',keep_default_na=False)
        assert len(configs)==32
        for row in configs.itertuples(index=False):
            group=[r for r in verified.values() if r['configuration_id']==row.configuration_id and r['budget']==int(row.budget) and r['status']=='completed']
            if len(group)==3:
                accuracies=[r['metrics']['validation']['accuracy'] for r in group]
                assert abs(float(row.mean_accuracy)-np.mean(accuracies))<1e-12
                assert abs(float(row.sample_std_accuracy)-np.std(accuracies,ddof=1))<1e-12
        readback=json.loads((ROOT/'records/workbook_readback.json').read_text())
        source=json.loads((ROOT/'records/source_lock.json').read_text())
        sources_match=all(sha(ROOT/file)==digest for file,digest in source['files'].items())
        assert readback['allPassed'] and sources_match and max_error<=1e-12
        failed=sum(r['status']=='failed' for r in verified.values())
        report={'verified':True,'verified_at':now(),'configurations':16,'seed_pairs':3,'budgets':[100,200],'expected_outcomes':96,'completed':96-failed,'failed':failed,'pending':0,'both_stages_complete':all(sum(r['budget']==b for r in verified.values())==48 for b in (100,200)),'all_valid_panels':failed==0,'train_rows':364,'validation_rows':91,'split_seed':20261109,'split_differs_from_first_pilot':json.loads((ROOT/'split.json').read_text())['differs_from_first_split'],'original_test_inspected':False,'training_only_fit_access_verified':True,'initial_vectors_identical_across_budgets':True,'max_table_metric_recomputation_error':max_error,'workbook_all_sheets_match_tables':True,'source_files_match_production_freeze':sources_match,'original_engine_unchanged':sha(ROOT/'vqc_engine.py')==json.loads((ROOT/'records/manifest_lock.json').read_text())['original_engine_sha256'],'workbook_sha256':sha(ROOT/'WDBC_VQC_Budget_Confirmation_Split2_Master.xlsx'),'failures_and_interruptions_accounted_for':True,'budget_selection':'deferred_to_later_user_review'}
        atomic_json(ROOT/'records/final_verification.json',report)
        return report

    def run(self):
        self.sync()
        settings=self.choose_resources()
        print(f'CPU FROZEN workers={settings["selected_workers"]} threads=1; budgets 100/200 only',flush=True)
        for budget in (100,200):
            rows=[row for row in self.manifest['fits'] if row['budget']==budget]
            summary=self.run_batch(rows,settings['selected_workers'])
            counts={'completed':sum(self.results.get(r['fit_id'],{}).get('status')=='completed' for r in rows),'failed':sum(self.results.get(r['fit_id'],{}).get('status')=='failed' for r in rows)}
            atomic_json(ROOT/f'records/stage_{budget}.json',{'budget':budget,'expected':48,**counts,'execution_measurement':summary,'finished':now()})
            if counts['completed']+counts['failed']!=48:raise RuntimeError(f'Stage {budget} has missing outcomes')
            print(f'STAGE {budget} finished outcomes={counts["completed"]+counts["failed"]}/48 failed={counts["failed"]}; continuing fixed protocol',flush=True)
        self.uncertainty=final_analysis(self.manifest,self.results)
        self.sync()
        write_report(self.manifest,self.results,self.payload,settings,self.status)
        final=self.writer.finish()
        if not final or not final['allPassed']:raise RuntimeError('Final workbook readback failed; receipts retained')
        verification=self.verify_final()
        if verification['failed']:raise RuntimeError('Study outcomes include failed fits; see failure receipts. Completion marker withheld.')
        self.status.update({'execution_state':'complete','controller_active':False,'final_verification':'records/final_verification.json'})
        atomic_json(ROOT/'status.json',self.status)
        atomic_json(ROOT/'records/execution_completion.json',{'finished':now(),**verification})
        atomic_json(ROOT/'COMPLETED.json',{'completed':now(),'expected_fits':96,'verified_fits':96,'final_verification_sha256':sha(ROOT/'records/final_verification.json'),'interpretation':'deferred_to_later_user_prompt'})
        print('SPLIT2 BOTH BUDGETS COMPLETE AND VERIFIED '+json.dumps(verification),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rebuild',action='store_true');args=parser.parse_args()
    lockfile=ROOT/'controller.lock'
    if lockfile.exists():
        old=json.loads(lockfile.read_text())
        if psutil.pid_exists(old['pid']):raise RuntimeError(f'Controller already active: {old["pid"]}')
        lockfile.unlink()
    fd=os.open(lockfile,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as handle:json.dump({'pid':os.getpid(),'started':now()},handle)
    try:
        files=['vqc_engine.py','prepare_pilot.py','pilot_fit.py','pilot_analysis.py','pilot_controller.py','workbook_support/workbook_writer.mjs']
        source_path=ROOT/'records/source_lock.json'
        if source_path.exists():
            source=json.loads(source_path.read_text());assert all(sha(ROOT/file)==digest for file,digest in source['files'].items())
        else:atomic_json(source_path,{'frozen':now(),'files':{file:sha(ROOT/file) for file in files}})
        Controller().run()
    except Exception as error:
        event('controller_errors.jsonl',{'error_type':type(error).__name__,'error':str(error)})
        status=json.loads((ROOT/'status.json').read_text()) if (ROOT/'status.json').exists() else {'expected_fits':96}
        status.update({'execution_state':'failed','controller_active':False,'error_type':type(error).__name__,'error':str(error),'failed_at':now()})
        atomic_json(ROOT/'status.json',status)
        atomic_json(ROOT/'FAILED.json',{'failed_at':now(),'error_type':type(error).__name__,'error':str(error),'receipts_preserved':True})
        raise
    finally:
        if lockfile.exists() and json.loads(lockfile.read_text())['pid']==os.getpid():lockfile.unlink()

if __name__=='__main__':main()
