from __future__ import annotations
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
os.environ['PYTHONIOENCODING']='utf-8'
import concurrent.futures
import datetime
import json
import multiprocessing
import queue
import copy
import shutil
import subprocess
import threading
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from vqc_engine import atomic_json,sha
from fit_worker import fit_one,validate_receipt
from sequence_analysis import snapshot,compare,paired_analysis,planning,final_report

ROOT=Path(__file__).resolve().parent
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
def event(name,values):
    with (ROOT/'records'/name).open('a',encoding='utf-8') as handle:handle.write(json.dumps({'time':now(),**values},default=str)+'\n');handle.flush()

class ExcelWriter:
    def __init__(self):self.process=None
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
                event('workbook_errors.jsonl',{'error':str(error),'retry':retry+1,'fit_receipts_preserved':True})
                if self.process and self.process.poll() is None:self.process.terminate();self.process.wait()
                self.process=None;time.sleep(.5)
        return None

class AsyncExcelWriter:
    """One sole writer, bounded ordered queue; verified receipts never wait on Excel."""
    def __init__(self):
        self.writer=ExcelWriter();self.tasks=queue.Queue(maxsize=4);self.error=None
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
    def _run(self):
        while True:
            payload=self.tasks.get()
            try:
                if payload is None:return
                atomic_json(ROOT/'workbook_support/excel_snapshot.json',payload)
                if self.writer.command('sync') is None:self.error='Excel synchronization deferred; receipts retained'
                else:self.error=None
            except Exception as error:
                self.error=str(error);event('workbook_errors.jsonl',{'error':str(error),'fit_receipts_preserved':True})
            finally:self.tasks.task_done()
    def submit(self,payload):self.tasks.put(copy.deepcopy(payload))
    def finish(self):
        self.tasks.join()
        return self.writer.command('final')

class Monitor:
    def __enter__(self):
        self.samples=[];self.stop=threading.Event();psutil.cpu_percent(interval=None)
        def observe():
            last=time.perf_counter()
            while not self.stop.wait(1):
                current=time.perf_counter()
                self.samples.append((psutil.cpu_percent(interval=None),psutil.virtual_memory().available,max(0,current-last-1)))
                last=current
        self.thread=threading.Thread(target=observe,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join()
    def summary(self):
        return {'mean_cpu_percent':float(np.mean([s[0] for s in self.samples])) if self.samples else None,'min_available_ram':min((s[1] for s in self.samples),default=psutil.virtual_memory().available),'maximum_scheduler_delay_seconds':max((s[2] for s in self.samples),default=0)}

class Controller:
    def __init__(self):
        self.master=json.loads((ROOT/'manifest.json').read_text())
        assert self.master['initial_fits']==480 and self.master['maximum_fits']==624
        assert [r['qubits'] for r in self.master['studies']]==[4,7,11,17]
        self.manifests={s['study_id']:json.loads(Path(s['manifest_path']).read_text()) for s in self.master['studies']}
        self.results={};self.attempts={};self.activated={sid:([200] if 'full-grid' in sid else [100,200]) for sid in self.manifests}
        self.uncertainties={};self.recommendations={};self.resources={};self.writer=AsyncExcelWriter();self.phase='startup'
        self.last_payload=None;self.pool=None;self.pool_workers=None;self.pool_width=None
        inputs=json.loads((ROOT/'records/input_lock.json').read_text())
        for path,digest in inputs['files'].items():
            if sha(ROOT/path)!=digest:raise RuntimeError('Frozen input hash mismatch '+path)
        assert json.loads((ROOT/'records/numerical_validation.json').read_text())['passed']
        for sid,m in self.manifests.items():
            directory=Path(m['fits'][0]['study_root'])
            decision=directory/'records/budget_decision.json'
            if decision.exists():
                d=json.loads(decision.read_text());self.recommendations[sid]=d
                if d['add_300']:self.activated[sid]=[100,200,300]
            paired=directory/'records/paired_analysis.json'
            if paired.exists():self.uncertainties[sid]=json.loads(paired.read_text())['uncertainty']
            resource=directory/'records/execution_settings.json'
            if resource.exists():self.resources[sid]=json.loads(resource.read_text())
            for row in m['fits']:
                fitroot=directory/'artifacts'/row['artifact_key']
                attempts=[]
                for file in sorted(fitroot.glob('attempt_*/attempt.json')):
                    a=json.loads(file.read_text())
                    if a['status']=='running':
                        a.update({'status':'interrupted','ended':now(),'error':'Incomplete attempt recovered; restart original vector and seeds with new attempt ID.'});atomic_json(file,a)
                    attempts.append(a)
                self.attempts[row['fit_id']]=attempts
                result=validate_receipt(row)
                if result:self.results[row['fit_id']]=result
            # Keep numerical implementation provenance local to each study.
            shutil.copy2(ROOT/'records/source_lock.json',directory/'records/source_lock.json')
        atomic_json(ROOT/'records/startup_inventory.json',{'time':now(),'valid_completed':sum(r['status']=='completed' for r in self.results.values()),'failed':sum(r['status']=='failed' for r in self.results.values()),'recorded_attempts':sum(map(len,self.attempts.values()))})
        atomic_json(ROOT/'records/startup_confirmation.json',{'pid':os.getpid(),'initialized':True,'time':now(),'initial_fits':480,'maximum_fits':624,'phase_order':[4,7,11,17],'phase_a_fits':192,'mandatory_pilot_fits':288,'manifests_verified':True,'numerical_validation_passed':True,'manifest_sha256':sha(ROOT/'manifest.json')})
        print('INITIALIZED: 480 required fits, 624 maximum; full-grid k04 at 200 then pilots k07/k11/k17; no larger full grids.',flush=True)

    def sync(self):
        self.payload,self.status=snapshot(self.master,self.manifests,self.results,self.attempts,self.activated,self.uncertainties,self.recommendations,self.phase)
        if self.payload!=self.last_payload:
            self.writer.submit(self.payload);self.last_payload=copy.deepcopy(self.payload)

    def close_pool(self):
        if self.pool is not None:self.pool.shutdown(wait=True,cancel_futures=False)
        self.pool=None;self.pool_workers=None;self.pool_width=None

    def get_pool(self,workers,width):
        if self.pool is None or self.pool_workers!=workers or self.pool_width!=width:
            self.close_pool()
            self.pool=concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'))
            self.pool_workers=workers;self.pool_width=width
        return self.pool

    def pending(self,rows):
        output=[]
        for row in rows:
            result=self.results.get(row['fit_id'])
            failures=sum(a.get('status')=='failed' and a.get('technical_failure') for a in self.attempts[row['fit_id']])
            if result is None or (result['status']=='failed' and result.get('technical_failure') and failures<2):output.append(row)
        return output

    def accept(self,row,result):
        self.results[row['fit_id']]=result
        if result.get('attempt_record'):self.attempts[row['fit_id']].append(result['attempt_record'])
        self.sync()
        print(f"SAVED {row['fit_id']} {result['status']} nfev={result.get('nfev',0)}",flush=True)

    def crashed_result(self,row,number,error):
        directory=Path(row['study_root']);attempt=directory/'artifacts'/row['artifact_key']/f'attempt_{number:03d}'
        attempt.mkdir(parents=True,exist_ok=True)
        metadata={'fit_id':row['fit_id'],'attempt_id':f'attempt_{number:03d}','status':'failed','started':None,'ended':now(),'error':str(error),'technical_failure':True,'restarted_from':f'attempt_{number-1:03d}' if number>1 else None}
        atomic_json(attempt/'attempt.json',metadata)
        result={**row,'status':'failed','attempt_id':metadata['attempt_id'],'attempt_path':str(attempt.relative_to(directory)),'attempt_record':metadata,'technical_failure':True,'error':f'{type(error).__name__}: {error}','termination':'failed','nfev':0,'runtime':{'started':now(),'ended':now(),'total_seconds':0,'peak_rss_bytes':0}}
        atomic_json(attempt/'failure.json',result);atomic_json(attempt/'receipt.json',result);atomic_json(attempt.parent/'receipt.json',result)
        return result

    def safe_workers(self,requested,peak):
        vm=psutil.virtual_memory();reserve=max(1_500_000_000,int(.12*vm.total))
        policy=ROOT/'records/k17_shared_memory_policy.json'
        if 'k17' in self.phase and policy.exists():
            shared=json.loads(policy.read_text())
            # k17 retains one encoder/preprocessing group at a time. Read-only
            # mappings refer to the same file, so count that bank once and use
            # a guarded per-process private-memory allowance instead of summing RSS.
            slots=int(max(0,vm.available-reserve-shared['shared_bank_upper_bytes'])//shared['private_allowance_per_worker_bytes'])
            return max(1,min(requested,3,max(1,(psutil.cpu_count(logical=False) or 2)-1),slots))
        return max(1,min(requested,max(1,(psutil.cpu_count(logical=False) or 2)-1),int(max(0,vm.available-reserve)//max(1,int(2*peak)))))

    def batch(self,rows,workers,peak=250_000_000):
        rows=self.pending(rows)
        if not rows:return {'new_fits':0,'valid_fits':0,'workers':workers,'resumed':True,'fits_per_second':0,'max_worker_rss_bytes':peak,'min_available_ram':psutil.virtual_memory().available}
        workers=self.safe_workers(workers,peak)
        timer=time.perf_counter();outcomes=[]
        # Keep a bounded pool across configuration groups. No per-task recycling
        # (the earlier Python 3.11 recycling path stalled); close at width/resource boundaries.
        pool=self.get_pool(workers,rows[0]['qubits'])
        with Monitor() as monitor:
            pending={}
            for row in rows:
                number=len(self.attempts[row['fit_id']])+1
                pending[pool.submit(fit_one,row,number,workers)]=(row,number)
            for future in concurrent.futures.as_completed(pending):
                row,number=pending[future]
                try:result=future.result()
                except Exception as error:
                    # A worker may save its receipt before a pool/transport failure.
                    # Recover that verified outcome instead of overwriting/repeating it.
                    result=validate_receipt(row)
                    if result is None or result['status']!='completed':
                        result=self.crashed_result(row,number,error)
                self.accept(row,result);outcomes.append(result)
        if getattr(pool,'_broken',False):self.close_pool()
        summary={'new_fits':len(outcomes),'valid_fits':sum(r['status']=='completed' for r in outcomes),'workers':workers,'threads_per_worker':1,'wall_seconds_including_checkpoint_tables':time.perf_counter()-timer,'excel_synchronized_asynchronously':True,'worker_pool_reused':True,'max_worker_rss_bytes':max((r['runtime'].get('peak_rss_bytes',0) for r in outcomes),default=peak),**monitor.summary()}
        summary['wall_seconds_including_excel']=summary['wall_seconds_including_checkpoint_tables']
        summary['fits_per_second']=summary['valid_fits']/max(1,summary['wall_seconds_including_excel'])
        event('resource_measurements.jsonl',{'phase':self.phase,**summary})
        # Retry only technical failures once, with a fresh worker and original initialization.
        for row in rows:
            if row in self.pending([row]):
                result=self.results.get(row['fit_id'])
                if result and result['status']=='failed' and result.get('technical_failure'):
                    self.batch([row],1,max(peak,summary['max_worker_rss_bytes']))
        return summary

    def select_resources(self,sid):
        if sid in self.resources:return self.resources[sid]
        m=self.manifests[sid];k=m['fits'][0]['qubits'];directory=Path(m['fits'][0]['study_root'])
        vm=psutil.virtual_memory()
        if k==4:
            settings={'selected_workers':self.safe_workers(3,250_000_000),'threads_per_worker':1,'basis':'Previously validated four-qubit three-worker implementation; live headroom rechecked.','available_ram':vm.available,'physical_cores':psutil.cpu_count(logical=False),'logical_processors':psutil.cpu_count(),'cpu_only':True,'benchmarks':[],'peak_worker_rss_bytes':250_000_000}
        else:
            candidates=[r for r in m['fits'] if r['budget']==100]
            benchmarks=[]
            first=self.batch(candidates[:1],1,600_000_000 if k==17 else 250_000_000);benchmarks.append(first)
            valid=[self.results[r['fit_id']] for r in candidates[:1] if self.results.get(r['fit_id'],{}).get('status')=='completed']
            if not valid:
                settings={'selected_workers':1,'threads_per_worker':1,'basis':'Representative fit failed; no concurrency increase. Preserve failure and continue remaining declared fits conservatively.','available_ram':vm.available,'physical_cores':psutil.cpu_count(logical=False),'logical_processors':psutil.cpu_count(),'cpu_only':True,'benchmarks':benchmarks,'peak_worker_rss_bytes':1_500_000_000 if k==17 else 500_000_000}
                self.resources[sid]=settings;atomic_json(directory/'records/execution_settings.json',settings)
                return settings
            peak=max(250_000_000,max(r['runtime']['peak_rss_bytes'] for r in valid))
            proposed=self.safe_workers(2,peak)
            selected=1
            # Compatible seed fits count toward the frozen manifest.
            if proposed>=2:
                second=self.batch(candidates[1:3],2,peak);benchmarks.append(second)
                if second['valid_fits']==second['new_fits'] and second['fits_per_second']>first.get('fits_per_second',0) and second.get('maximum_scheduler_delay_seconds',0)<1 and second.get('min_available_ram',0)>max(1_500_000_000,.12*vm.total):
                    selected=2
                peak=max(peak,second['max_worker_rss_bytes'])
            selected=self.safe_workers(selected,peak)
            settings={'selected_workers':selected,'threads_per_worker':1,'basis':'Measured compatible production-manifest fits at one and, when safe, two workers; throughput, two-times memory margin and responsiveness only. No predictive metric used.','available_ram':vm.available,'physical_cores':psutil.cpu_count(logical=False),'logical_processors':psutil.cpu_count(),'cpu_only':True,'benchmarks':benchmarks,'peak_worker_rss_bytes':peak}
        self.resources[sid]=settings;atomic_json(directory/'records/execution_settings.json',settings)
        return settings

    def prune_cache(self,directory,limit=4_000_000_000):
        cache=directory/'state_cache'
        files=sorted(cache.glob('*.npy'),key=lambda p:p.stat().st_mtime)
        total=sum(p.stat().st_size for p in files)
        for path in files:
            if total<=limit:break
            total-=path.stat().st_size;path.unlink()
            receipt=path.with_suffix('.json')
            if receipt.exists():receipt.unlink()

    def stage(self,sid,budget,settings):
        m=self.manifests[sid];rows=[r for r in m['fits'] if r['budget']==budget]
        directory=Path(rows[0]['study_root']);measurements=[]
        if all(self.results.get(r['fit_id'],{}).get('status')=='completed' for r in rows):
            summary=compare(m,self.results,[budget])[0]
            if not (directory/f'records/stage_{budget}.json').exists():atomic_json(directory/f'records/stage_{budget}.json',{'finished':now(),'summary':summary,'recovered_from_verified_receipts':True,'measurements':[]})
            print(f'SKIPPED verified complete {sid}-b{budget}',flush=True)
            return summary
        self.phase=f'{sid}-b{budget}';self.sync()
        if rows[0]['qubits'] in (4,11):
            # Eleven-qubit banks are small enough for bounded cross-configuration
            # scheduling; each worker drops its previous model's encoder references.
            measurements.append(self.batch(rows,settings['selected_workers'],settings['peak_worker_rss_bytes']))
            if rows[0]['qubits']==11:self.prune_cache(directory)
        else:
            # Bound simultaneous large banks to one preprocessing/encoder configuration.
            for config in m['configurations']:
                group=[r for r in rows if r['configuration_id']==config['configuration_id']]
                if psutil.disk_usage(str(directory)).free<4_000_000_000:self.prune_cache(directory,0)
                measurements.append(self.batch(group,settings['selected_workers'],settings['peak_worker_rss_bytes']))
                self.prune_cache(directory)
        missing=[r['fit_id'] for r in rows if r['fit_id'] not in self.results]
        if missing:raise RuntimeError('Stage has missing outcomes: '+str(missing))
        summary=compare(m,self.results,[budget])[0]
        atomic_json(directory/f'records/stage_{budget}.json',{'finished':now(),'summary':summary,'measurements':measurements})
        print(f"STAGE FINISHED {self.phase}: {summary['completed']}/{summary['expected']} valid, {summary['failed']} failed",flush=True)
        return summary

    def run(self):
        self.sync()
        for study in self.master['studies']:
            sid=study['study_id'];m=self.manifests[sid];directory=Path(study['manifest_path']).parent
            # A completed width with a saved decision and paired analysis needs no
            # model reload, repeated bootstrap or workbook export on recovery.
            width_rows=[r for r in m['fits'] if r['budget'] in self.activated[sid]]
            if all(self.results.get(r['fit_id'],{}).get('status')=='completed' for r in width_rows) and (study['qubits']==4 or sid in self.recommendations):
                print(f'SKIPPED verified complete width {study["qubits"]}',flush=True)
                continue
            self.phase=sid+'-resource-check';self.sync()
            settings=self.select_resources(sid)
            if study['qubits']==4:
                summary=self.stage(sid,200,settings)
                if not summary['complete']:raise RuntimeError('Phase A not completely verified; larger-width pilots have not been started')
            else:
                for budget in (100,200):self.stage(sid,budget,settings)
                uncertainty=paired_analysis(m,self.results,[100,200])
                decision=planning(m,self.results,[100,200],uncertainty)
                prior=self.recommendations.get(sid)
                if prior and prior['add_300']!=decision['add_300']:raise RuntimeError('Frozen automatic-stage decision changed unexpectedly')
                self.uncertainties[sid]=uncertainty;self.recommendations[sid]=decision
                atomic_json(directory/'records/budget_decision.json',decision)
                if decision['add_300']:
                    self.activated[sid]=[100,200,300];self.sync()
                    self.stage(sid,300,settings)
                    uncertainty=paired_analysis(m,self.results,[100,200,300])
                    final=planning(m,self.results,[100,200,300],uncertainty)
                    # Preserve the original 200-based activation decision on resume.
                    final['add_300']=True
                    final['activation_decision_at_200']=decision
                    self.uncertainties[sid]=uncertainty;self.recommendations[sid]=final
                    atomic_json(directory/'records/budget_decision.json',final)
            self.sync()
            self.close_pool()
        self.phase='final-consolidation';self.sync()
        final_report(self.payload,self.resources,self.status)
        self.close_pool()
        readback=self.writer.finish()
        if not readback or not readback['allPassed']:raise RuntimeError('Workbook reconciliation incomplete; verified fit receipts retained for resume')
        self.verify_final()

    def verify_final(self):
        expected=[];failed=[]
        for sid,m in self.manifests.items():
            expected.extend(r for r in m['fits'] if r['budget'] in self.activated[sid])
            directory=Path(m['fits'][0]['study_root'])
            for config in m['configurations']:
                if config['qubits']!=4:assert config['feature_map_index']==(config['p']+config['r'])%4
                assert sha(directory/config['circuit_path'])==config['circuit_sha256']
            split=json.loads((directory/'split.json').read_text())
            for p in range(4):
                audit=json.loads((directory/f'preprocessing/p{p}.audit.json').read_text())
                assert audit['fit_rows']==m['fits'][0]['training_rows']
                assert audit['scaler_fit_index_sha256']==audit['rfe_fit_index_sha256']==split['training_index_sha256']
                assert audit['rfe_fit_labels_sha256']==split['training_label_sha256']
                assert not audit['validation_rows_in_fit'] and not audit['validation_labels_in_fit']
        for row in expected:
            result=validate_receipt(row)
            if result is None:raise RuntimeError('Missing completion receipt '+row['fit_id'])
            if result['status']!='completed':failed.append(row['fit_id'])
        source=json.loads((ROOT/'records/source_lock.json').read_text())
        assert all(sha(ROOT/p)==digest for p,digest in source['files'].items())
        inputs=json.loads((ROOT/'records/input_lock.json').read_text())
        assert all(sha(ROOT/p)==digest for p,digest in inputs['files'].items())
        readback=json.loads((ROOT/'records/workbook_readback.json').read_text());assert readback['allPassed']
        assert len(self.payload['Four_Qubit_200'])==192
        assert len(self.payload['Pilot_Fit_Results'])==len(expected)-192
        verification={'verified_at':now(),'all_activated_fits_verified':not failed,'activated_fits':len(expected),'completed':len(expected)-len(failed),'failed':len(failed),'failed_ids':failed,'initial_mandatory_fits':480,'maximum_allowed_fits':624,'no_larger_width_full_grids':True,'full_grid_training_rows':455,'full_grid_test_rows':114,'pilot_training_rows':364,'pilot_validation_rows':91,'preprocessing_fit_access_verified':True,'same_explicit_initializations_across_budgets':True,'workbook_matches_authoritative_tables':True,'immutable_input_and_source_hashes_match':True,'workbook_sha256':sha(ROOT/'WDBC_VQC_Execution_and_Budget_Master.xlsx'),'provisional_width_recommendations':self.recommendations}
        atomic_json(ROOT/'records/final_verification.json',verification)
        if failed:raise RuntimeError('Sequence has unresolved failed fits; see final_verification.json. No completion marker written.')
        self.status.update({'execution_state':'complete','controller_active':False,'phase':'complete','verified_at':now()})
        atomic_json(ROOT/'status.json',self.status)
        final_report(self.payload,self.resources,self.status)
        atomic_json(ROOT/'COMPLETED.json',{'completed_at':now(),'verified_fits':len(expected),'verification_sha256':sha(ROOT/'records/final_verification.json')})
        print(f'COMPLETE AND VERIFIED: {len(expected)} fits',flush=True)

def main():
    lock=ROOT/'controller.lock'
    if lock.exists():
        previous=json.loads(lock.read_text())
        if psutil.pid_exists(previous['pid']):raise RuntimeError(f"Controller already active PID {previous['pid']}")
        lock.unlink()
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(descriptor,'w') as handle:json.dump({'pid':os.getpid(),'started':now()},handle)
    try:
        files=['vqc_engine.py','prepare_pilot.py','prepare_sequence.py','verify_engine.py','fit_worker.py','sequence_analysis.py','sequence_controller.py','workbook_support/workbook_writer.mjs']
        sourcefile=ROOT/'records/source_lock.json'
        if sourcefile.exists():
            source=json.loads(sourcefile.read_text());assert all(sha(ROOT/p)==digest for p,digest in source['files'].items())
        else:atomic_json(sourcefile,{'frozen_at':now(),'files':{p:sha(ROOT/p) for p in files}})
        Controller().run()
    except Exception as error:
        failure={'failed_at':now(),'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),'receipts_preserved':True}
        event('controller_errors.jsonl',failure);atomic_json(ROOT/'FAILED.json',failure)
        status=json.loads((ROOT/'status.json').read_text()) if (ROOT/'status.json').exists() else {}
        status.update({'execution_state':'failed','controller_active':False,**failure});atomic_json(ROOT/'status.json',status)
        raise
    finally:
        if lock.exists() and json.loads(lock.read_text())['pid']==os.getpid():lock.unlink()

if __name__=='__main__':main()
