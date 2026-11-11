from __future__ import annotations
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ.setdefault(name,'1')
import concurrent.futures
import copy
import json
import multiprocessing
import queue
import subprocess
import threading
import time
from pathlib import Path
import psutil
from common import *
from execution import workload,validate_receipt,evaluate,receipt_for
from analysis import consolidate

def event(file,value):
    with (ROOT/'records'/file).open('a',encoding='utf-8') as stream:stream.write(json.dumps({'time':now(),**value},default=convert)+'\n')

class ExcelWriter:
    def __init__(self):self.process=None
    def command(self,command):
        for retry in range(3):
            try:
                if self.process is None or self.process.poll() is not None:
                    self.process=subprocess.Popen(['node',str(ROOT/'workbook_support/workbook_writer.mjs')],cwd=ROOT/'workbook_support',stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=(ROOT/'records/workbook_stderr.log').open('a',encoding='utf-8'),text=True,encoding='utf-8')
                self.process.stdin.write(command+'\n');self.process.stdin.flush()
                while True:
                    line=self.process.stdout.readline()
                    if not line:raise OSError('Workbook writer ended without acknowledgement')
                    if line.startswith('@@'):
                        answer=json.loads(line[2:])
                        if not answer['ok']:raise OSError(answer['error'])
                        return answer['result']
            except Exception as error:
                event('workbook_errors.jsonl',{'error':str(error),'retry':retry+1,'receipts_preserved':True})
                if self.process and self.process.poll() is None:self.process.terminate();self.process.wait()
                self.process=None
        return None
class AsyncWriter:
    def __init__(self):
        self.tasks=queue.Queue(maxsize=4);self.writer=ExcelWriter();self.last=None
        self.thread=threading.Thread(target=self.work,daemon=True);self.thread.start()
    def work(self):
        while True:
            item=self.tasks.get()
            try:
                if item is None:return
                atomic(ROOT/'workbook_support/excel_snapshot.json',item)
                self.writer.command('sync')
            finally:self.tasks.task_done()
    def submit(self,payload):
        if payload!=self.last:self.tasks.put(copy.deepcopy(payload));self.last=copy.deepcopy(payload)
    def finish(self):
        self.tasks.join();result=self.writer.command('final')
        if not result or not result['allPassed']:raise OSError('Excel final verification incomplete; receipts remain authoritative')
        self.tasks.put(None);self.thread.join()

def active_vqc():
    lock=ROOT.parent/'wdbc_vqc_execution_20261108/controller.lock'
    if not lock.exists():return False
    pid=read(lock)['pid']
    return psutil.pid_exists(pid) and 'sequence_controller.py' in ' '.join(psutil.Process(pid).cmdline())
def resources():
    vm=psutil.virtual_memory()
    return {'time':now(),'physical_cores':psutil.cpu_count(logical=False),'logical_processors':psutil.cpu_count(),'ram_total':vm.total,'ram_available':vm.available,'disk_free':psutil.disk_usage(str(ROOT)).free,'cpu_percent':psutil.cpu_percent(interval=1),'active_separate_vqc':active_vqc(),'recommended_initial_workers':1,'recommended_threads':2,'k17_train_bank_upper_bytes':456*(1<<17)*16}
def prune_cache():
    limit=read(ROOT/'protocol.json')['resources']['cache_limit_bytes']
    files=list((ROOT/'cache').glob('*'));total=sum(f.stat().st_size for f in files if f.is_file())
    # Cache entries are reconstructible, never authoritative evaluation artifacts.
    for metadata in sorted((ROOT/'cache').glob('*.json'),key=lambda f:f.stat().st_mtime):
        if total<=limit:break
        prefix=metadata.stem
        for file in (ROOT/'cache').glob(prefix+'.*'):
            assert file.resolve().parent==(ROOT/'cache').resolve()
            total-=file.stat().st_size;file.unlink()

def run(kind='all',candidate='all',workers=1):
    verify_frozen()
    if active_vqc():raise RuntimeError('The separate VQC pilot is active. Training is deferred to avoid oversubscribing it; this command launched no WDBC fit.')
    info=resources();workers=int(workers)
    assert 1<=workers<=2
    if info['ram_available']<2_147_483_648+workers*2_000_000_000:raise MemoryError('Insufficient live RAM for guarded execution')
    lock=ROOT/'controller.lock'
    if lock.exists():
        old=read(lock)
        if psutil.pid_exists(old['pid']):raise RuntimeError('A shared-fold controller already exists')
        lock.unlink()
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(descriptor,'w') as stream:json.dump({'pid':os.getpid(),'started':now()},stream)
    try:
        items=workload();selected=[i for i in items if (kind=='all' or i['kind']==kind) and (candidate=='all' or i['candidate_id']==candidate)]
        if not selected:raise ValueError('No matching candidate/scope')
        # Exactly one startup checkpoint inventory; valid fits are never repeated.
        outcomes={};attempts={}
        for item in items:
            folder=ROOT/'runs'/item['evaluation_id'];history=[]
            for file in sorted(folder.glob('attempt_*/attempt.json')):
                a=read(file)
                if a['status']=='running':a.update({'status':'interrupted','ended':now(),'error':'Restart incomplete evaluation under a new attempt ID'});atomic(file,a)
                history.append(a)
            attempts[item['evaluation_id']]=history
            result=validate_receipt(item)
            if result:outcomes[item['evaluation_id']]=result
        atomic(ROOT/'records/execution_resources.json',info)
        writer=AsyncWriter()
        # The preparation lock includes this workbook builder. Runtime changes
        # require an explicit new preparation revision, not silent defaults.
        for round_id in range(2):
            pending=[]
            for item in selected:
                result=outcomes.get(item['evaluation_id'])
                failures=sum(a['status']=='failed' and a.get('technical_failure') for a in attempts[item['evaluation_id']])
                if result is None or (result['status']=='failed' and result.get('technical_failure') and failures<2):pending.append(item)
            if not pending:break
            # One persistent bounded pool, not one fresh interpreter per fit.
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures={pool.submit(evaluate,item,len(attempts[item['evaluation_id']])+1,2 if workers==1 else 1):item for item in pending}
                for future in concurrent.futures.as_completed(futures):
                    item=futures[future]
                    try:result=future.result()
                    except Exception:
                        result=validate_receipt(item)
                        if result is None or result['status']=='failed':raise
                    outcomes[item['evaluation_id']]=result
                    attempt=ROOT/'runs'/item['evaluation_id']/result['attempt_id']/'attempt.json'
                    attempts[item['evaluation_id']].append(read(attempt))
                    # Receipts already verified by worker before this point.
                    payload=consolidate(verify=False,uncertainty=False,outcomes_cache=outcomes);writer.submit(payload)
                    complete=sum(r['status'] in ('completed','completed_reused') for r in outcomes.values())
                    atomic(ROOT/'status.json',{'state':'RUNNING','completed_outer_evaluations':complete,'expected_outer_evaluations':255,'selected_scope':kind,'selected_candidate':candidate,'WDBC_training_launched':True,'last_evaluation':item['evaluation_id']})
                    print('SAVED '+item['evaluation_id']+' '+result['status'],flush=True)
            prune_cache()
        all_complete=sum(r['status'] in ('completed','completed_reused') for r in outcomes.values())==255
        payload=consolidate(verify=True,uncertainty=all_complete);writer.submit(payload);writer.finish()
        valid=[validate_receipt(item) for item in selected]
        failed=[item['evaluation_id'] for item,r in zip(selected,valid) if r is None or r['status']=='failed']
        atomic(ROOT/'records/last_scope_verification.json',{'kind':kind,'candidate':candidate,'expected':len(selected),'valid':len(selected)-len(failed),'unresolved':failed,'workbook_matches_tables':True})
        atomic(ROOT/'status.json',{'state':'SCOPE_COMPLETE' if not failed else 'EXECUTION_INCOMPLETE','selected_scope':kind,'selected_candidate':candidate,'complete_total':sum(row['status'] in ('completed','completed_reused') for row in payload['Individual_Results']),'expected_total':255,'unresolved':failed,'WDBC_training_launched':True})
        if all_complete and not failed:
            final={'verified_at':now(),'outer_evaluations':255,'predictions_and_metrics_verified':True,'shared_fold_coverage_verified':True,'paired_patient_uncertainty_saved':True,'workbook_readback_all_passed':True,'workbook_sha256':sha(ROOT/'WDBC_QSVC_Classical_Shared_Folds_Master.xlsx'),'fold_manifest_sha256':sha(ROOT/'folds/outer_inner_assignments.json')}
            atomic(ROOT/'records/final_verification.json',final)
            atomic(ROOT/'COMPLETED.json',{'verified_at':now(),'verification_sha256':sha(ROOT/'records/final_verification.json')})
            atomic(ROOT/'status.json',{'state':'COMPLETE','completed_outer_evaluations':255,'WDBC_training_launched':True})
    except Exception as error:
        event('controller_errors.jsonl',{'error_type':type(error).__name__,'error':str(error)})
        raise
    finally:
        if lock.exists() and read(lock)['pid']==os.getpid():lock.unlink()
