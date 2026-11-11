"""Explicit opt-in job-mode runner. Never invoked by prepare, imports or local simulation."""
from pathlib import Path
import time,math
import numpy as np
from qiskit import qpy
from .io import read,atomic,identity,file_hash,runner_lock,now
from .backend import connect,remaining,job_allowance
from .ledger import Ledger,BudgetError
from .data import verify,preprocess
from .circuits import bindings
from .results import extract_pub_results,matrices,score

def cache_identity(directory,plan):
    directory=Path(directory);verify(directory);preprocess(directory)
    if plan['identity']!=identity({k:v for k,v in plan.items() if k!='identity'}):raise BudgetError('Backend plan integrity failure')
    if plan['inputs']!=identity(read(directory/'input_lock.json')) or plan['preprocessing']!=file_hash(directory/'preprocessed.npz'):
        raise BudgetError('Cache data/preprocessing identity mismatch')
    if plan['isa_sha256']!=file_hash(directory/'isa_template.qpy') or plan['logical_sha256']!=file_hash(directory/'logical_template.qpy'):
        raise BudgetError('Circuit cache integrity failure')
    import qiskit,qiskit_ibm_runtime
    return identity({'plan':plan['identity'],'protocol':read(directory/'protocol.json'),'qiskit':qiskit.__version__,'runtime':qiskit_ibm_runtime.__version__})

def collect(directory,ledger,key,service,p,pair_lookup,wait=True,usage_wait_seconds=60):
    entry=ledger.data['jobs'][key]
    if not entry['job_id']:
        found=service.jobs(job_tags=[entry['tag']],limit=10)
        if len(found)!=1:raise BudgetError('Ambiguous submission recovery; no replacement allowed for '+key)
        ledger.submitted(key,found[0].job_id());entry=ledger.data['jobs'][key]
    job=service.job(entry['job_id'])
    if not wait and not job.in_final_state():return False
    started=time.perf_counter();records=None;failure=None
    try:
        result=job.result();pairs=[pair_lookup[v] for v in entry['pairs']]
        records=extract_pub_results(result,pairs,p['measurement']['shots'],p['encoder']['qubits'])
    except Exception as error:
        failure=type(error).__name__ # Do not copy provider exceptions that might contain credentials.
    if not job.in_final_state():raise BudgetError('Job is not final; preserve reservation and reconcile later')
    out=Path(directory)/'hardware/jobs';out.mkdir(parents=True,exist_ok=True)
    if records is not None:
        for r in records:r.update(job_id=entry['job_id'],job_key=key)
        atomic(out/(key+'.counts.json'),{'identity':ledger.data['identity'],'records':records})
    # IBM may return a transient zero immediately after valid hardware results.
    # Preserve counts first, then allow billing metadata to settle. A completed
    # measured circuit cannot be treated as free while that value is unresolved.
    usage_started=time.perf_counter();reported=None;actual=None
    while True:
        try:reported=job.usage()
        except Exception:reported=None
        if isinstance(reported,(int,float)) and math.isfinite(reported) and reported>=0:
            if records is None or reported>0:actual=reported;break
        if time.perf_counter()-usage_started>=usage_wait_seconds:break
        time.sleep(min(5,max(0,usage_wait_seconds-(time.perf_counter()-usage_started))))
    try:
        metadata=job.metrics();usage_metadata=metadata.get('usage');timestamps=metadata.get('timestamps')
    except Exception:usage_metadata=None;timestamps=None
    atomic(out/(key+'.json'),{'job_id':entry['job_id'],'job_key':key,'actual_service_qpu_seconds':actual,
        'last_reported_service_qpu_seconds':reported,'usage_settlement_wait_seconds':time.perf_counter()-usage_started,
        'service_usage_metadata':usage_metadata,'service_timestamps':timestamps,
        'result_and_usage_wait_wall_seconds':time.perf_counter()-started,'failure_type':failure,'status':str(job.status()),'completed':now()})
    ledger.finish(key,'VALID' if records is not None else 'FAILED_OR_CANCELLED',actual)
    if records is None:raise BudgetError('Hardware result failed validation; retain billed attempt, no automatic paid retry: '+key)
    if actual is None:raise BudgetError('Service usage unknown; reservation retained and further submissions blocked')
    return True

def load_records(directory,ledger):
    records={}
    for key,entry in ledger.data['jobs'].items():
        if entry['state']!='VALID':continue
        path=Path(directory)/'hardware/jobs'/(key+'.counts.json');saved=read(path)
        if saved['identity']!=ledger.data['identity']:raise BudgetError('Saved raw-count identity mismatch')
        if [r['pair']['id'] for r in saved['records']]!=entry['pairs']:raise BudgetError('Saved pair order mismatch')
        from .results import counts_record
        for r in saved['records']:
            verified=counts_record(r['raw_counts'],r['requested_shots'],len(next(iter(r['raw_counts']))))
            if any(verified[k]!=r[k] for k in verified):raise BudgetError('Raw-count integrity failure')
            if r['pair']['id'] in records:raise BudgetError('Duplicate hardware pair')
            records[r['pair']['id']]=r
    return records

def run(directory,stage,allow_submission=False):
    if not allow_submission:raise BudgetError('Hardware disabled. Explicit --allow-qpu-submission is required; never use during preparation.')
    directory=Path(directory)
    if (directory/'HARDWARE_EXPERIMENTATION_STOPPED.json').exists():raise BudgetError('The authorized robustness experiment is complete; further hardware experimentation is stopped')
    with runner_lock(directory):
        p=read(directory/'protocol.json');plan=read(directory/'backend_plan.json');cid=cache_identity(directory,plan)
        if not plan['locally_estimated_feasible']:raise BudgetError('Local budget estimate is not feasible')
        service,instance_hash=connect()
        if instance_hash!=plan['instance_hash']:raise BudgetError('Instance changed; no fallback allowed')
        ledger=Ledger(directory/'hardware/ledger.json',cid,p['budget'])
        pairs=read(directory/'pairs.json');lookup={v['id']:v for v in pairs}
        for key,j in list(ledger.data['jobs'].items()):
            if j['state']!='VALID' or j['actual_seconds'] is None:
                collect(directory,ledger,key,service,p,lookup)
        records=load_records(directory,ledger)
        if stage=='reconcile':return {'complete_pairs':len(records),'charged_seconds':ledger.charged()}
        with (directory/'isa_template.qpy').open('rb') as f:circuit=qpy.load(f)[0]
        data=np.load(directory/'preprocessed.npz');backend=service.backend(plan['backend'])
        for i,chunk in enumerate(plan['chunks']):
            if stage=='pilot' and i>0:break
            key=f'job-{i:03d}'
            if all(v in records for v in chunk):continue
            if any(v in records for v in chunk):raise BudgetError('Partial chunk collision; manual reconciliation required')
            ratio=1.
            if i>0:
                pilot=ledger.data['jobs'].get('job-000')
                if not pilot or pilot['state']!='VALID' or pilot['actual_seconds'] is None:raise BudgetError('Verified pilot actual usage is required')
                ratio=max(1.,pilot['actual_seconds']/plan['base_estimates_seconds'][0])
            allowance=job_allowance(plan['base_estimates_seconds'][i],p,ratio)
            # Recheck the entire outstanding plan before every submission, not just the next chunk.
            outstanding=sum(job_allowance(plan['base_estimates_seconds'][j],p,ratio) for j in range(i,len(plan['chunks'])))
            fresh_remaining=remaining(service)
            if ledger.cumulative_charged()+outstanding>p['budget']['planned_cap_seconds'] or fresh_remaining-outstanding<p['budget']['safety_reserve_seconds']:
                raise BudgetError('Pilot-informed whole-plan forecast exceeds available safe budget')
            entry=ledger.reserve(key,chunk,allowance,fresh_remaining)
            entry.update(base_estimate_seconds=plan['base_estimates_seconds'][i],pilot_usage_ratio=ratio,
                remaining_account_seconds_before_submission=fresh_remaining,outstanding_plan_allowances_seconds=outstanding,
                prior_experiment_usage_seconds=ledger.prior_charged())
            ledger.save()
            from qiskit_ibm_runtime import SamplerV2
            sampler=SamplerV2(mode=backend,options={'default_shots':p['measurement']['shots'],'max_execution_time':allowance,
                'environment':{'job_tags':[entry['tag']]},'dynamical_decoupling':{'enable':False},
                'twirling':{'enable_gates':False,'enable_measure':False}})
            values=bindings(circuit,[lookup[v] for v in chunk],data['X_train'],data['X_test'])
            submitted=time.perf_counter()
            job=sampler.run([(circuit,values,p['measurement']['shots'])])
            ledger.submitted(key,job.job_id()) # Must persist ID BEFORE waiting or result access.
            submission_seconds=time.perf_counter()-submitted
            ledger.data['jobs'][key]['submission_wall_seconds']=submission_seconds;ledger.save()
            atomic(directory/'hardware/status.json',{'state':'JOB_SUBMITTED','job_key':key,'job_id':job.job_id(),'submission_wall_seconds':submission_seconds})
            collect(directory,ledger,key,service,p,lookup);records=load_records(directory,ledger)
        if len(records)==len(pairs):
            Ktr,Kte=matrices(records,pairs,p['cohort']['train'],p['cohort']['test'])
            out=directory/'hardware';score(out,p,Ktr,Kte,data['y_train'],data['y_test'],data['test_ids'])
            atomic(out/'COMPLETED.json',{'identity':cid,'pairs':len(records),'actual_qpu_seconds':ledger.charged(),
                'files':{f.name:file_hash(f) for f in out.iterdir() if f.is_file() and f.name not in ['COMPLETED.json','status.json'] and not f.name.endswith('.log')}})
        atomic(directory/'hardware/status.json',{'state':'COMPLETE' if len(records)==len(pairs) else 'PILOT_COMPLETE','pairs':len(records),'actual_qpu_seconds':ledger.charged()})
        return read(directory/'hardware/status.json')
