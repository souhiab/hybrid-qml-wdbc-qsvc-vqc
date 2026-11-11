from __future__ import annotations
import ast
import datetime
import hashlib
import importlib.util
import json
import logging
import os
import time
import traceback
from pathlib import Path
import numpy as np
import psutil
from qiskit.quantum_info import Statevector
from qiskit.primitives import StatevectorSampler
from threadpoolctl import threadpool_limits
import vqc_engine as engine
from prepare_pilot import circuits

ROOT=Path(__file__).resolve().parent
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):engine.atomic_json(path,value)

def numerical_check():
    oldpath=ROOT/'records/cache_lock_fix_v3/vqc_engine.py'
    oldast=ast.parse(oldpath.read_text());newast=ast.parse((ROOT/'vqc_engine.py').read_text())
    def elements(tree):return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
    a,b=elements(oldast),elements(newast)
    unchanged=['sampled_binary_probabilities','CachedStatevectorQNN','RecordedCOBYLA','clinical']
    assert all(a[name]==b[name] for name in unchanged)
    if a['evolve_ansatz']!=b['evolve_ansatz']:
        optimization=read(ROOT/'records/scheduling_numerical_validation.json')
        assert optimization['passed'] and optimization['probabilities_bitwise_identical']
        assert optimization['engine_sha256']==engine.sha(ROOT/'vqc_engine.py')
    spec=importlib.util.spec_from_file_location('prior_cache_engine',oldpath)
    prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
    qa=ROOT/'records/cache_assurance_numerics';qa.mkdir(exist_ok=True)
    oldroot,newroot=qa/'before',qa/'after'
    oldroot.mkdir(exist_ok=True);newroot.mkdir(exist_ok=True)
    prior.ROOT=oldroot;engine.ROOT=newroot
    logging.getLogger('qiskit_machine_learning').setLevel(logging.ERROR)
    checks=[]
    with threadpool_limits(limits=1):
        for k in (7,11,17):
            directory=ROOT/f'studies/internal-budget-pilot-k{k:02d}'
            with np.load(directory/'preprocessing/p0.npz',allow_pickle=False) as data:inputs=np.array(data['X_train'][:2],copy=True)
            for m in range(4):
                row={'qubits':k,'encoder':'Z' if m==0 else 'ZZ','topology':['','linear','full','circular'][m],'feature_map_reps':2,'ansatz_reps':2}
                fmap,ansatz=circuits(row)
                weights=np.random.default_rng(101).uniform(0,1,ansatz.num_parameters)
                for seed in (1101,1211,1307):
                    old=prior.CachedStatevectorQNN(fmap,ansatz,StatevectorSampler(default_shots=1024,seed=seed),seed)
                    new=engine.CachedStatevectorQNN(fmap,ansatz,StatevectorSampler(default_shots=1024,seed=seed),seed)
                    np.testing.assert_array_equal(old.forward(inputs,weights),new.forward(inputs,weights))
                    checks.append({'qubits':k,'feature_map_index':m,'sampler_seed':seed,'probabilities_identical':True})
    result={'passed':True,'unchanged_computational_ast':unchanged,'finite_shot_comparisons':len(checks),'checks':checks,'training_fits_executed':0,'optimizer_and_loss_unchanged':True,'precision_shots_initialization_readout_unchanged':True,'cache_publication_regression':read(ROOT/'records/cache_lock_validation.json')}
    save(ROOT/'records/cache_fix_numerical_confirmation.json',result)
    return result

def catalog():
    expected={};studies=[]
    master=read(ROOT/'manifest.json')
    for study in master['studies']:
        directory=Path(study['manifest_path']).parent
        manifest=read(study['manifest_path']);studies.append((directory,manifest))
        split=read(directory/'split.json')
        if study['qubits']==4:continue
        for p in range(4):
            cache=directory/f'preprocessing/p{p}.npz';audit=read(directory/f'preprocessing/p{p}.audit.json')
            assert engine.sha(cache)==audit['cache_sha256']
            assert audit['fit_rows']==364 and not audit['validation_rows_in_fit'] and not audit['validation_labels_in_fit']
            with np.load(cache,allow_pickle=False) as data:
                for cohort,key,indices in [('train','X_train','train_indices'),('validation','X_validation','validation_indices')]:
                    inputs=np.ascontiguousarray(data[key]);ids=data[indices]
                    expected_ids=split['pilot_training_indices' if cohort=='train' else 'validation_indices']
                    np.testing.assert_array_equal(ids,expected_ids)
                    assert engine.array_sha(inputs)==audit['X_train_sha256' if cohort=='train' else 'X_validation_sha256']
                    assert inputs.shape==(364 if cohort=='train' else 91,study['qubits'])
                    for config in manifest['configurations']:
                        if config['p']!=p:continue
                        fmap,_=circuits(config)
                        semantic=f'{fmap.num_qubits}-{fmap.name}-{fmap.reps}-{fmap.entanglement}-{engine.array_sha(inputs)}'
                        token=hashlib.sha256(semantic.encode()).hexdigest()
                        context={'configuration_id':config['configuration_id'],'qubits':study['qubits'],'cohort':cohort,'p':p,'scaler':config['scaler'],'rfe_estimator':config['rfe_estimator'],'feature_map_reps':config['feature_map_reps'],'encoder':config['encoder'],'topology':config['topology'],'input_sha256':engine.array_sha(inputs),'ordered_indices_sha256':engine.array_sha(ids),'selected_feature_indices':data['selected_indices'].tolist(),'preprocessing_cache_sha256':audit['cache_sha256'],'shape':[len(inputs),1<<study['qubits']],'dtype':'complex128','semantic':semantic}
                        entry=expected.setdefault((str(directory),token),{'contexts':[],'sample_inputs':inputs[[0,-1]].copy(),'sample_positions':[0,len(inputs)-1],'fmap':fmap})
                        entry['contexts'].append(context)
    return expected,studies

def check_cache(directory,receipt,entry):
    info=read(receipt);path=receipt.with_suffix('.npy');context=entry['contexts'][0]
    assert info['semantic']==context['semantic']
    assert info['rows']==context['shape'][0] and info['qubits']==context['qubits'] and info['precision']=='complex128'
    assert engine.sha(path)==info['sha256']
    bank=np.load(path,mmap_mode='r',allow_pickle=False)
    assert bank.shape==tuple(context['shape']) and bank.dtype==np.dtype('complex128')
    assert bank.offset+bank.nbytes==path.stat().st_size
    for position,inputs in zip(entry['sample_positions'],entry['sample_inputs']):
        reference=Statevector.from_instruction(entry['fmap'].assign_parameters(inputs)).data
        np.testing.assert_allclose(bank[position],reference,atol=2e-12,rtol=0)
    # Check completeness/finiteness/normalization throughout the stored bank in
    # bounded chunks, without altering or retaining the production mapping.
    for start in range(0,len(bank),8):
        chunk=bank[start:start+8]
        assert np.isfinite(chunk).all()
        np.testing.assert_allclose(np.sum(np.abs(chunk)**2,axis=1),1,atol=2e-12,rtol=0)
    del bank
    return {'file':str(path),'sha256':info['sha256'],'verified_at':now(),'identity_contexts':entry['contexts'],'file_complete':True,'all_rows_finite_and_normalized':True,'reference_statevector_samples':2,'encoder_only_cache':'Ansatz repetitions, optimizer and seed may share a bank only because encoding inputs and feature-map semantics match exactly.'}

def attempt_accounting(studies,boundary):
    failed=[];interrupted=[];unresolved=[];completed=0
    for directory,manifest in studies:
        bykey={row['artifact_key']:row for row in manifest['fits']}
        for key,row in bykey.items():
            parent=directory/'artifacts'/key
            receipt=parent/'receipt.json'
            final=read(receipt) if receipt.exists() else None
            if final and final['status']=='completed':completed+=1
            if final and final['status']=='failed':unresolved.append(row['fit_id'])
            for file in parent.glob('attempt_*/attempt.json'):
                attempt=read(file)
                if attempt['status']=='interrupted':interrupted.append({'fit_id':row['fit_id'],'attempt_id':attempt['attempt_id']})
                if attempt['status']!='failed':continue
                error=attempt.get('error','')
                failed.append({'fit_id':row['fit_id'],'attempt_id':attempt['attempt_id'],'started':attempt.get('started'),'error':error,'technical_failure':attempt.get('technical_failure',False),'cache_lock_error':('WinError 5' in error and 'state_cache' in error),'after_lock_fix':bool(attempt.get('started') and attempt['started']>=boundary),'recovered_by_valid_completed_result':bool(final and final['status']=='completed')})
    return {'completed_fits':completed,'failed_attempts':len(failed),'recovered_failed_attempts':sum(a['recovered_by_valid_completed_result'] for a in failed),'unresolved_final_failures':len(unresolved),'unresolved_fit_ids':unresolved,'interrupted_attempts':interrupted,'cache_lock_failed_attempts':sum(a['cache_lock_error'] for a in failed),'post_fix_cache_lock_failed_attempts':sum(a['cache_lock_error'] and a['after_lock_fix'] for a in failed),'failed_attempt_details':failed}

def main():
    lock=ROOT/'records/cache_assurance.lock'
    if lock.exists() and psutil.pid_exists(read(lock)['pid']):raise RuntimeError('Assurance monitor already running')
    save(lock,{'pid':os.getpid(),'started':now()})
    try:
        numerical=numerical_check()
        expected,studies=catalog();verified={};errors=[]
        baseline_path=ROOT/'records/preserved_completed_baseline.json'
        if baseline_path.exists():baseline=read(baseline_path)
        else:
            baseline={str(file):engine.sha(file) for directory,_ in studies for file in (directory/'artifacts').glob('*/receipt.json') if read(file)['status']=='completed'}
            assert len(baseline)>=344;save(baseline_path,baseline)
        previous_monitor=ROOT/'records/cache_assurance_started.json'
        boundary=read(previous_monitor)['post_lock_fix_attempt_boundary'] if previous_monitor.exists() else read(ROOT/'records/startup_inventory.json')['time']
        save(ROOT/'records/cache_assurance_started.json',{'pid':os.getpid(),'started':now(),'preserved_completed_results':len(baseline),'post_lock_fix_attempt_boundary':boundary,'training_fits_executed':0,'numerical_confirmation_passed':numerical['passed']})
        while True:
            for directory,_ in studies:
                for receipt in (directory/'state_cache').glob('*.json'):
                    try:
                        info=read(receipt);identity=(str(directory),receipt.stem)
                        key=str(receipt)
                        if key in verified and verified[key]['sha256']==info['sha256']:continue
                        if identity not in expected:raise AssertionError('Cache has no matching configuration/preprocessing/cohort identity: '+str(receipt))
                        with threadpool_limits(limits=1):verified[key]=check_cache(directory,receipt,expected[identity])
                        save(ROOT/'records/cache_integrity_verified.json',list(verified.values()))
                    except FileNotFoundError:continue # A completed group may legitimately prune its cold bank.
                    except Exception as error:
                        entry={'time':now(),'receipt':str(receipt),'error':str(error),'type':type(error).__name__}
                        if not any(e['receipt']==entry['receipt'] and e['error']==entry['error'] for e in errors):errors.append(entry)
            save(ROOT/'records/cache_assurance_status.json',{'updated':now(),'state':'monitoring','pid':os.getpid(),'numerical_confirmation_passed':True,'verified_cache_files':len(verified),'cache_verification_errors':errors,'baseline_completed_fits_preserved':len(baseline)})
            if (ROOT/'COMPLETED.json').exists() or (ROOT/'FAILED.json').exists():break
            time.sleep(20)
        preserved=all(Path(file).exists() and engine.sha(file)==digest for file,digest in baseline.items())
        accounting=attempt_accounting(studies,boundary)
        preference=read(ROOT/'records/budget_review_preferences.json')
        final={'verified_at':now(),'controller_completed':(ROOT/'COMPLETED.json').exists(),'baseline_completed_fits_preserved':preserved,'preserved_baseline_count':len(baseline),'numerical_training_procedure_unchanged':numerical['passed'],'cache_files_verified':len(verified),'cache_verification_errors':errors,'attempt_accounting':accounting,'seven_qubit_preference':preference,'trust_checks_passed':preserved and numerical['passed'] and not errors and accounting['unresolved_final_failures']==0}
        optimization=ROOT/'records/scheduling_numerical_validation.json'
        if optimization.exists():final['scheduling_optimization_numerical_validation']=read(optimization)
        save(ROOT/'records/cache_and_attempt_final_verification.json',final)
        lines=['## Cache assurance and final budget review','',f"Preserved completed baseline: {len(baseline)}; unchanged: {preserved}.",f"Cache identity/integrity checks: {len(verified)} files; verification errors: {len(errors)}.",f"Failed attempts: {accounting['failed_attempts']}; recovered: {accounting['recovered_failed_attempts']}; unresolved final failures: {accounting['unresolved_final_failures']}.",f"Cache locking failures after the fix: {accounting['post_fix_cache_lock_failed_attempts']}.",'The locking patch preserves the numerical engine, objective, optimizer, sampler, precision, initialization and scoring; 36 pre/post finite-shot comparisons were identical.',
            'Seven qubits: provisional preference is cap 300 for additional optimization allowance. This misses the predefined 90% completion target (41/48 converged, 7/48 budget-exhausted) and therefore requires an explicit documented exception. Retain all seven valid budget-limited outcomes. Do not claim all fits converged. The preference is not adopted as a rule-satisfying automatic recommendation.',
            'Finalize any seven-qubit exception alongside the remaining 11/17-qubit pilot results. No larger-width full grid is authorized by this record.','',json.dumps(preference,indent=2)]
        text='\n'.join(lines)+'\n';(ROOT/'BUDGET_REVIEW_AND_CACHE_AUDIT.md').write_text(text,encoding='utf-8')
        if (ROOT/'COMPLETED.json').exists():
            with (ROOT/'FINAL_REPORT.md').open('a',encoding='utf-8') as handle:handle.write('\n'+text)
            if final['trust_checks_passed']:
                save(ROOT/'CACHE_VERIFIED_COMPLETED.json',{'verified_at':now(),'verification_file':'records/cache_and_attempt_final_verification.json','sha256':engine.sha(ROOT/'records/cache_and_attempt_final_verification.json'),'locking_error_recurred':accounting['post_fix_cache_lock_failed_attempts']>0})
        save(ROOT/'records/cache_assurance_status.json',{'updated':now(),'state':'complete' if final['controller_completed'] else 'controller_failed','verification_file':'records/cache_and_attempt_final_verification.json','trust_checks_passed':final['trust_checks_passed'],'post_fix_cache_lock_failed_attempts':accounting['post_fix_cache_lock_failed_attempts']})
    except Exception as error:
        save(ROOT/'records/cache_assurance_status.json',{'state':'assurance_failed','time':now(),'error':str(error),'traceback':traceback.format_exc()})
        raise
    finally:
        if lock.exists():lock.unlink()

if __name__=='__main__':main()
