from __future__ import annotations
import datetime
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
import joblib
from qiskit import QuantumCircuit,qpy
from qiskit.circuit.library import ZFeatureMap,ZZFeatureMap,PauliFeatureMap,RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits
from vqc_engine import ROOT,VERSION,atomic_json,array_sha,sha,CachedStatevectorQNN

PRIOR=ROOT.parent/'wdbc_vqc_k04_20261108'
SEEDS=[(1,101,1101),(2,211,1211),(3,307,1307)]

def circuits(row):
    fmap=ZFeatureMap(4,reps=row['feature_map_reps']) if row['encoder']=='Z' else ZZFeatureMap(4,reps=row['feature_map_reps'],entanglement=row['topology'])
    ansatz=RealAmplitudes(4,reps=row['ansatz_reps'],entanglement='reverse_linear',skip_final_rotation_layer=False)
    return fmap,ansatz

def audit_histories():
    manifest=json.loads((PRIOR/'manifest.json').read_text())
    rows=[]
    for row in manifest['fits']:
        root=PRIOR/'artifacts'/row['fit_id']
        attempts=[p for p in sorted(root.glob('attempt_*')) if json.loads((p/'attempt.json').read_text()).get('state')=='completed']
        if len(attempts)!=1:raise RuntimeError(f'Expected one completed attempt: {row["fit_id"]}')
        attempt=attempts[0]
        history=[json.loads(line) for line in (attempt/'training_loss.jsonl').read_text().splitlines()]
        status=json.loads((attempt/'optimizer_status.json').read_text())
        losses=np.asarray([h['loss'] for h in history],dtype=float)
        cumulative=np.minimum.accumulate(losses);tail=losses[-20:]
        improvement20=float(cumulative[max(0,len(losses)-21)]-cumulative[-1])
        improvement10=float(cumulative[max(0,len(losses)-11)]-cumulative[-1])
        tail_spread=float(np.ptp(tail));upward=float(np.mean(np.diff(tail)>0))
        assert len(losses)==status['nfev'] and len(losses)<=100
        rows.append({**{k:row[k] for k in ['fit_id','configuration_id','scaler','rfe_estimator','encoder','topology','feature_map_reps','ansatz_reps','seed_pair_id']},
            'actual_nfev':int(status['nfev']),'termination':status['termination'],'optimizer_status':status['status'],
            'initial_training_loss':float(losses[0]),'best_evaluated_training_loss':float(losses.min()),'final_evaluated_training_loss':float(losses[-1]),'fitted_training_loss':status['fun'],
            'best_improvement_last_20':improvement20,'best_improvement_last_10':improvement10,'tail20_loss_range':tail_spread,'tail20_loss_std':float(np.std(tail,ddof=1)),'tail20_upward_step_fraction':upward,
            'finite_shot_plateau_uncertain':bool(tail_spread>improvement20 and upward>0),
            'interpretation_limit':'Finite-shot quantization plus optimizer moves; these histories alone cannot isolate shot fluctuations from parameter changes. Integer sampler seed makes repeated identical parameters deterministic.',
            'training_trace_path':str(attempt/'training_loss.jsonl'),'trace_sha256':sha(attempt/'training_loss.jsonl')})
    frame=pd.DataFrame(rows)
    assert len(frame)==192
    frame.to_csv(ROOT/'tables/existing_training_audit.csv',index=False)
    grouping=['encoder','topology','feature_map_reps','ansatz_reps']
    summary=frame.groupby(grouping,dropna=False).agg(fits=('fit_id','size'),mean_nfev=('actual_nfev','mean'),budget_exhausted=('termination',lambda s:int((s=='budget_exhausted').sum())),converged=('termination',lambda s:int((s=='converged').sum())),mean_best_loss=('best_evaluated_training_loss','mean'),mean_last20_improvement=('best_improvement_last_20','mean'),uncertain_plateaus=('finite_shot_plateau_uncertain','sum')).reset_index()
    summary.to_csv(ROOT/'tables/existing_training_audit_summary.csv',index=False)
    atomic_json(ROOT/'records/existing_training_audit.json',{'rows':192,'source_manifest_sha256':sha(PRIOR/'manifest.json'),'training_only':True,'original_test_metrics_read':False,'budget_exhausted':int((frame.termination=='budget_exhausted').sum()),'converged':int((frame.termination=='converged').sum()),'uncertain_plateau_flags':int(frame.finite_shot_plateau_uncertain.sum()),'flag_definition':'Last-20 loss range exceeds best-loss improvement over that window and contains upward steps; this is an ambiguity flag, not a measured shot-noise estimate or a convergence test.'})
    return rows

def main():
    if (ROOT/'manifest.json').exists():raise RuntimeError('Locked pilot exists; use controller for recovery')
    audit_histories()
    original=json.loads((PRIOR/'source/historical/split_idx.json').read_text())
    original_train=np.asarray(original['train_idx'],dtype=np.int64)
    assert len(original_train)==455
    with np.load(PRIOR/'source/data/wdbc_arrays.npz',allow_pickle=False) as archive:
        X=np.asarray(archive['X'][original_train],dtype=np.float64)
        y=np.asarray(archive['y'][original_train],dtype=np.int64)
        names=archive['feature_names'].astype(str)
    # No original test observations or labels are indexed, inspected or scored.
    fit_pos,val_pos=train_test_split(np.arange(455),train_size=364,test_size=91,stratify=y,random_state=20261108)
    fit_ids=original_train[fit_pos];val_ids=original_train[val_pos]
    assert not set(fit_ids).intersection(val_ids) and set(fit_ids).union(val_ids)==set(original_train)
    np.savez_compressed(ROOT/'source/original_training_only.npz',X=X,y=y,feature_names=names,original_indices=original_train)
    split={'random_state':20261108,'original_training_indices':original_train,'pilot_training_positions':fit_pos,'validation_positions':val_pos,'pilot_training_indices':fit_ids,'validation_indices':val_ids,'pilot_training_labels':y[fit_pos],'validation_labels':y[val_pos],
        'pilot_training_class_counts':{'malignant':int(np.sum(y[fit_pos]==0)),'benign':int(np.sum(y[fit_pos]==1))},'validation_class_counts':{'malignant':int(np.sum(y[val_pos]==0)),'benign':int(np.sum(y[val_pos]==1))},
        'original_train_index_sha256':array_sha(original_train),'pilot_train_index_sha256':array_sha(fit_ids),'validation_index_sha256':array_sha(val_ids),'pilot_train_label_sha256':array_sha(y[fit_pos]),'validation_label_sha256':array_sha(y[val_pos]),'original_test_used':False}
    atomic_json(ROOT/'split.json',split)
    preprocessing=[]
    settings=[('MinMaxScaler','Logistic Regression'),('MinMaxScaler','Random Forest'),('StandardScaler','Logistic Regression'),('StandardScaler','Random Forest')]
    with threadpool_limits(limits=1):
        for p,(scaler_name,estimator_name) in enumerate(settings):
            scaler=MinMaxScaler() if scaler_name=='MinMaxScaler' else StandardScaler()
            estimator=LogisticRegression(penalty='l1',solver='saga',max_iter=5000,random_state=42) if estimator_name=='Logistic Regression' else RandomForestClassifier(random_state=42)
            scaled=scaler.fit_transform(X[fit_pos])
            selector=RFE(estimator,n_features_to_select=4,step=1).fit(scaled,y[fit_pos])
            train=np.ascontiguousarray(selector.transform(scaled),dtype=np.float64)
            val=np.ascontiguousarray(selector.transform(scaler.transform(X[val_pos])),dtype=np.float64)
            selected=np.flatnonzero(selector.support_)
            np.savez_compressed(ROOT/f'preprocessing/p{p}.npz',X_train=train,X_validation=val,y_train=y[fit_pos],y_validation=y[val_pos],train_indices=fit_ids,validation_indices=val_ids,selected_indices=selected,selected_names=names[selected])
            joblib.dump({'scaler':scaler,'selector':selector},ROOT/f'preprocessing/p{p}.joblib')
            params={key:np.asarray(getattr(scaler,key)).tolist() for key in ('mean_','scale_','var_','min_','data_min_','data_max_','data_range_') if hasattr(scaler,key)}
            audit={'p':p,'scaler':scaler_name,'rfe_estimator':estimator_name,'scaler_parameters':params,'selector_settings':{k:(repr(v) if k=='estimator' else v) for k,v in selector.get_params(deep=False).items()},'selected_feature_indices':selected,'selected_feature_names':names[selected].tolist(),'fit_rows':364,'scaler_fit_index_sha256':array_sha(fit_ids),'rfe_fit_index_sha256':array_sha(fit_ids),'rfe_fit_labels_sha256':array_sha(y[fit_pos]),'validation_rows_in_fit':False,'validation_labels_in_fit':False,'original_test_used':False,'X_train_sha256':array_sha(train),'X_validation_sha256':array_sha(val),'cache_sha256':sha(ROOT/f'preprocessing/p{p}.npz')}
            atomic_json(ROOT/f'preprocessing/p{p}.audit.json',audit);preprocessing.append(audit)
    configs=[];fits=[]
    pairs=[(1,1),(1,2),(2,1),(2,2)]
    maps=[('Z',''),('ZZ','linear'),('ZZ','full'),('ZZ','circular')]
    for p,(scaler,selector) in enumerate(settings):
        for r,(freps,areps) in enumerate(pairs):
            fmap_index=(p+r)%4;encoder,topology=maps[fmap_index]
            cid=f'pilot-p{p}-r{r}-m{fmap_index}'
            row={'configuration_id':cid,'p':p,'r':r,'feature_map_index':fmap_index,'scaler':scaler,'rfe_estimator':selector,'encoder':encoder,'topology':topology,'feature_map_reps':freps,'ansatz_reps':areps,'ansatz_entanglement':'reverse_linear','final_rotation_layer':True}
            fmap,ansatz=circuits(row)
            with (ROOT/f'circuits/{cid}.qpy').open('wb') as handle:qpy.dump([QuantumCircuit(4).compose(c) for c in (fmap,ansatz,fmap.compose(ansatz))],handle)
            row.update({'circuit_path':f'circuits/{cid}.qpy','circuit_sha256':sha(ROOT/f'circuits/{cid}.qpy'),'parameter_order':[str(param) for param in ansatz.parameters],'feature_parameter_order':[str(param) for param in fmap.parameters]})
            configs.append(row)
            for budget in (100,200,300):
                for pair,initial_seed,sampler_seed in SEEDS:
                    initial=np.random.default_rng(initial_seed).uniform(0,1,ansatz.num_parameters)
                    fits.append({**row,'fit_id':f'{cid}-s{pair}-b{budget}','seed_pair_id':pair,'initialization_seed':initial_seed,'sampler_seed':sampler_seed,'budget':budget,'initial_parameters':initial.tolist(),'initial_parameter_sha256':array_sha(initial)})
    policy={'design':'Balanced 16-configuration pilot; feature_map_index=(p+r) modulo 4; not full factorial grid','budgets':[100,200,300],'fits':144,'configurations':16,'seed_pairs':SEEDS,'objective_limit':'COBYLA maxiter is maximum objective evaluations in installed SciPy; record actual nfev and raw stopping status','loss':'cross_entropy, installed log2 with clipping at 1e-10','shots':1024,'qubits':4,'precision':'complex128 states, float64 probabilities','warm_start':False,'ansatz':'RealAmplitudes, reverse_linear, final rotation enabled','initialization':'Fresh seeded NumPy generator, uniform [0,1); identical vector for matching configuration/seed across all budgets','readout':'basis-state integer modulo 2; source probability order [0,1]; malignant=0','decision':'argmax from one final validation probability array; ties select malignant column 0','primary_metric':'Equal mean of validation accuracies across the fixed 16 configurations and three seeds (48 fits)','primary_comparisons':[[200,100],[300,200]],'secondary_comparison':[300,100],'bootstrap':'10000 ordinary paired patient bootstrap draws from the 91 validation patients, identical draw indices across every budget/configuration/seed; fixed fitted-model panel; percentile 2.5/97.5','bootstrap_seed':20261108,'bootstrap_replicates':10000,'bootstrap_conditions':'Conditional on pilot-training split, fitted models, 16 configurations and three seed pairs. Excludes uncertainty from retraining on other samples or pilot selection. No formal multiple-comparison or equivalence claims.','practical_margin_pp':0.5,'margin_status':'Proposed practical relevance margin, not a universal scientific standard; all three stages mandatory irrespective of results','exact_optimizer_resume_supported':False,'restart_rule':'New independent fit from original initialization and seeds for each budget and interrupted attempt; never use final angles as optimizer state','technical_retry_rule':'One retry maximum for OSError, MemoryError, TimeoutError or worker-process failure; retain all attempts. No retry for poor scores, budget exhaustion or numerical/scientific errors.','incomplete_rule':'No complete-panel budget mean or bootstrap comparison when any expected fit is missing/failed; report counts and unavailable summaries; never silently average a successful subset','stage_order':[100,200,300],'original_test_partition':'Not copied, inspected or scored; only original 455 training cases used','numerical_implementation':VERSION,'cpu_only':True,'cpu_benchmarks':'Matched six predeclared configurations at 100 evaluations: seed pair 1 for one worker, pair 2 for two workers, pair 3 for three workers if live RAM permits. One BLAS/OpenMP thread per worker; selection uses throughput/resources only.','comparison_timing':'Local controller may compute/save metrics; substantive interpretation and recommendation only after 144 outcomes are accounted for and all budget stages verified.'}
    assert len(configs)==16 and len(fits)==144
    atomic_json(ROOT/'manifest.json',{'protocol':policy,'configurations':configs,'fits':fits})
    pd.DataFrame([{k:v for k,v in row.items() if not isinstance(v,list)} for row in fits]).to_csv(ROOT/'manifest.csv',index=False)
    atomic_json(ROOT/'records/manifest_lock.json',{'manifest_sha256':sha(ROOT/'manifest.json'),'split_sha256':sha(ROOT/'split.json'),'training_data_sha256':sha(ROOT/'source/original_training_only.npz'),'engine_sha256':sha(ROOT/'vqc_engine.py'),'original_engine_sha256':sha(PRIOR/'vqc_engine.py'),'identical_verified_engine':sha(ROOT/'vqc_engine.py')==sha(PRIOR/'vqc_engine.py'),'preprocessing':[{'p':p,'sha256':sha(ROOT/f'preprocessing/p{p}.npz'),'audit_sha256':sha(ROOT/f'preprocessing/p{p}.audit.json')} for p in range(4)]})
    vm=psutil.virtual_memory()
    atomic_json(ROOT/'records/environment.json',{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':{name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas','scikit-learn','qiskit','qiskit-machine-learning','qiskit-algorithms','threadpoolctl','psutil','joblib']},'physical_cores':psutil.cpu_count(logical=False),'logical_processors':psutil.cpu_count(),'ram_total':vm.total,'ram_available':vm.available,'cpu_percent':psutil.cpu_percent(interval=1)})
    checks=[]
    with threadpool_limits(limits=1):
        for row in configs:
            with np.load(ROOT/f'preprocessing/p{row["p"]}.npz') as data:inputs=data['X_train'][:2]
            fmap,ansatz=circuits(row)
            if row['encoder']=='ZZ':
                alias=PauliFeatureMap(4,reps=row['feature_map_reps'],entanglement=row['topology'],paulis=['Z','ZZ'],alpha=2.)
                for x in inputs:np.testing.assert_allclose(Statevector.from_instruction(fmap.assign_parameters(x)).data,Statevector.from_instruction(alias.assign_parameters(x)).data,atol=1e-12,rtol=0)
            weights=np.linspace(-2,2,ansatz.num_parameters)
            for seed in (1101,1211,1307):
                sampler=StatevectorSampler(default_shots=1024,seed=seed)
                reference=SamplerQNN(circuit=fmap.compose(ansatz),sampler=sampler,input_params=list(fmap.parameters),weight_params=list(ansatz.parameters),interpret=lambda x:x%2,output_shape=2)
                candidate=CachedStatevectorQNN(fmap,ansatz,StatevectorSampler(default_shots=1024,seed=seed),seed)
                expected=reference.forward(inputs,weights);actual=candidate.forward(inputs,weights)
                np.testing.assert_array_equal(actual,expected)
                checks.append({'configuration_id':row['configuration_id'],'sampler_seed':seed,'max_probability_error':float(np.max(np.abs(actual-expected)))})
    atomic_json(ROOT/'records/numerical_validation.json',{'passed':True,'cases':len(checks),'max_probability_error':0,'alias_semantics_verified':True,'input_partition':'pilot_training_only','checks':checks})
    print('FROZEN: 16 configurations, 144 fit IDs, internal split 364/91; 48 seeded numerical checks passed. Original test partition not inspected.')

if __name__=='__main__':main()
