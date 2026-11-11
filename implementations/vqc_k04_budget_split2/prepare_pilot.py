from __future__ import annotations
import importlib.metadata
import json
import platform
import shutil
import sys
import numpy as np
import pandas as pd
import psutil
import joblib
from qiskit.circuit.library import ZFeatureMap,ZZFeatureMap,RealAmplitudes,PauliFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits
from vqc_engine import ROOT,VERSION,atomic_json,sha,array_sha,CachedStatevectorQNN

PRIOR=ROOT.parent/'wdbc_vqc_budget_pilot_20261108'

def circuits(row):
    fmap=ZFeatureMap(4,reps=int(row['feature_map_reps'])) if row['encoder']=='Z' else ZZFeatureMap(4,reps=int(row['feature_map_reps']),entanglement=row['topology'])
    ansatz=RealAmplitudes(4,reps=int(row['ansatz_reps']),entanglement='reverse_linear',skip_final_rotation_layer=False)
    return fmap,ansatz

def main():
    if (ROOT/'manifest.json').exists():raise RuntimeError('Locked manifest exists; resume instead of preparing again')
    first=json.loads((ROOT/'source/first_pilot_manifest.json').read_text())
    first_split=json.loads((ROOT/'source/first_pilot_split.json').read_text())
    with np.load(ROOT/'source/original_training_only.npz',allow_pickle=False) as data:
        X=data['X'];y=data['y'];original_ids=data['original_indices'];names=data['feature_names']
    assert X.shape==(455,30) and len(y)==455
    train_pos,val_pos=train_test_split(np.arange(455),train_size=364,test_size=91,stratify=y,random_state=20261109)
    train_ids=original_ids[train_pos];val_ids=original_ids[val_pos]
    assert train_ids.tolist()!=first_split['pilot_training_indices'] and val_ids.tolist()!=first_split['validation_indices']
    assert set(val_ids)!=set(first_split['validation_indices'])
    assert not set(train_ids).intersection(val_ids) and set(train_ids).union(val_ids)==set(original_ids)
    split={'random_state':20261109,'original_training_indices':original_ids,'pilot_training_positions':train_pos,'validation_positions':val_pos,'pilot_training_indices':train_ids,'validation_indices':val_ids,'pilot_training_labels':y[train_pos],'validation_labels':y[val_pos],'pilot_training_class_counts':{'malignant':int(np.sum(y[train_pos]==0)),'benign':int(np.sum(y[train_pos]==1))},'validation_class_counts':{'malignant':int(np.sum(y[val_pos]==0)),'benign':int(np.sum(y[val_pos]==1))},'original_train_index_sha256':array_sha(original_ids),'pilot_train_index_sha256':array_sha(train_ids),'validation_index_sha256':array_sha(val_ids),'pilot_train_label_sha256':array_sha(y[train_pos]),'validation_label_sha256':array_sha(y[val_pos]),'differs_from_first_split':True,'validation_overlap_with_split1':len(set(val_ids).intersection(first_split['validation_indices'])),'original_test_used':False}
    atomic_json(ROOT/'split.json',split)
    settings=[('MinMaxScaler','Logistic Regression'),('MinMaxScaler','Random Forest'),('StandardScaler','Logistic Regression'),('StandardScaler','Random Forest')]
    with threadpool_limits(limits=1):
        for p,(scaling,selection) in enumerate(settings):
            scaler=MinMaxScaler() if scaling=='MinMaxScaler' else StandardScaler()
            estimator=LogisticRegression(penalty='l1',solver='saga',max_iter=5000,random_state=42) if selection=='Logistic Regression' else RandomForestClassifier(random_state=42)
            scaled=scaler.fit_transform(X[train_pos]);selector=RFE(estimator,n_features_to_select=4,step=1).fit(scaled,y[train_pos])
            train=np.ascontiguousarray(selector.transform(scaled),dtype=np.float64);validation=np.ascontiguousarray(selector.transform(scaler.transform(X[val_pos])),dtype=np.float64)
            selected=np.flatnonzero(selector.support_)
            np.savez_compressed(ROOT/f'preprocessing/p{p}.npz',X_train=train,X_validation=validation,y_train=y[train_pos],y_validation=y[val_pos],train_indices=train_ids,validation_indices=val_ids,selected_indices=selected,selected_names=names[selected])
            joblib.dump({'scaler':scaler,'selector':selector},ROOT/f'preprocessing/p{p}.joblib')
            audit={'p':p,'scaler':scaling,'rfe_estimator':selection,'fit_rows':364,'freshly_fitted_for_split2':True,'previous_fitted_preprocessing_reused':False,'scaler_fit_index_sha256':array_sha(train_ids),'rfe_fit_index_sha256':array_sha(train_ids),'rfe_fit_labels_sha256':array_sha(y[train_pos]),'validation_rows_in_fit':False,'validation_labels_in_fit':False,'original_test_used':False,'selected_feature_indices':selected,'selected_feature_names':names[selected].tolist(),'scaler_parameters':{k:np.asarray(getattr(scaler,k)).tolist() for k in ('mean_','scale_','var_','min_','data_min_','data_max_','data_range_') if hasattr(scaler,k)},'selector_settings':{k:repr(v) if k=='estimator' else v for k,v in selector.get_params(deep=False).items()},'X_train_sha256':array_sha(train),'X_validation_sha256':array_sha(validation),'cache_sha256':sha(ROOT/f'preprocessing/p{p}.npz')}
            atomic_json(ROOT/f'preprocessing/p{p}.audit.json',audit)
    configs=first['configurations']
    assert len(configs)==16
    fits=[]
    for config in configs:
        shutil.copy2(PRIOR/config['circuit_path'],ROOT/config['circuit_path'])
        assert sha(ROOT/config['circuit_path'])==config['circuit_sha256']
        original=[r for r in first['fits'] if r['configuration_id']==config['configuration_id'] and r['budget']==100]
        for budget in (100,200):
            for row in original:
                fit={**row,'fit_id':f'split2-{row["configuration_id"]}-s{row["seed_pair_id"]}-b{budget}','budget':budget,'internal_split_seed':20261109}
                fmap,ansatz=circuits(fit)
                np.testing.assert_array_equal(np.asarray(fit['initial_parameters']),np.random.default_rng(fit['initialization_seed']).uniform(0,1,ansatz.num_parameters))
                fits.append(fit)
    assert len(fits)==96 and {r['budget'] for r in fits}=={100,200}
    policy={**first['protocol'],'design':'Second internal-split confirmation of the same balanced 16-configuration pilot; not a full factorial grid','budgets':[100,200],'fits':96,'internal_split_seed':20261109,'primary_comparisons':[[200,100]],'secondary_comparison':None,'bootstrap_seed':20261109,'bootstrap_conditions':'Conditional on split 2, these fitted models, the same 16 configurations and three seed pairs. Overlap with split 1 does not create independent patients. No pooling of overlapping splits.','margin_status':'Proposed 0.5-percentage-point relevance margin recorded for later user review; no automatic final budget-selection decision','stage_order':[100,200],'cpu_benchmarks':'Reuse validated 3 workers x 1 thread; current RAM/cores check may only reduce concurrency. No additional tuning fits.','comparison_timing':'Automatically produce descriptive summaries after both stages; interpretation and budget selection deferred to a later user prompt.'}
    atomic_json(ROOT/'manifest.json',{'protocol':policy,'configurations':configs,'fits':fits})
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,list)} for r in fits]).to_csv(ROOT/'manifest.csv',index=False)
    vm=psutil.virtual_memory()
    atomic_json(ROOT/'records/environment.json',{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','qiskit','qiskit-machine-learning','qiskit-algorithms','threadpoolctl','psutil','joblib']},'physical_cores':psutil.cpu_count(logical=False),'logical_processors':psutil.cpu_count(),'ram_total':vm.total,'ram_available':vm.available,'cpu_percent':psutil.cpu_percent(interval=1)})
    assert sha(ROOT/'vqc_engine.py')==sha(PRIOR/'vqc_engine.py')
    atomic_json(ROOT/'records/manifest_lock.json',{'manifest_sha256':sha(ROOT/'manifest.json'),'split_sha256':sha(ROOT/'split.json'),'training_data_sha256':sha(ROOT/'source/original_training_only.npz'),'engine_sha256':sha(ROOT/'vqc_engine.py'),'original_engine_sha256':sha(PRIOR/'vqc_engine.py'),'first_pilot_manifest_sha256':sha(PRIOR/'manifest.json'),'same_16_configurations':True,'same_explicit_initial_vectors':True,'fresh_split2_preprocessing':True,'expected_outcomes':96})
    checks=[]
    with threadpool_limits(limits=1):
        for config in configs:
            fmap,ansatz=circuits(config)
            with np.load(ROOT/f'preprocessing/p{config["p"]}.npz') as data:values=data['X_train'][:2]
            weights=np.linspace(-2,2,ansatz.num_parameters)
            if config['encoder']=='ZZ':
                alias=PauliFeatureMap(4,reps=config['feature_map_reps'],entanglement=config['topology'],paulis=['Z','ZZ'],alpha=2)
                for x in values:np.testing.assert_allclose(Statevector.from_instruction(fmap.assign_parameters(x)).data,Statevector.from_instruction(alias.assign_parameters(x)).data,rtol=0,atol=1e-12)
            for seed in (1101,1211,1307):
                reference=SamplerQNN(circuit=fmap.compose(ansatz),sampler=StatevectorSampler(default_shots=1024,seed=seed),input_params=list(fmap.parameters),weight_params=list(ansatz.parameters),interpret=lambda x:x%2,output_shape=2)
                candidate=CachedStatevectorQNN(fmap,ansatz,StatevectorSampler(default_shots=1024,seed=seed),seed)
                np.testing.assert_array_equal(reference.forward(values,weights),candidate.forward(values,weights))
                checks.append({'configuration_id':config['configuration_id'],'sampler_seed':seed,'max_probability_error':0})
    atomic_json(ROOT/'records/numerical_validation.json',{'passed':True,'cases':48,'engine_unchanged':True,'max_probability_error':0,'alias_semantics_verified':True,'checks':checks})
    print('FROZEN split2: 96 fits, 16 matching configurations, caps 100/200, seed 20261109, fresh 364-case preprocessing; 48 numerical checks passed.')

if __name__=='__main__':main()
