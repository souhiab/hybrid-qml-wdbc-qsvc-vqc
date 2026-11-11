import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from qiskit import qpy,QuantumCircuit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits
from vqc_engine import ROOT,atomic_json,sha,array_sha
from prepare_pilot import circuits

def main():
    if (ROOT/'manifest.json').exists():raise RuntimeError('Frozen package exists; use grid_controller.py --check-only')
    with np.load(ROOT/'source/wdbc_arrays.npz',allow_pickle=False) as data:X=data['X'];y=data['y'];names=data['feature_names'].astype(str)
    original=json.loads((ROOT/'source/original_split.json').read_text())
    train=np.asarray(original['train_idx'],dtype=np.int64);test=np.asarray(original['test_idx'],dtype=np.int64)
    recovered=train_test_split(np.arange(569),test_size=.2,stratify=y,random_state=42)
    np.testing.assert_array_equal(train,recovered[0]);np.testing.assert_array_equal(test,recovered[1])
    assert len(train)==455 and len(test)==114 and not set(train)&set(test)
    split={'pilot_training_indices':train,'validation_indices':test,'pilot_training_labels':y[train],'validation_labels':y[test],'training_index_sha256':array_sha(train),'evaluation_index_sha256':array_sha(test),'training_label_sha256':array_sha(y[train]),'evaluation_label_sha256':array_sha(y[test]),'random_state':42,'original_test_used':True,'evaluation_cohort':'original 114-case test; internal worker field validation means test in this full-grid package'}
    atomic_json(ROOT/'split.json',split)
    protocol={'evidence_type':'full-grid','qubits':11,'budget':400,'budget_unit':'maximum objective evaluations','configurations':64,'seed_pairs':[[101,1101],[211,1211],[307,1307]],'expected_fits':192,'training_rows':455,'test_rows':114,'fit_access':'Scaler and supervised RFE fit only original 455 raw training cases and training labels. Test transformations attached after fitting. No pilot preprocessing or pilot models reused.','feature_maps':'Z without entanglement argument; ZZ linear/full/circular (verified Pauli equivalence)','feature_map_reps':[1,2],'ansatz':'RealAmplitudes reverse_linear, final rotation layer retained','ansatz_reps':[1,2],'optimizer':'COBYLA','rhobeg':1.0,'tol':None,'loss':'cross_entropy, installed base-2 and clipping 1e-10 semantics','shots':1024,'precision':'complex128 and float64','positive_source_class':0,'decision':'argmax of class probabilities [0,1]; exact ties malignant 0','evaluation':'One final test probability evaluation per fit; predictions and all clinical metrics derived from that saved array','ranking':'Mean accuracy over exactly three valid seeds; retain every exact accuracy tie by summed correct counts; incomplete configuration summaries unavailable','budget_exhaustion':'Retain valid budget-limited fits and poor outcomes; record actual nfev and optimizer termination; do not label every completed fit converged','interruption':'No exact COBYLA restoration; preserve attempts and restart only incomplete fits from original saved initialization and seeds; fresh model/optimizer/sampler','technical_retry':'One retry for technical failures only; preserve all failed attempts','selection_history':'Exploratory fixed original test split with historical test-selection history; not independent validation','budget_exception':{'cap_prepared':400,'pilot_converged':24,'pilot_budget_exhausted':24,'pilot_total':48,'convergence_fraction':24/48,'predeclared_target':.9,'target_met':False,'exception_required':True,'reason':'Additional optimizer allowance, not a demonstrated meaningful accuracy improvement','status':'User-requested 400-cap preparation; retain explicit exception for final scientific budget review alongside larger-width pilots'},'execution_authorization':'PREPARE ONLY. No launch, scheduling or connection to the current pilot controller. launch.ps1 requires a future deliberate invocation.','cpu':'CPU only; start with conservative live RAM check and at most three workers, one thread each; execution resources may be reduced without changing scientific settings.'}
    configs=[];fits=[]
    settings=[('MinMaxScaler','Logistic Regression','minmax','lr'),('MinMaxScaler','Random Forest','minmax','rf'),('StandardScaler','Logistic Regression','standard','lr'),('StandardScaler','Random Forest','standard','rf')]
    with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as captured:
        for p,(scaling,selection,st,rt) in enumerate(settings):
            scaler=MinMaxScaler() if st=='minmax' else StandardScaler()
            estimator=LogisticRegression(penalty='l1',solver='saga',max_iter=5000,random_state=42) if rt=='lr' else RandomForestClassifier(random_state=42)
            scaled=scaler.fit_transform(X[train]);selector=RFE(estimator,n_features_to_select=11,step=1).fit(scaled,y[train])
            xt=np.ascontiguousarray(selector.transform(scaled),dtype=np.float64);xe=np.ascontiguousarray(selector.transform(scaler.transform(X[test])),dtype=np.float64);selected=np.flatnonzero(selector.support_)
            file=ROOT/f'preprocessing/p{p}.npz'
            np.savez_compressed(file,X_train=xt,X_validation=xe,y_train=y[train],y_validation=y[test],train_indices=train,validation_indices=test,selected_indices=selected,selected_names=names[selected])
            joblib.dump({'scaler':scaler,'selector':selector},ROOT/f'preprocessing/p{p}.joblib')
            atomic_json(ROOT/f'preprocessing/p{p}.audit.json',{'p':p,'qubits':11,'fit_rows':455,'scaler':scaling,'rfe_estimator':selection,'scaler_fit_index_sha256':array_sha(train),'rfe_fit_index_sha256':array_sha(train),'rfe_fit_labels_sha256':array_sha(y[train]),'test_in_fit':False,'test_labels_in_fit':False,'pilot_preprocessing_reused':False,'selected_feature_indices':selected,'selected_feature_names':names[selected].tolist(),'scaler_settings':scaler.get_params(),'rfe_settings':{'n_features_to_select':11,'step':1},'estimator_settings':estimator.get_params(),'X_train_sha256':array_sha(xt),'X_validation_sha256':array_sha(xe),'cache_sha256':sha(file)})
            for m,(encoder,topology) in enumerate([('Z',''),('ZZ','linear'),('ZZ','full'),('ZZ','circular')]):
                for fr in (1,2):
                    for ar in (1,2):
                        token='z' if m==0 else 'zz-'+topology
                        cid=f'full-grid-k11-{st}-{rt}-{token}-f{fr}-a{ar}'
                        config={'configuration_id':cid,'evidence_type':'full-grid','qubits':11,'p':p,'feature_map_index':m,'scaler':scaling,'rfe_estimator':selection,'encoder':encoder,'topology':topology,'feature_map_reps':fr,'ansatz_reps':ar}
                        fmap,ansatz=circuits(config);short=f'p{p}-m{m}-f{fr}-a{ar}'
                        config['circuit_path']=f'circuits/{short}.qpy';config['parameter_order']=[str(v) for v in ansatz.parameters];config['input_parameter_order']=[str(v) for v in fmap.parameters]
                        with (ROOT/config['circuit_path']).open('wb') as handle:qpy.dump([QuantumCircuit(11).compose(c) for c in (fmap,ansatz,fmap.compose(ansatz))],handle)
                        config['circuit_sha256']=sha(ROOT/config['circuit_path']);configs.append(config)
                        for pair,(iseed,sseed) in enumerate([(101,1101),(211,1211),(307,1307)],1):
                            initial=np.random.default_rng(iseed).uniform(0,1,ansatz.num_parameters)
                            fits.append({**config,'fit_id':f'{cid}-i{iseed}-s{sseed}-b400','artifact_key':f'{short}-i{iseed}-s{sseed}-b400','budget':400,'seed_pair_id':pair,'initialization_seed':iseed,'sampler_seed':sseed,'initial_parameters':initial,'initial_parameter_sha256':array_sha(initial),'study_root':str(ROOT),'training_rows':455,'evaluation_rows':114,'evaluation_cohort':'original-test'})
    assert len(configs)==64 and len(fits)==192
    atomic_json(ROOT/'manifest.json',{'protocol':protocol,'configurations':configs,'fits':fits})
    pd.DataFrame([{k:v for k,v in row.items() if not isinstance(v,(list,dict,np.ndarray))} for row in fits]).to_csv(ROOT/'manifest.csv',index=False)
    atomic_json(ROOT/'records/preprocessing_warnings.json',[str(w.message) for w in captured])
    atomic_json(ROOT/'records/input_lock.json',{'files':{str(file.relative_to(ROOT)):sha(file) for file in [ROOT/'manifest.json',ROOT/'split.json',*list((ROOT/'source').glob('*')), *list((ROOT/'preprocessing').glob('*')), *list((ROOT/'circuits').glob('*'))]}})
    atomic_json(ROOT/'records/source_lock.json',{'files':{file:sha(ROOT/file) for file in ['vqc_engine.py','fit_worker.py','prepare_pilot.py','prepare_grid.py','grid_controller.py','workbook_support/workbook_writer.mjs']}})
    atomic_json(ROOT/'status.json',{'execution_state':'prepared_not_started','qubits':11,'budget':400,'configurations':64,'expected_fits':192,'completed':0,'failed':0,'pending':192,'training_launched':False})
    print('Prepared: 64 configurations x 3 seeds = 192 fits, cap 400. No VQC fit or test scoring executed.')

if __name__=='__main__':main()
