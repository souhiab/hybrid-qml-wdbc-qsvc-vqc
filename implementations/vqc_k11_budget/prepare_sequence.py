from __future__ import annotations
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import psutil
from qiskit import qpy,QuantumCircuit
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from vqc_engine import ROOT,VERSION,atomic_json,array_sha,sha
from prepare_pilot import circuits

SEEDS=[(1,101,1101),(2,211,1211),(3,307,1307)]
SETTINGS=[('MinMaxScaler','Logistic Regression','minmax','lr'),('MinMaxScaler','Random Forest','minmax','rf'),('StandardScaler','Logistic Regression','standard','lr'),('StandardScaler','Random Forest','standard','rf')]
POLICY={
 'phase_order':['full-grid-k04-b200','internal-budget-pilot-k07','internal-budget-pilot-k11','internal-budget-pilot-k17'],
 'initial_fit_count':480,'maximum_fit_count':624,
 'full_grid_k04':{'configurations':64,'seeds':3,'budget':200,'training_rows':455,'test_rows':114,'justification':'Additional optimization allowance and improved optimizer termination; no demonstrated meaningful accuracy gain.'},
 'larger_widths':[7,11,17],'larger_width_full_grids_authorized':False,
 'pilot_design':'Balanced 16 configurations: feature map index=(p+r)%4. Not a full factorial grid.',
 'pilot_initial_budgets':[100,200],'pilot_optional_budget':300,'maximum_automatic_budget':300,
 'primary_metric':'Validation accuracy, equally averaged over 16 configurations and three seed pairs.',
 'accuracy_ties':'Retain every exact accuracy tie; full-grid ranking uses summed correct counts of all three seeds.',
 'incomplete_panels':'No complete configuration mean or aggregate comparison unless every declared seed/fit is valid. Keep partial outcomes and failure rates.',
 'optional_300_rule':'After both 48-fit stages finish: add all 48 at 300 if convergence fraction at 200 <0.90 OR upper descriptive 95% paired 200-100 accuracy interval >=+0.5 percentage points. Incomplete panels require review and all predeclared 300 fits, never silently drop failed seeds.',
 'recommendation_rule':'Among tested budgets with >=90% optimizer-reported convergence, smallest budget with mean validation accuracy within 0.5 percentage points of best tested mean. Otherwise none satisfies criterion. Recommendations are provisional; never launch larger-width grids.',
 'unresolved_rule':'After 300, flag review if completion target unmet, any incomplete panel, or 300-200 upper interval >=0.5 points; continue to next width.',
 'practical_margin_pp':0.5,'convergence_target':0.90,'planning_criteria_status':'Practical study-design choices, not universal standards or guarantees of global optimization.',
 'bootstrap_resamples':10000,'bootstrap_seed':20261108,'bootstrap_method':'Paired patient percentile bootstrap; same 91 selected patient indices for every width, configuration, seed and budget; model panels fixed; linear 2.5/97.5 percentiles.',
 'uncertainty_conditions':'Conditional on this internal training split, these fitted models, the 16 configurations and three seed pairs; excludes retraining and pilot-selection uncertainty. Repeated predictions are not extra patients; no equivalence or formal multiple-comparison claims.',
 'internal_split_seed':20261108,'pilot_training_rows':364,'pilot_validation_rows':91,
 'seed_pairs':SEEDS,'initialization':'Explicit default_rng(initialization_seed).uniform(0,1,n_parameters); identical across budgets within width/configuration/seed.',
 'loss':'cross_entropy','loss_log_base':2,'clipping':1e-10,'shots':1024,'precision':'complex128 states; float64 probabilities/parameters',
 'optimizer':'COBYLA','rhobeg':1.0,'tol':None,'evaluation_cap_semantics':'SciPy maxiter/maxfun objective calls, verified actual nfev; genuine early termination allowed',
 'feature_maps':'ZFeatureMap without entanglement; ZZFeatureMap linear/full/circular, alpha=2, verified equivalent PauliFeatureMap paulis=[Z,ZZ]',
 'ansatz':'RealAmplitudes reverse_linear with final rotation layer, repetitions 1 or 2',
 'readout':'basis state integer %2, least-significant bit, NOT Hamming-weight parity',
 'positive_source_label':0,'decision_rule':'argmax of source-class columns [0,1], exact ties malignant 0; one final evaluation probability array per fit',
 'sampler_rng':'Fresh integer-seeded StatevectorSampler per fit; same common random uniforms per circuit, preserving installed reset semantics.',
 'optimizer_resume_supported':False,'recovery':'Preserve interrupted attempt; restart original saved initialization and seeds with a new attempt ID; no warm starts.',
 'technical_retry':'One automatic retry for OSError, MemoryError, TimeoutError or BrokenProcessPool. Keep poor and budget-limited fits; scientific failures require review.',
 'test_access':'Original test scores never determine budgets, scheduling, coverage or implementation. Phase A is exploratory fixed-split evidence with test-selection history. Pilots read only separate original-training source.',
 'cpu_only':True,'threads_per_worker':1,'resource_policy':'Four-qubit up to three workers after live check. Each larger width begins with one compatible manifest fit; measure memory/throughput, compare two workers only if safe. Bound workers by RAM reserve, twice peak working set, cores and responsiveness; no forced CPU saturation.',
 'cache_policy':'At most one encoder bank per model, disk-backed for larger widths, chunked state evolution; one active configuration group per large width; release worker processes and prune cold disk banks to 4 GB after groups. Rebuild invariant banks when needed, never retrain completed fits.',
 'completion':'Authoritative verified per-fit atomic receipts first; coordinator alone writes Excel. Completion requires all activated fits, recomputed metrics and workbook readback agreement.',
}

def prepare():
    if (ROOT/'manifest.json').exists():raise RuntimeError('Frozen manifest exists: use resume.ps1')
    with np.load(ROOT/'source/wdbc_arrays.npz',allow_pickle=False) as data:
        X=data['X'];y=data['y'];names=data['feature_names'].astype(str)
    original=json.loads((ROOT/'source/original_split.json').read_text())
    train=np.asarray(original['train_idx'],dtype=np.int64);test=np.asarray(original['test_idx'],dtype=np.int64)
    expected=train_test_split(np.arange(569),test_size=.2,random_state=42,stratify=y)
    np.testing.assert_array_equal(train,expected[0]);np.testing.assert_array_equal(test,expected[1])
    first=json.loads((ROOT/'source/internal_split.json').read_text())
    tr,va=train_test_split(np.arange(455),train_size=364,test_size=91,stratify=y[train],random_state=20261108)
    np.testing.assert_array_equal(first['original_training_indices'],train)
    np.testing.assert_array_equal(first['pilot_training_indices'],train[tr]);np.testing.assert_array_equal(first['validation_indices'],train[va])
    assert first['pilot_train_index_sha256']==array_sha(train[tr]) and first['validation_index_sha256']==array_sha(train[va])
    np.testing.assert_array_equal(first['pilot_training_labels'],y[train[tr]]);np.testing.assert_array_equal(first['validation_labels'],y[train[va]])
    np.savez_compressed(ROOT/'source/original_training_only.npz',X=X[train],y=y[train],original_indices=train,feature_names=names)
    historical=ROOT.parent/'wdbc_vqc_k04_20261108'
    hmanifest=json.loads((ROOT/'source/historical_100_manifest.json').read_text())
    historical_rows=[]
    for row in hmanifest['fits']:
        receipt=json.loads((historical/'artifacts'/row['fit_id']/'receipt.json').read_text())
        assert receipt['status']=='completed'
        path=historical/receipt['attempt_path']
        for file,digest in receipt['file_hashes'].items():assert sha(path/file)==digest
        historical_rows.append({'source_fit_id':row['fit_id'],'configuration_id':row['configuration_id'],'budget':100,'cohort':'original 114-case test','source_receipt_path':str(historical/'artifacts'/row['fit_id']/'receipt.json'),'source_receipt_sha256':sha(historical/'artifacts'/row['fit_id']/'receipt.json'),**receipt['metrics']['test']})
    pd.DataFrame(historical_rows).to_csv(ROOT/'tables/Historical_100_Reference.csv',index=False)
    atomic_json(ROOT/'records/reuse_audit.json',{'historical_reference_fits':192,'historical_reference_hashes_verified':True,'reused_200_cap_full_training_fits':0,'reason':'Known 200-cap fits in prior pilots trained on 364 cases and are ineligible; original full grid protocol cap is 100.'})
    studies=[];allfits=[]
    for k,evidence in [(4,'full-grid'),(7,'internal-budget-pilot'),(11,'internal-budget-pilot'),(17,'internal-budget-pilot')]:
        sid=f'{evidence}-k{k:02d}';study=ROOT/'studies'/sid
        for directory in ('records','artifacts','circuits','preprocessing','analysis','state_cache'):(study/directory).mkdir(parents=True,exist_ok=True)
        ti=train if k==4 else train[tr];ei=test if k==4 else train[va]
        split={'pilot_training_indices':ti,'validation_indices':ei,'pilot_training_labels':y[ti],'validation_labels':y[ei],'training_index_sha256':array_sha(ti),'evaluation_index_sha256':array_sha(ei),'training_label_sha256':array_sha(y[ti]),'evaluation_label_sha256':array_sha(y[ei]),'class_counts_training':{'malignant':int(sum(y[ti]==0)),'benign':int(sum(y[ti]==1))},'class_counts_evaluation':{'malignant':int(sum(y[ei]==0)),'benign':int(sum(y[ei]==1))},'random_state':42 if k==4 else 20261108,'original_test_used':k==4}
        atomic_json(study/'split.json',split)
        with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as captured:
            for p,(scaling,selection,st,rt) in enumerate(SETTINGS):
                scaler=MinMaxScaler() if st=='minmax' else StandardScaler()
                estimator=LogisticRegression(penalty='l1',solver='saga',max_iter=5000,random_state=42) if rt=='lr' else RandomForestClassifier(random_state=42)
                transformed=scaler.fit_transform(X[ti]);selector=RFE(estimator,n_features_to_select=k,step=1).fit(transformed,y[ti])
                xt=np.ascontiguousarray(selector.transform(transformed),dtype=np.float64)
                xe=np.ascontiguousarray(selector.transform(scaler.transform(X[ei])),dtype=np.float64)
                selected=np.flatnonzero(selector.support_)
                cache=study/f'preprocessing/p{p}.npz'
                np.savez_compressed(cache,X_train=xt,X_validation=xe,y_train=y[ti],y_validation=y[ei],train_indices=ti,validation_indices=ei,selected_indices=selected,selected_names=names[selected])
                joblib.dump({'scaler':scaler,'selector':selector},study/f'preprocessing/p{p}.joblib')
                audit={'p':p,'qubits':k,'scaler':scaling,'rfe_estimator':selection,'fit_rows':len(ti),'fit_indices':ti,'fit_labels_sha256':array_sha(y[ti]),'scaler_fit_index_sha256':array_sha(ti),'rfe_fit_index_sha256':array_sha(ti),'rfe_fit_labels_sha256':array_sha(y[ti]),'validation_rows_in_fit':False,'validation_labels_in_fit':False,'original_test_in_fit':False,'selected_feature_indices':selected,'selected_feature_names':names[selected].tolist(),'X_train_sha256':array_sha(xt),'X_validation_sha256':array_sha(xe),'cache_sha256':sha(cache),'scaler_settings':scaler.get_params(),'rfe_settings':{'n_features_to_select':k,'step':1},'estimator_settings':estimator.get_params()}
                if k==4:
                    original_cache=historical/f'preprocessing/fs-k04-{st}-{rt}.npz'
                    with np.load(original_cache,allow_pickle=False) as old:
                        np.testing.assert_array_equal(xt,old['X_train']);np.testing.assert_array_equal(xe,old['X_test']);np.testing.assert_array_equal(selected,old['selected_indices'])
                    audit['matches_verified_historical_full_training_transformations']=True
                atomic_json(study/f'preprocessing/p{p}.audit.json',audit)
        atomic_json(study/'records/preprocessing_warnings.json',[str(w.message) for w in captured])
        configs=[];fits=[]
        for p,(scaling,selection,st,rt) in enumerate(SETTINGS):
            for r,(fr,ar) in enumerate([(1,1),(1,2),(2,1),(2,2)]):
                for m in (range(4) if k==4 else [(p+r)%4]):
                    encoder,topology=[('Z',''),('ZZ','linear'),('ZZ','full'),('ZZ','circular')][m]
                    token='z' if m==0 else f'zz-{topology}'
                    cid=f'{evidence}-k{k:02d}-{st}-{rt}-{token}-f{fr}-a{ar}'
                    config={'configuration_id':cid,'evidence_type':evidence,'qubits':k,'p':p,'r':r,'feature_map_index':m,'scaler':scaling,'rfe_estimator':selection,'encoder':encoder,'topology':topology,'feature_map_reps':fr,'ansatz_reps':ar,'ansatz_entanglement':'reverse_linear','final_rotation_layer':True}
                    fmap,ansatz=circuits(config);config['parameter_order']=[str(v) for v in ansatz.parameters];config['input_parameter_order']=[str(v) for v in fmap.parameters]
                    short=f'p{p}-m{m}-f{fr}-a{ar}'
                    config['circuit_path']=f'circuits/{short}.qpy'
                    with (study/config['circuit_path']).open('wb') as handle:qpy.dump([QuantumCircuit(k).compose(c) for c in (fmap,ansatz,fmap.compose(ansatz))],handle)
                    config['circuit_sha256']=sha(study/config['circuit_path']);configs.append(config)
                    for budget in ([200] if k==4 else [100,200,300]):
                        for pair,iseed,sseed in SEEDS:
                            initial=np.random.default_rng(iseed).uniform(0,1,ansatz.num_parameters)
                            fid=f'{cid}-i{iseed}-s{sseed}-b{budget}'
                            fits.append({**config,'fit_id':fid,'artifact_key':f'{short}-i{iseed}-s{sseed}-b{budget}','study_id':sid,'study_root':str(study),'budget':budget,'conditional':budget==300,'seed_pair_id':pair,'initialization_seed':iseed,'sampler_seed':sseed,'initial_parameters':initial,'initial_parameter_sha256':array_sha(initial),'training_rows':len(ti),'evaluation_rows':len(ei),'evaluation_cohort':'original-test' if k==4 else 'internal-validation'})
        assert len(configs)==(64 if k==4 else 16) and len(fits)==(192 if k==4 else 144)
        manifest={'protocol':POLICY,'configurations':configs,'fits':fits}
        atomic_json(study/'manifest.json',manifest)
        pd.DataFrame([{key:value for key,value in row.items() if not isinstance(value,(list,dict,np.ndarray))} for row in fits]).to_csv(study/'manifest.csv',index=False)
        studies.append({'study_id':sid,'qubits':k,'evidence_type':evidence,'manifest_path':str(study/'manifest.json'),'manifest_sha256':sha(study/'manifest.json'),'split_sha256':sha(study/'split.json'),'configurations':len(configs),'initial_fits':192 if k==4 else 96,'maximum_fits':len(fits)})
        allfits.extend(fits)
    assert len(allfits)==624 and sum(not r['conditional'] for r in allfits)==480
    environment={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':{name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','qiskit','qiskit-machine-learning','qiskit-algorithms','psutil','dill','threadpoolctl']},'physical_cores':psutil.cpu_count(logical=False),'logical_processors':psutil.cpu_count(),'ram_total':psutil.virtual_memory().total,'ram_available':psutil.virtual_memory().available}
    atomic_json(ROOT/'records/environment.json',environment)
    for study in studies:shutil.copy2(ROOT/'records/environment.json',Path(study['manifest_path']).parent/'records/environment.json')
    atomic_json(ROOT/'manifest.json',{'protocol':POLICY,'studies':studies,'initial_fits':480,'maximum_fits':624})
    pd.DataFrame([{key:value for key,value in row.items() if not isinstance(value,(list,dict,np.ndarray))} for row in allfits]).to_csv(ROOT/'manifest.csv',index=False)
    atomic_json(ROOT/'records/input_lock.json',{'files':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'manifest.json',*list((ROOT/'source').glob('*')),*(ROOT/'studies').glob('*/manifest.json'),*(ROOT/'studies').glob('*/split.json'),*(ROOT/'studies').glob('*/preprocessing/*.npz'),*(ROOT/'studies').glob('*/preprocessing/*.audit.json')]}})
    status=subprocess.run(['git','status','--porcelain=v1'],cwd=ROOT.parent/'implementations',capture_output=True,text=True)
    (ROOT/'records/repository_starting_status.txt').write_text(status.stdout,encoding='utf-8')
    print('FROZEN: 192 full-grid fits; 288 mandatory pilot fits; 144 conditional fits. Fresh preprocessing and exact saved split verified.',flush=True)

if __name__=='__main__':prepare()
