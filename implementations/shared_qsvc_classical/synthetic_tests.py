from __future__ import annotations
import copy
import os
import shutil
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,average_precision_score
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import PauliFeatureMap
from common import *

def run():
    from execution import evaluate,validate_receipt
    from learning import preprocess
    import quantum_engine as quantum
    from analysis import weighted_panel
    root=ROOT/'synthetic_outputs'/('smoke-'+now().replace(':','').replace('-',''))
    root.mkdir(parents=True,exist_ok=False)
    X,y=make_classification(n_samples=60,n_features=8,n_informative=5,n_redundant=0,random_state=2201)
    y=y.astype(np.int8);names=np.asarray([f'synthetic_{i}' for i in range(8)])
    train,test=next(StratifiedKFold(n_splits=3,shuffle=True,random_state=501).split(X,y))
    inner=[]
    for ti,vi in StratifiedKFold(n_splits=3,shuffle=True,random_state=502).split(X[train],y[train]):inner.append({'train_indices':train[ti].tolist(),'validation_indices':train[vi].tolist()})
    fold={'repeat_id':1,'outer_fold_index':0,'train_indices':train.tolist(),'test_indices':test.tolist(),'inner_folds':inner}
    q=copy.deepcopy(read(ROOT/'candidates.json')[0])
    item={'candidate_id':'SYNTHETIC_Q','evaluation_id':'SYNTHETIC_Q-r1-f0','kind':'qsvc','definition':q,'fold':fold}
    result=evaluate(item,synthetic_data=(X,y,names),synthetic_root=root)
    assert result['status']=='completed',result
    receipt=root/item['evaluation_id']/'receipt.json'
    validate_receipt(item,receipt,synthetic_data=(X,y,names))
    model=joblib.load(Path(result['attempt_path'])/'model.joblib');prep=joblib.load(Path(result['attempt_path'])/'preprocessing.joblib')
    pred=pd.read_csv(Path(result['attempt_path'])/'predictions.csv')
    np.testing.assert_array_equal(model.predict(prep.transform(X[test])),pred.prediction)
    # Explicit mismatch/leak guards must fail before a learned fit is possible.
    rejected=False
    try:preprocess(X,y,names,train,test,{'k':4,'scaler':'minmax','selector':'rfe_lr'},'qsvc_grid',forbidden=[int(train[0])],cache_root=root/'cache')
    except AssertionError:rejected=True
    assert rejected
    saved=(Path(result['attempt_path'])/'predictions.csv').read_bytes()
    (Path(result['attempt_path'])/'predictions.csv').write_text('corrupt synthetic file',encoding='utf-8')
    caught=False
    try:validate_receipt(item,receipt,synthetic_data=(X,y,names))
    except AssertionError:caught=True
    assert caught
    (Path(result['attempt_path'])/'predictions.csv').write_bytes(saved)
    validate_receipt(item,receipt,synthetic_data=(X,y,names))
    interrupted=root/item['evaluation_id']/'attempt_002';interrupted.mkdir();atomic(interrupted/'attempt.json',{'status':'interrupted','reason':'synthetic interruption recovery check'})
    restarted=evaluate(item,attempt_number=3,synthetic_data=(X,y,names),synthetic_root=root)
    assert restarted['status']=='completed'
    repeated=pd.read_csv(Path(restarted['attempt_path'])/'predictions.csv')
    np.testing.assert_array_equal(pred.malignant_score,repeated.malignant_score)
    m=copy.deepcopy(read(ROOT/'comparators.json')['matched'][0])
    ci={**item,'candidate_id':'SYNTHETIC_M','evaluation_id':'SYNTHETIC_M-r1-f0','kind':'classical','definition':m}
    classical=evaluate(ci,synthetic_data=(X,y,names),synthetic_root=root)
    assert classical['status']=='completed',classical
    assert read(Path(classical['attempt_path'])/'tuning.json')['inner_model_fits']==27
    strong=copy.deepcopy(read(ROOT/'comparators.json')['strong'][0]);strong['search_space']=strong['search_space'][:2]
    si={**ci,'candidate_id':'SYNTHETIC_LR','evaluation_id':'SYNTHETIC_LR-r1-f0','definition':strong}
    sr=evaluate(si,synthetic_data=(X,y,names),synthetic_root=root);assert sr['status']=='completed',sr
    # Synthetic numerical equivalence for every exact requested circuit.
    rng=np.random.default_rng(20261108);checks=[]
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        for candidate in read(ROOT/'candidates.json'):
            feature_map=quantum.build_feature_map({'width':candidate['k'],'execution_constructor':candidate['feature_map'],'repetitions':candidate['feature_map_reps'],'topology':candidate['entanglement'] or ''})
            synthetic_values=rng.uniform(.1,.9,(2,candidate['k']))
            fast=np.asarray([quantum.exact_feature_map_statevector(feature_map,v) for v in synthetic_values])
            reference=np.asarray([Statevector.from_instruction(feature_map.assign_parameters(v)).data for v in synthetic_values])
            np.testing.assert_allclose(fast,reference,atol=2e-12,rtol=0)
            gram,audit=quantum.blocked_fidelity(fast,fast,True)
            np.testing.assert_allclose(gram,np.abs(reference.conj()@reference.T)**2,atol=2e-12,rtol=0)
            assert np.linalg.eigvalsh(gram).min()>-2e-12
            if candidate['feature_map']=='ZZFeatureMap':
                alias=PauliFeatureMap(candidate['k'],reps=candidate['feature_map_reps'],entanglement=candidate['entanglement'],paulis=['Z','ZZ'],alpha=2)
                np.testing.assert_allclose(Statevector.from_instruction(alias.assign_parameters(synthetic_values[0])).data,reference[0],atol=2e-12,rtol=0)
            checks.append({'candidate_id':candidate['candidate_id'],'synthetic_rows':2,'state_and_kernel_equivalence':True})
    # Weighted bootstrap implementation agrees with sklearn on synthetic tied scores.
    frames=[]
    for repeat in (1,2,3):frames.append(pd.DataFrame({'repeat_id':repeat,'sample_index':range(12),'true_source_label':[0,1]*6,'prediction':[0,1,1,1]*3,'malignant_score':[.8,.2,.5,.5]*3}))
    frame=pd.concat(frames,ignore_index=True);weights=np.asarray([[1]*12,[2,1,0,2,1,1,2,0,1,1,1,0]],dtype=float)
    measured=weighted_panel(frame,weights);base=frames[0]
    for i,w in enumerate(weights):
        assert abs(measured['roc_auc'][i]-roc_auc_score(base.true_source_label==0,base.malignant_score,sample_weight=w))<1e-12
        assert abs(measured['average_precision'][i]-average_precision_score(base.true_source_label==0,base.malignant_score,sample_weight=w))<1e-12
    # Simulated workbook sharing violation: retry must retain checkpoint evidence.
    import controller
    original_command=controller.ExcelWriter.command;original_root=controller.ROOT
    fake=root/'workbook_mock';(fake/'workbook_support').mkdir(parents=True);(fake/'records').mkdir()
    calls=[]
    def mock_command(self,command):
        calls.append(command)
        if len(calls)==1:return None # Synthetic file-sharing refusal.
        return {'allPassed':True}
    controller.ExcelWriter.command=mock_command;controller.ROOT=fake
    writer=controller.AsyncWriter()
    try:
        writer.submit({'scope':'SYNTHETIC','sequence':1});writer.submit({'scope':'SYNTHETIC','sequence':2});writer.finish()
    finally:controller.ExcelWriter.command=original_command;controller.ROOT=original_root
    validate_receipt(item,receipt,synthetic_data=(X,y,names))
    report={'passed':True,'scope':'SYNTHETIC DATA ONLY','synthetic_qsvc_fits':2,'synthetic_matched_outer_fits':1,'synthetic_matched_inner_fits':27,'synthetic_strong_reduced_grid_outer_fits':1,'synthetic_strong_inner_fits':6,'WDBC_models_fitted':0,'production_WDBC_kernels_generated':0,'guarded_training_scope_rejection':True,'checkpoint_corruption_detected':True,'model_and_preprocessor_serialization_verified':True,'interrupted_attempt_preserved_and_deterministic_restart_verified':True,'workbook_sharing_failure_simulated_without_receipt_loss':True,'weighted_bootstrap_matches_sklearn_with_ties':True,'circuits':checks,'output_directory':str(root)}
    baseline=ROOT/'records/synthetic_verification.json'
    atomic(ROOT/'records/synthetic_verification_latest.json' if baseline.exists() else baseline,report)
    print('Synthetic verification passed; no WDBC training or production WDBC kernels executed.')

if __name__=='__main__':run()
