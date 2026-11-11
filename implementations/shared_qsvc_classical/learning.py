from __future__ import annotations
import gc
import itertools
import json
import os
import shutil
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from qiskit_machine_learning.algorithms import QSVC
from historical.evaluation.preprocessing import build_preprocessor
from historical.evaluation.classical_models import make_logistic_regression,make_rbf_svm,select_auc_tolerant
from common import *
import quantum_engine as quantum

def preprocess(X,y,names,train_ids,eval_ids,config,regime,forbidden=(),cache_root=None,_locked=False):
    train_ids=np.asarray(train_ids,dtype=int);eval_ids=np.asarray(eval_ids,dtype=int)
    assert len(set(train_ids))==len(train_ids) and not set(train_ids)&set(eval_ids)
    assert not set(train_ids)&set(forbidden),'Held-out cases reached preprocessing fit'
    settings={'scaler':config['scaler'],'selector':config['selector'],'k':int(config['k']),'regime':regime}
    key=identity({'settings':settings,'train_ids':train_ids,'eval_ids':eval_ids,'X_train':array_hash(X[train_ids]),'y_train':array_hash(y[train_ids]),'X_eval':array_hash(X[eval_ids]),'code':sha(Path(__file__))})
    cache_root=Path(cache_root or ROOT/'cache');cache_root.mkdir(exist_ok=True,parents=True)
    prefix=cache_root/('pp-'+key[:24]);meta=prefix.with_suffix('.json');job=prefix.with_suffix('.joblib');arrays=prefix.with_suffix('.npz')
    if not _locked:
        with cache_lock(prefix.with_suffix('.lock')):
            return preprocess(X,y,names,train_ids,eval_ids,config,regime,forbidden,cache_root,_locked=True)
    if meta.exists():
        audit=read(meta)
        assert audit['key']==key and sha(job)==audit['pipeline_sha256'] and sha(arrays)==audit['arrays_sha256']
        with np.load(arrays,allow_pickle=False) as d:xt=np.array(d['train']);xe=np.array(d['eval'])
        return xt,xe,joblib.load(job),audit,job
    if regime=='historical_strong':
        frame=pd.DataFrame(X[train_ids],index=train_ids,columns=names);target=pd.Series(y[train_ids],index=train_ids)
        pipeline,log=build_preprocessor(scaler=config['scaler'],selector=config['selector'],k=int(config['k']),expected_fit_indices=train_ids,forbidden_fit_indices=forbidden)
        xt=pipeline.fit_transform(frame,target).to_numpy()
        xe=pipeline.transform(pd.DataFrame(X[eval_ids],index=eval_ids,columns=names)).to_numpy()
        selector=pipeline.named_steps['selector'].estimator_;events=log.events
    else:
        assert config['scaler'] in ('minmax','MinMaxScaler')
        estimator=LogisticRegression(penalty='l1',solver='saga',max_iter=5000,random_state=42) if config['selector']=='rfe_lr' else RandomForestClassifier(random_state=42)
        pipeline=Pipeline([('scaler',MinMaxScaler(feature_range=(0,1),clip=False)),('selector',RFE(estimator,n_features_to_select=int(config['k']),step=1))])
        xt=pipeline.fit_transform(X[train_ids],y[train_ids]);xe=pipeline.transform(X[eval_ids]);selector=pipeline.named_steps['selector']
        events=[{'step':step,'fit_indices':train_ids.tolist(),'fit_label_hash':array_hash(y[train_ids])} for step in ('scaler','supervised_RFE')]
    if hasattr(selector,'get_support'):
        selected=np.flatnonzero(selector.get_support()).tolist();selected_names=[str(names[i]) for i in selected]
    else:selected=None;selected_names=[f'PC{i+1}' for i in range(int(config['k']))]
    xt=np.ascontiguousarray(xt,dtype=np.float64);xe=np.ascontiguousarray(xe,dtype=np.float64)
    assert xt.shape==(len(train_ids),int(config['k'])) and xe.shape==(len(eval_ids),int(config['k']))
    temp=job.with_suffix('.joblib.tmp');joblib.dump(pipeline,temp);os.replace(temp,job)
    with arrays.with_suffix('.npz.tmp').open('wb') as stream:np.savez_compressed(stream,train=xt,eval=xe)
    os.replace(arrays.with_suffix('.npz.tmp'),arrays)
    audit={'key':key,'settings':settings,'train_indices':train_ids,'evaluation_indices':eval_ids,'fit_index_hash':array_hash(train_ids),'fit_label_hash':array_hash(y[train_ids]),'selected_indices':selected,'selected_names':selected_names,'fit_events':events,'outer_heldout_in_fit':False,'pipeline_sha256':sha(job),'arrays_sha256':sha(arrays)}
    atomic(meta,audit)
    return xt,xe,pipeline,audit,job

def fit_classifier(config,X,y):
    model=make_logistic_regression(float(config['C'])) if config['model_family']=='logistic_regression' else make_rbf_svm(float(config['C']),config['gamma'])
    model.fit(X,y);return model
def score_classifier(model,X):
    if hasattr(model,'predict_proba'):return positive_probability(model,X),'malignant_probability',.5
    return positive_decision(model,X),'malignant_decision_score',0.0

def classical(procedure,fold,X,y,names,output,cache_root=None):
    train=fold['train_indices'];test=fold['test_indices'];inner=fold['inner_folds']
    regime=procedure.get('preprocessing_regime','historical_strong')
    if procedure['family']=='matched_rbf_svm':
        grid=[{'model_family':'rbf_svm','scaler':'minmax','selector':procedure['selector'],'k':procedure['k'],'C':c,'gamma':g} for c,g in itertools.product(procedure['C_values'],procedure['gamma_values'])]
    else:grid=procedure['search_space']
    records=[];audits=[]
    for config in grid:
        aucs=[]
        for partition in inner:
            ti=partition['train_indices'];vi=partition['validation_indices']
            assert set(ti)|set(vi)==set(train) and not (set(ti)|set(vi))&set(test)
            xt,xv,_,audit,_=preprocess(X,y,names,ti,vi,config,regime,forbidden=test,cache_root=cache_root)
            model=fit_classifier(config,xt,y[ti]);score,_,_=score_classifier(model,xv)
            aucs.append(metrics(y[vi],np.where(score>= (.5 if config['model_family']=='logistic_regression' else 0),0,1),score)['roc_auc'])
            audits.append({'configuration':config,'train_indices':ti,'validation_indices':vi,'preprocessing_key':audit['key']})
        records.append({**config,'configuration':config,'auc_mean':float(np.mean(aucs)),'inner_fold_auc':aucs})
    selection=select_auc_tolerant(records);chosen=selection.selected['configuration']
    xt,xe,pipeline,audit,job=preprocess(X,y,names,train,test,chosen,regime,forbidden=test,cache_root=cache_root)
    model=fit_classifier(chosen,xt,y[train]);score,kind,threshold=score_classifier(model,xe)
    prediction=np.where(score>=threshold,0,1).astype(np.int8)
    atomic(output/'tuning.json',{'search_candidates':len(grid),'inner_model_fits':3*len(grid),'selection_rule':procedure.get('selection','0.005 inner-AUC tolerance, smaller C then gamma scale/.01/.1'),'selected':chosen,'records':records,'inner_fit_access':audits,'outer_heldout_used_for_selection':False})
    return prediction,score,{'selected_configuration':chosen,'score_type':kind,'threshold':threshold,'classes':model.classes_.tolist(),'inner_model_fits':3*len(grid)},audit,job,model

def qsvc(candidate,fold,X,y,names,output,evaluation_id,cache_root=None):
    train=fold['train_indices'];test=fold['test_indices']
    config={'k':candidate['k'],'scaler':'minmax','selector':candidate['selector']}
    xt,xe,pipeline,audit,job=preprocess(X,y,names,train,test,config,'qsvc_grid',forbidden=test,cache_root=cache_root)
    fmap=quantum.build_feature_map({'width':candidate['k'],'execution_constructor':candidate['feature_map'],'repetitions':candidate['feature_map_reps'],'topology':candidate['entanglement'] or ''})
    kernel_key=identity({'candidate':candidate,'fold_id':fold_id(fold),'train_ids':train,'heldout_ids':test,'preprocessing_key':audit['key'],'engine_sha256':sha(ROOT/'quantum_engine.py')})
    cache=Path(cache_root or ROOT/'cache');kp=cache/('kernel-'+kernel_key[:24]+'.npz');km=kp.with_suffix('.json')
    cached=km.exists()
    if cached:
        previous=read(km);assert previous['key']==kernel_key and sha(kp)==previous['sha256']
        with np.load(kp,allow_pickle=False) as d:kt=np.array(d['train']);kh=np.array(d['heldout'])
        bank=None;kernel_audit=previous['audit']
    else:
        bank_path=output/'training_states.tmp.npy'
        bank=np.lib.format.open_memmap(bank_path,mode='w+',dtype=np.complex128,shape=(len(train),1<<candidate['k']))
        for i,row in enumerate(xt):bank[i]=quantum.exact_feature_map_statevector(fmap,row)
        bank.flush()
        kt,train_audit=quantum.blocked_fidelity(bank,bank,symmetric=True,block_size=64)
        kernel_audit={'key':kernel_key,'training':train_audit,'statevector_dtype':'complex128','kernel_dtype':'float64','analytic':True,'shots':None,'PSD_projection_applied':False}
    callback=quantum.CachedKernel(xt,kt)
    model=QSVC(quantum_kernel=callback,C=1.0,probability=True,class_weight=None,shrinking=True,tol=.001,random_state=seed_for(evaluation_id),cache_size=200,max_iter=-1,decision_function_shape='ovr',break_ties=False)
    model.fit(xt,y[train])
    assert all(call['name']=='train_train' for call in callback.calls),'Nontraining kernel reached QSVC fit'
    if not cached:
        kh=np.empty((len(test),len(train)),dtype=np.float64)
        for start in range(0,len(test),4):
            vectors=np.asarray([quantum.exact_feature_map_statevector(fmap,row) for row in xe[start:start+4]],dtype=np.complex128)
            kh[start:start+len(vectors)]=np.abs(np.conjugate(vectors)@bank.T)**2
        bank._mmap.close();del bank
        bank_path.unlink()
        kernel_audit['heldout']={'shape':list(kh.shape),'generated_after_model_fit':True,'stream_batch':4}
        with kp.with_suffix('.npz.tmp').open('wb') as stream:np.savez_compressed(stream,train=kt,heldout=kh)
        os.replace(kp.with_suffix('.npz.tmp'),kp);atomic(km,{'key':kernel_key,'sha256':sha(kp),'audit':kernel_audit})
    callback.attach_test(xe,kh)
    prediction=np.asarray(model.predict(xe),dtype=np.int8);score=positive_probability(model,xe)
    from qiskit import qpy
    with (output/'feature_map.qpy').open('wb') as stream:qpy.dump([fmap],stream)
    circuit=fmap.decompose()
    details={'C':1.0,'probability':True,'probability_calibration_seed':seed_for(evaluation_id),'classes':model.classes_.tolist(),'score_type':'malignant_probability','decision':'estimator.predict, not probability threshold','malignant_decision_scores':positive_decision(model,xe),'kernel_calls':callback.calls,'kernel_audit':kernel_audit,'kernel_cache_reused':cached,'circuit_qubits':candidate['k'],'feature_map_reps':candidate['feature_map_reps'],'circuit_depth':circuit.depth(),'circuit_operations':dict(circuit.count_ops()),'input_parameter_order':[str(p) for p in fmap.parameters]}
    assert np.isfinite(score).all()
    gc.collect()
    return prediction,score,details,audit,job,model
