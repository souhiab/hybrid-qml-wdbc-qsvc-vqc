from __future__ import annotations
import json
import os
import shutil
import time
import traceback
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
from common import *

def workload():
    candidates=read(ROOT/'candidates.json');classical=read(ROOT/'comparators.json')
    folds=read(ROOT/'folds/outer_inner_assignments.json')['outer_records']
    return [{'evaluation_id':f"{p['candidate_id']}-{fold_id(f)}",'candidate_id':p['candidate_id'],'kind':'qsvc' if p['candidate_id'].startswith('Q') else 'classical','definition':p,'fold':f} for p in candidates+classical['strong']+classical['matched'] for f in folds]

def receipt_for(item):
    regular=ROOT/'runs'/item['evaluation_id']/'receipt.json'
    reused=ROOT/'reused_predictions'/item['evaluation_id']/'receipt.json'
    return regular if regular.exists() else reused

def validate_receipt(item,path=None,synthetic_data=None):
    receipt_path=Path(path or receipt_for(item))
    if not receipt_path.exists():return None
    receipt=read(receipt_path)
    if receipt['status']=='failed':return receipt
    assert receipt['evaluation_id']==item['evaluation_id']
    attempt=Path(receipt['attempt_path'])
    if not attempt.is_absolute():attempt=ROOT/attempt
    for name,digest in receipt['file_hashes'].items():assert sha(attempt/name)==digest,(receipt['evaluation_id'],name)
    frame=pd.read_csv(attempt/'predictions.csv')
    assert frame.sample_index.tolist()==item['fold']['test_indices']
    X,y,names=dataset() if synthetic_data is None else synthetic_data
    np.testing.assert_array_equal(frame.true_source_label,y[item['fold']['test_indices']])
    assert frame.evaluation_id.nunique()==1 and frame.evaluation_id.iloc[0]==item['evaluation_id']
    for key,value in metrics(frame.true_source_label,frame.prediction,frame.malignant_score).items():assert abs(value-receipt['metrics'][key])<=1e-12
    assert receipt['train_indices']==item['fold']['train_indices'] and receipt['test_indices']==item['fold']['test_indices']
    if not receipt.get('reused'):
        audit=read(attempt/'fit_audit.json')
        assert not audit['outer_heldout_in_preprocessing_fit'] and not audit['outer_heldout_in_model_fit'] and not audit['outer_heldout_in_tuning']
        assert audit['preprocessing']['train_indices']==item['fold']['train_indices']
        if item['kind']=='qsvc':assert audit['model_details']['C']==1 and audit['model_details']['classes']==[0,1]
    return receipt

def evaluate(item,attempt_number=1,threads=1,synthetic_data=None,synthetic_root=None):
    from learning import qsvc,classical
    synthetic=synthetic_data is not None
    if synthetic:
        X,y,names=synthetic_data;assert len(X)<=80 and synthetic_root is not None
        root=Path(synthetic_root).resolve();assert ROOT/'synthetic_outputs' in root.parents or root==ROOT/'synthetic_outputs'
        cache=root/'cache'
    else:X,y,names=dataset();root=ROOT/'runs';cache=ROOT/'cache'
    folder=root/item['evaluation_id'];folder.mkdir(parents=True,exist_ok=True)
    attempt=folder/f'attempt_{attempt_number:03d}';attempt.mkdir(exist_ok=False)
    timer=time.perf_counter();cpu=time.process_time()
    record={'evaluation_id':item['evaluation_id'],'attempt_id':attempt.name,'status':'running','started':now(),'pid':os.getpid(),'synthetic':synthetic,'restarted_from':f'attempt_{attempt_number-1:03d}' if attempt_number>1 else None}
    atomic(attempt/'attempt.json',record)
    try:
        with threadpool_limits(limits=threads),warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            definition=item['definition'];fold=item['fold']
            if item['kind']=='qsvc':pred,score,details,audit,prepfile,model=qsvc(definition,fold,X,y,names,attempt,item['evaluation_id'],cache_root=cache)
            else:pred,score,details,audit,prepfile,model=classical(definition,fold,X,y,names,attempt,cache_root=cache)
            heldout=fold['test_indices'];train=fold['train_indices']
            frame=pd.DataFrame({'candidate_id':item['candidate_id'],'evaluation_id':item['evaluation_id'],'repeat_id':fold['repeat_id'],'outer_fold_index':fold['outer_fold_index'],'sample_index':heldout,'true_source_label':y[heldout],'prediction':pred,'malignant_score':score,'score_type':details['score_type']})
            if 'malignant_decision_scores' in details:frame['malignant_decision_score']=details['malignant_decision_scores']
            frame.to_csv(attempt/'predictions.csv',index=False)
            measurement=metrics(y[heldout],pred,score);atomic(attempt/'metrics.json',measurement)
            atomic(attempt/'configuration.json',{'definition':definition,'fold_id':fold_id(fold),'protocol_sha256':sha(ROOT/'protocol.json'),'fold_manifest_sha256':sha(ROOT/'folds/outer_inner_assignments.json'),'environment':environment(),'controlled_seed':seed_for(item['evaluation_id']),'synthetic':synthetic})
            atomic(attempt/'split.json',{'train_indices':train,'heldout_indices':heldout,'train_labels':y[train],'heldout_labels':y[heldout]})
            atomic(attempt/'features.json',{'selected_indices':audit['selected_indices'],'selected_names':audit['selected_names']})
            atomic(attempt/'fit_audit.json',{'outer_heldout_in_preprocessing_fit':False,'outer_heldout_in_model_fit':False,'outer_heldout_in_tuning':False,'preprocessing':audit,'model_details':details,'fit_indices':train,'fit_label_hash':array_hash(y[train])})
            shutil.copy2(prepfile,attempt/'preprocessing.joblib');joblib.dump(model,attempt/'model.joblib',compress=3)
            atomic(attempt/'model_info.json',{'class':type(model).__name__,'settings':{k:repr(v) if k in ['kernel','quantum_kernel'] else v for k,v in model.get_params(deep=False).items()},'details':details})
            atomic(attempt/'warnings.json',[{'category':w.category.__name__,'message':str(w.message)} for w in caught])
        memory=psutil.Process().memory_info()
        runtime={'seconds':time.perf_counter()-timer,'cpu_seconds':time.process_time()-cpu,'rss_bytes':memory.rss,'peak_rss_bytes':getattr(memory,'peak_wset',memory.rss),'threads':threads}
        atomic(attempt/'runtime.json',runtime)
        record.update({'status':'completed','ended':now()});atomic(attempt/'attempt.json',record)
        resource_record={**runtime,**{key:details[key] for key in ['circuit_qubits','feature_map_reps','circuit_depth','circuit_operations'] if key in details}}
        receipt={'status':'completed','evaluation_id':item['evaluation_id'],'candidate_id':item['candidate_id'],'repeat_id':fold['repeat_id'],'outer_fold_index':fold['outer_fold_index'],'train_indices':train,'test_indices':heldout,'metrics':measurement,'score_type':details['score_type'],'runtime_seconds':runtime['seconds'],'resources':resource_record,'attempt_path':str(attempt.resolve()),'attempt_id':attempt.name,'reused':False,'synthetic':synthetic,'file_hashes':{f.name:sha(f) for f in attempt.iterdir() if f.is_file()}}
        # Verify all saved evidence before publishing the authoritative receipt.
        atomic(attempt/'verification_candidate.json',receipt)
        validate_receipt(item,attempt/'verification_candidate.json',synthetic_data=synthetic_data)
        atomic(attempt/'verification.json',{'verified':True,'metrics_recomputed_from_saved_predictions':True,'training_local_fit_scope':True})
        receipt['file_hashes']={f.name:sha(f) for f in attempt.iterdir() if f.is_file() and f.name not in ('receipt.json','verification_candidate.json')}
        atomic(attempt/'receipt.json',receipt);atomic(folder/'receipt.json',receipt)
        return receipt
    except Exception as error:
        record.update({'status':'failed','ended':now(),'error':str(error),'error_type':type(error).__name__,'technical_failure':isinstance(error,(OSError,MemoryError,TimeoutError))})
        atomic(attempt/'attempt.json',record)
        failure={'status':'failed','candidate_id':item['candidate_id'],'evaluation_id':item['evaluation_id'],'attempt_id':attempt.name,'attempt_path':str(attempt.resolve()),'repeat_id':item['fold']['repeat_id'],'outer_fold_index':item['fold']['outer_fold_index'],'error':str(error),'traceback':traceback.format_exc(),'technical_failure':record['technical_failure'],'synthetic':synthetic}
        atomic(attempt/'failure.json',failure);atomic(folder/'receipt.json',failure)
        # Large unfinished state banks are temporary caches, not scientific results.
        for file in attempt.glob('training_states.tmp.npy'):
            assert file.resolve().parent==attempt.resolve();file.unlink()
        return failure
