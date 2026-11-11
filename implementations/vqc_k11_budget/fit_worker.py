from __future__ import annotations
import datetime
import json
import logging
import os
import shutil
import threading
import time
import traceback
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.utils import algorithm_globals as algorithm_rng
from qiskit_machine_learning.utils import algorithm_globals as ml_rng
from threadpoolctl import threadpool_limits
from vqc_engine import ROOT,VERSION,atomic_json,sha,array_sha,CachedStatevectorQNN,RecordedCOBYLA,clinical
from prepare_pilot import circuits

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')

class MemoryObserver:
    def __enter__(self):
        self.process=psutil.Process();self.peak=self.process.memory_info().rss;self.stop=threading.Event()
        def sample():
            while not self.stop.wait(.1):self.peak=max(self.peak,self.process.memory_info().rss)
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.peak=max(self.peak,self.process.memory_info().rss);self.stop.set();self.thread.join()

def verify_artifacts(row,path,metrics):
    split=json.loads((ROOT/'split.json').read_text())
    with np.load(path/'predictions.npz',allow_pickle=False) as saved:
        for prefix,expected,n in [('train','pilot_training_indices',row['training_rows']),('validation','validation_indices',row['evaluation_rows'])]:
            indices=saved[prefix+'_indices'];labels=saved[prefix+'_labels'];pred=saved[prefix+'_predictions'];prob=saved[prefix+'_probabilities']
            assert indices.tolist()==split[expected] and len(indices)==n and len(np.unique(indices))==n
            assert prob.shape==(n,2) and np.isfinite(prob).all()
            np.testing.assert_array_equal(pred,np.argmax(prob,axis=1))
            np.testing.assert_allclose(prob.sum(axis=1),1,atol=1e-12,rtol=0)
            np.testing.assert_allclose(prob*1024,np.round(prob*1024),atol=1e-12,rtol=0)
            truth=split['pilot_training_labels' if prefix=='train' else 'validation_labels']
            np.testing.assert_array_equal(labels,truth)
            recalc=clinical(labels,pred,prob)
            assert all(abs(value-metrics[prefix][key])<=1e-12 for key,value in recalc.items())
        np.testing.assert_array_equal(saved['initial_parameters'],row['initial_parameters'])
        assert array_sha(saved['initial_parameters'])==row['initial_parameter_sha256']
    trace=[json.loads(line) for line in (path/'training_loss.jsonl').read_text().splitlines()]
    status=json.loads((path/'optimizer_status.json').read_text())
    assert len(trace)==status['nfev']<=row['budget']
    assert trace[0]['weights']==row['initial_parameters']
    assert status['termination'] in ('converged','budget_exhausted')
    assert (status['status']==1 and status['success']) or (status['status']==2 and status['nfev']==row['budget'])
    audit=json.loads((path/'fit_audit.json').read_text())
    assert audit['classes']==[0,1] and audit['validation_probability_calls']==1
    assert audit['optimizer_input_sha256']==audit['training_input_sha256'] and not audit['validation_in_fit'] and not audit['validation_labels_in_fit'] and audit['original_test_used']==(row['evidence_type']=='full-grid')
    return {'verified':True,'prediction_rows':row['evaluation_rows'],'metric_max_error':0,'nfev':len(trace)}

def validate_receipt(row):
    global ROOT
    ROOT=Path(row['study_root'])
    path=ROOT/'artifacts'/row['artifact_key']/'receipt.json'
    if not path.exists():return None
    receipt=json.loads(path.read_text())
    if receipt['status']=='failed':return receipt
    assert receipt['fit_id']==row['fit_id'] and receipt['budget']==row['budget'] and receipt['seed_pair_id']==row['seed_pair_id']
    attempt=ROOT/receipt['attempt_path']
    for file,digest in receipt['file_hashes'].items():
        if sha(attempt/file)!=digest:raise RuntimeError(f'Checkpoint hash mismatch: {row["fit_id"]}/{file}')
    verify_artifacts(row,attempt,receipt['metrics'])
    return receipt

def fit_one(row,attempt_number=1,benchmark_workers=None):
    global ROOT
    import vqc_engine
    ROOT=Path(row['study_root']);vqc_engine.ROOT=ROOT
    vqc_engine.ENCODER_BANKS.clear()
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
    logging.getLogger('qiskit_machine_learning').setLevel(logging.ERROR)
    root=ROOT/'artifacts'/row['artifact_key'];root.mkdir(exist_ok=True)
    attempt_id=f'attempt_{attempt_number:03d}';attempt=root/attempt_id
    attempt.mkdir(exist_ok=False)
    started=now();timer=time.perf_counter();cpu_start=time.process_time();peak=psutil.Process().memory_info().rss
    metadata={'fit_id':row['fit_id'],'attempt_id':attempt_id,'status':'running','started':started,'pid':os.getpid(),'restarted_from':f'attempt_{attempt_number-1:03d}' if attempt_number>1 else None,'restart_initialization':'original_saved_vector_and_seed_pair','exact_optimizer_resume':False}
    atomic_json(attempt/'attempt.json',metadata)
    atomic_json(attempt/'initialization.json',{'initial_parameters':row['initial_parameters'],'parameter_order':row['parameter_order'],'initialization_seed':row['initialization_seed'],'sampler_seed':row['sampler_seed']})
    optimizer=None
    try:
        with MemoryObserver() as memory,threadpool_limits(limits=1),warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            np.random.seed(row['initialization_seed']);algorithm_rng.random_seed=row['initialization_seed'];ml_rng.random_seed=row['initialization_seed']
            with np.load(ROOT/f'preprocessing/p{row["p"]}.npz',allow_pickle=False) as data:
                Xtrain=data['X_train'];ytrain=data['y_train'];itrain=data['train_indices'];names=data['selected_names'].astype(str).tolist()
            fmap,ansatz=circuits(row)
            initial=np.asarray(row['initial_parameters'],dtype=np.float64)
            optimizer=RecordedCOBYLA(attempt,budget=int(row['budget']))
            sampler=StatevectorSampler(default_shots=1024,seed=int(row['sampler_seed']))
            model=VQC(feature_map=fmap,ansatz=ansatz,loss='cross_entropy',optimizer=optimizer,sampler=sampler,warm_start=False,initial_point=initial.copy())
            qnn=CachedStatevectorQNN(fmap,ansatz,sampler,row['sampler_seed']);model._neural_network=qnn
            started_fit=time.perf_counter();model.fit(Xtrain,ytrain);fit_seconds=time.perf_counter()-started_fit
            status=optimizer.raw_status
            if status['termination'] not in ('converged','budget_exhausted'):raise ArithmeticError('Optimizer failed: '+str(status))
            assert all(call['rows']==row['training_rows'] and call['input_sha256']==array_sha(Xtrain) for call in qnn.forward_calls)
            fit_calls=len(qnn.forward_calls)
            classes=model._target_encoder.categories_[0].astype(int).tolist();assert classes==[0,1]
            final=np.asarray(model.weights,dtype=np.float64);assert final.shape==initial.shape and np.isfinite(final).all()
            train_prob=np.asarray(model.predict_proba(Xtrain),dtype=np.float64)
            # Validation observations and labels load only after optimizer termination.
            with np.load(ROOT/f'preprocessing/p{row["p"]}.npz',allow_pickle=False) as data:
                Xval=data['X_validation'];yval=data['y_validation'];ival=data['validation_indices']
            before=len(qnn.forward_calls);val_prob=np.asarray(model.predict_proba(Xval),dtype=np.float64)
            assert len(qnn.forward_calls)==before+1
            train_pred=np.argmax(train_prob,axis=1).astype(np.int64);val_pred=np.argmax(val_prob,axis=1).astype(np.int64)
            metrics={'train':clinical(ytrain,train_pred,train_prob),'validation':clinical(yval,val_pred,val_prob)}
            np.savez_compressed(attempt/'predictions.npz',train_indices=itrain,validation_indices=ival,train_labels=ytrain,validation_labels=yval,train_predictions=train_pred,validation_predictions=val_pred,train_probabilities=train_prob,validation_probabilities=val_prob,initial_parameters=initial,final_parameters=final)
            pd.DataFrame({'sample_index':ival,'source_label':yval,'predicted_source_label':val_pred,'malignant_probability':val_prob[:,0],'benign_probability':val_prob[:,1]}).to_csv(attempt/'validation_predictions.csv',index=False)
            atomic_json(attempt/'parameters.json',{'initial':initial,'final':final,'parameter_order':row['parameter_order']})
            audit={'training_input_sha256':array_sha(Xtrain),'optimizer_input_sha256':array_sha(Xtrain),'train_index_sha256':array_sha(itrain),'training_label_sha256':array_sha(ytrain),'validation_index_sha256':array_sha(ival),'classes':classes,'optimizer_forward_calls':fit_calls,'validation_probability_calls':1,'validation_loaded_after_fit':True,'validation_in_fit':False,'validation_labels_in_fit':False,'original_test_used':row['evidence_type']=='full-grid','positive_source_label':0,'decision':'argmax; ties malignant'}
            atomic_json(attempt/'fit_audit.json',audit)
            atomic_json(attempt/'metrics.json',metrics)
            shutil.copy2(ROOT/row['circuit_path'],attempt/'circuits.qpy')
            shutil.copy2(ROOT/f'preprocessing/p{row["p"]}.audit.json',attempt/'preprocessing.json')
            atomic_json(attempt/'configuration.json',{'fit':row,'protocol_sha256':sha(ROOT/'manifest.json'),'data_split_sha256':sha(ROOT/'split.json'),'environment_sha256':sha(ROOT/'records/environment.json'),'source_lock_sha256':sha(ROOT/'records/source_lock.json'),'implementation_version':VERSION,'cpu_threads':1,'original_test_used':row['evidence_type']=='full-grid'})
            serialization={'supported':True,'path':'model.dill'}
            try:
                qnn.encoder_banks.clear()
                if row['qubits']<=4:model.save(str(attempt/'model.dill'))
                else:serialization={'supported':False,'reason':'Large-width models preserved via QPY, initial/final parameters and protocol; avoid serializing cached state banks.'}
            except Exception as error:serialization={'supported':False,'error':str(error)}
            atomic_json(attempt/'serialization.json',serialization)
            atomic_json(attempt/'warnings.json',[{'category':w.category.__name__,'message':str(w.message)} for w in captured])
            verification=verify_artifacts(row,attempt,metrics)
            atomic_json(attempt/'verification.json',verification)
            metadata.update({'status':'completed','ended':now(),'termination':status['termination']});atomic_json(attempt/'attempt.json',metadata)
            peak=max(memory.peak,psutil.Process().memory_info().rss)
            runtime={'started':started,'ended':now(),'total_seconds':time.perf_counter()-timer,'fit_seconds':fit_seconds,'cpu_seconds':time.process_time()-cpu_start,'peak_rss_bytes':peak,'threads':1,'benchmark_workers':benchmark_workers,'pid':os.getpid()}
            trace=optimizer.history;losses=np.asarray([h['loss'] for h in trace]);running=np.minimum.accumulate(losses)
            receipt={**row,'status':'completed','attempt_id':attempt_id,'attempt_path':str(attempt.relative_to(ROOT)),'attempt_record':metadata,'technical_failure':False,'implementation_version':VERSION,'metrics':metrics,'termination':status['termination'],'optimizer_status':status,'nfev':len(trace),'initial_training_loss':float(losses[0]),'best_training_loss':float(losses.min()),'last_training_loss':float(losses[-1]),'fitted_training_loss':float(status['fun']),'last20_best_improvement':float(running[max(0,len(trace)-21)]-running[-1]),'tail20_loss_range':float(np.ptp(losses[-20:])),'runtime':runtime,'selected_features':names,'model_serialization':serialization,'verification':verification,'file_hashes':{p.name:sha(p) for p in attempt.iterdir() if p.is_file()}}
            atomic_json(attempt/'receipt.json',receipt)
            atomic_json(root/'receipt.json',receipt)
            return receipt
    except Exception as error:
        technical=isinstance(error,(OSError,MemoryError,TimeoutError))
        metadata.update({'status':'failed','ended':now(),'error_type':type(error).__name__,'error':str(error),'technical_failure':technical})
        atomic_json(attempt/'attempt.json',metadata)
        failure={**row,'status':'failed','attempt_id':attempt_id,'attempt_path':str(attempt.relative_to(ROOT)),'attempt_record':metadata,'technical_failure':technical,'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),'termination':'failed','nfev':len(optimizer.history) if optimizer else 0,'runtime':{'started':started,'ended':now(),'total_seconds':time.perf_counter()-timer,'peak_rss_bytes':peak,'threads':1}}
        atomic_json(attempt/'failure.json',failure);atomic_json(attempt/'receipt.json',failure);atomic_json(root/'receipt.json',failure)
        return failure
