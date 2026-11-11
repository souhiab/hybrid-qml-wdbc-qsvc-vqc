from __future__ import annotations
import hashlib
import json
import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Operator, Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_algorithms.optimizers import COBYLA, OptimizerResult
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
SHOTS = 1024
VERSION = 'vqc-k04-exact-cached-v2'
ENCODER_BANKS = {}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

def json_default(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(str(type(value)))

def atomic_json(path, payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(payload,indent=2,sort_keys=True,default=json_default)+'\n',encoding='utf-8')
    os.replace(temp,path)

def sampled_binary_probabilities(states, seed, shots=SHOTS):
    """Match StatevectorSampler(integer seed): RNG resets per bound circuit."""
    probabilities=np.abs(states)**2
    probabilities/=probabilities.sum(axis=1,keepdims=True)
    cdf=np.cumsum(probabilities,axis=1)
    draws=np.sort(np.random.default_rng(seed).random(shots))
    result=np.empty((len(states),2),dtype=np.float64)
    cumulative_counts=np.searchsorted(draws,cdf,side='left')
    if np.any(cumulative_counts[:,-1]!=shots):raise RuntimeError('Sampling CDF overflow')
    counts=cumulative_counts[:,0]+np.sum(cumulative_counts[:,2::2]-cumulative_counts[:,1:-1:2],axis=1)
    result[:,0]=counts/shots
    result[:,1]=1-result[:,0]
    return result

class CachedStatevectorQNN(SamplerQNN):
    """Same finite-shot VQC readout with cached encoders and exact ansatz matrices."""
    def __init__(self, feature_map, ansatz, sampler, sampler_seed):
        super().__init__(circuit=feature_map.compose(ansatz),sampler=sampler,
            input_params=list(feature_map.parameters),weight_params=list(ansatz.parameters),
            interpret=lambda x:x%2,output_shape=2,sparse=False)
        self.encoder=feature_map
        self.ansatz_template=ansatz
        self.sampler_seed=int(sampler_seed)
        self.encoder_banks={}
        self.forward_calls=[]

    def _forward(self,input_data,weights):
        inputs=np.asarray(input_data,dtype=np.float64)
        key=array_sha(inputs)
        if key not in self.encoder_banks:
            cache_key=(self.encoder.name,int(self.encoder.reps),str(self.encoder.entanglement),key)
            if cache_key not in ENCODER_BANKS:
                bank=np.ascontiguousarray([
                    Statevector.from_instruction(self.encoder.assign_parameters(row)).data for row in inputs
                ],dtype=np.complex128)
                bank.flags.writeable=False
                ENCODER_BANKS[cache_key]=bank
            self.encoder_banks[key]=ENCODER_BANKS[cache_key]
        unitary=np.asarray(Operator(self.ansatz_template.assign_parameters(weights)).data,dtype=np.complex128)
        states=self.encoder_banks[key]@unitary.T
        result=sampled_binary_probabilities(states,self.sampler_seed)
        self.forward_calls.append({'rows':len(inputs),'input_sha256':key})
        return result

class RecordedCOBYLA(COBYLA):
    """Preserve SciPy COBYLA status which Qiskit's wrapper otherwise discards."""
    def __init__(self, attempt_path, budget=100):
        super().__init__(maxiter=budget,rhobeg=1.0,tol=None,disp=False)
        self.attempt_path=Path(attempt_path)
        self.budget=budget
        self.history=[]
        self.raw_status=None

    def minimize(self,fun,x0,jac=None,bounds=None):
        started=time.perf_counter()
        def objective(weights):
            if len(self.history)>=self.budget: raise RuntimeError('Objective budget exceeded')
            value=float(fun(weights))
            item={'evaluation':len(self.history)+1,'loss':value,'elapsed_seconds':time.perf_counter()-started,'weights':np.asarray(weights).tolist()}
            self.history.append(item)
            with (self.attempt_path/'training_loss.jsonl').open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(item)+'\n');handle.flush()
            atomic_json(self.attempt_path/'optimization_checkpoint.json',{'exact_optimizer_resume_supported':False,'last_evaluation':item,'state':'running'})
            return value
        raw=minimize(objective,np.asarray(x0),method='COBYLA',options={'maxiter':self.budget,'rhobeg':1.0,'disp':False},tol=None)
        if int(raw.nfev)!=len(self.history): raise RuntimeError('Objective evaluation count mismatch')
        termination='converged' if raw.success else ('budget_exhausted' if int(raw.status)==2 and raw.nfev>=self.budget else 'optimizer_failed')
        self.raw_status={'status':int(raw.status),'success':bool(raw.success),'message':str(raw.message),'nfev':int(raw.nfev),'fun':float(raw.fun),'termination':termination,'maxcv':float(raw.get('maxcv',0))}
        result=OptimizerResult();result.x=raw.x;result.fun=raw.fun;result.nfev=raw.nfev
        result.nit=raw.get('nit');result.njev=raw.get('njev')
        atomic_json(self.attempt_path/'optimizer_status.json',self.raw_status)
        return result

def clinical(labels,predictions,probabilities):
    labels=np.asarray(labels);predictions=np.asarray(predictions)
    truth=(labels==0).astype(int);pred=(predictions==0).astype(int)
    tn,fp,fn,tp=confusion_matrix(truth,pred,labels=[0,1]).ravel()
    return {'accuracy':float(accuracy_score(truth,pred)),'correct_count':int(tp+tn),'n':len(labels),
        'roc_auc':float(roc_auc_score(truth,probabilities[:,0])),
        'average_precision':float(average_precision_score(truth,probabilities[:,0])),
        'sensitivity':float(recall_score(truth,pred,zero_division=0)),
        'specificity':float(tn/(tn+fp)),'precision':float(precision_score(truth,pred,zero_division=0)),
        'f1':float(f1_score(truth,pred,zero_division=0)),'TP':int(tp),'FN':int(fn),'FP':int(fp),'TN':int(tn)}
