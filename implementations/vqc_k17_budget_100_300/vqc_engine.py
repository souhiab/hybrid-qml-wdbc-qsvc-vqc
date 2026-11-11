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
VERSION = 'vqc-native-strict-complex128-v5'
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
    import uuid
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    temp.write_text(json.dumps(payload,indent=2,sort_keys=True,default=json_default)+'\n',encoding='utf-8')
    # Windows readers/antivirus may briefly hold the destination without delete
    # sharing. Retry publication; keep the complete temporary file on failure.
    for attempt in range(40):
        try:
            os.replace(temp,path)
            return
        except PermissionError:
            if attempt==39:raise
            time.sleep(min(.02*(attempt+1),.5))

def sampled_binary_probabilities(states, seed, shots=SHOTS):
    """Same sorted uniform draws/CDF, finding only the sampled outcomes.

    Original count(draw < CDF) uses strict comparison. searchsorted(CDF,
    draw, side='right') preserves exactly that tie convention.
    """
    probabilities=np.abs(states)**2
    probabilities/=probabilities.sum(axis=1,keepdims=True)
    cdf=np.cumsum(probabilities,axis=1)
    draws=np.sort(np.random.default_rng(seed).random(shots))
    if np.any(cdf[:,-1]<=draws[-1]):raise RuntimeError('Sampling CDF overflow')
    result=np.empty((len(states),2),dtype=np.float64)
    for i,row in enumerate(cdf):
        outcomes=np.searchsorted(row,draws,side='right')
        result[i,0]=np.count_nonzero((outcomes & 1)==0)/shots
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

    def _forward_dense(self,input_data,weights):
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

    def _forward(self,input_data,weights):
        if self.encoder.num_qubits<=4:
            return self._forward_dense(input_data,weights)
        inputs=np.asarray(input_data,dtype=np.float64)
        key=array_sha(inputs)
        # Exactly one read-only encoder bank is retained per model. At 17 qubits
        # use a verified disk-backed bank shared between matching seed/budget fits.
        if key not in self.encoder_banks:
            self.encoder_banks.clear()
            self.encoder_banks[key]=encoded_bank(self.encoder,inputs)
        bank=self.encoder_banks[key]
        bound=self.ansatz_template.assign_parameters(weights).decompose()
        operations=[]
        for inst in bound.data:
            name=inst.operation.name
            bits=[bound.find_bit(q).index for q in inst.qubits]
            if name=='ry':operations.append(('ry',bits[0],float(inst.operation.params[0])))
            elif name=='cx':operations.append(('cx',bits[0],bits[1]))
            elif name!='barrier':raise RuntimeError('Unexpected ansatz operation '+name)
        chunk=max(1,int(os.environ.get('VQC_CHUNK_ROWS','8')))
        result=np.empty((len(inputs),2),dtype=np.float64)
        for start in range(0,len(inputs),chunk):
            states=np.array(bank[start:start+chunk],dtype=np.complex128,copy=True,order='C')
            evolve_ansatz(states,operations)
            result[start:start+chunk]=sampled_binary_probabilities(states,self.sampler_seed)
        self.forward_calls.append({'rows':len(inputs),'input_sha256':key})
        return result

@lru_cache(maxsize=32)
def cx_permutation(dim,control,target):
    indices=np.arange(dim,dtype=np.int64)
    indices=indices^(((indices>>control)&1)<<target)
    indices.flags.writeable=False
    return indices

def evolve_ansatz(states,operations):
    # Native in-place complex128 gates; same operations, strict double arithmetic.
    from native_backend.native_gates import evolve_ansatz as native_evolve
    return native_evolve(states,operations)


def encoded_bank(encoder,inputs):
    cache=ROOT/'state_cache';cache.mkdir(exist_ok=True)
    semantic=f'{encoder.num_qubits}-{encoder.name}-{encoder.reps}-{encoder.entanglement}-{array_sha(inputs)}'
    token=hashlib.sha256(semantic.encode()).hexdigest()
    path=cache/(token+'.npy');receipt=cache/(token+'.json')
    if receipt.exists() and path.exists():
        info=json.loads(receipt.read_text())
        if sha(path)==info['sha256']:
            return np.load(path,mmap_mode='r',allow_pickle=False)
        raise RuntimeError('Encoder cache hash mismatch '+token)
    # Coordinator groups matching encoder/preprocessing configurations, so at
    # large widths only one worker creates a given bank. O_EXCL also prevents
    # accidental concurrent producers.
    lock=cache/(token+'.lock')
    descriptor=None
    while descriptor is None:
        try:descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:
            if receipt.exists():return encoded_bank(encoder,inputs)
            try:
                import psutil
                owner=json.loads(lock.read_text())
                if not psutil.pid_exists(owner['pid']):lock.unlink();continue
            except (ValueError,FileNotFoundError):
                if lock.exists() and time.time()-lock.stat().st_mtime>10:
                    lock.unlink()
                    continue
            time.sleep(.5)
    try:
        with os.fdopen(descriptor,'w') as handle:json.dump({'pid':os.getpid()},handle)
        # Another producer can finish after our initial existence check but before
        # we acquire the lock. Never replace its published, memory-mapped bank.
        if receipt.exists() and path.exists():
            info=json.loads(receipt.read_text())
            if sha(path)!=info['sha256']:raise RuntimeError('Encoder cache hash mismatch '+token)
            return np.load(path,mmap_mode='r',allow_pickle=False)
        temp=cache/(token+f'.{os.getpid()}.tmp.npy')
        bank=np.lib.format.open_memmap(temp,mode='w+',dtype=np.complex128,shape=(len(inputs),1<<encoder.num_qubits))
        for index,row in enumerate(inputs):
            bank[index]=Statevector.from_instruction(encoder.assign_parameters(row)).data
        bank.flush();del bank
        os.replace(temp,path)
        atomic_json(receipt,{'sha256':sha(path),'rows':len(inputs),'qubits':encoder.num_qubits,'precision':'complex128','semantic':semantic})
        return np.load(path,mmap_mode='r',allow_pickle=False)
    finally:
        if lock.exists():lock.unlink()

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
