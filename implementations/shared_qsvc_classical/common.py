from __future__ import annotations
import datetime
import hashlib
import importlib.metadata
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score,average_precision_score,confusion_matrix

ROOT=Path(__file__).resolve().parent
PYTHON=Path(r'D:\envs\stage02c-qiskit142-py311\Scripts\python.exe')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),default=convert)
def convert(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)
def identity(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def array_hash(value):
    value=np.ascontiguousarray(value)
    return hashlib.sha256(canonical({'shape':value.shape,'dtype':value.dtype.str}).encode()+value.tobytes()).hexdigest()
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,default=convert)+'\n',encoding='utf-8');os.replace(temp,path)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
def seed_for(evaluation_id,role='libsvm'):
    return int.from_bytes(hashlib.sha256(f'20261108|{evaluation_id}|{role}'.encode()).digest()[:4],'big')%(2**31-1)
def dataset():
    with np.load(ROOT/'data/wdbc_arrays.npz',allow_pickle=False) as d:return np.array(d['X']),np.asarray(d['y'],dtype=np.int8),d['feature_names'].astype(str)
def metrics(y,pred,score):
    y=np.asarray(y);pred=np.asarray(pred);score=np.asarray(score,dtype=float)
    assert len(y)==len(pred)==len(score) and set(np.unique(y)).issubset({0,1}) and np.isfinite(score).all()
    truth=y==0;positive=pred==0;tn,fp,fn,tp=confusion_matrix(truth,positive,labels=[False,True]).ravel()
    ratio=lambda a,b:float(a/b) if b else 0.0
    return {'n':len(y),'roc_auc':float(roc_auc_score(truth,score)),'average_precision':float(average_precision_score(truth,score)),'accuracy':float(accuracy_score(y,pred)),'sensitivity':ratio(tp,tp+fn),'specificity':ratio(tn,tn+fp),'precision':ratio(tp,tp+fp),'f1':ratio(2*tp,2*tp+fp+fn),'TP':int(tp),'FN':int(fn),'FP':int(fp),'TN':int(tn)}
def positive_probability(model,X):
    classes=np.asarray(model.classes_);assert classes.tolist()==[0,1]
    return np.asarray(model.predict_proba(X))[:,int(np.flatnonzero(classes==0)[0])]
def positive_decision(model,X):
    classes=np.asarray(model.classes_);assert classes.tolist()==[0,1]
    return -np.asarray(model.decision_function(X),dtype=float)
def fold_id(record):return f"r{record['repeat_id']}-f{record['outer_fold_index']}"
def environment():
    import sys
    return {'python':sys.version,'executable':sys.executable,'packages':{name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','qiskit','qiskit-machine-learning','qiskit-algorithms','joblib','psutil','threadpoolctl']}}
def verify_frozen():
    for file,digest in read(ROOT/'records/preparation_lock.json')['files'].items():
        assert sha(ROOT/file)==digest,f'Frozen preparation changed: {file}'
    assert environment()['packages']==read(ROOT/'records/environment.json')['packages'],'Environment versions changed'

@contextmanager
def cache_lock(path):
    import psutil
    path=Path(path);started=time.monotonic();descriptor=None
    while descriptor is None:
        try:descriptor=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:
            try:
                owner=read(path)
                if not psutil.pid_exists(owner['pid']):path.unlink();continue
            except (ValueError,FileNotFoundError):
                if path.exists() and time.time()-path.stat().st_mtime>10:path.unlink();continue
            if time.monotonic()-started>1200:raise TimeoutError('Cache producer lock wait exceeded 1200 seconds')
            time.sleep(.25)
    try:
        with os.fdopen(descriptor,'w') as stream:json.dump({'pid':os.getpid()},stream)
        yield
    finally:
        if path.exists():path.unlink()
