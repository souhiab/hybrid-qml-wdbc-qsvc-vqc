from pathlib import Path
import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from .io import atomic,read,identity,file_hash
from .protocol import workload

def array_hash(a):
    import hashlib
    a=np.ascontiguousarray(a);return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()
def pair_manifest(p):
    n=p['cohort']['train'];m=p['cohort']['test']
    return ([{'id':f'train-{i:03d}-{j:03d}','kind':'train','i':i,'j':j} for i in range(n) for j in range(i+1,n)]+
            [{'id':f'test-{i:03d}-{j:03d}','kind':'test','i':i,'j':j} for i in range(m) for j in range(n)])
def prepare(directory,p):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'protocol.json').exists():
        if read(directory/'protocol.json')!=p:raise ValueError('Frozen protocol differs; use a separate directory')
        return verify(directory)
    raw=load_breast_cancer();X=raw.data;y=raw.target;ids=np.arange(len(y))
    excluded=p['cohort'].get('excluded_source_row_ids',[])
    if len(set(excluded))!=len(excluded) or not set(excluded)<=set(ids):raise ValueError('Invalid excluded dataset row identifiers')
    eligible=ids[~np.isin(ids,excluded)]
    selected,_=train_test_split(eligible,train_size=p['cohort']['size'],stratify=y[eligible],random_state=p['cohort']['selection_seed'])
    train,test=train_test_split(selected,train_size=p['cohort']['train'],test_size=p['cohort']['test'],stratify=y[selected],random_state=p['cohort']['split_seed'])
    split={'dataset':'sklearn.datasets.load_breast_cancer','patient_identifier':'original sklearn row ordinal; no clinical patient identifier',
           'dataset_x_sha256':array_hash(X),'dataset_y_sha256':array_hash(y),'feature_names':raw.feature_names.tolist(),
           'selected_ids':selected.tolist(),'train_ids':train.tolist(),'test_ids':test.tolist(),
           'train_labels':y[train].tolist(),'test_labels':y[test].tolist(),
           'train_class_counts':{str(c):int(np.sum(y[train]==c)) for c in [0,1]},
           'test_class_counts':{str(c):int(np.sum(y[test]==c)) for c in [0,1]},
           'selection_uses_model_scores':False,'selection_uses_hardware_outcomes':False}
    if excluded:
        split.update(excluded_source_row_ids=excluded,eligible_cohort_size=len(eligible),exclusion_uses_first_cohort_membership_only=True)
    atomic(directory/'protocol.json',p);atomic(directory/'case_manifest.json',split)
    pairs=pair_manifest(p);atomic(directory/'pairs.json',pairs)
    np.savez_compressed(directory/'raw_cases.npz',X_train=X[train],X_test=X[test],y_train=y[train],y_test=y[test],train_ids=train,test_ids=test,feature_names=raw.feature_names)
    atomic(directory/'input_lock.json',{'protocol_hash':identity(p),'case_manifest_hash':identity(split),'pairs_hash':identity(pairs),'raw_cases_sha256':file_hash(directory/'raw_cases.npz')})
    atomic(directory/'status.json',{'state':'PREPARED_NOT_EXECUTED','workload':workload(p),'hardware_jobs_submitted':0})
    return verify(directory)
def verify(directory):
    directory=Path(directory);p=read(directory/'protocol.json');s=read(directory/'case_manifest.json');pairs=read(directory/'pairs.json');lock=read(directory/'input_lock.json')
    assert identity(p)==lock['protocol_hash'] and identity(s)==lock['case_manifest_hash'] and identity(pairs)==lock['pairs_hash']
    assert file_hash(directory/'raw_cases.npz')==lock['raw_cases_sha256']
    assert not set(s['train_ids'])&set(s['test_ids']) and set(s['train_ids'])|set(s['test_ids'])==set(s['selected_ids'])
    assert len(s['train_ids'])==p['cohort']['train'] and len(s['test_ids'])==p['cohort']['test']
    assert not set(s['selected_ids'])&set(p['cohort'].get('excluded_source_row_ids',[]))
    assert len(pairs)==workload(p)['main_circuits'] and len({r['id'] for r in pairs})==len(pairs)
    return {'passed':True,'workload':workload(p),'case_manifest_hash':identity(s),'protocol_hash':identity(p)}
def fit_preprocessing(X_train,y_train,X_test,p,feature_names):
    settings=p['preprocessing'];scaler=MinMaxScaler(feature_range=tuple(settings['feature_range']),clip=settings['clip'])
    scaled=scaler.fit_transform(X_train);selector=RFE(LogisticRegression(**settings['estimator']),n_features_to_select=settings['k'],step=settings['rfe_step'])
    train=selector.fit_transform(scaled,y_train)
    test=selector.transform(scaler.transform(X_test));indices=np.flatnonzero(selector.support_)
    audit={'fit_training_rows':len(X_train),'evaluation_rows_in_fit':False,'evaluation_labels_in_fit':False,
           'raw_training_sha256':array_hash(X_train),'training_labels_sha256':array_hash(y_train),
           'selected_indices':indices.tolist(),'selected_names':[str(feature_names[i]) for i in indices],
           'scaler_params':scaler.get_params(),'estimator_params':settings['estimator'],'rfe_step':settings['rfe_step'],
           'training_transformed_sha256':array_hash(train),'test_transformed_sha256':array_hash(test),
           'extra_pi_scaling':False,'test_clipping':False}
    return train,test,audit,scaler,selector
def preprocess(directory):
    directory=Path(directory);verify(directory);p=read(directory/'protocol.json');started=time.perf_counter()
    if (directory/'preprocessing_receipt.json').exists():
        r=read(directory/'preprocessing_receipt.json')
        assert file_hash(directory/'preprocessed.npz')==r['preprocessed_sha256']
        assert identity(read(directory/'preprocessing_audit.json'))==r['audit_hash']
        assert r['input_lock_hash']==identity(read(directory/'input_lock.json'))
        return r
    with np.load(directory/'raw_cases.npz') as raw:
        train,test,audit,scaler,selector=fit_preprocessing(raw['X_train'],raw['y_train'],raw['X_test'],p,raw['feature_names'])
        np.savez_compressed(directory/'preprocessed.npz',X_train=train,X_test=test,y_train=raw['y_train'],y_test=raw['y_test'],train_ids=raw['train_ids'],test_ids=raw['test_ids'])
    import joblib
    joblib.dump({'scaler':scaler,'selector':selector},directory/'preprocessing.joblib')
    atomic(directory/'preprocessing_audit.json',audit)
    r={'preprocessed_sha256':file_hash(directory/'preprocessed.npz'),'audit_hash':identity(audit),'input_lock_hash':identity(read(directory/'input_lock.json')),'local_preparation_seconds':time.perf_counter()-started}
    atomic(directory/'preprocessing_receipt.json',r);return r
