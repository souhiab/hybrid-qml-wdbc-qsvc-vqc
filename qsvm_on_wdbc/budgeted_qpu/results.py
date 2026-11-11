from pathlib import Path
import csv,math,warnings
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,roc_auc_score,average_precision_score,confusion_matrix,precision_score,f1_score
from .io import atomic,identity

class ResultValidationError(ValueError):pass
def counts_record(counts,requested,bits):
    if not isinstance(counts,dict) or not counts:raise ResultValidationError('Missing or empty counts; never substitute a zero kernel value')
    normalized={}
    for key,value in counts.items():
        key=str(key).replace(' ','')
        if key.startswith('0x'):key=format(int(key,16),f'0{bits}b')
        if len(key)!=bits or any(c not in '01' for c in key):raise ResultValidationError('Malformed measured bitstring')
        if not isinstance(value,(int,np.integer)) or value<0:raise ResultValidationError('Invalid returned shot count')
        normalized[key]=normalized.get(key,0)+int(value)
    total=sum(normalized.values())
    if total<=0:raise ResultValidationError('Zero returned shots')
    if total!=requested:raise ResultValidationError('Returned shots differ from the frozen request')
    return {'raw_counts':normalized,'requested_shots':requested,'returned_shots':total,'zero_probability':normalized.get('0'*bits,0)/total}
def extract_pub_results(result,pairs,shots,bits,register='meas'):
    if len(result)!=1:raise ResultValidationError('Expected exactly one broadcast Sampler PUB')
    pub=result[0];data=getattr(pub,'data',None)
    if data is None or not hasattr(data,register):raise ResultValidationError('Expected measurement register is absent')
    bitarray=getattr(data,register);out=[]
    if tuple(getattr(bitarray,'shape',()))!=(len(pairs),):raise ResultValidationError('Broadcast result shape mismatch')
    for i,pair in enumerate(pairs):
        try:counts=bitarray.get_counts(loc=i)
        except Exception as error:raise ResultValidationError('Cannot extract counts for pair '+pair['id']) from error
        out.append({'pair':pair,**counts_record(counts,shots,bits)})
    return out
def matrices(records,pairs,n,m):
    expected={p['id']:p for p in pairs}
    if set(records)!=set(expected):raise ResultValidationError('Incomplete pair coverage; metrics are not final')
    train=np.eye(n,dtype=float);test=np.empty((m,n),dtype=float)
    for pair_id,pair in expected.items():
        record=records[pair_id]
        if record['pair']!=pair:raise ResultValidationError('Cached pair identity mismatch')
        value=float(record['zero_probability'])
        if not math.isfinite(value) or not 0<=value<=1:raise ResultValidationError('Invalid overlap probability')
        i,j=pair['i'],pair['j']
        if pair['kind']=='train':train[i,j]=train[j,i]=value
        else:test[i,j]=value
    return train,test
def center_kernels(train,test):
    columns=train.mean(axis=0,keepdims=True);grand=train.mean()
    return train-train.mean(axis=1,keepdims=True)-columns+grand,test-test.mean(axis=1,keepdims=True)-columns+grand
def clinical(labels,predictions,malignant_probability):
    truth=np.asarray(labels)==0;pred=np.asarray(predictions)==0
    tn,fp,fn,tp=confusion_matrix(truth,pred,labels=[False,True]).ravel()
    return {'n':len(truth),'accuracy':float(accuracy_score(truth,pred)),'correct':int(tp+tn),
        'roc_auc':float(roc_auc_score(truth,malignant_probability)),'average_precision':float(average_precision_score(truth,malignant_probability)),
        'sensitivity':float(tp/(tp+fn)),'specificity':float(tn/(tn+fp)),
        'precision':float(precision_score(truth,pred,zero_division=0)),'f1':float(f1_score(truth,pred,zero_division=0)),
        'TP':int(tp),'FN':int(fn),'FP':int(fp),'TN':int(tn)}
def wilson(correct,n):
    z=1.959963984540054;p=correct/n;den=1+z*z/n;middle=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [middle-half,middle+half]
def score(directory,p,K_train,K_test,train_labels,test_labels,test_ids):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    eig=np.linalg.eigvalsh((K_train+K_train.T)/2);spectrum={'minimum_eigenvalue':float(eig.min()),'negative_eigenvalues_below_minus_1e-10':int(np.sum(eig< -1e-10)),'PSD_repair_applied':False}
    if p['kernel']['psd_repair'] or p['kernel']['diagonal_normalization']:raise ValueError('Optional repair/normalization has no frozen out-of-sample policy; refusing')
    original_train=K_train.copy();original_test=K_test.copy()
    if p['kernel']['center']:K_train,K_test=center_kernels(K_train,K_test)
    model=SVC(**p['svc'])
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always');model.fit(K_train,train_labels)
    assert model.classes_.tolist()==[0,1]
    atomic(directory/'svc_fit_diagnostics.json',{'fit_status':int(model.fit_status_),'n_iter':model.n_iter_.tolist(),
        'n_support_per_class':model.n_support_.tolist(),'warnings':[{'category':w.category.__name__,'message':str(w.message)} for w in captured],
        'interpretation':'Successful termination is not a proof of a global optimum for an indefinite kernel or of generalization.'})
    predictions=model.predict(K_test);probabilities=model.predict_proba(K_test);index=int(np.flatnonzero(model.classes_==0)[0]);scores=-model.decision_function(K_test)
    metrics=clinical(test_labels,predictions,probabilities[:,index]);metrics['decision_score_auc']=float(roc_auc_score(np.asarray(test_labels)==0,scores))
    metrics['accuracy_wilson95_descriptive']=wilson(metrics['correct'],metrics['n'])
    metrics['uncertainty_note']='24-case feasibility uncertainty; not independent validation or uncertainty over the prior configuration selection.'
    np.savez_compressed(directory/'kernels_predictions.npz',kernel_train=original_train,kernel_test=original_test,test_ids=test_ids,source_labels=test_labels,predictions=predictions,malignant_probabilities=probabilities[:,index],malignant_decision_scores=scores,classes=model.classes_)
    with (directory/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['source_row_id','source_label','prediction','malignant_probability','malignant_decision_score'])
        w.writerows(zip(test_ids,test_labels,predictions,probabilities[:,index],scores))
    atomic(directory/'metrics.json',metrics);atomic(directory/'kernel_spectrum.json',spectrum)
    import joblib
    joblib.dump(model,directory/'svc.joblib')
    saved=np.load(directory/'kernels_predictions.npz');recomputed=clinical(saved['source_labels'],saved['predictions'],saved['malignant_probabilities'])
    assert all(metrics[k]==v for k,v in recomputed.items())
    atomic(directory/'verification.json',{'saved_predictions_recomputed':True,'model_classes':[0,1],'malignant_positive':True,'native_prediction_used':True,'no_threshold_optimization':True,'input_kernel_hash':identity({'train':original_train,'test':original_test})})
    return metrics,spectrum
