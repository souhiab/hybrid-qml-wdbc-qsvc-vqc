from __future__ import annotations
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from common import *
from historical.evaluation.dataset import load_wdbc
from historical.evaluation.preprocessing import indexed_object_hash
from historical.evaluation.classical_models import select_auc_tolerant

SOURCES={'LR':Path('D:/lr3_261014_01'),'RBF':Path('D:/sv3_261014_01')}
EXPECTED={'LR':'08b36cf014873aa923455b7ff5a10a46455135b251b98484a641ef2c385fa50b','RBF':'849dafd7eeaaa5545acd711a30c09dfca95b66873231a34d0a955ab352f7fd0a'}

def audit_one(cid,source,checkpoint,fold,bundle):
    cp=read(checkpoint)
    assert cp['status']=='complete' and cp['configuration_sha256']==EXPECTED[cid]
    assert cp['split_sha256']==sha(ROOT/'folds/outer_inner_assignments.json')
    assert cp['key']['repeat_id']==fold['repeat_id'] and cp['key']['outer_fold_index']==fold['outer_fold_index']
    files={}
    for record in cp['artifact_receipts']:
        file=source/record['path'];assert sha(file)==record['sha256'],str(file)
        files[record['path'].split('/')[0]]=file
    required={'predictions','results','decision_evidence','features','timing'};assert required.issubset(files)
    evidence=read(files['decision_evidence'])
    train=fold['train_indices'];test=fold['test_indices']
    assert evidence['refit_audit']['fit_sample_indices']==train
    assert evidence['refit_audit']['feature_identity_sha256']==indexed_object_hash(bundle.X.loc[train])
    assert evidence['refit_audit']['target_identity_sha256']==indexed_object_hash(bundle.y.loc[train])
    expected_logs=[]
    for inner in fold['inner_folds']:expected_logs.extend([('inner_fit',inner['train_indices']),('inner_validation',inner['validation_indices'])])
    expected_logs.extend([('outer_training_refit',train),('outer_test_prediction',test)])
    assert [(r['purpose'],r['sample_indices']) for r in evidence['access_log']]==expected_logs
    assert set(evidence['selection']['evidence_sample_indices'])==set(train)
    # Every saved inner validation vector must use the frozen patients/labels.
    for row in evidence['inner_evidence']:
        inner=fold['inner_folds'][row['inner_fold_index']]
        ids=row.get('sample_indices',row.get('validation_indices'))
        assert ids==inner['validation_indices']
        truth=row.get('true_source_labels')
        np.testing.assert_array_equal(truth,bundle.y.loc[ids].to_numpy())
        assert len(row['malignant_scores'])==len(ids) and np.isfinite(row['malignant_scores']).all()
    records=[{**row['configuration'],'configuration':row['configuration'],'auc_mean':row['auc_mean']} for row in evidence['selection']['candidate_evaluations']]
    assert len(records)==(120 if cid=='LR' else 360)
    assert select_auc_tolerant(records).selected['configuration']==evidence['selection']['configuration']
    predictions=pd.read_csv(files['predictions'])
    assert predictions.sample_index.tolist()==test
    np.testing.assert_array_equal(predictions.true_source_label,bundle.y.loc[test].to_numpy())
    assert set(predictions.repeat_id)=={fold['repeat_id']} and set(predictions.outer_fold_index)=={fold['outer_fold_index']}
    threshold=.5 if cid=='LR' else 0.0
    assert set(predictions.default_threshold)=={threshold}
    assert set(predictions.score_type)==({'malignant_probability'} if cid=='LR' else {'malignant_decision_score'})
    np.testing.assert_array_equal(predictions.default_prediction,np.where(predictions.malignant_score>=threshold,0,1))
    recalculated=metrics(predictions.true_source_label,predictions.default_prediction,predictions.malignant_score)
    old=pd.read_csv(files['results']).iloc[0].to_dict()
    for metric in ['roc_auc','average_precision','accuracy','sensitivity','specificity','precision','f1']:
        assert abs(recalculated[metric]-float(old[metric]))<1e-12
    evaluation_id=f'{cid}-{fold_id(fold)}';dest=ROOT/'reused_predictions'/evaluation_id;dest.mkdir(exist_ok=True)
    canonical_predictions=pd.DataFrame({'candidate_id':cid,'evaluation_id':evaluation_id,'repeat_id':fold['repeat_id'],'outer_fold_index':fold['outer_fold_index'],'sample_index':predictions.sample_index,'true_source_label':predictions.true_source_label,'prediction':predictions.default_prediction,'malignant_score':predictions.malignant_score,'score_type':predictions.score_type})
    canonical_predictions.to_csv(dest/'predictions.csv',index=False)
    supporting=[]
    for kind,file in files.items():
        target=dest/(('source_predictions' if kind=='predictions' else kind)+file.suffix);shutil.copy2(file,target)
        supporting.append({'source':str(file),'copied':target.name,'sha256':sha(target)})
    atomic(dest/'verification.json',{'verified':True,'folds_and_order':True,'full_learning_procedure':True,'training_local_preprocessing_provenance':True,'inner_selection_scope':True,'class_orientation_and_decision':True,'metric_max_error':0.0})
    receipt={'status':'completed_reused','candidate_id':cid,'evaluation_id':evaluation_id,'repeat_id':fold['repeat_id'],'outer_fold_index':fold['outer_fold_index'],'train_indices':train,'test_indices':test,'metrics':recalculated,'score_type':'malignant_probability' if cid=='LR' else 'malignant_decision_score','source_checkpoint':str(checkpoint),'source_checkpoint_sha256':sha(checkpoint),'procedure_configuration_sha256':cp['configuration_sha256'],'source_artifacts':supporting,'attempt_path':str(dest.relative_to(ROOT)),'runtime_seconds':None,'reused':True,'file_hashes':{f.name:sha(f) for f in dest.iterdir() if f.is_file() and f.name!='receipt.json'}}
    atomic(dest/'receipt.json',receipt)
    return {'candidate_id':cid,'repeat_id':fold['repeat_id'],'outer_fold_index':fold['outer_fold_index'],'reusable':True,'reason':'Exact folds/data, all artifact hashes, 120/360-candidate inner procedure, training-local refit/access scope, ordered predictions, labels, orientation and locked default decision verified','source':str(checkpoint)}

def run():
    bundle=load_wdbc();folds=read(ROOT/'folds/outer_inner_assignments.json')['outer_records'];rows=[]
    for cid,source in SOURCES.items():
        checkpoints=list((source/'checkpoints').glob('*.json'))
        for fold in folds:
            try:
                cp=next(p for p in checkpoints if read(p)['key']['repeat_id']==fold['repeat_id'] and read(p)['key']['outer_fold_index']==fold['outer_fold_index'])
                rows.append(audit_one(cid,source,cp,fold,bundle))
            except Exception as error:
                rows.append({'candidate_id':cid,'repeat_id':fold['repeat_id'],'outer_fold_index':fold['outer_fold_index'],'reusable':False,'reason':f'{type(error).__name__}: {error}','source':str(source)})
    for cid in ['M1','M2','M3','M4','M5']:rows.append({'candidate_id':cid,'repeat_id':'all','outer_fold_index':'all','reusable':False,'reason':'Historical fixed-split feature subsets/scores are not these training-local repeated-fold matched procedures. New nested matched evaluations required.','source':'historical fixed-split classical tables'})
    rows.append({'candidate_id':'historical_fixed_split_QSVC','repeat_id':'all','outer_fold_index':'all','reusable':False,'reason':'455/114 single split differs from the frozen 15 outer partitions; used only to establish fixed candidate definitions.','source':'completed QSVC grid'})
    atomic(ROOT/'prediction_reuse_audit.json',rows);pd.DataFrame(rows).to_csv(ROOT/'tables/Reuse_Audit.csv',index=False)
    print(f"Reuse audit: {sum(row['reusable'] for row in rows)} verified historical outer-fold results reusable; remaining procedures require future execution.")
    return rows

if __name__=='__main__':run()
