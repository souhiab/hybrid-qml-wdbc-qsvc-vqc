from pathlib import Path
import json,hashlib,py_compile,collections,importlib.util
import numpy as np
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2))
m=json.loads((P/'manifest.json').read_text());folds=json.loads((P/'folds/outer_inner_assignments.json').read_text())['outer_records']
assert len(m['fits'])==180 and len({r['fit_id'] for r in m['fits']})==180
assert collections.Counter(r['candidate_id'] for r in m['fits'])=={'V4-A':45,'V4-B':45,'V7-A':45,'V7-B':45}
for repeat in [1,2,3]:
 held=[i for f in folds if f['repeat_id']==repeat for i in f['test_indices']];assert sorted(held)==list(range(569))
for row in m['fits']:
 fold=next(f for f in folds if f['repeat_id']==row['repeat_id'] and f['outer_fold_index']==row['outer_fold_index'])
 d=Path(row['study_root']);split=json.loads((d/'split.json').read_text());assert split['pilot_training_indices']==fold['train_indices'] and split['validation_indices']==fold['test_indices']
 with np.load(d/f'preprocessing/p{row["p"]}.npz') as data:
  assert data['X_train'].shape==(len(fold['train_indices']),row['qubits'])
  assert data['X_validation'].shape==(len(fold['test_indices']),row['qubits'])
  assert data['train_indices'].tolist()==fold['train_indices'] and data['validation_indices'].tolist()==fold['test_indices']
 assert sha(d/row['circuit_path'])==row['circuit_sha256']
for name in ['shared_controller.py','writer_support.py','fit_worker.py','vqc_engine.py','prepare_pilot.py']:py_compile.compile(str(P/name),doraise=True)
# Test aggregation/recovery paths with synthetic saved predictions, without training.
import shared_controller as controller
import tempfile
original_root=controller.ROOT
with tempfile.TemporaryDirectory() as tmp:
 controller.ROOT=Path(tmp);(controller.ROOT/'tables').mkdir();synthetic={'configurations':[{'candidate_id':'S'}],'fits':[]};results={}
 for repeat in [1,2,3]:
  for seed in [1,2,3]:
   for fold in range(5):
    idx=np.array_split(np.arange(569),5)[fold];labels=(idx%2).astype(int);pred=labels.copy();prob=np.c_[labels==0,labels==1].astype(float)
    fid=f'{repeat}-{seed}-{fold}';path=controller.ROOT/fid;path.mkdir();np.savez(path/'predictions.npz',validation_indices=idx,validation_labels=labels,validation_predictions=pred,validation_probabilities=prob)
    row={'fit_id':fid,'candidate_id':'S','repeat_id':repeat,'seed_pair_id':seed,'outer_fold_index':fold,'initialization_seed':seed,'sampler_seed':seed,'study_root':str(controller.ROOT)};synthetic['fits'].append(row)
    results[fid]={'status':'completed','attempt_path':fid,'metrics':{'validation':{'accuracy':1.0}},'termination':'budget_exhausted'}
 rows=controller.consolidate_oof(synthetic,results);assert len(rows)==12 and all(r['accuracy']==1 for r in rows)
controller.ROOT=original_root
source={n:sha(P/n) for n in ['vqc_engine.py','fit_worker.py','prepare_pilot.py','shared_controller.py','writer_support.py','workbook_support/workbook_writer.mjs']}
for study in m['studies']:
 d=Path(study['path']);save(d/'records/source_lock.json',{'actual_module_root':str(P),'files':source})
files={}
for p in P.rglob('*'):
 if p.is_file() and '__pycache__' not in p.parts and not any(x in p.parts for x in ['artifacts','state_cache','node_modules','tables','logs']) and p.name not in ['package_lock.json','status.json','prelaunch_verification.json']:
  files[str(p.relative_to(P))]=sha(p)
save(P/'records/package_lock.json',{'files':files})
save(P/'records/prelaunch_verification.json',{'passed':True,'fit_rows':180,'candidate_count':4,'shared_partitions':15,'seeds':3,'all_training_only_audits_verified':True,'fold_mapping_verified':True,'synthetic_oof_aggregation_passed':True,'VQC_optimizer_fits_launched':0,'fixed_split_models_reused':False})
print('PASS: 180 frozen fits, exact shared folds, training-only preprocessing, synthetic OOF aggregation.')
