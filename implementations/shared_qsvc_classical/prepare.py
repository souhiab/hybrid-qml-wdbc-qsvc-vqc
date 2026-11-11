from __future__ import annotations
import ast
import inspect
import itertools
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from qiskit.circuit.library import ZFeatureMap,ZZFeatureMap
from historical.evaluation.dataset import load_wdbc
from historical.evaluation.splits import validate_split_artifact
from historical.evaluation.classical_models import classical_candidate_grid
from common import *

GRID=ROOT.parent/'wdbc_qsvc_scaled_grid_20261107'
REPO=ROOT.parent/'implementations'
SPECS=[(4,'lr','ZZFeatureMap','circular',4),(7,'rf','ZZFeatureMap','circular',3),(11,'lr','ZFeatureMap',None,1),(11,'lr','ZFeatureMap',None,2),(11,'lr','ZZFeatureMap','linear',4),(17,'lr','ZFeatureMap',None,1),(17,'lr','ZFeatureMap',None,2),(17,'rf','ZFeatureMap',None,1),(4,'lr','ZFeatureMap',None,1),(7,'rf','ZFeatureMap',None,1)]

def verify_folds():
    bundle=load_wdbc();X,y,names=dataset()
    np.testing.assert_array_equal(X,bundle.X.to_numpy());np.testing.assert_array_equal(y,bundle.y.to_numpy());assert names.tolist()==list(bundle.feature_names)
    folds=validate_split_artifact(ROOT/'folds/outer_inner_assignments.json',ROOT/'evidence/historical_fold_receipt.json',bundle=bundle)
    # Exact reconstruction and patient coverage checked by historical validator.
    for record in folds['outer_records']:
        assert set(record['train_indices']).isdisjoint(record['test_indices'])
        for inner in record['inner_folds']:assert not (set(inner['train_indices'])|set(inner['validation_indices']))&set(record['test_indices'])
    return folds,bundle.identity()

def main():
    if (ROOT/'candidates.json').exists():raise RuntimeError('Preparation already frozen; use verify')
    folds,data_identity=verify_folds()
    source=pd.read_csv(ROOT/'evidence/qsvc_completed_grid.csv').fillna('')
    execution=pd.read_csv(ROOT/'evidence/qsvc_execution_manifest.csv').fillna('')
    candidates=[]
    for number,(k,selector,constructor,topology,reps) in enumerate(SPECS,1):
        estimator='Logistic Regression' if selector=='lr' else 'Random Forest'
        matches=source[(source.width==k)&(source.scaler=='MinMaxScaler')&(source.rfe_estimator==estimator)&(source.execution_constructor==constructor)&(source.topology==(topology or ''))&(source.repetitions==reps)]
        assert len(matches)==1,(number,'exact archived match missing')
        old=matches.iloc[0].to_dict();configuration=execution[execution.configuration_id==old['configuration_id']].iloc[0].to_dict()
        assert old['verification_status']=='verified'
        artifact=GRID/old['artifact_path'];assert artifact.is_dir()
        prep_audit=GRID/f'preprocessing/fs-k{k:02d}-minmax-{selector}.audit.json'
        prep=read(prep_audit);assert prep['rfe_settings']['step']==1 and prep['rfe_fit_rows']==455 and not prep['test_indices_in_fit']
        params=LogisticRegression(penalty='l1',solver='saga',max_iter=5000,random_state=42).get_params() if selector=='lr' else RandomForestClassifier(random_state=42).get_params()
        # Unfitted constructors recover their installed defaults, not learned data.
        model_options=SVC().get_params();model_options.update({'C':1.0,'probability':True,'class_weight':None,'shrinking':True,'tol':.001,'kernel':'verified analytic squared-overlap callback','random_state':'seed_for(evaluation_id,libsvm)'})
        fm=ZFeatureMap(k,reps=reps) if constructor=='ZFeatureMap' else ZZFeatureMap(k,reps=reps,entanglement=topology)
        candidate={'candidate_id':f'Q{number}','role':'selected accuracy leader' if number<=8 else 'shallow control','k':k,'scaler':'MinMaxScaler','scaler_params':MinMaxScaler().get_params(),'selector':'rfe_'+selector,'rfe_step':1,'rfe_estimator_params':params,'feature_map':constructor,'entanglement':topology,'feature_map_reps':reps,'feature_map_alpha':float(fm.alpha),'feature_map_input_order':[str(p) for p in fm.parameters],'data_map_function':'Qiskit default self_product; one-body x_i, two-body (pi-x_i)*(pi-x_j), alpha=2','C':1.0,'svc_options':model_options,'kernel':json.loads(configuration['kernel_settings']),'statevector_dtype':'complex128','kernel_dtype':'float64','enforce_psd_semantics':'Historical callback uses exact Gram, unit diagonal, block symmetry; no numerical PSD projection. Analytic fidelity is PSD mathematically. Preserve this rather than adding a projection.','archived_configuration_id':old['configuration_id'],'archived_calibration_seed':old['calibration_seed'],'archived_artifact_path':str(artifact),'archived_artifact_config_sha256':sha(artifact/'configuration.json') if (artifact/'configuration.json').exists() else None,'evidence':{'completed_grid':{'path':str(GRID/'tables/fixed_split_results.csv'),'sha256':sha(GRID/'tables/fixed_split_results.csv')},'preprocessing_audit':{'path':str(prep_audit),'sha256':sha(prep_audit)},'training_code':{'path':str(GRID/'run_scaled_grid.py'),'sha256':sha(GRID/'run_scaled_grid.py')}}}
        assert candidate['kernel']['analytic'] and candidate['kernel']['shots'] is None
        candidates.append(candidate)
    atomic(ROOT/'candidates.json',candidates)
    pd.DataFrame([{key:c[key] for key in ['candidate_id','role','k','scaler','selector','feature_map','entanglement','feature_map_reps','C','archived_configuration_id','archived_calibration_seed','archived_artifact_path']} for c in candidates]).to_csv(ROOT/'tables/Candidates.csv',index=False)
    headings=['candidate_id','role','k','scaler','selector','feature_map','entanglement','feature_map_reps','C','archived_configuration_id']
    lines=['| '+' | '.join(headings)+' |','| '+' | '.join(['---']*len(headings))+' |']
    lines.extend('| '+' | '.join(str(c.get(h) if c.get(h) is not None else 'not applicable') for h in headings)+' |' for c in candidates)
    (ROOT/'CANDIDATES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    matched=[{'candidate_id':f'M{i}','family':'matched_rbf_svm','k':k,'scaler':'minmax','selector':selector,'preprocessing_regime':'qsvc_grid','C_values':[.1,1.,10.],'gamma_values':['scale',.01,.1],'probability':False,'inner_folds':3,'inner_auc_tolerance':.005,'outer_evaluations':15,'inner_model_fits_per_outer':27,'comparisons':partners} for i,(k,selector,partners) in enumerate([(4,'rfe_lr',['Q1','Q9']),(7,'rfe_rf',['Q2','Q10']),(11,'rfe_lr',['Q3','Q4','Q5']),(17,'rfe_lr',['Q6','Q7']),(17,'rfe_rf',['Q8'])],1)]
    strong=[]
    for cid,family in [('LR','logistic_regression'),('RBF','rbf_svm')]:
        grid=[row for row in classical_candidate_grid() if row['model_family']==family]
        strong.append({'candidate_id':cid,'family':'strong_'+family,'search_space':grid,'search_candidates':len(grid),'inner_folds':3,'inner_model_fits_per_outer':len(grid)*3,'preprocessing_regime':'historical_strong','selection':'Mean inner malignant AUC; eligible within 0.005 of best; ascending k, selector order select_k_best/pca/rfe_lr/rfe_rf, scaler standard/minmax, C, gamma scale/.01/.1, canonical configuration','default_decision':'malignant probability >=0.5' if cid=='LR' else 'malignant decision score >=0','tie_positive_class':0,'historical_secondary_threshold':'Inner Youden rule existed as a secondary analysis; not used for primary predictions here. Stored legacy secondary predictions are not reused as default results.','source':['historical/evaluation/config.py','historical/evaluation/classical_models.py','historical/evaluation/preprocessing.py'],'outer_evaluations':15})
    atomic(ROOT/'comparators.json',{'strong':strong,'matched':matched})
    protocol={'preparation_state':'READY_NOT_STARTED','dataset':data_identity,'evaluation':'Internal repeated evaluation on WDBC: shortlist informed by exploration of the same dataset; no independent validation or corrected maxima claims','outer_seeds':folds['outer_seeds'],'outer_partitions':15,'inner_folds':3,'inner_seed_rule':'outer_seed + zero-based outer_fold_index + 1000','qs_candidates':10,'qs_outer_fits':150,'matched_procedures':5,'matched_outer_evaluations':75,'matched_inner_model_fits':2025,'strong_outer_evaluations':30,'strong_inner_model_fits_if_no_reuse':21600,'fit_scope':'Raw outer-training cases only; each inner fold refits all learned preprocessing on its own training cases and training labels. Outer held-out rows never enter fitting, tuning or calibration.','preprocessing_reuse':'Only when ordered training/transform IDs, X/y hashes, all estimator/scaler/selector settings and provenance match; Q and matched controls share qsvc-grid preprocessing, not strong-procedure RF500 preprocessing. No fixed-split cache/model reuse.','calibration_seed_rule':'int.from_bytes(SHA256(UTF8(20261108|evaluation_id|libsvm))[:4],big) mod (2**31-1); RFE randomness remains historical 42 for Q/matched, 20261009 for strong.','probability_calibration':'QSVC probability=True: installed libsvm training-only internal calibration; seed frozen before fit. No held-out calibration or thresholds. Historical uncontrolled_None cases explicitly recorded; new repeated evaluation intentionally controls all calibration randomness.','positive_class':'malignant, original source label 0','qsvc_decision':'Preserve estimator.predict and installed libsvm voting/tie semantics; ranking metrics use probability column identified by classes_ == 0. Do not replace decisions by a calibrated probability threshold.','classical_decision':'LR malignant probability >=0.5; RBF malignant-oriented decision score >=0; equality malignant. No outer-label threshold tuning.','metrics':['roc_auc','average_precision','accuracy','sensitivity','specificity','precision','f1','TP','FN','FP','TN'],'undefined_ratio':0.0,'aggregation':'Report arithmetic fold means separately from per-repeat pooled OOF metrics over exactly 569 patients; report mean and sample SD across three repeat summaries. A pooled OOF AUC mixes separately fitted fold score scales and is not the fold-mean AUC. Never pool 1707 repeated records as independent patients.','paired_comparisons':'All Q1-Q10 versus LR and RBF plus each Q versus its prespecified matched M control. Align exact patient IDs, repeats/folds and truth labels.','uncertainty':'Descriptive paired patient bootstrap: 10000 draws, fixed seed 20261108; same resampled patient IDs across models and all repeats, panel/folds fixed; recompute each repeat pooled OOF metrics and average repeat differences. Excludes training-resampling, feature/candidate selection, and full dependence induced by overlapping training folds. No naive independent-samples 15-fold tests; no formal multiple-comparison or equivalence claims.','training_authorization':'PREPARATION ONLY in this task. Explicit future CLI execute command required; importing or --help never fits.','vqc':{'shortlist_state':'PENDING','candidates':[],'seed_pairs':[[101,1101],[211,1211],[307,1307]],'integration':'Register a later frozen VQC shortlist with these same saved folds and result schema; no QSVC-to-VQC winner assumption; registration does not train.'},'resources':{'default_workers':1,'threads_per_worker':2,'max_workers':2,'system_ram_reserve_bytes':2147483648,'kernel_block_size':64,'heldout_statevector_batch':4,'statevector_precision':'complex128','kernel_precision':'float64','cache_limit_bytes':2147483648,'active_vqc_policy':'Future training refuses concurrent execution with the active VQC controller by default. No training launched during preparation.'},'optimizations':['Verified vectorized exact Z/ZZ simulator extracted verbatim','Symmetric blocked analytic train kernel; stream held-out rows after QSVC fit','Fold/settings/content-bound preprocessing and kernel reuse','No duplicate verified evaluations; interruption-safe attempts','Single asynchronous Excel writer with bounded queue','Classical preprocessing reused across classifier hyperparameters within each inner training fold'],'unresolved_settings':[]}
    atomic(ROOT/'protocol.json',protocol)
    atomic(ROOT/'vqc_candidates.json',protocol['vqc'])
    # Extract only the verified numerical helpers. No historical runner dispatch.
    text=(ROOT/'evidence/qsvc_run_scaled_grid.py').read_text(encoding='utf-8');tree=ast.parse(text)
    selected=['array_hash','build_feature_map','computational_basis_bits','normalized_fwht_inplace','exact_feature_map_statevector','blocked_fidelity','CachedKernel']
    chunks=[]
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in selected:
            if node.decorator_list:
                chunks.append('\n'.join(text.splitlines()[node.decorator_list[0].lineno-1:node.end_lineno]))
            else:chunks.append(ast.get_source_segment(text,node))
    header='from __future__ import annotations\nimport hashlib,time\nfrom functools import lru_cache\nimport numpy as np\nfrom qiskit.circuit.library import ZFeatureMap,ZZFeatureMap\nBLOCK_SIZE=64\n\n'
    (ROOT/'quantum_engine.py').write_text(header+'\n\n'.join(chunks)+'\n',encoding='utf-8')
    foldtable=[{'fold_id':fold_id(r),'repeat_id':r['repeat_id'],'outer_fold_index':r['outer_fold_index'],'seed':r['outer_seed'],'train_n':len(r['train_indices']),'heldout_n':len(r['test_indices']),'train_malignant':r['train_counts']['malignant_0'],'test_malignant':r['test_counts']['malignant_0'],'train_hash':r['train_indices_sha256'],'test_hash':r['test_indices_sha256']} for r in folds['outer_records']]
    pd.DataFrame(foldtable).to_csv(ROOT/'tables/Folds.csv',index=False)
    atomic(ROOT/'records/fold_verification.json',{'verified':True,'source':'existing exact historical manifest and receipt','source_file_sha256':sha(ROOT/'folds/outer_inner_assignments.json'),'new_folds_generated':False,'patient_id_definition':'Stable sklearn row ordinal 0..568, not a newly invented clinical ID','rows':569,'outer_partitions':15,'each_patient_heldout_once_per_repeat':True,'all_inner_scopes_inside_outer_training':True})
    provenance={'repository':'https://github.com/souhiab/hybrid-qml-wdbc-qsvc-vqc','commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'starting_git_status':subprocess.check_output(['git','status','--porcelain=v1'],cwd=REPO,text=True),'source_grid':str(GRID),'source_grid_completion_sha256':sha(GRID/'records/final_completion_verification.json'),'historical_fold_artifact_sha256':sha(ROOT/'folds/outer_inner_assignments.json'),'prepared_at':now()}
    atomic(ROOT/'records/provenance.json',provenance);atomic(ROOT/'records/environment.json',environment())
    atomic(ROOT/'status.json',{'state':'PREPARATION_INCOMPLETE','WDBC_training_launched':False,'production_kernels_generated':0,'blockers':['Finish reuse audit, synthetic tests, workbook and source freeze']})
    print('Ten exact candidates and 15 historical shared folds verified; no WDBC fitting performed.')

if __name__=='__main__':main()
