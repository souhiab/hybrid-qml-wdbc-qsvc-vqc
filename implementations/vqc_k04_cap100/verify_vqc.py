from __future__ import annotations
import json
import hashlib
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from vqc_engine import ROOT,sha,array_sha,atomic_json,clinical
from run_vqc import validate_receipt,make_snapshot

def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    assert len(manifest['configurations'])==64 and len({c['configuration_id'] for c in manifest['configurations']})==64
    assert len(manifest['fits'])==192 and len({r['fit_id'] for r in manifest['fits']})==192
    assert sha(ROOT/'manifest.json')==json.loads((ROOT/'records/manifest_lock.json').read_text())['manifest_sha256']
    results={};errors=[];max_error=0;attempt_counts={};termination={}
    with np.load(ROOT/'source/data/wdbc_arrays.npz',allow_pickle=False) as data:source_y=data['y']
    split=json.loads((ROOT/'source/historical/split_idx.json').read_text())
    assert len(split['train_idx'])==455 and len(split['test_idx'])==114
    assert np.sum(source_y[split['test_idx']]==0)==42 and np.sum(source_y[split['test_idx']]==1)==72
    for row in manifest['fits']:
        result=validate_receipt(row)
        if result is None:continue
        results[row['fit_id']]=result
        for path in (ROOT/'artifacts'/row['fit_id']).glob('attempt_*/attempt.json'):
            state=json.loads(path.read_text()).get('state','unknown');attempt_counts[state]=attempt_counts.get(state,0)+1
        if result['status']!='completed':continue
        termination[result['termination']]=termination.get(result['termination'],0)+1
        attempt=ROOT/result['attempt_path']
        history=[json.loads(line) for line in (attempt/'training_loss.jsonl').read_text().splitlines()]
        assert len(history)==result['objective_evaluations']==result['optimizer_status']['nfev']<=100
        assert [h['evaluation'] for h in history]==list(range(1,len(history)+1))
        assert result['termination']=='budget_exhausted' if result['optimizer_status']['status']==2 else result['termination']=='converged'
        assert history[0]['weights']==row['initial_parameters']
        audit=json.loads((attempt/'fit_audit.json').read_text())
        assert not audit['test_rows_in_any_fit'] and not audit['test_labels_in_any_fit']
        assert audit['test_inputs_loaded_after_fit'] and audit['final_test_probability_calls']==1 and audit['classes']==[0,1]
        with np.load(attempt/'predictions.npz',allow_pickle=False) as arrays:
            for partition in ('train','test'):
                idx=arrays[partition+'_indices'];labels=arrays[partition+'_labels'];probs=arrays[partition+'_probabilities'];pred=arrays[partition+'_predictions']
                np.testing.assert_array_equal(labels,source_y[idx])
                assert len(set(idx.tolist()))==len(idx)
                assert np.allclose(probs*1024,np.round(probs*1024),atol=1e-12)
                metrics=clinical(labels,pred,probs)
                max_error=max(max_error,max(abs(value-result['metrics'][partition][key]) for key,value in metrics.items()))
            np.testing.assert_array_equal(arrays['initial_parameters'],row['initial_parameters'])
            assert array_sha(arrays['initial_parameters'])==row['initial_parameter_sha256']
        csv=pd.read_csv(attempt/'test_predictions.csv')
        assert len(csv)==114 and csv.sample_index.tolist()==split['test_idx']
    snapshot=json.loads((ROOT/'workbook_support/snapshot.json').read_text())
    status=json.loads((ROOT/'status.json').read_text())
    assert status['completed']==sum(r['status']=='completed' for r in results.values())
    for config in snapshot['Configuration_Summary']:
        group=[r for r in results.values() if r['configuration_id']==config['configuration_id'] and r['status']=='completed']
        pairs={r['seed_pair_id'] for r in manifest['fits'] if r['configuration_id']==config['configuration_id']}
        assert pairs=={1,2,3}
        if config['complete']:
            assert len(group)==3
            acc=[r['metrics']['test']['accuracy'] for r in group]
            assert abs(config['mean_accuracy']-np.mean(acc))<1e-12
            assert abs(config['sample_std_accuracy']-np.std(acc,ddof=1))<1e-12
            assert config['min_accuracy']==min(acc) and config['max_accuracy']==max(acc)
            for metric in ('roc_auc','sensitivity','specificity','f1'):
                assert abs(config['mean_'+metric]-np.mean([r['metrics']['test'][metric] for r in group]))<1e-12
        else:
            assert config['mean_accuracy'] is None and not config['accuracy_shortlist']
    workbook=json.loads((ROOT/'records/workbook_readback.json').read_text()) if (ROOT/'records/workbook_readback.json').exists() else {'allPassed':False}
    lock=json.loads((ROOT/'records/production_source_lock.json').read_text())
    scientific_source_unchanged=all(sha(ROOT/file)==digest for file,digest in lock['files'].items() if file!='workbook_support/workbook_writer.mjs')
    workbook_update=json.loads((ROOT/'records/workbook_writer_update.json').read_text())
    workbook_revision_verified=(sha(ROOT/'versions/v2_workbook_writer.mjs')==lock['files']['workbook_support/workbook_writer.mjs']==workbook_update['old_sha256'] and sha(ROOT/'workbook_support/workbook_writer.mjs')==workbook_update['new_sha256'])
    source_unchanged=scientific_source_unchanged and workbook_revision_verified
    report={'verified_at':status['updated'],'expected_configurations':64,'expected_fits':192,**status,'termination_counts':termination,'attempt_state_counts':attempt_counts,'max_metric_recomputation_error':max_error,'ordered_split_verified':True,'test_counts':{'malignant':42,'benign':72},'train_only_fit_audits_verified':True,'malignant_column':0,'finite_shot_probabilities_verified':True,'source_unchanged_since_production_freeze':source_unchanged,'workbook_readback_passed':workbook['allPassed'],'all_outcomes_accounted_for':status['completed']+status['failed']+status['pending']==192,'passed':len(results)==192 and status['pending']==0 and max_error<=1e-12 and source_unchanged and workbook['allPassed']}
    atomic_json(ROOT/'records/final_verification.json',report)
    print(json.dumps(report,indent=2))
    if not report['passed']:raise RuntimeError('Final verification not complete')

if __name__=='__main__':main()
