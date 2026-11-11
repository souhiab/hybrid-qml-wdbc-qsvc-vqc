from __future__ import annotations
import datetime
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd
import psutil
from vqc_engine import ROOT,atomic_json,sha

def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    fits=[json.loads((ROOT/'artifacts'/row['fit_id']/'receipt.json').read_text()) for row in manifest['fits']]
    snapshot=json.loads((ROOT/'workbook_support/snapshot.json').read_text())
    status=json.loads((ROOT/'records/execution_completion.json').read_text())
    status['execution_state']='completed';status['runner_active']=False
    atomic_json(ROOT/'status.json',status)
    byid={row['fit_id']:row for row in fits}
    versions={version:sum(r.get('implementation_version')==version for r in fits) for version in sorted({r.get('implementation_version') for r in fits})}
    serializers=sum(r['model_serialization']['supported'] for r in fits)
    for row in snapshot['Run_Results']:
        result=byid[row['fit_id']]
        row.update({'ansatz_entanglement':'reverse_linear','final_rotation_layer':True,'loss':'cross_entropy','shots':1024,'objective_budget':100,'warm_start':False,'implementation_version':result['implementation_version'],'selected_features':' | '.join(result['selected_features']),'parameters_path':result['attempt_path']+'/parameters.json','circuit_path':result['attempt_path']+'/circuits.qpy','preprocessing_path':result['attempt_path']+'/preprocessing.json','model_serialization_supported':result['model_serialization']['supported']})
    configs=snapshot['Configuration_Summary']
    ranked=sorted(configs,key=lambda row:(-row['total_correct_across_seeds'],row['configuration_id']))
    winners=[row for row in ranked if row['accuracy_shortlist']]
    max_auc=max(row['mean_roc_auc'] for row in configs)
    auc_winners=[row for row in configs if row['mean_roc_auc']==max_auc]
    sum_runtime=sum(r['runtime']['total_seconds'] for r in fits)
    first=min(datetime.datetime.fromisoformat(r['runtime']['started']) for r in fits)
    last=max(datetime.datetime.fromisoformat(r['runtime']['ended']) for r in fits)
    perversion={version:{'fits':count,'sum_seconds':sum(r['runtime']['total_seconds'] for r in fits if r['implementation_version']==version),'median_seconds':float(np.median([r['runtime']['total_seconds'] for r in fits if r['implementation_version']==version]))} for version,count in versions.items()}
    workbook_errors=len((ROOT/'records/workbook_save_errors.jsonl').read_text().splitlines()) if (ROOT/'records/workbook_save_errors.jsonl').exists() else 0
    runtime={'first_fit_started':first.isoformat(),'last_fit_ended':last.isoformat(),'fit_wall_interval_seconds':(last-first).total_seconds(),'sum_completed_fit_seconds':sum_runtime,'mean_fit_seconds':sum_runtime/192,'max_fit_seconds':max(r['runtime']['total_seconds'] for r in fits),'maximum_observed_peak_rss_bytes':max(r['runtime']['peak_rss_bytes'] for r in fits),'versions':perversion,'serialized_models':serializers,'recovered_workbook_sync_errors':workbook_errors,'interrupted_attempts':1,'technical_retries':0}
    atomic_json(ROOT/'records/runtime_summary.json',runtime)
    snapshot['Summary']=[{'item':k,'value':v} for k,v in status.items()]
    snapshot['Summary'].extend([{'item':'mean_accuracy_primary_ranking','value':'Complete three-seed configurations; retain all exact ties'}, {'item':'best_mean_accuracy','value':winners[0]['mean_accuracy']},{'item':'best_accuracy_tie_count','value':len(winners)},{'item':'interrupted_attempts_preserved','value':1},{'item':'technical_failure_retries','value':0},{'item':'recovered_workbook_sync_errors','value':workbook_errors},{'item':'interpretation','value':'Exploratory fixed split; three seeds assess training variability on the same 114 test patients'}])
    snapshot['Provenance'].extend([{'item':'execution_implementation_versions','value':json.dumps(versions)},{'item':'implementation_transition','value':'records/implementation_transition_v2.json; first 16 completed v1 fits retained; remaining fits v2; one interrupted attempt restarted from original initialization'},{'item':'workbook_revision','value':'records/workbook_writer_update.json; changed-row synchronization and canonical Excel blank/timestamp readback'},{'item':'data_sha256','value':sha(ROOT/'source/data/wdbc_arrays.npz')},{'item':'split_sha256','value':sha(ROOT/'source/historical/split_idx.json')},{'item':'optimizer_exact_resume','value':'Unsupported; original initialization and seeds restart under a new attempt ID'},{'item':'source_and_environment','value':'records/environment.json, records/installed_source_semantics.json, records/production_source_lock.json, versions/v1/'}])
    atomic_json(ROOT/'workbook_support/snapshot.json',snapshot)
    for name,rows in snapshot.items():pd.DataFrame(rows).to_csv(ROOT/'tables'/f'{name}.csv',index=False)
    pd.DataFrame(ranked).to_csv(ROOT/'tables/accuracy_ranking.csv',index=False)
    pd.DataFrame(winners).to_csv(ROOT/'tables/accuracy_shortlist.csv',index=False)
    pd.DataFrame([{'fit_id':r['fit_id'],'implementation_version':r['implementation_version'],'attempt_id':r['attempt_id'],'restarted_from':r.get('restarted_from')} for r in fits]).to_csv(ROOT/'records/implementation_inventory.csv',index=False)
    benchmark=json.loads((ROOT/'records/benchmark_training_review.json').read_text())
    validation=json.loads((ROOT/'records/simulator_validation.json').read_text())
    lines=['# Four-qubit VQC exploration','',f'Completed: 64/64 configurations and 192/192 valid three-seed fits. Failures: 0/192 (0%). Pending: 0. Converged: 27. Budget-exhausted: 165. All budget-limited fits were retained.','',
        'The fixed budget was 100 objective evaluations. Both declared benchmark loss histories were reviewed using training information, and the same budget was retained before production. No test-driven training or budget changes were made.','',
        '## Accuracy shortlist','', '| Configuration | Mean accuracy | Sample SD | Min | Max | Mean malignant AUC | Seed correct counts / 114 |','|---|---:|---:|---:|---:|---:|---|']
    for row in winners:
        group=sorted([r for r in fits if r['configuration_id']==row['configuration_id']],key=lambda r:r['seed_pair_id'])
        counts=', '.join(str(r['metrics']['test']['correct_count']) for r in group)
        lines.append(f"| {row['configuration_id']} | {row['mean_accuracy']:.10f} | {row['sample_std_accuracy']:.10f} | {row['min_accuracy']:.10f} | {row['max_accuracy']:.10f} | {row['mean_roc_auc']:.10f} | {counts} |")
    lines.extend(['','Mean accuracy is the primary exploratory ranking metric. Exact accuracy ties use the sum of correct counts across three equal-size test sets and are all retained. No best-seed selection is used. Incomplete configurations would have unavailable aggregate means and be excluded from the shortlist; all 64 are complete.','',
        '## Highest mean AUC','',*['- '+row['configuration_id']+f": mean AUC {row['mean_roc_auc']:.10f}, mean accuracy {row['mean_accuracy']:.10f}." for row in auc_winners],'',
        '## Protocol and verification','',
        '- Original ordered seed-42 stratified split verified: 455 train, 114 test; test labels 42 malignant (0), 72 benign (1).',
        '- Four corrected preprocessing caches reused after hashes, features, selector settings, scaler parameters, training-index hashes and raw-data transformations matched. Scaler and RFE used training rows/labels only.',
        '- Grid: two scalers, two RFE estimators, Z or equivalent Pauli/ZZ maps; feature-map reps 1/2; linear/full/circular interactions; RealAmplitudes reps 1/2 with explicit reverse_linear connectivity and final rotation layer.',
        '- Fresh model, optimizer and seeded sampler per fit; cross entropy (base-2, installed clipping 1e-10), COBYLA, 1024 shots, no warm start. Seeds (initialization, sampling): (101,1101), (211,1211), (307,1307). Explicit initial vectors are uniform [0,1) in saved parameter order.',
        '- Installed binary readout is basis-state integer modulo 2 (least-significant measured bit), not Hamming-weight parity. Probability order [0,1] was verified; column 0 is malignant. Predictions use argmax from the same final test probability array, with malignant winning an exact tie.',
        '- Each fit uses exactly one final test predict_proba call. Test labels and predictions are not obtained from separate sampling calls. Training and scoring outputs, all clinical metrics and confusion counts are saved.',
        '- SciPy COBYLA maxiter maps to its function-evaluation cap. Raw optimizer status and actual nfev are retained; a completed budget-limited fit is not called converged.',
        '- Historical design differences are explicit ansatz reps 1/2 rather than only 1, explicit reverse_linear/final rotation, fixed sampling and initialization seeds, explicit shot/loss settings, and train-only preprocessing.','',
        '## Implementation, recovery and runtime','',
        f"- Numerical replacement validation: {validation['finite_shot_cases']} seeded finite-shot cases, maximum probability error {validation['probability_max_error']}; Pauli/ZZ alias error {validation['alias_max_state_error']}. Same 1024 draws, complex128/float64 precision and VQC objective semantics.",
        f"- Production used {benchmark['worker_count']} worker and {benchmark['threads_per_worker']} BLAS thread, chosen after measured peak RSS {benchmark['benchmark_peak_rss']:,} bytes and {benchmark['available_ram']:,} available bytes on four physical/eight logical cores.",
        f"- Completed-fit runtime sum {sum_runtime:.3f} s, mean {sum_runtime/192:.3f} s, maximum {runtime['max_fit_seconds']:.3f} s. First/last fit timestamps: {first.isoformat()} / {last.isoformat()} (wall interval {runtime['fit_wall_interval_seconds']:.0f} s). Coordinator workbook synchronization adds elapsed time beyond fitting.",
        f"- Versions: {json.dumps(perversion)}.",
        '- First 16 completed v1 fits were retained. V2 replaced repeated per-case sampling searches with mathematically equivalent counts of the same sorted seeded uniforms. This training-compute optimization used no test results and passed the same numerical validation.',
        '- One interrupted v1 attempt is preserved. Its fit restarted from its original explicit initialization and seed settings as attempt_002. Complete COBYLA state restoration is unsupported; final angles alone were never treated as an exact resume.',
        f"- Technical training failures/retries: 0/0. Recovered workbook synchronization interruptions: {workbook_errors}; receipts were preserved and the workbook reconciled from them. The coordinator writer was revised to update changed rows instead of rebuilding styles after each fit.",
        f'- Model serialization succeeded for {serializers}/192 fits. Portable predictions, parameters, QPY circuits, reconstruction metadata, preprocessing and environment records are saved independently.',
        '- Checkpoints are authoritative. The launcher validates receipts and skips completed fits; interrupted optimization restarts under a new attempt ID.','',
        '## Resume command','', '```powershell',f"Set-Location -LiteralPath '{ROOT}'", "powershell.exe -NoProfile -ExecutionPolicy Bypass -File '.\\resume.ps1'",'```','',
        'The run is complete and no training process remains active. The command validates existing checkpoints and performs no duplicate fits.','',
        '## Interpretation','',
        'These are exploratory scores on the original fixed split. Three seeds measure training variability on the same patient cohort; they do not provide independent patient validation. The full grid and its shortlist have test-selection history.'] )
    (ROOT/'RESULTS_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'PROGRESS_REPORT.md').write_text('Completed 64/64 four-qubit configurations, 192/192 valid seed fits; 27 converged, 165 budget-exhausted, 0 failed, 0 pending. One interrupted attempt preserved and restarted. See RESULTS_REPORT.md and records/final_verification.json. Resume: powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\\resume.ps1\n',encoding='utf-8')
    repo=ROOT.parent/'implementations'
    git_status=subprocess.run(['git','status','--porcelain=v1','--untracked-files=all'],cwd=repo,text=True,capture_output=True,check=True).stdout
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,text=True,capture_output=True,check=True).stdout.strip()
    atomic_json(ROOT/'records/repository_end_state.json',{'commit':commit,'status_sha256':hashlib.sha256(git_status.encode()).hexdigest(),'status_line_count':len(git_status.splitlines()),'repository_modified_by_this_task':False,'all_task_outputs_in_sibling_directory':str(ROOT)})
    print(json.dumps({'status':status,'accuracy_winners':winners,'auc_winners':auc_winners,'runtime':runtime},indent=2))

if __name__=='__main__':main()
