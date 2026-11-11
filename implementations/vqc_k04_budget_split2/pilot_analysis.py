from __future__ import annotations
import datetime
import json
from pathlib import Path
import numpy as np
import pandas as pd
from vqc_engine import ROOT,atomic_json,sha,clinical

METRICS=['accuracy','roc_auc','average_precision','sensitivity','specificity','precision','f1']
BUDGETS=(100,200)
COMPARISONS=[(200,100)]

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')

def compare_panel(manifest,results):
    rows=[]
    for budget in BUDGETS:
        panel=[results.get(row['fit_id']) for row in manifest['fits'] if row['budget']==budget]
        valid=[r for r in panel if r and r['status']=='completed'];failed=sum(bool(r) and r['status']=='failed' for r in panel)
        complete=len(valid)==48
        item={'budget':budget,'expected':48,'completed':len(valid),'failed':failed,'pending':48-len(valid)-failed,'complete':complete,'failure_rate':failed/48}
        for metric in METRICS:item['mean_'+metric]=float(np.mean([r['metrics']['validation'][metric] for r in valid])) if complete else None
        item.update({'mean_actual_nfev':float(np.mean([r['nfev'] for r in valid])) if complete else None,'sum_actual_nfev':sum(r['nfev'] for r in valid),'converged':sum(r['termination']=='converged' for r in valid),'budget_exhausted':sum(r['termination']=='budget_exhausted' for r in valid),'convergence_rate':sum(r['termination']=='converged' for r in valid)/48,'budget_exhaustion_rate':sum(r['termination']=='budget_exhausted' for r in valid)/48,'sum_fit_runtime_seconds':sum(r['runtime']['total_seconds'] for r in valid),'mean_fit_runtime_seconds':float(np.mean([r['runtime']['total_seconds'] for r in valid])) if complete else None,'mean_best_training_loss':float(np.mean([r['best_training_loss'] for r in valid])) if complete else None,'mean_fitted_training_loss':float(np.mean([r['fitted_training_loss'] for r in valid])) if complete else None,'mean_last20_best_improvement':float(np.mean([r['last20_best_improvement'] for r in valid])) if complete else None})
        rows.append(item)
    return rows

def snapshot(manifest,results,attempts,audit_rows,resource_benchmarks=None,uncertainty=None):
    run=[];config_summary=[];coverage=[]
    for row in manifest['fits']:
        result=results.get(row['fit_id']);valid=result and result['status']=='completed'
        item={key:row[key] for key in ['fit_id','configuration_id','p','r','feature_map_index','scaler','rfe_estimator','encoder','topology','feature_map_reps','ansatz_reps','budget','seed_pair_id','initialization_seed','sampler_seed']}
        item.update({'status':result['status'] if result else 'pending','termination':result['termination'] if result else None,'nfev':result.get('nfev') if result else None})
        for metric in METRICS+['correct_count','TP','FN','FP','TN']:item[metric]=result['metrics']['validation'][metric] if valid else None
        item.update({'initial_training_loss':result.get('initial_training_loss') if result else None,'best_training_loss':result.get('best_training_loss') if result else None,'fitted_training_loss':result.get('fitted_training_loss') if result else None,'last20_best_improvement':result.get('last20_best_improvement') if result else None,'runtime_seconds':result['runtime']['total_seconds'] if result else None,'peak_rss_bytes':result['runtime'].get('peak_rss_bytes') if result else None,'attempt_id':result['attempt_id'] if result else None,'artifact_path':result['attempt_path'] if result else None,'verification':bool(valid),'error':result.get('error') if result else None})
        run.append(item)
    for config in manifest['configurations']:
        for budget in BUDGETS:
            group=[r for r in run if r['configuration_id']==config['configuration_id'] and r['budget']==budget]
            valid=[r for r in group if r['status']=='completed'];fail=sum(r['status']=='failed' for r in group);complete=len(valid)==3
            item={k:config[k] for k in ['configuration_id','p','r','encoder','topology','feature_map_reps','ansatz_reps','scaler','rfe_estimator']}
            item.update({'budget':budget,'complete':complete,'valid_seed_outcomes':len(valid),'failed_seed_outcomes':fail,'pending_seed_outcomes':3-len(valid)-fail,'failure_rate':fail/3})
            cover=dict(item);coverage.append(cover)
            for metric in METRICS:item['mean_'+metric]=float(np.mean([r[metric] for r in valid])) if complete else None
            item.update({'sample_std_accuracy':float(np.std([r['accuracy'] for r in valid],ddof=1)) if complete else None,'min_accuracy':min(r['accuracy'] for r in valid) if complete else None,'max_accuracy':max(r['accuracy'] for r in valid) if complete else None})
            config_summary.append(item)
    pilot=[]
    for c in manifest['configurations']:
        pilot.append({**{k:c[k] for k in ['configuration_id','p','r','feature_map_index','scaler','rfe_estimator','encoder','topology','feature_map_reps','ansatz_reps']},'ansatz_entanglement':'reverse_linear','final_rotation_layer':True,'seed_pairs':'101/1101; 211/1211; 307/1307','budgets':'100; 200','circuit_path':c['circuit_path'],'parameter_order':json.dumps(c['parameter_order'])})
    protocol=[{'item':k,'value':json.dumps(v) if isinstance(v,(list,dict)) else v} for k,v in manifest['protocol'].items()]
    provenance=[{'item':file,'value':sha(ROOT/file)} for file in ['manifest.json','split.json','source/original_training_only.npz','vqc_engine.py','pilot_fit.py','prepare_pilot.py','pilot_controller.py','pilot_analysis.py','records/environment.json','records/numerical_validation.json']]
    provenance.extend([{'item':'original_test_partition','value':'No original test cases, labels, predictions or performance metrics are copied or inspected by this pilot.'},{'item':'patient_bootstrap_unit','value':'91 validation patients; the same resampled patient indices are shared by every budget/configuration/seed; model panel fixed.'},{'item':'bootstrap_conditions','value':manifest['protocol']['bootstrap_conditions']}])
    if resource_benchmarks:
        provenance.append({'item':'resource_benchmarks','value':json.dumps(resource_benchmarks)})
    attempt_rows=[]
    for row in manifest['fits']:
        current=attempts.get(row['fit_id'],[])
        if not current:current=[{'attempt_id':None,'status':'not_started','started':None,'ended':None,'technical_failure':False,'error':None,'restarted_from':None}]
        for item in current:attempt_rows.append({'fit_id':row['fit_id'],'budget':row['budget'],'attempt_id':item.get('attempt_id'),'status':item.get('status'),'started':item.get('started'),'ended':item.get('ended'),'technical_failure':item.get('technical_failure',False),'error':item.get('error'),'restarted_from':item.get('restarted_from'),'exact_optimizer_resume':False})
    if uncertainty is None:uncertainty=[{'comparison':'200-100','priority':'primary','mean_change_pp':None,'ci95_low_pp':None,'ci95_high_pp':None,'proposed_margin_pp':.5,'margin_interpretation':'deferred_to_later_user_review','bootstrap_resamples':10000,'bootstrap_seed':manifest['protocol']['bootstrap_seed']}]
    payload={'Protocol':protocol,'Pilot_Manifest':pilot,'Run_Results':run,'Configuration_Summary':config_summary,'Budget_Comparison':compare_panel(manifest,results),'Paired_Uncertainty':uncertainty,'Coverage':coverage,'Failures_Attempts':attempt_rows,'Provenance':provenance}
    atomic_json(ROOT/'workbook_support/snapshot.json',payload)
    for name,rows in payload.items():
        temp=ROOT/'tables'/f'{name}.csv.tmp';pd.DataFrame(rows).to_csv(temp,index=False);temp.replace(ROOT/'tables'/f'{name}.csv')
    status={'updated':now(),'expected_fits':96,'internal_split_seed':20261109,'completed':sum(r['status']=='completed' for r in run),'failed':sum(r['status']=='failed' for r in run),'pending':sum(r['status']=='pending' for r in run),'budgets':{str(b):{'completed':sum(r['status']=='completed' and r['budget']==b for r in run),'failed':sum(r['status']=='failed' and r['budget']==b for r in run),'expected':48} for b in BUDGETS},'execution_state':'running' if len(results)<96 else 'outcomes_accounted_for','comparison_ready':uncertainty[0]['mean_change_pp'] is not None,'budget_selection':'deferred_to_later_user_prompt'}
    atomic_json(ROOT/'status.json',status)
    return payload,status

def final_analysis(manifest,results):
    if len(results)!=96 or any(r['status']!='completed' for r in results.values()):
        atomic_json(ROOT/'records/analysis_blocker.json',{'reason':'Complete paired 48-model panels at both budgets are required; failures or missing fits prevent aggregate/uncertainty claims.'})
        return None
    split=json.loads((ROOT/'split.json').read_text());ids=split['validation_indices']
    correctness=np.empty((2,16,3,91),dtype=np.float64)
    byid={row['configuration_id']:i for i,row in enumerate(manifest['configurations'])}
    metrics_error=0;prefix_checks=[];config_diffs=[]
    for row in manifest['fits']:
        result=results[row['fit_id']]
        with np.load(ROOT/result['attempt_path']/'predictions.npz',allow_pickle=False) as arrays:
            assert arrays['validation_indices'].tolist()==ids
            correctness[BUDGETS.index(row['budget']),byid[row['configuration_id']],row['seed_pair_id']-1]=(arrays['validation_labels']==arrays['validation_predictions']).astype(float)
            recalc=clinical(arrays['validation_labels'],arrays['validation_predictions'],arrays['validation_probabilities'])
            metrics_error=max(metrics_error,max(abs(value-result['metrics']['validation'][key]) for key,value in recalc.items()))
    analysis=ROOT/'analysis';analysis.mkdir(exist_ok=True)
    seed=manifest['protocol']['bootstrap_seed'];rng=np.random.default_rng(seed)
    draw_indices=rng.integers(0,91,size=(10000,91),dtype=np.int16)
    np.savez_compressed(analysis/'patient_bootstrap.npz',draw_indices=draw_indices,validation_indices=np.asarray(ids),correctness=correctness)
    uncertainty=[];distributions={}
    per_patient=correctness.mean(axis=(1,2))
    for high,low in COMPARISONS:
        hi=BUDGETS.index(high);lo=BUDGETS.index(low)
        differences=100*(per_patient[hi]-per_patient[lo])
        draws=differences[draw_indices].mean(axis=1)
        lower,upper=np.percentile(draws,[2.5,97.5],method='linear')
        uncertainty.append({'comparison':f'{high}-{low}','priority':'primary','mean_change_pp':float(differences.mean()),'ci95_low_pp':float(lower),'ci95_high_pp':float(upper),'proposed_margin_pp':.5,'margin_interpretation':'deferred_to_later_user_review','bootstrap_resamples':10000,'bootstrap_seed':seed})
        distributions[f'{high}-{low}']=draws
        for cid,ci in byid.items():
            for pair in range(3):config_diffs.append({'configuration_id':cid,'seed_pair_id':pair+1,'comparison':f'{high}-{low}','accuracy_change_pp':float(100*(correctness[hi,ci,pair].mean()-correctness[lo,ci,pair].mean()))})
    pd.DataFrame(distributions).to_csv(analysis/'bootstrap_differences_pp.csv',index=False)
    pd.DataFrame(config_diffs).to_csv(analysis/'paired_configuration_seed_differences.csv',index=False)
    for config in manifest['configurations']:
        for pair in (1,2,3):
            histories=[]
            for budget in BUDGETS:
                result=results[f'split2-{config["configuration_id"]}-s{pair}-b{budget}']
                histories.append([json.loads(line) for line in (ROOT/result['attempt_path']/'training_loss.jsonl').read_text().splitlines()])
            common=min(map(len,histories))
            same=all(np.array_equal(histories[0][i]['weights'],hist[i]['weights']) and histories[0][i]['loss']==hist[i]['loss'] for hist in histories[1:] for i in range(common))
            prefix_checks.append({'configuration_id':config['configuration_id'],'seed_pair_id':pair,'common_prefix_evaluations':common,'identical_objective_prefix':same})
    pd.DataFrame(prefix_checks).to_csv(analysis/'objective_prefix_checks.csv',index=False)
    groups=[]
    config_map={c['configuration_id']:c for c in manifest['configurations']}
    for group_name,key in [('feature_map','feature_map_index'),('repetition_pair','r'),('preprocessing','p')]:
        for level in range(4):
            cids=[c['configuration_id'] for c in manifest['configurations'] if c[key]==level]
            for high,low in COMPARISONS:
                values=[d['accuracy_change_pp'] for d in config_diffs if d['configuration_id'] in cids and d['comparison']==f'{high}-{low}']
                groups.append({'grouping':group_name,'level':level,'comparison':f'{high}-{low}','configurations':len(cids),'fit_pairs':len(values),'mean_accuracy_change_pp':float(np.mean(values)),'negative_fit_pairs':int(np.sum(np.asarray(values)<0)),'zero_fit_pairs':int(np.sum(np.asarray(values)==0)),'positive_fit_pairs':int(np.sum(np.asarray(values)>0))})
    pd.DataFrame(groups).to_csv(analysis/'group_differences.csv',index=False)
    verification={'completed_panels':2,'fits_per_panel':48,'patients':91,'bootstrap_replicates':10000,'bootstrap_seed':seed,'paired_patient_indices_shared_across_entire_model_panel':True,'bootstrap_indices_sha256':sha(analysis/'patient_bootstrap.npz'),'metric_max_error':metrics_error,'objective_prefix_matches':sum(c['identical_objective_prefix'] for c in prefix_checks),'objective_prefix_cases':48,'conditional_uncertainty':manifest['protocol']['bootstrap_conditions']}
    atomic_json(ROOT/'records/paired_analysis_verification.json',verification)
    return uncertainty

def write_report(manifest,results,payload,resources,status):
    lines=['# Second-split VQC budget confirmation','',
        'Descriptive results only. Budget selection and substantive interpretation are deferred to a later user review.',
        '16 fixed balanced configurations, three seed pairs, two independent fresh fits at budgets 100 and 200: 96 expected fits.',
        'Split seed 20261109: 364 training and 91 validation cases from the original 455 training cases only. Scaling and RFE fitted on the 364 cases only. The original test partition is not used.',
        'The two internal splits overlap and must not be pooled as independent patient samples.','',
        '## Budget summaries','',
        '| Budget | Completed | Failed | Mean accuracy | Mean malignant AUC | Mean nfev | Runtime seconds |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for row in payload['Budget_Comparison']:
        def fmt(key):
            value=row.get(key)
            return 'unavailable' if value is None else f'{value:.6f}'
        lines.append(f"| {row['budget']} | {row['completed']} | {row['failed']} | {fmt('mean_accuracy')} | {fmt('mean_roc_auc')} | {fmt('mean_actual_nfev')} | {fmt('sum_fit_runtime_seconds')} |")
    lines.extend(['','## Paired descriptive uncertainty','',json.dumps(payload['Paired_Uncertainty'],indent=2),
        '10,000 paired patient resamples, seed 20261109; the same patient draws apply to every configuration, training seed and budget. The model panel remains fixed.',
        'Intervals are conditional on this training split, fitted models, configurations and seeds. They exclude retraining and pilot-selection uncertainty. The proposed 0.5 percentage-point margin is recorded for later review; no automatic decision is made.',
        'Budget-exhausted fits are valid budget-limited outcomes, not automatically converged. Failures and interrupted attempts remain recorded.',
        'COBYLA restoration is unsupported: interrupted fits restart from their original vector and seeds under a new attempt ID.',
        '## Execution','',json.dumps(resources,indent=2),'',
        'The unchanged verified numerical engine caches invariant encoder states and vectorizes exact seeded shot counting. BLAS/OpenMP threads are bounded to one per worker. Excel is written by one coordinator after authoritative artifact receipts.',
        '## Resume','',f"Run resume.ps1 in {ROOT}. Verified receipts are skipped."])
    (ROOT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    atomic_json(ROOT/'records/recommendation.json',{'budget_selection':'deferred_to_later_user_review','automatic_selection':False})
