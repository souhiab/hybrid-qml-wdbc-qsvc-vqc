from __future__ import annotations
import datetime
import json
from pathlib import Path
import numpy as np
import pandas as pd
from vqc_engine import ROOT,atomic_json,sha,clinical

METRICS=['accuracy','roc_auc','average_precision','sensitivity','specificity','precision','f1']

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')

def compare_panel(manifest,results):
    rows=[]
    for budget in (100,200,300):
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
        for budget in (100,200,300):
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
        pilot.append({**{k:c[k] for k in ['configuration_id','p','r','feature_map_index','scaler','rfe_estimator','encoder','topology','feature_map_reps','ansatz_reps']},'ansatz_entanglement':'reverse_linear','final_rotation_layer':True,'seed_pairs':'101/1101; 211/1211; 307/1307','budgets':'100; 200; 300','circuit_path':c['circuit_path'],'parameter_order':json.dumps(c['parameter_order'])})
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
    if uncertainty is None:uncertainty=[{'comparison':f'{high}-{low}','priority':'primary' if (high,low)!=(300,100) else 'secondary','mean_change_pp':None,'ci95_low_pp':None,'ci95_high_pp':None,'proposed_margin_pp':.5,'margin_interpretation':'pending_all_stages','bootstrap_resamples':10000,'bootstrap_seed':manifest['protocol']['bootstrap_seed']} for high,low in [(200,100),(300,200),(300,100)]]
    payload={'Protocol':protocol,'Existing_Training_Audit':audit_rows,'Pilot_Manifest':pilot,'Run_Results':run,'Configuration_Summary':config_summary,'Budget_Comparison':compare_panel(manifest,results),'Paired_Uncertainty':uncertainty,'Coverage':coverage,'Failures_Attempts':attempt_rows,'Provenance':provenance}
    atomic_json(ROOT/'workbook_support/snapshot.json',payload)
    for name,rows in payload.items():
        temp=ROOT/'tables'/f'{name}.csv.tmp';pd.DataFrame(rows).to_csv(temp,index=False);temp.replace(ROOT/'tables'/f'{name}.csv')
    status={'updated':now(),'expected_fits':144,'completed':sum(r['status']=='completed' for r in run),'failed':sum(r['status']=='failed' for r in run),'pending':sum(r['status']=='pending' for r in run),'budgets':{str(b):{'completed':sum(r['status']=='completed' and r['budget']==b for r in run),'failed':sum(r['status']=='failed' and r['budget']==b for r in run),'expected':48} for b in (100,200,300)},'execution_state':'running' if len(results)<144 else 'outcomes_accounted_for','comparison_ready':uncertainty[0]['mean_change_pp'] is not None}
    atomic_json(ROOT/'status.json',status)
    return payload,status

def final_analysis(manifest,results):
    if len(results)!=144 or any(r['status']!='completed' for r in results.values()):
        atomic_json(ROOT/'records/analysis_blocker.json',{'reason':'Complete paired 48-model panels at all three budgets are required; failures or missing fits prevent aggregate/uncertainty claims.'})
        return None
    split=json.loads((ROOT/'split.json').read_text());ids=split['validation_indices']
    correctness=np.empty((3,16,3,91),dtype=np.float64)
    byid={row['configuration_id']:i for i,row in enumerate(manifest['configurations'])}
    metrics_error=0;prefix_checks=[];config_diffs=[]
    for row in manifest['fits']:
        result=results[row['fit_id']]
        with np.load(ROOT/result['attempt_path']/'predictions.npz',allow_pickle=False) as arrays:
            assert arrays['validation_indices'].tolist()==ids
            correctness[(100,200,300).index(row['budget']),byid[row['configuration_id']],row['seed_pair_id']-1]=(arrays['validation_labels']==arrays['validation_predictions']).astype(float)
            recalc=clinical(arrays['validation_labels'],arrays['validation_predictions'],arrays['validation_probabilities'])
            metrics_error=max(metrics_error,max(abs(value-result['metrics']['validation'][key]) for key,value in recalc.items()))
    analysis=ROOT/'analysis';analysis.mkdir(exist_ok=True)
    seed=manifest['protocol']['bootstrap_seed'];rng=np.random.default_rng(seed)
    draw_indices=rng.integers(0,91,size=(10000,91),dtype=np.int16)
    np.savez_compressed(analysis/'patient_bootstrap.npz',draw_indices=draw_indices,validation_indices=np.asarray(ids),correctness=correctness)
    uncertainty=[];distributions={}
    per_patient=correctness.mean(axis=(1,2))
    for high,low in [(200,100),(300,200),(300,100)]:
        hi=(100,200,300).index(high);lo=(100,200,300).index(low)
        differences=100*(per_patient[hi]-per_patient[lo])
        draws=differences[draw_indices].mean(axis=1)
        lower,upper=np.percentile(draws,[2.5,97.5],method='linear')
        interpretation='improvement_exceeds_proposed_margin_within_pilot' if lower>.5 else ('incremental_gain_below_proposed_margin' if upper<.5 else 'inconclusive_relative_to_proposed_margin')
        uncertainty.append({'comparison':f'{high}-{low}','priority':'primary' if (high,low)!=(300,100) else 'secondary','mean_change_pp':float(differences.mean()),'ci95_low_pp':float(lower),'ci95_high_pp':float(upper),'proposed_margin_pp':.5,'margin_interpretation':interpretation,'bootstrap_resamples':10000,'bootstrap_seed':seed})
        distributions[f'{high}-{low}']=draws
        for cid,ci in byid.items():
            for pair in range(3):config_diffs.append({'configuration_id':cid,'seed_pair_id':pair+1,'comparison':f'{high}-{low}','accuracy_change_pp':float(100*(correctness[hi,ci,pair].mean()-correctness[lo,ci,pair].mean()))})
    pd.DataFrame(distributions).to_csv(analysis/'bootstrap_differences_pp.csv',index=False)
    pd.DataFrame(config_diffs).to_csv(analysis/'paired_configuration_seed_differences.csv',index=False)
    for config in manifest['configurations']:
        for pair in (1,2,3):
            histories=[]
            for budget in (100,200,300):
                result=results[f'{config["configuration_id"]}-s{pair}-b{budget}']
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
            for high,low in [(200,100),(300,200),(300,100)]:
                values=[d['accuracy_change_pp'] for d in config_diffs if d['configuration_id'] in cids and d['comparison']==f'{high}-{low}']
                groups.append({'grouping':group_name,'level':level,'comparison':f'{high}-{low}','configurations':len(cids),'fit_pairs':len(values),'mean_accuracy_change_pp':float(np.mean(values)),'negative_fit_pairs':int(np.sum(np.asarray(values)<0)),'zero_fit_pairs':int(np.sum(np.asarray(values)==0)),'positive_fit_pairs':int(np.sum(np.asarray(values)>0))})
    pd.DataFrame(groups).to_csv(analysis/'group_differences.csv',index=False)
    verification={'completed_panels':3,'fits_per_panel':48,'patients':91,'bootstrap_replicates':10000,'bootstrap_seed':seed,'paired_patient_indices_shared_across_entire_model_panel':True,'bootstrap_indices_sha256':sha(analysis/'patient_bootstrap.npz'),'metric_max_error':metrics_error,'objective_prefix_matches':sum(c['identical_objective_prefix'] for c in prefix_checks),'objective_prefix_cases':48,'conditional_uncertainty':manifest['protocol']['bootstrap_conditions']}
    atomic_json(ROOT/'records/paired_analysis_verification.json',verification)
    return uncertainty

def write_report(manifest,results,payload,resources,status):
    budgets=payload['Budget_Comparison'];uncertainty=payload['Paired_Uncertainty']
    comparison_ready=all(row['complete'] for row in budgets) and all(row['mean_change_pp'] is not None for row in uncertainty)
    bybudget={row['budget']:row for row in budgets}
    first=uncertainty[0];second=uncertainty[1]
    if comparison_ready:
        if first['ci95_high_pp']<.5:
            recommended=100;reason='The interval for 200 minus 100 supports an incremental gain below the proposed 0.5-point margin. The 100-evaluation budget is the lower-cost working choice for this pilot.'
            if uncertainty[2]['ci95_low_pp']>.5:recommended=300;reason='The cumulative 300 minus 100 improvement exceeds the proposed practical margin, although the first increment alone is small; 300 is the tested performance-oriented choice.'
        elif second['ci95_low_pp']>.5:recommended=300;reason='The 300 minus 200 interval exceeds the proposed practical margin within this pilot; 300 is the tested performance-oriented choice.'
        elif first['ci95_low_pp']>.5 and second['ci95_high_pp']<.5:recommended=200;reason='The first increment exceeds the proposed margin, while the 300 minus 200 interval supports an incremental gain below it. 200 is the working performance/runtime balance.'
        else:
            recommended=min((100,200,300),key=lambda b:(-bybudget[b]['mean_accuracy'],b))
            reason=f'{recommended} has the highest observed panel mean (lower budget retained for an exact tie), but at least one primary interval is inconclusive relative to the proposed margin; this is a provisional choice, not a demonstrated practical optimum.'
    else:recommended=None;reason='Incomplete panels prevent a complete budget comparison or recommendation.'
    audit=json.loads((ROOT/'records/existing_training_audit.json').read_text())
    failures=[r for r in results.values() if r['status']=='failed']
    lines=['# Four-qubit VQC internal-validation budget pilot','',f"Execution: {status['completed']}/144 valid fits, {status['failed']} failed, {status['pending']} pending; 16 balanced configurations, three fixed seed pairs, all three mandatory budgets.",'',
        'This balanced pilot is not a full factorial grid. Feature-map index is (preprocessing index + repetition-pair index) modulo 4. It uses 364 pilot-training and 91 validation cases drawn only from the original 455 training cases. The original 114-case test partition was not inspected or scored.','',
        '## Existing training-history audit','',f"Inspected 192 training traces and optimizer records only: {audit['converged']} converged and {audit['budget_exhausted']} budget-exhausted. {audit['uncertain_plateau_flags']} traces meet the predefined finite-shot plateau-ambiguity flag. The flag compares late best-loss progress with the tail spread and upward steps; histories cannot isolate sampling variation from optimizer moves. With the installed integer sampler seed, repeated identical parameter vectors are deterministic.",'',
        '## Complete budget comparison','', '| Budget | Valid / expected | Mean validation accuracy | Mean malignant AUC | Mean sensitivity | Mean specificity | Mean F1 | Mean nfev | Converged | Budget exhausted | Sum fit runtime (s) |','|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in budgets:
        def fmt(key,digits=6):return f'{row[key]:.{digits}f}' if row[key] is not None else 'unavailable'
        lines.append(f"| {row['budget']} | {row['completed']}/48 | {fmt('mean_accuracy')} | {fmt('mean_roc_auc')} | {fmt('mean_sensitivity')} | {fmt('mean_specificity')} | {fmt('mean_f1')} | {fmt('mean_actual_nfev',2)} | {row['converged']} | {row['budget_exhausted']} | {row['sum_fit_runtime_seconds']:.3f} |")
    lines.extend(['','Accuracy and secondary clinical metrics are averaged equally across the fixed 16 configurations and three seed pairs. Each fit uses the same 91 validation patients; predictions are never pooled as extra patients. Budget exhaustion is retained as a valid budget-limited result. Per-configuration three-seed means, sample SDs, minima and maxima are in Configuration_Summary.csv.','',
        '## Paired accuracy differences and uncertainty','', '| Comparison | Priority | Mean change (pp) | Descriptive 95% patient-bootstrap interval (pp) | Interpretation relative to proposed +0.5 pp |','|---|---|---:|---|---|'] )
    for row in uncertainty:
        lines.append(f"| {row['comparison']} | {row['priority']} | {row['mean_change_pp']:.6f} | [{row['ci95_low_pp']:.6f}, {row['ci95_high_pp']:.6f}] | {row['margin_interpretation']} |" if row['mean_change_pp'] is not None else f"| {row['comparison']} | {row['priority']} | unavailable | unavailable | incomplete panels |")
    lines.extend(['','The bootstrap uses 10,000 ordinary resamples of the 91 validation patients, seed 20261108. The same sampled patient indices apply simultaneously to every budget, configuration and seed. The fitted-model panel stays fixed. Percentile endpoints are 2.5% and 97.5% using NumPy linear interpolation.','',
        'Intervals are conditional on this pilot-training split, these fitted models, these 16 configurations and these three seed pairs. They do not capture uncertainty from retraining on different training samples or selecting the pilot. They are descriptive: no formal multiple-comparison, equivalence or nonsignificance-as-equivalence claim is made. The 0.5-percentage-point margin is proposed for this comparison, not a universal scientific standard.','',
        '## Training progress and exceptions',''])
    for row in budgets:
        lines.append(f"- Budget {row['budget']}: mean best training loss {row['mean_best_training_loss']:.8f}; mean fitted loss {row['mean_fitted_training_loss']:.8f}; mean best-loss improvement in the final 20 evaluations {row['mean_last20_best_improvement']:.8f}." if row['complete'] else f"- Budget {row['budget']}: incomplete.")
    if comparison_ready:
        differences=pd.read_csv(ROOT/'analysis/paired_configuration_seed_differences.csv')
        group_diffs=pd.read_csv(ROOT/'analysis/group_differences.csv')
        for comparison in ['200-100','300-200','300-100']:
            subset=differences[differences.comparison==comparison]
            avg=subset.groupby('configuration_id').accuracy_change_pp.mean()
            lines.append(f"- {comparison}: {int((subset.accuracy_change_pp<0).sum())}/48 seed-fit pairs regressed, {int((subset.accuracy_change_pp==0).sum())} were unchanged, {int((subset.accuracy_change_pp>0).sum())} improved. Configuration-level three-seed mean changes ranged from {avg.min():.6f} to {avg.max():.6f} pp.")
            negative=avg[avg<0]
            if len(negative):lines.append('  Regressing configurations: '+', '.join(f'{cid} ({value:+.6f} pp)' for cid,value in negative.items())+'.')
        lines.extend(['','| Grouping | Level | Comparison | Mean accuracy change (pp) | Negative / zero / positive seed-fit pairs |','|---|---:|---|---:|---|'])
        for row in group_diffs.itertuples(index=False):lines.append(f'| {row.grouping} | {row.level} | {row.comparison} | {row.mean_accuracy_change_pp:.6f} | {row.negative_fit_pairs}/{row.zero_fit_pairs}/{row.positive_fit_pairs} |')
    lines.extend(['','Group labels follow the frozen p/r/feature-map indices in the manifest. These subgroup patterns are descriptive and do not estimate all factorial interactions.','',
        '## CPU execution and recovery','',f"Resource experiments: {json.dumps(resources,sort_keys=True)}.",'',
        'All execution was CPU-only with one BLAS/OpenMP thread per worker and independent initialization/sampling seeds. Resource selection used measured throughput, CPU load, available RAM and responsiveness, never validation results. Invariant encoder states and internal-split preprocessing were reused; every budget used a fresh optimizer/model/sampler from the same original vector. The verified numerical engine was reused unchanged.','',
        'COBYLA optimizer-state restoration remains unsupported. An interrupted fit restarts from its original initialization and seeds under a new attempt ID. Technical failures alone receive at most one retry; all poor and budget-exhausted outcomes are retained. Receipts are verified and atomically saved before the sole coordinator updates Excel.','',
        '## Recommendation','',f"Tested budget recommendation: {recommended if recommended else 'unavailable'}. {reason}",'',
        'Use another predeclared internal split as the more useful next robustness test before committing to a full-grid budget revision. This one conditional split does not establish a universally optimal budget. Larger untested budgets may still improve some configurations, and four-qubit findings do not determine budgets for larger widths. No full-grid, seven-qubit or larger-budget runs were launched.','',
        'Full four-qubit grid: retain the existing 100-evaluation evidence with its budget-limited labels. Treat the pilot recommendation as provisional planning evidence; confirm its practical balance on another internal split before scheduling a complete revised grid.','',
        '## Output and resume','',f'Output: `{ROOT}`.','',
        '```powershell',f"& '{ROOT / 'resume.ps1'}'",'```','',
        'The launcher performs one startup checkpoint inventory and skips verified fits. Tables and receipts are authoritative; Excel can be rebuilt without repeating completed fits.'] )
    (ROOT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    atomic_json(ROOT/'records/recommendation.json',{'tested_budget':recommended,'reason':reason,'conditional_pilot_only':True,'next_test':'another predeclared internal split before full-grid revision','full_four_qubit_grid_revision':'not launched; confirm provisional budget on another internal split','larger_width_inference':False})
    return recommended
