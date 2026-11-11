from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from vqc_engine import atomic_json,sha,clinical

ROOT=Path(__file__).resolve().parent
METRICS=['accuracy','roc_auc','average_precision','sensitivity','specificity','precision','f1']

def compare(manifest,results,budgets):
    output=[]
    for budget in budgets:
        expected=[row for row in manifest['fits'] if row['budget']==budget]
        outcomes=[results.get(row['fit_id']) for row in expected]
        valid=[r for r in outcomes if r and r['status']=='completed']
        failed=sum(bool(r) and r['status']=='failed' for r in outcomes)
        complete=len(valid)==len(expected)
        row={'qubits':expected[0]['qubits'],'evidence_type':expected[0]['evidence_type'],'budget':budget,'expected':len(expected),'completed':len(valid),'failed':failed,'pending':len(expected)-len(valid)-failed,'complete':complete,'failure_rate':failed/len(expected)}
        for metric in METRICS:row['mean_'+metric]=float(np.mean([r['metrics']['validation'][metric] for r in valid])) if complete else None
        row.update({'converged':sum(r['termination']=='converged' for r in valid),'budget_exhausted':sum(r['termination']=='budget_exhausted' for r in valid),'convergence_fraction':sum(r['termination']=='converged' for r in valid)/len(expected),'mean_nfev':float(np.mean([r['nfev'] for r in valid])) if complete else None,'sum_nfev':sum(r['nfev'] for r in valid),'sum_fit_seconds':sum(r['runtime']['total_seconds'] for r in valid),'mean_best_training_loss':float(np.mean([r['best_training_loss'] for r in valid])) if complete else None,'mean_final_training_loss':float(np.mean([r['fitted_training_loss'] for r in valid])) if complete else None,'mean_last20_best_improvement':float(np.mean([r['last20_best_improvement'] for r in valid])) if complete else None})
        output.append(row)
    return output

def paired_analysis(manifest,results,budgets):
    directory=Path(manifest['fits'][0]['study_root'])
    panels={}
    for budget in budgets:
        rows=[row for row in manifest['fits'] if row['budget']==budget]
        if any(results.get(row['fit_id'],{}).get('status')!='completed' for row in rows):
            atomic_json(directory/'records/analysis_blocker.json',{'reason':'Incomplete matched panel; no successful-seed-only averaging or uncertainty estimate'})
            return []
        panel=[]
        for row in rows:
            r=results[row['fit_id']]
            with np.load(directory/r['attempt_path']/'predictions.npz',allow_pickle=False) as d:
                panel.append((d['validation_predictions']==d['validation_labels']).astype(float))
                recalc=clinical(d['validation_labels'],d['validation_predictions'],d['validation_probabilities'])
                assert all(abs(recalc[key]-value)<=1e-12 for key,value in r['metrics']['validation'].items())
        panels[budget]=np.asarray(panel)
    # Every budget uses identical manifest order: configuration then seed pair.
    order={b:[(r['configuration_id'],r['seed_pair_id']) for r in manifest['fits'] if r['budget']==b] for b in budgets}
    assert all(order[b]==order[budgets[0]] for b in budgets)
    draws=np.random.default_rng(20261108).integers(0,91,size=(10000,91),dtype=np.int16)
    comparisons=[(200,100)]+([(300,200),(300,100)] if 300 in budgets else [])
    output=[];differences=[];distributions={}
    for high,low in comparisons:
        per_patient=100*(panels[high]-panels[low]).mean(axis=0)
        samples=per_patient[draws].mean(axis=1)
        lower,upper=np.percentile(samples,[2.5,97.5],method='linear')
        interpretation='improvement exceeds proposed margin' if lower>.5 else ('incremental gain below proposed margin' if upper<.5 else 'inconclusive relative to proposed margin')
        output.append({'qubits':manifest['fits'][0]['qubits'],'comparison':f'{high}-{low}','mean_change_pp':float(per_patient.mean()),'ci95_low_pp':float(lower),'ci95_high_pp':float(upper),'proposed_margin_pp':.5,'margin_interpretation':interpretation,'bootstrap_resamples':10000,'bootstrap_seed':20261108,'conditional_fixed_panel':True})
        distributions[f'{high}-{low}']=samples
        delta=100*(panels[high].mean(axis=1)-panels[low].mean(axis=1))
        for (cid,pair),value in zip(order[high],delta):
            differences.append({'qubits':manifest['fits'][0]['qubits'],'configuration_id':cid,'seed_pair_id':pair,'comparison':f'{high}-{low}','accuracy_change_pp':float(value)})
    np.savez_compressed(directory/'analysis/patient_bootstrap.npz',draw_indices=draws,**{f'correctness_b{b}':p for b,p in panels.items()})
    pd.DataFrame(differences).to_csv(directory/'analysis/paired_fit_differences.csv',index=False)
    pd.DataFrame(distributions).to_csv(directory/'analysis/bootstrap_differences_pp.csv',index=False)
    changes=pd.DataFrame(differences)
    configmap={r['configuration_id']:r for r in manifest['configurations']}
    grouped=[]
    for key in ['feature_map_index','r','p']:
        for level in range(4):
            for high,low in comparisons:
                part=changes[(changes.comparison==f'{high}-{low}') & changes.configuration_id.map(lambda cid:configmap[cid][key]==level)]
                grouped.append({'qubits':manifest['fits'][0]['qubits'],'grouping':key,'level':level,'comparison':f'{high}-{low}','mean_change_pp':float(part.accuracy_change_pp.mean()),'improved':int(sum(part.accuracy_change_pp>0)),'unchanged':int(sum(part.accuracy_change_pp==0)),'declined':int(sum(part.accuracy_change_pp<0))})
    pd.DataFrame(grouped).to_csv(directory/'analysis/group_differences.csv',index=False)
    atomic_json(directory/'records/paired_analysis.json',{'uncertainty':output,'patients':91,'fixed_model_panel':48,'same_patient_draws_all_budgets_seeds_configurations':True,'conditions':manifest['protocol']['uncertainty_conditions']})
    return output

def planning(manifest,results,budgets,uncertainty):
    summaries=compare(manifest,results,budgets)
    bybudget={r['budget']:r for r in summaries}
    complete=all(r['complete'] for r in summaries)
    delta=next((r for r in uncertainty if r['comparison']=='200-100'),None)
    completion_trigger=bybudget[200]['convergence_fraction']<.9
    accuracy_trigger=delta is None or delta['ci95_high_pp']>=.5
    add300=completion_trigger or accuracy_trigger or not complete
    best=max((r['mean_accuracy'] for r in summaries if r['complete']),default=None)
    eligible=[r['budget'] for r in summaries if r['complete'] and r['convergence_fraction']>=.9 and best is not None and r['mean_accuracy']>=best-.005]
    recommendation=min(eligible) if eligible and complete else None
    latest=max(budgets)
    last_delta=next((r for r in uncertainty if r['comparison']=='300-200'),None)
    needs_review=not complete or recommendation is None or (latest==300 and (last_delta is None or last_delta['ci95_high_pp']>=.5))
    return {'qubits':manifest['fits'][0]['qubits'],'tested_budgets':';'.join(map(str,budgets)),'add_300':bool(add300),'completion_trigger':bool(completion_trigger),'accuracy_interval_trigger':bool(accuracy_trigger),'complete_panels':complete,'convergence_fraction_at_200':bybudget[200]['convergence_fraction'],'upper_200_minus_100_pp':delta['ci95_high_pp'] if delta else None,'recommended_budget':recommendation,'recommendation_status':'provisional for later review' if recommendation else 'no recommendation satisfies complete-panel and completion criterion','requires_review':bool(needs_review),'full_grid_launched':False}

def snapshot(master,manifests,results,attempts,activated,uncertainties,recommendations,phase):
    full=[];pilot=[];summaries=[];coverage=[];comparisons=[];attempt_rows=[]
    for sid,manifest in manifests.items():
        budgets=activated[sid]
        comparisons.extend(compare(manifest,results,budgets))
        for row in manifest['fits']:
            if row['budget'] not in budgets:continue
            result=results.get(row['fit_id']);valid=result and result['status']=='completed'
            item={key:row[key] for key in ['fit_id','configuration_id','qubits','budget','seed_pair_id','initialization_seed','sampler_seed','scaler','rfe_estimator','encoder','topology','feature_map_reps','ansatz_reps','evidence_type','evaluation_cohort']}
            item.update({'status':result['status'] if result else 'pending','termination':result.get('termination') if result else None,'nfev':result.get('nfev') if result else None,'runtime_seconds':result['runtime']['total_seconds'] if result else None,'artifact_path':str(Path(row['study_root'])/result['attempt_path']) if result else None,'error':result.get('error') if result else None})
            for metric in METRICS+['correct_count','TP','FN','FP','TN']:item[metric]=result['metrics']['validation'][metric] if valid else None
            (full if row['qubits']==4 else pilot).append(item)
            for attempt in attempts.get(row['fit_id'],[]):
                attempt_rows.append({'fit_id':row['fit_id'],'qubits':row['qubits'],'budget':row['budget'],**{key:attempt.get(key) for key in ['attempt_id','status','started','ended','technical_failure','error','restarted_from']}})
        for config in manifest['configurations']:
            for budget in budgets:
                rows=[r for r in manifest['fits'] if r['configuration_id']==config['configuration_id'] and r['budget']==budget]
                valid=[results[r['fit_id']] for r in rows if results.get(r['fit_id'],{}).get('status')=='completed']
                failed=sum(results.get(r['fit_id'],{}).get('status')=='failed' for r in rows)
                item={key:config[key] for key in ['configuration_id','qubits','evidence_type','scaler','rfe_estimator','encoder','topology','feature_map_reps','ansatz_reps']}
                item.update({'budget':budget,'valid_seeds':len(valid),'failed_seeds':failed,'pending_seeds':3-len(valid)-failed,'complete':len(valid)==3,'failure_rate':failed/3})
                coverage.append(dict(item))
                for metric in METRICS:item['mean_'+metric]=float(np.mean([r['metrics']['validation'][metric] for r in valid])) if len(valid)==3 else None
                item['sample_sd_accuracy']=float(np.std([r['metrics']['validation']['accuracy'] for r in valid],ddof=1)) if len(valid)==3 else None
                summaries.append(item)
    historical=pd.read_csv(ROOT/'tables/Historical_100_Reference.csv').where(pd.notna(pd.read_csv(ROOT/'tables/Historical_100_Reference.csv')),None).to_dict('records')
    payload={'Four_Qubit_200':full,'Four_Qubit_Summary':[r for r in summaries if r['qubits']==4],'Historical_100_Reference':historical,'Pilot_Fit_Results':pilot,'Pilot_Config_Summary':[r for r in summaries if r['qubits']!=4],'Budget_Comparisons':comparisons,'Paired_Uncertainty':[r for rows in uncertainties.values() for r in rows],'Budget_Recommendations':list(recommendations.values()),'Coverage':coverage,'Attempts_Errors':attempt_rows,'Protocol':[{'item':key,'value':json.dumps(value) if isinstance(value,(dict,list)) else value} for key,value in master['protocol'].items()],'Provenance':[{'item':str(path.relative_to(ROOT)),'value':sha(path)} for path in [ROOT/'manifest.json',ROOT/'records/input_lock.json',ROOT/'records/source_lock.json',ROOT/'records/environment.json',ROOT/'records/numerical_validation.json']]}
    for name,rows in payload.items():
        file=ROOT/'tables'/f'{name}.csv';temp=file.with_suffix('.csv.tmp');pd.DataFrame(rows).to_csv(temp,index=False);temp.replace(file)
    atomic_json(ROOT/'workbook_support/snapshot.json',payload)
    active_ids={r['fit_id'] for sid,m in manifests.items() for r in m['fits'] if r['budget'] in activated[sid]}
    status={'phase':phase,'expected_initial':480,'maximum_possible':624,'activated_fits':len(active_ids),'completed':sum(results.get(fid,{}).get('status')=='completed' for fid in active_ids),'failed':sum(results.get(fid,{}).get('status')=='failed' for fid in active_ids),'pending':sum(fid not in results for fid in active_ids),'activated_budgets':activated,'execution_state':'running','controller_active':True}
    atomic_json(ROOT/'status.json',status)
    return payload,status

def final_report(payload,resources,status):
    lines=['# WDBC VQC execution and budget report','',
        'Four-qubit results are exploratory scores on the original fixed test split with historical test-selection history. Internal-validation pilot scores are kept separate and do not establish complete-grid maxima.',
        'Phase A adopts 200 for optimization allowance and termination, not a demonstrated meaningful predictive gain.',
        'Budget exhaustion is a valid budget-limited result, not convergence. Every failure and interrupted attempt is preserved.',
        '## Execution coverage','',json.dumps(status,indent=2),'','## Budget summaries','',
        pd.DataFrame(payload['Budget_Comparisons']).to_csv(index=False),
        '## Paired descriptive uncertainty','',json.dumps(payload['Paired_Uncertainty'],indent=2),
        'Intervals use 10,000 paired resamples of the same 91 validation patients with fixed model panels and seed 20261108. They exclude retraining and pilot-selection uncertainty. Repeated predictions are not additional patients. No equivalence claims.',
        '## Provisional recommendations','',json.dumps(payload['Budget_Recommendations'],indent=2),
        'Recommendations use the frozen 90% convergence and 0.5-point accuracy planning rules. They are not universal standards or evidence of global optimization. No larger-width full grids were run.',
        'Configuration-specific regressions and exceptions, mean changes by feature map/repetition/preprocessing, and saved bootstrap draws are in each pilot analysis directory.',
        '## CPU and resource measurements','',json.dumps(resources,indent=2),
        '## Evidence','',
        'The workbook and tables retain full-grid test scores, historical 100-cap reference scores, and internal-validation pilot scores in separate tables. Per-fit receipts, initial/final parameters, QPY, selected features, audits and prediction arrays are authoritative.']
    (ROOT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
