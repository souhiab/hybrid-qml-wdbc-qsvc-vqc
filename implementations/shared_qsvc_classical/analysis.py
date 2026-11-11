from __future__ import annotations
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from common import *
from execution import workload,receipt_for,validate_receipt

METRICS=['roc_auc','average_precision','accuracy','sensitivity','specificity','precision','f1']
PREDICTION_FRAME_CACHE={}
def pooled_frame(candidate,results):
    key=(candidate,tuple((r['evaluation_id'],r['file_hashes']['predictions.csv']) for r in results if r['candidate_id']==candidate))
    if key in PREDICTION_FRAME_CACHE:return PREDICTION_FRAME_CACHE[key]
    frames=[]
    for result in results:
        if result['candidate_id']==candidate:
            path=Path(result['attempt_path']);path=path if path.is_absolute() else ROOT/path
            frames.append(pd.read_csv(path/'predictions.csv'))
    frame=pd.concat(frames,ignore_index=True);PREDICTION_FRAME_CACHE[key]=frame
    return frame

def weighted_panel(frame,counts):
    """Vectorized patient-bootstrap metrics; same patient counts in every repeat."""
    output={metric:[] for metric in METRICS}
    for repeat in (1,2,3):
        part=frame[frame.repeat_id==repeat].sort_values('sample_index')
        assert part.sample_index.tolist()==list(range(counts.shape[1]))
        truth=(part.true_source_label.to_numpy()==0);pred=(part.prediction.to_numpy()==0);score=part.malignant_score.to_numpy()
        order=np.argsort(score,kind='stable');starts=np.r_[0,1+np.flatnonzero(np.diff(score[order]))]
        pos=np.add.reduceat(counts[:,order]*truth[order],starts,axis=1)
        neg=np.add.reduceat(counts[:,order]*(~truth[order]),starts,axis=1)
        totalpos=pos.sum(axis=1);totalneg=neg.sum(axis=1)
        assert np.all(totalpos>0)&np.all(totalneg>0)
        auc=np.sum(pos*(np.cumsum(neg,axis=1)-.5*neg),axis=1)/(totalpos*totalneg)
        p=pos[:,::-1];n=neg[:,::-1];cp=np.cumsum(p,axis=1);cn=np.cumsum(n,axis=1)
        precision=np.divide(cp,cp+cn,out=np.zeros_like(cp,dtype=float),where=(cp+cn)>0)
        ap=np.sum(p/totalpos[:,None]*precision,axis=1)
        tp=counts[:,truth&pred].sum(axis=1);fn=counts[:,truth&~pred].sum(axis=1);fp=counts[:,~truth&pred].sum(axis=1);tn=counts[:,~truth&~pred].sum(axis=1)
        divide=lambda a,b:np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b>0)
        values={'roc_auc':auc,'average_precision':ap,'accuracy':(tp+tn)/counts.sum(axis=1),'sensitivity':divide(tp,tp+fn),'specificity':divide(tn,tn+fp),'precision':divide(tp,tp+fp),'f1':divide(2*tp,2*tp+fp+fn)}
        for key,value in values.items():output[key].append(value)
    return {key:np.mean(value,axis=0) for key,value in output.items()}

def bootstrap_pairs(complete,results,comparators):
    wanted=[(f'Q{i}',base) for i in range(1,11) for base in ('LR','RBF')]
    wanted += [(q,m['candidate_id']) for m in comparators['matched'] for q in m['comparisons']]
    wanted += [('RBF','LR')]
    wanted=[p for p in wanted if p[0] in complete and p[1] in complete]
    if not wanted:return []
    rng=np.random.default_rng(20261108);n=569
    draws=rng.integers(0,n,size=(10000,n),dtype=np.int16)
    counts=np.zeros((10000,n),dtype=np.float64)
    for i,row in enumerate(draws):counts[i]=np.bincount(row,minlength=n)
    np.savez_compressed(ROOT/'tables/patient_bootstrap_draws.npz',patient_indices=draws)
    values={}
    for cid in sorted(set(itertools.chain.from_iterable(wanted))):
        frame=pooled_frame(cid,results);batches=[]
        for start in range(0,len(counts),256):batches.append(weighted_panel(frame,counts[start:start+256]))
        values[cid]={key:np.concatenate([batch[key] for batch in batches]) for key in METRICS}
    rows=[]
    for left,right in wanted:
        for metric in METRICS:
            differences=values[left][metric]-values[right][metric]
            lo,hi=np.percentile(differences,[2.5,97.5],method='linear')
            exact=weighted_panel(pooled_frame(left,results),np.ones((1,n)))[metric][0]-weighted_panel(pooled_frame(right,results),np.ones((1,n)))[metric][0]
            factor=100 if metric=='accuracy' else 1
            rows.append({'left':left,'right':right,'metric':metric,'aggregation':'mean of three repeat-pooled OOF metrics','difference':float(exact*factor),'ci95_low':float(lo*factor),'ci95_high':float(hi*factor),'units':'percentage points' if factor==100 else 'absolute units','bootstrap_draws':10000,'bootstrap_seed':20261108,'inference':'descriptive fixed-panel paired patient bootstrap; not independent-fold inference'})
    return rows

def consolidate(verify=True,uncertainty=False,outcomes_cache=None):
    expected=workload();results=[];individual=[];coverage=[];resources=[];attempts=[]
    for item in expected:
        file=receipt_for(item)
        result=outcomes_cache.get(item['evaluation_id']) if outcomes_cache is not None else ((validate_receipt(item) if verify else read(file)) if file.exists() else None)
        valid=result and result['status'] in ('completed','completed_reused')
        if valid:results.append(result)
        row={'evaluation_id':item['evaluation_id'],'candidate_id':item['candidate_id'],'kind':item['kind'],'repeat_id':item['fold']['repeat_id'],'outer_fold_index':item['fold']['outer_fold_index'],'status':result['status'] if result else 'pending','runtime_seconds':result.get('runtime_seconds') if result else None}
        for key in METRICS+['TP','FN','FP','TN']:row[key]=result['metrics'][key] if valid else None
        individual.append(row)
        if valid and result.get('resources'):resources.append({'evaluation_id':item['evaluation_id'],**result['resources']})
        for file in (ROOT/'runs'/item['evaluation_id']).glob('attempt_*/attempt.json'):
            a=read(file);attempts.append({'evaluation_id':item['evaluation_id'],**a})
    summaries=[];complete=[]
    for cid in dict.fromkeys(item['candidate_id'] for item in expected):
        group=[r for r in results if r['candidate_id']==cid]
        coverage.append({'candidate_id':cid,'expected_outer_evaluations':15,'completed':len(group),'pending_or_failed':15-len(group),'complete':len(group)==15,'reused':sum(r.get('reused',False) for r in group)})
        if len(group)!=15:
            summaries.append({'candidate_id':cid,'complete':False,'outer_evaluations':len(group)});continue
        complete.append(cid);frame=pooled_frame(cid,results);repeat_records=[]
        assert len(frame)==1707
        for repeat in (1,2,3):
            part=frame[frame.repeat_id==repeat].sort_values('sample_index')
            assert part.sample_index.tolist()==list(range(569))
            repeat_records.append({'candidate_id':cid,'repeat_id':repeat,**metrics(part.true_source_label,part.prediction,part.malignant_score)})
        pd.DataFrame(repeat_records).to_csv(ROOT/f'tables/{cid}_repeat_pooled_metrics.csv',index=False)
        row={'candidate_id':cid,'complete':True,'outer_evaluations':15}
        for metric in METRICS:
            row['fold_mean_'+metric]=float(np.mean([r['metrics'][metric] for r in group]))
            row['repeat_pooled_mean_'+metric]=float(np.mean([r[metric] for r in repeat_records]))
            row['repeat_pooled_sample_sd_'+metric]=float(np.std([r[metric] for r in repeat_records],ddof=1))
        summaries.append(row)
    paired=bootstrap_pairs(complete,results,read(ROOT/'comparators.json')) if uncertainty else []
    candidates=read(ROOT/'candidates.json');comparators=read(ROOT/'comparators.json');protocol=read(ROOT/'protocol.json')
    classical_rows=[{'candidate_id':r['candidate_id'],'family':r['family'],'search_candidates':r.get('search_candidates',9),'inner_folds':3,'inner_model_fits_per_outer':r['inner_model_fits_per_outer'],'preprocessing_regime':r['preprocessing_regime'] if 'preprocessing_regime' in r else 'historical_strong','settings':canonical({k:v for k,v in r.items() if k!='search_space'}),'complete_search_space':'comparators.json'} for r in comparators['strong']+comparators['matched']]
    payload={'Protocol':[{'item':k,'value':canonical(v) if isinstance(v,(dict,list)) else v} for k,v in protocol.items()],'Candidates':[{k:r[k] for k in ['candidate_id','role','k','selector','feature_map','entanglement','feature_map_reps','C','archived_configuration_id']} for r in candidates],'Folds':pd.read_csv(ROOT/'tables/Folds.csv').to_dict('records'),'Classical_Procedures':classical_rows,'Reuse_Audit':read(ROOT/'prediction_reuse_audit.json'),'Individual_Results':individual,'Candidate_Summaries':summaries,'Paired_Comparisons':paired,'Resources':resources,'Coverage':coverage,'Attempts':attempts,'Provenance':[{'item':name,'value':sha(ROOT/name)} for name in ['protocol.json','candidates.json','comparators.json','folds/outer_inner_assignments.json','records/environment.json','records/provenance.json']]}
    for name,rows in payload.items():pd.DataFrame(rows).to_csv(ROOT/f'tables/{name}.csv',index=False)
    atomic(ROOT/'workbook_support/tables_snapshot.json',payload)
    atomic(ROOT/'records/consolidation_verification.json',{'expected_outer_evaluations':255,'complete_outer_evaluations':len(results),'reused_outer_evaluations':sum(r.get('reused',False) for r in results),'complete_candidates':complete,'prediction_metrics_verified':verify,'paired_uncertainty_calculated':uncertainty,'new_training_performed':False})
    lines=['# WDBC shared-fold evaluation','',protocol['evaluation'],'',f'Completed outer results: {len(results)}/255; reused: {sum(r.get("reused",False) for r in results)}.','',protocol['aggregation'],'',protocol['uncertainty'],'','Candidate summaries: tables/Candidate_Summaries.csv. Individual results and coverage are separate; pending cells are unavailable, not fabricated scores.']
    (ROOT/'ANALYSIS_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return payload
