from __future__ import annotations
import numpy as np
from common import *
from prepare import verify_folds,SPECS
from execution import workload,validate_receipt

def verify():
    if (ROOT/'records/preparation_lock.json').exists():verify_frozen()
    candidates=read(ROOT/'candidates.json');assert len(candidates)==10
    for i,(candidate,expected) in enumerate(zip(candidates,SPECS),1):
        k,selector,constructor,topology,reps=expected
        assert candidate['candidate_id']==f'Q{i}' and candidate['k']==k and candidate['selector']=='rfe_'+selector
        assert candidate['feature_map']==constructor and candidate['entanglement']==topology and candidate['feature_map_reps']==reps
        assert candidate['C']==1 and candidate['scaler_params']['feature_range']==[0,1] and not candidate['scaler_params']['clip']
        assert candidate['rfe_step']==1 and candidate['rfe_estimator_params']['random_state']==42
        assert candidate['kernel']['analytic'] and candidate['kernel']['shots'] is None
    verify_folds()
    items=workload();assert len(items)==255 and len({i['evaluation_id'] for i in items})==255
    assert sum(i['kind']=='qsvc' for i in items)==150
    assert len(read(ROOT/'comparators.json')['matched'])==5
    reused=0
    for item in items:
        file=ROOT/'reused_predictions'/item['evaluation_id']/'receipt.json'
        if file.exists():assert validate_receipt(item,file)['status']=='completed_reused';reused+=1
    tests=read(ROOT/'records/synthetic_verification.json');assert tests['passed']
    assert read(ROOT/'vqc_candidates.json')['shortlist_state']=='PENDING'
    return {'verified':True,'candidate_count':10,'outer_partitions':15,'total_outer_evaluations':255,'qsvc_fits':150,'matched_outer_evaluations':75,'matched_inner_fits':2025,'strong_reusable':reused,'new_outer_evaluations_pending':255-reused,'WDBC_models_fitted_during_preparation':0,'production_WDBC_kernels_generated':0}
