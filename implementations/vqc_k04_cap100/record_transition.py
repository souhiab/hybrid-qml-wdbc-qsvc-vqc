import json
from vqc_engine import ROOT,atomic_json,sha

def main():
    completed=[];interrupted=[]
    for path in (ROOT/'artifacts').glob('*/receipt.json'):
        row=json.loads(path.read_text())
        if row['status']=='completed':completed.append(row['fit_id'])
    for path in (ROOT/'artifacts').glob('*/attempt_*/attempt.json'):
        row=json.loads(path.read_text())
        if row['state']=='running':interrupted.append({'fit_id':row['fit_id'],'attempt_id':row['attempt_id']})
    atomic_json(ROOT/'records/implementation_transition_v2.json',{
        'from_version':'vqc-k04-exact-cached-v1','to_version':'vqc-k04-exact-cached-v2',
        'reason':'Training execution and workbook synchronization had low throughput under measured memory pressure. Exact aggregation of the same seeded 1024 uniforms and reuse of the coordinator workbook object reduce CPU and memory overhead.',
        'scientific_settings_changed':False,'test_information_used_for_change':False,
        'v1_completed_fits_retained':sorted(completed),'interrupted_attempts_retained':interrupted,
        'restart_policy':'Only incomplete fits restart using original saved initialization and seeds under a new attempt ID; completed fits are never repeated.',
        'validation_path':'records/simulator_validation.json','validation_sha256':sha(ROOT/'records/simulator_validation.json'),
        'old_source_and_locks':'versions/v1','budget':100,'shots':1024,
        'sampling_equivalence':'Same deterministic 1024 uniforms per integer sampler seed. Count uniforms in each CDF interval with searchsorted(sorted_uniforms,cdf,side=left); matches original per-draw basis-state searchsorted(cdf,uniform,side=right).'
    })

if __name__=='__main__':main()
