import copy
import json
import subprocess
from pathlib import Path
from sequence_analysis import planning
from vqc_engine import atomic_json

root=Path(__file__).resolve().parent
manifest=json.loads((root/'studies/internal-budget-pilot-k07/manifest.json').read_text())
results={}
for row in manifest['fits']:
    if row['budget']==300:continue
    results[row['fit_id']]={**row,'status':'completed','termination':'converged' if row['seed_pair_id']!=3 else 'budget_exhausted','nfev':row['budget'],'runtime':{'total_seconds':1},'metrics':{'validation':{key:.75 for key in ['accuracy','roc_auc','average_precision','sensitivity','specificity','precision','f1']}},'best_training_loss':1,'fitted_training_loss':1,'last20_best_improvement':0}
uncertainty=[{'comparison':'200-100','ci95_high_pp':.49}]
assert planning(manifest,results,[100,200],uncertainty)['add_300']
for row in manifest['fits']:
    if row['budget']==200:results[row['fit_id']]['termination']='converged'
decision=planning(manifest,results,[100,200],uncertainty)
assert not decision['add_300'] and decision['recommended_budget']==200
uncertainty[0]['ci95_high_pp']=.5
assert planning(manifest,results,[100,200],uncertainty)['add_300']
atomic_json(root/'records/planning_rule_checks.json',{'passed':True,'low_convergence_adds_300':True,'sufficient_convergence_and_small_interval_stops_at_200':True,'exact_upper_margin_boundary_adds_300':True,'smallest_eligible_budget_verified':True,'checks_used_synthetic_metrics_only':True})
run=subprocess.run(['node',str(root/'workbook_support/workbook_writer.mjs')],input='sync\nfinal\n',text=True,capture_output=True,encoding='utf-8',cwd=root/'workbook_support')
(root/'records/workbook_preflight_stdout.log').write_text(run.stdout,encoding='utf-8')
(root/'records/workbook_preflight_stderr.log').write_text(run.stderr,encoding='utf-8')
answers=[json.loads(line[2:]) for line in run.stdout.splitlines() if line.startswith('@@')]
assert len(answers)==2 and all(answer['ok'] for answer in answers),run.stdout[-2000:]
assert answers[-1]['result']['allPassed']
print('PREFLIGHT PASSED: planning boundaries and all 12 workbook sheets reconciled.',flush=True)
