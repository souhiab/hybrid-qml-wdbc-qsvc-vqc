import json
import numpy as np
import pandas as pd
from vqc_engine import ROOT,atomic_json

def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    with np.load(ROOT/'analysis/patient_bootstrap.npz',allow_pickle=False) as saved:
        draws=saved['draw_indices'];correctness=saved['correctness'];indices=saved['validation_indices']
    expected=np.random.default_rng(20261108).integers(0,91,size=(10000,91),dtype=np.int16)
    np.testing.assert_array_equal(draws,expected)
    np.testing.assert_array_equal(indices,json.loads((ROOT/'split.json').read_text())['validation_indices'])
    summary=pd.read_csv(ROOT/'tables/Budget_Comparison.csv')
    intervals=pd.read_csv(ROOT/'tables/Paired_Uncertainty.csv')
    maximum_error=0
    for index,budget in enumerate((100,200,300)):
        stored=float(summary.loc[summary.budget==budget,'mean_accuracy'].iloc[0])
        maximum_error=max(maximum_error,abs(stored-float(correctness[index].mean())))
    for hi,lo in ((200,100),(300,200),(300,100)):
        delta=100*np.mean(correctness[(100,200,300).index(hi)]-correctness[(100,200,300).index(lo)],axis=(0,1))
        independent=np.asarray([np.mean(delta[patients]) for patients in draws])
        low,high=np.percentile(independent,[2.5,97.5],method='linear')
        row=intervals[intervals.comparison==f'{hi}-{lo}'].iloc[0]
        maximum_error=max(maximum_error,abs(row.mean_change_pp-delta.mean()),abs(row.ci95_low_pp-low),abs(row.ci95_high_pp-high))
    equal_predictions=0;probability_changes=0
    for config in manifest['configurations']:
        for pair in (1,2,3):
            panels=[]
            for budget in (200,300):
                receipt=json.loads((ROOT/'artifacts'/f'{config["configuration_id"]}-s{pair}-b{budget}'/'receipt.json').read_text())
                with np.load(ROOT/receipt['attempt_path']/'predictions.npz',allow_pickle=False) as data:
                    panels.append((data['validation_predictions'],data['validation_probabilities']))
            equal_predictions+=int(np.array_equal(panels[0][0],panels[1][0]))
            probability_changes+=int(not np.array_equal(panels[0][1],panels[1][1]))
    assert maximum_error<1e-12
    report={'verified':True,'patient_resamples':10000,'validation_patients':91,'fixed_configuration_seed_panel':48,'bootstrap_seed':20261108,'all_saved_indices_reproduced':True,'maximum_independent_recalculation_error':maximum_error,'identical_200_300_validation_prediction_arrays':equal_predictions,'changed_200_300_probability_arrays':probability_changes,'degenerate_interval_explanation':'The 200 and 300 budgets give identical saved class predictions for this panel; a zero-width conditional accuracy interval does not establish general equivalence.'}
    atomic_json(ROOT/'records/independent_bootstrap_verification.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
