"""Local saved-kernel diagnostics. No preprocessing fits, quantum simulation or QPU calls."""
from pathlib import Path
import argparse,csv,warnings,time,shutil,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import joblib,sklearn
from sklearn.svm import SVC
from sklearn.exceptions import ConvergenceWarning
from qsvm_on_wdbc.budgeted_qpu.io import read,atomic,file_hash,identity,now
from qsvm_on_wdbc.budgeted_qpu.results import clinical

METHODS=['local_exact','local_shots','hardware']

def spectrum(K,tolerance):
    eigenvalues=np.linalg.eigvalsh((K+K.T)/2)
    neg=eigenvalues[eigenvalues < -tolerance];full=float(np.abs(eigenvalues).sum());mass=float(-neg.sum())
    return eigenvalues,{'minimum_eigenvalue':float(eigenvalues.min()),'maximum_eigenvalue':float(eigenvalues.max()),
        'trace':float(eigenvalues.sum()),'absolute_spectrum_sum':full,'positive_eigenvalue_sum':float(eigenvalues[eigenvalues>0].sum()),
        'negative_eigenvalue_count_below_tolerance':len(neg),'negative_eigenvalue_magnitude_sum':mass,
        'negative_mass_fraction_of_absolute_spectrum':mass/full if full else 0.,
        'unthresholded_negative_eigenvalue_magnitude_sum':float(-eigenvalues[eigenvalues<0].sum()),
        'negative_mass_to_signed_trace_ratio':mass/float(eigenvalues.sum()) if eigenvalues.sum()>0 else None,
        'negative_spectral_energy_fraction':float(np.sum(neg**2)/np.sum(eigenvalues**2)) if np.any(eigenvalues) else 0.,
        'symmetry_max_absolute_error':float(np.max(np.abs(K-K.T))),'negative_threshold':-tolerance,'matrix_repair_applied':False}

def captured_fit(K,y,params):
    start=time.perf_counter()
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always');model=SVC(**params);model.fit(K,y)
    all_warnings=[{'category':w.category.__name__,'message':str(w.message)} for w in recorded]
    return model,{'fit_status':int(model.fit_status_),'n_iter':model.n_iter_.tolist(),
        'convergence_warnings':[w for w in all_warnings if w['category']=='ConvergenceWarning'],
        'all_warnings':all_warnings,'fit_wall_seconds':time.perf_counter()-start}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True);args=parser.parse_args()
    d=args.run_dir.resolve();out=d/'solver_spectrum_audit_20261111';out.mkdir(exist_ok=True)
    if (out/'COMPLETED.json').exists():
        r=read(out/'COMPLETED.json');assert all(file_hash(out/k)==v for k,v in r['files'].items());print(out/'SOLVER_SPECTRUM_REPORT.md');return
    # Freeze diagnostics before any exploratory refit. These do not select a model.
    plan={'created':now(),'negative_eigenvalue_absolute_tolerance':1e-10,'permutation_seed':20261111,'random_permutations':10,
        'include_identity_and_reverse_order':True,'material_prediction_change_rule':'Any changed native held-out class prediction versus identity refit is flagged; numerical margin/probability changes also reported without equivalence claims.',
        'score_orientation':'malignant0; negative SVC decision_function; probability column classes_==0',
        'permutation_rule':'K_train[P,P], y_train[P], K_test[:,P]; training predictions and margins mapped back by inverse permutation.',
        'calibration_caveat':'probability=True and the original fixed random_state are retained. Row order changes internal calibration folds; probability differences are not attributable solely to the SVM decision solver.',
        'scope':'Exploratory classical solver diagnostics on saved kernels; no model selection, new patients, preprocessing fit, QPU jobs, kernels, or production-result replacement.'}
    if (out/'diagnostic_plan.json').exists():plan=read(out/'diagnostic_plan.json')
    else:atomic(out/'diagnostic_plan.json',plan)
    tol=plan['negative_eigenvalue_absolute_tolerance']
    _,check=spectrum(np.diag([-2.,1.,3.]),tol);assert abs(check['negative_mass_fraction_of_absolute_spectrum']-1/3)<1e-14
    _,check=spectrum(np.diag([1.,2.,3.]),tol);assert check['negative_eigenvalue_magnitude_sum']==0
    protocol=read(d/'protocol.json');raw=np.load(d/'raw_cases.npz');y=raw['y_train'];n=len(y)
    rng=np.random.default_rng(plan['permutation_seed']);perms=[('identity',np.arange(n)),('reverse',np.arange(n)[::-1])]
    perms += [(f'random-{i+1:02d}',rng.permutation(n)) for i in range(plan['random_permutations'])]
    atomic(out/'permutations.json',[{'id':name,'training_positions':v.tolist(),'training_source_row_ids':raw['train_ids'][v].tolist()} for name,v in perms])
    input_hashes={str((d/m/name).relative_to(d)):file_hash(d/m/name) for m in METHODS for name in ['svc.joblib','kernels_predictions.npz','metrics.json','predictions.csv','kernel_spectrum.json']}
    atomic(out/'input_hashes.json',input_hashes)
    saved_models={};spectra={};runs=[];outputs={};spectrum_rows=[];eigen_rows=[]
    for method in METHODS:
        a=np.load(d/method/'kernels_predictions.npz');K=a['kernel_train'];T=a['kernel_test'];saved=joblib.load(d/method/'svc.joblib')
        assert saved.get_params()==SVC(**protocol['svc']).get_params();assert saved.classes_.tolist()==[0,1]
        np.testing.assert_array_equal(saved.predict(T),a['predictions'])
        eigenvalues,stats=spectrum(K,tol);spectra[method]=stats;spectrum_rows.append({'method':method,**stats})
        for i,value in enumerate(eigenvalues):eigen_rows.append({'method':method,'ascending_index':i,'eigenvalue':float(value),'below_negative_tolerance':bool(value < -tol)})
        saved_models[method]={'fit_status':int(saved.fit_status_),'n_iter':saved.n_iter_.tolist(),'n_support_per_class':saved.n_support_.tolist(),
            'original_warning_capture':'Not captured as structured data at original fit. Cannot reconstruct historical warning events from the pickle.',
            'original_hardware_console_convergence_warning_mentions':None}
        if method=='hardware':
            path=d/'hardware/main_console.log';data=path.read_bytes();text=data.decode('utf-16') if data.startswith((b'\xff\xfe',b'\xfe\xff')) else data.decode('utf-8',errors='replace')
            saved_models[method]['original_hardware_console_convergence_warning_mentions']=text.lower().count('convergencewarning')
        identity_pred=None;identity_margin=None;identity_prob=None;identity_train_pred=None;identity_train_margin=None
        for name,P in perms:
            inverse=np.argsort(P);Kp=K[np.ix_(P,P)];Tp=T[:,P];yp=y[P]
            np.testing.assert_array_equal(Kp[np.ix_(inverse,inverse)],K);np.testing.assert_array_equal(Tp[:,inverse],T);np.testing.assert_array_equal(yp[inverse],y)
            model,fit=captured_fit(Kp,yp,protocol['svc']);assert model.classes_.tolist()==[0,1]
            pred=model.predict(Tp);margin=-model.decision_function(Tp);prob=model.predict_proba(Tp)[:,0]
            train_pred=model.predict(Kp)[inverse];train_margin=(-model.decision_function(Kp))[inverse]
            if name=='identity':
                identity_pred=pred;identity_margin=margin;identity_prob=prob;identity_train_pred=train_pred;identity_train_margin=train_margin
                np.testing.assert_array_equal(pred,a['predictions']);np.testing.assert_allclose(margin,a['malignant_decision_scores'],rtol=0,atol=1e-12)
                np.testing.assert_allclose(prob,a['malignant_probabilities'],rtol=0,atol=1e-12)
            row={'method':method,'permutation_id':name,'fit_status':fit['fit_status'],'n_iter':fit['n_iter'][0],
                'convergence_warning_count':len(fit['convergence_warnings']),'held_out_prediction_changes_vs_identity':int(np.sum(pred!=identity_pred)),
                'training_prediction_changes_vs_identity':int(np.sum(train_pred!=identity_train_pred)),
                'maximum_absolute_held_out_margin_change':float(np.max(np.abs(margin-identity_margin))),
                'maximum_absolute_training_margin_change':float(np.max(np.abs(train_margin-identity_train_margin))),
                'maximum_absolute_malignant_probability_change':float(np.max(np.abs(prob-identity_prob))),
                'held_out_correct':int(np.sum(pred==a['source_labels'])),'fit_wall_seconds':fit['fit_wall_seconds']}
            runs.append(row);outputs[method+'__'+name+'__predictions']=pred;outputs[method+'__'+name+'__malignant_margin']=margin;outputs[method+'__'+name+'__malignant_probability']=prob
            atomic(out/(method+'__'+name+'__fit.json'),{'permutation':name,'fit':fit,'summary':row})
    for name,rows in [('Spectrum_Comparison.csv',spectrum_rows),('Eigenvalues.csv',eigen_rows),('Permutation_Stability.csv',runs)]:
        with (out/name).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(out/'permutation_predictions_scores.npz',test_ids=raw['test_ids'],source_labels=raw['y_test'],**outputs)
    atomic(out/'saved_svc_diagnostics.json',saved_models);atomic(out/'spectrum_comparison.json',spectra)
    summary={}
    for method in METHODS:
        panel=[r for r in runs if r['method']==method and r['permutation_id']!='identity']
        summary[method]={'permuted_refits':len(panel),'refits_changing_any_held_out_prediction':sum(r['held_out_prediction_changes_vs_identity']>0 for r in panel),
            'maximum_changed_held_out_predictions':max(r['held_out_prediction_changes_vs_identity'] for r in panel),
            'maximum_changed_training_predictions':max(r['training_prediction_changes_vs_identity'] for r in panel),
            'maximum_absolute_held_out_margin_change':max(r['maximum_absolute_held_out_margin_change'] for r in panel),
            'maximum_absolute_malignant_probability_change':max(r['maximum_absolute_malignant_probability_change'] for r in panel),
            'minimum_n_iter':min(r['n_iter'] for r in panel),'maximum_n_iter':max(r['n_iter'] for r in panel),
            'nonzero_fit_status_count':sum(r['fit_status']!=0 for r in panel),'convergence_warning_count':sum(r['convergence_warning_count'] for r in panel)}
    atomic(out/'stability_summary.json',summary)
    lines=['# Saved SVC, kernel spectrum and permutation audit','',f'Generated: {now()}. sklearn {sklearn.__version__}. No QPU submission or new quantum computation. Original models, kernels, predictions and metrics are unchanged.','',
        '## Saved fitted models','','|Kernel|fit_status_|n_iter_|Support vectors (class0 / class1)|','|---|---:|---:|---|']
    for method,v in saved_models.items():lines.append(f"|{method}|{v['fit_status']}|{v['n_iter']}|{v['n_support_per_class']}|")
    lines+=['','fit_status_=0 records successful solver termination, not proof of a global optimum or generalization. Structured warnings were not captured at the original fits, so their historical absence cannot be inferred from the model files. No ConvergenceWarning occurrence was found in the saved original hardware console. All identity replays and permutation refits explicitly capture warnings; see per-fit JSON. n_iter_ describes the binary decision SVM fit, not every internal probability-calibration optimization.','',
        '## Spectrum magnitude','','Matrices: saved raw 48x48 training kernels, symmetrized only for eigensolver round-off; no PSD correction or diagonal adjustment. A negative eigenvalue is counted when lambda < -1e-10. M- = sum(abs(lambda)) over these negative eigenvalues; S = sum(abs(lambda)) over the entire spectrum. Reported negative mass percentage =100*M-/S. The denominator is the absolute full spectrum, not the signed trace. Unthresholded mass, signed-trace ratio and squared-eigenvalue energy fraction are also saved.','',
        '|Kernel|Minimum eigenvalue|Negative count|M-|Full absolute spectrum S|100 M-/S|','|---|---:|---:|---:|---:|---:|']
    for method,v in spectra.items():lines.append(f"|{method}|{v['minimum_eigenvalue']:.12g}|{v['negative_eigenvalue_count_below_tolerance']}|{v['negative_eigenvalue_magnitude_sum']:.12g}|{v['absolute_spectrum_sum']:.12g}|{100*v['negative_mass_fraction_of_absolute_spectrum']:.8f}%|")
    lines+=['','The ideal exact kernel provides the numerical reference; finite-shot and hardware matrices are compared with it. Negative mass measures the magnitude of indefiniteness but has no universal acceptability threshold. It does not establish that a downstream solver is stable or unstable on its own.','',
        '## Exploratory permutation check','','Predeclared identity replay, reverse order and10 fixed random permutations (seed20261111) for each kernel:36 classical fits in total. All reuse the same saved kernels and original SVC settings. Each permutation uses K[P,P], y[P] and K_test[:,P]; training outputs are inverse-permuted. No test-label feedback changes a fit or selects a permutation.','',
        '|Kernel|Permuted refits|Refits with any changed test prediction|Max changed /24|Max absolute malignant margin change|Max malignant probability change|Iteration range|Warnings|','|---|---:|---:|---:|---:|---:|---|---:|']
    for method,v in summary.items():lines.append(f"|{method}|{v['permuted_refits']}|{v['refits_changing_any_held_out_prediction']}|{v['maximum_changed_held_out_predictions']}|{v['maximum_absolute_held_out_margin_change']:.9g}|{v['maximum_absolute_malignant_probability_change']:.9g}|{v['minimum_n_iter']}–{v['maximum_n_iter']}|{v['convergence_warning_count']}|")
    lines+=['','Identity replays reproduced the saved held-out classes, decision scores and probabilities. Row order can change probability=True internal calibration folds even with a fixed random seed; probability changes therefore do not isolate the decision solver. Native class predictions and decision margins are reported separately. This finite set of permutations is exploratory evidence only: unchanged predictions do not prove uniqueness, convexity, general stability or external validity. The original production model and score are retained regardless of the audit outcomes.','',
        'Files: diagnostic_plan.json, input_hashes.json, saved_svc_diagnostics.json, Spectrum_Comparison.csv, Eigenvalues.csv, Permutation_Stability.csv, stability_summary.json, per-fit warning/status JSON and permutation_predictions_scores.npz.']
    (out/'SOLVER_SPECTRUM_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    assert all(file_hash(d/name)==digest for name,digest in input_hashes.items())
    shutil.copy2(Path(__file__),out/'audit_q9_solver_spectra.py')
    atomic(out/'COMPLETED.json',{'completed':now(),'qpu_jobs_submitted':0,'classical_diagnostic_refits':len(runs),'original_artifacts_unchanged':True,
        'input_hashes':input_hashes,'files':{f.name:file_hash(f) for f in out.iterdir() if f.is_file() and f.name!='COMPLETED.json'}})
    print(__import__('json').dumps({'report':str(out/'SOLVER_SPECTRUM_REPORT.md'),'saved_models':saved_models,'spectra':spectra,'stability':summary},indent=2))

if __name__=='__main__':main()
