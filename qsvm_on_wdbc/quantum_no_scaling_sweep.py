import os
import time
import traceback

import pandas as pd
from tqdm.auto import tqdm

from sklearn.datasets        import load_breast_cancer
from fixed_split_preprocessing import fixed_split, prepare_features
from sklearn.metrics         import (
    accuracy_score, roc_auc_score, f1_score, confusion_matrix
)

from qiskit.circuit.library                      import (
    ZZFeatureMap, ZFeatureMap, PauliFeatureMap
)
from qiskit_machine_learning.kernels             import FidelityStatevectorKernel
from qiskit_machine_learning.algorithms.classifiers import QSVC

def run_job(k, fmap_name, ent, reps, X_tr_sel, X_te_sel, y_tr, y_te,
            fmap_cache, out_csv):
    t0 = time.time()

    # look up cached feature map & run QSVC
    fmap = fmap_cache[(k, fmap_name, ent, reps)]
    qk   = FidelityStatevectorKernel(feature_map=fmap)
    qsvc = QSVC(quantum_kernel=qk, probability=True)
    qsvc.fit(X_tr_sel, y_tr)

    # predictions & metrics
    y_tr_q = qsvc.predict(X_tr_sel)
    y_te_q = qsvc.predict(X_te_sel)

    train_acc = accuracy_score(y_tr, y_tr_q)
    test_acc  = accuracy_score(y_te, y_te_q)
    overfit   = train_acc - test_acc
    auc       = roc_auc_score(y_te, qsvc.predict_proba(X_te_sel)[:,1])
    f1        = f1_score(y_te, y_te_q)
    tn, fp, fn, tp = confusion_matrix(y_te, y_te_q).ravel()

    duration = time.time() - t0

    # append one row of results
    row = {
        'n_features':     k,
        'Feature_Map':    fmap_name,
        'Entanglement':   ent,
        'Reps':           reps,
        'QSVC_Train_Acc': round(train_acc,4),
        'QSVC_Test_Acc':  round(test_acc,4),
        'QSVC_Overfit':   round(overfit,4),
        'QSVC_AUC':       round(auc,4),
        'QSVC_F1':        round(f1,4),
        'QSVC_TP':        int(tp),
        'QSVC_FP':        int(fp),
        'QSVC_TN':        int(tn),
        'QSVC_FN':        int(fn),
        'Duration_sec':   round(duration,4)
    }
    pd.DataFrame([row]).to_csv(out_csv, mode='a', header=False, index=False)

def safe_run_job(*args, **kwargs):
    try:
        return run_job(*args, **kwargs)
    except Exception:
        print("❌ Exception in job:", args[:4])
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # The original output key has no scaler or estimator. Keep the first
    # configuration per width, as used by the historical resume key.
    results_df = pd.read_csv('classical_rfe_fixed_counts.csv')[
        ['Scaling', 'Estimator', 'n_features']
    ].drop_duplicates(subset='n_features')

    data = load_breast_cancer()
    X_raw = pd.DataFrame(data.data, columns=data.feature_names)
    X_tr, X_te, y_tr, y_te = fixed_split(X_raw, data.target)

    # feature‐map factories
    feature_map_builders = {
        'ZZFeatureMap':    lambda d,r,e: ZZFeatureMap(feature_dimension=d, reps=r, entanglement=e),
        'ZFeatureMap':     lambda d,r,e: ZFeatureMap(feature_dimension=d, reps=r),
        'PauliFeatureMap': lambda d,r,e: PauliFeatureMap(feature_dimension=d, reps=r, entanglement=e),
    }
    entanglements = ['full','linear','circular']
    reps_list     = [1,2,3,4]   # <-- now all four reps

    # CSV setup
    out_csv = 'quantum_no_scaling_qsvc_results_train_only.csv'
    header = [
        'n_features','Feature_Map','Entanglement','Reps',
        'QSVC_Train_Acc','QSVC_Test_Acc','QSVC_Overfit',
        'QSVC_AUC','QSVC_F1','QSVC_TP','QSVC_FP','QSVC_TN','QSVC_FN',
        'Duration_sec'
    ]
    if not os.path.exists(out_csv):
        pd.DataFrame(columns=header).to_csv(out_csv, index=False)

    # skip any already-computed rows
    df_done  = pd.read_csv(out_csv)
    existing = set(zip(
        df_done['n_features'],
        df_done['Feature_Map'],
        df_done['Entanglement'],
        df_done['Reps']
    ))

    # cache all feature maps
    fmap_cache = {}
    for _, row in results_df.iterrows():
        k = int(row['n_features'])
        for fmap_name, builder in feature_map_builders.items():
            for ent in entanglements:
                for reps in reps_list:
                    fmap_cache[(k, fmap_name, ent, reps)] = builder(d=k, r=reps, e=ent)

    # build the job list
    jobs = []
    for _, row in results_df.iterrows():
        k = int(row['n_features'])
        pending = []
        for fmap_name in feature_map_builders:
            for ent in entanglements:
                for reps in reps_list:
                    key = (k, fmap_name, ent, reps)
                    if key not in existing:
                        pending.append(key)
        if pending:
            train_selected, test_selected, _ = prepare_features(
                X_tr, X_te, y_tr, row['Scaling'], row['Estimator'], k, raw=True
            )
            jobs.extend(
                (*key, train_selected, test_selected, y_tr, y_te, fmap_cache, out_csv)
                for key in pending
            )

    print(f"🚀 {len(jobs)} new jobs to run.\n")

    # run them with progress bar
    for job in tqdm(jobs, desc="QSVC no-scaling sweep"):
        safe_run_job(*job)

    print(f"\n✅ Done — results in '{out_csv}'")
