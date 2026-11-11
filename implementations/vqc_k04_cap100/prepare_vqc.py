from __future__ import annotations
import ast
import hashlib
import importlib.metadata
import inspect
import json
import platform
import subprocess
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from qiskit import qpy
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZFeatureMap,ZZFeatureMap,PauliFeatureMap,RealAmplitudes
from qiskit.quantum_info import Operator,Statevector
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_algorithms.optimizers import COBYLA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from vqc_engine import ROOT,VERSION,sha,array_sha,atomic_json,CachedStatevectorQNN

SEEDS=[(1,101,1101),(2,211,1211),(3,307,1307)]

def circuits(row):
    fmap=ZFeatureMap(4,reps=int(row['feature_map_reps'])) if row['encoder']=='Z' else ZZFeatureMap(4,reps=int(row['feature_map_reps']),entanglement=row['topology'])
    ansatz=RealAmplitudes(4,reps=int(row['ansatz_reps']),entanglement='reverse_linear',skip_final_rotation_layer=False)
    return fmap,ansatz

def main():
    if (ROOT/'manifest.json').exists(): raise RuntimeError('Manifest exists; use runner to resume rather than replacing it')
    records=ROOT/'records'
    with np.load(ROOT/'source/data/wdbc_arrays.npz',allow_pickle=False) as d:
        X=d['X'];y=d['y'];names=d['feature_names'].astype(str).tolist()
    split=json.loads((ROOT/'source/historical/split_idx.json').read_text())
    train=np.asarray(split['train_idx'],dtype=np.int64);test=np.asarray(split['test_idx'],dtype=np.int64)
    regenerated=train_test_split(np.arange(len(y)),test_size=.2,random_state=42,stratify=y)
    assert train.tolist()==regenerated[0].tolist() and test.tolist()==regenerated[1].tolist()
    assert len(train)==455 and len(test)==114 and np.sum(y[test]==0)==42 and np.sum(y[test]==1)==72
    assert not set(train).intersection(test)
    prep_checks=[]
    for scaling,token in [('StandardScaler','standard'),('MinMaxScaler','minmax')]:
        for estimator,etoken in [('Logistic Regression','lr'),('Random Forest','rf')]:
            pid=f'fs-k04-{token}-{etoken}'
            audit=json.loads((ROOT/f'preprocessing/{pid}.audit.json').read_text())
            cache_path=ROOT/f'preprocessing/{pid}.npz'
            assert sha(cache_path)==audit['cache_sha256']
            assert audit['scaler']==scaling and audit['rfe_estimator']==estimator
            assert audit['rfe_settings']['n_features_to_select']==4 and audit['rfe_settings']['step']==1
            assert audit['scaler_fit_rows']==audit['rfe_fit_rows']==455
            assert not audit['test_indices_in_fit'] and not audit['test_labels_in_fit']
            assert audit['scaler_fit_index_sha256']==audit['rfe_fit_index_sha256']==array_sha(train)
            assert audit['rfe_fit_labels_sha256']==array_sha(y[train])
            with np.load(cache_path,allow_pickle=False) as cache:
                assert cache['train_idx'].tolist()==train.tolist() and cache['test_idx'].tolist()==test.tolist()
                np.testing.assert_array_equal(cache['y_train'],y[train]);np.testing.assert_array_equal(cache['y_test'],y[test])
                scaler=(StandardScaler() if token=='standard' else MinMaxScaler()).fit(X[train])
                selected=cache['selected_indices'].astype(int)
                np.testing.assert_array_equal(scaler.transform(X[train])[:,selected],cache['X_train'])
                np.testing.assert_array_equal(scaler.transform(X[test])[:,selected],cache['X_test'])
                assert cache['selected_names'].tolist()==[names[i] for i in selected]
                assert array_sha(cache['X_train'])==audit['X_train_sha256'] and array_sha(cache['X_test'])==audit['X_test_sha256']
            prep_checks.append({'preprocessing_id':pid,'cache_sha256':sha(cache_path),'audit_sha256':sha(ROOT/f'preprocessing/{pid}.audit.json'),'verified':True})
    atomic_json(records/'preprocessing_verification.json',{'verified':True,'preprocessing':prep_checks,'train_index_sha256':array_sha(train),'test_index_sha256':array_sha(test),'positive_source_label':0,'test_class_counts':{'malignant':42,'benign':72}})
    configs=[];fits=[]
    circuit_path=ROOT/'circuits';circuit_path.mkdir(exist_ok=True)
    for scaler,st in [('StandardScaler','standard'),('MinMaxScaler','minmax')]:
        for selector,rt in [('Logistic Regression','lr'),('Random Forest','rf')]:
            for encoder,topology in [('Z',''),('ZZ','linear'),('ZZ','full'),('ZZ','circular')]:
                for freps in (1,2):
                    for areps in (1,2):
                        cid=f'vqc-k04-{st}-{rt}-{encoder.lower()}{"-"+topology if topology else ""}-f{freps}-a{areps}'
                        row={'configuration_id':cid,'scaler':scaler,'rfe_estimator':selector,'preprocessing_id':f'fs-k04-{st}-{rt}','encoder':encoder,'topology':topology,'feature_map_reps':freps,'ansatz_reps':areps,'ansatz_entanglement':'reverse_linear','final_rotation_layer':True,'qubits':4}
                        fmap,ansatz=circuits(row)
                        portable=[QuantumCircuit(4).compose(circuit) for circuit in (fmap,ansatz,fmap.compose(ansatz))]
                        with (circuit_path/f'{cid}.qpy').open('wb') as handle:qpy.dump(portable,handle)
                        row['circuit_path']=f'circuits/{cid}.qpy';row['circuit_sha256']=sha(circuit_path/f'{cid}.qpy')
                        row['input_parameter_order']=[str(p) for p in fmap.parameters]
                        row['weight_parameter_order']=[str(p) for p in ansatz.parameters]
                        row['circuit_operations']=[{'name':inst.operation.name,'qubits':[fmap.compose(ansatz).decompose().find_bit(q).index for q in inst.qubits],'parameters':[str(p) for p in inst.operation.params]} for inst in fmap.compose(ansatz).decompose().data]
                        configs.append(row)
                        for pair,initial_seed,sampler_seed in SEEDS:
                            initial=np.random.default_rng(initial_seed).uniform(0,1,ansatz.num_parameters)
                            fits.append({**row,'fit_id':f'{cid}-s{pair}','seed_pair_id':pair,'initialization_seed':initial_seed,'sampler_seed':sampler_seed,'initial_parameters':initial.tolist(),'initial_parameter_sha256':array_sha(initial)})
    assert len(configs)==64 and len(fits)==192
    policy={
        'implementation_version':VERSION,'budget':100,'budget_unit':'objective evaluations','loss':'cross_entropy','loss_log_base':2,'shots':1024,'warm_start':False,
        'optimizer':'COBYLA','rhobeg':1.0,'tol':None,'maxiter_meaning':'SciPy 1.15 COBYLA maxiter is passed as maxfun, the maximum number of function evaluations',
        'decision_rule':'argmax of one final probability array in source-class order [0,1]; ties choose first column, malignant source class 0',
        'primary_ranking':'mean test accuracy of exactly three valid seed outcomes','accuracy_ties':'retain all exact ties using sum of correct counts across three equal 114-case test sets; no metric tie-break',
        'incomplete_configurations':'mean/std/min/max and mean clinical metrics unavailable until all three valid outcomes exist; show counts, failures and partial outcomes separately; excluded from shortlist',
        'seed_pairs':SEEDS,'sampler_rng':'fresh StatevectorSampler(default_shots=1024, seed=integer sampler_seed) per fit; installed implementation resets default_rng(integer_seed) per bound circuit; common random uniforms across samples and evaluations preserved',
        'global_rngs':'numpy.random.seed(initialization_seed), qiskit_algorithms.utils.algorithm_globals.random_seed and qiskit_machine_learning.utils.algorithm_globals.random_seed set per fit; initial point from separate default_rng(initialization_seed)',
        'exact_optimizer_resume_supported':False,'exact_sampler_resume_supported':'integer sampler seed reproduces each circuit; COBYLA internal optimizer state is not serializable for exact continuation',
        'interruption_rule':'preserve interrupted attempt and restart from original saved vector and seed pair under a new attempt ID; never resume from final angles alone',
        'technical_retry_rule':'at most one automatic retry for OSError, MemoryError, TimeoutError, BrokenProcessPool or serialization/I/O failure; preserved new attempt; no retry for budget exhaustion, poor scores, numerical validation or scientific failures',
        'historical_differences':'explicit reverse_linear RealAmplitudes connectivity and final layer; ansatz reps 1 and 2 versus historical reps 1; 3 declared seed pairs with explicit uniform initial vectors versus uncontrolled defaults; explicit shots 1024 and cross entropy; training-only preprocessing',
        'interpretation':'basis_state % 2 (least-significant measured bit), not Hamming-weight parity; OneHotEncoder sorts source classes [0,1]; column 0 malignant',
        'test_access':'test inputs attach only after fitting; final predict_proba once per fit on test; all labels/predictions derived from that array; no test-driven training changes',
        'benchmark_ids':[fits[0]['fit_id'],next(f['fit_id'] for f in fits if f['scaler']=='StandardScaler' and f['rfe_estimator']=='Logistic Regression' and f['encoder']=='ZZ' and f['topology']=='full' and f['feature_map_reps']==2 and f['ansatz_reps']==2 and f['seed_pair_id']==1)],
    }
    manifest={'protocol':policy,'configurations':configs,'fits':fits}
    atomic_json(ROOT/'manifest.json',manifest)
    pd.DataFrame([{k:v for k,v in row.items() if not isinstance(v,(list,dict))} for row in fits]).to_csv(ROOT/'manifest.csv',index=False)
    pkgs={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','qiskit','qiskit-machine-learning','qiskit-algorithms','psutil','dill','threadpoolctl']}
    vm=psutil.virtual_memory()
    atomic_json(records/'environment.json',{'python':sys.version,'executable':sys.executable,'packages':pkgs,'platform':platform.platform(),'physical_cores':psutil.cpu_count(logical=False),'logical_processors':psutil.cpu_count(),'ram_total':vm.total,'ram_available':vm.available,'cpu_percent':psutil.cpu_percent(interval=1,percpu=True)})
    snapshot={'StatevectorSampler':inspect.getsource(StatevectorSampler._run_pub),'COBYLA_minimize':inspect.getsource(COBYLA.minimize),'VQC_readout':inspect.getsource(VQC._get_interpret)}
    atomic_json(records/'installed_source_semantics.json',snapshot)
    atomic_json(records/'manifest_lock.json',{'manifest_sha256':sha(ROOT/'manifest.json'),'fits':192,'configurations':64,'preprocessing_verified':True})
    print('LOCKED 64 configurations / 192 fits; four preprocessing caches verified')

if __name__=='__main__': main()
