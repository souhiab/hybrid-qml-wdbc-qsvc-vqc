import json
import time
import warnings
import numpy as np
from qiskit.circuit.library import PauliFeatureMap
from qiskit.quantum_info import Statevector
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from vqc_engine import ROOT,atomic_json,CachedStatevectorQNN
from prepare_vqc import circuits

def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    rng=np.random.default_rng(20261107)
    inputs=np.vstack([rng.uniform(0,1,(2,4)),rng.normal(0,2,(2,4))])
    checks=[];alias_checks=[]
    for topology in ('linear','full','circular'):
        for reps in (1,2):
            row={'encoder':'ZZ','topology':topology,'feature_map_reps':reps,'ansatz_reps':1}
            fmap,_=circuits(row)
            alias=PauliFeatureMap(4,reps=reps,entanglement=topology,paulis=['Z','ZZ'],alpha=2.0)
            errors=[]
            for x in inputs:
                a=Statevector.from_instruction(fmap.assign_parameters(x)).data
                b=Statevector.from_instruction(alias.assign_parameters(x)).data
                errors.append(float(np.max(np.abs(a-b))))
            alias_checks.append({'topology':topology,'reps':reps,'max_state_error':max(errors)})
    unique={tuple(row[key] for key in ('encoder','topology','feature_map_reps','ansatz_reps')):row for row in manifest['configurations']}
    for row in unique.values():
        fmap,ansatz=circuits(row)
        weights=rng.uniform(0,1,ansatz.num_parameters)
        for seed in (1101,1211,1307):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                reference=SamplerQNN(circuit=fmap.compose(ansatz),sampler=StatevectorSampler(default_shots=1024,seed=seed),input_params=list(fmap.parameters),weight_params=list(ansatz.parameters),interpret=lambda x:x%2,output_shape=2)
                candidate=CachedStatevectorQNN(fmap,ansatz,StatevectorSampler(default_shots=1024,seed=seed),seed)
            start=time.perf_counter();expected=reference.forward(inputs,weights);ref_seconds=time.perf_counter()-start
            start=time.perf_counter();actual=candidate.forward(inputs,weights);fast_seconds=time.perf_counter()-start
            error=float(np.max(np.abs(expected-actual)))
            if error!=0:raise RuntimeError(f'Seeded finite-shot outputs differ: {row} seed={seed}, error={error}')
            checks.append({'encoder':row['encoder'],'topology':row['topology'],'feature_map_reps':row['feature_map_reps'],'ansatz_reps':row['ansatz_reps'],'sampler_seed':seed,'rows':len(inputs),'sampled_probability_max_error':error,'reference_seconds':ref_seconds,'optimized_seconds':fast_seconds})
    total_ref=sum(row['reference_seconds'] for row in checks);total_fast=sum(row['optimized_seconds'] for row in checks)
    result={'passed':True,'finite_shot_cases':len(checks),'probability_max_error':max(c['sampled_probability_max_error'] for c in checks),'alias_max_state_error':max(c['max_state_error'] for c in alias_checks),'reference_seconds':total_ref,'optimized_seconds':total_fast,'speedup':total_ref/total_fast,'precision':'complex128 / float64','shots':1024,'checks':checks,'alias_checks':alias_checks,'readout':'integer basis state modulo 2','tie_rule':'argmax, column 0 malignant wins exact ties'}
    assert result['alias_max_state_error']<1e-12
    atomic_json(ROOT/'records/simulator_validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','alias_checks')}))

if __name__=='__main__': main()
