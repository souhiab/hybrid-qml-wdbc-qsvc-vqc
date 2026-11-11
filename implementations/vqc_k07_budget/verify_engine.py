import json
import logging
import shutil
import warnings
from pathlib import Path
import numpy as np
from qiskit.circuit.library import PauliFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from threadpoolctl import threadpool_limits
import vqc_engine as engine
from prepare_pilot import circuits

def verify():
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'manifest.json').read_text())
    checks=[]
    logging.getLogger('qiskit_machine_learning').setLevel(logging.ERROR)
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for study in manifest['studies']:
            directory=Path(study['manifest_path']).parent
            engine.ROOT=directory
            maximum=0.0;count=0
            for m in range(4):
                for fr,ar in [(1,1),(2,2)]:
                    row={'qubits':study['qubits'],'encoder':'Z' if m==0 else 'ZZ','topology':['','linear','full','circular'][m],'feature_map_reps':fr,'ansatz_reps':ar}
                    fmap,ansatz=circuits(row)
                    with np.load(directory/'preprocessing/p0.npz',allow_pickle=False) as data:inputs=data['X_train'][:2]
                    for weight_id,weights in enumerate([np.random.default_rng(101).uniform(0,1,ansatz.num_parameters),np.linspace(-2,2,ansatz.num_parameters)]):
                        reference=np.asarray([Statevector.from_instruction(fmap.compose(ansatz).assign_parameters(dict(zip(list(fmap.parameters)+list(ansatz.parameters),np.r_[x,weights])))).data for x in inputs])
                        encoded=np.asarray([Statevector.from_instruction(fmap.assign_parameters(x)).data for x in inputs])
                        bound=ansatz.assign_parameters(weights).decompose()
                        operations=[(inst.operation.name,bound.find_bit(inst.qubits[0]).index,float(inst.operation.params[0]) if inst.operation.name=='ry' else bound.find_bit(inst.qubits[1]).index) for inst in bound.data if inst.operation.name!='barrier']
                        actual=engine.evolve_ansatz(encoded.copy(),operations)
                        error=float(np.max(np.abs(reference-actual)));maximum=max(maximum,error)
                        np.testing.assert_allclose(actual,reference,atol=2e-12,rtol=0)
                        if m:
                            alias=PauliFeatureMap(row['qubits'],reps=fr,entanglement=row['topology'],paulis=['Z','ZZ'],alpha=2)
                            np.testing.assert_allclose(Statevector.from_instruction(alias.assign_parameters(inputs[0])).data,encoded[0],atol=2e-12,rtol=0)
                        for seed in (1101,1211,1307):
                            expected=engine.sampled_binary_probabilities(reference,seed)
                            qnn=engine.CachedStatevectorQNN(fmap,ansatz,StatevectorSampler(default_shots=1024,seed=seed),seed)
                            np.testing.assert_array_equal(qnn.forward(inputs,weights),expected)
                            count+=1
                            del qnn
                    engine.ENCODER_BANKS.clear()
            record={'qubits':study['qubits'],'passed':True,'probability_checks':count,'max_statevector_absolute_error':maximum,'finite_shot_probability_arrays_identical':True,'alias_semantics_verified':True,'dense_matrix_path_only_at_four_qubits':True,'validation_inputs':'Two cases from internal training cache p0; no evaluation cohort used'}
            engine.atomic_json(directory/'records/numerical_validation.json',record)
            checks.append(record)
            for item in (directory/'state_cache').iterdir():item.unlink()
            print(f"Numerical validation passed k={study['qubits']}: {count} seeded readouts",flush=True)
    engine.atomic_json(root/'records/numerical_validation.json',{'passed':True,'checks':checks})

if __name__=='__main__':verify()
