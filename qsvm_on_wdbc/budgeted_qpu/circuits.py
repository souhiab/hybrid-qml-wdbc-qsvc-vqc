from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit,qpy
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import Statevector
from .io import atomic,read,file_hash,identity

def feature_map(p,prefix='x'):
    e=p['encoder']
    if e['constructor']!='ZFeatureMap' or e['entanglement_argument'] is not None:raise ValueError('Only the explicitly configured non-entangling Z encoder is supported')
    circuit=ZFeatureMap(feature_dimension=e['qubits'],reps=e['reps'],parameter_prefix=prefix)
    if circuit.alpha!=e['alpha']:raise ValueError('Installed ZFeatureMap default alpha differs from frozen alpha')
    return circuit
def template(p):
    left=feature_map(p,'a');right=feature_map(p,'b');circuit=left.compose(right.inverse());circuit.measure_all()
    return circuit
def pair_values(pair,train,test):
    return (train[pair['i']],train[pair['j']]) if pair['kind']=='train' else (test[pair['i']],train[pair['j']])
def bindings(circuit,pairs,train,test):
    k=train.shape[1];order=[str(v) for v in circuit.parameters]
    expected={f'{prefix}[{q}]' for prefix in ['a','b'] for q in range(k)}
    if set(order)!=expected:raise ValueError('Transpilation changed the parameter set')
    result=[]
    for pair in pairs:
        a,b=pair_values(pair,train,test);mapping={**{f'a[{q}]':a[q] for q in range(k)},**{f'b[{q}]':b[q] for q in range(k)}}
        result.append([mapping[name] for name in order])
    return np.asarray(result,dtype=float)
def logical_state(p,values):
    fmap=feature_map(p);return Statevector.from_instruction(fmap.assign_parameters(dict(zip(fmap.parameters,values)))).data
def overlap_state(circuit,values):
    circuit=circuit.remove_final_measurements(inplace=False)
    return Statevector.from_instruction(circuit.assign_parameters(dict(zip(circuit.parameters,values)))).data
def persist_template(directory,p):
    directory=Path(directory);path=directory/'logical_template.qpy'
    if not path.exists():
        circuit=template(p)
        with path.open('wb') as f:qpy.dump(circuit,f)
        atomic(directory/'logical_template.json',{'protocol_hash':identity(p),'parameter_order':[str(v) for v in circuit.parameters],'measurement_register':'meas','compute_uncompute':'U_b inverse after U_a','alpha':2.0,'extra_pi_rescaling':False,'qpy_sha256':file_hash(path)})
    metadata=read(directory/'logical_template.json');assert metadata['protocol_hash']==identity(p) and metadata['qpy_sha256']==file_hash(path)
    with path.open('rb') as f:circuit=qpy.load(f)[0]
    return circuit
