from qiskit.circuit.library import ZFeatureMap,ZZFeatureMap,RealAmplitudes

def circuits(row):
    k=int(row['qubits'])
    fmap=ZFeatureMap(k,reps=int(row['feature_map_reps'])) if row['encoder']=='Z' else ZZFeatureMap(k,reps=int(row['feature_map_reps']),entanglement=row['topology'])
    ansatz=RealAmplitudes(k,reps=int(row['ansatz_reps']),entanglement='reverse_linear',skip_final_rotation_layer=False)
    return fmap,ansatz
