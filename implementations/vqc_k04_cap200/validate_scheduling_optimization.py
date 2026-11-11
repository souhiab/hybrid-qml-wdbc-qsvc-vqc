import cProfile
import importlib.util
import json
import logging
import time
from pathlib import Path
import numpy as np
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from threadpoolctl import threadpool_limits
import vqc_engine as new
from prepare_pilot import circuits

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('before_scheduling_engine',ROOT/'records/scheduling_optimization_v5/vqc_engine.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
logging.getLogger('qiskit_machine_learning').setLevel(logging.ERROR)

def pair(k,m,fr,ar,rows):
    directory=ROOT/f'studies/internal-budget-pilot-k{k:02d}'
    with np.load(directory/'preprocessing/p0.npz',allow_pickle=False) as data:inputs=np.array(data['X_train'][:rows],copy=True)
    row={'qubits':k,'encoder':'Z' if m==0 else 'ZZ','topology':['','linear','full','circular'][m],'feature_map_reps':fr,'ansatz_reps':ar}
    fmap,ansatz=circuits(row)
    bank=np.asarray([Statevector.from_instruction(fmap.assign_parameters(x)).data for x in inputs])
    engines=[]
    for module in (old,new):
        qnn=module.CachedStatevectorQNN(fmap,ansatz,StatevectorSampler(default_shots=1024,seed=1101),1101)
        qnn.encoder_banks[module.array_sha(inputs)]=bank
        engines.append(qnn)
    return inputs,ansatz,engines

def main():
    checks=0
    with threadpool_limits(limits=1):
        for k in (7,11,17):
            for m in range(4):
                for fr,ar in ((1,1),(2,2)):
                    inputs,ansatz,(a,b)=pair(k,m,fr,ar,2)
                    weights=np.random.default_rng(101).uniform(0,1,ansatz.num_parameters)
                    for seed in (1101,1211,1307):
                        a.sampler_seed=b.sampler_seed=seed
                        np.testing.assert_array_equal(a.forward(inputs,weights),b.forward(inputs,weights));checks+=1
        inputs,ansatz,(a,b)=pair(11,1,1,2,364)
        vectors=np.random.default_rng(211).uniform(0,1,(12,ansatz.num_parameters))
        for vector in vectors:np.testing.assert_array_equal(a.forward(inputs,vector),b.forward(inputs,vector))
        timings=[]
        for name,qnn in [('before',a),('candidate',b),('before',a),('candidate',b)]:
            started=time.perf_counter()
            for vector in vectors:qnn.forward(inputs,vector)
            timings.append({'engine':name,'seconds':time.perf_counter()-started,'forward_calls':len(vectors)})
        profiler=cProfile.Profile();profiler.enable();b.forward(inputs,vectors[0]);profiler.disable()
        profile=[]
        for entry in sorted(profiler.getstats(),key=lambda s:s.totaltime,reverse=True)[:12]:
            code=entry.code
            profile.append({'function':code if isinstance(code,str) else code.co_name,'calls':entry.callcount,'total_seconds':entry.totaltime,'self_seconds':entry.inlinetime})
        before=np.median([r['seconds'] for r in timings if r['engine']=='before'])
        after=np.median([r['seconds'] for r in timings if r['engine']=='candidate'])
        report={'passed':True,'finite_shot_probability_checks':checks,'full_364_row_probability_checks':12,'probabilities_bitwise_identical':True,'optimizer_fits_executed':0,'optimizer_loss_shots_precision_seeds_circuits_unchanged':True,'timings':timings,'candidate_speedup_factor':float(before/after),'retain_inner_loop_change':bool(before/after>=1.02),'profile':profile,'engine_sha256':new.sha(ROOT/'vqc_engine.py'),'candidate':'Bounded read-only cache of exact controlled-NOT index permutations; no floating-point calculation changed.'}
        new.atomic_json(ROOT/'records/scheduling_numerical_validation.json',report)
        print(json.dumps({k:report[k] for k in ['passed','finite_shot_probability_checks','candidate_speedup_factor','retain_inner_loop_change']}),flush=True)

if __name__=='__main__':main()
