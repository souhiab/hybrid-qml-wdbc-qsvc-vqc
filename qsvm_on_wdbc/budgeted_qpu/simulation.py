from pathlib import Path
import time
import numpy as np
from .io import read,atomic,file_hash,identity,runner_lock
from .data import verify,preprocess
from .circuits import logical_state,persist_template,bindings,overlap_state
from .results import counts_record,matrices,score

def run_local(directory,mode='exact'):
    directory=Path(directory)
    with runner_lock(directory):
        verify(directory);preprocess(directory);p=read(directory/'protocol.json');pairs=read(directory/'pairs.json');circuit=persist_template(directory,p)
        inputs=np.load(directory/'preprocessed.npz');train=inputs['X_train'];test=inputs['X_test'];started=time.perf_counter()
        out=directory/('local_'+mode);out.mkdir(exist_ok=True)
        lock={'protocol':identity(p),'inputs':file_hash(directory/'preprocessed.npz'),'circuit':file_hash(directory/'logical_template.qpy'),'mode':mode}
        if (out/'COMPLETED.json').exists():
            r=read(out/'COMPLETED.json');assert r['identity']==identity(lock)
            for name,digest in r['files'].items():assert file_hash(out/name)==digest
            return read(out/'metrics.json')
        if mode=='exact':
            states_train=np.asarray([logical_state(p,x) for x in train]);states_test=np.asarray([logical_state(p,x) for x in test])
            K_train=np.abs(states_train.conj()@states_train.T)**2;K_test=np.abs(states_test.conj()@states_train.T)**2;np.fill_diagonal(K_train,1)
        elif mode=='shots':
            records={};values=bindings(circuit,pairs,train,test);shots=p['measurement']['shots'];seed=p['simulation']['seed']
            for pair,value in zip(pairs,values):
                state=overlap_state(circuit,value);prob=np.abs(state)**2;prob/=prob.sum()
                rng=np.random.default_rng(int(identity({'seed':seed,'pair':pair['id']})[:16],16));sample=rng.multinomial(shots,prob)
                counts={format(i,f'0{p["encoder"]["qubits"]}b'):int(n) for i,n in enumerate(sample) if n}
                records[pair['id']]={'pair':pair,**counts_record(counts,shots,p['encoder']['qubits']),'simulator':'noiseless finite-shot compute-uncompute','seed_rule':'SHA256(seed,pair_id) first64bits'}
            atomic(out/'raw_counts.json',records);K_train,K_test=matrices(records,pairs,len(train),len(test))
        else:raise ValueError('Use exact or shots; backend-noise simulation is an explicit separate command')
        metrics,spectrum=score(out,p,K_train,K_test,inputs['y_train'],inputs['y_test'],inputs['test_ids'])
        atomic(out/'runtime.json',{'local_simulation_and_model_seconds':time.perf_counter()-started,'hardware_execution':False,'qpu_jobs_submitted':0,'kernel_pairs':len(pairs),'shot_executions':len(pairs)*p['measurement']['shots'] if mode=='shots' else 0})
        atomic(out/'COMPLETED.json',{'identity':identity(lock),'files':{f.name:file_hash(f) for f in out.iterdir() if f.is_file() and f.name!='COMPLETED.json'},'pair_coverage':len(pairs),'n_test':len(test)})
        return metrics
