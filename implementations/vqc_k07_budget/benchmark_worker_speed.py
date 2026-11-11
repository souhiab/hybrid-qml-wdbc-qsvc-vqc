from __future__ import annotations
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
import concurrent.futures
import datetime
import hashlib
import json
import multiprocessing
import time
from pathlib import Path
import numpy as np
import psutil
from qiskit.primitives import StatevectorSampler
from threadpoolctl import threadpool_limits
import vqc_engine as engine
from prepare_pilot import circuits

ROOT=Path(__file__).resolve().parent

def forward_task(payload):
    row,bankfile,chunk,repetitions=payload
    os.environ['VQC_CHUNK_ROWS']=str(chunk)
    with threadpool_limits(limits=1):
        directory=Path(row['study_root'])
        with np.load(directory/'preprocessing/p0.npz',allow_pickle=False) as data:inputs=np.array(data['X_train'],copy=True)
        fmap,ansatz=circuits(row)
        qnn=engine.CachedStatevectorQNN(fmap,ansatz,StatevectorSampler(default_shots=1024,seed=1101),1101)
        qnn.encoder_banks[engine.array_sha(inputs)]=np.load(bankfile,mmap_mode='r',allow_pickle=False)
        weights=np.random.default_rng(101).uniform(0,1,(repetitions,ansatz.num_parameters))
        started=time.perf_counter();fingerprints=[];peak=0
        for vector in weights:
            output=qnn.forward(inputs,vector)
            fingerprints.append(engine.array_sha(output));peak=max(peak,psutil.Process().memory_info().rss)
        return {'seconds':time.perf_counter()-started,'fingerprints':fingerprints,'peak_rss_bytes':peak}

def main():
    directory=ROOT/'studies/internal-budget-pilot-k11'
    manifest=json.loads((directory/'manifest.json').read_text())
    row=next(r for r in manifest['fits'] if r['p']==0 and r['feature_map_index']==1 and r['feature_map_reps']==1 and r['ansatz_reps']==2)
    with np.load(directory/'preprocessing/p0.npz',allow_pickle=False) as data:inputs=data['X_train']
    fmap,_=circuits(row)
    semantic=f'{fmap.num_qubits}-{fmap.name}-{fmap.reps}-{fmap.entanglement}-{engine.array_sha(inputs)}'
    bank=directory/'state_cache'/(hashlib.sha256(semantic.encode()).hexdigest()+'.npy')
    receipt=json.loads(bank.with_suffix('.json').read_text());assert engine.sha(bank)==receipt['sha256']
    controller=json.loads((ROOT/'controller.lock').read_text())['pid']
    suspended=[]
    try:
        # Preserve complete optimizer/RNG state by suspending, not interrupting fits.
        for process in psutil.Process(controller).children():
            if process.name().lower().startswith('python'):
                process.suspend();suspended.append(process)
        chunks=[]
        for chunk in (8,32,64,128,364):
            result=forward_task((row,str(bank),chunk,3));chunks.append({'chunk_rows':chunk,**result})
        assert all(r['fingerprints']==chunks[0]['fingerprints'] for r in chunks)
        best=min(chunks,key=lambda r:r['seconds'])['chunk_rows']
        parallel=[]
        for workers in (2,3):
            started=time.perf_counter()
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                results=list(pool.map(forward_task,[(row,str(bank),best,12)]*6))
            elapsed=time.perf_counter()-started
            assert all(r['fingerprints']==results[0]['fingerprints'] for r in results)
            parallel.append({'workers':workers,'wall_seconds_including_startup':elapsed,'forward_calls':72,'calls_per_second':72/elapsed,'peak_worker_rss_bytes':max(r['peak_rss_bytes'] for r in results),'minimum_available_ram_after':psutil.virtual_memory().available})
        selected=max(parallel,key=lambda r:r['calls_per_second'])['workers']
        result={'measured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'width':11,'training_inputs_only':True,'optimizer_fits_executed':0,'existing_fits_suspended_and_resumed_without_state_loss':len(suspended),'chunk_benchmarks':chunks,'parallel_benchmarks':parallel,'selected_chunk_rows':best,'selected_workers':selected,'all_probability_fingerprints_identical':True,'numerical_engine_modified':False,'physical_cores':psutil.cpu_count(logical=False),'logical_processors':psutil.cpu_count(),'available_ram':psutil.virtual_memory().available}
        engine.atomic_json(ROOT/'records/worker_speed_benchmark.json',result)
        print(json.dumps({k:result[k] for k in ['selected_chunk_rows','selected_workers','all_probability_fingerprints_identical','parallel_benchmarks']}),flush=True)
    finally:
        for process in suspended:
            try:process.resume()
            except psutil.NoSuchProcess:pass

if __name__=='__main__':main()
