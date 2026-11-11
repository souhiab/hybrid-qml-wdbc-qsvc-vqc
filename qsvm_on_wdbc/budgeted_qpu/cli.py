"""Explicit entry points. Importing or --help cannot train or submit."""
import argparse,json
from pathlib import Path
from .io import read,atomic

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','verify','local-exact','local-shots','backend-plan','pilot','execute','resume','analyze','status'])
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--cohort-size',type=int,default=72,choices=[72,96]);parser.add_argument('--qubits',type=int,default=4,choices=[4,7])
    parser.add_argument('--backend',default='ibm_fez');parser.add_argument('--allow-qpu-submission',action='store_true')
    args=parser.parse_args();d=args.run_dir
    if args.command=='prepare':
        from .data import prepare
        from .protocol import protocol
        result=prepare(d,protocol(args.cohort_size,args.qubits,args.backend))
    elif args.command=='verify':
        from .data import verify
        result=verify(d)
    elif args.command.startswith('local-'):
        from .simulation import run_local
        result=run_local(d,args.command.split('-')[1])
    elif args.command=='backend-plan':
        from .backend import plan
        full=plan(d)
        result={k:full[k] for k in ['backend','layout','planned_reserved_seconds','remaining_account_seconds_at_plan','locally_estimated_feasible','hardware_jobs_submitted']}
    elif args.command in ['pilot','execute','resume']:
        from .hardware import run
        result=run(d,'execute' if args.command=='resume' else args.command,args.allow_qpu_submission)
    elif args.command=='analyze':
        from .results import clinical
        import numpy as np
        result={}
        for name in ['local_exact','local_shots','hardware']:
            path=d/name/'kernels_predictions.npz'
            if path.exists():
                with np.load(path) as a:result[name]=clinical(a['source_labels'],a['predictions'],a['malignant_probabilities'])
        atomic(d/'comparison.json',result)
    else:
        result={'preparation':read(d/'status.json')}
        for name in ['local_exact','local_shots','hardware']:
            result[name]={'complete':(d/name/'COMPLETED.json').exists()}
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
