from __future__ import annotations
import argparse
from pathlib import Path
from common import *

def register_vqc(path):
    supplied=read(path)
    assert supplied.get('frozen') is True and supplied.get('candidates'),'A later frozen VQC shortlist is required'
    assert supplied.get('seed_pairs')==[[101,1101],[211,1211],[307,1307]],'Exactly the current three seed pairs are required'
    assert supplied.get('fold_manifest_sha256')==sha(ROOT/'folds/outer_inner_assignments.json')
    for candidate in supplied['candidates']:
        assert candidate['candidate_id'].startswith('V') and all(key in candidate for key in ['qubits','feature_map','feature_map_reps','ansatz_reps','scaler','selector','optimizer_budget'])
    atomic(ROOT/'manifests/vqc_shortlist_registered.json',supplied)
    plan=[{'candidate_id':c['candidate_id'],'repeat_id':f['repeat_id'],'outer_fold_index':f['outer_fold_index'],'initialization_seed':pair[0],'sampler_seed':pair[1],'evaluation_id':f"{c['candidate_id']}-{fold_id(f)}-i{pair[0]}-s{pair[1]}",'train_indices':f['train_indices'],'test_indices':f['test_indices']} for c in supplied['candidates'] for f in read(ROOT/'folds/outer_inner_assignments.json')['outer_records'] for pair in supplied['seed_pairs']]
    atomic(ROOT/'manifests/vqc_shared_fold_plan.json',plan)
    print('VQC shortlist registered to the same frozen folds. Registration performs no training; the shortlist-specific VQC adapter must be validated separately.')

def main():
    parser=argparse.ArgumentParser(description='Prepared WDBC QSVC/classical shared-fold framework. No training on import or --help.')
    sub=parser.add_subparsers(dest='command',required=True)
    for cmd in ['verify','folds','resources','audit-reuse','status','synthetic-tests','export']:sub.add_parser(cmd)
    analysis=sub.add_parser('consolidate');analysis.add_argument('--uncertainty',action='store_true');analysis.add_argument('--require-complete',action='store_true')
    for cmd in ['execute','resume']:
        run=sub.add_parser(cmd);run.add_argument('--kind',choices=['qsvc','classical','all'],default='all');run.add_argument('--candidate',default='all');run.add_argument('--workers',type=int,default=1)
    vqc=sub.add_parser('add-vqc');vqc.add_argument('--file',required=True)
    args=parser.parse_args()
    if args.command=='verify':
        from verify import verify
        verify();print('Frozen preparation verified. No training performed.')
    elif args.command=='folds':
        from prepare import verify_folds
        folds,_=verify_folds();print(f"VERIFIED: {len(folds['outer_records'])} shared outer partitions; exact historical seeds and patient mapping.")
    elif args.command=='resources':
        from controller import resources
        print(json.dumps(resources(),indent=2))
    elif args.command=='audit-reuse':
        from reuse_audit import run
        run()
    elif args.command in ('execute','resume'):
        from controller import run
        run(args.kind,args.candidate,args.workers)
    elif args.command=='consolidate':
        from analysis import consolidate
        consolidate(verify=True,uncertainty=args.uncertainty)
        if args.require_complete:assert read(ROOT/'records/consolidation_verification.json')['complete_outer_evaluations']==255,'Evaluation coverage is incomplete; inspect tables/Coverage.csv'
        print('Saved prediction metrics and coverage verified; no models trained.')
    elif args.command=='export':
        from controller import ExcelWriter
        lock=ROOT/'controller.lock'
        if lock.exists():
            import psutil
            assert not psutil.pid_exists(read(lock)['pid']),'The coordinator is active; it is the sole Excel writer'
        atomic(ROOT/'workbook_support/excel_snapshot.json',read(ROOT/'workbook_support/tables_snapshot.json'))
        writer=ExcelWriter();a=writer.command('sync');b=writer.command('final')
        assert a and b and b['allPassed'];print('Workbook saved and readback verified.')
    elif args.command=='status':
        print(json.dumps(read(ROOT/'status.json'),indent=2))
    elif args.command=='synthetic-tests':
        from synthetic_tests import run
        run()
    elif args.command=='add-vqc':register_vqc(args.file)

if __name__=='__main__':main()
