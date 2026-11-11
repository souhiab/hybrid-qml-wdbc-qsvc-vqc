"""Adaptive pilot scheduling with isolated, single-threaded fit processes."""
import json
import multiprocessing
import time
from pathlib import Path
import pandas as pd
import psutil

MAX_WORKERS = 6

def vscode_open():
    for p in psutil.process_iter(['name']):
        try:
            if (p.info['name'] or '').lower() == 'code.exe': return True
        except (psutil.NoSuchProcess, psutil.AccessDenied): return True
    return False

def adjust(target, closed, healthy, pressure, stable, since_change):
    if pressure and since_change >= 60: return max(1, target-1), 'memory_pressure'
    if healthy and stable >= 600 and since_change >= 600:
        return min(MAX_WORKERS, target+1), 'stable_resources'
    return target, 'holding'

def worker(row, attempt, concurrency):
    from fit_worker import fit_one
    r = fit_one(row, attempt, concurrency)
    if r['status'] != 'completed': raise RuntimeError(str(r.get('error', 'fit failed')))

def run_adaptive(root, budget, rows, results, validate, atomic):
    root = Path(root)
    context = multiprocessing.get_context('spawn')
    pending = [r for r in rows if r['fit_id'] not in results]
    active = {}
    retries = {}
    target = MAX_WORKERS
    changed = healthy_since = time.monotonic()
    psutil.cpu_percent(interval=None)
    def event(reason):
        with (root/'records/scaling_events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps({'time':time.time(), 'target':target, 'reason':reason})+'\n')
    try:
        while pending or active:
            now = time.monotonic()
            vm = psutil.virtual_memory()
            reserve = max(2_000_000_000, int(.15*vm.total))
            allowance = 1_600_000_000
            # Live execution-only policy; hard ceiling six and one thread per worker.
            policy_file = root/'records/worker_policy.json'
            policy = json.loads(policy_file.read_text()) if policy_file.exists() else {}
            maximum = max(1,min(MAX_WORKERS,int(policy.get('maximum_workers',MAX_WORKERS))))
            target = min(target,maximum)
            closed = not vscode_open()
            cpu = psutil.cpu_percent(interval=None)
            pressure = vm.available < reserve
            healthy = vm.available >= reserve+allowance and cpu < 95
            if not healthy: healthy_since = now
            updated, reason = adjust(target, closed, healthy, pressure, now-healthy_since, now-changed)
            updated = min(updated,maximum)
            if updated != target:
                target = updated
                changed = healthy_since = now
                event(reason)
            for key, (process, row) in list(active.items()):
                if process.is_alive(): continue
                process.join()
                receipt = validate(row)
                if process.exitcode == 0 and receipt and receipt['status']=='completed':
                    results[key] = receipt
                else:
                    technical = receipt is None or receipt.get('technical_failure', False)
                    retries[key] = retries.get(key, 0)+1
                    attempts = list((root/'artifacts'/row['artifact_key']).glob('attempt_*/attempt.json'))
                    failures = sum(json.loads(p.read_text()).get('technical_failure', False) for p in attempts)
                    if not technical or failures >= 2 or retries[key] > 1:
                        raise RuntimeError('Fit requires review: '+key)
                    target = max(1, target-1)
                    changed = healthy_since = now
                    event('technical_worker_failure')
                    pending.insert(0, row)
                process.close()
                del active[key]
                pd.DataFrame([{'fit_id':v['fit_id'], 'budget':budget, 'termination':v['termination'],
                    'nfev':v['nfev'], 'runtime_seconds':v['runtime']['total_seconds'],
                    **v['metrics']['validation']} for v in results.values()]).to_csv(root/f'tables/budget_{budget}.csv', index=False)
            capacity = min(target, maximum, max(1, (psutil.cpu_count(logical=True) or 2)-1))
            while pending and len(active)<capacity and psutil.virtual_memory().available>=reserve+allowance and (len(active)<5 or cpu<90):
                row = pending.pop(0)
                attempts = sorted((root/'artifacts'/row['artifact_key']).glob('attempt_*/attempt.json'))
                for path in attempts:
                    record = json.loads(path.read_text())
                    if record['status']=='running':
                        if psutil.pid_exists(record['pid']): raise RuntimeError('Existing worker owns fit: '+row['fit_id'])
                        record.update(status='interrupted', error='Restart original initialization; no optimizer restoration')
                        atomic(path, record)
                if sum(json.loads(p.read_text()).get('technical_failure',False) for p in attempts)>=2:
                    raise RuntimeError('Technical retry limit reached: '+row['fit_id'])
                process = context.Process(target=worker, args=(row,len(attempts)+1,capacity))
                process.start()
                active[row['fit_id']] = (process,row)
                time.sleep(3)
            state = {'state':'running', 'budget':budget, 'completed':len(results), 'expected':len(rows),
                'target_workers':target, 'active_workers':len(active), 'maximum_workers':maximum,
                'vscode_open':not closed, 'available':vm.available, 'reserve':reserve,
                'launch_allowance':allowance, 'cpu_percent':cpu, 'threads':1,
                'sixth_worker_requires':'available RAM >= reserve + 1.6 GB; CPU below 90%; pending fit available',
                'ramp_seconds':600, 'down_seconds':60, 'updated_unix':time.time(),
                'reduction':'finish active fits; withhold replacement slots'}
            atomic(root/'records/resources_latest.json', state)
            atomic(root/'status.json', state)
            time.sleep(5)
    finally:
        # Preserve in-flight fits and wait for their atomic receipts on errors.
        for process,row in active.values(): process.join()
