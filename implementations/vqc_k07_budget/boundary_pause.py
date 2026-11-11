import datetime
import json
import time
from pathlib import Path
import psutil
from vqc_engine import atomic_json

root=Path(__file__).resolve().parent
manifest=json.loads((root/'studies/internal-budget-pilot-k11/manifest.json').read_text())
rows=[r for r in manifest['fits'] if r['budget']==100]
controller=json.loads((root/'controller.lock').read_text())['pid']
while True:
    if not psutil.pid_exists(controller):raise RuntimeError('Controller exited before requested boundary')
    ready=True
    for row in rows:
        receipt=Path(row['study_root'])/'artifacts'/row['artifact_key']/'receipt.json'
        if not receipt.exists() or json.loads(receipt.read_text())['status']!='completed':ready=False;break
    if ready:
        process=psutil.Process(controller)
        assert 'sequence_controller.py' in ' '.join(process.cmdline())
        children=process.children()
        process.kill()
        for child in children:
            try:
                if child.name().lower().startswith(('python','node')):child.kill()
            except psutil.NoSuchProcess:pass
        record={'paused_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'User-requested optimization boundary after all 48 eleven-qubit 100-cap receipts were saved','controller_pid':controller,'verified_receipt_status_count':48,'completed_fits_retrained':False}
        atomic_json(root/'records/optimization_boundary_pause.json',record)
        status=json.loads((root/'status.json').read_text());status.update({'execution_state':'paused_for_optimization','controller_active':False,'phase':'internal-budget-pilot-k11-b100-complete','pause_record':'records/optimization_boundary_pause.json'})
        atomic_json(root/'status.json',status)
        break
    time.sleep(.5)
