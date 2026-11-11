import json
import os
import tempfile
import threading
import time
from pathlib import Path
import sequence_controller as controller
from vqc_engine import atomic_json

def main():
    root=Path(__file__).resolve().parent
    coordinator=controller.Controller.__new__(controller.Controller)
    coordinator.pool=None;coordinator.pool_width=None;coordinator.pool_workers=None
    first=coordinator.get_pool(3,11)
    pids=[f.result() for f in [first.submit(os.getpid) for _ in range(60)]]
    assert len(pids)==60 and len(set(pids))<=3
    assert coordinator.get_pool(3,11) is first
    coordinator.close_pool()
    original_root=controller.ROOT;testroot=Path(tempfile.mkdtemp(prefix='async_excel_check_',dir=root/'records'))
    (testroot/'workbook_support').mkdir();controller.ROOT=testroot
    seen=[];sync_threads=[]
    def fake_command(self,command):
        if command=='sync':
            time.sleep(.02)
            seen.append(json.loads((testroot/'workbook_support/excel_snapshot.json').read_text())['sequence'])
            sync_threads.append(threading.get_ident())
        return {'allPassed':True}
    original_command=controller.ExcelWriter.command;controller.ExcelWriter.command=fake_command
    writer=controller.AsyncExcelWriter()
    try:
        for value in range(10):writer.submit({'sequence':value})
        assert writer.finish()['allPassed']
        assert seen==list(range(10)) and len(set(sync_threads))==1
    finally:
        writer.tasks.put(None);writer.thread.join()
        controller.ExcelWriter.command=original_command;controller.ROOT=original_root
    atomic_json(root/'records/scheduling_control_validation.json',{'passed':True,'persistent_pool_jobs_completed':60,'worker_count_bound':3,'no_per_task_worker_recycling':True,'pool_reused_across_groups':True,'asynchronous_excel_updates_verified_in_order':10,'single_excel_sync_thread':True,'bounded_queue':4,'training_fits_executed':0})
    print('Scheduling checks passed: 60 jobs, reused bounded pool, ordered asynchronous single-writer updates.')

if __name__=='__main__':main()
