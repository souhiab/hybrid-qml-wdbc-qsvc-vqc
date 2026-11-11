import queue,threading,copy,json,subprocess
from pathlib import Path
from vqc_engine import atomic_json
ROOT=Path(__file__).resolve().parent
class Writer:
    def __init__(self):
        self.queue=queue.Queue(maxsize=4);self.process=None;self.error=None
        self.thread=threading.Thread(target=self.work,daemon=True);self.thread.start()
    def work(self):
        while True:
            item=self.queue.get()
            try:
                if item is None:return
                command,payload=item
                atomic_json(ROOT/'workbook_support/excel_snapshot.json',payload)
                answer=None
                for attempt in range(3):
                    try:
                        if self.process is None or self.process.poll() is not None:
                            stderr=(ROOT/'records/workbook_stderr.log').open('a',encoding='utf-8')
                            self.process=subprocess.Popen(['node',str(ROOT/'workbook_support/workbook_writer.mjs')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,encoding='utf-8')
                        self.process.stdin.write(command+'\n');self.process.stdin.flush()
                        while True:
                            line=self.process.stdout.readline()
                            if not line:raise OSError('Workbook writer exited')
                            if line.startswith('@@'):
                                answer=json.loads(line[2:])
                                if not answer['ok']:raise OSError(answer['error'])
                                break
                        self.error=None;break
                    except Exception as error:
                        self.error=str(error)
                        if self.process and self.process.poll() is None:self.process.terminate();self.process.wait()
                        self.process=None
                atomic_json(ROOT/'records/workbook_writer_status.json',{'last_command':command,'error':self.error,'receipts_preserved':True})
            finally:self.queue.task_done()
    def submit(self,payload,final=False):self.queue.put(('final' if final else 'sync',copy.deepcopy(payload)))
    def finish(self,payload):
        self.submit(payload,True);self.queue.join()
        if self.error:raise OSError(self.error)
        assert json.loads((ROOT/'records/workbook_readback.json').read_text())['allPassed']
