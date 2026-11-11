from pathlib import Path
from contextlib import contextmanager
import datetime,hashlib,json,math,os,socket,time,uuid

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def convert(value):
    if hasattr(value,'tolist'):return value.tolist()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),default=convert,allow_nan=False)
def identity(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,default=convert,allow_nan=False),encoding='utf-8')
    for attempt in range(40):
        try:os.replace(temp,path);return
        except PermissionError:
            if attempt==39:raise
            time.sleep(min(.02*(attempt+1),.5))
@contextmanager
def runner_lock(directory):
    import psutil
    path=Path(directory)/'runner.lock';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        previous=read(path)
        if previous.get('host')!=socket.gethostname():raise RuntimeError('Runner lock belongs to another host')
        if psutil.pid_exists(previous['pid']):
            if abs(psutil.Process(previous['pid']).create_time()-previous['create_time'])<.001:raise RuntimeError('A runner already owns this experiment')
        path.rename(path.with_name('runner.stale.'+uuid.uuid4().hex+'.json'))
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:json.dump({'pid':os.getpid(),'create_time':psutil.Process().create_time(),'host':socket.gethostname()},f)
    try:yield
    finally:
        if path.exists() and read(path)['pid']==os.getpid():path.unlink()
