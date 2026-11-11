"""Read-only account/calibration queries and local transpilation. Never submits a job."""
from pathlib import Path
import math
from qiskit import transpile,qpy
from .io import read,atomic,identity,file_hash,now
from .data import preprocess
from .circuits import persist_template
from .ledger import BudgetError,number

def connect():
    from qiskit_ibm_runtime import QiskitRuntimeService
    path=Path.home()/'.config/wdbc_ibm_quantum/credentials.json'
    credentials=read(path)
    token=credentials.get('api_key');instance=credentials.get('instance_crn')
    if not token or not instance:raise BudgetError('Explicit credentials and instance CRN are required')
    service=QiskitRuntimeService(channel='ibm_quantum_platform',token=token,instance=instance)
    matches=[item for item in service.instances() if item.get('crn')==instance]
    if len(matches)!=1 or str(matches[0].get('plan','')).lower()!='open':
        raise BudgetError('Selected explicit instance is not verified as Open Plan; no fallback is allowed')
    return service,identity(instance)

def remaining(service):
    usage=service.usage()
    value=usage.get('usage_remaining_seconds')
    if value is None:
        limit=usage.get('usage_limit_seconds',usage.get('usage_allocation_seconds'))
        consumed=usage.get('usage_consumed_seconds')
        value=number(limit,'usage limit')-number(consumed,'consumed usage')
    return number(value,'remaining usage')

def job_allowance(base,p,ratio=1.0):
    b=p['budget']
    return math.ceil(max(b['minimum_job_allowance_seconds'],base*max(1.,ratio)*b['estimate_safety_factor']+b['extra_job_margin_seconds']))

def plan(directory):
    directory=Path(directory);preprocess(directory);p=read(directory/'protocol.json')
    if (directory/'backend_plan.json').exists():
        existing=read(directory/'backend_plan.json')
        if existing['inputs']!=identity(read(directory/'input_lock.json')) or existing['preprocessing']!=file_hash(directory/'preprocessed.npz'):
            raise BudgetError('Frozen backend plan differs from current inputs')
        if file_hash(directory/'isa_template.qpy')!=existing['isa_sha256']:raise BudgetError('ISA circuit integrity failure')
        return existing
    service,instance_hash=connect();available=remaining(service);backend=service.backend(p['hardware']['backend'])
    if not backend.status().operational:raise BudgetError('Backend is not operational')
    props=backend.properties(refresh=True);bad=set(props.faulty_qubits());rank=[]
    for q in range(backend.num_qubits):
        if q in bad:continue
        try:
            if not props.is_gate_operational('sx',[q]):continue
            ro=props.readout_error(q);sx=props.gate_error('sx',[q]);number(ro,'readout error');number(sx,'SX error')
            rank.append((ro+sx,ro,sx,q))
        except Exception:continue
    rank.sort();k=p['encoder']['qubits']
    if len(rank)<k:raise BudgetError('Insufficient calibrated operational qubits')
    layout=[r[3] for r in rank[:k]]
    circuit=transpile(persist_template(directory,p),backend=backend,initial_layout=layout,
        optimization_level=p['hardware']['optimization_level'],seed_transpiler=p['hardware']['transpiler_seed'],approximation_degree=1.0)
    if any(len(item.qubits)>1 and item.operation.name!='barrier' for item in circuit.data):raise BudgetError('Unexpected two-qubit instruction in Z-only circuit')
    duration=float(circuit.estimate_duration(backend.target,unit='s'))
    repetition=float(backend.configuration().default_rep_delay)
    # Include reset explicitly even if scheduling estimate omitted it; conservative double counting is safe.
    resets=[backend.target['reset'][(q,)].duration for q in layout]
    if any(v is None for v in resets):raise BudgetError('Unknown reset timing; cannot estimate hardware budget')
    per_shot=number(duration,'scheduled duration')+number(repetition,'default repetition delay')+max(resets)
    with (directory/'isa_template.qpy').open('wb') as f:qpy.dump(circuit,f)
    pairs=read(directory/'pairs.json');n=len(pairs);pilot=p['hardware']['pilot_pairs']
    # Uniformly spaced deterministic pilot; no labels or scores involved. Pilot pairs are reused.
    pilot_indices=sorted({i*(n-1)//(pilot-1) for i in range(pilot)})
    pilot_ids=[pairs[i]['id'] for i in pilot_indices];rest=[v['id'] for v in pairs if v['id'] not in set(pilot_ids)]
    chunks=[pilot_ids]+[rest[i:i+p['hardware']['chunk_pairs']] for i in range(0,len(rest),p['hardware']['chunk_pairs'])]
    base=[p['budget']['subjob_overhead_seconds']+len(chunk)*p['measurement']['shots']*per_shot for chunk in chunks]
    allowances=[job_allowance(v,p) for v in base]
    result={'created':now(),'backend':backend.name,'instance_hash':instance_hash,'open_plan_verified':True,
        'inputs':identity(read(directory/'input_lock.json')),'preprocessing':file_hash(directory/'preprocessed.npz'),
        'logical_sha256':file_hash(directory/'logical_template.qpy'),'isa_sha256':file_hash(directory/'isa_template.qpy'),
        'layout':layout,'layout_selection':'Independent qubits sorted by readout_error + SX_error; no connectivity requirement',
        'qubit_calibration':[{'qubit':r[3],'readout_error':r[1],'sx_error':r[2]} for r in rank[:k]],
        'parameter_order':[str(x) for x in circuit.parameters],
        'measurement_map':[{'physical_qubit':circuit.find_bit(item.qubits[0]).index,'classical_bit':circuit.find_bit(item.clbits[0]).index} for item in circuit.data if item.operation.name=='measure'],
        'circuit_duration_seconds':duration,'default_rep_delay_seconds':repetition,'reset_seconds_conservative':max(resets),
        'per_shot_seconds':per_shot,'chunks':chunks,'base_estimates_seconds':base,'job_allowances_seconds':allowances,
        'planned_reserved_seconds':sum(allowances),'remaining_account_seconds_at_plan':available,
        'locally_estimated_feasible':sum(allowances)<=min(p['budget']['planned_cap_seconds'],available-p['budget']['safety_reserve_seconds']),
        'estimate_limitations':'Local conservative estimate, not a service guarantee. Pilot actual usage must be known before main jobs; every job has a separate execution cap.',
        'hardware_jobs_submitted':0}
    result['identity']=identity(result);atomic(directory/'backend_plan.json',result)
    return result
