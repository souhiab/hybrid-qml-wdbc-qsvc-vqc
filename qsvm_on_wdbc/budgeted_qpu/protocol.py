"""Frozen primary configuration; optional workloads are explicit, never auto-selected."""
import copy
from .io import identity
DEFAULT={
 'schema':'wdbc-budgeted-kernel-v1','configuration':'Q9-budgeted-feasibility',
 'cohort':{'size':72,'train':48,'test':24,'selection_seed':20261110,'split_seed':20261110},
 'preprocessing':{'scaler':'MinMaxScaler','feature_range':[0,1],'clip':False,'rfe_step':1,'k':4,
    'estimator':{'penalty':'l1','solver':'saga','C':1.0,'max_iter':5000,'random_state':42,'tol':.0001,
                 'fit_intercept':True,'class_weight':None}},
 'encoder':{'constructor':'ZFeatureMap','qubits':4,'reps':1,'alpha':2.0,'data_map':'Qiskit default self_product','extra_input_rescaling':None,'entanglement_argument':None},
 'svc':{'C':1.0,'kernel':'precomputed','probability':True,'random_state':20261110,'tol':.001,'shrinking':True,'class_weight':None,
        'cache_size':200,'max_iter':-1,'decision_function_shape':'ovr','break_ties':False},
 'classical_calibration_note':'Archived Q9 random_state=None was uncontrolled; this new feasibility experiment freezes calibration seed20261110. No threshold optimization.',
 'kernel':{'training_diagonal':1.0,'symmetry_reuse':True,'center':False,'diagonal_normalization':False,'psd_repair':False},
 'measurement':{'shots':256,'adaptive':False,'m3':False,'dynamical_decoupling':False,'twirling':False,'rep_delay':'backend default','init_qubits':'runtime default','register':'meas'},
 'hardware':{'backend':'ibm_fez','plan_required':'open','chunk_pairs':256,'pilot_pairs':32,'transpiler_seed':20261110,'optimization_level':1,'approximation_degree':1.0},
 'budget':{'absolute_cap_seconds':600,'safety_reserve_seconds':120,'planned_cap_seconds':480,'minimum_job_allowance_seconds':5,
           'estimate_safety_factor':1.5,'extra_job_margin_seconds':2,'subjob_overhead_seconds':2,'require_known_usage':True},
 'simulation':{'seed':20261110,'statevector_dtype':'complex128','kernel_dtype':'float64'},
 'metrics':{'positive_source_label':0,'prediction':'native SVC.predict; binary zero-margin tie is benign1','auc_primary':'probability column classes_==0','threshold_tuning':False},
 'hardware_submission_enabled_by_default':False,
 'interpretation':'Matched hardware feasibility on a new small fixed cohort; not an identical rerun or independent validation of the historical model search.'}
def protocol(cohort_size=72,qubits=4,backend='ibm_fez'):
    p=copy.deepcopy(DEFAULT)
    if cohort_size not in [72,96]:raise ValueError('Only frozen72/48/24 or explicit96/64/32 cohorts are supported')
    if qubits not in [4,7]:raise ValueError('Only primary4 and explicitly optional7 widths are supported')
    p['cohort'].update(size=cohort_size,train=cohort_size*2//3,test=cohort_size//3)
    p['preprocessing']['k']=qubits;p['encoder']['qubits']=qubits;p['hardware']['backend']=backend
    if qubits==7:p['configuration']='explicit-optional-seven-qubit-Q9-style; not historical-seven-qubit-workflow'
    return p
def workload(p):
    n=p['cohort']['train'];m=p['cohort']['test'];pairs=n*(n-1)//2+n*m
    return {'train_off_diagonal_pairs':n*(n-1)//2,'test_train_pairs':n*m,'main_circuits':pairs,'base_shot_executions':pairs*p['measurement']['shots'],'diagonal_circuits':0,'additional_diagnostic_circuits':0}
