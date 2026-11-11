"""Conservative cumulative usage accounting. Unknown usage never releases a reservation."""
import math
from pathlib import Path
from .io import atomic,read,identity,now,file_hash

class BudgetError(RuntimeError): pass

def number(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
        raise BudgetError('Unknown or invalid '+name)
    return float(value)

class Ledger:
    def __init__(self,path,cache_identity,budget):
        self.path=Path(path);self.budget=budget
        if self.path.exists():
            self.data=read(self.path)
            if self.data.get('identity')!=cache_identity:raise BudgetError('Ledger identity mismatch')
        else:
            self.data={'schema':1,'identity':cache_identity,'jobs':{},'created':now()};self.save()
    def save(self):atomic(self.path,self.data)
    def charged(self):
        return sum(number(j['actual_seconds'],'job usage') if j.get('actual_seconds') is not None else number(j['allowance_seconds'],'reservation') for j in self.data['jobs'].values())
    def prior_charged(self):
        total=0.
        for prior in self.budget.get('prior_experiments',[]):
            path=Path(prior['ledger_path'])
            if file_hash(path)!=prior['ledger_sha256']:raise BudgetError('Prior experiment ledger changed; cumulative usage requires review')
            jobs=read(path)['jobs']
            if any(j.get('actual_seconds') is None or j.get('state')!='VALID' for j in jobs.values()):raise BudgetError('Prior experiment has unsettled usage or failures')
            total+=sum(number(j['actual_seconds'],'prior settled usage') for j in jobs.values())
        return total
    def cumulative_charged(self):return self.prior_charged()+self.charged()
    def reserve(self,key,pairs,allowance,remaining):
        remaining=number(remaining,'remaining account allowance');allowance=number(allowance,'next job allowance')
        if key in self.data['jobs']:raise BudgetError('Duplicate reservation; reconcile existing job')
        if any(j.get('actual_seconds') is None for j in self.data['jobs'].values()):
            raise BudgetError('Unresolved prior submission or unknown actual usage; reconcile before another job')
        cap=min(self.budget['planned_cap_seconds'],self.budget['absolute_cap_seconds']-self.budget['safety_reserve_seconds'])
        if self.cumulative_charged()+allowance>cap or remaining-allowance<self.budget['safety_reserve_seconds']:
            raise BudgetError('Next job would consume cumulative cap or safety reserve')
        self.data['jobs'][key]={'pairs':pairs,'allowance_seconds':allowance,'actual_seconds':None,'state':'RESERVED','job_id':None,'tag':'wdbc-'+identity({'identity':self.data['identity'],'key':key})[:40],'reserved':now()};self.save()
        return self.data['jobs'][key]
    def submitted(self,key,job_id):
        self.data['jobs'][key].update(job_id=str(job_id),state='SUBMITTED',submitted=now());self.save()
    def finish(self,key,state,actual):
        # Failed/cancelled work is charged too. None remains an outstanding reservation.
        if actual is not None:actual=number(actual,'actual service usage')
        self.data['jobs'][key].update(state=state,actual_seconds=actual,finished=now());self.save()
