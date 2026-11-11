"""Local-only safety/numerical regression tests; contains no IBM connection or submission."""
import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from qiskit.primitives.containers import BitArray,DataBin,SamplerPubResult,PrimitiveResult
from sklearn.preprocessing import KernelCenterer
from qsvm_on_wdbc.budgeted_qpu.protocol import protocol,workload
from qsvm_on_wdbc.budgeted_qpu.data import fit_preprocessing,pair_manifest
from qsvm_on_wdbc.budgeted_qpu.circuits import template,bindings,overlap_state,logical_state
from qsvm_on_wdbc.budgeted_qpu.results import counts_record,extract_pub_results,matrices,clinical,center_kernels,ResultValidationError
from qsvm_on_wdbc.budgeted_qpu.ledger import Ledger,BudgetError
from qsvm_on_wdbc.budgeted_qpu.io import runner_lock,atomic,read
from qsvm_on_wdbc.budgeted_qpu.hardware import run,collect

class SafetyTests(unittest.TestCase):
    def test_workload(self):
        self.assertEqual(workload(protocol())['main_circuits'],2280)
        self.assertEqual(workload(protocol())['base_shot_executions'],583680)
        self.assertEqual(len(pair_manifest(protocol())),2280)
    def test_train_only_preprocessing(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(40,6));y=np.tile([0,1],20)
        a,b,audit,scaler,selector=fit_preprocessing(x,y,np.ones((5,6))*100,protocol(),list('abcdef'))
        self.assertEqual(a.shape,(40,4));self.assertTrue((b>1).all())
        np.testing.assert_array_equal(scaler.data_max_,x.max(axis=0))
        self.assertFalse(audit['evaluation_rows_in_fit']);self.assertFalse(audit['evaluation_labels_in_fit'])
    def test_compute_uncompute_order(self):
        p=protocol();a=np.array([.1,.3,.6,.9]);b=np.array([.8,.2,.4,.7]);pair={'id':'x','kind':'train','i':0,'j':1}
        circuit=template(p);values=bindings(circuit,[pair],np.array([a,b]),np.empty((0,4)))[0]
        measured=abs(overlap_state(circuit,values)[0])**2
        expected=abs(np.vdot(logical_state(p,b),logical_state(p,a)))**2
        self.assertAlmostEqual(measured,expected,places=12)
        self.assertAlmostEqual(measured,float(np.prod(np.cos(a-b)**2)),places=12)
    def test_malignant(self):
        m=clinical([0,0,1,1],[0,1,0,1],[.9,.7,.4,.1])
        self.assertEqual((m['TP'],m['FN'],m['FP'],m['TN']),(1,1,1,1));self.assertEqual(m['roc_auc'],1)
    def test_centering(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(6,4));z=rng.normal(size=(3,4));k=x@x.T;t=z@x.T
        a,b=center_kernels(k,t);c=KernelCenterer().fit(k)
        np.testing.assert_allclose(a,c.transform(k),atol=1e-12);np.testing.assert_allclose(b,c.transform(t),atol=1e-12)
    def test_counts(self):
        for invalid in [{},{'0000':0},{'0000':3},{'garbage':256}]:
            with self.assertRaises(ResultValidationError):counts_record(invalid,256,4)
        self.assertEqual(counts_record({'1111':256},256,4)['zero_probability'],0.)
        pairs=[{'id':'one'}];bits=BitArray.from_counts([{'0000':128,'1111':128}])
        result=PrimitiveResult([SamplerPubResult(DataBin(meas=bits))])
        self.assertEqual(extract_pub_results(result,pairs,256,4)[0]['zero_probability'],.5)
        with self.assertRaises(ResultValidationError):extract_pub_results(PrimitiveResult([SamplerPubResult(DataBin())]),pairs,256,4)
    def test_pair_coverage(self):
        with self.assertRaises(ResultValidationError):matrices({},[{'id':'a'}],2,1)
    def test_budget_and_unknown_usage(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=Ledger(Path(d)/'ledger.json','identity',protocol()['budget'])
            for unknown in [None,float('nan'),-1]:
                with self.assertRaises(BudgetError):ledger.reserve('a',['pair'],10,unknown)
            ledger.reserve('a',['pair'],100,600)
            with self.assertRaises(BudgetError):ledger.reserve('b',['pair2'],10,600)
            ledger.submitted('a','job-id');ledger.finish('a','CANCELLED',None)
            with self.assertRaises(BudgetError):ledger.reserve('b',['pair2'],10,600)
            ledger.finish('a','CANCELLED',90)
            self.assertEqual(ledger.charged(),90)
            with self.assertRaises(BudgetError):ledger.reserve('b',['pair2'],400,600)
            with self.assertRaises(BudgetError):ledger.reserve('a',['pair'],10,600)
            with self.assertRaises(BudgetError):Ledger(Path(d)/'ledger.json','wrong',protocol()['budget'])
    def test_lock_and_opt_in(self):
        with tempfile.TemporaryDirectory() as d:
            with runner_lock(d):
                with self.assertRaises(RuntimeError):
                    with runner_lock(d):pass
            with self.assertRaises(BudgetError):run(d,'execute')
    def test_recovery_id_before_wait(self):
        # Fake provider only: verifies recovery persists ID before accessing results.
        with tempfile.TemporaryDirectory() as d:
            ledger=Ledger(Path(d)/'ledger.json','cache',protocol()['budget']);entry=ledger.reserve('job-000',['p'],10,600)
            pair={'id':'p','kind':'train','i':0,'j':1}
            class FakeJob:
                def job_id(self):return 'synthetic-job'
                def in_final_state(self):return True
                def result(self):
                    assert read(Path(d)/'ledger.json')['jobs']['job-000']['job_id']=='synthetic-job'
                    return PrimitiveResult([SamplerPubResult(DataBin(meas=BitArray.from_counts([{'0000':256}],num_bits=4)) )])
                def usage(self):return 2.
                def status(self):return 'DONE'
            class FakeService:
                def jobs(self,**kw):return [FakeJob()]
                def job(self,jobid):return FakeJob()
            collect(d,ledger,'job-000',FakeService(),protocol(),{'p':pair})
            self.assertEqual(ledger.data['jobs']['job-000']['state'],'VALID')
            self.assertEqual(ledger.charged(),2.)
    def test_zero_usage_retains_reservation(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=Ledger(Path(d)/'ledger.json','cache',protocol()['budget']);ledger.reserve('job-000',['p'],9,600)
            ledger.submitted('job-000','synthetic-zero-usage')
            class FakeJob:
                def in_final_state(self):return True
                def result(self):return PrimitiveResult([SamplerPubResult(DataBin(meas=BitArray.from_counts([{'0000':256}],num_bits=4)))])
                def usage(self):return 0
                def status(self):return 'DONE'
            class FakeService:
                def job(self,jobid):return FakeJob()
            with self.assertRaises(BudgetError):collect(d,ledger,'job-000',FakeService(),protocol(),{'p':{'id':'p','kind':'train','i':0,'j':1}},usage_wait_seconds=0)
            self.assertTrue((Path(d)/'hardware/jobs/job-000.counts.json').exists())
            self.assertIsNone(ledger.data['jobs']['job-000']['actual_seconds'])
            self.assertEqual(ledger.charged(),9)
    def test_cumulative_prior_guard(self):
        from qsvm_on_wdbc.budgeted_qpu.io import file_hash
        with tempfile.TemporaryDirectory() as d:
            prior=Path(d)/'prior.json';atomic(prior,{'jobs':{'j':{'state':'VALID','actual_seconds':180}}})
            budget=protocol()['budget'];budget['prior_experiments']=[{'ledger_path':str(prior),'ledger_sha256':file_hash(prior)}]
            ledger=Ledger(Path(d)/'new.json','new',budget)
            self.assertEqual(ledger.cumulative_charged(),180)
            with self.assertRaises(BudgetError):ledger.reserve('too-large',['p'],301,600)
            ledger.reserve('allowed',['p'],274,420)
            atomic(prior,{'jobs':{'j':{'state':'VALID','actual_seconds':181}}})
            with self.assertRaises(BudgetError):ledger.prior_charged()
    def test_patient_exclusion(self):
        from qsvm_on_wdbc.budgeted_qpu.data import prepare
        with tempfile.TemporaryDirectory() as d:
            p=protocol();p['cohort']['excluded_source_row_ids']=list(range(72))
            prepare(d,p);s=read(Path(d)/'case_manifest.json')
            self.assertEqual(len(s['selected_ids']),72)
            self.assertFalse(set(s['selected_ids'])&set(range(72)))

if __name__=='__main__':unittest.main()
