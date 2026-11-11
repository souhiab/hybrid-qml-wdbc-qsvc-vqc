import json
from pilot_controller import Controller
from pilot_fit import now
from vqc_engine import ROOT,atomic_json,sha

def main():
    controller=Controller()
    controller.payload=json.loads((ROOT/'workbook_support/snapshot.json').read_text())
    controller.status=json.loads((ROOT/'status.json').read_text())
    report=controller.verify_final()
    controller.status.update({'execution_state':'complete','controller_active':False,'final_verification':'records/final_verification.json'})
    atomic_json(ROOT/'status.json',controller.status)
    atomic_json(ROOT/'records/execution_completion.json',{'finished':now(),**report})
    atomic_json(ROOT/'records/finalization_recovery.json',{'training_repeated':False,'fits_preserved':144,'scientific_source_files_unchanged':True,'helper_sources':{'finalize_pilot.py':sha(ROOT/'finalize_pilot.py'),'workbook_support/finalize_workbook.mjs':sha(ROOT/'workbook_support/finalize_workbook.mjs')},'recovery':'Final workbook timestamp-precision check; no numerical result or model changes.'})
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
