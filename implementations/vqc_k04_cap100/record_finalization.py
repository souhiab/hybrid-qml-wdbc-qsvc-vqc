import json
from vqc_engine import ROOT,atomic_json,sha
path=ROOT/'records/workbook_writer_update.json'
record=json.loads(path.read_text())
record.setdefault('runtime_new_sha256',record['new_sha256'])
record['new_sha256']=sha(ROOT/'workbook_support/workbook_writer.mjs')
record['final_readback_change']='Compare Excel blanks against empty strings and normalize ISO timestamps to the same UTC instant; all other values compared exactly.'
record['final_presentation_change']='Order identifiers before settings and metrics, wrap identifiers/paths, format timestamp and summary percentages. No numerical artifact or training change.'
atomic_json(path,record)
