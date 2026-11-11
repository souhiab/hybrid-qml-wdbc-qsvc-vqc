from vqc_engine import ROOT,atomic_json,sha
import json
atomic_json(ROOT/'records/workbook_writer_update.json',{
    'component':'coordinator workbook writer only',
    'old_sha256':sha(ROOT/'versions/v2_workbook_writer.mjs'),
    'new_sha256':sha(ROOT/'workbook_support/workbook_writer.mjs'),
    'reason':'Update changed rows and append training-log rows rather than clear and reformat the workbook after each fit. Preserve atomic export and recovery backup.',
    'training_code_changed':False,'scientific_settings_changed':False,
    'receipts_authoritative':True,'reload_rule':'Restart workbook writer only. Any in-flight synchronization error is logged; the next synchronization rebuilds from the full authoritative snapshot.',
    'status_at_update':json.loads((ROOT/'status.json').read_text()),
})
