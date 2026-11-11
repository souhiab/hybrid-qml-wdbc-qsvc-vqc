"""Read-only verification. Never fits models or contacts IBM."""
import argparse, csv, hashlib, json, zipfile
from pathlib import Path
import numpy as np

def verify(root):
    index=json.loads((root/'results/STUDIES.json').read_text())
    count=0
    for study in index['studies']:
        folder=root/study['data']
        if study['study']=='hardware_q9':
            with (folder/'comparison.csv').open(newline='') as f: hardware=list(csv.DictReader(f))
            assert len(hardware)==6
            cohorts=[]
            for row in hardware:
                with np.load(folder/'evidence'/row['cohort']/row['method']/'kernels_predictions.npz') as a:
                    y=a['source_labels'];p=a['predictions'];s=a['malignant_probabilities']
                    assert len(y)==24
                    assert int((y==p).sum())==int(row['correct_count'])
                    pos=s[y==0];neg=s[y==1]
                    auc=((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg))
                    assert abs(auc-float(row['roc_auc']))<1e-12
        if (folder/'evidence.zip').exists():
            hashes=json.loads((folder/'evidence_hashes.json').read_text())
            with zipfile.ZipFile(folder/'evidence.zip') as z:
                assert z.testzip() is None
                for name, digest in hashes.items():
                    assert hashlib.sha256(z.read(name)).hexdigest()==digest, name
        if not (folder/'predictions.csv').exists(): continue
        panel={}
        with (folder/'predictions.csv').open(newline='') as f:
            for row in csv.DictReader(f): panel.setdefault(row['evaluation_id'],[]).append(row)
        with (folder/'metrics.csv').open(newline='') as f:
            metrics=list(csv.DictReader(f))
        assert len(metrics)==study['expected']==len(panel),study['study']
        for m in metrics:
            rows=panel[m['evaluation_id']]
            y=np.array([int(r['source_label']) for r in rows]); p=np.array([int(r['prediction']) for r in rows])
            s=np.array([float(r['malignant_score']) for r in rows])
            assert len({r['source_row_id'] for r in rows})==len(rows)
            assert np.isfinite(s).all()
            assert int(m['correct_count'])==int((y==p).sum())
            assert abs(float(m['accuracy'])-float((y==p).mean()))<1e-12
            positive=s[y==0]; negative=s[y==1]
            auc=((positive[:,None]>negative).sum()+.5*(positive[:,None]==negative).sum())/(len(positive)*len(negative))
            assert abs(float(m['roc_auc'])-auc)<1e-12
            for key,value in {'TP':((y==0)&(p==0)).sum(),'FN':((y==0)&(p==1)).sum(),'FP':((y==1)&(p==0)).sum(),'TN':((y==1)&(p==1)).sum()}.items():
                assert int(m[key])==value
        count+=len(metrics)
    manifest=root/'PUBLICATION_HASHES.json'
    if manifest.exists():
        for name,digest in json.loads(manifest.read_text()).items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    print(json.dumps({'status':'VERIFIED','studies':len(index['studies']),'simulator_outer_evaluations':count,'training_started':False}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    verify(parser.parse_args().root)
