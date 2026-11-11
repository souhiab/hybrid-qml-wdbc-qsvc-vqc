"""Read-only evidence dashboard. No IBM imports, connection, training or submission."""
import argparse,datetime,html,json,os,time,uuid
from pathlib import Path
from .io import read,atomic,now,runner_lock

def esc(x):return html.escape(str(x))
def stamp(path):
    if not path.exists():return 'Not saved yet'
    return datetime.datetime.fromtimestamp(path.stat().st_mtime,datetime.timezone.utc).astimezone().isoformat(timespec='seconds')
def load(path,default=None):return read(path) if path.exists() else ({} if default is None else default)
def link(path,label):return f'<a href="{esc(path.resolve().as_uri())}" target="_blank">{esc(label)}</a>'

def snapshot(d):
    p=load(d/'protocol.json');plan=load(d/'backend_plan.json');ledger=load(d/'hardware/ledger.json')
    status=load(d/'status.json');jobs=ledger.get('jobs',{});total=sum(len(v) for v in plan.get('chunks',[])) or 2280
    valid=[];warnings=[];rows=[]
    for key,j in sorted(jobs.items()):
        counts=d/'hardware/jobs'/(key+'.counts.json');saved=counts.exists()
        if j.get('state')=='VALID' and saved:valid.extend(j.get('pairs',[]))
        elif j.get('state')=='VALID':warnings.append(key+': valid ledger entry has no saved counts file')
        if j.get('actual_seconds') is None and j.get('state') not in ['SUBMITTED']:
            warnings.append(key+': actual usage unresolved; reservation remains charged')
        rows.append({'key':key,'job_id':j.get('job_id') or 'Not recorded / reconcile required','state':j.get('state','UNKNOWN'),
            'pairs':len(j.get('pairs',[])),'actual':j.get('actual_seconds'),'reserved':j.get('allowance_seconds'),
            'saved':saved,'saved_at':stamp(counts),'record':d/'hardware/jobs'/(key+'.json')})
    if len(valid)!=len(set(valid)):warnings.append('Duplicate pair IDs detected in ledger')
    actual=sum(j['actual_seconds'] for j in jobs.values() if isinstance(j.get('actual_seconds'),(int,float)))
    reserved=sum(j.get('allowance_seconds',0) for j in jobs.values() if j.get('actual_seconds') is None)
    verified=len(set(valid));hw=load(d/'hardware/status.json')
    if not jobs:phase='READY — QPU NOT STARTED'
    elif hw.get('state')=='COMPLETE_VERIFIED':phase='COMPLETE — HARDWARE RESULTS VERIFIED'
    elif (d/'hardware/COMPLETED.json').exists():phase='QPU RESULTS SAVED — SEE FINAL VERIFICATION'
    elif warnings:phase='CHECK REQUIRED — REVIEW JOB RECORDS'
    else:phase=hw.get('state','QPU WORK IN PROGRESS')
    prior=sum(float(j['actual_seconds']) for r in p.get('budget',{}).get('prior_experiments',[]) for j in load(Path(r['ledger_path'])).get('jobs',{}).values())
    return {'p':p,'plan':plan,'status':status,'jobs':rows,'actual':actual,'prior_actual':prior,'reserved':reserved,'verified':verified,'total':total,
        'warnings':warnings,'phase':phase,'last_save':max([d/'status.json',d/'hardware/ledger.json',d/'hardware/status.json']+[d/'hardware/jobs'/(r['key']+'.counts.json') for r in rows],key=lambda f:f.stat().st_mtime if f.exists() else 0)}

def render(d):
    s=snapshot(d);plan=s['plan'];budget=s['p'].get('budget',{});percent=100*s['verified']/s['total'];ntrain=s['p'].get('cohort',{}).get('train',48);ntest=s['p'].get('cohort',{}).get('test',24)
    metricrows=''
    for name,label in [('local_exact','Ideal exact simulation'),('local_shots','Ideal 256-shot simulation'),('hardware','IBM QPU')]:
        m=load(d/name/'metrics.json');done=(d/name/'COMPLETED.json').exists()
        values=[label,'Saved + verified receipt' if done else 'Not complete',f"{m.get('correct','—')}/{m.get('n',ntest)}" if m else '—',f"{m['accuracy']*100:.2f}%" if m else '—',f"{m['roc_auc']:.6f}" if m else '—',stamp(d/name/'metrics.json')]
        metricrows+='<tr>'+''.join('<td>'+esc(v)+'</td>' for v in values)+'</tr>'
    jobrows=''
    for j in s['jobs']:
        values=[j['key'],j['state'],j['job_id'],j['pairs'],f"{j['actual']:.3f}s" if j['actual'] is not None else 'Unknown',str(j['reserved'])+'s','Saved '+j['saved_at'] if j['saved'] else 'Not saved yet']
        jobrows+='<tr>'+''.join('<td>'+esc(v)+'</td>' for v in values)+'</tr>'
    if not jobrows:jobrows='<tr><td colspan="7">No QPU jobs submitted. The dashboard does not start jobs.</td></tr>'
    files=[('protocol.json','Frozen protocol'),('case_manifest.json','Ordered split and source rows'),('preprocessing_audit.json','Training-only preprocessing audit'),('backend_plan.json','Backend and time estimate'),('hardware/ledger.json','Cumulative budget ledger'),('hardware/status.json','Hardware status'),('comparison.json','Local / hardware comparison'),('final_verification.json','Preparation verification'),('REPORT.md','Preparation report'),('hardware/EXECUTION_REPORT_20261111.md','Hardware execution report'),('hardware/final_verification_20261111.json','Hardware final verification'),('RUN_ORDER.md','Execution / resume commands'),('local_exact/predictions.csv','Exact predictions'),('local_shots/predictions.csv','256-shot predictions'),('hardware/predictions.csv','QPU predictions'),('hardware/COMPLETED.json','QPU completion receipt')]
    filerows=''.join('<tr><td>'+link(d/path,label)+'</td><td>'+('Saved' if (d/path).exists() else 'Waiting')+'</td><td>'+esc(stamp(d/path))+'</td></tr>' for path,label in files)
    warning=''.join('<p class="warn">'+esc(w)+'</p>' for w in s['warnings'])
    stats=[('Saved QPU pairs',f"{s['verified']:,} / {s['total']:,}"),('Recorded jobs',str(len(s['jobs']))),('Actual reported QPU usage',f"{s['actual']:.3f}s"),('Unresolved reservations',f"{s['reserved']:.1f}s")]
    cards=''.join('<article><small>'+esc(k)+'</small><strong>'+esc(v)+'</strong></article>' for k,v in stats)
    forecast=plan.get('planned_reserved_seconds',0);cap=budget.get('planned_cap_seconds',480);reserve=budget.get('safety_reserve_seconds',120)
    charged=s['actual']+s['reserved']+s['prior_actual'];width=min(100,100*charged/cap) if cap else 0
    final_remaining=load(d/'hardware/status.json').get('remaining_account_seconds')
    final_budget_note=f'<p>Account remaining at final verification: <b>{final_remaining}s</b>. This is a saved final reading, not a continuous account query.</p>' if final_remaining is not None else ''
    if s['prior_actual']:final_budget_note+=f'<p>Cumulative budget includes <b>{s["prior_actual"]}s</b> already used by the first cohort; the usage card above shows this cohort only.</p>'
    document=f'''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="10"><title>WDBC Q9 · Live progress</title><style>
    :root{{color-scheme:dark;font-family:Segoe UI,Arial,sans-serif;background:#0c1420;color:#e6edf7}}body{{margin:0}}main{{max-width:1240px;margin:auto;padding:32px 24px}}h1{{margin:6px 0 12px;font-size:30px}}h2{{font-size:21px;margin-top:30px}}p{{color:#b4c4d8;line-height:1.55}}.badge{{display:inline-block;background:#123d38;color:#84e3cf;border:1px solid #28766a;border-radius:20px;padding:8px 14px;font-weight:600}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}}article,.panel{{background:#152235;border:1px solid #2b405b;border-radius:14px;padding:20px}}small{{display:block;color:#b2c5df}}strong{{display:block;font-size:27px;margin-top:10px}}.track{{height:16px;background:#24334a;border-radius:10px;overflow:hidden;margin:12px 0}}.fill{{height:100%;background:#58d6b8;min-width:0}}.budget{{background:#7cafff}}.tablewrap{{overflow-x:auto;border:1px solid #2b405b;border-radius:12px}}table{{width:100%;border-collapse:collapse;background:#111e30}}th,td{{text-align:left;padding:13px;border-bottom:1px solid #26374d;font-size:14px;vertical-align:top}}th{{background:#1c2d44;color:#bed1eb}}a{{color:#8bc4ff}}.warn{{padding:12px;background:#493619;color:#ffe1a2;border-radius:8px}}.muted{{font-size:13px;color:#9aadc5}}code{{word-break:break-all}}button{{background:#23496f;color:white;border:1px solid #49759c;border-radius:8px;padding:9px 14px;cursor:pointer}}@media(max-width:800px){{.cards{{grid-template-columns:repeat(2,1fr)}}}}@media(max-width:450px){{.cards{{grid-template-columns:1fr}}}}
    </style></head><body><main><small>WDBC · FOUR-QUBIT Q9 · COMPUTE–UNCOMPUTE</small><h1>Execution and saved evidence</h1><span class="badge">{esc(s['phase'])}</span><p>This page refreshes every 10 seconds. It reads local saved evidence only; it cannot train models or submit QPU jobs.</p><button onclick="location.reload()">Refresh now</button>
    <p class="muted">Dashboard generated: <span id="generated">{esc(now())}</span> · Latest tracked execution save: {esc(stamp(s['last_save']))}<br>If the generated timestamp stops advancing, the dashboard updater is offline. Refreshing the browser alone does not restart it.</p>{warning}<section class="cards">{cards}</section>
    <section class="panel"><h2 style="margin-top:0">Hardware pair coverage · {percent:.1f}%</h2><div class="track"><div class="fill" style="width:{percent:.3f}%"></div></div><p>{s['verified']:,} pairs have VALID ledger entries and saved count files. Final model metrics require all {s['total']:,} pairs and a completion receipt. This display is not a replacement for the runner's verification.</p></section>
    <h2>Quantum-time budget</h2><section class="panel"><div class="track"><div class="fill budget" style="width:{width:.3f}%"></div></div><p>Actual usage + outstanding reservations: <b>{charged:.3f}s / {cap}s planned cap</b>. Protected reserve: <b>{reserve}s</b>. Absolute experiment ceiling: <b>{budget.get('absolute_cap_seconds',600)}s</b>.</p>{final_budget_note}<p>Pre-execution conservative allowance: {forecast}s. Account remaining at the saved planning snapshot: {plan.get('remaining_account_seconds_at_plan','unknown')}s — this is not a live account reading. The controller must refresh it before submissions. Queue waiting and local computation are separate from QPU usage.</p></section>
    <h2>Saved results</h2><div class="tablewrap"><table><thead><tr><th>Method</th><th>Completion</th><th>Correct</th><th>Accuracy</th><th>Malignant AUC</th><th>Metrics saved</th></tr></thead><tbody>{metricrows}</tbody></table></div><p class="muted">Frozen cohort: {ntrain} training / {ntest} test cases. Local scores are simulator results, not hardware results. Small-cohort feasibility with historical selection; not independent validation.</p>
    <h2>QPU job history</h2><div class="tablewrap"><table><thead><tr><th>Chunk</th><th>State</th><th>IBM job ID</th><th>Pairs</th><th>Actual usage</th><th>Allowance</th><th>Counts saved</th></tr></thead><tbody>{jobrows}</tbody></table></div>
    <h2>Evidence on this computer</h2><p><code>{esc(d)}</code></p><div class="tablewrap"><table><thead><tr><th>File</th><th>Availability</th><th>Last saved (local computer time, offset included)</th></tr></thead><tbody>{filerows}</tbody></table></div><p class="muted">File links may download or open in a text viewer, depending on your browser. Atomic page replacement prevents partially written dashboard pages.</p></main></body></html>'''
    out=d/'Q9_LIVE_PROGRESS.html';temp=out.with_name(out.name+'.'+uuid.uuid4().hex+'.tmp');temp.write_text(document,encoding='utf-8');os.replace(temp,out)
    atomic(d/'dashboard/status.json',{'state':'WATCHING','generated':now(),'dashboard':str(out),'hardware_jobs':len(s['jobs']),'saved_pairs':s['verified'],'qpu_submission_capability':False})
    return out

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--once',action='store_true');parser.add_argument('--interval',type=float,default=5)
    args=parser.parse_args();d=args.run_dir.resolve()
    if not (d/'protocol.json').exists():parser.error('Missing frozen protocol')
    if args.once:print(render(d));return
    watch=d/'dashboard';watch.mkdir(exist_ok=True)
    with runner_lock(watch):
        atomic(watch/'process.json',{'pid':os.getpid(),'started':now(),'command':'read-only dashboard updater','run_directory':str(d)})
        while True:
            try:render(d)
            except Exception as error:
                atomic(watch/'status.json',{'state':'DASHBOARD_READ_ERROR','time':now(),'error_type':type(error).__name__,'detail':str(error)})
            time.sleep(max(2,args.interval))

if __name__=='__main__':main()
