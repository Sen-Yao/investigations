#!/usr/bin/env python3
import json, subprocess, time, os
from pathlib import Path
SCRIPT = Path.home() / 'VoxG/nexus/investigations/2026-04-27-global-ga-score/experiments/scripts/run_clean_dual_sequence_gt.py'
OUTDIR = Path.home() / 'VoxG/nexus/investigations/2026-04-27-global-ga-score/experiments/outputs/clean_gt_mini'
LOGDIR = OUTDIR / 'logs'
OUTDIR.mkdir(parents=True, exist_ok=True); LOGDIR.mkdir(parents=True, exist_ok=True)
GPU=os.environ.get('CLEAN_GT_GPU','3')
RUNS=[('elliptic','baseline_qc'),('elliptic','normal_max_dev'),('elliptic','normal_soft_or')]
summary=[]; start=time.time()
for idx,(dataset,ga) in enumerate(RUNS,4):
    name=f'{idx:02d}_{dataset}_{ga}_add_seed0_retry'
    out=OUTDIR/f'{name}.json'; log=LOGDIR/f'{name}.log'
    cmd=['python3',str(SCRIPT),'--dataset',dataset,'--ga_mode',ga,'--out',str(out),'--device',GPU,'--seed','0','--train_rate','0.05','--num_epoch','200','--gl_combine','add','--encode_batch_size','2048']
    print('RUN_START',name,'cmd='+' '.join(cmd),flush=True)
    t0=time.time()
    with open(log,'w') as lf: p=subprocess.run(cmd,cwd=str(Path.home()/'VoxG'),stdout=lf,stderr=subprocess.STDOUT,text=True)
    item={'name':name,'dataset':dataset,'ga_mode':ga,'returncode':p.returncode,'time_sec':time.time()-t0,'out':str(out),'log':str(log)}
    if out.exists():
        try:
            d=json.loads(out.read_text()); item.update({'best':d.get('best'),'purity':d.get('purity')})
        except Exception as e: item['parse_error']=str(e)
    else: item['error']='missing output json'
    summary.append(item); (OUTDIR/'summary_elliptic_retry_live.json').write_text(json.dumps({'runs':summary,'elapsed_sec':time.time()-start},indent=2)); print('RUN_DONE',json.dumps(item,ensure_ascii=False),flush=True)
(OUTDIR/'summary_elliptic_retry.json').write_text(json.dumps({'runs':summary,'elapsed_sec':time.time()-start},indent=2)); print('ALL_DONE',OUTDIR/'summary_elliptic_retry.json')
