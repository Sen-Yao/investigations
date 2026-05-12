#!/usr/bin/env python3
import json, subprocess, time
from pathlib import Path
SCRIPT = Path.home() / "VoxG/nexus/investigations/2026-04-27-global-ga-score/experiments/scripts/run_clean_dual_sequence_gt.py"
OUTDIR = Path.home() / "VoxG/nexus/investigations/2026-04-27-global-ga-score/experiments/outputs/clean_gt_mini"
LOGDIR = OUTDIR / "logs"
OUTDIR.mkdir(parents=True, exist_ok=True); LOGDIR.mkdir(parents=True, exist_ok=True)
name = "05_elliptic_normal_max_dev_add_seed0_bs512"
out = OUTDIR / f"{name}.json"
log = LOGDIR / f"{name}.log"
gpu = "0"
cmd = ["python3", str(SCRIPT), "--dataset", "elliptic", "--ga_mode", "normal_max_dev", "--out", str(out), "--device", gpu, "--seed", "0", "--train_rate", "0.05", "--num_epoch", "200", "--gl_combine", "add", "--encode_batch_size", "512"]
print("RUN_START", name, "cmd=" + " ".join(cmd), flush=True)
t0=time.time()
with open(log, "w") as lf:
    p = subprocess.run(cmd, cwd=str(Path.home()/"VoxG"), stdout=lf, stderr=subprocess.STDOUT, text=True)
item={"name":name,"dataset":"elliptic","ga_mode":"normal_max_dev","returncode":p.returncode,"time_sec":time.time()-t0,"out":str(out),"log":str(log),"encode_batch_size":512}
if out.exists():
    try:
        d=json.loads(out.read_text()); item.update({"best":d.get("best"),"purity":d.get("purity")})
    except Exception as e: item["parse_error"]=str(e)
else:
    item["error"]="missing output json"
(OUTDIR/"summary_elliptic_normal_max_dev_bs512.json").write_text(json.dumps(item, indent=2))
print("RUN_DONE", json.dumps(item, ensure_ascii=False), flush=True)
