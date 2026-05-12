#!/usr/bin/env python3
import json, subprocess, sys, time
from pathlib import Path

ROOT = Path.home()/"VoxG"
SCRIPT = ROOT/"nexus/investigations/2026-04-27-global-ga-score/experiments/scripts/run_tokenization_theory_gt.py"
OUTDIR = ROOT/"nexus/investigations/2026-04-27-global-ga-score/experiments/outputs/tokenization_theory_mini_gt_best_ablation"
OUTDIR.mkdir(parents=True, exist_ok=True)

CONFIGS = {
  "photo": dict(device=4, encode_batch_size=1024, descriptor_mode="hop_attr", pn_estimator="pca_residual", gn_mode="label_gate", ln_mode="descriptor_similarity", ga_mode="normal_rejection", la_mode="residual_cosine"),
  "elliptic": dict(device=2, encode_batch_size=512, descriptor_mode="hop_attr", pn_estimator="pca_residual", gn_mode="label_gate", ln_mode="descriptor_similarity", ga_mode="normal_soft_or", la_mode="descriptor_similarity"),
}
ABLATIONS = ["full", "no_ra", "shuffled_ra", "fixed_labeled_normal"]

def ts():
    return time.strftime('%F %T')

def run_dataset(ds):
    cfg = CONFIGS[ds]
    summary=[]
    log_path=OUTDIR/f"{ds}.log"
    with open(log_path,"a",buffering=1) as lf:
        for abl in ABLATIONS:
            name=f"{ds}__{abl}.json"
            out=OUTDIR/name
            cmd=[sys.executable,str(SCRIPT),"--dataset",ds,"--seed","0","--train_rate","0.05","--num_epoch","200","--wandb","true","--device",str(cfg["device"]),"--encode_batch_size",str(cfg["encode_batch_size"]),"--descriptor_mode",cfg["descriptor_mode"],"--pn_estimator",cfg["pn_estimator"],"--gn_mode",cfg["gn_mode"],"--ln_mode",cfg["ln_mode"],"--ga_mode",cfg["ga_mode"],"--la_mode",cfg["la_mode"],"--ablation_mode",abl,"--normal_k","4","--anom_k","16","--pp_k","6","--out",str(out)]
            print(f"[{ts()}] START {ds} {abl} device={cfg['device']}", flush=True); lf.write(f"START {ds} {abl}\n")
            t0=time.time(); proc=subprocess.run(cmd,cwd=str(ROOT),text=True,capture_output=True); elapsed=time.time()-t0
            rec={"dataset":ds,"ablation_mode":abl,"returncode":proc.returncode,"elapsed_sec":elapsed,"file":str(out),"stdout_tail":proc.stdout[-4000:],"stderr_tail":proc.stderr[-4000:]}
            if proc.returncode==0 and out.exists():
                try: rec["result"]=json.loads(out.read_text())
                except Exception as e: rec["parse_error"]=str(e)
            else:
                rec["error"]=(proc.stderr or proc.stdout)[-4000:]
            summary.append(rec)
            (OUTDIR/f"{ds}_summary_partial.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False))
            lf.write(proc.stdout[-3000:]+"\n"+proc.stderr[-3000:]+"\n")
            print(f"[{ts()}] END {ds} {abl} rc={proc.returncode} elapsed={elapsed:.1f}s", flush=True)
            if proc.returncode != 0:
                break
    (OUTDIR/f"{ds}_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print("DATASET_DONE", ds, OUTDIR/f"{ds}_summary.json")

if __name__=="__main__":
    run_dataset(sys.argv[1])
