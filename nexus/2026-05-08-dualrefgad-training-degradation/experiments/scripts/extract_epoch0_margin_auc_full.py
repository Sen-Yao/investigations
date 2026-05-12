#!/usr/bin/env python3
from pathlib import Path
import wandb, statistics as st, math, json
ENTITY="HCCS"
PROJECT="DualRefGAD"
SWEEP="s9u6hm9g"

def main():
    api=wandb.Api()
    sweep=api.sweep(f"{ENTITY}/{PROJECT}/{SWEEP}")
    rows=[]
    for run in sweep.runs:
        summ=run.summary
        e0=summ.get("epoch0_diagnostic_auc") or {}
        margin_auc=e0.get("margin_auc")
        margin_ap=e0.get("margin_ap")
        seed=run.config.get("seed")
        rows.append({"seed":seed, "margin_auc":margin_auc, "margin_ap":margin_ap, "run_id":run.id, "state":run.state})
    rows.sort(key=lambda x:x["seed"])
    auc_vals=[r["margin_auc"] for r in rows if isinstance(r["margin_auc"],(int,float))]
    ap_vals=[r["margin_ap"] for r in rows if isinstance(r["margin_ap"],(int,float))]
    auc_mean=st.mean(auc_vals) if auc_vals else None
    auc_std=st.stdev(auc_vals) if len(auc_vals)>1 else 0.0
    ap_mean=st.mean(ap_vals) if ap_vals else None
    ap_std=st.stdev(ap_vals) if len(ap_vals)>1 else 0.0
    out={"sweep":SWEEP, "runs":rows, "auc_mean":auc_mean, "auc_std":auc_std, "ap_mean":ap_mean, "ap_std":ap_std}
    (Path(__file__).resolve().parents[1] / "outputs" / "margin_only_5seed.json").write_text(json.dumps(out,indent=2,ensure_ascii=False))
    print("margin-only AUC:", f"{auc_mean:.4f}±{auc_std:.4f}" if auc_mean else "N/A")
    print("margin-only AP: ", f"{ap_mean:.4f}±{ap_std:.4f}" if ap_mean else "N/A")

if __name__=="__main__":
    main()