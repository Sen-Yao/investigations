#!/usr/bin/env python3
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
        seed=run.config.get("seed")
        rows.append({"seed":seed, "margin_auc":margin_auc, "run_id":run.id, "state":run.state})
    rows.sort(key=lambda x:x["seed"])
    vals=[r["margin_auc"] for r in rows if isinstance(r["margin_auc"],(int,float))]
    mean=st.mean(vals) if vals else None
    std=st.stdev(vals) if len(vals)>1 else 0.0
    out={"sweep":SWEEP, "runs":rows, "mean":mean, "std":std}
    print(json.dumps(out,indent=2,ensure_ascii=False))
    print("margin-only AUC:", f"{mean:.4f}±{std:.4f}" if mean else "N/A")
if __name__=="__main__":
    main()