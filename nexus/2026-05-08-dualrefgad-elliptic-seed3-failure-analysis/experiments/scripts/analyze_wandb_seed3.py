#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import math
import statistics as stats

import wandb

ENTITY = "HCCS"
PROJECT = "DualRefGAD"
SWEEP_ID = "0d0py9y1"
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

KEYS = [
    "final_test_auc", "final_test_ap", "final_train_auc",
    "final_score_std", "final_sn_std", "final_sa_std", "final_margin_std", "final_margin_test_mean",
    "normal_ref_normal_ratio", "anom_ref_anom_ratio", "anom_ref_anom_ratio_on_anom_nodes",
    "n_labeled_normal", "best_val_auc", "best_test_auc", "best_epoch",
]

HIST_KEYS = ["epoch", "test_auc", "test_ap", "val_auc", "val_ap", "loss", "score_std", "sn_std", "sa_std", "margin_std", "margin_test_mean"]


def clean(v):
    try:
        if isinstance(v, float) and math.isnan(v):
            return None
    except Exception:
        pass
    return v


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    return stats.mean(xs), stats.stdev(xs)


def zscore(x, xs):
    xs = [v for v in xs if v is not None]
    if x is None or len(xs) < 2:
        return None
    m, s = stats.mean(xs), stats.stdev(xs)
    return None if s == 0 else (x - m) / s


def main():
    api = wandb.Api()
    sweep = api.sweep(f"{ENTITY}/{PROJECT}/{SWEEP_ID}")
    rows = []
    histories = {}
    for run in sweep.runs:
        cfg = dict(run.config or {})
        seed = cfg.get("seed")
        summ = dict(run.summary or {})
        row = {
            "run_id": run.id,
            "name": run.name,
            "state": run.state,
            "seed": seed,
            "url": run.url,
        }
        for k in KEYS:
            row[k] = clean(summ.get(k))
        rows.append(row)
        hist = run.history(keys=HIST_KEYS, pandas=True)
        histories[str(seed)] = hist.to_dict(orient="records")
    rows.sort(key=lambda r: r.get("seed"))

    # Per-key aggregate + seed3 z-scores against all seeds and non-seed3 seeds.
    seed3 = next((r for r in rows if r.get("seed") == 3), None)
    diag = {"sweep": f"{ENTITY}/{PROJECT}/{SWEEP_ID}", "runs": rows, "seed3": seed3, "metrics": {}}
    for k in KEYS:
        vals = [r.get(k) for r in rows if isinstance(r.get(k), (int, float))]
        m, s = mean_std(vals)
        vals_wo3 = [r.get(k) for r in rows if r.get("seed") != 3 and isinstance(r.get(k), (int, float))]
        m4, s4 = mean_std(vals_wo3)
        x3 = seed3.get(k) if seed3 else None
        diag["metrics"][k] = {
            "mean": m, "std": s,
            "mean_without_seed3": m4, "std_without_seed3": s4,
            "seed3": x3,
            "seed3_z_all": zscore(x3, vals),
            "seed3_delta_from_non3_mean": None if x3 is None or m4 is None else x3 - m4,
        }

    # History degradation: final - max over epochs for test_auc/ap (diagnostic only).
    deg = {}
    for r in rows:
        seed = str(r["seed"])
        hist = histories.get(seed) or []
        for metric in ["test_auc", "test_ap"]:
            series = [h.get(metric) for h in hist if isinstance(h.get(metric), (int, float))]
            if not series:
                continue
            final = series[-1]
            peak = max(series)
            peak_epoch = next((h.get("epoch") for h in hist if h.get(metric) == peak), None)
            deg.setdefault(seed, {})[metric] = {"final": final, "peak": peak, "drop": peak - final, "peak_epoch": peak_epoch}
    diag["epoch_degradation"] = deg

    (OUT / "wandb_runs_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "wandb_histories.json").write_text(json.dumps(histories, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "seed3_diagnosis.json").write_text(json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown report
    lines = ["# Seed 3 First-Pass Diagnosis", "", f"Sweep: `{ENTITY}/{PROJECT}/{SWEEP_ID}`", ""]
    lines += ["## Summary", "", "| seed | run | final AUC | final AP | score std | margin std | anom ref ratio | anom ref on anom |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append("| {seed} | `{run_id}` | {auc:.4f} | {ap:.4f} | {score:.4f} | {margin:.4f} | {ar:.4f} | {aroa:.4f} |".format(
            seed=r.get("seed"), run_id=r.get("run_id"), auc=r.get("final_test_auc"), ap=r.get("final_test_ap"),
            score=r.get("final_score_std"), margin=r.get("final_margin_std"), ar=r.get("anom_ref_anom_ratio"), aroa=r.get("anom_ref_anom_ratio_on_anom_nodes")))
    lines += ["", "## Seed 3 deltas from non-seed3 mean", "", "| metric | seed3 | non3 mean | delta | z(all) |", "|---|---:|---:|---:|---:|"]
    for k in ["final_test_auc", "final_test_ap", "final_score_std", "final_sn_std", "final_sa_std", "final_margin_std", "normal_ref_normal_ratio", "anom_ref_anom_ratio", "anom_ref_anom_ratio_on_anom_nodes"]:
        d = diag["metrics"][k]
        lines.append(f"| {k} | {d['seed3'] if d['seed3'] is not None else '-'} | {d['mean_without_seed3'] if d['mean_without_seed3'] is not None else '-'} | {d['seed3_delta_from_non3_mean'] if d['seed3_delta_from_non3_mean'] is not None else '-'} | {d['seed3_z_all'] if d['seed3_z_all'] is not None else '-'} |")
    lines += ["", "## Epoch degradation", "", "| seed | metric | peak | final | drop | peak epoch |", "|---:|---|---:|---:|---:|---:|"]
    for seed, dd in sorted(deg.items(), key=lambda x: int(x[0])):
        for metric, v in dd.items():
            lines.append(f"| {seed} | {metric} | {v['peak']:.4f} | {v['final']:.4f} | {v['drop']:.4f} | {v['peak_epoch']} |")
    (OUT / "seed3_first_pass_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT)
    print("seed3 final_auc", seed3.get("final_test_auc") if seed3 else None)

if __name__ == "__main__":
    main()
