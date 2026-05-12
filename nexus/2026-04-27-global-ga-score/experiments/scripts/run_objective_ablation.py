#!/usr/bin/env python3
"""
Objective Ablation Runner - 旧 objective vs 新 objective

对比：
  - self_residual: 旧版 pseudo anomaly（由 normal_emb 自己的 residual 方向）
  - ref_guided: 新版 pseudo anomaly（由 pooled R_a - R_n 方向）

数据集：Photo, Elliptic
"""

import subprocess, json, time, sys
from pathlib import Path

ROOT = Path.home() / "VoxG"
SCRIPT = ROOT / "nexus/investigations/2026-04-27-global-ga-score/experiments/scripts/run_tokenization_theory_gt_refguided.py"
OUTDIR = ROOT / "nexus/investigations/2026-04-27-global-ga-score/experiments/outputs/objective_ablation"

DATASETS = [
    {
        "name": "photo",
        "device": 4,
        "encode_batch_size": 1024,
        "ga_mode": "normal_rejection",
        "la_mode": "residual_cosine"
    },
    {
        "name": "elliptic",
        "device": 2,
        "encode_batch_size": 512,
        "ga_mode": "normal_soft_or",
        "la_mode": "descriptor_similarity"
    }
]

OBJECTIVE_MODES = ["self_residual", "ref_guided"]

def run_one(dataset, objective_mode):
    d = DATASETS[dataset]
    out = OUTDIR / f"{d['name']}__{objective_mode}.json"
    cmd = [
        "python3", str(SCRIPT),
        "--dataset", d['name'],
        "--seed", "0",
        "--train_rate", "0.05",
        "--num_epoch", "200",
        "--wandb", "true",
        "--device", str(d['device']),
        "--encode_batch_size", str(d['encode_batch_size']),
        "--descriptor_mode", "hop_attr",
        "--pn_estimator", "pca_residual",
        "--gn_mode", "label_gate",
        "--ln_mode", "descriptor_similarity",
        "--ga_mode", d['ga_mode'],
        "--la_mode", d['la_mode'],
        "--ablation_mode", "full",
        "--normal_k", "4",
        "--anom_k", "16",
        "--pp_k", "6",
        "--objective_mode", objective_mode,
        "--out", str(out)
    ]
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] START {d['name']} {objective_mode} device={d['device']}", flush=True)
    start = time.time()
    rc = subprocess.run(cmd).returncode
    elapsed = time.time() - start
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] END {d['name']} {objective_mode} rc={rc} elapsed={elapsed:.1f}s", flush=True)
    
    result = None
    if out.exists():
        try:
            result = json.loads(out.read_text())
        except:
            pass
    return {"dataset": d['name'], "objective_mode": objective_mode, "returncode": rc, "elapsed_sec": elapsed, "result": result}

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    results = []
    
    # 运行顺序：photo self_residual, photo ref_guided, elliptic self_residual, elliptic ref_guided
    # 这样可以并行两个数据集的不同 objective（如果用不同 GPU）
    # 但为了简单，我们串行
    
    for d_name in ["photo", "elliptic"]:
        d_idx = 0 if d_name == "photo" else 1
        for obj_mode in OBJECTIVE_MODES:
            r = run_one(d_idx, obj_mode)
            results.append(r)
    
    summary = OUTDIR / "summary.json"
    summary.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("ALL_DONE", flush=True)
    
    # 打印对比表格
    print("\n=== Objective Ablation Summary ===", flush=True)
    for r in results:
        b = r.get("result", {}).get("best", {})
        print(f"{r['dataset']} {r['objective_mode']}: val_auc={b.get('val_auc','N/A'):.4f} val_ap={b.get('val_ap','N/A'):.4f} test_auc={b.get('test_auc','N/A'):.4f} test_ap={b.get('test_ap','N/A'):.4f} best_epoch={b.get('epoch','N/A')}", flush=True)

if __name__ == '__main__':
    main()