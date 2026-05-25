#!/usr/bin/env python3
"""Parallel mat_mean reference-length scan across multiple GPUs.

Scientific question:
- After AMRF failed to beat simple matrix summaries in the old 0.80-ish regime,
  scan normal/reference lengths more thoroughly to see where mat_mean or
  -mat_mean is stable, and whether the signal is tied to reference pollution
  or degree correlation.

Protocol:
- Frozen VecGAD encoder / Route2.5 response matrix construction.
- No AE training.
- Labels diagnostic-only for AUC/AP and reference autopsy.
- Independent (variant, normal_k, anom_k, seed) tasks are scheduled across
  --devices with a thread queue, one worker per GPU.
"""
import argparse
import copy
import json
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from route25_ref_length_mat_mean_scan import run_one, mean_std  # noqa: E402


VARIANTS = {
    "current": {
        "cn_name": "当前构造",
        "definition": "Route2.5 当前参考点构造：hop_attr_rwse + diag_gaussian + label_gate_density + normal_rejection + residual_cosine + GT1 + approximate anomaly refs.",
        "changes": {
            "descriptor_mode": "hop_attr_rwse",
            "pn_estimator": "diag_gaussian",
            "gn_mode": "label_gate_density",
            "ga_mode": "normal_rejection",
            "la_mode": "residual_cosine",
            "GT_num_layers": 1,
            "use_approx_anom_refs": True,
        },
    },
    "old_semantic": {
        "cn_name": "旧语义",
        "definition": "恢复早期异常参考点语义方向：hop_attr_rwse + diag_gaussian + label_gate + normal_soft_or + descriptor_similarity + GT1.",
        "changes": {
            "descriptor_mode": "hop_attr_rwse",
            "pn_estimator": "diag_gaussian",
            "gn_mode": "label_gate",
            "ga_mode": "normal_soft_or",
            "la_mode": "descriptor_similarity",
            "GT_num_layers": 1,
            "use_approx_anom_refs": False,
        },
    },
    "old_exact": {
        "cn_name": "旧复刻",
        "definition": "尽量复刻早期 0.80 左右强 mat_mean 设置：hop_attr + pca_residual + label_gate + normal_soft_or + descriptor_similarity + GT3.",
        "changes": {
            "descriptor_mode": "hop_attr",
            "pn_estimator": "pca_residual",
            "gn_mode": "label_gate",
            "ga_mode": "normal_soft_or",
            "la_mode": "descriptor_similarity",
            "GT_num_layers": 3,
            "use_approx_anom_refs": False,
        },
    },
}


def parse_int_list(s):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def atomic_write_json(path, payload):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def make_variant_args(cli_args, variant, seed, normal_k, anom_k, device):
    cfg = vars(cli_args).copy()
    cfg.update(VARIANTS[variant]["changes"])
    cfg.update({
        "seed": int(seed),
        "normal_k": int(normal_k),
        "anom_k": int(anom_k),
        "device": int(device),
        "use_approx_anom_refs": bool(VARIANTS[variant]["changes"].get("use_approx_anom_refs", False)),
    })
    # Fields not consumed by the serial helper should not matter, but keeping the
    # Namespace close to argparse shape makes helper reuse safer.
    return argparse.Namespace(**cfg)


def summarize(rows):
    by_key = {}
    for r in rows:
        key = (r["variant"], r["normal_k"], r["anom_k"])
        by_key.setdefault(key, []).append(r)
    pair_summaries = []
    for (variant, nk, ak), vals in sorted(by_key.items()):
        mat_auc = [v["metrics"]["mat_mean"]["auc"] for v in vals]
        neg_auc = [v["metrics"]["neg_mat_mean"]["auc"] for v in vals]
        mat_ap = [v["metrics"]["mat_mean"]["ap"] for v in vals]
        neg_ap = [v["metrics"]["neg_mat_mean"]["ap"] for v in vals]
        margin_auc = [v["metrics"]["margin"]["auc"] for v in vals]
        neg_margin_auc = [v["metrics"]["neg_margin"]["auc"] for v in vals]
        ref_ratio = [v["reference_diagnostics"]["anom_ref_anom_ratio_diagnostic"] for v in vals]
        normal_ref_ratio = [v["reference_diagnostics"]["normal_ref_anom_ratio_diagnostic"] for v in vals]
        degree_corr = [v["correlations"]["mat_mean_spearman_with_degree"] for v in vals]
        margin_corr = [v["correlations"]["mat_mean_spearman_with_margin"] for v in vals]
        best_orientation_auc = [max(a, b) for a, b in zip(mat_auc, neg_auc)]
        pair_summaries.append({
            "variant": variant,
            "variant_cn_name": VARIANTS[variant]["cn_name"],
            "normal_k": int(nk),
            "anom_k": int(ak),
            "flatten_dim": int(nk * ak),
            "num_seeds": len(vals),
            "mat_mean_auc": mean_std(mat_auc),
            "neg_mat_mean_auc": mean_std(neg_auc),
            "best_orientation_auc": mean_std(best_orientation_auc),
            "mat_mean_ap": mean_std(mat_ap),
            "neg_mat_mean_ap": mean_std(neg_ap),
            "margin_auc": mean_std(margin_auc),
            "neg_margin_auc": mean_std(neg_margin_auc),
            "mat_mean_minus_neg_mat_mean_auc": mean_std([a - b for a, b in zip(mat_auc, neg_auc)]),
            "anom_ref_anom_ratio": mean_std(ref_ratio),
            "normal_ref_anom_ratio": mean_std(normal_ref_ratio),
            "mat_mean_degree_spearman": mean_std(degree_corr),
            "mat_mean_margin_spearman": mean_std(margin_corr),
            "positive_orientation_wins": int(sum(1 for v in vals if v["metrics"]["mat_mean"]["auc"] >= v["metrics"]["neg_mat_mean"]["auc"])),
            "negative_orientation_wins": int(sum(1 for v in vals if v["metrics"]["neg_mat_mean"]["auc"] > v["metrics"]["mat_mean"]["auc"])),
        })
    by_variant = {}
    for v in VARIANTS:
        pv = [p for p in pair_summaries if p["variant"] == v]
        if not pv:
            continue
        by_variant[v] = {
            "variant_cn_name": VARIANTS[v]["cn_name"],
            "definition": VARIANTS[v]["definition"],
            "pair_count": len(pv),
            "best_mat_mean_pair": max(pv, key=lambda x: x["mat_mean_auc"]["mean"]),
            "best_neg_mat_mean_pair": max(pv, key=lambda x: x["neg_mat_mean_auc"]["mean"]),
            "best_either_orientation_pair": max(pv, key=lambda x: x["best_orientation_auc"]["mean"]),
            "mean_best_orientation_over_grid": mean_std([p["best_orientation_auc"]["mean"] for p in pv]),
            "mean_anom_ref_ratio_over_grid": mean_std([p["anom_ref_anom_ratio"]["mean"] for p in pv]),
            "mean_degree_corr_over_grid": mean_std([p["mat_mean_degree_spearman"]["mean"] for p in pv]),
        }
    return {
        "pairs": pair_summaries,
        "by_variant": by_variant,
        "leaderboard_best_either": sorted(pair_summaries, key=lambda x: x["best_orientation_auc"]["mean"], reverse=True)[:20],
        "leaderboard_mat_mean": sorted(pair_summaries, key=lambda x: x["mat_mean_auc"]["mean"], reverse=True)[:20],
        "leaderboard_neg_mat_mean": sorted(pair_summaries, key=lambda x: x["neg_mat_mean_auc"]["mean"], reverse=True)[:20],
        "grid_count": len(pair_summaries),
        "run_count": len(rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(Path.home() / "DualRefGAD"))
    ap.add_argument("--dataset", default="elliptic")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--devices", default="")
    ap.add_argument("--variants", default="current,old_semantic,old_exact")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--normal_ks", default="1,2,4,8,16")
    ap.add_argument("--anom_ks", default="4,8,16,32,64")
    ap.add_argument("--train_rate", type=float, default=0.05)
    ap.add_argument("--val_rate", type=float, default=0.0)
    ap.add_argument("--descriptor_mode", choices=["hop_attr", "rwse", "hop_attr_rwse"], default="hop_attr_rwse")
    ap.add_argument("--pn_estimator", choices=["diag_gaussian", "pca_residual"], default="diag_gaussian")
    ap.add_argument("--gn_mode", choices=["label_gate", "normal_density", "label_gate_density"], default="label_gate_density")
    ap.add_argument("--ln_mode", default="descriptor_similarity")
    ap.add_argument("--ga_mode", choices=["normal_rejection", "residual_norm", "normal_soft_or"], default="normal_rejection")
    ap.add_argument("--la_mode", choices=["residual_cosine", "descriptor_similarity"], default="residual_cosine")
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--rw_steps", type=int, default=8)
    ap.add_argument("--pca_components", type=int, default=32)
    ap.add_argument("--embedding_dim", type=int, default=256)
    ap.add_argument("--GT_ffn_dim", type=int, default=256)
    ap.add_argument("--GT_dropout", type=float, default=0.4)
    ap.add_argument("--GT_attention_dropout", type=float, default=0.4)
    ap.add_argument("--GT_num_heads", type=int, default=2)
    ap.add_argument("--GT_num_layers", type=int, default=1)
    ap.add_argument("--pp_k", type=int, default=6)
    ap.add_argument("--sample_rate", type=float, default=0.15)
    ap.add_argument("--mean", type=float, default=0.02)
    ap.add_argument("--var", type=float, default=0.01)
    ap.add_argument("--outlier_beta", type=float, default=0.3)
    ap.add_argument("--ring_R_max", type=float, default=1.0)
    ap.add_argument("--ring_R_min", type=float, default=0.3)
    ap.add_argument("--lambda_rec_tok", type=float, default=1.0)
    ap.add_argument("--lambda_rec_emb", type=float, default=0.1)
    ap.add_argument("--encode_batch_size", type=int, default=1024)
    ap.add_argument("--ref_block_size", type=int, default=1024)
    ap.add_argument("--use_approx_anom_refs", action="store_true")
    ap.add_argument("--anom_approx_k", type=int, default=1000)
    ap.add_argument("--progress_out", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}; available={list(VARIANTS)}")
    seeds = parse_int_list(args.seeds)
    normal_ks = parse_int_list(args.normal_ks)
    anom_ks = parse_int_list(args.anom_ks)
    devices = parse_int_list(args.devices) if str(args.devices).strip() else [int(args.device)]
    if not devices:
        devices = [int(args.device)]

    start = time.time()
    task_q = queue.Queue()
    for variant in variants:
        for nk in normal_ks:
            for ak in anom_ks:
                for seed in seeds:
                    task_q.put((variant, nk, ak, seed))
    total = task_q.qsize()
    rows = []
    errors = []
    done = 0
    lock = threading.Lock()

    def snapshot(status="running", current=None):
        completed_by_variant = {v: 0 for v in variants}
        for r in rows:
            completed_by_variant[r["variant"]] = completed_by_variant.get(r["variant"], 0) + 1
        payload = {
            "status": status,
            "done": done,
            "total": total,
            "devices": devices,
            "variants": variants,
            "normal_ks": normal_ks,
            "anom_ks": anom_ks,
            "current": current,
            "completed_by_variant": completed_by_variant,
            "errors": errors[-5:],
            "elapsed_sec": time.time() - start,
        }
        if rows:
            payload["partial_best_either"] = summarize(rows)["leaderboard_best_either"][:5]
        atomic_write_json(args.progress_out, payload)

    def worker(device):
        nonlocal done
        while True:
            try:
                variant, nk, ak, seed = task_q.get_nowait()
            except queue.Empty:
                return
            current = {"variant": variant, "variant_cn_name": VARIANTS[variant]["cn_name"], "normal_k": int(nk), "anom_k": int(ak), "seed": int(seed), "device": int(device)}
            try:
                with lock:
                    print(json.dumps({"stage": "task_start", **current}, ensure_ascii=False), flush=True)
                    snapshot("running", current)
                base_args = make_variant_args(args, variant, seed, nk, ak, device)
                row = run_one(args, base_args, seed, nk, ak)
                row["variant"] = variant
                row["variant_cn_name"] = VARIANTS[variant]["cn_name"]
                row["variant_definition"] = VARIANTS[variant]["definition"]
                row["device"] = int(device)
                with lock:
                    rows.append(row)
                    done += 1
                    print(json.dumps({"stage": "task_done", **current, "best": row["best"]["name"], "best_auc": row["best"]["auc"]}, ensure_ascii=False), flush=True)
                    snapshot("running", current)
            except Exception as e:
                tb = traceback.format_exc()
                with lock:
                    done += 1
                    errors.append({**current, "error": repr(e), "traceback": tb[-4000:]})
                    print(json.dumps({"stage": "task_failed", **current, "error": repr(e)}, ensure_ascii=False), flush=True)
                    snapshot("failed_running", current)
            finally:
                task_q.task_done()

    threads = []
    for d in devices:
        t = threading.Thread(target=worker, args=(d,), daemon=False)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    status = "finished" if not errors and len(rows) == total else "failed"
    agg = summarize(rows) if rows else {"pairs": [], "by_variant": {}, "leaderboard_best_either": []}
    output = {
        "status": status,
        "probe": "route25_parallel_mat_mean_reference_scan",
        "protocol": "Frozen encoder; no AE training; labels diagnostic-only; parallel GPU queue across variants x normal_k x anom_k x seed.",
        "dataset": args.dataset,
        "variants": variants,
        "variant_definitions": VARIANTS,
        "seeds": seeds,
        "normal_ks": normal_ks,
        "anom_ks": anom_ks,
        "devices": devices,
        "config": vars(args),
        "rows": sorted(rows, key=lambda r: (r["variant"], r["normal_k"], r["anom_k"], r["seed"])),
        "aggregate": agg,
        "errors": errors,
        "time_sec": float(time.time() - start),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    atomic_write_json(args.progress_out, {"status": status, "done": done, "total": total, "elapsed_sec": time.time() - start, "out": str(out), "leaderboard_best_either": agg.get("leaderboard_best_either", [])[:10], "errors": errors[-10:]})
    print("FINAL " + json.dumps({"status": status, "top10": agg.get("leaderboard_best_either", [])[:10], "errors": errors[-3:], "time_sec": output["time_sec"]}, indent=2, ensure_ascii=False), flush=True)
    if status != "finished":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
