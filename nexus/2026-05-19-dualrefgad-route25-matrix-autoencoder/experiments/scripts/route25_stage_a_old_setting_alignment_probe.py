#!/usr/bin/env python3
"""Stage A old-setting alignment probe for DualRefGAD.

Runs the Route2.5 Stage-A response-matrix orientation/regime probe under
multiple explicitly named reference/encoder regimes so mat_mean can be compared
against the old 0.80+ setting without mixing protocol definitions.

No training. Frozen random VecGAD encoder. Labels are diagnostic-only for
AUC/AP, reference-purity autopsy, and aggregation.

This script supports parallel execution across multiple GPUs. Each
(variant, seed) task is independent; pass --devices 1,2,3,... to use all idle
cards while leaving any occupied GPU alone.
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
from route25_stage_a_matrix_orientation_regime_probe import run_one_seed, summarize  # noqa: E402


VARIANTS = {
    "old_exact_080_regime": {
        "definition": "Reproduce the old reference-response anatomy regime that produced mat_mean≈0.80: hop_attr descriptor, PCA residual normal model, labeled-normal normal refs, normal_soft_or anomaly scoring, descriptor-similarity anomaly refs, GT_num_layers=3, full candidate pool.",
        "changes": {
            "descriptor_mode": "hop_attr",
            "pn_estimator": "pca_residual",
            "gn_mode": "label_gate",
            "ga_mode": "normal_soft_or",
            "la_mode": "descriptor_similarity",
            "GT_num_layers": 3,
            "use_approx_anom_refs": False,
        },
        "hypothesis": "If the old mat_mean signal was mainly a reference-regime effect, mat_mean should recover positive orientation and high AUC here.",
    },
    "old_refs_gt1_low_layer": {
        "definition": "Old reference construction with GT_num_layers=1 to isolate whether the 0.80+ behavior requires the deeper 3-layer frozen GT stack.",
        "changes": {
            "descriptor_mode": "hop_attr",
            "pn_estimator": "pca_residual",
            "gn_mode": "label_gate",
            "ga_mode": "normal_soft_or",
            "la_mode": "descriptor_similarity",
            "GT_num_layers": 1,
            "use_approx_anom_refs": False,
        },
        "hypothesis": "If reference selection dominates, the signal should remain closer to old_exact than to current_stage_a even with one GT layer.",
    },
    "current_refs_gt3_layer_bridge": {
        "definition": "Current Route2.5 reference construction but with GT_num_layers=3 to isolate the GT-depth change from the reference-regime change.",
        "changes": {
            "descriptor_mode": "hop_attr_rwse",
            "pn_estimator": "diag_gaussian",
            "gn_mode": "label_gate_density",
            "ga_mode": "normal_rejection",
            "la_mode": "residual_cosine",
            "GT_num_layers": 3,
            "use_approx_anom_refs": True,
        },
        "hypothesis": "If depth alone explains the old/current discrepancy, this variant should move toward positive mat_mean; otherwise it should stay current-like.",
    },
    "current_stage_a_baseline": {
        "definition": "The already-used Route2.5 Stage A regime: hop_attr_rwse descriptor, diagonal Gaussian normal model, label_gate_density normal refs, normal_rejection/residual_cosine anomaly refs, GT_num_layers=1, approximate anomaly-ref candidate pool.",
        "changes": {
            "descriptor_mode": "hop_attr_rwse",
            "pn_estimator": "diag_gaussian",
            "gn_mode": "label_gate_density",
            "ga_mode": "normal_rejection",
            "la_mode": "residual_cosine",
            "GT_num_layers": 1,
            "use_approx_anom_refs": True,
        },
        "hypothesis": "Control condition matching the current negative-orientation Stage A result.",
    },
    "hybrid_current_descriptor_old_refs": {
        "definition": "Use current richer descriptor/diag normal model but old anomaly-reference semantics: normal_soft_or + descriptor_similarity + label-gated normal refs. This tests whether old behavior is specifically tied to old reference scoring rather than hop_attr/PCA alone.",
        "changes": {
            "descriptor_mode": "hop_attr_rwse",
            "pn_estimator": "diag_gaussian",
            "gn_mode": "label_gate",
            "ga_mode": "normal_soft_or",
            "la_mode": "descriptor_similarity",
            "GT_num_layers": 1,
            "use_approx_anom_refs": False,
        },
        "hypothesis": "If old anomaly-reference semantics are enough, mat_mean should recover more than current_stage_a even without PCA/hop_attr/GT3.",
    },
}


BASE_DEFAULTS = {
    "dataset": "elliptic",
    "device": 0,
    "train_rate": 0.05,
    "val_rate": 0.0,
    "descriptor_mode": "hop_attr_rwse",
    "pn_estimator": "diag_gaussian",
    "gn_mode": "label_gate_density",
    "ln_mode": "descriptor_similarity",
    "ga_mode": "normal_rejection",
    "la_mode": "residual_cosine",
    "normal_k": 4,
    "anom_k": 16,
    "hops": 2,
    "rw_steps": 8,
    "pca_components": 32,
    "embedding_dim": 256,
    "GT_ffn_dim": 256,
    "GT_dropout": 0.4,
    "GT_attention_dropout": 0.4,
    "GT_num_heads": 2,
    "GT_num_layers": 1,
    "pp_k": 6,
    "sample_rate": 0.15,
    "mean": 0.02,
    "var": 0.01,
    "outlier_beta": 0.3,
    "ring_R_max": 1.0,
    "ring_R_min": 0.3,
    "lambda_rec_tok": 1.0,
    "lambda_rec_emb": 0.1,
    "encode_batch_size": 1024,
    "ref_block_size": 1024,
    "use_approx_anom_refs": True,
    "anom_approx_k": 1000,
    "n_bins": 3,
    "top_k": 64,
}


def mean_std(vals):
    vals = np.asarray([v for v in vals if v is not None], dtype=np.float64)
    if vals.size == 0:
        return None
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}


def metric_from_row(row, name):
    for m in row.get("top_metrics", []):
        if m.get("name") == name:
            return m
    for fam in row.get("family_best", {}).values():
        if fam.get("name") == name:
            return fam
    return None


def summarize_variant(rows):
    base = summarize(rows)
    mat_mean_auc = []
    neg_mat_mean_auc = []
    margin_auc = []
    neg_margin_auc = []
    mat_mean_ap = []
    neg_mat_mean_ap = []
    for r in rows:
        for key, collector_auc, collector_ap in [
            ("mat_mean", mat_mean_auc, mat_mean_ap),
            ("neg_mat_mean", neg_mat_mean_auc, neg_mat_mean_ap),
            ("margin", margin_auc, None),
            ("neg_margin", neg_margin_auc, None),
        ]:
            m = metric_from_row(r, key)
            if m:
                collector_auc.append(m.get("auc"))
                if collector_ap is not None and "ap" in m:
                    collector_ap.append(m.get("ap"))
    base.update({
        "mat_mean_auc": mean_std(mat_mean_auc),
        "neg_mat_mean_auc": mean_std(neg_mat_mean_auc),
        "margin_auc": mean_std(margin_auc),
        "neg_margin_auc": mean_std(neg_margin_auc),
        "mat_mean_ap": mean_std(mat_mean_ap),
        "neg_mat_mean_ap": mean_std(neg_mat_mean_ap),
        "positive_mat_mean_wins_over_negative_count": int(sum((a or 0) > (b or 0) for a, b in zip(mat_mean_auc, neg_mat_mean_auc))),
        "negative_mat_mean_wins_over_positive_count": int(sum((b or 0) > (a or 0) for a, b in zip(mat_mean_auc, neg_mat_mean_auc))),
    })
    return base


def build_variant_args(cli_args, variant_name, seed, device):
    cfg = copy.deepcopy(BASE_DEFAULTS)
    cfg.update(VARIANTS[variant_name]["changes"])
    cfg.update({
        "project_root": cli_args.project_root,
        "dataset": cli_args.dataset,
        "device": int(device),
        "seed": int(seed),
        "seeds": cli_args.seeds,
        "train_rate": cli_args.train_rate,
        "val_rate": cli_args.val_rate,
        "ln_mode": cli_args.ln_mode,
        "normal_k": cli_args.normal_k,
        "anom_k": cli_args.anom_k,
        "out": cli_args.out,
    })
    return argparse.Namespace(**cfg)


def atomic_write_json(path, payload):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def leaderboard(results):
    return sorted([
        {
            "variant": r["variant"],
            "mat_mean_auc_mean": (r["aggregate"].get("mat_mean_auc") or {}).get("mean"),
            "neg_mat_mean_auc_mean": (r["aggregate"].get("neg_mat_mean_auc") or {}).get("mean"),
            "anom_ref_anom_ratio_mean": (r["aggregate"].get("anom_ref_anom_ratio") or {}).get("mean"),
            "best_scalar_auc_mean": (r["aggregate"].get("best_scalar_auc") or {}).get("mean"),
            "winner_counts": r["aggregate"].get("winner_counts"),
            "decision": r["aggregate"].get("decision"),
        }
        for r in results
    ], key=lambda x: (x["mat_mean_auc_mean"] if x["mat_mean_auc_mean"] is not None else -1), reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(Path.home() / "DualRefGAD"))
    ap.add_argument("--dataset", default="elliptic")
    ap.add_argument("--device", type=int, default=0, help="Fallback single GPU if --devices is omitted")
    ap.add_argument("--devices", default="", help="Comma-separated GPUs for parallel tasks, e.g. 1,2,3,4,5,6,7")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--variants", default="old_exact_080_regime,old_refs_gt1_low_layer,current_refs_gt3_layer_bridge,current_stage_a_baseline,hybrid_current_descriptor_old_refs")
    ap.add_argument("--train_rate", type=float, default=0.05)
    ap.add_argument("--val_rate", type=float, default=0.0)
    ap.add_argument("--ln_mode", default="descriptor_similarity")
    ap.add_argument("--normal_k", type=int, default=4)
    ap.add_argument("--anom_k", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress_out", default="")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}; available={list(VARIANTS)}")
    devices = [int(x.strip()) for x in args.devices.split(",") if x.strip()] if args.devices.strip() else [int(args.device)]
    if not devices:
        devices = [int(args.device)]

    start = time.time()
    task_q = queue.Queue()
    for v in variants:
        for s in seeds:
            task_q.put((v, s))
    total = task_q.qsize()
    rows_by_variant = {v: [] for v in variants}
    errors = []
    done = 0
    lock = threading.Lock()

    def snapshot(status="running", current=None):
        partial_results = []
        for vv in variants:
            rr = rows_by_variant[vv]
            entry = {"variant": vv, "completed_seeds": sorted([int(r["seed"]) for r in rr]), "n_completed": len(rr)}
            if len(rr) == len(seeds):
                entry["aggregate"] = summarize_variant(sorted(rr, key=lambda r: r["seed"]))
            partial_results.append(entry)
        payload = {
            "status": status,
            "done": done,
            "total": total,
            "devices": devices,
            "current": current,
            "errors": errors[-5:],
            "elapsed_sec": time.time() - start,
            "partial_results": partial_results,
        }
        atomic_write_json(args.progress_out, payload)

    def worker(device):
        nonlocal done
        # Restrict this Python thread/process work to the assigned visible GPU.
        # run_one_seed uses cuda:{args.device}; with a full system visibility we pass
        # the actual GPU id. This is one process with multiple threads, but each task
        # is sequential per worker device.
        while True:
            try:
                v, s = task_q.get_nowait()
            except queue.Empty:
                return
            current = {"variant": v, "seed": s, "device": int(device)}
            try:
                with lock:
                    print(json.dumps({"stage": "task_start", **current, "definition": VARIANTS[v]["definition"]}, ensure_ascii=False), flush=True)
                    snapshot("running", current)
                var_args = build_variant_args(args, v, s, device)
                row = run_one_seed(var_args, s)
                row["device"] = int(device)
                with lock:
                    rows_by_variant[v].append(row)
                    done += 1
                    print(json.dumps({"stage": "task_done", **current, "best": row["best_scalar"]["name"], "auc": row["best_scalar"]["auc"]}, ensure_ascii=False), flush=True)
                    snapshot("running", current)
            except Exception as e:
                tb = traceback.format_exc()
                with lock:
                    done += 1
                    errors.append({"variant": v, "seed": int(s), "device": int(device), "error": repr(e), "traceback": tb[-4000:]})
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

    results = []
    for v in variants:
        rows = sorted(rows_by_variant[v], key=lambda r: r["seed"])
        if len(rows) != len(seeds):
            errors.append({"variant": v, "error": f"completed {len(rows)}/{len(seeds)} seeds"})
        if rows:
            agg = summarize_variant(rows)
        else:
            agg = {}
        results.append({
            "variant": v,
            "definition": VARIANTS[v]["definition"],
            "hypothesis": VARIANTS[v]["hypothesis"],
            "changes_from_base": VARIANTS[v]["changes"],
            "rows": rows,
            "aggregate": agg,
        })

    status = "finished" if not errors and all(len(rows_by_variant[v]) == len(seeds) for v in variants) else "failed"
    lb = leaderboard(results)
    output = {
        "status": status,
        "probe": "route25_stage_a_old_setting_alignment_probe",
        "protocol": "Frozen encoder; no AE training; 5-seed multi-variant Stage-A alignment; labels diagnostic-only. Variants explicitly define reference/descriptor/GT regimes. Parallel GPU execution used independent variant/seed tasks.",
        "seeds": seeds,
        "devices": devices,
        "variants_requested": variants,
        "variant_definitions": {k: VARIANTS[k] for k in variants},
        "results": results,
        "leaderboard_by_mat_mean_auc": lb,
        "errors": errors,
        "time_sec": float(time.time() - start),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    atomic_write_json(args.progress_out, {"status": status, "done": done, "total": total, "devices": devices, "elapsed_sec": time.time() - start, "leaderboard_by_mat_mean_auc": lb, "errors": errors[-10:]})
    print("FINAL " + json.dumps({"status": status, "leaderboard_by_mat_mean_auc": lb, "errors": errors[-3:], "time_sec": output["time_sec"]}, indent=2, ensure_ascii=False), flush=True)
    if status != "finished":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
