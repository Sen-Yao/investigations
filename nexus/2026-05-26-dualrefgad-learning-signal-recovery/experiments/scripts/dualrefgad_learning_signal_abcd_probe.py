#!/usr/bin/env python3
"""DualRefGAD Learning Signal Recovery ABCD bundled probe.

This is a thin orchestration wrapper over the already validated C-LEG3 Layer-1
label-free shallow-gate probe. It reuses the trusted response-matrix and
candidate-score construction, then organizes the output into the report-defined
ABCD phases:

A. Top-K failure autopsy extension
B. Fragmentation decomposition probe
C. Reference relation reliability probe
D. Trainable target readiness

Scientific boundaries:
- no anomaly-label training;
- labels are diagnostic-only for AUC/AP and top-K autopsy;
- the output is target-readiness evidence, not a deployable trained method.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# On HCCS-25 this dependency already lives in ~/DualRefGAD/experiments/scripts.
# In the local investigation archive, the validator can still parse this file;
# the remote preflight verifies the dependency before launch.
from cleg3_layer1_label_free_shallow_gate_probe import (  # noqa: E402
    VARIANTS,
    atomic_write_json,
    mean_std,
    parse_floats,
    parse_ints,
    run_one as run_layer1_one,
    summarize as summarize_layer1,
)


def _mean(obj, default=None):
    if isinstance(obj, dict):
        return obj.get("mean", default)
    return default


def _best_by(rows, key, reverse=True):
    valid = [r for r in rows if r.get(key) is not None]
    if not valid:
        return None
    return sorted(valid, key=lambda r: r.get(key), reverse=reverse)[0]


def build_phase_a(rows, aggregate):
    """Top-K failure autopsy extension."""
    cats = aggregate.get("oracle_category_counts", {})
    effect_fields = {}
    for field in [
        "mat_std", "row_mean_range", "col_mean_range", "fragmentation_penalty",
        "row_effective_count", "col_effective_count", "joint_reliability",
        "mixture_support", "consensus_minus_fragmentation", "degree", "rejection", "residual_norm",
    ]:
        vals_lost = []
        vals_rescued = []
        for r in rows:
            fmap = r.get("category_effect_size_map", {}).get(field, {})
            vals_lost.append(fmap.get("lost_minus_removed_fp"))
            vals_rescued.append(fmap.get("rescued_minus_introduced_fp"))
        effect_fields[field] = {
            "lost_minus_removed_fp": mean_std(vals_lost),
            "rescued_minus_introduced_fp": mean_std(vals_rescued),
        }
    return {
        "phase": "A",
        "name": "Top-K failure autopsy extension",
        "definition": "Compare mat_mean/Layer-1 boundary categories to explain rescued anomalies, lost anomalies, introduced FPs, and reintroduced FPs.",
        "oracle_category_counts": cats,
        "diagnostic_effects": effect_fields,
        "interpretation_handle": "If fragmentation/reliability fields separate lost anomalies from removed FPs, the response matrix contains target-recovery evidence beyond scalar mat_mean.",
    }


def build_phase_b(aggregate):
    """Fragmentation decomposition probe."""
    leaderboard = aggregate.get("layer0_reference_leaderboard") or []
    fixed = [x for x in leaderboard if x.get("family") in {
        "affine_cmf", "affine_mix", "sigmoid_fixed_gate", "reliability_blend",
        "readout", "baseline",
    }]
    best_auc = _best_by(fixed, "auc_mean")
    best_ap = _best_by(fixed, "ap_mean")
    mat_auc = _mean(aggregate.get("mat_mean_auc"))
    mat_ap = _mean(aggregate.get("mat_mean_ap"))
    return {
        "phase": "B",
        "name": "Fragmentation decomposition probe",
        "definition": "Test fixed-formula targets that decompose fragmentation into consensus, mixture support, reliability, and fragmentation penalty components.",
        "best_fixed_by_auc": best_auc,
        "best_fixed_by_ap": best_ap,
        "mat_mean_auc": mat_auc,
        "mat_mean_ap": mat_ap,
        "fixed_formula_count": len(fixed),
        "gate_reading": {
            "ap_gate": None if not best_ap or mat_ap is None else bool(best_ap.get("ap_mean") is not None and best_ap.get("ap_mean") >= mat_ap),
            "auc_gate": None if not best_auc or mat_auc is None else bool(best_auc.get("auc_mean") is not None and best_auc.get("auc_mean") >= mat_auc),
            "non_monotone_hint": None if not best_auc else bool((best_auc.get("spearman_vs_mat_mean_mean") or 1.0) < 0.98),
        },
    }


def build_phase_c(aggregate):
    """Reference relation reliability probe."""
    tradeoff = aggregate.get("tradeoff", {})
    rel_like = []
    for name, t in tradeoff.items():
        if any(tok in name for tok in ["reliability", "cmf", "mix", "gate"]):
            rel_like.append({
                "strategy": name,
                "tradeoff": t,
                "meta": aggregate.get("strategy_meta", {}).get(name, {}),
            })
    # Compact top strategies by low FP reintroduction among candidates with usable AUC/AP.
    compact = []
    for item in rel_like:
        t = item["tradeoff"]
        compact.append({
            "strategy": item["strategy"],
            "family": item["meta"].get("family"),
            "recovered_lost_anomalies_mean": _mean(t.get("recovered_lost_anomalies")),
            "reintroduced_removed_false_positives_mean": _mean(t.get("reintroduced_removed_false_positives")),
            "spearman_vs_mat_mean_mean": _mean(t.get("spearman_vs_mat_mean")),
            "topk_overlap_with_mat_mean_mean": _mean(t.get("topk_overlap_with_mat_mean")),
        })
    compact = sorted(
        compact,
        key=lambda x: (
            -1e9 if x["reintroduced_removed_false_positives_mean"] is None else -x["reintroduced_removed_false_positives_mean"],
            0 if x["recovered_lost_anomalies_mean"] is None else x["recovered_lost_anomalies_mean"],
        ),
        reverse=True,
    )[:20]
    return {
        "phase": "C",
        "name": "Reference relation reliability probe",
        "definition": "Audit whether reliable reference relations are stable, non-hub-dominated, and not just mat_mean/margin/degree/rejection shortcuts.",
        "top_reliability_like_strategies": compact,
        "layer1_fit_diagnostics": aggregate.get("layer1_fit_diagnostics", {}),
        "shortcut_boundary": "Spearman/top-K overlap/correlation values are diagnostic-only; high mat_mean correlation weakens target-readiness even when AUC/AP improves.",
    }


def build_phase_d(aggregate):
    """Trainable target readiness."""
    continuation = aggregate.get("continuation_gate", {})
    decision = aggregate.get("decision")
    best_l1 = continuation.get("best_layer1")
    mat_auc = _mean(aggregate.get("mat_mean_auc"))
    mat_ap = _mean(aggregate.get("mat_mean_ap"))
    ready = False
    if best_l1:
        pass_ap = best_l1.get("ap_mean") is not None and mat_ap is not None and best_l1["ap_mean"] >= mat_ap + 0.005
        pass_auc = best_l1.get("auc_mean") is not None and mat_auc is not None and best_l1["auc_mean"] >= mat_auc + 0.003
        pass_rho = best_l1.get("spearman_vs_mat_mean_mean") is None or best_l1.get("spearman_vs_mat_mean_mean") < 0.95
        pass_shortcut = best_l1.get("abs_shortcut_corr_mean") is None or best_l1.get("abs_shortcut_corr_mean") < 0.20
        ready = bool((pass_ap or pass_auc) and pass_rho and pass_shortcut)
    if ready:
        readiness = "READY_FOR_SHALLOW_TARGET_LEARNING"
    elif best_l1:
        readiness = "FIXED_FORMULA_OR_DIAGNOSTIC_ONLY_REVIEW_REQUIRED"
    else:
        readiness = "REFERENCE_CONSTRUCTOR_BACKOFF"
    return {
        "phase": "D",
        "name": "Trainable target readiness",
        "definition": "Decide whether any fixed or shallow label-free target is reliable enough to become a later trainable objective.",
        "upstream_layer1_decision": decision,
        "readiness_decision": readiness,
        "best_layer1": best_l1,
        "criteria": continuation.get("criteria", []),
        "gate_flags": {k: v for k, v in continuation.items() if k.startswith("pass_")},
        "next_action_if_ready": "Design a shallow label-free objective that learns the validated reliability target, not true anomaly labels.",
        "next_action_if_not_ready": "Back off to reference constructor / fragmentation decomposition rather than increasing head capacity.",
    }


def build_abcd(rows, aggregate):
    return {
        "A": build_phase_a(rows, aggregate),
        "B": build_phase_b(aggregate),
        "C": build_phase_c(aggregate),
        "D": build_phase_d(aggregate),
    }


def worker_loop(worker_id, device, task_queue, result_queue, args_dict):
    """Run independent tasks in a child process bound to one GPU."""
    # Make CUDA visibility explicit but still pass the physical device id through
    # because the lower-level helper logs device identity and constructs cuda:{device}.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    cli_args = argparse.Namespace(**args_dict)
    strategy_meta = {}
    while True:
        task = task_queue.get()
        if task is None:
            return
        variant, seed = task
        t0 = time.time()
        try:
            print(json.dumps({"stage": "task_start", "event": "point", "pid": os.getpid(), "worker": worker_id, "variant": variant, "seed": seed, "device": device}, ensure_ascii=False), flush=True)
            row = run_layer1_one(cli_args, variant, seed, 0, strategy_meta)
            # Preserve physical device assignment in the aggregate row; lower-level
            # helper sees cuda:0 inside CUDA_VISIBLE_DEVICES.
            row["physical_device"] = int(device)
            result_queue.put({"ok": True, "variant": variant, "seed": seed, "device": device, "row": row, "strategy_meta": strategy_meta, "elapsed_sec": time.time() - t0})
            print(json.dumps({"stage": "task_done", "event": "point", "pid": os.getpid(), "worker": worker_id, "variant": variant, "seed": seed, "device": device, "elapsed_sec": time.time() - t0}, ensure_ascii=False), flush=True)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(tb, flush=True)
            result_queue.put({"ok": False, "variant": variant, "seed": seed, "device": device, "error": repr(e), "traceback": tb[-4000:], "elapsed_sec": time.time() - t0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(Path.home() / "DualRefGAD"))
    ap.add_argument("--dataset", default="elliptic")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--devices", default="")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--variants", default="old_exact_080_regime")
    ap.add_argument("--train_rate", type=float, default=0.05)
    ap.add_argument("--val_rate", type=float, default=0.0)
    ap.add_argument("--ln_mode", default="descriptor_similarity")
    ap.add_argument("--normal_k", type=int, default=4)
    ap.add_argument("--anom_k", type=int, default=16)
    ap.add_argument("--encode_batch_size", type=int, default=1024)
    ap.add_argument("--ref_block_size", type=int, default=1024)
    ap.add_argument("--top_k", type=int, default=96)
    ap.add_argument("--n_bins", type=int, default=3)
    ap.add_argument("--pca_dims", default="2,4,8,12,24")
    ap.add_argument("--grid_lambdas", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--grid_mus", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--grid_alphas", default="0.5,1.0,2.0")
    ap.add_argument("--grid_betas", default="0.5,1.0,1.5")
    ap.add_argument("--q_values", default="0.05,0.10")
    ap.add_argument("--qf_values", default="0.10,0.20")
    ap.add_argument("--alpha_anchor_values", default="0.5,1.0")
    ap.add_argument("--alpha_mono_values", default="0.1,0.5")
    ap.add_argument("--lambda_l2_values", default="0.001,0.01")
    ap.add_argument("--train_steps", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress_out", default="")
    args = ap.parse_args()

    seeds = parse_ints(args.seeds)
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    devices = parse_ints(args.devices) if args.devices.strip() else [int(args.device)]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}; available={list(VARIANTS)}")

    start = time.time()
    rows_by_variant = {v: [] for v in variants}
    strategy_meta = {}
    errors = []
    total = len(seeds) * len(variants)
    done = 0

    def snapshot(status="running"):
        partial = {}
        for v in variants:
            rows = rows_by_variant[v]
            partial[v] = summarize_layer1(rows, strategy_meta) if rows else {"n_rows": 0}
        atomic_write_json(args.progress_out, {
            "status": status,
            "probe": "dualrefgad_learning_signal_abcd_probe",
            "done": done,
            "total": total,
            "variants": variants,
            "seeds": seeds,
            "devices": devices,
            "parallel_workers": min(len(devices), total),
            "parallel_seed_discipline": "multiprocessing spawn; one independent process per GPU worker; each task still sets seed/data_split_seed inside Layer-1 helper",
            "partial": partial,
            "errors": errors[-5:],
            "elapsed_sec": time.time() - start,
        })

    snapshot("running")
    print(json.dumps({"stage": "main_snapshot_written", "event": "point", "done": done, "total": total, "progress_out": args.progress_out}, ensure_ascii=False), flush=True)
    tasks = [(variant, seed) for variant in variants for seed in seeds]
    if tasks:
        ctx = mp.get_context("spawn")
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()
        worker_count = min(len(devices), len(tasks))
        print(json.dumps({"stage": "main_prepare_workers", "event": "point", "worker_count": worker_count, "tasks": tasks, "devices": devices}, ensure_ascii=False), flush=True)
        for task in tasks:
            task_queue.put(task)
        for _ in range(worker_count):
            task_queue.put(None)
        args_dict = vars(args).copy()
        workers = []
        for wid, device in enumerate(devices[:worker_count]):
            print(json.dumps({"stage": "main_worker_start", "event": "enter", "worker": wid, "device": int(device)}, ensure_ascii=False), flush=True)
            p = ctx.Process(target=worker_loop, args=(wid, int(device), task_queue, result_queue, args_dict), daemon=False)
            p.start()
            print(json.dumps({"stage": "main_worker_start", "event": "exit", "worker": wid, "device": int(device), "pid": p.pid}, ensure_ascii=False), flush=True)
            workers.append(p)
        while done < total:
            print(json.dumps({"stage": "main_wait_result", "event": "enter", "done": done, "total": total}, ensure_ascii=False), flush=True)
            result = result_queue.get()
            print(json.dumps({"stage": "main_wait_result", "event": "exit", "done": done, "total": total, "ok": result.get("ok"), "variant": result.get("variant"), "seed": result.get("seed"), "device": result.get("device")}, ensure_ascii=False), flush=True)
            if result.get("ok"):
                rows_by_variant[result["variant"]].append(result["row"])
                strategy_meta.update(result.get("strategy_meta") or {})
            else:
                errors.append({
                    "variant": result.get("variant"),
                    "seed": result.get("seed"),
                    "device": result.get("device"),
                    "error": result.get("error"),
                    "traceback": result.get("traceback"),
                })
            done += 1
            snapshot("running")
        for p in workers:
            p.join()
            if p.exitcode not in (0, None):
                errors.append({"worker_exitcode": p.exitcode})

    variant_summaries = []
    for variant in variants:
        rows = sorted(rows_by_variant[variant], key=lambda r: r["seed"])
        aggregate = summarize_layer1(rows, strategy_meta)
        variant_summaries.append({
            "variant": variant,
            "report_codename": VARIANTS[variant]["report_codename"],
            "definition": VARIANTS[variant]["definition"],
            "changes_from_base": VARIANTS[variant]["changes"],
            "aggregate": aggregate,
            "abcd": build_abcd(rows, aggregate),
            "rows": rows,
        })

    payload = {
        "status": "finished" if not errors else "finished_with_errors",
        "probe": "dualrefgad_learning_signal_abcd_probe",
        "protocol": {
            "type": "runner-registered pure probe; ABCD target-readiness diagnostic; no anomaly-label training; multiprocessing GPU task queue",
            "source_report": "https://report.senyao.org/reports/2026/05/26/dualrefgad-learning-signal-recovery-discussion-2026-05-26.html",
            "label_boundary": "labels diagnostic-only for AUC/AP and oracle top-K boundary categories",
            "reuse_boundary": "reuses validated C-LEG3 Layer-1 response-matrix/proxy construction; this wrapper organizes ABCD interpretation",
            "parallel_boundary": "independent spawned process per GPU worker; CUDA_VISIBLE_DEVICES is set per worker; Layer-1 helper still sets seed and data_split_seed per task",
        },
        "dataset": args.dataset,
        "seeds": seeds,
        "variants": variants,
        "devices": devices,
        "config": vars(args),
        "variant_summaries": variant_summaries,
        "errors": errors,
        "elapsed_sec": time.time() - start,
    }
    atomic_write_json(args.out, payload)
    snapshot(payload["status"])
    print(json.dumps({"stage": "abcd_probe_done", "status": payload["status"], "done": done, "total": total, "out": args.out}, ensure_ascii=False), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
