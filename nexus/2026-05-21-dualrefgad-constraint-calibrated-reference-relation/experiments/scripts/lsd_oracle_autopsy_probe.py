#!/usr/bin/env python3
"""DualRefGAD Learning Signal Discovery oracle-autopsy pure probe.

Runner-registered bundled multi-seed diagnostic. This script freezes the C-LEG3 /
old_exact response regime and uses anomaly labels only for report-only oracle
autopsy: labels never enter loss, early stopping, checkpoint selection, or a
method claim. The goal is to reverse-engineer label-free candidate learning
signals from the oracle boundary between margin and response-matrix summaries.
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
from route25_mat_mean_margin_failure_autopsy import (  # noqa: E402
    mean_std,
    parse_ints,
    run_one as autopsy_run_one,
    safe_auc_ap,
    safe_spearman,
)
from route25_leg3_response_matrix_decomposition_probe import VARIANTS  # noqa: E402


def atomic_write_json(path, payload):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def metric_delta(m, a, b):
    av = m.get(a, {}) or {}
    bv = m.get(b, {}) or {}
    return {
        "auc_delta": None if av.get("auc") is None or bv.get("auc") is None else float(av["auc"] - bv["auc"]),
        "ap_delta": None if av.get("ap") is None or bv.get("ap") is None else float(av["ap"] - bv["ap"]),
    }


def flatten_category(rows, cat, field):
    vals = []
    for r in rows:
        d = ((r.get("category_summary") or {}).get(cat) or {}).get(field)
        if isinstance(d, dict) and d.get("mean") is not None:
            vals.append(d["mean"])
        elif isinstance(d, (int, float)):
            vals.append(d)
    return mean_std(vals)


def candidate_family_summary(rows):
    cats = {
        "mat_rescues_true_positive": "rescued_anomalies_mat_only_true_positive",
        "mat_introduces_false_positive": "introduced_false_positives_mat_only_normal",
        "margin_loses_true_positive": "lost_anomalies_margin_only_true_positive",
        "margin_removes_false_positive": "removed_false_positives_margin_only_normal",
    }
    fields = [
        "delta_mat_minus_margin", "row_mean_range", "col_mean_range", "mat_std",
        "normal_ref_anom_ratio", "anom_ref_anom_ratio", "degree", "rejection", "residual_norm",
        "attr_anom_rate", "struct_anom_rate",
    ]
    oracle = {}
    for out_name, cat in cats.items():
        oracle[out_name] = {f: flatten_category(rows, cat, f) for f in fields}
        oracle[out_name]["count"] = mean_std([r.get("topk_counts", {}).get(cat, 0) for r in rows])

    score_relationship = {
        "spearman_mat_mean_margin": mean_std([r["score_relationship"].get("spearman_mat_mean_margin") for r in rows]),
        "spearman_delta_margin": mean_std([r["score_relationship"].get("spearman_delta_margin") for r in rows]),
        "spearman_delta_degree": mean_std([r["score_relationship"].get("spearman_delta_degree") for r in rows]),
        "spearman_delta_rejection": mean_std([r["score_relationship"].get("spearman_delta_rejection") for r in rows]),
        "spearman_delta_residual_norm": mean_std([r["score_relationship"].get("spearman_delta_residual_norm") for r in rows]),
    }

    # Translate oracle categories into label-free proxy candidates. These are not
    # losses yet; they are candidate signals that must be label-free-ized later.
    candidates = {
        "pair_contribution": {
            "oracle_evidence": "mat_mean beats margin by averaging pairwise normal-ref × anomaly-ref contributions; rescued nodes show positive delta_mat_minus_margin.",
            "label_free_proxy": "weight/contrast individual response-matrix entries using normal-only split stability and entry residuals, not anomaly labels.",
            "support": oracle["mat_rescues_true_positive"].get("delta_mat_minus_margin"),
        },
        "direction_compatibility": {
            "oracle_evidence": "delta-vs-margin/residual correlations indicate whether mat_mean adds a direction rather than only monotone margin scaling.",
            "label_free_proxy": "learn compatibility between node residual direction and reference-pair response direction from normal-only consistency.",
            "support": {
                "delta_margin": score_relationship["spearman_delta_margin"],
                "delta_residual_norm": score_relationship["spearman_delta_residual_norm"],
            },
        },
        "row_column_reliability": {
            "oracle_evidence": "rescued/lost groups differ in row/column range and matrix dispersion.",
            "label_free_proxy": "estimate stable normal-anchor rows and anomaly-reference columns by normal-only bootstrap/dropout reliability.",
            "support": {
                "rescued_row_range": oracle["mat_rescues_true_positive"].get("row_mean_range"),
                "rescued_col_range": oracle["mat_rescues_true_positive"].get("col_mean_range"),
                "introduced_row_range": oracle["mat_introduces_false_positive"].get("row_mean_range"),
                "introduced_col_range": oracle["mat_introduces_false_positive"].get("col_mean_range"),
            },
        },
        "teacher_distillation": {
            "oracle_evidence": "mat_mean is the current oracle-positive teacher baseline; any trainable scorer must distill only label-free matrix structure.",
            "label_free_proxy": "distill from frozen C-LEG3 mat_mean / robust row-min teacher with normal-only calibration, then test against margin and degree confounds.",
            "support": None,
        },
        "anti_shortcut_confound": {
            "oracle_evidence": "degree/rejection/residual summaries on discordant oracle groups expose shortcut risks.",
            "label_free_proxy": "penalize high correlation with degree and simple normal-model rejection when these explain the proposed signal.",
            "support": {
                "delta_degree": score_relationship["spearman_delta_degree"],
                "delta_rejection": score_relationship["spearman_delta_rejection"],
            },
        },
    }
    return oracle, score_relationship, candidates


def summarize_variant(rows, variant):
    metrics = {"margin": [], "mat_mean": []}
    for r in rows:
        for name in metrics:
            metrics[name].append(r["metrics"][name])
    baselines = {
        "margin": {
            "auc": mean_std([m.get("auc") for m in metrics["margin"]]),
            "ap": mean_std([m.get("ap") for m in metrics["margin"]]),
            "family": "centroid_margin",
        },
        "mat_mean": {
            "auc": mean_std([m.get("auc") for m in metrics["mat_mean"]]),
            "ap": mean_std([m.get("ap") for m in metrics["mat_mean"]]),
            "family": "response_matrix_scalar_teacher",
        },
        "mat_minus_margin": {
            "auc_delta": mean_std([metric_delta(r["metrics"], "mat_mean", "margin")["auc_delta"] for r in rows]),
            "ap_delta": mean_std([metric_delta(r["metrics"], "mat_mean", "margin")["ap_delta"] for r in rows]),
        },
    }
    oracle, score_relationship, candidates = candidate_family_summary(rows)
    gates = {
        "label_boundary": "PASS: anomaly labels used only for report-only AUC/AP and oracle autopsy categories; no loss/early-stop/checkpoint/method claim.",
        "teacher_positive_control": "PASS" if (baselines["mat_minus_margin"]["auc_delta"] or {}).get("mean", 0) > 0 else "WEAK_OR_FAIL",
        "not_margin_only": "PASS" if abs((score_relationship["spearman_mat_mean_margin"] or {}).get("mean", 1.0)) < 0.9 else "WEAK_OR_FAIL",
        "anti_degree_shortcut": "PASS" if abs((score_relationship["spearman_delta_degree"] or {}).get("mean", 1.0)) < 0.35 else "REVIEW",
    }
    recommendations = []
    if gates["teacher_positive_control"] == "PASS":
        recommendations.append("Use frozen mat_mean / robust row-summary as teacher baselines before adding trainable losses.")
    if gates["not_margin_only"] == "PASS":
        recommendations.append("Prioritize row/column reliability and pair-contribution proxies because oracle delta is not merely margin rank.")
    recommendations.append("Do not train on anomaly labels; convert only stable normal-side reliability / dropout / split signals into future losses.")
    recommendations.append("Any future trainable method must beat mat_mean and report degree/rejection shortcut correlations.")
    return {
        "variant": variant,
        "report_codename": VARIANTS.get(variant, {}).get("report_codename"),
        "n_seeds": len(rows),
        "baselines": baselines,
        "oracle_autopsy": {"categories": oracle, "score_relationship": score_relationship},
        "candidate_signals": candidates,
        "proxy_candidates": {k: v["label_free_proxy"] for k, v in candidates.items()},
        "gates": gates,
        "recommendations": recommendations,
        "rows": rows,
    }


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
    ap.add_argument("--sample_nodes_per_category", type=int, default=12)
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
        partial = {v: {"done": len(rows_by_variant[v])} for v in variants}
        atomic_write_json(args.progress_out, {
            "status": status,
            "probe": "lsd_oracle_autopsy_probe",
            "done": done,
            "total": total,
            "devices": devices,
            "current": current,
            "partial": partial,
            "errors": errors[-5:],
            "elapsed_sec": time.time() - start,
        })

    def worker(device):
        nonlocal done
        while True:
            try:
                variant, seed = task_q.get_nowait()
            except queue.Empty:
                return
            cur = {"variant": variant, "seed": seed, "device": device}
            try:
                print(json.dumps({"stage": "task_start", **cur}, ensure_ascii=False), flush=True)
                snapshot("running", cur)
                row = autopsy_run_one(args, variant, seed, device)
                with lock:
                    rows_by_variant[variant].append(row)
                    done += 1
                    snapshot("running", cur)
            except Exception as e:
                tb = traceback.format_exc()
                print(tb, flush=True)
                with lock:
                    errors.append({"variant": variant, "seed": seed, "device": device, "error": repr(e), "traceback": tb[-4000:]})
                    done += 1
                    snapshot("running", cur)
            finally:
                task_q.task_done()

    snapshot("running")
    threads = [threading.Thread(target=worker, args=(d,), daemon=False) for d in devices]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    variant_summaries = [summarize_variant(sorted(rows_by_variant[v], key=lambda r: r["seed"]), v) for v in variants]
    primary = variant_summaries[0] if variant_summaries else {}
    payload = {
        "status": "finished" if not errors else "finished_with_errors",
        "protocol": {
            "name": "lsd_oracle_autopsy_probe",
            "design_report_url": "https://report.senyao.org/reports/2026/05/25/dualrefgad-learning-signal-discovery-probe-design-2026-05-25.html",
            "type": "runner-registered bundled multi-seed pure probe; no training; frozen C-LEG3/old_exact response regime",
            "label_boundary": "anomaly labels are report-only/oracle-autopsy only; never final loss, early stopping, checkpoint selection, or method claim",
            "candidate_signal_families": ["pair_contribution", "direction_compatibility", "row_column_reliability", "teacher_distillation", "anti_shortcut_confound"],
        },
        "seeds": seeds,
        "baselines": primary.get("baselines", {}),
        "oracle_autopsy": primary.get("oracle_autopsy", {}),
        "candidate_signals": primary.get("candidate_signals", {}),
        "proxy_candidates": primary.get("proxy_candidates", {}),
        "gates": primary.get("gates", {}),
        "recommendations": primary.get("recommendations", []),
        "dataset": args.dataset,
        "variants": variants,
        "devices": devices,
        "variant_summaries": variant_summaries,
        "errors": errors,
        "elapsed_sec": time.time() - start,
    }
    atomic_write_json(args.out, payload)
    snapshot(payload["status"])
    print(json.dumps({"stage": "probe_done", "status": payload["status"], "done": done, "total": total, "out": args.out}, ensure_ascii=False), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
