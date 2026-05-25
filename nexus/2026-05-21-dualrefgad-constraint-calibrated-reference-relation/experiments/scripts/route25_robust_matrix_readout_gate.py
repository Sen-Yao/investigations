#!/usr/bin/env python3
"""C-LEG3 robust matrix readout gate.

No training. Frozen C-LEG3 reference regime. This probe tests a set of
predefined matrix readout functions against centroid margin and mat_mean.
Labels are diagnostic-only for AUC/AP and top-K autopsy; no score uses labels.
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
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from route25_leg3_response_matrix_decomposition_probe import (  # noqa: E402
    BASE_DEFAULTS,
    VARIANTS,
    parse_ints,
    safe_corr,
    safe_spearman,
)
from route25_matrix_autoencoder_probe import (  # noqa: E402
    NormalModel,
    build_descriptor,
    build_tokens,
    encode_tokens_batched,
    metric_block,
    response_matrix_from_embeddings,
    select_refs,
    set_seed,
)


def atomic_write_json(path, payload):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tid = threading.get_ident()
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{tid}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    x = np.asarray(vals, dtype=np.float64)
    return {"mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()), "max": float(x.max())}


def safe_auc_ap(labels, score, idx):
    idx = np.asarray(idx, dtype=np.int64)
    y = np.asarray(labels).reshape(-1).astype(int)[idx]
    s = np.asarray(score, dtype=np.float64)[idx]
    if len(np.unique(y)) < 2:
        return None, None
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def top_set(idx_test, score, k):
    idx_test = np.asarray(idx_test, dtype=np.int64)
    s = np.asarray(score, dtype=np.float64)[idx_test]
    order = np.argsort(-s)[:k]
    return set(map(int, idx_test[order]))


def summarize_group(nodes, labels, scores, mat, normal_refs, anom_refs, meta):
    nodes = [int(n) for n in nodes]
    if not nodes:
        return {"count": 0}
    arr = np.asarray(nodes, dtype=np.int64)
    mats = mat[arr]
    flat = mats.reshape(len(arr), -1)
    row_mean = mats.mean(axis=2)
    col_mean = mats.mean(axis=1)
    normal_ref_labels = labels[normal_refs[arr]]
    anom_ref_labels = labels[anom_refs[arr]]
    out = {
        "count": int(len(arr)),
        "anom_rate": float(np.mean(labels[arr] == 1)),
        "degree": mean_std(meta["degree"][arr]),
        "rejection": mean_std(meta["rejection"][arr]),
        "residual_norm": mean_std(meta["residual_norm"][arr]),
        "normal_ref_anom_ratio": mean_std(np.mean(normal_ref_labels == 1, axis=1)),
        "anom_ref_anom_ratio": mean_std(np.mean(anom_ref_labels == 1, axis=1)),
        "mat_mean": mean_std(scores["mat_mean"][arr]),
        "margin": mean_std(scores["margin"][arr]),
        "mat_std": mean_std(flat.std(axis=1)),
        "mat_q50": mean_std(np.quantile(flat, 0.50, axis=1)),
        "mat_q75": mean_std(np.quantile(flat, 0.75, axis=1)),
        "mat_q90": mean_std(np.quantile(flat, 0.90, axis=1)),
        "row_mean_max": mean_std(row_mean.max(axis=1)),
        "row_mean_range": mean_std(row_mean.max(axis=1) - row_mean.min(axis=1)),
        "col_mean_max": mean_std(col_mean.max(axis=1)),
        "col_mean_range": mean_std(col_mean.max(axis=1) - col_mean.min(axis=1)),
    }
    for name, score in scores.items():
        if name in ("margin", "mat_mean") or name.startswith("robust") or name.startswith("top") or name.startswith("q") or name.startswith("row") or name.startswith("col") or name.startswith("winsor") or name.startswith("trim"):
            out[name] = mean_std(score[arr])
    return out


def build_readout_scores(mat, margin, normal_idx):
    """Predefined label-free readout candidates over a normal_k x anom_k response matrix.

    All calibration quantities use only known-normal training nodes where needed.
    Diagnostic labels are not used to define any score.
    """
    X = mat.reshape(mat.shape[0], -1).astype(np.float64)
    row_mean = mat.mean(axis=2)
    col_mean = mat.mean(axis=1)
    q10 = np.quantile(X, 0.10, axis=1)
    q25 = np.quantile(X, 0.25, axis=1)
    q50 = np.quantile(X, 0.50, axis=1)
    q75 = np.quantile(X, 0.75, axis=1)
    q90 = np.quantile(X, 0.90, axis=1)
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    sorted_x = np.sort(X, axis=1)
    trim10 = sorted_x[:, 6:-6].mean(axis=1) if X.shape[1] > 12 else mean
    winsor10 = np.clip(X, q10[:, None], q90[:, None]).mean(axis=1)
    top8_mean = sorted_x[:, -8:].mean(axis=1)
    top16_mean = sorted_x[:, -16:].mean(axis=1)
    row_top1 = row_mean.max(axis=1)
    row_top2 = np.sort(row_mean, axis=1)[:, -2:].mean(axis=1) if row_mean.shape[1] >= 2 else row_top1
    col_top4 = np.sort(col_mean, axis=1)[:, -4:].mean(axis=1) if col_mean.shape[1] >= 4 else col_mean.max(axis=1)
    col_top8 = np.sort(col_mean, axis=1)[:, -8:].mean(axis=1) if col_mean.shape[1] >= 8 else col_mean.max(axis=1)

    # Normal-only dispersion calibration. High dispersion alone is suspicious, but
    # Step-2 showed that real heterogeneous anomalies can also be high-dispersion.
    # Therefore penalties are deliberately soft and paired with upper-tail support.
    n_std = std[np.asarray(normal_idx, dtype=np.int64)]
    med = float(np.median(n_std))
    iqr = float(np.quantile(n_std, 0.75) - np.quantile(n_std, 0.25) + 1e-6)
    disp_z = np.maximum(0.0, (std - med) / iqr)
    disp_penalty = np.tanh(disp_z)

    # Robust gate family: start from mat_mean; add local strong-support evidence;
    # softly subtract global heterogeneity when no enough upper-tail support exists.
    support_gap = np.maximum(0.0, q75 - mean)
    strong_gap = np.maximum(0.0, q90 - mean)
    row_gap = np.maximum(0.0, row_top2 - mean)
    col_gap = np.maximum(0.0, col_top4 - mean)

    return {
        "margin": margin.astype(np.float64),
        "mat_mean": mean,
        "trim10_mean": trim10,
        "winsor10_mean": winsor10,
        "q50_median": q50,
        "q75": q75,
        "q90": q90,
        "top8_mean": top8_mean,
        "top16_mean": top16_mean,
        "row_top1_mean": row_top1,
        "row_top2_mean": row_top2,
        "col_top4_mean": col_top4,
        "col_top8_mean": col_top8,
        "mean_q75_blend": 0.5 * mean + 0.5 * q75,
        "mean_top16_blend": 0.5 * mean + 0.5 * top16_mean,
        "row_col_top_blend": 0.5 * row_top2 + 0.5 * col_top4,
        "robust_gate_q75_soft": mean + 0.65 * support_gap - 0.08 * disp_penalty,
        "robust_gate_q90_soft": mean + 0.45 * strong_gap - 0.08 * disp_penalty,
        "robust_gate_rowcol_soft": mean + 0.35 * row_gap + 0.25 * col_gap - 0.08 * disp_penalty,
        "robust_gate_consensus": mean + 0.25 * support_gap + 0.20 * row_gap + 0.20 * col_gap - 0.10 * disp_penalty,
        "heterogeneity_penalized_mean": mean - 0.10 * disp_penalty,
        "mat_std": std,
    }


def run_one(args, variant, seed, device):
    set_seed(seed)
    cfg = copy.deepcopy(BASE_DEFAULTS)
    cfg.update(VARIANTS[variant]["changes"])
    cfg.update(vars(args))
    cfg["variant"] = variant
    cfg["device"] = int(device)
    cfg["seed"] = int(seed)
    v_args = argparse.Namespace(**cfg)

    root = Path(v_args.project_root).expanduser().resolve()
    sys.path.insert(0, str(root))
    os.chdir(str(root))
    from utils import load_mat, preprocess_features, normalize_adj  # noqa: E402
    from VecGAD import VecGAD  # noqa: E402

    def to_dense_features(dataset, features):
        if dataset in ["Amazon", "tf_finace", "t_finance", "reddit", "elliptic"]:
            features, _ = preprocess_features(features)
            return np.asarray(features, dtype=np.float32)
        return np.asarray(features.todense(), dtype=np.float32)

    device_obj = torch.device(f"cuda:{device}" if torch.cuda.is_available() and int(device) >= 0 else "cpu")
    print(json.dumps({"stage": "seed_start", "variant": variant, "seed": seed, "device": str(device_obj)}, ensure_ascii=False), flush=True)
    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat(v_args.dataset, v_args.train_rate, v_args.val_rate, args=v_args)
    features_np = to_dense_features(v_args.dataset, features)
    labels_np = np.asarray(ano_label).reshape(-1).astype(int)
    normal_idx = np.asarray(normal_for_train_idx, dtype=np.int64)
    idx_test = np.asarray(idx_test, dtype=np.int64)
    assert np.sum(labels_np[normal_idx]) == 0, "Data leakage: normal_for_train_idx contains anomalies"

    z = build_descriptor(v_args.descriptor_mode, features_np, adj, normalize_adj, v_args.hops, v_args.rw_steps)
    nm = NormalModel(v_args.pn_estimator, z, normal_idx, v_args.pca_components)
    residual = nm.residual()
    normal_refs, anom_refs, meta = select_refs(z, residual, normal_idx, nm, features_np, adj, v_args, normalize_adj)
    token_tensor = build_tokens(features_np, normal_refs, anom_refs)
    model = VecGAD(features_np.shape[1], v_args.embedding_dim, "prelu", v_args).to(device_obj)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    with torch.no_grad():
        emb = encode_tokens_batched(model, token_tensor, device_obj, v_args.encode_batch_size)
    del token_tensor, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mat, margin = response_matrix_from_embeddings(emb, normal_refs, anom_refs)
    scores = build_readout_scores(mat, margin, normal_idx)

    metrics = metric_block(labels_np, idx_test, scores, base_name="mat_mean")
    n_anom_test = int(np.sum(labels_np[idx_test] == 1))
    k = max(1, n_anom_test)
    mat_top = top_set(idx_test, scores["mat_mean"], k)

    readout_autopsy = {}
    for name, score in scores.items():
        if name == "mat_std":
            continue
        top = top_set(idx_test, score, k)
        rescued_vs_mat = sorted([n for n in (top - mat_top) if labels_np[n] == 1], key=lambda n: -score[n])
        introduced_fp_vs_mat = sorted([n for n in (top - mat_top) if labels_np[n] == 0], key=lambda n: -score[n])
        lost_anom_vs_mat = sorted([n for n in (mat_top - top) if labels_np[n] == 1], key=lambda n: scores["mat_mean"][n] - score[n])
        removed_fp_vs_mat = sorted([n for n in (mat_top - top) if labels_np[n] == 0], key=lambda n: scores["mat_mean"][n] - score[n])
        readout_autopsy[name] = {
            "topk_counts_vs_mat_mean": {
                "rescued_anomalies": int(len(rescued_vs_mat)),
                "introduced_false_positives": int(len(introduced_fp_vs_mat)),
                "lost_anomalies": int(len(lost_anom_vs_mat)),
                "removed_false_positives": int(len(removed_fp_vs_mat)),
            },
            "net_topk_tp_gain_vs_mat_mean": int(len(rescued_vs_mat) - len(lost_anom_vs_mat)),
            "net_topk_fp_change_vs_mat_mean": int(len(introduced_fp_vs_mat) - len(removed_fp_vs_mat)),
            "rescued_summary": summarize_group(rescued_vs_mat, labels_np, scores, mat, normal_refs, anom_refs, meta),
            "introduced_fp_summary": summarize_group(introduced_fp_vs_mat, labels_np, scores, mat, normal_refs, anom_refs, meta),
            "lost_anom_summary": summarize_group(lost_anom_vs_mat, labels_np, scores, mat, normal_refs, anom_refs, meta),
            "removed_fp_summary": summarize_group(removed_fp_vs_mat, labels_np, scores, mat, normal_refs, anom_refs, meta),
        }

    metric_auc = {k: v.get("auc") for k, v in metrics.items() if isinstance(v, dict) and v.get("auc") is not None}
    best_auc_name = max(metric_auc, key=metric_auc.get)
    row = {
        "variant": variant,
        "report_codename": VARIANTS[variant]["report_codename"],
        "seed": int(seed),
        "device": int(device),
        "topk_protocol": "top K within test nodes where K equals number of test anomalies; labels diagnostic-only",
        "k_anomaly_count": int(k),
        "num_test": int(len(idx_test)),
        "test_anom_rate": float(np.mean(labels_np[idx_test] == 1)),
        "metrics": metrics,
        "best_auc_name": best_auc_name,
        "best_auc": float(metric_auc[best_auc_name]),
        "score_relationship": {
            name: {
                "spearman_with_mat_mean": safe_spearman(scores[name][idx_test], scores["mat_mean"][idx_test]),
                "spearman_with_margin": safe_spearman(scores[name][idx_test], scores["margin"][idx_test]),
                "pearson_like_with_mat_mean": safe_corr(scores[name][idx_test], scores["mat_mean"][idx_test]),
            }
            for name in scores if name != "mat_std"
        },
        "readout_autopsy_vs_mat_mean": readout_autopsy,
        "reference_global": {
            "normal_ref_anom_ratio_diagnostic": float(np.mean(labels_np[normal_refs] == 1)),
            "anom_ref_anom_ratio_diagnostic": float(np.mean(labels_np[anom_refs] == 1)),
        },
    }
    print(json.dumps({"stage": "seed_done", "variant": variant, "seed": seed, "best": best_auc_name, "best_auc": row["best_auc"], "mat_auc": metrics["mat_mean"]["auc"]}, ensure_ascii=False), flush=True)
    return row


def summarize(rows):
    if not rows:
        return {"n_rows": 0}
    score_names = sorted(rows[0]["metrics"].keys())
    out = {"n_rows": len(rows), "score_summary": {}, "best_auc_votes": {}}
    for name in score_names:
        vals_auc = [r["metrics"].get(name, {}).get("auc") for r in rows if isinstance(r["metrics"].get(name), dict)]
        vals_ap = [r["metrics"].get(name, {}).get("ap") for r in rows if isinstance(r["metrics"].get(name), dict)]
        vals_spear_mat = [r["score_relationship"].get(name, {}).get("spearman_with_mat_mean") for r in rows]
        out["score_summary"][name] = {
            "auc": mean_std(vals_auc),
            "ap": mean_std(vals_ap),
            "spearman_with_mat_mean": mean_std(vals_spear_mat),
        }
    for r in rows:
        out["best_auc_votes"][r["best_auc_name"]] = out["best_auc_votes"].get(r["best_auc_name"], 0) + 1
    mat_auc = out["score_summary"]["mat_mean"]["auc"]["mean"]
    candidates = []
    for name, block in out["score_summary"].items():
        auc = block["auc"]
        if name not in ("mat_mean", "margin", "mat_std") and auc:
            candidates.append((name, auc["mean"], auc["std"]))
    candidates.sort(key=lambda x: (-x[1], x[2]))
    out["leaderboard"] = [{"score": n, "auc_mean": a, "auc_std": s, "delta_vs_mat_mean": a - mat_auc} for n, a, s in candidates]
    best = out["leaderboard"][0] if out["leaderboard"] else None
    out["mat_mean_auc"] = out["score_summary"]["mat_mean"]["auc"]
    out["margin_auc"] = out["score_summary"]["margin"]["auc"]
    out["best_nonbaseline"] = best
    if best and best["delta_vs_mat_mean"] > 0.002:
        out["decision"] = "ROBUST_READOUT_CANDIDATE_BEATS_MAT_MEAN_GATE"
    elif best and best["delta_vs_mat_mean"] > -0.001:
        out["decision"] = "ROBUST_READOUT_TIES_MAT_MEAN_NEEDS_TOPK_AUTOPSY"
    else:
        out["decision"] = "MAT_MEAN_REMAINS_STRONGEST_SIMPLE_GATE"
    # Aggregate top-K autopsy for the top few candidates plus mat_mean/margin.
    selected = ["margin", "mat_mean"] + [x["score"] for x in out["leaderboard"][:8]]
    out["topk_autopsy_vs_mat_mean"] = {}
    for name in selected:
        if name not in rows[0]["readout_autopsy_vs_mat_mean"]:
            continue
        out["topk_autopsy_vs_mat_mean"][name] = {
            "net_topk_tp_gain_vs_mat_mean": mean_std([r["readout_autopsy_vs_mat_mean"][name]["net_topk_tp_gain_vs_mat_mean"] for r in rows]),
            "net_topk_fp_change_vs_mat_mean": mean_std([r["readout_autopsy_vs_mat_mean"][name]["net_topk_fp_change_vs_mat_mean"] for r in rows]),
            "counts": {
                c: mean_std([r["readout_autopsy_vs_mat_mean"][name]["topk_counts_vs_mat_mean"][c] for r in rows])
                for c in ["rescued_anomalies", "introduced_false_positives", "lost_anomalies", "removed_false_positives"]
            }
        }
    return out


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
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress_out", default="")
    args = ap.parse_args()

    seeds = parse_ints(args.seeds)
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}; available={list(VARIANTS)}")
    devices = parse_ints(args.devices) if args.devices.strip() else [int(args.device)]
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
        partial = []
        for v in variants:
            rows = rows_by_variant[v]
            partial.append({"variant": v, "done": len(rows), "summary": summarize(rows) if rows else None})
        atomic_write_json(args.progress_out, {
            "status": status,
            "probe": "route25_robust_matrix_readout_gate",
            "done": done,
            "total": total,
            "devices": devices,
            "current": current,
            "partial": partial,
            "errors": errors[-5:],
            "time_sec": time.time() - start,
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
                row = run_one(args, variant, seed, device)
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

    results = []
    for v in variants:
        rows = sorted(rows_by_variant[v], key=lambda r: r["seed"])
        results.append({
            "variant": v,
            "report_codename": VARIANTS[v]["report_codename"],
            "definition": VARIANTS[v]["definition"],
            "rows": rows,
            "aggregate": summarize(rows),
        })
    payload = {
        "status": "finished" if not errors else "finished_with_errors",
        "probe": "route25_robust_matrix_readout_gate",
        "protocol": "Frozen encoder; C-LEG3 reference regime fixed; no training; predefined label-free robust matrix readout candidates; labels diagnostic-only for AUC/AP and top-K autopsy.",
        "dataset": args.dataset,
        "seeds": seeds,
        "devices": devices,
        "variants_requested": variants,
        "config": vars(args),
        "results": results,
        "errors": errors,
        "time_sec": time.time() - start,
    }
    atomic_write_json(args.out, payload)
    snapshot(payload["status"])
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
