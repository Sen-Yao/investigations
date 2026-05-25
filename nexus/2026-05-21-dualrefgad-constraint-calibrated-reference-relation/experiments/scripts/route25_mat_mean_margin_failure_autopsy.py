#!/usr/bin/env python3
"""C-LEG3 mat_mean vs margin failure autopsy.

No training. Frozen C-LEG3 reference regime. Labels are diagnostic-only for
AUC/AP and for explaining which top-K mistakes are rescued or introduced.
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
    build_decomposition_arrays,
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
    # Multi-threaded probe workers may write progress concurrently. Use a
    # thread-unique temp path so one writer cannot replace/delete another
    # writer's temp file before rename.
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


def summarize_node_group(nodes, labels, str_labels, attr_labels, margin, mat_mean, mat, normal_refs, anom_refs, meta):
    nodes = [int(n) for n in nodes]
    if not nodes:
        return {"count": 0}
    arr = np.asarray(nodes, dtype=np.int64)
    mats = mat[arr]
    row_mean = mats.mean(axis=2)
    col_mean = mats.mean(axis=1)
    normal_ref_labels = labels[normal_refs[arr]]
    anom_ref_labels = labels[anom_refs[arr]]
    # repeated references inside each node's reference list
    def dup_rate(refs):
        vals = []
        for r in refs[arr]:
            vals.append(1.0 - len(set(map(int, r))) / max(1, len(r)))
        return float(np.mean(vals))
    out = {
        "count": int(len(nodes)),
        "anom_rate": float(np.mean(labels[arr] == 1)),
        "struct_anom_rate": float(np.mean(str_labels[arr] == 1)) if str_labels is not None else None,
        "attr_anom_rate": float(np.mean(attr_labels[arr] == 1)) if attr_labels is not None else None,
        "margin": mean_std(margin[arr]),
        "mat_mean": mean_std(mat_mean[arr]),
        "delta_mat_minus_margin": mean_std(mat_mean[arr] - margin[arr]),
        "degree": mean_std(meta["degree"][arr]),
        "rejection": mean_std(meta["rejection"][arr]),
        "residual_norm": mean_std(meta["residual_norm"][arr]),
        "normal_ref_anom_ratio": mean_std(np.mean(normal_ref_labels == 1, axis=1)),
        "anom_ref_anom_ratio": mean_std(np.mean(anom_ref_labels == 1, axis=1)),
        "normal_ref_dup_rate": mean_std([dup_rate(normal_refs)]),
        "anom_ref_dup_rate": mean_std([dup_rate(anom_refs)]),
        "mat_std": mean_std(mats.reshape(len(arr), -1).std(axis=1)),
        "mat_q25": mean_std(np.quantile(mats.reshape(len(arr), -1), 0.25, axis=1)),
        "mat_q50": mean_std(np.quantile(mats.reshape(len(arr), -1), 0.50, axis=1)),
        "mat_q75": mean_std(np.quantile(mats.reshape(len(arr), -1), 0.75, axis=1)),
        "row_mean_min": mean_std(row_mean.min(axis=1)),
        "row_mean_max": mean_std(row_mean.max(axis=1)),
        "row_mean_range": mean_std(row_mean.max(axis=1) - row_mean.min(axis=1)),
        "col_mean_min": mean_std(col_mean.min(axis=1)),
        "col_mean_max": mean_std(col_mean.max(axis=1)),
        "col_mean_range": mean_std(col_mean.max(axis=1) - col_mean.min(axis=1)),
    }
    return out


def sample_nodes(nodes, labels, str_labels, attr_labels, margin, mat_mean, mat, normal_refs, anom_refs, meta, limit):
    nodes = list(map(int, nodes))[:limit]
    samples = []
    for n in nodes:
        m = mat[n]
        rm = m.mean(axis=1)
        cm = m.mean(axis=0)
        samples.append({
            "node_index": int(n),
            "label": int(labels[n]),
            "struct_label": int(str_labels[n]) if str_labels is not None else None,
            "attr_label": int(attr_labels[n]) if attr_labels is not None else None,
            "margin": float(margin[n]),
            "mat_mean": float(mat_mean[n]),
            "delta_mat_minus_margin": float(mat_mean[n] - margin[n]),
            "degree": float(meta["degree"][n]),
            "rejection": float(meta["rejection"][n]),
            "residual_norm": float(meta["residual_norm"][n]),
            "normal_ref_anom_ratio": float(np.mean(labels[normal_refs[n]] == 1)),
            "anom_ref_anom_ratio": float(np.mean(labels[anom_refs[n]] == 1)),
            "row_mean": [float(x) for x in rm.tolist()],
            "row_argmin": int(np.argmin(rm)),
            "row_argmax": int(np.argmax(rm)),
            "col_mean_top5": [float(x) for x in sorted(cm.tolist(), reverse=True)[:5]],
            "mat_quantiles": {"q10": float(np.quantile(m, 0.10)), "q25": float(np.quantile(m, 0.25)), "q50": float(np.quantile(m, 0.50)), "q75": float(np.quantile(m, 0.75)), "q90": float(np.quantile(m, 0.90))},
        })
    return samples


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
    str_np = np.asarray(str_ano_label).reshape(-1).astype(int) if str_ano_label is not None else None
    attr_np = np.asarray(attr_ano_label).reshape(-1).astype(int) if attr_ano_label is not None else None
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
    arrays, _ = build_decomposition_arrays(mat, margin, meta, normal_idx, seed, parse_ints(v_args.pca_dims))
    mat_mean = arrays["mat_mean"]
    metrics = metric_block(labels_np, idx_test, {"margin": margin, "mat_mean": mat_mean, "degree": meta["degree"], "rejection": meta["rejection"], "residual_norm": meta["residual_norm"]}, base_name="margin")

    n_anom_test = int(np.sum(labels_np[idx_test] == 1))
    k = max(1, n_anom_test)
    margin_top = top_set(idx_test, margin, k)
    mat_top = top_set(idx_test, mat_mean, k)
    y = labels_np

    # Top-K-at-anomaly-count error autopsy.
    rescued_anomalies = sorted([n for n in (mat_top - margin_top) if y[n] == 1], key=lambda n: -(mat_mean[n] - margin[n]))
    introduced_false_positives = sorted([n for n in (mat_top - margin_top) if y[n] == 0], key=lambda n: -(mat_mean[n] - margin[n]))
    lost_anomalies = sorted([n for n in (margin_top - mat_top) if y[n] == 1], key=lambda n: (mat_mean[n] - margin[n]))
    removed_false_positives = sorted([n for n in (margin_top - mat_top) if y[n] == 0], key=lambda n: (mat_mean[n] - margin[n]))

    categories = {
        "rescued_anomalies_mat_only_true_positive": rescued_anomalies,
        "introduced_false_positives_mat_only_normal": introduced_false_positives,
        "lost_anomalies_margin_only_true_positive": lost_anomalies,
        "removed_false_positives_margin_only_normal": removed_false_positives,
    }
    cat_summary = {name: summarize_node_group(nodes, labels_np, str_np, attr_np, margin, mat_mean, mat, normal_refs, anom_refs, meta) for name, nodes in categories.items()}
    cat_samples = {name: sample_nodes(nodes, labels_np, str_np, attr_np, margin, mat_mean, mat, normal_refs, anom_refs, meta, v_args.sample_nodes_per_category) for name, nodes in categories.items()}

    # Tail-bin view: among discordant membership nodes, what direction dominates?
    discordant = sorted(list((margin_top ^ mat_top)))
    discordant_summary = summarize_node_group(discordant, labels_np, str_np, attr_np, margin, mat_mean, mat, normal_refs, anom_refs, meta)

    row = {
        "variant": variant,
        "report_codename": VARIANTS[variant]["report_codename"],
        "seed": int(seed),
        "device": int(device),
        "topk_protocol": "top K within test nodes where K equals number of test anomalies; labels diagnostic-only for autopsy categories",
        "k_anomaly_count": int(k),
        "num_test": int(len(idx_test)),
        "test_anom_rate": float(np.mean(labels_np[idx_test] == 1)),
        "metrics": metrics,
        "score_relationship": {
            "spearman_mat_mean_margin": safe_spearman(mat_mean[idx_test], margin[idx_test]),
            "spearman_delta_margin": safe_spearman((mat_mean - margin)[idx_test], margin[idx_test]),
            "spearman_delta_degree": safe_spearman((mat_mean - margin)[idx_test], meta["degree"][idx_test]),
            "spearman_delta_rejection": safe_spearman((mat_mean - margin)[idx_test], meta["rejection"][idx_test]),
            "spearman_delta_residual_norm": safe_spearman((mat_mean - margin)[idx_test], meta["residual_norm"][idx_test]),
            "pearson_like_delta_margin": safe_corr((mat_mean - margin)[idx_test], margin[idx_test]),
        },
        "topk_counts": {name: int(len(nodes)) for name, nodes in categories.items()},
        "category_summary": cat_summary,
        "category_samples": cat_samples,
        "discordant_summary": discordant_summary,
        "reference_global": {
            "normal_ref_anom_ratio_diagnostic": float(np.mean(labels_np[normal_refs] == 1)),
            "anom_ref_anom_ratio_diagnostic": float(np.mean(labels_np[anom_refs] == 1)),
        },
    }
    print(json.dumps({"stage": "seed_done", "variant": variant, "seed": seed, "mat_auc": metrics["mat_mean"]["auc"], "margin_auc": metrics["margin"]["auc"], "rescued": len(rescued_anomalies), "removed_fp": len(removed_false_positives)}, ensure_ascii=False), flush=True)
    return row


def summarize(rows):
    cats = [
        "rescued_anomalies_mat_only_true_positive",
        "introduced_false_positives_mat_only_normal",
        "lost_anomalies_margin_only_true_positive",
        "removed_false_positives_margin_only_normal",
    ]
    out = {
        "n_rows": len(rows),
        "mat_mean_auc": mean_std([r["metrics"]["mat_mean"]["auc"] for r in rows]),
        "margin_auc": mean_std([r["metrics"]["margin"]["auc"] for r in rows]),
        "mat_mean_ap": mean_std([r["metrics"]["mat_mean"]["ap"] for r in rows]),
        "margin_ap": mean_std([r["metrics"]["margin"]["ap"] for r in rows]),
        "spearman_mat_mean_margin": mean_std([r["score_relationship"]["spearman_mat_mean_margin"] for r in rows]),
        "spearman_delta_margin": mean_std([r["score_relationship"]["spearman_delta_margin"] for r in rows]),
        "topk_counts": {c: mean_std([r["topk_counts"][c] for r in rows]) for c in cats},
        "category_summary_mean_of_means": {},
    }
    # Aggregate selected summary means per category.
    fields = ["margin", "mat_mean", "delta_mat_minus_margin", "degree", "rejection", "residual_norm", "anom_ref_anom_ratio", "mat_std", "row_mean_range", "col_mean_range", "attr_anom_rate", "struct_anom_rate"]
    for c in cats:
        out["category_summary_mean_of_means"][c] = {}
        for f in fields:
            vals = []
            for r in rows:
                d = r["category_summary"][c].get(f)
                if isinstance(d, dict) and d.get("mean") is not None:
                    vals.append(d["mean"])
                elif isinstance(d, (int, float)) and d is not None:
                    vals.append(d)
            out["category_summary_mean_of_means"][c][f] = mean_std(vals)
    # Plain decision.
    rescued = out["topk_counts"]["rescued_anomalies_mat_only_true_positive"]["mean"] if out["topk_counts"]["rescued_anomalies_mat_only_true_positive"] else 0
    removed = out["topk_counts"]["removed_false_positives_margin_only_normal"]["mean"] if out["topk_counts"]["removed_false_positives_margin_only_normal"] else 0
    introduced = out["topk_counts"]["introduced_false_positives_mat_only_normal"]["mean"] if out["topk_counts"]["introduced_false_positives_mat_only_normal"] else 0
    if rescued > 0 and removed > 0 and introduced <= removed:
        out["decision"] = "MAT_MEAN_REORDERS_BOUNDARY_BY_RESCUING_ANOMALIES_AND_REMOVING_MARGIN_FALSE_POSITIVES"
    elif introduced > removed:
        out["decision"] = "MAT_MEAN_GAIN_COMES_WITH_MORE_FALSE_POSITIVE_BOUNDARY_COST"
    else:
        out["decision"] = "MAT_MEAN_MARGIN_DIFFERENCE_NEEDS_CASE_LEVEL_REVIEW"
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
    ap.add_argument("--sample_nodes_per_category", type=int, default=12)
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
            "probe": "route25_mat_mean_margin_failure_autopsy",
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
        "probe": "route25_mat_mean_margin_failure_autopsy",
        "protocol": "Frozen encoder; C-LEG3 reference regime fixed; no training; labels diagnostic-only for AUC/AP and top-K failure autopsy. Top-K uses K = number of test anomalies per seed.",
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
