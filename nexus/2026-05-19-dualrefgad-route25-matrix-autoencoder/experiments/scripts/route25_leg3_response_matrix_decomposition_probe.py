#!/usr/bin/env python3
"""LEG3 response-matrix decomposition probe for DualRefGAD Route2.5.

Purpose:
- Fix the old_exact_080_regime / C-LEG3 reference regime that previously
  recovers mat_mean≈0.80 AUC.
- Decompose the full response matrix M(v) into scalar, row, column, entry,
  normal-manifold residual, and complementarity diagnostics.
- No training. Frozen VecGAD encoder. Labels are diagnostic-only for AUC/AP.
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
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from route25_matrix_autoencoder_probe import (  # noqa: E402
    NormalModel,
    build_descriptor,
    build_tokens,
    encode_tokens_batched,
    jaccard_top,
    metric_block,
    response_matrix_from_embeddings,
    safe_spearman,
    select_refs,
    set_seed,
)

VARIANTS = {
    "old_exact_080_regime": {
        "report_codename": "C-LEG3",
        "definition": "Legacy GT-3 reproduction: hop_attr descriptor, PCA-residual normal model, labeled-normal normal references, normal_soft_or anomaly scoring, descriptor-similarity anomaly references, GT_num_layers=3, full anomaly candidate pool.",
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
    "old_refs_gt1_low_layer": {
        "report_codename": "C-LEG1-control",
        "definition": "Same legacy reference construction as C-LEG3, but GT_num_layers=1; used only as a depth-control if requested.",
        "changes": {
            "descriptor_mode": "hop_attr",
            "pn_estimator": "pca_residual",
            "gn_mode": "label_gate",
            "ga_mode": "normal_soft_or",
            "la_mode": "descriptor_similarity",
            "GT_num_layers": 1,
            "use_approx_anom_refs": False,
        },
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
    "use_approx_anom_refs": False,
    "anom_approx_k": 1000,
    "top_k": 96,
    "n_bins": 3,
    "pca_dims": "2,4,8,12,24",
}


def parse_ints(s):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def atomic_write_json(path, payload):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def safe_auc_ap(labels, score, idx):
    idx = np.asarray(idx, dtype=np.int64)
    y = np.asarray(labels).reshape(-1).astype(int)[idx]
    s = np.asarray(score, dtype=np.float64)[idx]
    if len(np.unique(y)) < 2:
        return None, None
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def safe_corr(a, b):
    try:
        v = spearmanr(np.asarray(a), np.asarray(b)).correlation
        return float(0.0 if np.isnan(v) else v)
    except Exception:
        return 0.0


def mean_std(vals):
    vals = np.asarray([v for v in vals if v is not None], dtype=np.float64)
    if vals.size == 0:
        return None
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}


def bucket_edges(x, n_bins):
    x = np.asarray(x, dtype=np.float64)
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    return edges


def stratified_auc(labels, score, idx, stratifier, n_bins=3):
    idx = np.asarray(idx, dtype=np.int64)
    labels = np.asarray(labels).reshape(-1).astype(int)
    score = np.asarray(score, dtype=np.float64)
    strat = np.asarray(stratifier, dtype=np.float64)
    edges = bucket_edges(strat[idx], n_bins)
    bins = np.digitize(strat[idx], edges[1:-1], right=True)
    out = []
    for b in range(n_bins):
        sub = idx[bins == b]
        block = {
            "bin": int(b),
            "count": int(len(sub)),
            "range": [float(edges[b]), float(edges[b + 1])],
        }
        if len(sub) and len(np.unique(labels[sub])) == 2:
            block["anom_rate"] = float(np.mean(labels[sub] == 1))
            block["auc"], block["ap"] = safe_auc_ap(labels, score, sub)
        else:
            block["anom_rate"] = float(np.mean(labels[sub] == 1)) if len(sub) else None
            block["auc"], block["ap"] = None, None
        out.append(block)
    return out


def top_overlap(idx, a, b, frac=0.05):
    return jaccard_top(np.asarray(a)[idx], np.asarray(b)[idx], frac)


def build_decomposition_arrays(mat, margin, meta, normal_idx, seed, pca_dims):
    n, kn, ka = mat.shape
    X = mat.reshape(n, -1).astype(np.float32)
    row_mean = mat.mean(axis=2)
    col_mean = mat.mean(axis=1)
    row_std = mat.std(axis=2)
    col_std = mat.std(axis=1)
    entry_mean_normals = X[normal_idx].mean(axis=0, keepdims=True)
    entry_std_normals = X[normal_idx].std(axis=0, keepdims=True) + 1e-6
    Z = (X - entry_mean_normals) / entry_std_normals

    # Normal-only reliability: stable entries have lower normal-side std and stable split means.
    rng = np.random.default_rng(seed + 1009)
    nidx = rng.permutation(np.asarray(normal_idx, dtype=np.int64))
    half = max(1, len(nidx) // 2)
    a, b = nidx[:half], nidx[half:] if len(nidx[half:]) else nidx[:half]
    split_diff = np.abs(X[a].mean(axis=0) - X[b].mean(axis=0))
    rel = 1.0 / (split_diff + entry_std_normals.reshape(-1) + 1e-3)
    rel = np.clip(rel / (np.mean(rel) + 1e-12), 0.05, 20.0).astype(np.float32)
    WZ = Z * rel[None, :]

    arrays = {
        # scalar baselines
        "margin": margin,
        "neg_margin": -margin,
        "mat_mean": mat.mean(axis=(1, 2)),
        "neg_mat_mean": -mat.mean(axis=(1, 2)),
        "mat_std": mat.std(axis=(1, 2)),
        "mat_median": np.median(X, axis=1),
        "mat_q75": np.quantile(X, 0.75, axis=1),
        "mat_q25": np.quantile(X, 0.25, axis=1),
        "mat_iqr": np.quantile(X, 0.75, axis=1) - np.quantile(X, 0.25, axis=1),
        "mat_top5_mean": np.sort(X, axis=1)[:, -5:].mean(axis=1),
        "mat_bottom5_mean": np.sort(X, axis=1)[:, :5].mean(axis=1),
        "neg_mat_bottom5_mean": -np.sort(X, axis=1)[:, :5].mean(axis=1),
        "mat_high08_ratio": (mat > 0.8).mean(axis=(1, 2)),
        "mat_lowneg08_ratio": (mat < -0.8).mean(axis=(1, 2)),
        # row family: does any normal anchor consistently expose anomaly direction?
        "row_mean_max": row_mean.max(axis=1),
        "row_mean_min": row_mean.min(axis=1),
        "row_mean_range": row_mean.max(axis=1) - row_mean.min(axis=1),
        "row_mean_std": row_mean.std(axis=1),
        "row_std_mean": row_std.mean(axis=1),
        "row_std_max": row_std.max(axis=1),
        # column family: does any anomaly reference act as a reliable probe?
        "col_mean_max": col_mean.max(axis=1),
        "col_mean_min": col_mean.min(axis=1),
        "col_mean_range": col_mean.max(axis=1) - col_mean.min(axis=1),
        "col_mean_std": col_mean.std(axis=1),
        "col_std_mean": col_std.mean(axis=1),
        "col_std_max": col_std.max(axis=1),
        # normal-manifold deviation over entry responses; label-free fit on labeled normals only
        "entry_z_l2": np.mean(Z ** 2, axis=1),
        "entry_reliable_z_l2": np.mean((WZ) ** 2, axis=1),
        "entry_reliable_abs": np.mean(np.abs(WZ), axis=1),
        "entry_reliable_mean": np.mean(WZ, axis=1),
        "entry_reliable_negmean": -np.mean(WZ, axis=1),
        # existing reference/proxy diagnostics
        "rejection": meta["rejection"],
        "residual_norm": meta["residual_norm"],
        "degree": meta["degree"],
    }

    for dim in pca_dims:
        dim = int(min(dim, max(1, len(normal_idx) - 1), WZ.shape[1]))
        if dim < 1:
            continue
        try:
            pca = PCA(n_components=dim, svd_solver="randomized", random_state=seed)
            pca.fit(WZ[normal_idx])
            emb = pca.transform(WZ)
            rec = pca.inverse_transform(emb)
            arrays[f"entry_reliable_pca{dim}_resid"] = np.mean((WZ - rec) ** 2, axis=1)
            mu = emb[normal_idx].mean(axis=0, keepdims=True)
            sd = emb[normal_idx].std(axis=0, keepdims=True) + 1e-6
            arrays[f"entry_reliable_pca{dim}_center"] = np.mean(((emb - mu) / sd) ** 2, axis=1)
            try:
                lw = LedoitWolf().fit(emb[normal_idx])
                arrays[f"entry_reliable_pca{dim}_mahal"] = lw.mahalanobis(emb).astype(np.float64)
            except Exception:
                pass
        except Exception:
            pass

    reliability_summary = {
        "entry_count": int(X.shape[1]),
        "mean": float(np.mean(rel)),
        "std": float(np.std(rel)),
        "top10_mean": float(np.mean(np.sort(rel)[-max(1, int(0.1 * len(rel))):])),
        "bottom10_mean": float(np.mean(np.sort(rel)[:max(1, int(0.1 * len(rel)))])),
        "top_entries": [
            {"entry_flat": int(i), "normal_ref_slot": int(i // ka), "anom_ref_slot": int(i % ka), "reliability": float(rel[i])}
            for i in np.argsort(-rel)[: min(12, len(rel))]
        ],
    }
    return arrays, reliability_summary


def family_of(name):
    if name in {"margin", "neg_margin"}:
        return "centroid_margin"
    if name.startswith("row_"):
        return "row_normal_anchor"
    if name.startswith("col_"):
        return "column_anomaly_reference"
    if name.startswith("entry_"):
        return "entry_normal_manifold"
    if name in {"rejection", "residual_norm", "degree"}:
        return "proxy_diagnostic"
    return "scalar_matrix_summary"


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
    arrays, reliability_summary = build_decomposition_arrays(mat, margin, meta, normal_idx, seed, parse_ints(v_args.pca_dims))
    metrics = metric_block(labels_np, idx_test, arrays, base_name="margin")
    ranked = sorted(metrics.items(), key=lambda kv: kv[1]["auc"], reverse=True)
    family_best = {}
    for name, m in metrics.items():
        fam = family_of(name)
        cand = {"name": name, **m}
        if fam not in family_best or cand["auc"] > family_best[fam]["auc"]:
            family_best[fam] = cand

    best_name, best_metric = ranked[0]
    row = {
        "variant": variant,
        "report_codename": VARIANTS[variant]["report_codename"],
        "seed": int(seed),
        "device": int(device),
        "best_metric": {"name": best_name, "family": family_of(best_name), **best_metric},
        "family_best": family_best,
        "top_metrics": [{"name": n, "family": family_of(n), **m} for n, m in ranked[: v_args.top_k]],
        "reference_diagnostics": {
            "anom_ref_anom_ratio_diagnostic": float(np.mean(labels_np[anom_refs] == 1)),
            "normal_ref_anom_ratio_diagnostic": float(np.mean(labels_np[normal_refs] == 1)),
            "test_anom_rate": float(np.mean(labels_np[idx_test] == 1)),
        },
        "complementarity": {
            "best_spearman_with_margin": safe_spearman(arrays[best_name][idx_test], margin[idx_test]),
            "best_top5_jaccard_with_margin": top_overlap(idx_test, arrays[best_name], margin, 0.05),
            "mat_mean_spearman_with_margin": safe_spearman(arrays["mat_mean"][idx_test], margin[idx_test]),
            "entry_reliable_l2_spearman_with_mat_mean": safe_corr(arrays["entry_reliable_z_l2"][idx_test], arrays["mat_mean"][idx_test]),
        },
        "proxy_correlation": {
            "best_with_degree": safe_corr(arrays[best_name][idx_test], arrays["degree"][idx_test]),
            "best_with_rejection": safe_corr(arrays[best_name][idx_test], arrays["rejection"][idx_test]),
            "best_with_residual_norm": safe_corr(arrays[best_name][idx_test], arrays["residual_norm"][idx_test]),
        },
        "stratified_best_auc": {
            "degree": stratified_auc(labels_np, arrays[best_name], idx_test, arrays["degree"], v_args.n_bins),
            "margin": stratified_auc(labels_np, arrays[best_name], idx_test, margin, v_args.n_bins),
            "rejection": stratified_auc(labels_np, arrays[best_name], idx_test, arrays["rejection"], v_args.n_bins),
        },
        "reliability_summary": reliability_summary,
        "counts": {
            "num_nodes": int(len(labels_np)),
            "num_test": int(len(idx_test)),
            "num_labeled_normals": int(len(normal_idx)),
            "matrix_shape": list(mat.shape),
        },
    }
    print(json.dumps({"stage": "seed_done", "variant": variant, "seed": seed, "best": best_name, "auc": best_metric["auc"]}, ensure_ascii=False), flush=True)
    return row


def summarize(rows):
    fam_names = sorted({fam for r in rows for fam in r["family_best"]})
    out = {
        "n_rows": len(rows),
        "best_auc": mean_std([r["best_metric"]["auc"] for r in rows]),
        "winner_counts": {},
        "winner_family_counts": {},
        "reference_anom_ratio": mean_std([r["reference_diagnostics"]["anom_ref_anom_ratio_diagnostic"] for r in rows]),
        "best_spearman_with_margin": mean_std([r["complementarity"]["best_spearman_with_margin"] for r in rows]),
        "best_top5_jaccard_with_margin": mean_std([r["complementarity"]["best_top5_jaccard_with_margin"] for r in rows]),
        "best_with_degree_abs": mean_std([abs(r["proxy_correlation"]["best_with_degree"]) for r in rows]),
        "family_best_auc": {},
    }
    for r in rows:
        n = r["best_metric"]["name"]
        f = r["best_metric"]["family"]
        out["winner_counts"][n] = out["winner_counts"].get(n, 0) + 1
        out["winner_family_counts"][f] = out["winner_family_counts"].get(f, 0) + 1
    for fam in fam_names:
        out["family_best_auc"][fam] = mean_std([r["family_best"].get(fam, {}).get("auc") for r in rows])
    # A plain-language decision useful for watchdog/status reports.
    fam_counts = out["winner_family_counts"]
    if fam_counts.get("entry_normal_manifold", 0) >= max(3, int(np.ceil(0.6 * len(rows)))):
        out["decision"] = "ENTRY_NORMAL_MANIFOLD_SIGNAL_PROMISING"
    elif fam_counts.get("row_normal_anchor", 0) + fam_counts.get("column_anomaly_reference", 0) >= max(3, int(np.ceil(0.6 * len(rows)))):
        out["decision"] = "ROW_OR_COLUMN_STRUCTURE_PROMISING"
    elif fam_counts.get("scalar_matrix_summary", 0) >= max(3, int(np.ceil(0.6 * len(rows)))):
        out["decision"] = "SIMPLE_SCALAR_SUMMARY_STILL_DOMINATES"
    elif (out["best_with_degree_abs"] or {}).get("mean", 0) > 0.5:
        out["decision"] = "BEST_SIGNAL_DEGREE_CONFOUNDED"
    else:
        out["decision"] = "MIXED_DECOMPOSITION_SIGNAL"
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
            rr = sorted(rows_by_variant[v], key=lambda r: r["seed"])
            ent = {"variant": v, "report_codename": VARIANTS[v]["report_codename"], "n_completed": len(rr), "completed_seeds": [int(r["seed"]) for r in rr]}
            if rr:
                ent["partial_summary"] = summarize(rr)
            partial.append(ent)
        atomic_write_json(args.progress_out, {
            "status": status,
            "probe": "route25_leg3_response_matrix_decomposition_probe",
            "done": int(done),
            "total": int(total),
            "devices": devices,
            "current": current,
            "elapsed_sec": float(time.time() - start),
            "partial_results": partial,
            "errors": errors[-5:],
        })

    def worker(device):
        nonlocal done
        while True:
            try:
                v, s = task_q.get_nowait()
            except queue.Empty:
                return
            cur = {"variant": v, "seed": int(s), "device": int(device)}
            try:
                with lock:
                    print(json.dumps({"stage": "task_start", **cur}, ensure_ascii=False), flush=True)
                    snapshot("running", cur)
                row = run_one(args, v, s, device)
                with lock:
                    rows_by_variant[v].append(row)
                    done += 1
                    print(json.dumps({"stage": "task_done", **cur, "best": row["best_metric"]["name"], "auc": row["best_metric"]["auc"]}, ensure_ascii=False), flush=True)
                    snapshot("running", cur)
            except Exception as e:
                tb = traceback.format_exc()
                with lock:
                    done += 1
                    errors.append({"variant": v, "seed": int(s), "device": int(device), "error": repr(e), "traceback": tb[-4000:]})
                    print(json.dumps({"stage": "task_failed", **cur, "error": repr(e)}, ensure_ascii=False), flush=True)
                    snapshot("failed_running", cur)
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
        rr = sorted(rows_by_variant[v], key=lambda r: r["seed"])
        if len(rr) != len(seeds):
            errors.append({"variant": v, "error": f"completed {len(rr)}/{len(seeds)} seeds"})
        results.append({
            "variant": v,
            "report_codename": VARIANTS[v]["report_codename"],
            "definition": VARIANTS[v]["definition"],
            "changes_from_base": VARIANTS[v]["changes"],
            "rows": rr,
            "aggregate": summarize(rr) if rr else {},
        })
    status = "finished" if not errors and all(len(rows_by_variant[v]) == len(seeds) for v in variants) else "failed"
    output = {
        "status": status,
        "probe": "route25_leg3_response_matrix_decomposition_probe",
        "protocol": "Frozen encoder; C-LEG3 reference regime fixed; no AE/training; labels diagnostic-only for AUC/AP/autopsy; decomposition families are scalar, row, column, entry normal-manifold, complementarity, and proxy confound diagnostics.",
        "dataset": args.dataset,
        "seeds": seeds,
        "devices": devices,
        "variants_requested": variants,
        "variant_definitions": {k: VARIANTS[k] for k in variants},
        "config": vars(args),
        "results": results,
        "errors": errors,
        "time_sec": float(time.time() - start),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    atomic_write_json(args.progress_out, {"status": status, "done": int(done), "total": int(total), "devices": devices, "elapsed_sec": float(time.time() - start), "results_summary": [{"variant": r["variant"], "aggregate": r["aggregate"]} for r in results], "errors": errors[-10:]})
    print("FINAL " + json.dumps({"status": status, "results_summary": [{"variant": r["variant"], "aggregate": r["aggregate"]} for r in results], "errors": errors[-3:], "time_sec": output["time_sec"]}, indent=2, ensure_ascii=False), flush=True)
    if status != "finished":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
