#!/usr/bin/env python3
"""Route2.5 AMRF-style adaptive matrix response diagnostic for DualRefGAD.

Question: can we borrow the *spirit* of AdaFreq without copying RHO by learning /
estimating which response-matrix entries are stable/reliable on labeled-normal
nodes, then scoring deviations from that normal response manifold?

Protocol:
- Frozen VecGAD encoder and Route2.5 response matrix construction.
- Labels are diagnostic-only for AUC/AP/autopsy.
- AMRF variants fit only on labeled-normal training nodes.
- Parallel execution across independent (variant, seed) tasks via --devices.
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
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
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
    "current_refs_amrf": {
        "definition": "Current Route2.5 reference construction; AMRF scores fit on labeled-normal response matrices only.",
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
    "old_semantic_refs_amrf": {
        "definition": "Old-style anomaly-reference semantics: label-gated normal refs + normal_soft_or + descriptor_similarity; tests whether AMRF depends on old reference orientation.",
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
    "old_exact_amrf": {
        "definition": "Reproduce old 0.80-ish response regime before AMRF scoring: hop_attr + PCA residual + label_gate + normal_soft_or + descriptor_similarity + GT3.",
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
}


def safe_auc_ap(labels, score, idx):
    idx = np.asarray(idx, dtype=np.int64)
    return float(roc_auc_score(labels[idx], score[idx])), float(average_precision_score(labels[idx], score[idx]))


def zscore_from_normals(X, normal_idx):
    nidx = np.asarray(normal_idx, dtype=np.int64)
    mu = X[nidx].mean(axis=0, keepdims=True)
    sd = X[nidx].std(axis=0, keepdims=True) + 1e-6
    Z = (X - mu) / sd
    return Z.astype(np.float32), mu.reshape(-1), sd.reshape(-1)


def split_reliability(X, normal_idx, seed):
    rng = np.random.default_rng(seed + 9173)
    nidx = rng.permutation(np.asarray(normal_idx, dtype=np.int64))
    h = max(1, len(nidx) // 2)
    a, b = nidx[:h], nidx[h:]
    if len(b) == 0:
        b = a
    mu_a = X[a].mean(axis=0)
    mu_b = X[b].mean(axis=0)
    # Stable normal entries have similar split means and low within-normal std.
    diff = np.abs(mu_a - mu_b)
    std = X[nidx].std(axis=0) + 1e-6
    raw = 1.0 / (diff + std + 1e-3)
    raw = raw / (np.mean(raw) + 1e-12)
    return np.clip(raw, 0.05, 20.0).astype(np.float32)


def fit_amrf_scores(mat, margin, meta, normal_idx, labels, idx_test, seed, pca_dim=12):
    X = mat.reshape(mat.shape[0], -1).astype(np.float32)
    Z, mu, sd = zscore_from_normals(X, normal_idx)
    rel = split_reliability(X, normal_idx, seed)
    WZ = Z * rel[None, :]
    nidx = np.asarray(normal_idx, dtype=np.int64)
    arrays = {
        "margin": margin,
        "neg_margin": -margin,
        "mat_mean": mat.mean(axis=(1, 2)),
        "neg_mat_mean": -mat.mean(axis=(1, 2)),
        "mat_std": mat.std(axis=(1, 2)),
        "amrf_invvar_l2": np.mean(Z ** 2, axis=1),
        "amrf_reliable_l2": np.mean(WZ ** 2, axis=1),
        "amrf_reliable_abs": np.mean(np.abs(WZ), axis=1),
        "amrf_reliable_posmean": np.mean(WZ, axis=1),
        "amrf_reliable_negmean": -np.mean(WZ, axis=1),
        "rejection": meta["rejection"],
        "residual_norm": meta["residual_norm"],
        "degree": meta["degree"],
    }
    ncomp = int(min(pca_dim, max(1, len(nidx) - 1), WZ.shape[1]))
    if ncomp >= 1:
        pca = PCA(n_components=ncomp, svd_solver="randomized", random_state=seed)
        pca.fit(WZ[nidx])
        rec = pca.inverse_transform(pca.transform(WZ))
        arrays["amrf_reliable_pca_resid"] = np.mean((WZ - rec) ** 2, axis=1)
        emb = pca.transform(WZ).astype(np.float32)
        emu = emb[nidx].mean(axis=0, keepdims=True)
        esd = emb[nidx].std(axis=0, keepdims=True) + 1e-6
        arrays["amrf_reliable_pca_center"] = np.mean(((emb - emu) / esd) ** 2, axis=1)
        try:
            lw = LedoitWolf().fit(emb[nidx])
            arrays["amrf_reliable_pca_mahal"] = lw.mahalanobis(emb).astype(np.float64)
        except Exception:
            pass
    # A tiny label-free pseudo-boundary sanity check: normal-vs-reference-dropout noise.
    # Positives here are corrupted copies of labeled normals, not anomaly labels.
    try:
        rng = np.random.default_rng(seed + 12345)
        train = WZ[nidx]
        noise = train.copy()
        mask = rng.random(noise.shape) < 0.35
        noise[mask] = 0.0
        Xb = np.vstack([train, noise])
        yb = np.concatenate([np.zeros(len(train)), np.ones(len(train))])
        clf = LogisticRegression(max_iter=200, class_weight="balanced", random_state=seed)
        clf.fit(Xb, yb)
        arrays["amrf_dropout_boundary"] = clf.decision_function(WZ)
    except Exception:
        pass
    metrics = metric_block(labels, idx_test, arrays, base_name="margin")
    ranked = sorted(metrics.items(), key=lambda kv: kv[1]["auc"], reverse=True)
    amrf_names = [k for k in arrays if k.startswith("amrf_")]
    best_amrf_name = max(amrf_names, key=lambda k: metrics[k]["auc"])
    best_scalar_name = max(["margin", "neg_margin", "mat_mean", "neg_mat_mean", "mat_std", "rejection", "residual_norm"], key=lambda k: metrics[k]["auc"])
    return {
        "arrays": arrays,
        "metrics": metrics,
        "ranked": ranked,
        "best_amrf": {"name": best_amrf_name, **metrics[best_amrf_name]},
        "best_scalar": {"name": best_scalar_name, **metrics[best_scalar_name]},
        "reliability_summary": {
            "mean": float(np.mean(rel)),
            "std": float(np.std(rel)),
            "top10_frac_mean": float(np.mean(np.sort(rel)[-max(1, int(0.1 * len(rel))):])),
            "bottom10_frac_mean": float(np.mean(np.sort(rel)[:max(1, int(0.1 * len(rel)))])),
        },
    }


def build_variant_args(cli_args, variant_name, seed, device):
    cfg = copy.deepcopy(BASE_DEFAULTS)
    cfg.update(VARIANTS[variant_name]["changes"])
    cfg.update({
        "project_root": cli_args.project_root,
        "dataset": cli_args.dataset,
        "device": int(device),
        "seed": int(seed),
        "train_rate": cli_args.train_rate,
        "val_rate": cli_args.val_rate,
        "normal_k": cli_args.normal_k,
        "anom_k": cli_args.anom_k,
        "ln_mode": cli_args.ln_mode,
        "pca_dim": cli_args.pca_dim,
    })
    return argparse.Namespace(**cfg)


def run_one_task(cli_args, variant_name, seed, device):
    args = build_variant_args(cli_args, variant_name, seed, device)
    set_seed(seed)
    root = Path(args.project_root).expanduser().resolve()
    sys.path.insert(0, str(root))
    os.chdir(str(root))
    from utils import load_mat, preprocess_features, normalize_adj  # noqa: E402
    from VecGAD import VecGAD  # noqa: E402
    import torch  # noqa: E402

    def to_dense_features(dataset, features):
        if dataset in ["Amazon", "tf_finace", "t_finance", "reddit", "elliptic"]:
            features, _ = preprocess_features(features)
            return np.asarray(features, dtype=np.float32)
        return np.asarray(features.todense(), dtype=np.float32)

    dev = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")
    print(json.dumps({"stage": "seed_start", "variant": variant_name, "seed": seed, "device": str(dev)}, ensure_ascii=False), flush=True)
    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat(args.dataset, args.train_rate, args.val_rate, args=args)
    features_np = to_dense_features(args.dataset, features)
    labels_np = np.asarray(ano_label).reshape(-1).astype(int)
    normal_idx = np.asarray(normal_for_train_idx, dtype=np.int64)
    idx_test = np.asarray(idx_test, dtype=np.int64)
    assert np.sum(labels_np[normal_idx]) == 0, "Data leakage: normal_for_train_idx contains anomalies"

    z = build_descriptor(args.descriptor_mode, features_np, adj, normalize_adj, args.hops, args.rw_steps)
    nm = NormalModel(args.pn_estimator, z, normal_idx, args.pca_components)
    residual = nm.residual()
    normal_refs, anom_refs, meta = select_refs(z, residual, normal_idx, nm, features_np, adj, args, normalize_adj)
    token_tensor = build_tokens(features_np, normal_refs, anom_refs)
    model = VecGAD(features_np.shape[1], args.embedding_dim, "prelu", args).to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    with torch.no_grad():
        emb = encode_tokens_batched(model, token_tensor, dev, args.encode_batch_size)
    del token_tensor, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mat, margin = response_matrix_from_embeddings(emb, normal_refs, anom_refs)
    fitted = fit_amrf_scores(mat, margin, meta, normal_idx, labels_np, idx_test, seed, args.pca_dim)
    row = {
        "variant": variant_name,
        "seed": int(seed),
        "device": int(device),
        "best_amrf": fitted["best_amrf"],
        "best_scalar": fitted["best_scalar"],
        "top_metrics": [{"name": n, **m} for n, m in fitted["ranked"][: int(cli_args.top_k)]],
        "reference_diagnostics": {
            "anom_ref_anom_ratio_diagnostic": float(np.mean(labels_np[anom_refs] == 1)),
            "normal_ref_anom_ratio_diagnostic": float(np.mean(labels_np[normal_refs] == 1)),
        },
        "comparisons": {
            "amrf_minus_best_scalar_auc": float(fitted["best_amrf"]["auc"] - fitted["best_scalar"]["auc"]),
            "amrf_spearman_with_margin": float(fitted["best_amrf"].get("spearman_with_margin", 0.0)),
            "amrf_top5_jaccard_with_margin": float(fitted["best_amrf"].get("top5_jaccard_with_margin", 0.0)),
        },
        "reliability_summary": fitted["reliability_summary"],
        "counts": {"num_nodes": int(len(labels_np)), "num_test": int(len(idx_test)), "num_labeled_normals": int(len(normal_idx)), "matrix_shape": list(mat.shape)},
    }
    print(json.dumps({"stage": "seed_done", "variant": variant_name, "seed": seed, "best_amrf": row["best_amrf"]["name"], "auc": row["best_amrf"]["auc"], "delta": row["comparisons"]["amrf_minus_best_scalar_auc"]}, ensure_ascii=False), flush=True)
    return row


def mean_std(vals):
    vals = np.asarray([v for v in vals if v is not None], dtype=np.float64)
    if vals.size == 0:
        return None
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}


def summarize_variant(rows):
    winners = {}
    for r in rows:
        winners[r["best_amrf"]["name"]] = winners.get(r["best_amrf"]["name"], 0) + 1
    deltas = [r["comparisons"]["amrf_minus_best_scalar_auc"] for r in rows]
    aucs = [r["best_amrf"]["auc"] for r in rows]
    aps = [r["best_amrf"]["ap"] for r in rows]
    scalar_aucs = [r["best_scalar"]["auc"] for r in rows]
    margin_corrs = [r["comparisons"]["amrf_spearman_with_margin"] for r in rows]
    ref_ratios = [r["reference_diagnostics"]["anom_ref_anom_ratio_diagnostic"] for r in rows]
    if len(rows) >= 3 and np.mean(deltas) > 0.01 and np.median(margin_corrs) < 0.75:
        decision = "PROMOTE_AMRF_SIGNAL__COMPLEMENTARY_TO_MARGIN"
    elif len(rows) >= 3 and np.mean(deltas) > 0.0:
        decision = "PROMOTE_CAUTION__AMRF_IMPROVES_BUT_MAY_TRACK_MARGIN"
    elif len(rows) >= 3 and np.mean(aucs) >= np.mean(scalar_aucs) - 0.01 and np.median(margin_corrs) < 0.5:
        decision = "KEEP_AS_COMPLEMENTARY_DIAGNOSTIC"
    else:
        decision = "DROP_OR_REPAIR_AMRF_FOR_THIS_REFERENCE_REGIME"
    return {
        "best_amrf_auc": mean_std(aucs),
        "best_amrf_ap": mean_std(aps),
        "best_scalar_auc": mean_std(scalar_aucs),
        "amrf_minus_best_scalar_auc": mean_std(deltas),
        "amrf_spearman_with_margin": mean_std(margin_corrs),
        "anom_ref_anom_ratio": mean_std(ref_ratios),
        "winner_counts": winners,
        "decision": decision,
    }


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
            "best_amrf_auc_mean": (r["aggregate"].get("best_amrf_auc") or {}).get("mean"),
            "amrf_minus_best_scalar_auc_mean": (r["aggregate"].get("amrf_minus_best_scalar_auc") or {}).get("mean"),
            "amrf_spearman_with_margin_mean": (r["aggregate"].get("amrf_spearman_with_margin") or {}).get("mean"),
            "winner_counts": r["aggregate"].get("winner_counts"),
            "decision": r["aggregate"].get("decision"),
        }
        for r in results
    ], key=lambda x: (x["amrf_minus_best_scalar_auc_mean"] if x["amrf_minus_best_scalar_auc_mean"] is not None else -999), reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(Path.home() / "DualRefGAD"))
    ap.add_argument("--dataset", default="elliptic")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--devices", default="")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--variants", default="current_refs_amrf,old_semantic_refs_amrf,old_exact_amrf")
    ap.add_argument("--train_rate", type=float, default=0.05)
    ap.add_argument("--val_rate", type=float, default=0.0)
    ap.add_argument("--ln_mode", default="descriptor_similarity")
    ap.add_argument("--normal_k", type=int, default=4)
    ap.add_argument("--anom_k", type=int, default=16)
    ap.add_argument("--pca_dim", type=int, default=12)
    ap.add_argument("--top_k", type=int, default=10)
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
        partial = []
        for vv in variants:
            rr = rows_by_variant[vv]
            entry = {"variant": vv, "completed_seeds": sorted([int(r["seed"]) for r in rr]), "n_completed": len(rr)}
            if len(rr) == len(seeds):
                entry["aggregate"] = summarize_variant(sorted(rr, key=lambda r: r["seed"]))
            partial.append(entry)
        atomic_write_json(args.progress_out, {"status": status, "done": done, "total": total, "devices": devices, "current": current, "errors": errors[-5:], "elapsed_sec": time.time() - start, "partial_results": partial})

    def worker(device):
        nonlocal done
        while True:
            try:
                v, s = task_q.get_nowait()
            except queue.Empty:
                return
            current = {"variant": v, "seed": int(s), "device": int(device)}
            try:
                with lock:
                    print(json.dumps({"stage": "task_start", **current, "definition": VARIANTS[v]["definition"]}, ensure_ascii=False), flush=True)
                    snapshot("running", current)
                row = run_one_task(args, v, s, device)
                with lock:
                    rows_by_variant[v].append(row)
                    done += 1
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
        results.append({"variant": v, "definition": VARIANTS[v]["definition"], "rows": rows, "aggregate": summarize_variant(rows) if rows else {}})
    status = "finished" if not errors and all(len(rows_by_variant[v]) == len(seeds) for v in variants) else "failed"
    lb = leaderboard(results)
    output = {
        "status": status,
        "probe": "route25_amrf_matrix_response_probe",
        "protocol": "Frozen encoder; AMRF-style adaptive response-matrix reliability scoring fitted only on labeled-normal nodes; labels diagnostic-only for AUC/AP/autopsy; parallel independent variant/seed tasks.",
        "seeds": seeds,
        "devices": devices,
        "variants_requested": variants,
        "variant_definitions": VARIANTS,
        "results": results,
        "leaderboard_by_delta_auc": lb,
        "errors": errors,
        "time_sec": float(time.time() - start),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    atomic_write_json(args.progress_out, {"status": status, "done": done, "total": total, "devices": devices, "elapsed_sec": time.time() - start, "leaderboard_by_delta_auc": lb, "errors": errors[-10:]})
    print("FINAL " + json.dumps({"status": status, "leaderboard_by_delta_auc": lb, "errors": errors[-3:], "time_sec": output["time_sec"]}, indent=2, ensure_ascii=False), flush=True)
    if status != "finished":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
