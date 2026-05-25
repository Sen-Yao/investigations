#!/usr/bin/env python3
"""Stage A response-matrix orientation/regime probe for DualRefGAD Route2.5.

Question: after normal-only Matrix AE failed as a stable method component,
which response-matrix families still carry signal, and do promote/drop splits
look different in reference/orientation/regime diagnostics?

Protocol:
- Frozen VecGAD encoder / response-matrix construction reused from Route2.5.
- No AE training.
- Labels are diagnostic-only for AUC/AP and autopsy.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

# Reuse the exact construction helpers from the original Route2.5 probe.
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
    set_seed,
)


def to_float(x):
    return float(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else None


def safe_corr(a, b):
    try:
        v = spearmanr(np.asarray(a), np.asarray(b)).correlation
        return float(0.0 if np.isnan(v) else v)
    except Exception:
        return 0.0


def quantile_gap(values, labels, idx, q):
    v = np.asarray(values)[idx]
    y = np.asarray(labels)[idx]
    if not np.any(y == 0) or not np.any(y == 1):
        return 0.0
    return float(np.quantile(v[y == 1], q) - np.quantile(v[y == 0], q))


def bucket_edges(x, n_bins):
    x = np.asarray(x, dtype=np.float64)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(x, qs)
    # Avoid duplicate-edge digitize collapse by adding tiny monotone jitter.
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    return edges


def stratified_auc(labels, score, idx, stratifier, n_bins=3):
    from sklearn.metrics import average_precision_score, roc_auc_score

    idx = np.asarray(idx, dtype=np.int64)
    labels = np.asarray(labels).reshape(-1).astype(int)
    score = np.asarray(score, dtype=np.float64)
    strat = np.asarray(stratifier, dtype=np.float64)
    edges = bucket_edges(strat[idx], n_bins)
    bins = np.digitize(strat[idx], edges[1:-1], right=True)
    out = []
    for b in range(n_bins):
        sub = idx[bins == b]
        if len(sub) == 0:
            continue
        y = labels[sub]
        block = {
            "bin": int(b),
            "count": int(len(sub)),
            "anom_rate": float(np.mean(y == 1)),
            "range": [float(edges[b]), float(edges[b + 1])],
        }
        if len(np.unique(y)) == 2:
            block["auc"] = float(roc_auc_score(y, score[sub]))
            block["ap"] = float(average_precision_score(y, score[sub]))
        else:
            block["auc"] = None
            block["ap"] = None
        out.append(block)
    return out


def build_arrays(mat, margin, meta):
    x = mat.reshape(mat.shape[0], -1)
    q25 = np.quantile(x, 0.25, axis=1)
    q50 = np.quantile(x, 0.50, axis=1)
    q75 = np.quantile(x, 0.75, axis=1)
    lo = np.quantile(x, 0.10, axis=1)
    hi = np.quantile(x, 0.90, axis=1)
    sorted_x = np.sort(x, axis=1)
    trim = sorted_x[:, max(1, int(x.shape[1] * 0.1)) : max(2, int(x.shape[1] * 0.9))].mean(axis=1)
    return {
        "margin": margin,
        "neg_margin": -margin,
        "mat_mean": mat.mean(axis=(1, 2)),
        "neg_mat_mean": -mat.mean(axis=(1, 2)),
        "mat_std": mat.std(axis=(1, 2)),
        "neg_mat_std": -mat.std(axis=(1, 2)),
        "mat_median": q50,
        "neg_mat_median": -q50,
        "mat_q25": q25,
        "neg_mat_q25": -q25,
        "mat_q75": q75,
        "neg_mat_q75": -q75,
        "mat_iqr": q75 - q25,
        "mat_trimmed_mean": trim,
        "neg_mat_trimmed_mean": -trim,
        "mat_middle80_width": hi - lo,
        "mat_top5_mean": sorted_x[:, -5:].mean(axis=1),
        "neg_mat_top5_mean": -sorted_x[:, -5:].mean(axis=1),
        "mat_bottom5_mean": sorted_x[:, :5].mean(axis=1),
        "neg_mat_bottom5_mean": -sorted_x[:, :5].mean(axis=1),
        "mat_high08_ratio": (mat > 0.8).mean(axis=(1, 2)),
        "mat_lowneg08_ratio": (mat < -0.8).mean(axis=(1, 2)),
        "rejection": meta["rejection"],
        "residual_norm": meta["residual_norm"],
        "degree": meta["degree"],
    }


def run_one_seed(args, seed):
    set_seed(seed)
    root = Path(args.project_root).expanduser().resolve()
    sys.path.insert(0, str(root))
    os.chdir(str(root))
    from utils import load_mat, preprocess_features, normalize_adj  # noqa: E402
    from VecGAD import VecGAD  # noqa: E402

    def to_dense_features(dataset, features):
        if dataset in ["Amazon", "tf_finace", "t_finance", "reddit", "elliptic"]:
            features, _ = preprocess_features(features)
            return np.asarray(features, dtype=np.float32)
        return np.asarray(features.todense(), dtype=np.float32)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")
    print(json.dumps({"stage": "seed_start", "seed": seed, "device": str(device)}, ensure_ascii=False), flush=True)
    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat(args.dataset, args.train_rate, args.val_rate, args=args)
    features_np = to_dense_features(args.dataset, features)
    labels_np = np.asarray(ano_label).reshape(-1).astype(int)
    normal_idx = np.asarray(normal_for_train_idx, dtype=np.int64)
    idx_test = np.asarray(idx_test, dtype=np.int64)
    assert np.sum(labels_np[normal_idx]) == 0, "Data leakage: normal_for_train_idx contains anomalies"

    z = build_descriptor(args.descriptor_mode, features_np, adj, normalize_adj, args.hops, args.rw_steps)
    nm = NormalModel(args.pn_estimator, z, normal_idx, args.pca_components)
    residual = nm.residual()
    # select_refs expects args.seed for deterministic candidate behavior if needed.
    args.seed = seed
    from route25_matrix_autoencoder_probe import select_refs  # noqa: E402
    normal_refs, anom_refs, meta = select_refs(z, residual, normal_idx, nm, features_np, adj, args, normalize_adj)

    token_tensor = build_tokens(features_np, normal_refs, anom_refs)
    model = VecGAD(features_np.shape[1], args.embedding_dim, "prelu", args).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    with torch.no_grad():
        emb = encode_tokens_batched(model, token_tensor, device, args.encode_batch_size)
    del token_tensor, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mat, margin = response_matrix_from_embeddings(emb, normal_refs, anom_refs)
    arrays = build_arrays(mat, margin, meta)
    metrics = metric_block(labels_np, idx_test, arrays, base_name="margin")
    ranked = sorted(metrics.items(), key=lambda kv: kv[1]["auc"], reverse=True)

    family = {}
    for name, m in metrics.items():
        if name in ["rejection", "residual_norm", "degree"]:
            fam = "proxy"
        elif name.startswith("neg_") or name.startswith("mat_low") or name.startswith("mat_bottom"):
            fam = "negative_orientation"
        elif name.startswith("mat_"):
            fam = "positive_or_shape"
        elif name.startswith("neg_margin"):
            fam = "negative_margin"
        else:
            fam = "margin"
        family.setdefault(fam, []).append({"name": name, **m})
    family_best = {fam: max(vals, key=lambda x: x["auc"]) for fam, vals in family.items()}

    best_name, best_metric = ranked[0]
    second_name, second_metric = ranked[1]
    orientation_pair = {
        "mat_mean_minus_neg_mat_mean_auc": float(metrics["mat_mean"]["auc"] - metrics["neg_mat_mean"]["auc"]),
        "margin_minus_neg_margin_auc": float(metrics["margin"]["auc"] - metrics["neg_margin"]["auc"]),
        "winner_is_negative_orientation": bool(best_name.startswith("neg_") or best_name.startswith("mat_low") or best_name.startswith("mat_bottom")),
    }
    diagnostics = {
        "anom_ref_anom_ratio_diagnostic": float(np.mean(labels_np[anom_refs] == 1)),
        "normal_ref_anom_ratio_diagnostic": float(np.mean(labels_np[normal_refs] == 1)),
        "test_anom_rate": float(np.mean(labels_np[idx_test] == 1)),
        "degree_spearman_with_best": safe_corr(arrays["degree"][idx_test], arrays[best_name][idx_test]),
        "rejection_spearman_with_best": safe_corr(arrays["rejection"][idx_test], arrays[best_name][idx_test]),
        "margin_spearman_with_best": safe_spearman(arrays[best_name][idx_test], margin[idx_test]),
        "top5_jaccard_best_with_margin": jaccard_top(arrays[best_name][idx_test], margin[idx_test], 0.05),
        "best_q25_gap_anom_minus_normal": quantile_gap(arrays[best_name], labels_np, idx_test, 0.25),
        "best_q50_gap_anom_minus_normal": quantile_gap(arrays[best_name], labels_np, idx_test, 0.50),
        "best_q75_gap_anom_minus_normal": quantile_gap(arrays[best_name], labels_np, idx_test, 0.75),
    }
    stratified = {
        "degree": stratified_auc(labels_np, arrays[best_name], idx_test, arrays["degree"], args.n_bins),
        "rejection": stratified_auc(labels_np, arrays[best_name], idx_test, arrays["rejection"], args.n_bins),
        "margin": stratified_auc(labels_np, arrays[best_name], idx_test, margin, args.n_bins),
    }
    row = {
        "seed": int(seed),
        "best_scalar": {"name": best_name, **best_metric},
        "second_scalar": {"name": second_name, **second_metric},
        "family_best": family_best,
        "orientation_pair": orientation_pair,
        "diagnostics": diagnostics,
        "stratified_best_auc": stratified,
        "top_metrics": [{"name": n, **m} for n, m in ranked[: args.top_k]],
        "counts": {
            "num_nodes": int(len(labels_np)),
            "num_test": int(len(idx_test)),
            "num_labeled_normals": int(len(normal_idx)),
            "matrix_shape": list(mat.shape),
        },
    }
    print(json.dumps({"stage": "seed_done", "seed": seed, "best": best_name, "auc": best_metric["auc"]}, ensure_ascii=False), flush=True)
    return row


def summarize(rows):
    def mean_std(vals):
        vals = np.asarray(vals, dtype=np.float64)
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}

    best_auc = [r["best_scalar"]["auc"] for r in rows]
    neg_mean_auc = []
    for r in rows:
        metric = next((m for m in r.get("top_metrics", []) if m.get("name") == "neg_mat_mean"), None)
        if metric is not None and metric.get("auc") is not None:
            neg_mean_auc.append(metric.get("auc"))
    winners = {}
    families = {}
    for r in rows:
        winners[r["best_scalar"]["name"]] = winners.get(r["best_scalar"]["name"], 0) + 1
        fam = "negative_orientation" if r["orientation_pair"]["winner_is_negative_orientation"] else "other"
        families[fam] = families.get(fam, 0) + 1
    delta_mean = [r["orientation_pair"]["mat_mean_minus_neg_mat_mean_auc"] for r in rows]
    ref_ratio = [r["diagnostics"]["anom_ref_anom_ratio_diagnostic"] for r in rows]
    margin_corr = [r["diagnostics"]["margin_spearman_with_best"] for r in rows]
    degree_corr = [r["diagnostics"]["degree_spearman_with_best"] for r in rows]
    return {
        "best_scalar_auc": mean_std(best_auc),
        "neg_mat_mean_auc": mean_std(neg_mean_auc) if neg_mean_auc else None,
        "winner_counts": winners,
        "orientation_family_counts": families,
        "mat_mean_minus_neg_mat_mean_auc": mean_std(delta_mean),
        "anom_ref_anom_ratio": mean_std(ref_ratio),
        "margin_spearman_with_best": mean_std(margin_corr),
        "degree_spearman_with_best": mean_std(degree_corr),
        "decision": decide_stage_a(rows),
    }


def decide_stage_a(rows):
    neg_wins = sum(1 for r in rows if r["orientation_pair"]["winner_is_negative_orientation"])
    mean_delta = float(np.mean([r["orientation_pair"]["mat_mean_minus_neg_mat_mean_auc"] for r in rows]))
    degree_abs = float(np.mean([abs(r["diagnostics"]["degree_spearman_with_best"]) for r in rows]))
    if neg_wins >= max(3, int(np.ceil(len(rows) * 0.6))) and mean_delta < 0:
        if degree_abs > 0.5:
            return "ORIENTATION_SIGNAL_PRESENT_BUT_DEGREE_REGIME_CONFOUNDED"
        return "NEGATIVE_ORIENTATION_SIGNAL_STABLE__PRIORITIZE_SIGN_AWARE_SCALAR_OR_REGIME_SCORE"
    if degree_abs > 0.5:
        return "REGIME_PROXY_DOMINATES__DO_NOT_PROMOTE_RAW_MATRIX_SCORE"
    return "MIXED_ORIENTATION_REGIME__NEEDS_REFERENCE_POOL_REPAIR_BEFORE_METHOD"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(Path.home() / "DualRefGAD"))
    ap.add_argument("--dataset", default="elliptic")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--train_rate", type=float, default=0.05)
    ap.add_argument("--val_rate", type=float, default=0.0)
    ap.add_argument("--descriptor_mode", choices=["hop_attr", "rwse", "hop_attr_rwse"], default="hop_attr_rwse")
    ap.add_argument("--pn_estimator", choices=["diag_gaussian", "pca_residual"], default="diag_gaussian")
    ap.add_argument("--gn_mode", choices=["label_gate", "normal_density", "label_gate_density"], default="label_gate_density")
    ap.add_argument("--ln_mode", default="descriptor_similarity")
    ap.add_argument("--ga_mode", choices=["normal_rejection", "residual_norm", "normal_soft_or"], default="normal_rejection")
    ap.add_argument("--la_mode", choices=["residual_cosine", "descriptor_similarity"], default="residual_cosine")
    ap.add_argument("--normal_k", type=int, default=4)
    ap.add_argument("--anom_k", type=int, default=16)
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
    ap.add_argument("--encode_batch_size", type=int, default=2048)
    ap.add_argument("--ref_block_size", type=int, default=1024)
    ap.add_argument("--use_approx_anom_refs", action="store_true")
    ap.add_argument("--anom_approx_k", type=int, default=1000)
    ap.add_argument("--n_bins", type=int, default=3)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    start = time.time()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    rows = []
    for s in seeds:
        rows.append(run_one_seed(args, s))
    result = {
        "status": "finished",
        "probe": "route25_stage_a_matrix_orientation_regime_probe",
        "protocol": "Frozen encoder; no AE training; response-matrix scalar/orientation/regime decomposition; labels diagnostic-only.",
        "dataset": args.dataset,
        "seeds": seeds,
        "config": vars(args),
        "rows": rows,
        "aggregate": summarize(rows),
        "time_sec": float(time.time() - start),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("FINAL " + json.dumps(result["aggregate"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
