#!/usr/bin/env python3
"""C-LEG3 Layer-0 fixed-formula reliability gate probe.

No training. No learned weights. No anomaly-label tuning.

Purpose:
  Test whether the v2-report consensus_minus_fragmentation signal can be used
  as a shallow fixed gate/rescoring rule around mat_mean while controlling
  false-positive reintroduction.

Labels are diagnostic-only for AUC/AP and oracle top-K boundary categories.
"""
import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
NEXUS_ROOT = SCRIPT_DIR.parents[3]
UPSTREAM = NEXUS_ROOT / "2026-05-21-dualrefgad-constraint-calibrated-reference-relation" / "experiments" / "scripts"
ROUTE25 = NEXUS_ROOT / "2026-05-19-dualrefgad-route25-matrix-autoencoder" / "experiments" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(ROUTE25))

from route25_leg3_response_matrix_decomposition_probe import (  # noqa: E402
    BASE_DEFAULTS,
    VARIANTS,
    build_decomposition_arrays,
    parse_ints,
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

EPS = 1e-9


def atomic_write_json(path, payload):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def mean_std(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return None
    x = np.asarray(vals, dtype=np.float64)
    return {"mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()), "max": float(x.max())}


def sha1_ints(xs):
    arr = np.asarray(xs, dtype=np.int64).reshape(-1)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def split_fingerprint(labels_np, idx_train, idx_val, idx_test, normal_idx):
    idx_train = np.asarray(idx_train, dtype=np.int64)
    idx_val = np.asarray(idx_val, dtype=np.int64)
    idx_test = np.asarray(idx_test, dtype=np.int64)
    normal_idx = np.asarray(normal_idx, dtype=np.int64)
    return {
        "idx_train_sha1": sha1_ints(idx_train),
        "idx_val_sha1": sha1_ints(idx_val),
        "idx_test_sha1": sha1_ints(idx_test),
        "normal_for_train_sha1": sha1_ints(normal_idx),
        "train_count": int(len(idx_train)),
        "val_count": int(len(idx_val)),
        "test_count": int(len(idx_test)),
        "train_anom_count": int(np.sum(labels_np[idx_train] == 1)),
        "test_anom_count": int(np.sum(labels_np[idx_test] == 1)),
        "normal_for_train_anom_count": int(np.sum(labels_np[normal_idx] == 1)),
    }


def to_dense_features(dataset, features, preprocess_features):
    if dataset in ["Amazon", "tf_finace", "t_finance", "reddit", "elliptic"]:
        features, _ = preprocess_features(features)
        return np.asarray(features, dtype=np.float32)
    return np.asarray(features.todense(), dtype=np.float32)


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
    return set(map(int, idx_test[np.argsort(-s)[:k]]))


def entropy_effective_count(x, axis):
    x = np.asarray(x, dtype=np.float64)
    w = np.maximum(x - x.min(axis=axis, keepdims=True), 0.0) + 1e-9
    p = w / np.sum(w, axis=axis, keepdims=True)
    ent = -np.sum(p * np.log(p + 1e-12), axis=axis)
    return np.exp(ent)


def robust_z(x):
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < EPS:
        sd = np.std(x)
        return (x - np.mean(x)) / (sd + EPS)
    return (x - med) / (1.4826 * mad + EPS)


def sigmoid(x):
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def candidate_readouts(mat):
    """Return no-label candidate scores and per-node proxy features."""
    flat = mat.reshape(mat.shape[0], -1)
    row = mat.mean(axis=2)
    col = mat.mean(axis=1)
    row_std = row.std(axis=1)
    col_std = col.std(axis=1)
    mat_std = flat.std(axis=1)
    row_range = row.max(axis=1) - row.min(axis=1)
    col_range = col.max(axis=1) - col.min(axis=1)
    row_eff = entropy_effective_count(row, axis=1)
    col_eff = entropy_effective_count(col, axis=1)
    row_rel = row_eff / row.shape[1] / (1.0 + row_std)
    col_rel = col_eff / col.shape[1] / (1.0 + col_std)
    reliability = np.sqrt(np.maximum(row_rel, 0) * np.maximum(col_rel, 0))
    sorted_flat = np.sort(flat, axis=1)
    n = sorted_flat.shape[1]
    lo, hi = int(0.10 * n), max(int(0.90 * n), int(0.10 * n) + 1)
    trimmed = sorted_flat[:, lo:hi].mean(axis=1)
    q75 = np.quantile(flat, 0.75, axis=1)
    top25 = sorted_flat[:, int(0.75 * n):].mean(axis=1)
    top_row2 = np.sort(row, axis=1)[:, -min(2, row.shape[1]):].mean(axis=1)
    top_col4 = np.sort(col, axis=1)[:, -min(4, col.shape[1]):].mean(axis=1)
    consensus_minus_fragmentation = 0.5 * (top_row2 + top_col4) - 0.25 * (row_range + col_range)
    mixture_support = 0.5 * top_row2 + 0.5 * q75 - 0.15 * mat_std
    fragmentation = 0.5 * (row_range + col_range)
    scores = {
        "trimmed_mean_10_90": trimmed,
        "top25_entry_mean": top25,
        "top2_row_mean": top_row2,
        "top4_col_mean": top_col4,
        "consensus_minus_fragmentation": consensus_minus_fragmentation,
        "mixture_support": mixture_support,
        "row_reliability": row_rel,
        "col_reliability": col_rel,
        "joint_reliability": reliability,
    }
    proxies = {
        "mat_std": mat_std,
        "row_mean_range": row_range,
        "col_mean_range": col_range,
        "fragmentation_penalty": fragmentation,
        "row_effective_count": row_eff,
        "col_effective_count": col_eff,
        "row_reliability": row_rel,
        "col_reliability": col_rel,
        "joint_reliability": reliability,
        "top2_row_mean": top_row2,
        "top4_col_mean": top_col4,
        "mixture_support": mixture_support,
        "consensus_minus_fragmentation": consensus_minus_fragmentation,
    }
    return scores, proxies


def build_layer0_scores(mat_mean, margin, cand, proxies, grid_lambdas, grid_mus, grid_alphas, grid_betas):
    """Fixed formula layer-0 scores. No learned parameters."""
    z_mat = robust_z(mat_mean)
    z_margin = robust_z(margin)
    z_cmf = robust_z(cand["consensus_minus_fragmentation"])
    z_mix = robust_z(cand["mixture_support"])
    z_frag = robust_z(proxies["fragmentation_penalty"])
    z_rel = robust_z(proxies["joint_reliability"])
    scores = {
        "L0_identity_mat_mean": mat_mean,
        "L0_identity_margin": margin,
        "L0_cmf_only": cand["consensus_minus_fragmentation"],
        "L0_mix_only": cand["mixture_support"],
    }
    meta = {
        "L0_identity_mat_mean": {"family": "baseline", "formula": "mat_mean"},
        "L0_identity_margin": {"family": "baseline", "formula": "margin"},
        "L0_cmf_only": {"family": "readout", "formula": "consensus_minus_fragmentation"},
        "L0_mix_only": {"family": "readout", "formula": "mixture_support"},
    }
    for lam in grid_lambdas:
        for mu in grid_mus:
            name = f"L0_affine_cmf_l{lam:g}_m{mu:g}"
            scores[name] = z_mat + lam * z_cmf - mu * z_frag
            meta[name] = {"family": "affine_cmf", "lambda_cmf": lam, "mu_fragmentation": mu, "formula": "z(mat_mean)+lambda*z(cmf)-mu*z(fragmentation)"}
            name2 = f"L0_affine_mix_l{lam:g}_m{mu:g}"
            scores[name2] = z_mat + lam * z_mix - mu * z_frag
            meta[name2] = {"family": "affine_mix", "lambda_mix": lam, "mu_fragmentation": mu, "formula": "z(mat_mean)+lambda*z(mixture_support)-mu*z(fragmentation)"}
    for alpha in grid_alphas:
        for beta in grid_betas:
            gate = sigmoid(alpha * (z_cmf - beta * z_frag))
            # Conservative gate: only promote cmf where gate says consensus is reliable; penalize fragmentation where gate is low.
            for lam in grid_lambdas:
                for mu in grid_mus:
                    name = f"L0_sigmoid_gate_a{alpha:g}_b{beta:g}_l{lam:g}_m{mu:g}"
                    scores[name] = z_mat + lam * gate * z_cmf - mu * (1.0 - gate) * z_frag
                    meta[name] = {"family": "sigmoid_fixed_gate", "alpha": alpha, "beta": beta, "lambda_cmf": lam, "mu_fragmentation": mu, "formula": "z(mat_mean)+lambda*sigmoid(alpha*(z(cmf)-beta*z(fragmentation)))*z(cmf)-mu*(1-gate)*z(fragmentation)"}
            name_rel = f"L0_reliability_gate_a{alpha:g}_b{beta:g}"
            rel_gate = sigmoid(alpha * (z_rel + z_cmf - beta * z_frag))
            scores[name_rel] = z_mat * rel_gate + z_margin * (1.0 - rel_gate)
            meta[name_rel] = {"family": "reliability_blend", "alpha": alpha, "beta": beta, "formula": "rel_gate*z(mat_mean)+(1-rel_gate)*z(margin)"}
    return scores, meta


def summarize_values(values, nodes):
    nodes = list(map(int, nodes))
    if not nodes:
        return {"count": 0}
    arr = np.asarray(nodes, dtype=np.int64)
    return {"count": int(len(arr)), "mean": float(np.mean(values[arr])), "std": float(np.std(values[arr])), "min": float(np.min(values[arr])), "max": float(np.max(values[arr]))}


def category_proxy_summary(categories, proxy_dict):
    return {cat: {name: summarize_values(vals, nodes) for name, vals in proxy_dict.items()} for cat, nodes in categories.items()}


def effect_delta(summary, a, b, field):
    da = summary.get(a, {}).get(field, {})
    db = summary.get(b, {}).get(field, {})
    if da.get("count", 0) == 0 or db.get("count", 0) == 0:
        return None
    return float(da["mean"] - db["mean"])


def run_one(cli_args, variant, seed, device, strategy_meta):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = copy.deepcopy(BASE_DEFAULTS)
    cfg.update(VARIANTS[variant]["changes"])
    cfg.update(vars(cli_args))
    cfg.update({"variant": variant, "device": int(device), "seed": int(seed), "data_split_seed": int(seed), "strict_sequential": True})
    v_args = argparse.Namespace(**cfg)

    root = Path(v_args.project_root).expanduser().resolve()
    sys.path.insert(0, str(root))
    os.chdir(str(root))
    from utils import load_mat, preprocess_features, normalize_adj  # noqa: E402
    from VecGAD import VecGAD  # noqa: E402

    device_obj = torch.device(f"cuda:{device}" if torch.cuda.is_available() and int(device) >= 0 else "cpu")
    print(json.dumps({"stage": "seed_start", "probe": "layer0_fixed_formula_gate", "variant": variant, "seed": seed, "device": str(device_obj), "data_split_seed": seed}, ensure_ascii=False), flush=True)
    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat(v_args.dataset, v_args.train_rate, v_args.val_rate, args=v_args)
    features_np = to_dense_features(v_args.dataset, features, preprocess_features)
    labels_np = np.asarray(ano_label).reshape(-1).astype(int)
    normal_idx = np.asarray(normal_for_train_idx, dtype=np.int64)
    idx_test = np.asarray(idx_test, dtype=np.int64)
    fp = split_fingerprint(labels_np, idx_train, idx_val, idx_test, normal_idx)
    assert fp["normal_for_train_anom_count"] == 0, "Data leakage: normal_for_train_idx contains anomalies"

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
    direct_mat_mean = mat.mean(axis=(1, 2))
    formula_diff = float(np.max(np.abs(np.asarray(mat_mean) - np.asarray(direct_mat_mean))))
    candidate_scores, proxy_features = candidate_readouts(mat)
    l0_scores, l0_meta = build_layer0_scores(
        mat_mean, margin, candidate_scores, proxy_features,
        parse_floats(v_args.grid_lambdas), parse_floats(v_args.grid_mus), parse_floats(v_args.grid_alphas), parse_floats(v_args.grid_betas)
    )
    strategy_meta.update(l0_meta)
    all_scores = {"margin": margin, "mat_mean": mat_mean, **candidate_scores, **l0_scores, "degree": meta["degree"], "rejection": meta["rejection"], "residual_norm": meta["residual_norm"]}
    metrics = metric_block(labels_np, idx_test, all_scores, base_name="margin")

    k = max(1, int(np.sum(labels_np[idx_test] == 1)))
    margin_top = top_set(idx_test, margin, k)
    mat_top = top_set(idx_test, mat_mean, k)
    y = labels_np
    categories = {
        "rescued_anomalies_mat_only_true_positive": sorted([n for n in (mat_top - margin_top) if y[n] == 1], key=lambda n: -(mat_mean[n] - margin[n])),
        "introduced_false_positives_mat_only_normal": sorted([n for n in (mat_top - margin_top) if y[n] == 0], key=lambda n: -(mat_mean[n] - margin[n])),
        "lost_anomalies_margin_only_true_positive": sorted([n for n in (margin_top - mat_top) if y[n] == 1], key=lambda n: (mat_mean[n] - margin[n])),
        "removed_false_positives_margin_only_normal": sorted([n for n in (margin_top - mat_top) if y[n] == 0], key=lambda n: (mat_mean[n] - margin[n])),
    }

    category_summary = category_proxy_summary(categories, {"margin": margin, "mat_mean": mat_mean, **proxy_features, "degree": meta["degree"], "rejection": meta["rejection"], "residual_norm": meta["residual_norm"]})

    layer0_tradeoff = {}
    for name, score in l0_scores.items():
        top = top_set(idx_test, score, k)
        auc, ap = safe_auc_ap(labels_np, score, idx_test)
        layer0_tradeoff[name] = {
            "auc": auc,
            "ap": ap,
            "spearman_vs_margin": safe_spearman(score[idx_test], margin[idx_test]),
            "spearman_vs_mat_mean": safe_spearman(score[idx_test], mat_mean[idx_test]),
            "topk_overlap_with_margin": len(top & margin_top) / max(1, k),
            "topk_overlap_with_mat_mean": len(top & mat_top) / max(1, k),
            "retained_rescued_anomalies": len(top & set(categories["rescued_anomalies_mat_only_true_positive"])),
            "recovered_lost_anomalies": len(top & set(categories["lost_anomalies_margin_only_true_positive"])),
            "reintroduced_removed_false_positives": len(top & set(categories["removed_false_positives_margin_only_normal"])),
            "retained_introduced_false_positives": len(top & set(categories["introduced_false_positives_mat_only_normal"])),
        }

    effect_size_map = {}
    for f in ["mat_std", "row_mean_range", "col_mean_range", "fragmentation_penalty", "row_effective_count", "col_effective_count", "joint_reliability", "mixture_support", "consensus_minus_fragmentation", "degree", "rejection", "residual_norm"]:
        effect_size_map[f] = {
            "lost_minus_removed_fp": effect_delta(category_summary, "lost_anomalies_margin_only_true_positive", "removed_false_positives_margin_only_normal", f),
            "rescued_minus_introduced_fp": effect_delta(category_summary, "rescued_anomalies_mat_only_true_positive", "introduced_false_positives_mat_only_normal", f),
        }

    best = max(layer0_tradeoff.items(), key=lambda kv: kv[1]["auc"] if kv[1]["auc"] is not None else -1)
    print(json.dumps({"stage": "seed_done", "seed": seed, "margin_auc": metrics["margin"]["auc"], "mat_mean_auc": metrics["mat_mean"]["auc"], "best_layer0": best[0], "best_auc": best[1]["auc"]}, ensure_ascii=False), flush=True)
    return {
        "variant": variant,
        "report_codename": VARIANTS[variant]["report_codename"],
        "seed": int(seed),
        "device": int(device),
        "effective_config": cfg,
        "split_fingerprint": fp,
        "formula_check": {"mat_mean_equals_direct_matrix_mean_max_abs_diff": formula_diff, "mat_shape": list(np.asarray(mat).shape)},
        "topk_protocol": "K equals number of test anomalies; labels diagnostic-only for oracle categories and layer-0 scoring evaluation",
        "k_anomaly_count": int(k),
        "num_test": int(len(idx_test)),
        "metrics": metrics,
        "oracle_category_counts": {k: len(v) for k, v in categories.items()},
        "category_effect_size_map": effect_size_map,
        "layer0_tradeoff": layer0_tradeoff,
        "anti_shortcut_correlations": {name: {"degree": safe_spearman(score[idx_test], meta["degree"][idx_test]), "rejection": safe_spearman(score[idx_test], meta["rejection"][idx_test]), "residual_norm": safe_spearman(score[idx_test], meta["residual_norm"][idx_test])} for name, score in l0_scores.items()},
    }


def parse_floats(s):
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def summarize(rows, strategy_meta):
    if not rows:
        return {"n_rows": 0}
    strategies = sorted(rows[0]["layer0_tradeoff"].keys())
    cats = sorted(rows[0]["oracle_category_counts"].keys())
    out = {
        "n_rows": len(rows),
        "margin_auc": mean_std([r["metrics"]["margin"]["auc"] for r in rows]),
        "mat_mean_auc": mean_std([r["metrics"]["mat_mean"]["auc"] for r in rows]),
        "mat_mean_ap": mean_std([r["metrics"]["mat_mean"]["ap"] for r in rows]),
        "mat_mean_formula_max_abs_diff": mean_std([r["formula_check"]["mat_mean_equals_direct_matrix_mean_max_abs_diff"] for r in rows]),
        "oracle_category_counts": {c: mean_std([r["oracle_category_counts"][c] for r in rows]) for c in cats},
        "layer0_auc": {s: mean_std([r["layer0_tradeoff"][s]["auc"] for r in rows]) for s in strategies},
        "layer0_ap": {s: mean_std([r["layer0_tradeoff"][s]["ap"] for r in rows]) for s in strategies},
        "layer0_tradeoff": {},
        "strategy_meta": {s: strategy_meta.get(s, {}) for s in strategies},
        "split_fingerprints": {str(r["seed"]): r["split_fingerprint"] for r in rows},
    }
    for s in strategies:
        out["layer0_tradeoff"][s] = {
            "recovered_lost_anomalies": mean_std([r["layer0_tradeoff"][s]["recovered_lost_anomalies"] for r in rows]),
            "reintroduced_removed_false_positives": mean_std([r["layer0_tradeoff"][s]["reintroduced_removed_false_positives"] for r in rows]),
            "retained_rescued_anomalies": mean_std([r["layer0_tradeoff"][s]["retained_rescued_anomalies"] for r in rows]),
            "retained_introduced_false_positives": mean_std([r["layer0_tradeoff"][s]["retained_introduced_false_positives"] for r in rows]),
            "spearman_vs_margin": mean_std([r["layer0_tradeoff"][s]["spearman_vs_margin"] for r in rows]),
            "spearman_vs_mat_mean": mean_std([r["layer0_tradeoff"][s]["spearman_vs_mat_mean"] for r in rows]),
            "topk_overlap_with_mat_mean": mean_std([r["layer0_tradeoff"][s]["topk_overlap_with_mat_mean"] for r in rows]),
        }
    leaderboard = []
    mat_auc = out["mat_mean_auc"]["mean"] if out["mat_mean_auc"] else None
    mat_ap = out["mat_mean_ap"]["mean"] if out["mat_mean_ap"] else None
    for s in strategies:
        auc = out["layer0_auc"][s]["mean"] if out["layer0_auc"][s] else None
        ap = out["layer0_ap"][s]["mean"] if out["layer0_ap"][s] else None
        rec = out["layer0_tradeoff"][s]["recovered_lost_anomalies"]["mean"] if out["layer0_tradeoff"][s]["recovered_lost_anomalies"] else 0
        bad = out["layer0_tradeoff"][s]["reintroduced_removed_false_positives"]["mean"] if out["layer0_tradeoff"][s]["reintroduced_removed_false_positives"] else 0
        rho = out["layer0_tradeoff"][s]["spearman_vs_mat_mean"]["mean"] if out["layer0_tradeoff"][s]["spearman_vs_mat_mean"] else None
        shortcut = mean_std([abs(v) for r in rows for v in r["anti_shortcut_correlations"][s].values() if v is not None])
        leaderboard.append({
            "strategy": s,
            "family": strategy_meta.get(s, {}).get("family"),
            "auc_mean": auc,
            "ap_mean": ap,
            "delta_auc_vs_mat_mean": None if auc is None or mat_auc is None else float(auc - mat_auc),
            "delta_ap_vs_mat_mean": None if ap is None or mat_ap is None else float(ap - mat_ap),
            "lost_recovery_minus_fp_reintro": float(rec - bad),
            "spearman_vs_mat_mean_mean": rho,
            "abs_shortcut_corr_mean": shortcut["mean"] if shortcut else None,
        })
    out["diagnostic_leaderboard"] = sorted(leaderboard, key=lambda x: ((x["auc_mean"] if x["auc_mean"] is not None else -1), x["lost_recovery_minus_fp_reintro"]), reverse=True)
    # Conservative decision: require near-mat_mean AUC while reducing FP reintro cost or adding non-identical ordering.
    best = out["diagnostic_leaderboard"][0]
    if best["auc_mean"] is not None and mat_auc is not None and best["auc_mean"] >= mat_auc - 0.003:
        if best["spearman_vs_mat_mean_mean"] is None or best["spearman_vs_mat_mean_mean"] < 0.995:
            out["decision"] = "LAYER0_FIXED_GATE_HAS_USABLE_PROXY_SIGNAL_REVIEW_REQUIRED"
        else:
            out["decision"] = "LAYER0_NEAR_BASELINE_BUT_MOSTLY_MONOTONIC_REWRITE"
    else:
        out["decision"] = "LAYER0_FIXED_GATE_NOT_READY_USE_PROXY_ANALYSIS_ONLY"
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
    ap.add_argument("--grid_lambdas", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--grid_mus", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--grid_alphas", default="0.5,1.0,2.0")
    ap.add_argument("--grid_betas", default="0.5,1.0,1.5")
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

    protocol = {
        "type": "runner-registered pure probe; Layer-0 fixed-formula gate; no training; no learned weights",
        "label_boundary": "labels diagnostic-only for AUC/AP and oracle top-K boundary categories; labels are not used for fitting, thresholding, early stopping, or formula parameter updates",
        "parameter_boundary": "grid parameters are predeclared mechanism scans; the leaderboard is diagnostic, not a deployable hyperparameter-selection claim",
        "formula_boundary": "mat_mean = response_matrix.mean(axis=(1,2)); consensus_minus_fragmentation = 0.5*(top2_row_mean+top4_col_mean)-0.25*(row_range+col_range)",
        "critical_fix": "force data_split_seed=seed and run seeds sequentially to avoid global random.shuffle cross-thread drift",
    }

    def compact_partial(v):
        if not rows_by_variant[v]:
            return {"n_rows": 0}
        s = summarize(rows_by_variant[v], strategy_meta)
        return {
            "n_rows": s["n_rows"],
            "margin_auc": s["margin_auc"],
            "mat_mean_auc": s["mat_mean_auc"],
            "decision": s.get("decision"),
            "top5": s.get("diagnostic_leaderboard", [])[:5],
        }

    def snapshot(status="running"):
        atomic_write_json(args.progress_out, {"status": status, "probe": "cleg3_layer0_fixed_formula_gate_probe", "done": done, "total": total, "variants": variants, "seeds": seeds, "sequential": True, "partial": {v: compact_partial(v) for v in variants}, "errors": errors[-5:], "elapsed_sec": time.time() - start})

    snapshot("running")
    for variant in variants:
        for i, seed in enumerate(seeds):
            device = devices[i % len(devices)]
            try:
                row = run_one(args, variant, seed, device, strategy_meta)
                rows_by_variant[variant].append(row)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(tb, flush=True)
                errors.append({"variant": variant, "seed": seed, "device": device, "error": repr(e), "traceback": tb[-4000:]})
            finally:
                done += 1
                snapshot("running")

    variant_summaries = []
    for variant in variants:
        rows = sorted(rows_by_variant[variant], key=lambda r: r["seed"])
        variant_summaries.append({"variant": variant, "report_codename": VARIANTS[variant]["report_codename"], "definition": VARIANTS[variant]["definition"], "changes_from_base": VARIANTS[variant]["changes"], "aggregate": summarize(rows, strategy_meta), "rows": rows})
    payload = {"status": "finished" if not errors else "finished_with_errors", "probe": "cleg3_layer0_fixed_formula_gate_probe", "protocol": protocol, "dataset": args.dataset, "seeds": seeds, "variants": variants, "devices": devices, "config": vars(args), "variant_summaries": variant_summaries, "errors": errors, "elapsed_sec": time.time() - start}
    atomic_write_json(args.out, payload)
    snapshot(payload["status"])
    print(json.dumps({"stage": "probe_done", "status": payload["status"], "done": done, "total": total, "out": args.out}, ensure_ascii=False), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
