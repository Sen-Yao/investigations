#!/usr/bin/env python3
"""Pseudo-vs-real anomaly distribution alignment probe for DualRefGAD.

This is a diagnostic script, not a method. It reuses the fixed C-LEG3 / old_exact
response-matrix regime and audits whether the V0 pseudo matrices generated from
known normals occupy the same region as true anomalies in response-matrix feature
space.

Labels are used only to form diagnostic real-anomaly/test-normal groups.
"""
import argparse
import copy
import json
import os
import queue
import random
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Reuse the audited Route25/C-LEG3 helper implementation from the prior investigation.
ROUTE25_SCRIPT_DIR = Path("/home/openclawvm/investigations/nexus/2026-05-19-dualrefgad-route25-matrix-autoencoder/experiments/scripts")
sys.path.insert(0, str(ROUTE25_SCRIPT_DIR))
from route25_leg3_response_matrix_decomposition_probe import BASE_DEFAULTS, VARIANTS, parse_ints  # noqa: E402
from route25_matrix_autoencoder_probe import (  # noqa: E402
    NormalModel,
    build_descriptor,
    build_tokens,
    encode_tokens_batched,
    response_matrix_from_embeddings,
    select_refs,
    set_seed,
)


def atomic_write_json(path, payload):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def mean_std(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
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


def _clip01(x):
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def make_pseudo(norm_x, mu, sd, rng, noise_scale=1.0, tail_boost=1.0, strategy="v0_mixed"):
    """Generate pseudo response-matrix vectors from known normals without anomaly labels.

    Strategies are deliberately simple and diagnostic-only.  They perturb the
    response-matrix feature vector after the fixed C-LEG3 matrix has been built;
    true anomaly labels are not used in generation.
    """
    n, d = norm_x.shape
    base = norm_x.copy().astype(np.float32)
    z_sd = sd.reshape(1, d).astype(np.float32)
    center = mu.reshape(1, d).astype(np.float32)
    pseudo = base.copy()

    def sparse_indices(frac_low, frac_high):
        lo = max(1, int(d * frac_low))
        hi = max(lo + 1, int(d * frac_high))
        return int(rng.integers(lo, hi + 1))

    strategy = (strategy or "v0_mixed").strip()

    if strategy == "v0_mixed":
        # Original V0 protocol: mixture of dense shift, sparse upper-tail, and mixed boost/suppression.
        typ = rng.integers(0, 3, size=n)
        for i in range(n):
            if typ[i] == 0:
                pseudo[i] = base[i] + rng.uniform(0.35, 0.90) * noise_scale * z_sd.reshape(-1)
            elif typ[i] == 1:
                k = int(rng.integers(max(4, d // 8), max(5, d // 3)))
                idx = rng.choice(d, size=k, replace=False)
                pseudo[i, idx] = base[i, idx] + rng.uniform(0.8, 1.8) * tail_boost * z_sd.reshape(-1)[idx]
            else:
                k_hi = int(rng.integers(max(4, d // 10), max(5, d // 4)))
                k_lo = int(rng.integers(max(4, d // 10), max(5, d // 4)))
                hi = rng.choice(d, size=k_hi, replace=False)
                lo = rng.choice(d, size=k_lo, replace=False)
                pseudo[i, hi] = base[i, hi] + rng.uniform(0.8, 1.6) * z_sd.reshape(-1)[hi]
                pseudo[i, lo] = center.reshape(-1)[lo] - rng.uniform(0.8, 1.6) * z_sd.reshape(-1)[lo]
        return _clip01(pseudo)

    if strategy == "dense_soft_shift":
        # Low-amplitude dense deviation: tests whether V0 was simply too far OOD.
        amp = rng.uniform(0.12, 0.35, size=(n, 1)).astype(np.float32) * noise_scale
        return _clip01(base + amp * z_sd)

    if strategy == "dense_hard_shift":
        # Strong dense deviation: positive control for trivially separable pseudo anomalies.
        amp = rng.uniform(0.65, 1.25, size=(n, 1)).astype(np.float32) * noise_scale
        return _clip01(base + amp * z_sd)

    if strategy == "sparse_tail_boost":
        # Only boost a sparse set of high-response candidates; keeps most entries normal-like.
        for i in range(n):
            k = sparse_indices(0.08, 0.22)
            idx = rng.choice(d, size=k, replace=False)
            pseudo[i, idx] = base[i, idx] + rng.uniform(0.55, 1.35) * tail_boost * z_sd.reshape(-1)[idx]
        return _clip01(pseudo)

    if strategy == "mixed_signed":
        # Symmetric positive/negative perturbation: tests whether sign-mixed distortion destroys real direction.
        for i in range(n):
            k = sparse_indices(0.16, 0.36)
            idx = rng.choice(d, size=k, replace=False)
            signs = rng.choice([-1.0, 1.0], size=k).astype(np.float32)
            pseudo[i, idx] = base[i, idx] + signs * rng.uniform(0.45, 1.10) * noise_scale * z_sd.reshape(-1)[idx]
        return _clip01(pseudo)

    if strategy == "normal_shell":
        # Move each normal away from the normal mean by a controlled radial factor in standardized space.
        z = (base - center) / (z_sd + 1e-6)
        norm = np.linalg.norm(z, axis=1, keepdims=True) + 1e-6
        direction = z / norm
        target_radius = np.quantile(norm, 0.75) + rng.uniform(0.25, 0.85, size=(n, 1)).astype(np.float32)
        return _clip01(center + direction * target_radius * z_sd)

    if strategy == "tail_quantile_match":
        # Replace a small fraction of entries by upper-tail values sampled from the known-normal entry distribution.
        q75 = np.quantile(base, 0.75, axis=0).astype(np.float32)
        q95 = np.quantile(base, 0.95, axis=0).astype(np.float32)
        for i in range(n):
            k = sparse_indices(0.10, 0.25)
            idx = rng.choice(d, size=k, replace=False)
            alpha = rng.uniform(0.2, 1.0, size=k).astype(np.float32)
            pseudo[i, idx] = q75[idx] * (1 - alpha) + q95[idx] * alpha
        return _clip01(pseudo)

    raise ValueError(f"Unknown pseudo strategy: {strategy}")

def build_readout_scores(mat, margin):
    x = mat.reshape(mat.shape[0], -1).astype(np.float64)
    row_mean = mat.mean(axis=2)
    col_mean = mat.mean(axis=1)
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    q10, q25, q50, q75, q90 = [np.quantile(x, q, axis=1) for q in [0.10, 0.25, 0.50, 0.75, 0.90]]
    top8 = np.sort(x, axis=1)[:, -8:].mean(axis=1)
    top16 = np.sort(x, axis=1)[:, -16:].mean(axis=1)
    trim10 = np.sort(x, axis=1)[:, 6:-6].mean(axis=1)
    lo = np.quantile(x, 0.10, axis=1, keepdims=True)
    hi = np.quantile(x, 0.90, axis=1, keepdims=True)
    winsor10 = np.clip(x, lo, hi).mean(axis=1)
    return {
        "margin": margin.astype(np.float64),
        "mat_mean": mean,
        "mat_std": std,
        "q10": q10,
        "q25": q25,
        "q50_median": q50,
        "q75": q75,
        "q90": q90,
        "top8_mean": top8,
        "top16_mean": top16,
        "trim10_mean": trim10,
        "winsor10_mean": winsor10,
        "mean_q75_blend": 0.5 * mean + 0.5 * q75,
        "mean_top16_blend": 0.5 * mean + 0.5 * top16,
        "row_top2_mean": np.sort(row_mean, axis=1)[:, -2:].mean(axis=1),
        "col_top4_mean": np.sort(col_mean, axis=1)[:, -4:].mean(axis=1),
        "flat_l2": np.linalg.norm(x, axis=1),
        "flat_min": x.min(axis=1),
        "flat_max": x.max(axis=1),
    }


def response_features(mat, margin, normal_train_idx=None):
    scores = build_readout_scores(mat, margin)
    x = mat.reshape(mat.shape[0], -1).astype(np.float64)
    # Normal-manifold residual in response-matrix space, fitted on known normals only.
    if normal_train_idx is not None and len(normal_train_idx) > 8:
        scaler = StandardScaler()
        xs_train = scaler.fit_transform(x[normal_train_idx])
        xs_all = scaler.transform(x)
        ncomp = int(min(16, xs_train.shape[0] - 1, xs_train.shape[1]))
        if ncomp > 0:
            pca = PCA(n_components=ncomp, svd_solver="randomized", random_state=0)
            pca.fit(xs_train)
            rec = pca.inverse_transform(pca.transform(xs_all))
            scores["response_pca_residual"] = np.mean((xs_all - rec) ** 2, axis=1)
            scores["response_z_l2"] = np.linalg.norm(xs_all, axis=1)
    return scores


def js_hist(a, b, bins=50):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return None
    lo = float(min(np.min(a), np.min(b)))
    hi = float(max(np.max(a), np.max(b)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0
    ha, edges = np.histogram(a, bins=bins, range=(lo, hi), density=False)
    hb, _ = np.histogram(b, bins=edges, density=False)
    ha = ha.astype(np.float64) + 1e-12
    hb = hb.astype(np.float64) + 1e-12
    ha /= ha.sum()
    hb /= hb.sum()
    return float(jensenshannon(ha, hb, base=2.0) ** 2)


def compare_1d(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return None
    return {
        "mean_delta_b_minus_a": float(np.mean(b) - np.mean(a)),
        "std_ratio_b_over_a": float((np.std(b) + 1e-12) / (np.std(a) + 1e-12)),
        "wasserstein": float(wasserstein_distance(a, b)),
        "ks_stat": float(ks_2samp(a, b).statistic),
        "js_hist": js_hist(a, b),
        "a": {"mean": float(np.mean(a)), "std": float(np.std(a)), "q25": float(np.quantile(a, 0.25)), "q50": float(np.quantile(a, 0.50)), "q75": float(np.quantile(a, 0.75))},
        "b": {"mean": float(np.mean(b)), "std": float(np.std(b)), "q25": float(np.quantile(b, 0.25)), "q50": float(np.quantile(b, 0.50)), "q75": float(np.quantile(b, 0.75))},
    }


def group_values(feature_map, idx_or_values):
    out = {}
    for name, v in feature_map.items():
        arr = np.asarray(v, dtype=np.float64)
        if isinstance(idx_or_values, np.ndarray) and idx_or_values.dtype == bool:
            out[name] = arr[idx_or_values]
        else:
            out[name] = arr[np.asarray(idx_or_values, dtype=np.int64)]
    return out


def summarize_group(values):
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "q10": float(np.quantile(v, 0.10)), "q50": float(np.quantile(v, 0.50)), "q90": float(np.quantile(v, 0.90))} for k, v in values.items() if len(v) > 0}


def compare_groups(group_a, group_b):
    out = {}
    for k in sorted(set(group_a) & set(group_b)):
        out[k] = compare_1d(group_a[k], group_b[k])
    return out


def alignment_rank(compare_real_pseudo, compare_normal_real, compare_normal_pseudo):
    rows = []
    for name, rp in compare_real_pseudo.items():
        nr = compare_normal_real.get(name) or {}
        npair = compare_normal_pseudo.get(name) or {}
        if rp is None:
            continue
        rows.append({
            "feature": name,
            "pseudo_real_wasserstein": rp.get("wasserstein"),
            "normal_real_wasserstein": nr.get("wasserstein"),
            "normal_pseudo_wasserstein": npair.get("wasserstein"),
            "pseudo_real_ks": rp.get("ks_stat"),
            "pseudo_real_js": rp.get("js_hist"),
            "pseudo_minus_real_mean": rp.get("mean_delta_b_minus_a"),
            "pseudo_std_over_real": rp.get("std_ratio_b_over_a"),
        })
    rows.sort(key=lambda r: (-(r["pseudo_real_wasserstein"] or 0.0)))
    return rows


def gate_decisions(summary):
    """Four gates for pseudo generation quality.

    G1 alignment: pseudo-real distance should not exceed the normal-real gap on key readouts.
    G2 difficulty: pseudo should not be much farther from known normals than real anomalies are.
    G3 direction: pseudo shift should have the same sign as real-anomaly shift for key readouts.
    G4 training relevance: the best real-anomaly readout should remain near the scalar positive control.
    """
    cmp = summary.get("comparison_summary", {})
    rp = cmp.get("real_anomaly_vs_pseudo_anomaly", {})
    nr = cmp.get("known_normal_vs_real_anomaly", {})
    npair = cmp.get("known_normal_vs_pseudo_anomaly", {})
    key_feats = ["mat_mean", "q75", "q90", "top16_mean", "mean_top16_blend", "response_pca_residual", "response_z_l2"]
    feat_rows = []
    align_pass = []
    difficulty_pass = []
    direction_pass = []
    for f in key_feats:
        if f not in rp or f not in nr or f not in npair:
            continue
        def m(src, field):
            v = src.get(f, {}).get(field)
            return v.get("mean") if isinstance(v, dict) else None
        pr_w = m(rp, "wasserstein")
        nr_w = m(nr, "wasserstein")
        np_w = m(npair, "wasserstein")
        nr_delta = m(nr, "mean_delta_b_minus_a")  # real - normal
        np_delta = m(npair, "mean_delta_b_minus_a")  # pseudo - normal
        if pr_w is None or nr_w is None or np_w is None:
            continue
        ar = float(pr_w / (nr_w + 1e-12))
        dr = float(np_w / (nr_w + 1e-12))
        same_dir = bool(nr_delta is not None and np_delta is not None and np.sign(nr_delta) == np.sign(np_delta))
        feat_rows.append({
            "feature": f,
            "pseudo_real_over_normal_real": ar,
            "normal_pseudo_over_normal_real": dr,
            "real_minus_normal_mean": nr_delta,
            "pseudo_minus_normal_mean": np_delta,
            "same_direction": same_dir,
        })
        align_pass.append(ar <= 1.10)
        difficulty_pass.append(0.55 <= dr <= 1.80)
        direction_pass.append(same_dir)
    aucs = []
    for name, vals in summary.get("diagnostic_auc_ap_summary", {}).items():
        auc = vals.get("auc", {}).get("mean") if vals.get("auc") else None
        ap = vals.get("ap", {}).get("mean") if vals.get("ap") else None
        if auc is not None:
            aucs.append({"feature": name, "auc": auc, "ap": ap})
    aucs.sort(key=lambda x: -x["auc"])
    mat_auc = summary.get("diagnostic_auc_ap_summary", {}).get("mat_mean", {}).get("auc", {}).get("mean")
    best_auc = aucs[0]["auc"] if aucs else None
    return {
        "gate_definitions": {
            "G1_alignment": "On key readouts, pseudo-real Wasserstein / normal-real Wasserstein <= 1.10 for at least half of available key features.",
            "G2_difficulty": "Normal-pseudo distance must be in [0.55, 1.80] × normal-real distance, avoiding too-easy OOD pseudo anomalies and too-weak near-normal pseudo anomalies.",
            "G3_direction": "Pseudo-minus-normal mean shift must have the same sign as real-minus-normal shift for at least half of key features.",
            "G4_readout_relevance": "Best diagnostic AUC among readouts remains within 0.03 of mat_mean or exceeds mat_mean, meaning the fixed matrix regime still contains the known anomaly signal.",
        },
        "feature_gate_rows": feat_rows,
        "G1_alignment_pass": bool(align_pass and np.mean(align_pass) >= 0.5),
        "G2_difficulty_pass": bool(difficulty_pass and np.mean(difficulty_pass) >= 0.5),
        "G3_direction_pass": bool(direction_pass and np.mean(direction_pass) >= 0.5),
        "G4_readout_relevance_pass": bool(best_auc is not None and mat_auc is not None and best_auc >= mat_auc - 0.03),
        "best_auc_features": aucs[:8],
        "mat_mean_auc": mat_auc,
        "decision": None,
    }


def run_one(args, variant, pseudo_strategy, seed, device):
    set_seed(seed)
    rng = np.random.default_rng(seed + 20260522 + (abs(hash(pseudo_strategy)) % 100000))
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
    encoder = VecGAD(features_np.shape[1], v_args.embedding_dim, "prelu", v_args).to(device_obj)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    with torch.no_grad():
        emb = encode_tokens_batched(encoder, token_tensor, device_obj, v_args.encode_batch_size)
    del token_tensor, encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mat, margin = response_matrix_from_embeddings(emb, normal_refs, anom_refs)
    real_features = response_features(mat, margin, normal_idx)

    normals = np.asarray(normal_idx, dtype=np.int64)
    tr_norm, va_norm = train_test_split(normals, test_size=args.normal_val_frac, random_state=seed, shuffle=True)
    x_all = mat.reshape(mat.shape[0], -1).astype(np.float32)
    mu = x_all[tr_norm].mean(axis=0)
    sd = x_all[tr_norm].std(axis=0) + 1e-6

    # Pseudo group: generated from held-out labeled normals, so true anomaly labels are not used.
    source_idx = va_norm
    source_x = x_all[source_idx].astype(np.float32)
    pseudo_x = make_pseudo(source_x, mu, sd, rng, args.pseudo_noise_scale, args.pseudo_tail_boost, args.pseudo_strategy)
    pseudo_mat = pseudo_x.reshape((-1,) + mat.shape[1:])
    pseudo_margin = pseudo_x.max(axis=1) - pseudo_x.min(axis=1)
    pseudo_features = response_features(pseudo_mat, pseudo_margin, np.arange(len(source_idx)))

    test_normal_idx = idx_test[labels_np[idx_test] == 0]
    test_real_anom_idx = idx_test[labels_np[idx_test] == 1]

    groups = {
        "known_normal_val": group_values(real_features, source_idx),
        "test_normal": group_values(real_features, test_normal_idx),
        "real_anomaly": group_values(real_features, test_real_anom_idx),
        "pseudo_anomaly": pseudo_features,
    }
    group_summary = {k: summarize_group(v) for k, v in groups.items()}

    compare_real_pseudo = compare_groups(groups["real_anomaly"], groups["pseudo_anomaly"])
    compare_normal_real = compare_groups(groups["known_normal_val"], groups["real_anomaly"])
    compare_normal_pseudo = compare_groups(groups["known_normal_val"], groups["pseudo_anomaly"])

    metric_auc = {}
    for name, score in real_features.items():
        auc, ap = safe_auc_ap(labels_np, score, idx_test)
        metric_auc[name] = {"auc": auc, "ap": ap}

    row = {
        "variant": variant,
        "pseudo_strategy": pseudo_strategy,
        "seed": int(seed),
        "device": int(device),
        "counts": {
            "num_nodes": int(len(labels_np)),
            "num_test": int(len(idx_test)),
            "num_test_normal": int(len(test_normal_idx)),
            "num_test_real_anomaly": int(len(test_real_anom_idx)),
            "num_labeled_normals": int(len(normal_idx)),
            "num_pseudo": int(len(source_idx)),
            "matrix_shape": list(mat.shape),
        },
        "group_summary": group_summary,
        "comparisons": {
            "real_anomaly_vs_pseudo_anomaly": compare_real_pseudo,
            "known_normal_vs_real_anomaly": compare_normal_real,
            "known_normal_vs_pseudo_anomaly": compare_normal_pseudo,
        },
        "alignment_rank_worst_first": alignment_rank(compare_real_pseudo, compare_normal_real, compare_normal_pseudo)[:12],
        "diagnostic_auc_ap": metric_auc,
        "reference_diagnostics": {
            "anom_ref_anom_ratio_diagnostic": float(np.mean(labels_np[anom_refs] == 1)),
            "normal_ref_anom_ratio_diagnostic": float(np.mean(labels_np[normal_refs] == 1)),
            "test_anom_rate": float(np.mean(labels_np[idx_test] == 1)),
        },
    }
    print(json.dumps({
        "stage": "seed_done",
        "variant": variant,
        "pseudo_strategy": pseudo_strategy,
        "seed": seed,
        "mat_mean_auc": metric_auc.get("mat_mean", {}).get("auc"),
        "worst_alignment": row["alignment_rank_worst_first"][:3],
    }, ensure_ascii=False), flush=True)
    return row


def summarize(rows):
    out = {"n_rows": len(rows)}
    if not rows:
        return out
    # Aggregate AUC/AP.
    metric_names = sorted({m for r in rows for m in r["diagnostic_auc_ap"]})
    out["diagnostic_auc_ap_summary"] = {
        m: {
            "auc": mean_std([r["diagnostic_auc_ap"][m].get("auc") for r in rows]),
            "ap": mean_std([r["diagnostic_auc_ap"][m].get("ap") for r in rows]),
        }
        for m in metric_names
    }
    # Aggregate distribution distances for the most interpretable features.
    cmp_names = ["real_anomaly_vs_pseudo_anomaly", "known_normal_vs_real_anomaly", "known_normal_vs_pseudo_anomaly"]
    out["comparison_summary"] = {}
    for cmp_name in cmp_names:
        out["comparison_summary"][cmp_name] = {}
        feat_names = sorted({f for r in rows for f in r["comparisons"][cmp_name]})
        for f in feat_names:
            vals = [r["comparisons"][cmp_name][f] for r in rows if r["comparisons"][cmp_name].get(f) is not None]
            out["comparison_summary"][cmp_name][f] = {
                "wasserstein": mean_std([v.get("wasserstein") for v in vals]),
                "ks_stat": mean_std([v.get("ks_stat") for v in vals]),
                "js_hist": mean_std([v.get("js_hist") for v in vals]),
                "mean_delta_b_minus_a": mean_std([v.get("mean_delta_b_minus_a") for v in vals]),
                "std_ratio_b_over_a": mean_std([v.get("std_ratio_b_over_a") for v in vals]),
            }
    # Worst pseudo-real misalignment features by mean Wasserstein.
    rp = out["comparison_summary"]["real_anomaly_vs_pseudo_anomaly"]
    worst = []
    for f, v in rp.items():
        w = v["wasserstein"]["mean"] if v.get("wasserstein") else None
        if w is not None:
            worst.append({"feature": f, "pseudo_real_wasserstein_mean": w, "ks_mean": v["ks_stat"]["mean"], "mean_delta_pseudo_minus_real": v["mean_delta_b_minus_a"]["mean"]})
    worst.sort(key=lambda x: -x["pseudo_real_wasserstein_mean"])
    out["worst_pseudo_real_alignment_features"] = worst[:12]

    # Direct decision flags.
    key_feats = ["mat_mean", "q75", "q90", "top16_mean", "mat_std", "response_pca_residual", "response_z_l2"]
    flags = []
    for f in key_feats:
        if f not in rp:
            continue
        pr_w = rp[f]["wasserstein"]["mean"] if rp[f].get("wasserstein") else None
        nr_w = out["comparison_summary"]["known_normal_vs_real_anomaly"].get(f, {}).get("wasserstein", {}).get("mean")
        np_w = out["comparison_summary"]["known_normal_vs_pseudo_anomaly"].get(f, {}).get("wasserstein", {}).get("mean")
        pr_delta = rp[f]["mean_delta_b_minus_a"]["mean"] if rp[f].get("mean_delta_b_minus_a") else None
        flags.append({"feature": f, "pseudo_real_w": pr_w, "normal_real_w": nr_w, "normal_pseudo_w": np_w, "pseudo_minus_real_mean_delta": pr_delta})
    out["key_alignment_flags"] = flags
    out["gates"] = gate_decisions(out)
    passed = [out["gates"].get(k) for k in ["G1_alignment_pass", "G2_difficulty_pass", "G3_direction_pass", "G4_readout_relevance_pass"]]
    out["gates"]["num_passed"] = int(sum(1 for x in passed if x))
    out["gates"]["decision"] = "PSEUDO_STRATEGY_GATE_PASS" if all(passed) else "PSEUDO_STRATEGY_GATE_FAIL"
    out["interpretation_hint"] = "If normal_pseudo distance is much larger than normal_real or pseudo-real distance remains large on high-response features, pseudo anomalies are over-strong / misaligned relative to real anomalies."
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
    ap.add_argument("--normal_val_frac", type=float, default=0.25)
    ap.add_argument("--pseudo_noise_scale", type=float, default=1.0)
    ap.add_argument("--pseudo_tail_boost", type=float, default=1.0)
    ap.add_argument("--pseudo_strategy", default="v0_mixed")
    ap.add_argument("--pseudo_strategies", default="v0_mixed")
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
    pseudo_strategies = [x.strip() for x in args.pseudo_strategies.split(",") if x.strip()] or [args.pseudo_strategy]
    task_q = queue.Queue()
    for v in variants:
        for ps in pseudo_strategies:
            for s in seeds:
                task_q.put((v, ps, s))
    total = task_q.qsize()
    rows_by_strategy = {ps: [] for ps in pseudo_strategies}
    errors = []
    done = 0
    lock = threading.Lock()

    def snapshot(status="running", current=None):
        partial = []
        for ps in pseudo_strategies:
            if rows_by_strategy[ps]:
                partial.append({"pseudo_strategy": ps, "summary": summarize(rows_by_strategy[ps])})
        return {"status": status, "done": done, "total": total, "devices": devices, "current": current, "errors": errors[-5:], "partial": partial, "elapsed_sec": round(time.time() - start, 2)}

    def worker(device):
        nonlocal done
        while True:
            try:
                variant, pseudo_strategy, seed = task_q.get_nowait()
            except queue.Empty:
                return
            current = {"variant": variant, "pseudo_strategy": pseudo_strategy, "seed": seed, "device": int(device)}
            with lock:
                atomic_write_json(args.progress_out, snapshot("running", current))
            try:
                row = run_one(args, variant, pseudo_strategy, seed, int(device))
                with lock:
                    rows_by_strategy[pseudo_strategy].append(row)
                    done += 1
                    atomic_write_json(args.progress_out, snapshot("running", current))
            except Exception as e:
                tb = traceback.format_exc()
                with lock:
                    errors.append({"variant": variant, "pseudo_strategy": pseudo_strategy, "seed": int(seed), "device": int(device), "error": repr(e), "traceback": tb})
                    done += 1
                    atomic_write_json(args.progress_out, snapshot("running", current))
                print(tb, flush=True)
            finally:
                task_q.task_done()

    threads = [threading.Thread(target=worker, args=(d,), daemon=False) for d in devices]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    status = "failed" if errors else "finished"
    payload = {
        "probe": "pseudo_real_distribution_alignment_probe",
        "codename": "C-MiT / V0 pseudo-real alignment audit",
        "protocol": {
            "reference_regime": "C-LEG3 old_exact_080_regime by default",
            "pseudo_generation": "multiple strategies in response-matrix space, generated only from held-out known-normal matrices",
            "pseudo_strategies": pseudo_strategies,
            "diagnostic_labels": "true anomaly labels are used only to form real-anomaly/test-normal diagnostic groups",
            "variant": variants,
            "seeds": seeds,
            "devices": devices,
        },
        "status": status,
        "elapsed_sec": round(time.time() - start, 2),
        "errors": errors,
        "by_strategy": {ps: {"rows": rows_by_strategy[ps], "summary": summarize(rows_by_strategy[ps]) if rows_by_strategy[ps] else None} for ps in pseudo_strategies},
    }
    atomic_write_json(args.out, payload)
    atomic_write_json(args.progress_out, snapshot(status, None))
    print(json.dumps({"stage": "done", "status": status, "out": args.out, "elapsed_sec": payload["elapsed_sec"]}, ensure_ascii=False), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
