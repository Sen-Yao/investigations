#!/usr/bin/env python3
"""C-LEG3 Layer-1 label-free shallow reliability gate probe.

This probe reuses the Layer-0 response-matrix construction and candidate proxy features,
then fits a tiny monotone logistic reliability gate with label-free pseudo anchors.

Scientific boundaries:
- anomaly labels are diagnostic-only for AUC/AP and top-K autopsy;
- labels are never used for fitting, early stopping, pseudo-anchor construction, or hyperparameter selection;
- the scanned Layer-1 settings are predeclared mechanism probes, not deployable label-selected hyperparameters.
"""
from __future__ import annotations

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
import torch.nn.functional as F
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
from cleg3_layer0_fixed_formula_gate_probe import (  # noqa: E402
    atomic_write_json,
    build_layer0_scores,
    candidate_readouts,
    category_proxy_summary,
    effect_delta,
    mean_std,
    parse_floats,
    robust_z,
    safe_auc_ap,
    sha1_ints,
    split_fingerprint,
    top_set,
    to_dense_features,
)

EPS = 1e-9


def sigmoid_np(x):
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def quantile_mask(x, q, side):
    x = np.asarray(x, dtype=np.float64)
    if side == "top":
        return x >= np.quantile(x, 1.0 - q)
    if side == "bottom":
        return x <= np.quantile(x, q)
    raise ValueError(side)


def build_anchor_masks(z_cmf, z_frag, q, qf):
    # pseudo-positive = high consensus and low fragmentation.
    pos = quantile_mask(z_cmf, q, "top") & quantile_mask(z_frag, qf, "bottom")
    # pseudo-negative = low consensus or high fragmentation.
    neg = quantile_mask(z_cmf, q, "bottom") | quantile_mask(z_frag, qf, "top")
    # Avoid overlap if quantiles are tied.
    neg = neg & (~pos)
    return pos, neg


def sample_pairs(pos_idx, neg_idx, max_pairs, seed):
    rng = np.random.default_rng(int(seed))
    pos_idx = np.asarray(pos_idx, dtype=np.int64)
    neg_idx = np.asarray(neg_idx, dtype=np.int64)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return pos_idx[:0], neg_idx[:0]
    n = min(int(max_pairs), int(len(pos_idx) * len(neg_idx)))
    p = rng.choice(pos_idx, size=n, replace=True)
    nn = rng.choice(neg_idx, size=n, replace=True)
    return p, nn


def sample_monotone_pairs(z_cmf, z_frag, max_pairs, seed):
    rng = np.random.default_rng(int(seed) + 1729)
    n = len(z_cmf)
    a = rng.integers(0, n, size=int(max_pairs) * 3)
    b = rng.integers(0, n, size=int(max_pairs) * 3)
    keep = (z_cmf[a] >= z_cmf[b]) & (z_frag[a] <= z_frag[b]) & (a != b)
    hi = a[keep][:int(max_pairs)]
    lo = b[keep][:int(max_pairs)]
    return hi.astype(np.int64), lo.astype(np.int64)


def fit_monotone_logistic_gate(z_cmf, z_rel, z_frag, q, qf, alpha_anchor, alpha_mono, lambda_l2, steps, seed, max_pairs=20000, max_mono_pairs=20000):
    """Fit a tiny constrained reliability gate using no anomaly labels."""
    z_cmf = np.asarray(z_cmf, dtype=np.float32)
    z_rel = np.asarray(z_rel, dtype=np.float32)
    z_frag = np.asarray(z_frag, dtype=np.float32)
    pos_mask, neg_mask = build_anchor_masks(z_cmf, z_frag, q, qf)
    pos_idx = np.where(pos_mask)[0]
    neg_idx = np.where(neg_mask)[0]
    if len(pos_idx) < 5 or len(neg_idx) < 5:
        raise ValueError(f"insufficient pseudo anchors: pos={len(pos_idx)} neg={len(neg_idx)} q={q} qf={qf}")
    p_idx, n_idx = sample_pairs(pos_idx, neg_idx, max_pairs, seed)
    hi_idx, lo_idx = sample_monotone_pairs(z_cmf, z_frag, max_mono_pairs, seed)

    x = torch.tensor(np.stack([z_cmf, z_rel, z_frag], axis=1), dtype=torch.float32)
    p_t = torch.tensor(p_idx, dtype=torch.long)
    n_t = torch.tensor(n_idx, dtype=torch.long)
    hi_t = torch.tensor(hi_idx, dtype=torch.long)
    lo_t = torch.tensor(lo_idx, dtype=torch.long)
    raw = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32))
    opt = torch.optim.Adam([raw], lr=0.05)
    losses = []
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        wc = F.softplus(raw[0])
        wr = F.softplus(raw[1])
        wf = F.softplus(raw[2])
        b = raw[3]
        h = wc * x[:, 0] + wr * x[:, 1] - wf * x[:, 2] + b
        g = torch.sigmoid(h)
        rank_loss = F.softplus(-(h[p_t] - h[n_t])).mean()
        anchor_loss = 0.5 * (F.binary_cross_entropy(g[torch.tensor(pos_idx, dtype=torch.long)], torch.ones(len(pos_idx))) + F.binary_cross_entropy(g[torch.tensor(neg_idx, dtype=torch.long)], torch.zeros(len(neg_idx))))
        if len(hi_t) > 0:
            mono_loss = F.relu(g[lo_t] - g[hi_t]).mean()
        else:
            mono_loss = torch.tensor(0.0)
        l2 = wc * wc + wr * wr + wf * wf + b * b
        loss = rank_loss + float(alpha_anchor) * anchor_loss + float(alpha_mono) * mono_loss + float(lambda_l2) * l2
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        wc = F.softplus(raw[0]).item()
        wr = F.softplus(raw[1]).item()
        wf = F.softplus(raw[2]).item()
        b = raw[3].item()
        h = wc * x[:, 0] + wr * x[:, 1] - wf * x[:, 2] + b
        gate = torch.sigmoid(h).cpu().numpy().astype(np.float64)
    return {
        "gate": gate,
        "weights": {"w_cmf": float(wc), "w_reliability": float(wr), "w_fragmentation": float(wf), "bias": float(b)},
        "anchor_counts": {"positive": int(len(pos_idx)), "negative": int(len(neg_idx)), "rank_pairs": int(len(p_idx)), "mono_pairs": int(len(hi_idx))},
        "loss": {"initial": losses[0] if losses else None, "final": losses[-1] if losses else None, "steps": int(steps)},
    }


def build_layer1_scores(mat_mean, margin, cand, proxies, q_values, qf_values, anchor_weights, mono_weights, l2_values, steps, seed):
    z_mat = robust_z(mat_mean)
    z_margin = robust_z(margin)
    z_cmf = robust_z(cand["consensus_minus_fragmentation"])
    z_rel = robust_z(proxies["joint_reliability"])
    z_frag = robust_z(proxies["fragmentation_penalty"])
    scores = {}
    meta = {}
    diagnostics = {}
    for q in q_values:
        for qf in qf_values:
            for aa in anchor_weights:
                for am in mono_weights:
                    for l2 in l2_values:
                        name = f"L1_lfgate_q{q:g}_qf{qf:g}_aa{aa:g}_am{am:g}_l2{l2:g}"
                        fit = fit_monotone_logistic_gate(z_cmf, z_rel, z_frag, q, qf, aa, am, l2, steps, seed)
                        gate = fit["gate"]
                        scores[name] = gate * z_mat + (1.0 - gate) * z_margin
                        meta[name] = {
                            "family": "label_free_monotone_logistic_gate",
                            "q": float(q), "q_fragmentation": float(qf),
                            "alpha_anchor": float(aa), "alpha_mono": float(am), "lambda_l2": float(l2),
                            "formula": "g*z(mat_mean)+(1-g)*z(margin); g=sigmoid(w_c*z(cmf)+w_r*z(joint_reliability)-w_f*z(fragmentation)+b); constrained w>=0",
                            "label_boundary": "fit uses pseudo reliability anchors only; anomaly labels diagnostic-only after fitting",
                        }
                        diagnostics[name] = {k: v for k, v in fit.items() if k != "gate"}
    return scores, meta, diagnostics


def summarize_values(values, nodes):
    nodes = list(map(int, nodes))
    if not nodes:
        return {"count": 0}
    arr = np.asarray(nodes, dtype=np.int64)
    return {"count": int(len(arr)), "mean": float(np.mean(values[arr])), "std": float(np.std(values[arr])), "min": float(np.min(values[arr])), "max": float(np.max(values[arr]))}


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
    print(json.dumps({"stage": "seed_start", "probe": "layer1_label_free_shallow_gate", "variant": variant, "seed": seed, "device": str(device_obj), "data_split_seed": seed}, ensure_ascii=False), flush=True)
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
    l1_scores, l1_meta, l1_diagnostics = build_layer1_scores(
        mat_mean, margin, candidate_scores, proxy_features,
        parse_floats(v_args.q_values), parse_floats(v_args.qf_values),
        parse_floats(v_args.alpha_anchor_values), parse_floats(v_args.alpha_mono_values),
        parse_floats(v_args.lambda_l2_values), int(v_args.train_steps), int(seed)
    )
    strategy_meta.update(l0_meta)
    strategy_meta.update(l1_meta)
    all_scores = {"margin": margin, "mat_mean": mat_mean, **candidate_scores, **l0_scores, **l1_scores, "degree": meta["degree"], "rejection": meta["rejection"], "residual_norm": meta["residual_norm"]}
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

    strategies = {**l0_scores, **l1_scores}
    tradeoff = {}
    anti = {}
    for name, score in strategies.items():
        top = top_set(idx_test, score, k)
        auc, ap = safe_auc_ap(labels_np, score, idx_test)
        tradeoff[name] = {
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
        anti[name] = {"degree": safe_spearman(score[idx_test], meta["degree"][idx_test]), "rejection": safe_spearman(score[idx_test], meta["rejection"][idx_test]), "residual_norm": safe_spearman(score[idx_test], meta["residual_norm"][idx_test])}

    effect_size_map = {}
    for f in ["mat_std", "row_mean_range", "col_mean_range", "fragmentation_penalty", "row_effective_count", "col_effective_count", "joint_reliability", "mixture_support", "consensus_minus_fragmentation", "degree", "rejection", "residual_norm"]:
        effect_size_map[f] = {
            "lost_minus_removed_fp": effect_delta(category_summary, "lost_anomalies_margin_only_true_positive", "removed_false_positives_margin_only_normal", f),
            "rescued_minus_introduced_fp": effect_delta(category_summary, "rescued_anomalies_mat_only_true_positive", "introduced_false_positives_mat_only_normal", f),
        }

    best_l1 = max(l1_scores.keys(), key=lambda n: tradeoff[n]["auc"] if tradeoff[n]["auc"] is not None else -1)
    print(json.dumps({"stage": "seed_done", "seed": seed, "margin_auc": metrics["margin"]["auc"], "mat_mean_auc": metrics["mat_mean"]["auc"], "best_layer1": best_l1, "best_l1_auc": tradeoff[best_l1]["auc"]}, ensure_ascii=False), flush=True)
    return {
        "variant": variant,
        "report_codename": VARIANTS[variant]["report_codename"],
        "seed": int(seed),
        "device": int(device),
        "effective_config": cfg,
        "split_fingerprint": fp,
        "formula_check": {"mat_mean_equals_direct_matrix_mean_max_abs_diff": formula_diff, "mat_shape": list(np.asarray(mat).shape)},
        "topk_protocol": "K equals number of test anomalies; labels diagnostic-only for oracle categories and scoring evaluation",
        "k_anomaly_count": int(k),
        "num_test": int(len(idx_test)),
        "metrics": metrics,
        "oracle_category_counts": {k: len(v) for k, v in categories.items()},
        "category_effect_size_map": effect_size_map,
        "tradeoff": tradeoff,
        "layer1_diagnostics": l1_diagnostics,
        "anti_shortcut_correlations": anti,
    }


def summarize(rows, strategy_meta):
    if not rows:
        return {"n_rows": 0}
    strategies = sorted(rows[0]["tradeoff"].keys())
    l1_strategies = [s for s in strategies if strategy_meta.get(s, {}).get("family") == "label_free_monotone_logistic_gate"]
    l0_strategies = [s for s in strategies if s not in l1_strategies]
    cats = sorted(rows[0]["oracle_category_counts"].keys())
    out = {
        "n_rows": len(rows),
        "margin_auc": mean_std([r["metrics"]["margin"]["auc"] for r in rows]),
        "margin_ap": mean_std([r["metrics"]["margin"]["ap"] for r in rows]),
        "mat_mean_auc": mean_std([r["metrics"]["mat_mean"]["auc"] for r in rows]),
        "mat_mean_ap": mean_std([r["metrics"]["mat_mean"]["ap"] for r in rows]),
        "mat_mean_formula_max_abs_diff": mean_std([r["formula_check"]["mat_mean_equals_direct_matrix_mean_max_abs_diff"] for r in rows]),
        "oracle_category_counts": {c: mean_std([r["oracle_category_counts"][c] for r in rows]) for c in cats},
        "score_auc": {s: mean_std([r["tradeoff"][s]["auc"] for r in rows]) for s in strategies},
        "score_ap": {s: mean_std([r["tradeoff"][s]["ap"] for r in rows]) for s in strategies},
        "tradeoff": {},
        "strategy_meta": {s: strategy_meta.get(s, {}) for s in strategies},
        "split_fingerprints": {str(r["seed"]): r["split_fingerprint"] for r in rows},
        "layer1_fit_diagnostics": {},
    }
    for s in strategies:
        out["tradeoff"][s] = {
            "recovered_lost_anomalies": mean_std([r["tradeoff"][s]["recovered_lost_anomalies"] for r in rows]),
            "reintroduced_removed_false_positives": mean_std([r["tradeoff"][s]["reintroduced_removed_false_positives"] for r in rows]),
            "retained_rescued_anomalies": mean_std([r["tradeoff"][s]["retained_rescued_anomalies"] for r in rows]),
            "retained_introduced_false_positives": mean_std([r["tradeoff"][s]["retained_introduced_false_positives"] for r in rows]),
            "spearman_vs_margin": mean_std([r["tradeoff"][s]["spearman_vs_margin"] for r in rows]),
            "spearman_vs_mat_mean": mean_std([r["tradeoff"][s]["spearman_vs_mat_mean"] for r in rows]),
            "topk_overlap_with_mat_mean": mean_std([r["tradeoff"][s]["topk_overlap_with_mat_mean"] for r in rows]),
        }
        if s in l1_strategies:
            out["layer1_fit_diagnostics"][s] = {
                "w_cmf": mean_std([r["layer1_diagnostics"][s]["weights"]["w_cmf"] for r in rows]),
                "w_reliability": mean_std([r["layer1_diagnostics"][s]["weights"]["w_reliability"] for r in rows]),
                "w_fragmentation": mean_std([r["layer1_diagnostics"][s]["weights"]["w_fragmentation"] for r in rows]),
                "bias": mean_std([r["layer1_diagnostics"][s]["weights"]["bias"] for r in rows]),
                "positive_anchors": mean_std([r["layer1_diagnostics"][s]["anchor_counts"]["positive"] for r in rows]),
                "negative_anchors": mean_std([r["layer1_diagnostics"][s]["anchor_counts"]["negative"] for r in rows]),
                "final_loss": mean_std([r["layer1_diagnostics"][s]["loss"]["final"] for r in rows]),
            }

    mat_auc = out["mat_mean_auc"]["mean"] if out["mat_mean_auc"] else None
    mat_ap = out["mat_mean_ap"]["mean"] if out["mat_mean_ap"] else None
    leaderboard = []
    for s in strategies:
        auc = out["score_auc"][s]["mean"] if out["score_auc"][s] else None
        ap = out["score_ap"][s]["mean"] if out["score_ap"][s] else None
        rec = out["tradeoff"][s]["recovered_lost_anomalies"]["mean"] if out["tradeoff"][s]["recovered_lost_anomalies"] else 0
        bad = out["tradeoff"][s]["reintroduced_removed_false_positives"]["mean"] if out["tradeoff"][s]["reintroduced_removed_false_positives"] else 0
        rho = out["tradeoff"][s]["spearman_vs_mat_mean"]["mean"] if out["tradeoff"][s]["spearman_vs_mat_mean"] else None
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
    out["layer1_leaderboard"] = [x for x in out["diagnostic_leaderboard"] if x["family"] == "label_free_monotone_logistic_gate"]
    out["layer0_reference_leaderboard"] = [x for x in out["diagnostic_leaderboard"] if x["family"] != "label_free_monotone_logistic_gate"]
    best_l1 = out["layer1_leaderboard"][0] if out["layer1_leaderboard"] else None
    best_l0 = out["layer0_reference_leaderboard"][0] if out["layer0_reference_leaderboard"] else None
    out["continuation_gate"] = {
        "criteria": [
            "AUC_mean >= mat_mean_AUC_mean + 0.003 OR AP_mean >= mat_mean_AP_mean + 0.005",
            "FP reintroduction does not worsen materially vs best Layer-0 reliability gate",
            "Spearman vs mat_mean < 0.95 to avoid pure monotone rewrite",
            "abs shortcut correlation does not increase meaningfully",
            "cross-seed variance remains controlled",
        ],
        "best_layer1": best_l1,
        "best_layer0_reference": best_l0,
    }
    if best_l1 and mat_auc is not None and mat_ap is not None:
        pass_metric = (best_l1["auc_mean"] is not None and best_l1["auc_mean"] >= mat_auc + 0.003) or (best_l1["ap_mean"] is not None and best_l1["ap_mean"] >= mat_ap + 0.005)
        pass_rho = best_l1["spearman_vs_mat_mean_mean"] is None or best_l1["spearman_vs_mat_mean_mean"] < 0.95
        pass_shortcut = best_l1["abs_shortcut_corr_mean"] is None or best_l1["abs_shortcut_corr_mean"] < 0.20
        if pass_metric and pass_rho and pass_shortcut:
            out["decision"] = "LAYER1_LABEL_FREE_GATE_PROMISING_REVIEW_FOR_NEXT_MECHANISM_STEP"
        elif pass_metric:
            out["decision"] = "LAYER1_IMPROVES_METRIC_BUT_REQUIRES_FAILURE_MODE_AUDIT"
        else:
            out["decision"] = "LAYER1_LABEL_FREE_GATE_NOT_PROMOTED_USE_AS_DIAGNOSTIC"
        out["continuation_gate"].update({"pass_metric": bool(pass_metric), "pass_rho": bool(pass_rho), "pass_shortcut": bool(pass_shortcut)})
    else:
        out["decision"] = "LAYER1_NO_VALID_RESULT"
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

    protocol = {
        "type": "runner-registered pure probe; Layer-1 label-free shallow monotone logistic reliability gate",
        "label_boundary": "anomaly labels diagnostic-only for AUC/AP and top-K boundary categories; labels are not used for fitting, pseudo-anchor construction, early stopping, or hyperparameter selection",
        "training_signal": "pseudo reliability anchors from high consensus/low fragmentation and low consensus or high fragmentation; no anomaly labels",
        "formula_boundary": "g=sigmoid(w_c*z(cmf)+w_r*z(joint_reliability)-w_f*z(fragmentation)+b), constrained w>=0; score=g*z(mat_mean)+(1-g)*z(margin)",
        "critical_fix": "force data_split_seed=seed and run seeds sequentially to avoid global random.shuffle cross-thread drift",
        "predeclared_scan": {"q_values": args.q_values, "qf_values": args.qf_values, "alpha_anchor_values": args.alpha_anchor_values, "alpha_mono_values": args.alpha_mono_values, "lambda_l2_values": args.lambda_l2_values, "train_steps": args.train_steps},
    }

    def compact_partial(v):
        if not rows_by_variant[v]:
            return {"n_rows": 0}
        s = summarize(rows_by_variant[v], strategy_meta)
        return {"n_rows": s["n_rows"], "mat_mean_auc": s["mat_mean_auc"], "mat_mean_ap": s["mat_mean_ap"], "decision": s.get("decision"), "top5_layer1": s.get("layer1_leaderboard", [])[:5]}

    def snapshot(status="running"):
        atomic_write_json(args.progress_out, {"status": status, "probe": "cleg3_layer1_label_free_shallow_gate_probe", "done": done, "total": total, "variants": variants, "seeds": seeds, "sequential": True, "partial": {v: compact_partial(v) for v in variants}, "errors": errors[-5:], "elapsed_sec": time.time() - start})

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
    payload = {"status": "finished" if not errors else "finished_with_errors", "probe": "cleg3_layer1_label_free_shallow_gate_probe", "protocol": protocol, "dataset": args.dataset, "seeds": seeds, "variants": variants, "devices": devices, "config": vars(args), "variant_summaries": variant_summaries, "errors": errors, "elapsed_sec": time.time() - start}
    atomic_write_json(args.out, payload)
    snapshot(payload["status"])
    print(json.dumps({"stage": "probe_done", "status": payload["status"], "done": done, "total": total, "out": args.out}, ensure_ascii=False), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
