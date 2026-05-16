#!/usr/bin/env python3
"""Elliptic orientation/sign diagnostic for DualRefGAD.

No-training diagnostic. It probes why margin/rejection orientation flips on Elliptic
and whether a label-free orientation rule can decide score vs -score.
Labels are used only for evaluation/autopsy metrics.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path.home() / "DualRefGAD"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))

from utils import load_mat  # noqa: E402
from VecGAD import VecGAD  # noqa: E402
import stage4_rho_normality_alignment_probe as s4  # noqa: E402


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_auc_ap(labels, score, idx):
    idx = np.asarray(idx, dtype=int)
    y = np.asarray(labels)[idx]
    s = np.asarray(score)[idx]
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def safe_spearman(a, b):
    try:
        v = spearmanr(np.asarray(a), np.asarray(b)).correlation
        return float(0.0 if np.isnan(v) else v)
    except Exception:
        return 0.0


def rank_percentile(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x)
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x))
    return r / max(1, len(x) - 1)


def degree_features(adj):
    csr = adj.tocsr() if sp.issparse(adj) else sp.csr_matrix(adj)
    deg = np.asarray(csr.sum(axis=1)).reshape(-1).astype(np.float64)
    logdeg = np.log1p(deg)
    return deg, logdeg


def time_feature_candidates(features_np):
    """Heuristic: Elliptic often carries timestep-like feature columns.

    Return columns with low unique integer-ish values and enough variation. This is
    diagnostic-only: it does not assume a specific Elliptic schema.
    """
    cands = []
    n, d = features_np.shape
    for j in range(min(d, 128)):
        col = np.asarray(features_np[:, j], dtype=np.float64)
        if not np.all(np.isfinite(col)):
            continue
        unique = np.unique(col)
        if 5 <= len(unique) <= min(100, max(10, n // 20)):
            # integer-ish or discretized after preprocessing
            rounded = np.round(col)
            intish = float(np.mean(np.abs(col - rounded) < 1e-6))
            cands.append({
                'col': int(j),
                'n_unique': int(len(unique)),
                'intish_frac': intish,
                'min': float(np.min(col)),
                'max': float(np.max(col)),
                'label_spearman': None,
            })
    return cands[:12]


def score_report(name, score, labels, idx_test, aux):
    auc, ap = safe_auc_ap(labels, score, idx_test)
    nauc, nap = safe_auc_ap(labels, -np.asarray(score), idx_test)
    out = {
        'score': name,
        'auc': auc,
        'ap': ap,
        'neg_auc': nauc,
        'neg_ap': nap,
        'best_orientation': 'positive' if auc >= nauc else 'negative',
        'best_auc': max(auc, nauc),
        'mean': float(np.mean(score)),
        'std': float(np.std(score)),
    }
    for k, v in aux.items():
        out[f'spearman_{k}'] = safe_spearman(score, v)
    return out


def group_stats(score, labels, idx, group_values, n_bins=10):
    idx = np.asarray(idx, dtype=int)
    score = np.asarray(score)
    labels = np.asarray(labels)
    g = np.asarray(group_values)
    vals = g[idx]
    # quantile bins robust to skew
    qs = np.unique(np.quantile(vals, np.linspace(0, 1, n_bins + 1)))
    bins = []
    if len(qs) <= 2:
        return bins
    for lo, hi in zip(qs[:-1], qs[1:]):
        mask = (vals >= lo) & (vals <= hi if hi == qs[-1] else vals < hi)
        if mask.sum() < 20:
            continue
        sub = idx[mask]
        if len(np.unique(labels[sub])) < 2:
            auc = None
            neg_auc = None
        else:
            auc, _ = safe_auc_ap(labels, score, sub)
            neg_auc, _ = safe_auc_ap(labels, -score, sub)
        bins.append({
            'lo': float(lo), 'hi': float(hi), 'n': int(len(sub)),
            'anom_rate': float(np.mean(labels[sub])),
            'score_mean': float(np.mean(score[sub])),
            'auc': auc, 'neg_auc': neg_auc,
        })
    return bins


def orientation_rules(scores, aux, normal_idx, all_unlabeled_idx):
    """Label-free orientation heuristics.

    Assumption family: anomaly-high scores should align with high normal-model
    rejection / structural rarity, and should be larger on non-train nodes than
    clean train normals. This returns which sign each rule would select.
    """
    rules = []
    normal_idx = np.asarray(normal_idx, dtype=int)
    all_unlabeled_idx = np.asarray(all_unlabeled_idx, dtype=int)
    for name, score in scores.items():
        s = np.asarray(score)
        train_mean = float(np.mean(s[normal_idx]))
        unlabeled_mean = float(np.mean(s[all_unlabeled_idx]))
        reject_corr = safe_spearman(s, aux['rejection'])
        ga_corr = safe_spearman(s, aux['ga'])
        deg_corr = safe_spearman(s, aux['log_degree'])
        rule_unlabeled_gt_train = 'positive' if unlabeled_mean >= train_mean else 'negative'
        rule_rejection_align = 'positive' if reject_corr >= 0 else 'negative'
        rule_ga_align = 'positive' if ga_corr >= 0 else 'negative'
        votes = [rule_unlabeled_gt_train, rule_rejection_align, rule_ga_align]
        pred = 'positive' if votes.count('positive') >= votes.count('negative') else 'negative'
        rules.append({
            'score': name,
            'train_normal_mean': train_mean,
            'unlabeled_mean': unlabeled_mean,
            'spearman_rejection': reject_corr,
            'spearman_ga': ga_corr,
            'spearman_log_degree': deg_corr,
            'rule_unlabeled_gt_train': rule_unlabeled_gt_train,
            'rule_rejection_align': rule_rejection_align,
            'rule_ga_align': rule_ga_align,
            'majority_orientation_prediction': pred,
        })
    return rules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='elliptic')
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train_rate', type=float, default=0.05)
    ap.add_argument('--val_rate', type=float, default=0.0)
    ap.add_argument('--descriptor_mode', choices=['hop_attr', 'rwse', 'hop_attr_rwse'], default='hop_attr_rwse')
    ap.add_argument('--pn_estimator', choices=['diag_gaussian', 'pca_residual'], default='diag_gaussian')
    ap.add_argument('--gn_mode', choices=['label_gate', 'normal_density', 'label_gate_density'], default='label_gate_density')
    ap.add_argument('--ln_mode', choices=['descriptor_similarity', 'reconstruction_gain'], default='descriptor_similarity')
    ap.add_argument('--ga_mode', choices=['normal_rejection', 'residual_norm', 'normal_soft_or'], default='normal_rejection')
    ap.add_argument('--la_mode', choices=['residual_cosine', 'descriptor_similarity'], default='residual_cosine')
    ap.add_argument('--normal_k', type=int, default=4)
    ap.add_argument('--anom_k', type=int, default=16)
    ap.add_argument('--pp_k', type=int, default=6)
    ap.add_argument('--hops', type=int, default=2)
    ap.add_argument('--rw_steps', type=int, default=8)
    ap.add_argument('--pca_components', type=int, default=32)
    ap.add_argument('--embedding_dim', type=int, default=256)
    ap.add_argument('--GT_ffn_dim', type=int, default=256)
    ap.add_argument('--GT_dropout', type=float, default=0.4)
    ap.add_argument('--GT_attention_dropout', type=float, default=0.4)
    ap.add_argument('--GT_num_heads', type=int, default=2)
    ap.add_argument('--GT_num_layers', type=int, default=1)
    ap.add_argument('--sample_rate', type=float, default=0.15)
    ap.add_argument('--mean', type=float, default=0.02)
    ap.add_argument('--var', type=float, default=0.01)
    ap.add_argument('--outlier_beta', type=float, default=0.3)
    ap.add_argument('--ring_R_max', type=float, default=1.0)
    ap.add_argument('--ring_R_min', type=float, default=0.3)
    ap.add_argument('--lambda_rec_tok', type=float, default=1.0)
    ap.add_argument('--lambda_rec_emb', type=float, default=0.1)
    ap.add_argument('--encode_batch_size', type=int, default=2048)
    ap.add_argument('--ref_block_size', type=int, default=1024)
    ap.add_argument('--use_approx_anom_refs', action='store_true')
    ap.add_argument('--anom_approx_k', type=int, default=500)
    ap.add_argument('--out', default='outputs/elliptic_orientation_sign_probe.json')
    args = ap.parse_args()

    start = time.time()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')

    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, _, _, normal_for_train_idx, _ = load_mat(
        args.dataset, args.train_rate, args.val_rate, args=args
    )
    features_np = s4.to_dense_features(args.dataset, features)
    labels_np = np.asarray(ano_label).reshape(-1).astype(int)
    normal_idx = np.asarray(normal_for_train_idx, dtype=int)
    idx_test = np.asarray(idx_test, dtype=int)
    all_idx = np.asarray(all_idx, dtype=int)
    assert np.sum(labels_np[normal_idx]) == 0, 'Data leakage: normal_for_train_idx contains anomalies'

    z = s4.build_descriptor(args.descriptor_mode, features_np, adj, args.hops, args.rw_steps)
    nm = s4.NormalModel(args.pn_estimator, z, normal_idx, args.pca_components)
    residual = nm.residual()
    normal_refs, anom_refs, score_meta = s4.select_refs(z, residual, normal_idx, nm, features_np, adj, args)
    token_tensor = s4.build_tokens(features_np, normal_refs, anom_refs)

    model = VecGAD(features_np.shape[1], args.embedding_dim, 'prelu', args).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    with torch.no_grad():
        emb_t = s4.encode_tokens_batched(model, token_tensor, device, args.encode_batch_size)
        normal_refs_t = torch.tensor(normal_refs, dtype=torch.long, device=device)
        anom_refs_t = torch.tensor(anom_refs, dtype=torch.long, device=device)
        zn_feat_t, zd_feat_t, margin_t = s4.build_relation_features(emb_t, normal_refs_t, anom_refs_t)

    emb = emb_t.detach().cpu().numpy().astype(np.float32)
    rn = emb[normal_refs].mean(axis=1)
    ra = emb[anom_refs].mean(axis=1)
    u = emb - rn
    d = ra - rn
    margin = margin_t.detach().cpu().numpy().astype(np.float32)

    deg, logdeg = degree_features(adj)
    rejection = np.asarray(score_meta.get('rejection'), dtype=np.float32)
    ga = np.asarray(score_meta.get('ga'), dtype=np.float32)
    residual_norm = np.asarray(score_meta.get('residual_norm'), dtype=np.float32)
    anom_ref_ratio = np.mean(labels_np[anom_refs] == 1, axis=1)
    normal_ref_degree = np.mean(logdeg[normal_refs], axis=1)
    anom_ref_degree = np.mean(logdeg[anom_refs], axis=1)
    rn_dist = np.linalg.norm(emb - rn, axis=1)
    ra_dist = np.linalg.norm(emb - ra, axis=1)
    d_norm = np.linalg.norm(d, axis=1)
    u_norm = np.linalg.norm(u, axis=1)
    raw_dot = np.sum(u * d, axis=1)
    cos_neg_u_d = -margin

    aux = {
        'rejection': rejection,
        'ga': ga,
        'residual_norm': residual_norm,
        'log_degree': logdeg,
        'degree': deg,
        'anom_ref_ratio': anom_ref_ratio,
        'normal_ref_degree': normal_ref_degree,
        'anom_ref_degree': anom_ref_degree,
        'rn_dist': rn_dist,
        'ra_dist': ra_dist,
        'd_norm': d_norm,
        'u_norm': u_norm,
    }
    scores = {
        'margin_cos_u_d': margin,
        'neg_margin_cos_u_d': cos_neg_u_d,
        'raw_dot_u_d': raw_dot,
        'neg_raw_dot_u_d': -raw_dot,
        'normal_rejection': rejection,
        'neg_normal_rejection': -rejection,
        'ga_score': ga,
        'neg_ga_score': -ga,
        'residual_norm': residual_norm,
        'neg_residual_norm': -residual_norm,
        'rn_dist': rn_dist,
        'neg_rn_dist': -rn_dist,
        'ra_dist': ra_dist,
        'neg_ra_dist': -ra_dist,
        'd_norm': d_norm,
        'neg_d_norm': -d_norm,
        'u_norm': u_norm,
        'neg_u_norm': -u_norm,
        'anom_ref_ratio_diagnostic_label_only': anom_ref_ratio,
    }

    reports = [score_report(name, score, labels_np, idx_test, aux) for name, score in scores.items()]
    reports.sort(key=lambda r: r['best_auc'], reverse=True)
    orient = orientation_rules({k: v for k, v in scores.items() if not k.endswith('label_only')}, aux, normal_idx, idx_test)

    # time-feature candidates and score drift by candidate columns
    time_cands = time_feature_candidates(features_np)
    for c in time_cands:
        col = features_np[:, c['col']]
        c['label_spearman'] = safe_spearman(col[idx_test], labels_np[idx_test])
        c['margin_spearman'] = safe_spearman(col, margin)
        c['rejection_spearman'] = safe_spearman(col, rejection)

    binned = {
        'by_log_degree_margin': group_stats(margin, labels_np, idx_test, logdeg, n_bins=10),
        'by_log_degree_rejection': group_stats(rejection, labels_np, idx_test, logdeg, n_bins=10),
        'by_rejection_margin': group_stats(margin, labels_np, idx_test, rejection, n_bins=10),
        'by_ga_margin': group_stats(margin, labels_np, idx_test, ga, n_bins=10),
    }
    if time_cands:
        best_time_col = max(time_cands, key=lambda c: abs(c.get('label_spearman') or 0))['col']
        binned['by_candidate_time_margin'] = group_stats(margin, labels_np, idx_test, features_np[:, best_time_col], n_bins=10)
        binned['by_candidate_time_rejection'] = group_stats(rejection, labels_np, idx_test, features_np[:, best_time_col], n_bins=10)
        binned['candidate_time_col'] = best_time_col

    result = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'protocol': 'no-training orientation/sign diagnostic; labels diagnostic-only for metrics/autopsy',
        'dataset': args.dataset,
        'seed': args.seed,
        'device': str(device),
        'n_nodes': int(len(labels_np)),
        'n_train_normal': int(len(normal_idx)),
        'n_test': int(len(idx_test)),
        'test_anom_rate': float(np.mean(labels_np[idx_test])),
        'config': vars(args),
        'score_reports_by_best_auc': reports,
        'label_free_orientation_rules': orient,
        'time_feature_candidates': time_cands,
        'binned_autopsy': binned,
        'reference_summary': {
            'global_anom_ref_anom_ratio': float(np.mean(labels_np[anom_refs] == 1)),
            'global_normal_ref_anom_ratio': float(np.mean(labels_np[normal_refs] == 1)),
            'anom_ref_ratio_auc': safe_auc_ap(labels_np, anom_ref_ratio, idx_test)[0],
            'anom_ref_ratio_ap': safe_auc_ap(labels_np, anom_ref_ratio, idx_test)[1],
            'mean_normal_ref_log_degree': float(np.mean(normal_ref_degree)),
            'mean_anom_ref_log_degree': float(np.mean(anom_ref_degree)),
        },
        'time_sec': float(time.time() - start),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({
        'out': str(out),
        'top_scores': reports[:10],
        'orientation_rules': orient[:8],
        'reference_summary': result['reference_summary'],
        'time_feature_candidates': time_cands[:5],
        'time_sec': result['time_sec'],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
