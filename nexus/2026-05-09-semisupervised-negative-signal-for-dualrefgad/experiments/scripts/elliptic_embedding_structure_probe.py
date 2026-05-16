#!/usr/bin/env python3
"""Elliptic embedding structure diagnostic for DualRefGAD Stage4 failure.

No-training diagnostic. It inspects whether frozen GT/DualRef relation spaces have
compact normal centers and whether center-distance scores separate anomalies.
Labels are used only for evaluation/autopsy metrics.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ks_2samp, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

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


def diag_mahal_score(x, normal_idx):
    mu = x[normal_idx].mean(axis=0, keepdims=True)
    std = x[normal_idx].std(axis=0, keepdims=True) + 1e-6
    return np.mean(((x - mu) / std) ** 2, axis=1)


def pca_residual_score(x, normal_idx, ncomp=32):
    scaler = StandardScaler(with_mean=True, with_std=True)
    xn = scaler.fit_transform(x[normal_idx])
    xa = scaler.transform(x)
    nc = int(min(ncomp, xn.shape[0] - 1, xn.shape[1]))
    if nc <= 0:
        return diag_mahal_score(x, normal_idx)
    pca = PCA(n_components=nc, svd_solver="randomized", random_state=0)
    pca.fit(xn)
    rec = pca.inverse_transform(pca.transform(xa))
    return np.mean((xa - rec) ** 2, axis=1)


def sample_pairwise_dist(x, idx, max_points=1200, seed=0):
    idx = np.asarray(idx, dtype=int)
    if len(idx) < 2:
        return {"mean": None, "std": None}
    rng = np.random.default_rng(seed)
    if len(idx) > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)
    xs = x[idx].astype(np.float32)
    # compute upper triangle distances in chunks to avoid scipy dependency overhead
    dists = []
    for st in range(0, xs.shape[0], 256):
        block = xs[st: st + 256]
        diff = block[:, None, :] - xs[None, :, :]
        dd = np.sqrt(np.sum(diff * diff, axis=-1))
        rows = np.arange(st, min(st + 256, xs.shape[0]))[:, None]
        cols = np.arange(xs.shape[0])[None, :]
        mask = rows < cols
        if np.any(mask):
            dists.append(dd[mask])
    if not dists:
        return {"mean": None, "std": None}
    d = np.concatenate(dists)
    return {"mean": float(np.mean(d)), "std": float(np.std(d))}


def effective_rank(x, idx, max_points=5000, ncomp=64, seed=0):
    idx = np.asarray(idx, dtype=int)
    rng = np.random.default_rng(seed)
    if len(idx) > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)
    xs = StandardScaler(with_mean=True, with_std=True).fit_transform(x[idx])
    nc = int(min(ncomp, xs.shape[0] - 1, xs.shape[1]))
    if nc <= 1:
        return {"ncomp": nc, "pr": None, "top1": None, "top5": None, "top10": None}
    pca = PCA(n_components=nc, svd_solver="randomized", random_state=seed)
    pca.fit(xs)
    ev = pca.explained_variance_ratio_.astype(np.float64)
    pr = (ev.sum() ** 2) / (np.sum(ev ** 2) + 1e-12)
    return {
        "ncomp": nc,
        "participation_ratio": float(pr),
        "top1_var": float(ev[:1].sum()),
        "top5_var": float(ev[:5].sum()),
        "top10_var": float(ev[:10].sum()),
    }


def summarize_space(name, x, labels, normal_train_idx, idx_test, margin, seed=0):
    labels = np.asarray(labels).astype(int)
    normal_train_idx = np.asarray(normal_train_idx, dtype=int)
    idx_test = np.asarray(idx_test, dtype=int)
    test_normal_idx = idx_test[labels[idx_test] == 0]
    test_anom_idx = idx_test[labels[idx_test] == 1]

    center = x[normal_train_idx].mean(axis=0, keepdims=True)
    center_dist = np.linalg.norm(x - center, axis=1)
    diag_score = diag_mahal_score(x, normal_train_idx)
    pca_score = pca_residual_score(x, normal_train_idx, ncomp=32)

    auc, ap = safe_auc_ap(labels, center_dist, idx_test)
    auc_m, ap_m = safe_auc_ap(labels, diag_score, idx_test)
    auc_p, ap_p = safe_auc_ap(labels, pca_score, idx_test)

    train_d = center_dist[normal_train_idx]
    test_n_d = center_dist[test_normal_idx]
    test_a_d = center_dist[test_anom_idx]
    pooled = math.sqrt((float(np.var(test_n_d)) + float(np.var(test_a_d))) / 2.0 + 1e-12) if len(test_n_d) and len(test_a_d) else None
    cohen = (float(np.mean(test_a_d) - np.mean(test_n_d)) / pooled) if pooled and pooled > 0 else None
    ks_train_test = ks_2samp(train_d, test_n_d).statistic if len(train_d) and len(test_n_d) else None
    ks_norm_anom = ks_2samp(test_n_d, test_a_d).statistic if len(test_n_d) and len(test_a_d) else None

    return {
        "space": name,
        "shape": list(x.shape),
        "center_distance_auc": auc,
        "center_distance_ap": ap,
        "diag_mahal_auc": auc_m,
        "diag_mahal_ap": ap_m,
        "pca_residual_auc": auc_p,
        "pca_residual_ap": ap_p,
        "spearman_center_margin": safe_spearman(center_dist, margin),
        "train_normal_radius_mean": float(np.mean(train_d)),
        "train_normal_radius_std": float(np.std(train_d)),
        "test_normal_radius_mean": float(np.mean(test_n_d)) if len(test_n_d) else None,
        "test_normal_radius_std": float(np.std(test_n_d)) if len(test_n_d) else None,
        "test_anom_radius_mean": float(np.mean(test_a_d)) if len(test_a_d) else None,
        "test_anom_radius_std": float(np.std(test_a_d)) if len(test_a_d) else None,
        "anom_minus_normal_radius": float(np.mean(test_a_d) - np.mean(test_n_d)) if len(test_n_d) and len(test_a_d) else None,
        "cohen_d_anom_vs_normal": cohen,
        "ks_train_normal_vs_test_normal": float(ks_train_test) if ks_train_test is not None else None,
        "ks_test_normal_vs_anom": float(ks_norm_anom) if ks_norm_anom is not None else None,
        "normal_pairwise_dist": sample_pairwise_dist(x, normal_train_idx, seed=seed),
        "test_normal_pairwise_dist": sample_pairwise_dist(x, test_normal_idx, seed=seed),
        "test_anom_pairwise_dist": sample_pairwise_dist(x, test_anom_idx, seed=seed),
        "effective_rank_train_normal": effective_rank(x, normal_train_idx, seed=seed),
    }


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
    ap.add_argument('--out', default='outputs/elliptic_embedding_structure_probe.json')
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
    zn_feat = zn_feat_t.detach().cpu().numpy().astype(np.float32)
    zd_feat = zd_feat_t.detach().cpu().numpy().astype(np.float32)

    margin_auc, margin_ap = safe_auc_ap(labels_np, margin, idx_test)
    neg_margin_auc, neg_margin_ap = safe_auc_ap(labels_np, -margin, idx_test)

    spaces = {
        'descriptor_z_pre_encoder': z.astype(np.float32),
        'gt_emb_h': emb,
        'rn_normal_ref_centroid': rn,
        'ra_anom_ref_centroid': ra,
        'u_target_minus_rn': u,
        'd_ra_minus_rn': d,
        'zn_feat_stage4_input': zn_feat,
        'zd_feat_stage4_input': zd_feat,
        'interaction_u_d': np.concatenate([u, d, u * d, np.abs(u - d)], axis=1).astype(np.float32),
    }
    space_reports = [summarize_space(name, x, labels_np, normal_idx, idx_test, margin, seed=args.seed) for name, x in spaces.items()]
    space_reports.sort(key=lambda r: r['center_distance_auc'], reverse=True)

    # Reference autopsy (diagnostic-only labels)
    anom_ref_ratio_per_node = np.mean(labels_np[anom_refs] == 1, axis=1)
    normal_ref_anom_ratio_per_node = np.mean(labels_np[normal_refs] == 1, axis=1)
    ref_report = {
        'global_anom_ref_anom_ratio': float(np.mean(labels_np[anom_refs] == 1)),
        'global_normal_ref_anom_ratio': float(np.mean(labels_np[normal_refs] == 1)),
        'anom_ref_ratio_auc': safe_auc_ap(labels_np, anom_ref_ratio_per_node, idx_test)[0],
        'anom_ref_ratio_ap': safe_auc_ap(labels_np, anom_ref_ratio_per_node, idx_test)[1],
        'spearman_anom_ref_ratio_margin': safe_spearman(anom_ref_ratio_per_node, margin),
        'ga_auc': safe_auc_ap(labels_np, np.asarray(score_meta.get('ga')), idx_test)[0],
        'rejection_auc': safe_auc_ap(labels_np, np.asarray(score_meta.get('rejection')), idx_test)[0],
    }

    result = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'protocol': 'no-training embedding structure diagnostic; labels diagnostic-only for metrics/autopsy',
        'dataset': args.dataset,
        'seed': args.seed,
        'device': str(device),
        'n_nodes': int(len(labels_np)),
        'n_train_normal': int(len(normal_idx)),
        'n_test': int(len(idx_test)),
        'test_anom_rate': float(np.mean(labels_np[idx_test])),
        'config': vars(args),
        'margin': {'auc': margin_auc, 'ap': margin_ap, 'neg_auc': neg_margin_auc, 'neg_ap': neg_margin_ap},
        'reference_autopsy': ref_report,
        'spaces_by_center_auc': space_reports,
        'time_sec': float(time.time() - start),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({
        'out': str(out),
        'margin': result['margin'],
        'reference_autopsy': ref_report,
        'top_spaces': [
            {k: r[k] for k in ['space', 'center_distance_auc', 'center_distance_ap', 'spearman_center_margin', 'cohen_d_anom_vs_normal', 'ks_train_normal_vs_test_normal']}
            for r in space_reports[:8]
        ],
        'time_sec': result['time_sec'],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
