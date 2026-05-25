#!/usr/bin/env python3
"""
Route2.5 Matrix Autoencoder Probe
=================================

No-training / light-training diagnostic for DualRefGAD.

Purpose
-------
Test whether the full K_n × K_a response matrix pattern contains usable
normal-only anomaly signal beyond the scalar margin / matrix summaries.

Protocol
--------
- Build descriptors and references using the same Stage4/RHO probe utilities.
- Freeze the GT encoder; no graph encoder training.
- Construct response matrix M_ij(v)=cos(h_v-r_n_i, r_a_j-r_n_i).
- Train small autoencoders only on labeled-normal training nodes.
- Score all nodes by reconstruction error; labels are used only for evaluation.

This script is intended to run as an experiment-runner registered probe.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path.home() / "DualRefGAD"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))

try:
    from utils import load_mat  # noqa: E402
except Exception:
    from utils_lite import load_mat  # noqa: E402
from VecGAD import VecGAD  # noqa: E402
import stage4_rho_normality_alignment_probe as s4  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_auc_ap(labels: np.ndarray, score: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    idx = np.asarray(idx, dtype=int)
    y = np.asarray(labels)[idx]
    s = np.asarray(score)[idx]
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def safe_spearman(a, b) -> float:
    try:
        v = spearmanr(np.asarray(a), np.asarray(b)).correlation
        return float(0.0 if np.isnan(v) else v)
    except Exception:
        return 0.0


def topk_jaccard(a, b, pct: float = 0.05) -> float:
    n = len(a)
    k = max(1, int(n * pct))
    ia = set(np.argsort(-np.asarray(a))[:k].tolist())
    ib = set(np.argsort(-np.asarray(b))[:k].tolist())
    return float(len(ia & ib) / max(1, len(ia | ib)))


def degree_score(adj) -> np.ndarray:
    csr = adj.tocsr()
    return np.asarray(csr.sum(axis=1)).reshape(-1).astype(np.float32)


def l2_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def rank_percentile(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x)
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x))
    return (r / max(1, len(x) - 1)).astype(np.float32)


def cosine_topk_refs(a: np.ndarray, b: np.ndarray, score_b: np.ndarray, k: int, block: int = 1024, exclude_global_self: np.ndarray | None = None) -> np.ndarray:
    """Top-k refs by cosine(a_i,b_j)+score_b[j] without materializing full N×N."""
    an = l2_rows(a)
    bn = l2_rows(b)
    score_b = np.asarray(score_b, dtype=np.float32)
    refs = np.empty((a.shape[0], k), dtype=np.int64)
    for st in range(0, a.shape[0], block):
        ed = min(st + block, a.shape[0])
        scores = an[st:ed] @ bn.T
        scores += score_b[None, :]
        if exclude_global_self is not None:
            # exclude_global_self maps global row id -> local candidate column, or -1
            cols = exclude_global_self[st:ed]
            rows = np.where(cols >= 0)[0]
            if len(rows):
                scores[rows, cols[rows]] = -1e9
        kk = min(k, scores.shape[1])
        part = np.argpartition(-scores, kth=kk - 1, axis=1)[:, :kk]
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        refs[st:ed, :kk] = np.take_along_axis(part, order, axis=1)
    return refs


def select_refs_memory_safe(z: np.ndarray, residual: np.ndarray, normal_idx: np.ndarray, nm, features_np: np.ndarray, adj, args):
    """Route2.5 local reference selection; no full N×N allocation for Elliptic."""
    n = z.shape[0]
    rejection = nm.rejection()
    density = nm.density_score()
    residual_norm = np.linalg.norm(residual, axis=1)

    if args.gn_mode == 'label_gate':
        normal_pool = np.asarray(normal_idx, dtype=np.int64)
        gn = np.zeros(n, dtype=np.float32)
        gn[normal_pool] = 1.0
    elif args.gn_mode == 'normal_density':
        normal_pool = np.arange(n, dtype=np.int64)
        gn = rank_percentile(density)
    elif args.gn_mode == 'label_gate_density':
        normal_pool = np.asarray(normal_idx, dtype=np.int64)
        gn = rank_percentile(density)
        mask = np.ones(n, bool)
        mask[normal_pool] = False
        gn[mask] = -1e9
    else:
        raise ValueError(args.gn_mode)

    if args.ga_mode == 'normal_rejection':
        ga = rank_percentile(rejection)
    elif args.ga_mode == 'residual_norm':
        ga = rank_percentile(residual_norm)
    elif args.ga_mode == 'normal_soft_or':
        ga = s4.normal_soft_or_score(features_np, adj, normal_idx).astype(np.float32)
    else:
        raise ValueError(args.ga_mode)

    normal_score_b = gn[normal_pool]
    normal_refs_local = cosine_topk_refs(z, z[normal_pool], normal_score_b, args.normal_k, args.ref_block_size)
    normal_refs = normal_pool[normal_refs_local]

    if args.la_mode == 'residual_cosine':
        a = residual
    elif args.la_mode == 'descriptor_similarity':
        a = z
    else:
        raise ValueError(args.la_mode)

    if args.use_approx_anom_refs or n > 30000:
        cand_global = np.argsort(-ga)[:min(args.anom_approx_k, n)].astype(np.int64)
    else:
        cand_global = np.arange(n, dtype=np.int64)
    local_of_global = np.full(n, -1, dtype=np.int64)
    local_of_global[cand_global] = np.arange(len(cand_global), dtype=np.int64)
    anom_refs_local = cosine_topk_refs(a, a[cand_global], ga[cand_global], args.anom_k, args.ref_block_size, exclude_global_self=local_of_global)
    anom_refs = cand_global[anom_refs_local]
    return normal_refs.astype(np.int64), anom_refs.astype(np.int64), {'ga': ga, 'rejection': rejection, 'residual_norm': residual_norm, 'anom_candidate_count': int(len(cand_global))}


def build_response_matrix_batched(
    emb: torch.Tensor,
    normal_refs: np.ndarray,
    anom_refs: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    """Build response matrix in batches.

    emb: [N,D]; refs: [N,K]. Returns [N,K_n,K_a] on CPU as float32.
    """
    emb = emb.to(device)
    n = emb.shape[0]
    chunks = []
    for st in range(0, n, batch_size):
        ed = min(st + batch_size, n)
        nr = torch.as_tensor(normal_refs[st:ed], dtype=torch.long, device=device)
        ar = torch.as_tensor(anom_refs[st:ed], dtype=torch.long, device=device)
        h = emb[st:ed]                                   # [B,D]
        rn = emb[nr]                                     # [B,K_n,D]
        ra = emb[ar]                                     # [B,K_a,D]
        u = h[:, None, :] - rn                           # [B,K_n,D]
        d = ra[:, None, :, :] - rn[:, :, None, :]         # [B,K_n,K_a,D]
        u_n = F.normalize(u, p=2, dim=-1)
        d_n = F.normalize(d, p=2, dim=-1)
        m = (u_n[:, :, None, :] * d_n).sum(dim=-1)        # [B,K_n,K_a]
        chunks.append(m.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


class MatrixAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def train_autoencoder(
    x_train: np.ndarray,
    latent_dim: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: torch.device,
    dropout: float = 0.0,
) -> tuple[MatrixAutoencoder, list[float]]:
    model = MatrixAutoencoder(x_train.shape[1], latent_dim, hidden_dim, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    x = torch.as_tensor(x_train, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(x, batch_size=min(batch_size, len(x)), shuffle=True, drop_last=False)
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb in loader:
            xb = xb.to(device)
            opt.zero_grad()
            rec, _ = model(xb)
            loss = F.mse_loss(rec, xb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.shape[0]
            count += xb.shape[0]
        losses.append(total / max(1, count))
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(json.dumps({'ae_epoch': epoch, 'latent_dim': latent_dim, 'loss': losses[-1]}, ensure_ascii=False), flush=True)
    return model, losses


def reconstruction_error(model: MatrixAutoencoder, x_all: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    errs = []
    x = torch.as_tensor(x_all, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            rec, _ = model(xb)
            errs.append(((rec - xb) ** 2).mean(dim=1).detach().cpu().numpy())
    return np.concatenate(errs).astype(np.float32)


def score_report(name: str, score: np.ndarray, labels: np.ndarray, idx_test: np.ndarray, margin: np.ndarray, degree: np.ndarray, rejection: np.ndarray):
    auc, ap = safe_auc_ap(labels, score, idx_test)
    neg_auc, neg_ap = safe_auc_ap(labels, -np.asarray(score), idx_test)
    return {
        'score': name,
        'auc': auc,
        'ap': ap,
        'neg_auc': neg_auc,
        'neg_ap': neg_ap,
        'best_auc_either_sign': max(auc, neg_auc),
        'best_orientation': 'positive' if auc >= neg_auc else 'negative',
        'spearman_margin': safe_spearman(score[idx_test], margin[idx_test]),
        'spearman_neg_margin': safe_spearman(score[idx_test], (-margin)[idx_test]),
        'spearman_degree': safe_spearman(score[idx_test], degree[idx_test]),
        'spearman_rejection': safe_spearman(score[idx_test], rejection[idx_test]),
        'top1_jaccard_margin': topk_jaccard(score[idx_test], margin[idx_test], 0.01),
        'top5_jaccard_margin': topk_jaccard(score[idx_test], margin[idx_test], 0.05),
    }


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description='Route2.5 Matrix AE seed diagnostic')
    ap.add_argument('--dataset', default='elliptic')
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train_rate', type=float, default=0.05)
    ap.add_argument('--val_rate', type=float, default=0.0)
    ap.add_argument('--latent_dims', default='4,8,16')
    ap.add_argument('--ae_hidden_dim', type=int, default=32)
    ap.add_argument('--ae_epochs', type=int, default=150)
    ap.add_argument('--ae_lr', type=float, default=1e-3)
    ap.add_argument('--ae_weight_decay', type=float, default=1e-5)
    ap.add_argument('--ae_batch_size', type=int, default=4096)
    ap.add_argument('--ae_dropout', type=float, default=0.0)
    ap.add_argument('--standardize_matrix', action='store_true', default=True)
    ap.add_argument('--no_standardize_matrix', dest='standardize_matrix', action='store_false')
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
    ap.add_argument('--matrix_batch_size', type=int, default=2048)
    ap.add_argument('--ref_block_size', type=int, default=1024)
    ap.add_argument('--use_approx_anom_refs', action='store_true')
    ap.add_argument('--anom_approx_k', type=int, default=500)
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--out', default='outputs/route25_matrix_ae_elliptic_s0.json')
    args = ap.parse_args()

    start = time.time()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')
    latent_dims = parse_int_list(args.latent_dims)

    print(json.dumps({'event': 'start', 'dataset': args.dataset, 'seed': args.seed, 'device': str(device), 'latent_dims': latent_dims}, ensure_ascii=False), flush=True)

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
    normal_refs, anom_refs, score_meta = select_refs_memory_safe(z, residual, normal_idx, nm, features_np, adj, args)
    token_tensor = s4.build_tokens(features_np, normal_refs, anom_refs)

    dry = {
        'dataset': args.dataset,
        'seed': args.seed,
        'device': str(device),
        'n_nodes': int(len(labels_np)),
        'n_features': int(features_np.shape[1]),
        'token_shape': list(token_tensor.shape),
        'normal_train_count': int(len(normal_idx)),
        'test_count': int(len(idx_test)),
        'test_anom_rate': float(np.mean(labels_np[idx_test])),
        'anom_ref_anom_ratio_diagnostic': float(np.mean(labels_np[anom_refs] == 1)),
    }
    print(json.dumps({'event': 'data_ready', **dry}, ensure_ascii=False), flush=True)
    if args.dry_run:
        print(json.dumps({'dry': dry}, indent=2, ensure_ascii=False), flush=True)
        return

    model = VecGAD(features_np.shape[1], args.embedding_dim, 'prelu', args).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    with torch.no_grad():
        emb = s4.encode_tokens_batched(model, token_tensor, device, args.encode_batch_size).detach()
        nr_t = torch.as_tensor(normal_refs, dtype=torch.long, device=device)
        ar_t = torch.as_tensor(anom_refs, dtype=torch.long, device=device)
        _, _, margin_t = s4.build_relation_features(emb, nr_t, ar_t)
    margin = margin_t.detach().cpu().numpy().astype(np.float32)

    print(json.dumps({'event': 'building_response_matrix'}, ensure_ascii=False), flush=True)
    M = build_response_matrix_batched(emb, normal_refs, anom_refs, device, args.matrix_batch_size)
    X = M.reshape(M.shape[0], -1).astype(np.float32)
    mat_mean = M.mean(axis=(1, 2))
    mat_median = np.median(X, axis=1)
    mat_std = X.std(axis=1)
    degree = degree_score(adj)
    rejection = np.asarray(score_meta.get('rejection'), dtype=np.float32)
    ga = np.asarray(score_meta.get('ga'), dtype=np.float32)

    if args.standardize_matrix:
        mu = X[normal_idx].mean(axis=0, keepdims=True)
        sd = X[normal_idx].std(axis=0, keepdims=True) + 1e-6
        X_ae = (X - mu) / sd
    else:
        mu = np.zeros((1, X.shape[1]), dtype=np.float32)
        sd = np.ones((1, X.shape[1]), dtype=np.float32)
        X_ae = X

    reports = []
    baseline_scores = {
        'margin': margin,
        'neg_margin': -margin,
        'mat_mean': mat_mean,
        'neg_mat_mean': -mat_mean,
        'mat_median': mat_median,
        'mat_std': mat_std,
        'degree': degree,
        'neg_degree': -degree,
        'rejection': rejection,
        'neg_rejection': -rejection,
        'ga': ga,
        'neg_ga': -ga,
    }
    for name, score in baseline_scores.items():
        reports.append(score_report(name, score, labels_np, idx_test, margin, degree, rejection))

    ae_runs = []
    for latent_dim in latent_dims:
        print(json.dumps({'event': 'train_ae', 'latent_dim': latent_dim}, ensure_ascii=False), flush=True)
        ae, losses = train_autoencoder(
            X_ae[normal_idx], latent_dim, args.ae_hidden_dim, args.ae_epochs, args.ae_lr,
            args.ae_weight_decay, args.ae_batch_size, device, args.ae_dropout,
        )
        err = reconstruction_error(ae, X_ae, args.ae_batch_size, device)
        rep = score_report(f'recon_error_ld{latent_dim}', err, labels_np, idx_test, margin, degree, rejection)
        rep['final_train_loss'] = float(losses[-1])
        rep['first_train_loss'] = float(losses[0])
        reports.append(rep)
        ae_runs.append({'latent_dim': latent_dim, 'losses': losses, 'report': rep})
        print(json.dumps({'event': 'ae_result', **rep}, ensure_ascii=False), flush=True)

    reports.sort(key=lambda r: r['auc'], reverse=True)
    best_ae = max([r for r in reports if r['score'].startswith('recon_error')], key=lambda r: r['auc'])
    neg_margin = next(r for r in reports if r['score'] == 'neg_margin')
    margin_rep = next(r for r in reports if r['score'] == 'margin')
    decision = {
        'best_ae_score': best_ae['score'],
        'best_ae_auc': best_ae['auc'],
        'best_ae_ap': best_ae['ap'],
        'neg_margin_auc': neg_margin['auc'],
        'margin_auc': margin_rep['auc'],
        'delta_vs_neg_margin': best_ae['auc'] - neg_margin['auc'],
        'best_ae_spearman_margin': best_ae['spearman_margin'],
        'best_ae_spearman_degree': best_ae['spearman_degree'],
        'success_rule': 'AUC > neg_margin + 0.02 OR comparable AUC with |Spearman(margin)| < 0.4',
    }
    if best_ae['auc'] > neg_margin['auc'] + 0.02:
        decision['recommendation'] = 'PROMOTE: Matrix AE beats -margin by >0.02; run 5-seed validation.'
    elif abs(best_ae['auc'] - neg_margin['auc']) <= 0.02 and abs(best_ae['spearman_margin']) < 0.4:
        decision['recommendation'] = 'PROMOTE_CAUTION: Matrix AE is competitive and relatively orthogonal; run 5-seed validation.'
    elif best_ae['auc'] < 0.58:
        decision['recommendation'] = 'DROP: Matrix AE below minimum AUC threshold; likely learns typicality not anomaly.'
    elif abs(best_ae['spearman_degree']) > 0.5:
        decision['recommendation'] = 'DROP_OR_REPAIR: Matrix AE strongly degree-correlated; consider regime-conditioned variant only.'
    else:
        decision['recommendation'] = 'INCONCLUSIVE: inspect score distributions before promotion.'

    result = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'protocol': 'Route2.5 Matrix AE normal-only diagnostic; labels diagnostic-only for metrics',
        'config': vars(args),
        'dry': dry,
        'matrix': {
            'shape': list(M.shape),
            'flat_shape': list(X.shape),
            'mean': float(M.mean()),
            'std': float(M.std()),
            'min': float(M.min()),
            'max': float(M.max()),
            'standardized': bool(args.standardize_matrix),
        },
        'score_results': reports,
        'ae_runs': ae_runs,
        'decision': decision,
        'time_sec': float(time.time() - start),
    }
    print('FINAL', json.dumps({'decision': decision, 'top_scores': reports[:8], 'time_sec': result['time_sec']}, indent=2, ensure_ascii=False), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'event': 'saved', 'out': str(out)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
