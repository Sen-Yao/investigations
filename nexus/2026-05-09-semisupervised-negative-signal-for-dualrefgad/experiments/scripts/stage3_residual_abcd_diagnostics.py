#!/usr/bin/env python3
"""Stage-3 residual ABCD diagnostics.

This is a diagnostic-only companion to stage3_margin_residual_normalonly_probe.py.
It asks why score = margin + correction remains almost rank-identical to margin.

ABCD hypotheses tested:
A. Global shift / calibration: correction mostly acts like a constant downward shift.
B. Monotonic margin-linked correction: correction is largely a function of margin.
C. Too weak / wrong-locality perturbation: correction magnitude is insufficient to change ranks/top-k.
D. Normal-only supervision insufficiency: correction separates known normals weakly but not anomalies.

Protocol: same as Stage-3 residual probe. Training uses known normal nodes only;
anomaly labels are used only for post-hoc diagnostic metrics.
"""
import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr, pearsonr


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_auc(y, s):
    try:
        return float(roc_auc_score(y, s)), float(average_precision_score(y, s))
    except Exception:
        return 0.0, 0.0


def safe_spearman(a, b):
    try:
        v = spearmanr(np.asarray(a), np.asarray(b)).correlation
        return float(0.0 if np.isnan(v) else v)
    except Exception:
        return 0.0


def safe_pearson(a, b):
    try:
        v = pearsonr(np.asarray(a), np.asarray(b))[0]
        return float(0.0 if np.isnan(v) else v)
    except Exception:
        return 0.0


def top_indices(scores, frac):
    n = max(1, int(len(scores) * frac))
    return np.argsort(-scores)[:n]


def top_ratio(labels, scores, frac):
    idx = top_indices(scores, frac)
    return float(np.mean(labels[idx]))


def jaccard(a, b):
    sa, sb = set(map(int, a)), set(map(int, b))
    return float(len(sa & sb) / max(1, len(sa | sb)))


def cohen_d(x0, x1):
    x0 = np.asarray(x0, dtype=float); x1 = np.asarray(x1, dtype=float)
    if len(x0) < 2 or len(x1) < 2:
        return 0.0
    pooled = np.sqrt(((len(x0)-1)*np.var(x0, ddof=1) + (len(x1)-1)*np.var(x1, ddof=1)) / max(1, len(x0)+len(x1)-2))
    return float((np.mean(x1) - np.mean(x0)) / (pooled + 1e-12))


def linear_r2(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0, float(np.std(y))
    X = np.stack([np.ones_like(x), x], axis=1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    resid_std = float(np.std(y - pred))
    return float(r2), resid_std


def sampled_rank_flip_rate(base, new, rng, pairs=200000):
    base = np.asarray(base); new = np.asarray(new)
    n = len(base)
    if n < 2:
        return 0.0
    i = rng.integers(0, n, size=pairs)
    j = rng.integers(0, n, size=pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    b = np.sign(base[i] - base[j])
    s = np.sign(new[i] - new[j])
    valid = b != 0
    if np.sum(valid) == 0:
        return 0.0
    return float(np.mean(b[valid] != s[valid]))


def boundary_flip_rate(base, new, frac, width_frac=0.2):
    """Rank changes around the top-k boundary.

    Select a band around the margin top-k cutoff and compute how many nodes cross
    the top-k membership after correction.
    """
    base = np.asarray(base); new = np.asarray(new)
    n = len(base); k = max(1, int(n * frac)); width = max(1, int(k * width_frac))
    order = np.argsort(-base)
    lo = max(0, k - width); hi = min(n, k + width)
    band = order[lo:hi]
    base_top = np.zeros(n, dtype=bool); base_top[order[:k]] = True
    new_top = np.zeros(n, dtype=bool); new_top[np.argsort(-new)[:k]] = True
    return float(np.mean(base_top[band] != new_top[band])) if len(band) else 0.0


def split_normals(normal_idx, seed, val_frac):
    rng = np.random.default_rng(seed)
    arr = np.asarray(normal_idx, dtype=np.int64).copy(); rng.shuffle(arr)
    n_val = max(1, int(len(arr) * val_frac)) if val_frac > 0 else 0
    val = arr[:n_val]; train = arr[n_val:]
    if len(train) == 0:
        train, val = arr, arr
    return train, val


def build_components(emb, normal_refs, anom_refs, node_idx, rn_idx=None, ra_idx=None):
    node_idx = np.asarray(node_idx, dtype=np.int64)
    rn_base = node_idx if rn_idx is None else np.asarray(rn_idx, dtype=np.int64)
    ra_base = node_idx if ra_idx is None else np.asarray(ra_idx, dtype=np.int64)
    h = emb[node_idx]
    rn = emb[normal_refs[rn_base]].mean(dim=1)
    ra = emb[anom_refs[ra_base]].mean(dim=1)
    u = h - rn; d = ra - rn
    u_norm = F.normalize(u, p=2, dim=1, eps=1e-12)
    d_norm = F.normalize(d, p=2, dim=1, eps=1e-12)
    margin = torch.sum(u_norm * d_norm, dim=1)
    return h, rn, ra, u, d, u_norm, d_norm, margin


def build_relation_features(emb, normal_refs, anom_refs, node_idx, input_mode, rn_idx=None, ra_idx=None):
    h, rn, ra, u, d, u_norm, d_norm, margin = build_components(emb, normal_refs, anom_refs, node_idx, rn_idx, ra_idx)
    if input_mode == 'ud_norm':
        x = torch.cat([u_norm, d_norm], dim=1)
    elif input_mode == 'ud_prod_absdiff_norm':
        x = torch.cat([u_norm, d_norm, u_norm * d_norm, torch.abs(u_norm - d_norm)], dim=1)
    elif input_mode == 'ud_mixed_norm':
        x = torch.cat([u, d, u_norm, d_norm], dim=1)
    elif input_mode == 'compact_geometry':
        dot = torch.sum(u_norm * d_norm, dim=1, keepdim=True)
        u_mag = torch.norm(u, p=2, dim=1, keepdim=True)
        d_mag = torch.norm(d, p=2, dim=1, keepdim=True)
        orth = torch.norm(u_norm - dot * d_norm, p=2, dim=1, keepdim=True)
        x = torch.cat([dot, u_mag, d_mag, orth, torch.abs(u_mag - d_mag)], dim=1)
    else:
        raise ValueError(input_mode)
    return x, margin


class Standardizer(nn.Module):
    def __init__(self, mean, std):
        super().__init__(); self.register_buffer('mean', mean); self.register_buffer('std', std.clamp_min(1e-6))
    def forward(self, x): return (x - self.mean) / self.std


class BoundedCorrectionHead(nn.Module):
    def __init__(self, in_dim, hidden=64, dropout=0.1, corr_scale=0.25):
        super().__init__(); self.corr_scale = corr_scale
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        last = self.net[-1]
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
    def forward(self, x): return self.corr_scale * torch.tanh(self.net(x).squeeze(-1))


def diagnostic_metrics(labels, test_idx, margin, corr, score, tau, seed):
    labels_t = labels[test_idx]
    m = margin[test_idx]; c = corr[test_idx]; s = score[test_idx]
    rng = np.random.default_rng(seed + 777)
    auc_m, ap_m = safe_auc(labels_t, m)
    auc_s, ap_s = safe_auc(labels_t, s)
    auc_c, ap_c = safe_auc(labels_t, c)
    auc_negc, ap_negc = safe_auc(labels_t, -c)
    normal_mask = labels_t == 0; anom_mask = labels_t == 1
    corr_r2_margin, corr_resid_std = linear_r2(m, c)
    centered_c = c - np.mean(c)
    metrics = {
        'margin_auc': auc_m, 'margin_ap': ap_m,
        'score_auc': auc_s, 'score_ap': ap_s,
        'delta_auc_vs_margin': auc_s - auc_m,
        'delta_ap_vs_margin': ap_s - ap_m,
        'corr_only_auc': auc_c, 'corr_only_ap': ap_c,
        'neg_corr_auc': auc_negc, 'neg_corr_ap': ap_negc,
        'spearman_score_margin': safe_spearman(s, m),
        'pearson_score_margin': safe_pearson(s, m),
        'corr_mean': float(np.mean(c)),
        'corr_std': float(np.std(c)),
        'corr_abs_mean': float(np.mean(np.abs(c))),
        'corr_centered_std': float(np.std(centered_c)),
        'margin_std': float(np.std(m)),
        'score_std': float(np.std(s)),
        'corr_std_over_margin_std': float(np.std(c) / (np.std(m) + 1e-12)),
        'corr_abs_mean_over_margin_std': float(np.mean(np.abs(c)) / (np.std(m) + 1e-12)),
        'spearman_corr_margin': safe_spearman(c, m),
        'pearson_corr_margin': safe_pearson(c, m),
        'linear_r2_corr_from_margin': corr_r2_margin,
        'corr_resid_std_after_linear_margin': corr_resid_std,
        'rank_flip_rate_sampled': sampled_rank_flip_rate(m, s, rng),
        'top1_margin': top_ratio(labels_t, m, 0.01),
        'top1_score': top_ratio(labels_t, s, 0.01),
        'top5_margin': top_ratio(labels_t, m, 0.05),
        'top5_score': top_ratio(labels_t, s, 0.05),
        'top1_jaccard_margin_score': jaccard(top_indices(m, 0.01), top_indices(s, 0.01)),
        'top5_jaccard_margin_score': jaccard(top_indices(m, 0.05), top_indices(s, 0.05)),
        'top1_boundary_flip_rate': boundary_flip_rate(m, s, 0.01),
        'top5_boundary_flip_rate': boundary_flip_rate(m, s, 0.05),
        'normal_corr_mean': float(np.mean(c[normal_mask])) if np.any(normal_mask) else 0.0,
        'anom_corr_mean': float(np.mean(c[anom_mask])) if np.any(anom_mask) else 0.0,
        'normal_corr_std': float(np.std(c[normal_mask])) if np.any(normal_mask) else 0.0,
        'anom_corr_std': float(np.std(c[anom_mask])) if np.any(anom_mask) else 0.0,
        'corr_anom_minus_normal': float(np.mean(c[anom_mask]) - np.mean(c[normal_mask])) if np.any(normal_mask) and np.any(anom_mask) else 0.0,
        'corr_cohen_d_anom_vs_normal': cohen_d(c[normal_mask], c[anom_mask]) if np.any(normal_mask) and np.any(anom_mask) else 0.0,
        'normal_score_above_tau_rate': float(np.mean(s[normal_mask] > tau)) if np.any(normal_mask) else 0.0,
        'anom_score_above_tau_rate': float(np.mean(s[anom_mask] > tau)) if np.any(anom_mask) else 0.0,
    }
    # Heuristic flags: not conclusions, just routing hints.
    metrics['flag_A_global_shift'] = bool(abs(metrics['corr_mean']) > metrics['corr_std'] and metrics['rank_flip_rate_sampled'] < 0.01)
    metrics['flag_B_margin_linked'] = bool(abs(metrics['spearman_corr_margin']) > 0.7 or metrics['linear_r2_corr_from_margin'] > 0.5)
    metrics['flag_C_too_weak_to_change_rank'] = bool(metrics['rank_flip_rate_sampled'] < 0.01 and metrics['top5_jaccard_margin_score'] > 0.95)
    metrics['flag_D_no_anomaly_separation'] = bool(abs(metrics['corr_cohen_d_anom_vs_normal']) < 0.2 and max(metrics['corr_only_auc'], metrics['neg_corr_auc']) < 0.6)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-root', '--project_root', dest='project_root', default='/data/linziyao/DualRefGAD')
    ap.add_argument('--dataset', default='elliptic')
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train_rate', type=float, default=0.05)
    ap.add_argument('--val_rate', type=float, default=0.0)
    ap.add_argument('--descriptor_mode', choices=['hop_attr','rwse','hop_attr_rwse'], default='hop_attr')
    ap.add_argument('--pn_estimator', choices=['diag_gaussian','pca_residual'], default='pca_residual')
    ap.add_argument('--gn_mode', choices=['label_gate','normal_density','label_gate_density'], default='label_gate')
    ap.add_argument('--ln_mode', choices=['descriptor_similarity','reconstruction_gain'], default='descriptor_similarity')
    ap.add_argument('--ga_mode', choices=['normal_rejection','residual_norm','normal_soft_or'], default='normal_soft_or')
    ap.add_argument('--la_mode', choices=['residual_cosine','descriptor_similarity'], default='descriptor_similarity')
    ap.add_argument('--ablation_mode', choices=['full','no_ra','shuffled_ra','fixed_labeled_normal'], default='full')
    ap.add_argument('--normal_k', type=int, default=4); ap.add_argument('--anom_k', type=int, default=16)
    ap.add_argument('--pp_k', type=int, default=6); ap.add_argument('--hops', type=int, default=2); ap.add_argument('--rw_steps', type=int, default=8); ap.add_argument('--pca_components', type=int, default=32)
    ap.add_argument('--embedding_dim', type=int, default=256); ap.add_argument('--GT_ffn_dim', type=int, default=256); ap.add_argument('--GT_dropout', type=float, default=0.4); ap.add_argument('--GT_attention_dropout', type=float, default=0.4); ap.add_argument('--GT_num_heads', type=int, default=2); ap.add_argument('--GT_num_layers', type=int, default=3)
    ap.add_argument('--sample_rate', type=float, default=0.15); ap.add_argument('--mean', type=float, default=0.02); ap.add_argument('--var', type=float, default=0.01); ap.add_argument('--outlier_beta', type=float, default=0.3); ap.add_argument('--ring_R_max', type=float, default=1.0); ap.add_argument('--ring_R_min', type=float, default=0.3); ap.add_argument('--lambda_rec_tok', type=float, default=1.0); ap.add_argument('--lambda_rec_emb', type=float, default=0.1)
    ap.add_argument('--encode_batch_size', type=int, default=512)
    ap.add_argument('--input_mode', choices=['ud_norm','ud_prod_absdiff_norm','ud_mixed_norm','compact_geometry'], default='ud_prod_absdiff_norm')
    ap.add_argument('--hidden', type=int, default=64); ap.add_argument('--dropout', type=float, default=0.1); ap.add_argument('--corr_scale', type=float, default=0.25); ap.add_argument('--beta', type=float, default=1.0)
    ap.add_argument('--normal_val_frac', type=float, default=0.2); ap.add_argument('--tau_quantile', type=float, default=0.75); ap.add_argument('--temp', type=float, default=0.08); ap.add_argument('--corr_l2', type=float, default=0.05); ap.add_argument('--corr_center', type=float, default=0.01)
    ap.add_argument('--lr', type=float, default=5e-4); ap.add_argument('--weight_decay', type=float, default=1e-3); ap.add_argument('--num_epoch', type=int, default=80)
    ap.add_argument('--wandb', type=lambda x: str(x).lower() in ['1','true','yes'], default=False)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    root = Path(args.project_root); sys.path.insert(0, str(root)); sys.path.insert(0, str(root / 'scripts')); os.chdir(str(root))
    from utils import load_mat
    from VecGAD import VecGAD
    from run_training_degradation_diagnosis import to_dense_features, build_descriptor, NormalModel, select_refs, apply_ablation, reference_purity, build_tokens, encode_tokens_batched

    t0 = time.time(); set_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')
    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat(args.dataset, args.train_rate, args.val_rate, args=args)
    features_np = to_dense_features(args.dataset, features)
    labels_np = np.asarray(ano_label).reshape(-1).astype(int)
    normal_idx = np.asarray(normal_for_train_idx, dtype=int); idx_test = np.asarray(idx_test, dtype=int)
    assert np.sum(labels_np[normal_idx]) == 0, 'Data leakage: train normal contains anomalies'

    z = build_descriptor(args.descriptor_mode, features_np, adj, args.hops, args.rw_steps)
    nm = NormalModel(args.pn_estimator, z, normal_idx, args.pca_components); residual = nm.residual()
    normal_refs, anom_refs, score_meta = select_refs(z, residual, normal_idx, nm, features_np, adj, args, labels_np)
    normal_refs, anom_refs = apply_ablation(normal_refs, anom_refs, normal_idx, labels_np, args)
    pur = reference_purity(normal_refs, anom_refs, labels_np)

    token_tensor = build_tokens(features_np, normal_refs, anom_refs)
    model = VecGAD(features_np.shape[1], args.embedding_dim, 'prelu', args).to(device); model.eval()
    with torch.no_grad(): emb = encode_tokens_batched(model, token_tensor, device, args.encode_batch_size).detach()
    normal_refs_t = torch.as_tensor(normal_refs, dtype=torch.long, device=device)
    anom_refs_t = torch.as_tensor(anom_refs, dtype=torch.long, device=device)
    all_nodes = np.arange(len(labels_np), dtype=np.int64)
    train_normals, val_normals = split_normals(normal_idx, args.seed, args.normal_val_frac)

    with torch.no_grad():
        x_all, margin0 = build_relation_features(emb, normal_refs_t, anom_refs_t, all_nodes, args.input_mode)
    train_t = torch.as_tensor(train_normals, dtype=torch.long, device=device); val_t = torch.as_tensor(val_normals, dtype=torch.long, device=device)
    mean = x_all[train_t].mean(dim=0); std = x_all[train_t].std(dim=0).clamp_min(1e-6)
    scaler = Standardizer(mean, std).to(device); xs_all = scaler(x_all).detach()
    head = BoundedCorrectionHead(xs_all.shape[1], args.hidden, args.dropout, args.corr_scale).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tau = torch.quantile(margin0[train_t].detach(), args.tau_quantile).detach()

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project='DualRefGAD', entity='HCCS', config=vars(args), name=f'stage3_abcd_diag_{args.input_mode}_h{args.hidden}_s{args.seed}')
        wandb.summary.update(pur)

    def normal_loss(nodes_t):
        corr = head(xs_all[nodes_t])
        score = margin0[nodes_t].detach() + args.beta * corr
        suppress = F.softplus((score - tau) / args.temp).mean()
        reg = args.corr_l2 * (corr * corr).mean() + args.corr_center * corr.mean().pow(2)
        return suppress + reg, suppress.detach(), reg.detach(), corr.detach(), score.detach()

    best = {'val_loss': float('inf'), 'epoch': -1}; best_state = None; last = {}
    history = []
    for epoch in range(args.num_epoch + 1):
        if epoch > 0:
            head.train(); opt.zero_grad(); loss, *_ = normal_loss(train_t); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            val_loss, val_suppress, val_reg, _, _ = normal_loss(val_t)
            train_loss, train_suppress, train_reg, _, _ = normal_loss(train_t)
            corr_all = head(xs_all).detach().cpu().numpy()
            margin_np = margin0.detach().cpu().numpy()
            score_all = margin_np + args.beta * corr_all
        row_diag = diagnostic_metrics(labels_np, idx_test, margin_np, corr_all, score_all, float(tau.detach().cpu().item()), args.seed)
        row = {
            'epoch': epoch,
            'train_loss': float(train_loss.cpu().item()), 'train_suppress': float(train_suppress.cpu().item()), 'train_reg': float(train_reg.cpu().item()),
            'val_loss': float(val_loss.cpu().item()), 'val_suppress': float(val_suppress.cpu().item()), 'val_reg': float(val_reg.cpu().item()),
            **row_diag,
        }
        last = row
        if epoch in [0, 1, 5, 10, 20, 40, args.num_epoch]:
            history.append(row)
        if row['val_loss'] < best.get('val_loss', float('inf')):
            best = dict(row); best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if run and (epoch % 10 == 0 or epoch == args.num_epoch):
            import wandb; wandb.log(row, step=epoch)

    result = {
        'status': 'stage3_residual_abcd_diagnostics',
        'dataset': args.dataset,
        'seed': args.seed,
        'config': vars(args),
        'protocol': 'normal-only training; anomaly labels used only for ABCD diagnostic metrics; best epoch selected by normal validation loss',
        'purity': pur,
        'normal_split': {'train_normals': int(len(train_normals)), 'val_normals': int(len(val_normals))},
        'tau': float(tau.detach().cpu().item()),
        'best': best,
        'last': last,
        'history_sparse': history,
        'interpretation_guide': {
            'A_global_shift': 'corr mean dominates corr std and rank_flip is tiny',
            'B_margin_linked': 'corr is strongly correlated with margin or linearly explained by margin',
            'C_too_weak_or_wrong_locality': 'rank_flip/top-k membership changes are tiny despite nonzero correction',
            'D_normal_only_insufficient': 'corr-only has weak anomaly AUC and normal/anomaly corr distributions barely separate',
        },
        'time_sec': time.time() - t0,
    }
    out = Path(args.out) if args.out else root / f'outputs/stage3_probe/stage3_residual_abcd_diag_s{args.seed}_e{args.num_epoch}.json'
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'out': str(out), 'protocol': result['protocol'], 'tau': result['tau'], 'normal_split': result['normal_split'], 'best': best, 'purity': pur, 'time_sec': result['time_sec']}, indent=2, ensure_ascii=False), flush=True)
    if run:
        import wandb
        wandb.summary.update({k: v for k, v in best.items() if isinstance(v, (int, float, bool))})
        run.finish()


if __name__ == '__main__':
    main()
