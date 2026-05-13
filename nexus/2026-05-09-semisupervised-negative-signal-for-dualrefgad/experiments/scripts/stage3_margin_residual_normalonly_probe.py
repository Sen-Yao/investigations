#!/usr/bin/env python3
"""Stage-3 Protocol-clean Margin Residual Probe.

Purpose
-------
This is a diagnostic probe, not a proposed final method. It tests whether a
small, bounded correction head trained ONLY on labeled normal nodes can improve
DualRef margin ranking. The final score is

    s_i = margin_i + beta * correction_theta(phi_i)

but the correction is deliberately constrained:
  - small MLP capacity;
  - tanh-bounded output;
  - L2 penalty on correction magnitude;
  - model selection by normal-only validation loss, never anomaly labels.

Training objective
------------------
Known-normal nodes should not receive high anomaly scores. We set a threshold tau
from train-normal margin quantile and minimize softplus((score - tau) / temp)
for normal nodes. This only uses labeled normal nodes. Anomaly labels are used
only for diagnostic evaluation after each epoch.
"""
import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr


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


def top_ratio(labels, scores, frac):
    n = max(1, int(len(scores) * frac))
    idx = np.argsort(-scores)[:n]
    return float(np.mean(labels[idx]))


def split_normals(normal_idx, seed, val_frac):
    rng = np.random.default_rng(seed)
    arr = np.asarray(normal_idx, dtype=np.int64).copy()
    rng.shuffle(arr)
    n_val = max(1, int(len(arr) * val_frac)) if val_frac > 0 else 0
    val = arr[:n_val]
    train = arr[n_val:]
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
    u = h - rn
    d = ra - rn
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
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std.clamp_min(1e-6))
    def forward(self, x):
        return (x - self.mean) / self.std


class BoundedCorrectionHead(nn.Module):
    def __init__(self, in_dim, hidden=64, dropout=0.1, corr_scale=0.25):
        super().__init__()
        self.corr_scale = corr_scale
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        # Start as exact no-op: score == margin at epoch 0.
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x):
        return self.corr_scale * torch.tanh(self.net(x).squeeze(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-root', default='/data/linziyao/DualRefGAD')
    ap.add_argument('--dataset', default='elliptic')
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train_rate', type=float, default=0.05)
    ap.add_argument('--val_rate', type=float, default=0.0)
    # Descriptor/ref parameters
    ap.add_argument('--descriptor_mode', choices=['hop_attr','rwse','hop_attr_rwse'], default='hop_attr')
    ap.add_argument('--pn_estimator', choices=['diag_gaussian','pca_residual'], default='pca_residual')
    ap.add_argument('--gn_mode', choices=['label_gate','normal_density','label_gate_density'], default='label_gate')
    ap.add_argument('--ln_mode', choices=['descriptor_similarity','reconstruction_gain'], default='descriptor_similarity')
    ap.add_argument('--ga_mode', choices=['normal_rejection','residual_norm','normal_soft_or'], default='normal_soft_or')
    ap.add_argument('--la_mode', choices=['residual_cosine','descriptor_similarity'], default='descriptor_similarity')
    ap.add_argument('--ablation_mode', choices=['full','no_ra','shuffled_ra','fixed_labeled_normal'], default='full')
    ap.add_argument('--normal_k', type=int, default=4)
    ap.add_argument('--anom_k', type=int, default=16)
    ap.add_argument('--pp_k', type=int, default=6)
    ap.add_argument('--hops', type=int, default=2)
    ap.add_argument('--rw_steps', type=int, default=8)
    ap.add_argument('--pca_components', type=int, default=32)
    # Encoder parameters
    ap.add_argument('--embedding_dim', type=int, default=256)
    ap.add_argument('--GT_ffn_dim', type=int, default=256)
    ap.add_argument('--GT_dropout', type=float, default=0.4)
    ap.add_argument('--GT_attention_dropout', type=float, default=0.4)
    ap.add_argument('--GT_num_heads', type=int, default=2)
    ap.add_argument('--GT_num_layers', type=int, default=3)
    # Token/sampling parameters
    ap.add_argument('--sample_rate', type=float, default=0.15)
    ap.add_argument('--mean', type=float, default=0.02)
    ap.add_argument('--var', type=float, default=0.01)
    ap.add_argument('--outlier_beta', type=float, default=0.3)
    ap.add_argument('--ring_R_max', type=float, default=1.0)
    ap.add_argument('--ring_R_min', type=float, default=0.3)
    ap.add_argument('--lambda_rec_tok', type=float, default=1.0)
    ap.add_argument('--lambda_rec_emb', type=float, default=0.1)
    ap.add_argument('--encode_batch_size', type=int, default=512)
    # Residual probe parameters
    ap.add_argument('--input_mode', choices=['ud_norm','ud_prod_absdiff_norm','ud_mixed_norm','compact_geometry'], default='ud_prod_absdiff_norm')
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--corr_scale', type=float, default=0.25)
    ap.add_argument('--beta', type=float, default=1.0)
    ap.add_argument('--normal_val_frac', type=float, default=0.2)
    ap.add_argument('--tau_quantile', type=float, default=0.75)
    ap.add_argument('--temp', type=float, default=0.08)
    ap.add_argument('--corr_l2', type=float, default=0.05)
    ap.add_argument('--corr_center', type=float, default=0.01)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-3)
    ap.add_argument('--num_epoch', type=int, default=80)
    ap.add_argument('--wandb', type=lambda x: str(x).lower() in ['1','true','yes'], default=False)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    root = Path(args.project_root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / 'scripts'))
    os.chdir(str(root))

    from utils import load_mat
    from VecGAD import VecGAD
    from run_training_degradation_diagnosis import (
        to_dense_features, build_descriptor, NormalModel, select_refs, apply_ablation,
        reference_purity, build_tokens, encode_tokens_batched,
    )

    t0 = time.time()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')

    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat(args.dataset, args.train_rate, args.val_rate, args=args)
    features_np = to_dense_features(args.dataset, features)
    labels_np = np.asarray(ano_label).reshape(-1).astype(int)
    normal_idx = np.asarray(normal_for_train_idx, dtype=int)
    idx_test = np.asarray(idx_test, dtype=int)
    assert np.sum(labels_np[normal_idx]) == 0, 'Data leakage: train normal contains anomalies'

    z = build_descriptor(args.descriptor_mode, features_np, adj, args.hops, args.rw_steps)
    nm = NormalModel(args.pn_estimator, z, normal_idx, args.pca_components)
    residual = nm.residual()
    normal_refs, anom_refs, score_meta = select_refs(z, residual, normal_idx, nm, features_np, adj, args, labels_np)
    normal_refs, anom_refs = apply_ablation(normal_refs, anom_refs, normal_idx, labels_np, args)
    pur = reference_purity(normal_refs, anom_refs, labels_np)

    token_tensor = build_tokens(features_np, normal_refs, anom_refs)
    model = VecGAD(features_np.shape[1], args.embedding_dim, 'prelu', args).to(device)
    model.eval()
    with torch.no_grad():
        emb = encode_tokens_batched(model, token_tensor, device, args.encode_batch_size).detach()

    normal_refs_t = torch.as_tensor(normal_refs, dtype=torch.long, device=device)
    anom_refs_t = torch.as_tensor(anom_refs, dtype=torch.long, device=device)
    all_nodes = np.arange(len(labels_np), dtype=np.int64)
    train_normals, val_normals = split_normals(normal_idx, args.seed, args.normal_val_frac)

    with torch.no_grad():
        x_all, margin0 = build_relation_features(emb, normal_refs_t, anom_refs_t, all_nodes, args.input_mode)
        margin0_np = margin0.detach().cpu().numpy()

    train_t = torch.as_tensor(train_normals, dtype=torch.long, device=device)
    val_t = torch.as_tensor(val_normals, dtype=torch.long, device=device)
    x_train_raw = x_all[train_t]
    mean = x_train_raw.mean(dim=0)
    std = x_train_raw.std(dim=0).clamp_min(1e-6)
    scaler = Standardizer(mean, std).to(device)
    xs_all = scaler(x_all).detach()

    in_dim = xs_all.shape[1]
    head = BoundedCorrectionHead(in_dim, args.hidden, args.dropout, args.corr_scale).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    margin_train = margin0[train_t].detach()
    tau = torch.quantile(margin_train, args.tau_quantile).detach()

    margin_auc, margin_ap = safe_auc(labels_np[idx_test], margin0_np[idx_test])
    margin_top1 = top_ratio(labels_np[idx_test], margin0_np[idx_test], 0.01)
    margin_top5 = top_ratio(labels_np[idx_test], margin0_np[idx_test], 0.05)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(
            project='DualRefGAD', entity='HCCS', config=vars(args),
            name=f'stage3_margin_residual_normalonly_{args.input_mode}_h{args.hidden}_s{args.seed}'
        )
        wandb.summary.update(pur)

    def normal_loss(nodes_t):
        corr = head(xs_all[nodes_t])
        score = margin0[nodes_t].detach() + args.beta * corr
        suppress = F.softplus((score - tau) / args.temp).mean()
        reg = args.corr_l2 * (corr * corr).mean() + args.corr_center * corr.mean().pow(2)
        return suppress + reg, suppress.detach(), reg.detach(), corr.detach(), score.detach()

    best = {'val_loss': float('inf'), 'epoch': -1}
    last = {}
    best_state = None

    for epoch in range(args.num_epoch + 1):
        if epoch > 0:
            head.train(); opt.zero_grad()
            loss, _, _, _, _ = normal_loss(train_t)
            loss.backward(); opt.step()
        else:
            loss = torch.tensor(0.0, device=device)

        head.eval()
        with torch.no_grad():
            val_loss, val_suppress, val_reg, val_corr, val_score = normal_loss(val_t)
            train_loss, train_suppress, train_reg, train_corr, train_score = normal_loss(train_t)
            corr_all = head(xs_all).detach()
            score_all = (margin0.detach() + args.beta * corr_all).detach().cpu().numpy()
            corr_np = corr_all.detach().cpu().numpy()

        auc, apv = safe_auc(labels_np[idx_test], score_all[idx_test])
        row = {
            'epoch': epoch,
            'train_loss': float(train_loss.cpu().item()),
            'val_loss': float(val_loss.cpu().item()),
            'val_suppress': float(val_suppress.cpu().item()),
            'val_reg': float(val_reg.cpu().item()),
            'test_auc': auc,
            'test_ap': apv,
            'top1_ratio': top_ratio(labels_np[idx_test], score_all[idx_test], 0.01),
            'top5_ratio': top_ratio(labels_np[idx_test], score_all[idx_test], 0.05),
            'spearman_score_margin': safe_spearman(score_all[idx_test], margin0_np[idx_test]),
            'corr_mean_test': float(np.mean(corr_np[idx_test])),
            'corr_std_test': float(np.std(corr_np[idx_test])),
            'corr_abs_mean_test': float(np.mean(np.abs(corr_np[idx_test]))),
            'margin_auc': margin_auc,
            'margin_ap': margin_ap,
        }
        last = row
        if row['val_loss'] < best.get('val_loss', float('inf')):
            best = dict(row)
            best['delta_auc_vs_margin'] = row['test_auc'] - margin_auc
            best['delta_ap_vs_margin'] = row['test_ap'] - margin_ap
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if run and (epoch % 10 == 0 or epoch == args.num_epoch):
            import wandb
            wandb.log(row, step=epoch)

    result = {
        'status': 'stage3_margin_residual_normalonly_probe',
        'dataset': args.dataset,
        'seed': args.seed,
        'config': vars(args),
        'protocol': 'normal-only training; anomaly labels used only for diagnostic evaluation; best epoch selected by normal validation loss',
        'purity': pur,
        'normal_split': {'train_normals': int(len(train_normals)), 'val_normals': int(len(val_normals))},
        'tau': float(tau.detach().cpu().item()),
        'margin_baseline': {
            'auc': margin_auc,
            'ap': margin_ap,
            'top1_ratio': margin_top1,
            'top5_ratio': margin_top5,
            'spearman': 1.0,
        },
        'best': best,
        'last': last,
        'time_sec': time.time() - t0,
    }

    out = Path(args.out) if args.out else root / 'outputs/stage3_probe/stage3_margin_residual_normalonly.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({
        'out': str(out),
        'protocol': result['protocol'],
        'tau': result['tau'],
        'normal_split': result['normal_split'],
        'margin_baseline': result['margin_baseline'],
        'best': best,
        'last': last,
        'purity': pur,
        'time_sec': result['time_sec'],
    }, indent=2, ensure_ascii=False), flush=True)

    if run:
        import wandb
        wandb.summary.update({
            'best_val_loss': best['val_loss'],
            'best_epoch': best['epoch'],
            'best_test_auc': best['test_auc'],
            'best_test_ap': best['test_ap'],
            'delta_auc_vs_margin': best['delta_auc_vs_margin'],
            'delta_ap_vs_margin': best['delta_ap_vs_margin'],
            'margin_auc': margin_auc,
            'margin_ap': margin_ap,
        })
        run.finish()


if __name__ == '__main__':
    main()
