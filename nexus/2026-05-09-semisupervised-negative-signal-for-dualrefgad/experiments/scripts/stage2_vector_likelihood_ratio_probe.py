#!/usr/bin/env python3
"""Stage-2 Vector-Conditioned Likelihood-Ratio Probe.

Core idea: Train conditional density p(phi | normalize(d)) on normal nodes only.
Anomaly score = likelihood ratio:
    log p(phi_i | normalize(d_i)) - log p(phi_i | mean_normal_direction)

This diagnostic tests whether scalar ||d|| failed because it discarded directional
geometry. It still uses normal-only training; anomaly labels are diagnostics only.
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
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False


def safe_auc(y, s):
    try: return float(roc_auc_score(y, s)), float(average_precision_score(y, s))
    except Exception: return 0.0, 0.0


def safe_spearman(a,b):
    try:
        v=spearmanr(np.asarray(a), np.asarray(b)).correlation
        return float(0.0 if np.isnan(v) else v)
    except Exception: return 0.0


def top_ratio(labels, scores, frac):
    n=max(1, int(len(scores)*frac)); idx=np.argsort(-scores)[:n]
    return float(np.mean(labels[idx]))


def build_components(emb, normal_refs, anom_refs, node_idx, rn_idx=None, ra_idx=None):
    node_idx=np.asarray(node_idx,dtype=np.int64)
    rn_base=node_idx if rn_idx is None else np.asarray(rn_idx,dtype=np.int64)
    ra_base=node_idx if ra_idx is None else np.asarray(ra_idx,dtype=np.int64)
    h=emb[node_idx]
    rn=emb[normal_refs[rn_base]].mean(dim=1)
    ra=emb[anom_refs[ra_base]].mean(dim=1)
    u=h-rn; d=ra-rn
    margin=torch.sum(F.normalize(u,p=2,dim=1,eps=1e-12)*F.normalize(d,p=2,dim=1,eps=1e-12), dim=1)
    d_mag=torch.norm(d, p=2, dim=1)
    d_cond=F.normalize(d, p=2, dim=1, eps=1e-12)  # vector conditioning
    return h,rn,ra,u,d,margin,d_mag,d_cond


def build_relation_features(emb, normal_refs, anom_refs, node_idx, input_mode, rn_idx=None, ra_idx=None):
    h,rn,ra,u,d,margin,d_mag,d_cond=build_components(emb, normal_refs, anom_refs, node_idx, rn_idx, ra_idx)
    u_norm = F.normalize(u, p=2, dim=1, eps=1e-12)
    d_norm_vec = F.normalize(d, p=2, dim=1, eps=1e-12)
    if input_mode == 'ud_prod_absdiff_norm':
        x=torch.cat([u_norm,d_norm_vec,u_norm*d_norm_vec,torch.abs(u_norm-d_norm_vec)], dim=1)
    elif input_mode == 'ud_norm':
        x=torch.cat([u_norm,d_norm_vec], dim=1)
    elif input_mode == 'ud_mixed_norm':
        x=torch.cat([u,d,u_norm,d_norm_vec], dim=1)
    else:
        raise ValueError(input_mode)
    return x, margin, d_mag, d_cond


class Standardizer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std.clamp_min(1e-6))
    def forward(self,x): return (x-self.mean)/self.std


class ConditionScaler(nn.Module):
    """Scale conditioning vector feature-wise to reasonable range."""
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std.clamp_min(1e-6))
    def forward(self, c): return (c - self.mean) / self.std


class ConditionalCoupling(nn.Module):
    """Coupling layer with conditioning variable."""
    def __init__(self, dim, hidden, cond_dim, mask):
        super().__init__()
        self.register_buffer('mask', mask)
        # Net takes [masked_x, condition] -> scale and shift for unmasked
        self.net = nn.Sequential(
            nn.Linear(dim + cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim * 2)
        )

    def forward(self, x, c):
        # x: [batch, dim], c: [batch, cond_dim]
        xm = x * self.mask  # masked portion stays unchanged
        xc = torch.cat([xm, c], dim=1)  # concat with condition
        st = self.net(xc)
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s) * 2.0  # limit scale
        inv = 1.0 - self.mask
        y = xm + inv * (x * torch.exp(s) + t)
        logdet = (inv * s).sum(dim=1)
        return y, logdet


class ConditionalRealNVP(nn.Module):
    """Conditional Normalizing Flow (RealNVP style)."""
    def __init__(self, dim, hidden=256, layers=4, cond_dim=1):
        super().__init__()
        self.cond_dim = cond_dim
        mods = []
        for k in range(layers):
            mask = ((torch.arange(dim) + k) % 2).float()
            mods.append(ConditionalCoupling(dim, hidden, cond_dim, mask))
        self.mods = nn.ModuleList(mods)

    def forward(self, x, c):
        """Forward pass: x -> z, returns log determinant."""
        logdet = torch.zeros(x.shape[0], device=x.device)
        z = x
        for m in self.mods:
            z, ld = m(z, c)
            logdet = logdet + ld
        return z, logdet

    def log_prob(self, x, c):
        """Compute log probability under standard normal base."""
        z, ld = self.forward(x, c)
        # Base distribution: standard normal N(0, I)
        base = -0.5 * (z * z + np.log(2 * np.pi)).sum(dim=1)
        return base + ld


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
    # Model parameters
    ap.add_argument('--embedding_dim', type=int, default=256)
    ap.add_argument('--GT_ffn_dim', type=int, default=256)
    ap.add_argument('--GT_dropout', type=float, default=0.4)
    ap.add_argument('--GT_attention_dropout', type=float, default=0.4)
    ap.add_argument('--GT_num_heads', type=int, default=2)
    ap.add_argument('--GT_num_layers', type=int, default=3)
    # Sampling parameters
    ap.add_argument('--sample_rate', type=float, default=0.15)
    ap.add_argument('--mean', type=float, default=0.02)
    ap.add_argument('--var', type=float, default=0.01)
    ap.add_argument('--outlier_beta', type=float, default=0.3)
    ap.add_argument('--ring_R_max', type=float, default=1.0)
    ap.add_argument('--ring_R_min', type=float, default=0.3)
    ap.add_argument('--lambda_rec_tok', type=float, default=1.0)
    ap.add_argument('--lambda_rec_emb', type=float, default=0.1)
    ap.add_argument('--encode_batch_size', type=int, default=512)
    # Vector likelihood-ratio specific
    ap.add_argument('--input_mode', choices=['ud_prod_absdiff_norm','ud_norm','ud_mixed_norm'], default='ud_prod_absdiff_norm')
    ap.add_argument('--flow_layers', type=int, default=4)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--num_epoch', type=int, default=80)
    # Likelihood ratio strategy
    ap.add_argument('--lr_strategy', choices=['self_vs_mean_direction'], default='self_vs_mean_direction')
    ap.add_argument('--wandb', type=lambda x: str(x).lower() in ['1','true','yes'], default=False)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    root = Path(args.project_root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / 'scripts'))
    os.chdir(str(root))

    from utils import load_mat
    from VecGAD import VecGAD
    from run_training_degradation_diagnosis import to_dense_features, build_descriptor, NormalModel, select_refs, apply_ablation, reference_purity, build_tokens, encode_tokens_batched

    t0 = time.time()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')

    # Load data
    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat(args.dataset, args.train_rate, args.val_rate, args=args)
    features_np = to_dense_features(args.dataset, features)
    labels_np = np.asarray(ano_label).reshape(-1).astype(int)
    normal_idx = np.asarray(normal_for_train_idx, dtype=int)
    idx_test = np.asarray(idx_test, dtype=int)
    assert np.sum(labels_np[normal_idx]) == 0, 'Data leakage: train normal contains anomalies'

    # Build descriptor and references
    z = build_descriptor(args.descriptor_mode, features_np, adj, args.hops, args.rw_steps)
    nm = NormalModel(args.pn_estimator, z, normal_idx, args.pca_components)
    residual = nm.residual()
    normal_refs, anom_refs, score_meta = select_refs(z, residual, normal_idx, nm, features_np, adj, args, labels_np)
    normal_refs, anom_refs = apply_ablation(normal_refs, anom_refs, normal_idx, labels_np, args)
    pur = reference_purity(normal_refs, anom_refs, labels_np)

    # Encode with frozen VecGAD
    token_tensor = build_tokens(features_np, normal_refs, anom_refs)
    model = VecGAD(features_np.shape[1], args.embedding_dim, 'prelu', args).to(device)
    model.eval()
    with torch.no_grad():
        emb = encode_tokens_batched(model, token_tensor, device, args.encode_batch_size).detach()

    normal_refs_t = torch.as_tensor(normal_refs, dtype=torch.long, device=device)
    anom_refs_t = torch.as_tensor(anom_refs, dtype=torch.long, device=device)
    all_nodes = np.arange(len(labels_np), dtype=np.int64)

    # Build relation features and vector conditioning variable normalize(d)
    with torch.no_grad():
        x_all, margin0, d_mag_all, d_cond_all = build_relation_features(emb, normal_refs_t, anom_refs_t, all_nodes, args.input_mode)

    x_train = x_all[torch.as_tensor(normal_idx, dtype=torch.long, device=device)]
    d_cond_train = d_cond_all[torch.as_tensor(normal_idx, dtype=torch.long, device=device)]

    # Standardize features
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0).clamp_min(1e-6)
    scaler = Standardizer(mean, std).to(device)
    xs_train = scaler(x_train).detach()
    xs_all = scaler(x_all).detach()

    # Standardize vector conditioning variable normalize(d) feature-wise
    c_mean = d_cond_train.mean(dim=0)
    c_std = d_cond_train.std(dim=0).clamp_min(1e-6)
    c_scaler = ConditionScaler(c_mean, c_std).to(device)
    c_train = c_scaler(d_cond_train).detach()
    c_all = c_scaler(d_cond_all).detach()

    margin0_np = margin0.detach().cpu().numpy()
    margin_auc, margin_ap = safe_auc(labels_np[idx_test], margin0_np[idx_test])

    # WandB setup
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(
            project='DualRefGAD',
            entity='HCCS',
            config=vars(args),
            name=f'stage2_vector_lr_{args.lr_strategy}_{args.input_mode}_h{args.hidden}_l{args.flow_layers}_s{args.seed}'
        )
        wandb.summary.update(pur)

    # Train conditional density p(phi | ||d||)
    density = ConditionalRealNVP(xs_train.shape[1], args.hidden, args.flow_layers, cond_dim=c_train.shape[1]).to(device)
    opt = torch.optim.AdamW(density.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = None
    best_nll = 1e9
    best_epoch = -1
    c_normal_mean = c_train.mean(dim=0)  # reference direction for likelihood ratio

    for epoch in range(args.num_epoch + 1):
        if epoch > 0:
            density.train()
            opt.zero_grad()
            nll = (-density.log_prob(xs_train, c_train)).mean()
            nll.backward()
            opt.step()

        density.eval()
        with torch.no_grad():
            train_nll = float((-density.log_prob(xs_train, c_train)).mean().item())

            # Compute likelihood ratio score
            # s_i = log p(phi_i | normalize(d_i)) - log p(phi_i | mean_normal_direction)
            log_p_self = density.log_prob(xs_all, c_all)
            log_p_mean = density.log_prob(xs_all, c_normal_mean.unsqueeze(0).expand(xs_all.shape[0], -1))
            lr_score = (log_p_self - log_p_mean).detach().cpu().numpy()

        if train_nll < best_nll:
            best_nll = train_nll
            best_score = lr_score
            best_epoch = epoch

        if run and (epoch % 10 == 0 or epoch == args.num_epoch):
            auc, apv = safe_auc(labels_np[idx_test], lr_score[idx_test])
            import wandb
            wandb.log({
                'epoch': epoch,
                'train_nll': train_nll,
                'test_auc': auc,
                'test_ap': apv,
                'top1_ratio': top_ratio(labels_np[idx_test], lr_score[idx_test], 0.01),
                'top5_ratio': top_ratio(labels_np[idx_test], lr_score[idx_test], 0.05),
                'spearman_lr_margin': safe_spearman(lr_score[idx_test], margin0_np[idx_test])
            }, step=epoch)

    score = best_score
    auc, apv = safe_auc(labels_np[idx_test], score[idx_test])

    result = {
        'status': 'stage2_vector_likelihood_ratio_probe',
        'dataset': args.dataset,
        'seed': args.seed,
        'config': vars(args),
        'purity': pur,
        'best': {
            'epoch': best_epoch,
            'train_nll': best_nll,
            'test_auc': auc,
            'test_ap': apv,
            'top1_ratio': top_ratio(labels_np[idx_test], score[idx_test], 0.01),
            'top5_ratio': top_ratio(labels_np[idx_test], score[idx_test], 0.05),
            'spearman_lr_margin': safe_spearman(score[idx_test], margin0_np[idx_test])
        },
        'margin_baseline': {
            'auc': margin_auc,
            'ap': margin_ap,
            'top1_ratio': top_ratio(labels_np[idx_test], margin0_np[idx_test], 0.01),
            'top5_ratio': top_ratio(labels_np[idx_test], margin0_np[idx_test], 0.05)
        },
        'density_probe_baseline': {
            'auc': 0.6674,  # from previous experiment
            'ap': 0.3213,
        },
        'time_sec': time.time() - t0
    }

    out = Path(args.out) if args.out else root / 'outputs/stage2_probe' / f'stage2_vector_lr_probe_s{args.seed}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

    print(json.dumps({
        'out': str(out),
        'margin_baseline': result['margin_baseline'],
        'density_probe_baseline': result['density_probe_baseline'],
        'best': result['best'],
        'purity': pur,
        'time_sec': result['time_sec']
    }, indent=2, ensure_ascii=False), flush=True)

    if run:
        import wandb
        wandb.summary.update({
            'best_epoch': best_epoch,
            'best_train_nll': best_nll,
            'best_test_auc': auc,
            'best_test_ap': apv,
            'best_top1_ratio': result['best']['top1_ratio'],
            'best_top5_ratio': result['best']['top5_ratio'],
            'best_spearman_margin': result['best']['spearman_lr_margin'],
            'margin_auc': margin_auc,
            'margin_ap': margin_ap,
            'margin_top1_ratio': result['margin_baseline']['top1_ratio'],
            'margin_top5_ratio': result['margin_baseline']['top5_ratio']
        })
        run.finish()


if __name__ == '__main__':
    main()