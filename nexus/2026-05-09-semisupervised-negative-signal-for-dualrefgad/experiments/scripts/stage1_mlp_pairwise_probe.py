#!/usr/bin/env python3
"""Stage-1 DualRef margin calibration probe.

Head-only, encoder/reference frozen diagnostic.
Uses MLP + pairwise ranking loss, with match-sweep reference construction.
No WandB required by default.
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


def l2_rows_t(x):
    return F.normalize(x, p=2, dim=1, eps=1e-12)


def metrics(y, s):
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def safe_spearman(a,b):
    try:
        v=spearmanr(np.asarray(a), np.asarray(b)).correlation
        return float(0.0 if np.isnan(v) else v)
    except Exception:
        return 0.0


def zscore_np(x):
    x=np.asarray(x, dtype=np.float64)
    return (x-x.mean())/(x.std()+1e-12)


def top_ratio(labels, scores, frac):
    n=max(1, int(len(scores)*frac))
    idx=np.argsort(-scores)[:n]
    return float(np.mean(labels[idx]))


def build_features(emb, normal_refs, anom_refs, node_idx, rn_idx=None, ra_idx=None):
    node_idx=np.asarray(node_idx, dtype=np.int64)
    rn_base=node_idx if rn_idx is None else np.asarray(rn_idx, dtype=np.int64)
    ra_base=node_idx if ra_idx is None else np.asarray(ra_idx, dtype=np.int64)
    h=emb[node_idx]
    rn=emb[normal_refs[rn_base]].mean(dim=1)
    ra=emb[anom_refs[ra_base]].mean(dim=1)
    u=h-rn; d=ra-rn
    margin=torch.sum(l2_rows_t(u)*l2_rows_t(d), dim=1)
    u_norm=torch.linalg.norm(u, dim=1)
    d_norm=torch.linalg.norm(d, dim=1)
    dist_rn=torch.linalg.norm(h-rn, dim=1)
    dist_ra=torch.linalg.norm(h-ra, dim=1)
    ra_closer=dist_rn-dist_ra
    return torch.stack([margin, u_norm, d_norm, dist_rn, dist_ra, ra_closer], dim=1)


class MLPHead(nn.Module):
    def __init__(self, in_dim=6, hidden=16, dropout=0.0):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project-root', default='/data/linziyao/DualRefGAD')
    ap.add_argument('--dataset', default='elliptic')
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train_rate', type=float, default=0.05)
    ap.add_argument('--val_rate', type=float, default=0.0)
    # Fixed match-sweep ref defaults
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
    ap.add_argument('--hidden', type=int, default=16); ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--num_epoch', type=int, default=200); ap.add_argument('--gamma', type=float, default=0.1)
    ap.add_argument('--neg_mode', choices=['N1_context','N2_direction'], default='N2_direction')
    ap.add_argument('--preserve_weight', type=float, default=0.05)
    ap.add_argument('--out', default='')
    args=ap.parse_args()

    root=Path(args.project_root); sys.path.insert(0, str(root)); sys.path.insert(0, str(root/'scripts')); os.chdir(str(root))
    from utils import load_mat
    from VecGAD import VecGAD
    from run_training_degradation_diagnosis import to_dense_features, build_descriptor, NormalModel, select_refs, apply_ablation, reference_purity, build_tokens, encode_tokens_batched

    t0=time.time(); set_seed(args.seed)
    device=torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device>=0 else 'cpu')
    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx=load_mat(args.dataset,args.train_rate,args.val_rate,args=args)
    features_np=to_dense_features(args.dataset, features); labels_np=np.asarray(ano_label).reshape(-1).astype(int)
    normal_idx=np.asarray(normal_for_train_idx,dtype=int); idx_test=np.asarray(idx_test,dtype=int); idx_val=np.asarray(idx_val,dtype=int)
    assert np.sum(labels_np[normal_idx]) == 0, 'Data leakage: train normal contains anomalies'
    z=build_descriptor(args.descriptor_mode, features_np, adj, args.hops, args.rw_steps)
    nm=NormalModel(args.pn_estimator, z, normal_idx, args.pca_components); residual=nm.residual()
    normal_refs, anom_refs, score_meta=select_refs(z, residual, normal_idx, nm, features_np, adj, args, labels_np)
    normal_refs, anom_refs=apply_ablation(normal_refs, anom_refs, normal_idx, labels_np, args)
    pur=reference_purity(normal_refs, anom_refs, labels_np)
    token_tensor=build_tokens(features_np, normal_refs, anom_refs)
    model=VecGAD(features_np.shape[1], args.embedding_dim, 'prelu', args).to(device); model.eval()
    with torch.no_grad(): emb=encode_tokens_batched(model, token_tensor, device, args.encode_batch_size).detach()
    normal_refs_t=torch.as_tensor(normal_refs, dtype=torch.long, device=device); anom_refs_t=torch.as_tensor(anom_refs, dtype=torch.long, device=device)

    all_nodes=np.arange(len(labels_np), dtype=np.int64)
    with torch.no_grad():
        x_all0=build_features(emb, normal_refs_t, anom_refs_t, all_nodes)
        margin0=x_all0[:,0].detach().cpu().numpy()
    margin_auc, margin_ap=metrics(labels_np[idx_test], margin0[idx_test])

    # Standardize features with train-normal positive tuples only.
    x_train0=build_features(emb, normal_refs_t, anom_refs_t, normal_idx).detach()
    mu=x_train0.mean(dim=0, keepdim=True); std=x_train0.std(dim=0, keepdim=True)+1e-6
    head=MLPHead(6,args.hidden,args.dropout).to(device)
    opt=torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng=np.random.default_rng(args.seed)
    best={'auc':-1,'ap':-1,'epoch':-1,'spearman':0,'top1':0,'top5':0}
    history=[]
    normal_np=normal_idx.copy(); n=len(normal_np)
    for epoch in range(args.num_epoch+1):
        if epoch>0:
            head.train(); opt.zero_grad()
            c=normal_np[rng.permutation(n)]
            same=c==normal_np
            if np.any(same): c[same]=normal_np[(np.where(same)[0]+1)%n]
            x_pos=(build_features(emb, normal_refs_t, anom_refs_t, normal_np)-mu)/std
            if args.neg_mode=='N1_context':
                x_neg=(build_features(emb, normal_refs_t, anom_refs_t, normal_np, rn_idx=c, ra_idx=c)-mu)/std
            else:
                x_neg=(build_features(emb, normal_refs_t, anom_refs_t, normal_np, rn_idx=None, ra_idx=c)-mu)/std
            s_pos=head(x_pos); s_neg=head(x_neg)
            loss_rank=F.softplus(s_pos - s_neg + args.gamma).mean()
            x_all=(x_all0-mu)/std
            s_all=head(x_all)
            # Preservation: z-scored node scores should not invert epoch0 margin.
            s_train=s_all[torch.as_tensor(normal_np, dtype=torch.long, device=device)]
            m_train=torch.as_tensor(margin0[normal_np], dtype=torch.float32, device=device)
            s_z=(s_train-s_train.mean())/(s_train.std()+1e-6); m_z=(m_train-m_train.mean())/(m_train.std()+1e-6)
            loss_pres=F.mse_loss(s_z, m_z)
            loss=loss_rank + args.preserve_weight*loss_pres
            loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            x_all=(x_all0-mu)/std
            s_all=head(x_all).detach().cpu().numpy()
        auc, apv=metrics(labels_np[idx_test], s_all[idx_test])
        sp=safe_spearman(s_all[idx_test], margin0[idx_test])
        row={'epoch':epoch,'test_auc':auc,'test_ap':apv,'spearman_score_margin':sp,'top1_ratio':top_ratio(labels_np[idx_test], s_all[idx_test], 0.01),'top5_ratio':top_ratio(labels_np[idx_test], s_all[idx_test], 0.05)}
        if epoch%10==0 or epoch==args.num_epoch: history.append(row)
        if auc+apv > best['auc']+best['ap']:
            best={'auc':auc,'ap':apv,'epoch':epoch,'spearman':sp,'top1':row['top1_ratio'],'top5':row['top5_ratio']}
    result={'status':'stage1_mlp_pairwise_head_only_probe','dataset':args.dataset,'seed':args.seed,'config':vars(args),'purity':pur,'margin_baseline':{'auc':margin_auc,'ap':margin_ap,'top1_ratio':top_ratio(labels_np[idx_test], margin0[idx_test], 0.01),'top5_ratio':top_ratio(labels_np[idx_test], margin0[idx_test], 0.05)},'best':best,'last':history[-1] if history else None,'history':history,'time_sec':time.time()-t0}
    out=Path(args.out) if args.out else root/'outputs/stage1_probe/stage1_mlp_pairwise_seed0.json'
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'out':str(out),'margin_baseline':result['margin_baseline'],'best':best,'last':result['last'],'purity':pur,'time_sec':result['time_sec']}, indent=2, ensure_ascii=False))

if __name__=='__main__': main()
