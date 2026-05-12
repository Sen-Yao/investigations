#!/usr/bin/env python3
"""Closed-form DualRef normality diagnostic.

Read-only mechanism diagnostic: no training, no sweep, no wandb.
Compares whether DualRef relation geometry gives better normality scores than raw embedding one-class center.
"""
import argparse, json, os, sys, time, random
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False


def l2_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def safe_metrics(labels, scores, idx):
    y=np.asarray(labels)[idx]; s=np.asarray(scores)[idx]
    return {"auc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s))}


def center_distance_score(z, normal_idx):
    c=z[normal_idx].mean(axis=0, keepdims=True)
    return np.linalg.norm(z-c, axis=1)


def mahalanobis_diag_score(z, normal_idx):
    mu=z[normal_idx].mean(axis=0, keepdims=True)
    std=z[normal_idx].std(axis=0, keepdims=True)+1e-6
    return np.mean(((z-mu)/std)**2, axis=1)


def pca_residual_score(z, normal_idx, ncomp=32):
    scaler=StandardScaler(with_mean=True, with_std=True)
    zn=scaler.fit_transform(z[normal_idx]); za=scaler.transform(z)
    nc=int(min(ncomp, zn.shape[0]-1, zn.shape[1]))
    if nc <= 0:
        return mahalanobis_diag_score(z, normal_idx)
    pca=PCA(n_components=nc, svd_solver="randomized", random_state=0)
    pca.fit(zn)
    rec=pca.inverse_transform(pca.transform(za))
    return np.mean((za-rec)**2, axis=1)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project-root', default='/data/linziyao/DualRefGAD')
    ap.add_argument('--dataset', default='elliptic')
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train_rate', type=float, default=0.05)
    ap.add_argument('--descriptor_mode', choices=['hop_attr','rwse','hop_attr_rwse'], default='hop_attr_rwse')
    ap.add_argument('--pn_estimator', choices=['diag_gaussian','pca_residual','pca'], default='pca_residual')
    ap.add_argument('--gn_mode', default='label_gate_density')
    ap.add_argument('--ln_mode', default='descriptor_similarity')
    ap.add_argument('--ga_mode', default='normal_soft_or')
    ap.add_argument('--la_mode', default='descriptor_similarity')
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
    ap.add_argument('--GT_num_layers', type=int, default=3)
    ap.add_argument('--encode_batch_size', type=int, default=512)
    ap.add_argument('--sample_rate', type=float, default=0.15)
    ap.add_argument('--mean', type=float, default=0.02)
    ap.add_argument('--var', type=float, default=0.01)
    ap.add_argument('--outlier_beta', type=float, default=0.3)
    ap.add_argument('--ring_R_max', type=float, default=1.0)
    ap.add_argument('--ring_R_min', type=float, default=0.3)
    ap.add_argument('--lambda_rec_tok', type=float, default=1.0)
    ap.add_argument('--lambda_rec_emb', type=float, default=0.1)
    ap.add_argument('--out', default='')
    args=ap.parse_args()

    root=Path(args.project_root)
    sys.path.insert(0, str(root))
    script_dir_candidates=[
        root/'nexus/investigations/2026-05-05-elliptic-training-degradation/experiments/scripts',
        root/'scripts',
    ]
    for script_dir in script_dir_candidates:
        if script_dir.exists():
            sys.path.insert(0, str(script_dir))
    os.chdir(str(root))

    from utils import load_mat
    from VecGAD import VecGAD
    from run_training_degradation_diagnosis import (
        to_dense_features, build_descriptor, NormalModel, select_refs, apply_ablation,
        reference_purity, build_tokens, encode_tokens_batched
    )

    t0=time.time(); set_seed(args.seed)
    device=torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device>=0 else 'cpu')
    # select_refs expects pca_residual name in the older helper.
    if args.pn_estimator == 'pca':
        args.pn_estimator = 'pca_residual'
    args.ablation_mode='full'

    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat(args.dataset, args.train_rate, 0.0, args=args)
    features_np=to_dense_features(args.dataset, features)
    labels_np=np.asarray(ano_label).reshape(-1).astype(int)
    normal_idx=np.asarray(normal_for_train_idx, dtype=int)
    idx_test=np.asarray(idx_test, dtype=int)
    assert np.sum(labels_np[normal_idx]) == 0, 'Data leakage: normal_for_train_idx contains anomalies'

    z_desc=build_descriptor(args.descriptor_mode, features_np, adj, args.hops, args.rw_steps)
    nm=NormalModel(args.pn_estimator, z_desc, normal_idx, args.pca_components)
    residual=nm.residual()
    normal_refs, anom_refs, score_meta=select_refs(z_desc, residual, normal_idx, nm, features_np, adj, args, labels_np)
    normal_refs, anom_refs=apply_ablation(normal_refs, anom_refs, normal_idx, labels_np, args)
    pur=reference_purity(normal_refs, anom_refs, labels_np)

    token_tensor=build_tokens(features_np, normal_refs, anom_refs)
    model=VecGAD(features_np.shape[1], args.embedding_dim, 'prelu', args).to(device)
    model.eval()
    with torch.no_grad():
        emb=encode_tokens_batched(model, token_tensor, device, args.encode_batch_size).detach().cpu().numpy().astype(np.float32)

    rn=emb[normal_refs].mean(axis=1)
    ra=emb[anom_refs].mean(axis=1)
    u=emb-rn
    d=ra-rn
    u_norm=l2_rows(u); d_norm=l2_rows(d)
    margin=np.sum(u_norm*d_norm, axis=1)  # anomaly-high in previous margin-only audit

    features_dict={
        'raw_h': emb,
        'u_target_minus_rn': u,
        'd_ra_minus_rn': d,
        'concat_u_d': np.concatenate([u,d], axis=1),
        'concat_h_rn_ra': np.concatenate([emb,rn,ra], axis=1),
        'relation_interaction': np.concatenate([u,d,u*d,np.abs(u-d)], axis=1),
    }
    scores={
        'margin_only_cos_u_d': margin,
        'neg_margin': -margin,
        'raw_dot_u_d': np.sum(u*d, axis=1),
        'neg_raw_dot_u_d': -np.sum(u*d, axis=1),
        'u_norm': np.linalg.norm(u, axis=1),
        'neg_u_norm': -np.linalg.norm(u, axis=1),
        'd_norm': np.linalg.norm(d, axis=1),
        'neg_d_norm': -np.linalg.norm(d, axis=1),
        'target_to_rn_l2': np.linalg.norm(emb-rn, axis=1),
        'target_to_ra_l2': np.linalg.norm(emb-ra, axis=1),
        'rn_to_ra_l2': np.linalg.norm(rn-ra, axis=1),
        'ra_closer_than_rn': np.linalg.norm(emb-rn, axis=1)-np.linalg.norm(emb-ra, axis=1),
        'rn_closer_than_ra': np.linalg.norm(emb-ra, axis=1)-np.linalg.norm(emb-rn, axis=1),
        'normal_rejection_ref_score': np.asarray(score_meta.get('rejection', np.zeros(len(labels_np))), dtype=np.float32),
        'ga_ref_score': np.asarray(score_meta.get('ga', np.zeros(len(labels_np))), dtype=np.float32),
        'residual_norm_ref_score': np.asarray(score_meta.get('residual_norm', np.zeros(len(labels_np))), dtype=np.float32),
    }
    for name, feat in features_dict.items():
        scores[f'{name}_center_l2']=center_distance_score(feat, normal_idx)
        scores[f'{name}_diag_mahal']=mahalanobis_diag_score(feat, normal_idx)
        scores[f'{name}_pca_residual']=pca_residual_score(feat, normal_idx, args.pca_components)

    result_scores=[]
    for name, score in scores.items():
        m=safe_metrics(labels_np, score, idx_test)
        train_normal_mean=float(np.mean(score[normal_idx]))
        test_normal_mean=float(np.mean(score[idx_test][labels_np[idx_test]==0])) if np.any(labels_np[idx_test]==0) else None
        test_anom_mean=float(np.mean(score[idx_test][labels_np[idx_test]==1])) if np.any(labels_np[idx_test]==1) else None
        result_scores.append({
            'score': name, 'auc': m['auc'], 'ap': m['ap'],
            'train_normal_mean': train_normal_mean,
            'test_normal_mean': test_normal_mean,
            'test_anom_mean': test_anom_mean,
            'score_std': float(np.std(score)),
        })
    result_scores.sort(key=lambda r: r['auc'], reverse=True)
    report={
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'closed_form_read_only_no_training',
        'dataset': args.dataset,
        'seed': args.seed,
        'train_rate': args.train_rate,
        'n_nodes': int(len(labels_np)),
        'n_train_normal': int(len(normal_idx)),
        'n_test': int(len(idx_test)),
        'reference_purity': pur,
        'config': vars(args),
        'scores': result_scores,
        'time_sec': time.time()-t0,
    }
    out=Path(args.out) if args.out else root/'outputs/normality_diagnostic/elliptic_seed0_closed_form_normality.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'out': str(out), 'top_scores': result_scores[:10], 'purity': pur, 'time_sec': report['time_sec']}, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
