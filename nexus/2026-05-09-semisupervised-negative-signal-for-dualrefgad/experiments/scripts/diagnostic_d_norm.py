#!/usr/bin/env python3
"""Quick diagnostic: Is ||d|| discriminative for anomaly detection?"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, './scripts')

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

# Simplified imports
from utils import load_mat
from VecGAD import VecGAD
import argparse

# Create minimal args
args = argparse.Namespace(
    descriptor_mode='hop_attr', pn_estimator='pca_residual',
    gn_mode='label_gate', ln_mode='descriptor_similarity',
    ga_mode='normal_soft_or', la_mode='descriptor_similarity',
    ablation_mode='full', normal_k=4, anom_k=16, pp_k=6,
    hops=2, rw_steps=8, pca_components=32, embedding_dim=256,
    GT_ffn_dim=256, GT_dropout=0.4, GT_attention_dropout=0.4,
    GT_num_heads=2, GT_num_layers=3, sample_rate=0.15,
    mean=0.02, var=0.01, outlier_beta=0.3,
    ring_R_max=1.0, ring_R_min=0.3,
    lambda_rec_tok=1.0, lambda_rec_emb=0.1,
    encode_batch_size=512
)

device = torch.device('cuda:0')

# Load data
adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat('elliptic', 0.05, 0.0, args=args)

from run_training_degradation_diagnosis import to_dense_features, build_descriptor, NormalModel, select_refs, apply_ablation, build_tokens, encode_tokens_batched

features_np = to_dense_features('elliptic', features)
labels_np = np.asarray(ano_label).reshape(-1).astype(int)
normal_idx = np.asarray(normal_for_train_idx, dtype=int)
idx_test = np.asarray(idx_test, dtype=int)

# Build references
z = build_descriptor('hop_attr', features_np, adj, 2, 8)
nm = NormalModel('pca_residual', z, normal_idx, 32)
residual = nm.residual()
normal_refs, anom_refs, _ = select_refs(z, residual, normal_idx, nm, features_np, adj, args, labels_np)
normal_refs, anom_refs = apply_ablation(normal_refs, anom_refs, normal_idx, labels_np, args)

# Encode
token_tensor = build_tokens(features_np, normal_refs, anom_refs)
model = VecGAD(features_np.shape[1], 256, 'prelu', args).to(device)
model.eval()
with torch.no_grad():
    emb = encode_tokens_batched(model, token_tensor, device, 512).detach()

# Compute ||d|| directly
normal_refs_t = torch.as_tensor(normal_refs, dtype=torch.long, device=device)
anom_refs_t = torch.as_tensor(anom_refs, dtype=torch.long, device=device)

h = emb
rn = emb[normal_refs_t].mean(dim=1)
ra = emb[anom_refs_t].mean(dim=1)
d = ra - rn
d_norm = torch.norm(d, p=2, dim=1).cpu().numpy()

# Compute margin
u = h - rn
margin = torch.sum(F.normalize(u, p=2, dim=1) * F.normalize(d, p=2, dim=1), dim=1).cpu().numpy()

# Diagnostic
normal_mask = labels_np == 0
anom_mask = labels_np == 1

print('=== ||d|| Distribution ===')
print(f'Normal nodes: ||d|| mean={d_norm[normal_mask].mean():.4f}, std={d_norm[normal_mask].std():.4f}')
print(f'Anomaly nodes: ||d|| mean={d_norm[anom_mask].mean():.4f}, std={d_norm[anom_mask].std():.4f}')

print('\n=== Is ||d|| discriminative? ===')
d_auc = roc_auc_score(labels_np[idx_test], d_norm[idx_test])
d_auc_inv = roc_auc_score(labels_np[idx_test], -d_norm[idx_test])
print(f'||d|| as score AUC: {d_auc:.4f}')
print(f'-||d|| as score AUC: {d_auc_inv:.4f}')

print('\n=== Spearman(||d||, margin) ===')
sp = spearmanr(d_norm[idx_test], margin[idx_test]).correlation
print(f'Spearman: {sp:.4f}')

print('\n=== ||d|| by Group ===')
train_d = d_norm[normal_idx]
test_normal_d = d_norm[np.intersect1d(idx_test, np.where(normal_mask)[0])]
test_anom_d = d_norm[np.intersect1d(idx_test, np.where(anom_mask)[0])]
print(f'Train normal: {train_d.mean():.4f} ± {train_d.std():.4f}')
print(f'Test normal: {test_normal_d.mean():.4f} ± {test_normal_d.std():.4f}')
print(f'Test anomaly: {test_anom_d.mean():.4f} ± {test_anom_d.std():.4f}')