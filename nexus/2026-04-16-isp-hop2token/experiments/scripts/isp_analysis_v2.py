#!/usr/bin/env python3
import sys
sys.path.insert(0, '/root/gpufree-data/linziyao/VoxG')

import torch
import numpy as np
from scipy import stats
import scipy.io as sio
import scipy.sparse as sp

dataset = sys.argv[1] if len(sys.argv) > 1 else 'photo'

# Load data
data = sio.loadmat(f'./dataset/{dataset}.mat')
features = data.get('Attributes', data.get('X'))
if sp.issparse(features):
    features = features.todense()
adj = data.get('Network', data.get('A'))
if sp.issparse(adj):
    adj = adj.todense()
labels = data.get('Label', data.get('Y')).flatten()
labels = labels - labels.min()
features = np.array(features)
adj = np.array(adj)

print(f'{dataset}: N={len(labels)}, anomaly={sum(labels)}, ratio={sum(labels)/len(labels)*100:.2f}%')

# Normalize adj
degree = adj.sum(axis=1)
d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree.flatten()), 0)
d_inv_sqrt = np.diag(d_inv_sqrt)
adj_norm = d_inv_sqrt @ adj @ d_inv_sqrt

# Compute hop features
features_t = torch.FloatTensor(features)
adj_norm_t = torch.FloatTensor(adj_norm)
K = 6
N, D = features.shape
hop_features = torch.zeros((N, K+1, D))
hop_features[:, 0] = features_t
agg = features_t.clone()
for k in range(1, K+1):
    agg = adj_norm_t @ agg
    hop_features[:, k] = agg

# Compute ISP
normal_mask = labels == 0
anomaly_mask = labels == 1

isp_total = torch.norm(hop_features[:, K] - hop_features[:, 0], p=2, dim=1).numpy()

isp_per_hop = torch.zeros(N, K)
for k in range(K):
    isp_per_hop[:, k] = torch.norm(hop_features[:, k+1] - hop_features[:, k], p=2, dim=1)
isp_per_hop = isp_per_hop.numpy()

isp_rate = np.zeros(N)
for i in range(N):
    changes = isp_per_hop[i]
    isp_rate[i] = np.std(changes) / (np.mean(changes) + 1e-8)

print()
print('='*60)
print(f'{dataset} ISP 分析')
print('='*60)

# ISP_total
normal_vals = isp_total[normal_mask]
anomaly_vals = isp_total[anomaly_mask]
ks_stat, ks_pval = stats.ks_2samp(normal_vals, anomaly_vals)
auc_like = 0.5 + abs(np.mean(anomaly_vals) - np.mean(normal_vals)) / (2 * (np.std(normal_vals) + np.std(anomaly_vals) + 1e-8))
print(f'ISP_total: Normal={np.mean(normal_vals):.2f}, Anomaly={np.mean(anomaly_vals):.2f}, AUC-like={auc_like:.4f}, KS-p={ks_pval:.4e}')

# ISP_rate
normal_vals = isp_rate[normal_mask]
anomaly_vals = isp_rate[anomaly_mask]
ks_stat, ks_pval = stats.ks_2samp(normal_vals, anomaly_vals)
auc_like = 0.5 + abs(np.mean(anomaly_vals) - np.mean(normal_vals)) / (2 * (np.std(normal_vals) + np.std(anomaly_vals) + 1e-8))
print(f'ISP_rate:  Normal={np.mean(normal_vals):.2f}, Anomaly={np.mean(anomaly_vals):.2f}, AUC-like={auc_like:.4f}, KS-p={ks_pval:.4e}')

# Per hop
print()
print('各 Hop ISP:')
for k in range(K):
    normal_vals = isp_per_hop[normal_mask, k]
    anomaly_vals = isp_per_hop[anomaly_mask, k]
    ks_stat, ks_pval = stats.ks_2samp(normal_vals, anomaly_vals)
    auc_like = 0.5 + abs(np.mean(anomaly_vals) - np.mean(normal_vals)) / (2 * (np.std(normal_vals) + np.std(anomaly_vals) + 1e-8))
    print(f'  Hop {k}: Normal={np.mean(normal_vals):.2f}, Anomaly={np.mean(anomaly_vals):.2f}, AUC={auc_like:.4f}, KS-p={ks_pval:.4e}')

# Best hop
best_auc = 0
best_hop = 0
for k in range(K):
    normal_vals = isp_per_hop[normal_mask, k]
    anomaly_vals = isp_per_hop[anomaly_mask, k]
    auc_like = 0.5 + abs(np.mean(anomaly_vals) - np.mean(normal_vals)) / (2 * (np.std(normal_vals) + np.std(anomaly_vals) + 1e-8))
    if auc_like > best_auc:
        best_auc = auc_like
        best_hop = k

print()
print(f'最佳 Hop: {best_hop}, AUC-like={best_auc:.4f}')
if best_auc > 0.7:
    print('✅ ISP 有较好区分能力')
elif best_auc > 0.55:
    print('⚠️ ISP 有一定区分能力')
else:
    print('❌ ISP 区分能力弱')
