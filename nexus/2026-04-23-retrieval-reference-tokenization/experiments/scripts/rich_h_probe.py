#!/usr/bin/env python3
"""
Probe richer anomaly-affinity representation h_v = [q, c, ndc, a].
"""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score


def load_mat_dataset(path):
    data = sio.loadmat(path)
    label = data['Label'] if 'Label' in data else data['gnd']
    attr = data['Attributes'] if 'Attributes' in data else data['X']
    network = data['Network'] if 'Network' in data else data['A']
    feat = attr.toarray() if sp.issparse(attr) else np.asarray(attr)
    adj = network.toarray() if sp.issparse(network) else np.asarray(network)
    labels = np.asarray(label).reshape(-1)
    labels = labels - labels.min()
    return adj.astype(np.float32), feat.astype(np.float32), labels.astype(int)


def row_normalize(x):
    rs = x.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return x / rs


def normalize_adj_dense(adj):
    degree = adj.sum(axis=1)
    d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    return d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]


def compute_original_ndc(features, adj, k=6):
    features = row_normalize(features.copy())
    adj_norm = normalize_adj_dense(adj)
    N, D = features.shape
    hop_features = np.zeros((N, k + 1, D), dtype=np.float32)
    hop_features[:, 0] = features
    agg = features.copy()
    for hop in range(1, k + 1):
        agg = adj_norm @ agg
        hop_features[:, hop] = agg
    delta_vectors = hop_features[:, 1:] - hop_features[:, :-1]
    ndc = np.zeros(N, dtype=np.float32)
    for i in range(N):
        neighbors = np.where(adj[i] > 0)[0]
        if len(neighbors) == 0:
            continue
        neighbor_delta_mean = np.mean(delta_vectors[neighbors], axis=0)
        a = delta_vectors[i].reshape(-1)
        b = neighbor_delta_mean.reshape(-1)
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            ndc[i] = 0.0
        else:
            ndc[i] = np.corrcoef(a, b)[0, 1]
    return ndc


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def cosine_matrix(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


def build_h(features, adj, k=6):
    ndc = compute_original_ndc(features, adj, k=k)
    q = sigmoid((ndc - ndc.mean()) / (ndc.std() + 1e-8))
    n = len(q)
    c = np.zeros(n, dtype=np.float32)
    a = np.zeros(n, dtype=np.float32)
    feats = row_normalize(features.copy())
    for i in range(n):
        neigh = np.where(adj[i] > 0)[0]
        if len(neigh) == 0:
            c[i] = q[i]
            a[i] = 0.0
            continue
        c[i] = float(np.mean(q[neigh]))
        neigh_center = feats[neigh].mean(axis=0)
        mismatch = 1.0 - float(np.dot(feats[i], neigh_center) / ((np.linalg.norm(feats[i]) + 1e-12) * (np.linalg.norm(neigh_center) + 1e-12)))
        a[i] = 0.7 * c[i] + 0.3 * mismatch
    h_small = np.stack([q, c], axis=1)
    h_rich = np.stack([q, c, ndc, a], axis=1)
    return ndc, q, c, a, h_small, h_rich


def eval_score(values, labels, topk):
    ks = ks_2samp(values[labels == 0], values[labels == 1])
    idx = np.argsort(values)[::-1][:topk]
    return {
        'auc': float(roc_auc_score(labels, values)),
        'ap': float(average_precision_score(labels, values)),
        'normal_mean': float(values[labels == 0].mean()),
        'anomaly_mean': float(values[labels == 1].mean()),
        'difference': float(values[labels == 1].mean() - values[labels == 0].mean()),
        'ks_stat': float(ks.statistic),
        'ks_pvalue': float(ks.pvalue),
        'topk_anomaly_ratio': float(np.mean(labels[idx] == 1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--topk', type=int, default=32)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    adj, features, labels = load_mat_dataset(args.data)
    ndc, q, c, a, h_small, h_rich = build_h(features, adj, k=args.k)
    local_small = cosine_matrix(h_small, h_small)
    local_rich = cosine_matrix(h_rich, h_rich)
    g_anom = q * c
    score_small = g_anom[:, None] * local_small
    score_rich = g_anom[:, None] * local_rich
    # For each node, take best matching anomaly-like candidate excluding self, then summarize by candidate-side score max.
    np.fill_diagonal(score_small, -1e9)
    np.fill_diagonal(score_rich, -1e9)
    best_small = score_small.max(axis=1)
    best_rich = score_rich.max(axis=1)

    result = {
        'data': args.data,
        'k': args.k,
        'topk': args.topk,
        'small_h_affinity': eval_score(best_small, labels, args.topk),
        'rich_h_affinity': eval_score(best_rich, labels, args.topk),
        'base_anom_prior': eval_score(g_anom, labels, args.topk),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
