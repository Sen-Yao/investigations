#!/usr/bin/env python3
"""
NDC-cluster score probe.
Use original delta-NDC as weak anomaly prior, then aggregate it over neighborhood.
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
            ndc[i] = 0.0
            continue
        neighbor_delta_mean = np.mean(delta_vectors[neighbors], axis=0)
        node_delta_flat = delta_vectors[i].reshape(-1)
        neighbor_delta_flat = neighbor_delta_mean.reshape(-1)
        if np.std(node_delta_flat) < 1e-8 or np.std(neighbor_delta_flat) < 1e-8:
            ndc[i] = 0.0
        else:
            ndc[i] = np.corrcoef(node_delta_flat, neighbor_delta_flat)[0, 1]
    return ndc


def build_ndc_cluster_score(ndc, adj):
    # high ndc => more anomaly-like based on reproduced phenomenon
    q = (ndc - ndc.mean()) / (ndc.std() + 1e-8)
    q = 1.0 / (1.0 + np.exp(-q))
    n = len(q)
    cluster = np.zeros(n, dtype=np.float32)
    for i in range(n):
        neighbors = np.where(adj[i] > 0)[0]
        if len(neighbors) == 0:
            cluster[i] = q[i]
        else:
            cluster[i] = 0.5 * q[i] + 0.5 * float(np.mean(q[neighbors]))
    return q, cluster


def eval_score(values, labels):
    ks = ks_2samp(values[labels == 0], values[labels == 1])
    return {
        'auc': float(roc_auc_score(labels, values)),
        'ap': float(average_precision_score(labels, values)),
        'normal_mean': float(values[labels == 0].mean()),
        'anomaly_mean': float(values[labels == 1].mean()),
        'difference': float(values[labels == 1].mean() - values[labels == 0].mean()),
        'ks_stat': float(ks.statistic),
        'ks_pvalue': float(ks.pvalue),
    }


def topk_purity(values, labels, k):
    idx = np.argsort(values)[::-1][:k]
    return {
        'topk_anomaly_ratio': float(np.mean(labels[idx] == 1)),
        'topk_indices': idx.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--topk', type=int, default=32)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    adj, features, labels = load_mat_dataset(args.data)
    ndc = compute_original_ndc(features, adj, k=args.k)
    q, cluster = build_ndc_cluster_score(ndc, adj)

    result = {
        'data': args.data,
        'k': args.k,
        'base_ndc': eval_score(ndc, labels),
        'ndc_prior_q': eval_score(q, labels),
        'ndc_cluster_score': eval_score(cluster, labels),
        'topk_base_ndc': topk_purity(ndc, labels, min(args.topk, len(labels))),
        'topk_ndc_cluster': topk_purity(cluster, labels, min(args.topk, len(labels))),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
