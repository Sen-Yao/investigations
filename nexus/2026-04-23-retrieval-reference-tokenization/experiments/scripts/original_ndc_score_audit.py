#!/usr/bin/env python3
"""
Audit the original delta-based NDC score itself.
No fusion, no GT, just score quality and top-k reference quality.
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


def eval_score(values, labels, anomaly_high):
    score = values if anomaly_high else -values
    ks = ks_2samp(values[labels == 0], values[labels == 1])
    return {
        'auc': float(roc_auc_score(labels, score)),
        'ap': float(average_precision_score(labels, score)),
        'normal_mean': float(values[labels == 0].mean()),
        'anomaly_mean': float(values[labels == 1].mean()),
        'difference': float(values[labels == 1].mean() - values[labels == 0].mean()),
        'ks_stat': float(ks.statistic),
        'ks_pvalue': float(ks.pvalue),
        'anomaly_high': anomaly_high,
    }


def topk_purity(values, labels, k, anomaly_high):
    order = np.argsort(values)
    if anomaly_high:
        anomaly_idx = order[::-1][:k]
        normal_idx = order[:k]
    else:
        anomaly_idx = order[:k]
        normal_idx = order[::-1][:k]
    return {
        'normal_topk_normal_ratio': float(np.mean(labels[normal_idx] == 0)),
        'anomaly_topk_anomaly_ratio': float(np.mean(labels[anomaly_idx] == 1)),
        'normal_topk_indices': normal_idx.tolist(),
        'anomaly_topk_indices': anomaly_idx.tolist(),
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

    high = eval_score(ndc, labels, anomaly_high=True)
    low = eval_score(ndc, labels, anomaly_high=False)
    purity_high = topk_purity(ndc, labels, min(args.topk, len(labels)), anomaly_high=True)
    purity_low = topk_purity(ndc, labels, min(args.topk, len(labels)), anomaly_high=False)

    result = {
        'data': args.data,
        'k': args.k,
        'score_eval_assume_anomaly_high': high,
        'score_eval_assume_anomaly_low': low,
        'topk_assume_anomaly_high': purity_high,
        'topk_assume_anomaly_low': purity_low,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
