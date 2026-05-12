#!/usr/bin/env python3
"""
Two-stage anomaly reference purification.
Stage 1: NDC-cluster coarse candidate selection.
Stage 2: local ANR-like proxy rerank within candidate pool.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


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
        a = delta_vectors[i].reshape(-1)
        b = neighbor_delta_mean.reshape(-1)
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            ndc[i] = 0.0
        else:
            ndc[i] = np.corrcoef(a, b)[0, 1]
    return ndc


def build_ndc_cluster_score(ndc, adj, self_weight=0.25):
    q = (ndc - ndc.mean()) / (ndc.std() + 1e-8)
    q = 1.0 / (1.0 + np.exp(-q))
    n = len(q)
    cluster = np.zeros(n, dtype=np.float32)
    for i in range(n):
        neighbors = np.where(adj[i] > 0)[0]
        if len(neighbors) == 0:
            cluster[i] = q[i]
        else:
            cluster[i] = self_weight * q[i] + (1 - self_weight) * float(np.mean(q[neighbors]))
    return q, cluster


def cosine(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


def build_local_concentration_proxy(features, adj, seed_score):
    features = row_normalize(features.copy())
    n = len(seed_score)
    proxy = np.zeros(n, dtype=np.float32)
    for i in range(n):
        neighbors = np.where(adj[i] > 0)[0]
        if len(neighbors) == 0:
            proxy[i] = seed_score[i]
            continue
        # local anomaly-likeness concentration + feature mismatch to local center
        neigh_score = float(np.mean(seed_score[neighbors]))
        neigh_center = features[neighbors].mean(axis=0)
        mismatch = 1.0 - cosine(features[i], neigh_center)
        proxy[i] = 0.7 * neigh_score + 0.3 * mismatch
    return proxy


def purity(indices, labels):
    return float(np.mean(labels[indices] == 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--topm', type=int, default=256)
    ap.add_argument('--topk', type=int, default=32)
    ap.add_argument('--self-weight', type=float, default=0.25)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    adj, features, labels = load_mat_dataset(args.data)
    ndc = compute_original_ndc(features, adj, k=args.k)
    q, ndc_cluster = build_ndc_cluster_score(ndc, adj, self_weight=args.self_weight)
    local_proxy = build_local_concentration_proxy(features, adj, ndc_cluster)

    stage1_idx = np.argsort(ndc_cluster)[::-1][:args.topm]
    rerank_order = stage1_idx[np.argsort(local_proxy[stage1_idx])[::-1]]
    stage2_idx = rerank_order[:args.topk]
    single_stage_idx = np.argsort(ndc_cluster)[::-1][:args.topk]

    result = {
        'data': args.data,
        'k': args.k,
        'topm': args.topm,
        'topk': args.topk,
        'self_weight': args.self_weight,
        'single_stage': {
            'topk_anomaly_ratio': purity(single_stage_idx, labels),
            'indices': single_stage_idx.tolist(),
        },
        'two_stage': {
            'topk_anomaly_ratio': purity(stage2_idx, labels),
            'indices': stage2_idx.tolist(),
        },
        'delta': float(purity(stage2_idx, labels) - purity(single_stage_idx, labels)),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
