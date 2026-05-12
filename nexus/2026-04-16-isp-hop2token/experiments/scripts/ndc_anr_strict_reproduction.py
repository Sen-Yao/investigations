#!/usr/bin/env python3
"""
Strict reproduction of original NDC / ANR style following 2026-04-16 report.
Uses symmetric normalized adjacency and pure hop propagation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.stats import ks_2samp


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


def normalize_adj_dense(adj):
    degree = adj.sum(axis=1)
    d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    return d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]


def compute_original_ndc(features, adj, k=6):
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


def compute_original_anr(adj, labels):
    N = len(labels)
    anr = np.zeros(N, dtype=np.float32)
    for i in range(N):
        neighbors = np.where(adj[i] > 0)[0]
        if len(neighbors) == 0:
            anr[i] = 0.0
            continue
        anr[i] = float(np.mean(labels[neighbors] == 1))
    return anr


def summarize(values, labels, anomaly_high=True):
    normal = values[labels == 0]
    anomaly = values[labels == 1]
    ks = ks_2samp(normal, anomaly)
    diff = float(anomaly.mean() - normal.mean())
    return {
        'normal_mean': float(normal.mean()),
        'normal_var': float(normal.var()),
        'anomaly_mean': float(anomaly.mean()),
        'anomaly_var': float(anomaly.var()),
        'difference': diff,
        'support_expected_direction': bool(diff > 0 if anomaly_high else diff < 0),
        'ks_stat': float(ks.statistic),
        'ks_pvalue': float(ks.pvalue),
    }


def resolve_dataset_path(name):
    candidates = [
        f"/root/gpufree-data/linziyao/VoxG/dataset/{name}.mat",
        f"/root/gpufree-data/linziyao/VoxG/dataset/{name.lower()}.mat",
        f"/root/gpufree-data/linziyao/VecGAD/dataset/{name}.mat",
        f"/root/gpufree-data/linziyao/VecGAD/dataset/{name.lower()}.mat",
        f"/root/gpufree-data/linziyao/VecGAD/GGAD/dataset/{name}.mat",
        f"/root/gpufree-data/linziyao/VecGAD/GGAD/dataset/{name.lower()}.mat",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', required=True)
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    results = {}
    for ds in [d.strip() for d in args.datasets.split(',') if d.strip()]:
        path = resolve_dataset_path(ds)
        if path is None:
            results[ds] = {'error': 'dataset not found'}
            continue
        adj, features, labels = load_mat_dataset(path)
        ndc = compute_original_ndc(features, adj, k=args.k)
        anr = compute_original_anr(adj, labels)
        results[ds] = {
            'path': path,
            'n_nodes': int(len(labels)),
            'anomaly_rate': float(labels.mean()),
            'ndc': summarize(ndc, labels, anomaly_high=True),
            'anr': summarize(anr, labels, anomaly_high=True),
            'anr_leakage_warning': True,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
