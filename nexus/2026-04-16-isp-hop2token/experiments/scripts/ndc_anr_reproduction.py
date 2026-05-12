#!/usr/bin/env python3
"""
Reproduce NDC / ANR experiments across datasets.
NDC uses no anomaly labels in construction.
ANR is computed only for phenomenon verification and explicitly uses labels.
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
    adj = network if sp.issparse(network) else sp.csr_matrix(network)
    labels = np.asarray(label).reshape(-1)
    labels = labels - labels.min()
    return adj.tocsr(), feat.astype(np.float32), labels.astype(int)


def row_normalize(x):
    rs = x.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return x / rs


def make_hop_features(features, adj, k=6, alpha=0.1):
    x = row_normalize(features.copy())
    hops = []
    cur = x
    for _ in range(k + 1):
        hops.append(cur.copy())
        cur = (1 - alpha) * x + alpha * adj.dot(cur)
    return np.stack(hops, axis=1)  # [N, K+1, D]


def compute_ndc(hop_feats, adj):
    n = hop_feats.shape[0]
    deltas = hop_feats[:, 1:, :] - hop_feats[:, :-1, :]
    ndc = np.zeros(n, dtype=np.float32)
    for i in range(n):
        neigh = adj.indices[adj.indptr[i]:adj.indptr[i+1]]
        if len(neigh) == 0:
            ndc[i] = 0.0
            continue
        neigh_delta = deltas[neigh].mean(axis=0)
        a = deltas[i].reshape(-1)
        b = neigh_delta.reshape(-1)
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            ndc[i] = 0.0
        else:
            ndc[i] = np.corrcoef(a, b)[0, 1]
    return ndc


def compute_anr(adj, labels):
    n = adj.shape[0]
    anr = np.zeros(n, dtype=np.float32)
    for i in range(n):
        neigh = adj.indices[adj.indptr[i]:adj.indptr[i+1]]
        if len(neigh) == 0:
            anr[i] = 0.0
        else:
            anr[i] = float(np.mean(labels[neigh] == 1))
    return anr


def summarize_metric(values, labels, anomaly_high=True):
    normal = values[labels == 0]
    anomaly = values[labels == 1]
    ks = ks_2samp(normal, anomaly)
    diff = float(anomaly.mean() - normal.mean())
    support = diff > 0 if anomaly_high else diff < 0
    return {
        'normal_mean': float(normal.mean()),
        'normal_var': float(normal.var()),
        'anomaly_mean': float(anomaly.mean()),
        'anomaly_var': float(anomaly.var()),
        'difference': diff,
        'support_expected_direction': bool(support),
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
    ap.add_argument('--datasets', required=True, help='comma-separated list')
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--alpha', type=float, default=0.1)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    results = {}
    for ds in [d.strip() for d in args.datasets.split(',') if d.strip()]:
        path = resolve_dataset_path(ds)
        if path is None:
            results[ds] = {'error': 'dataset not found'}
            continue
        adj, features, labels = load_mat_dataset(path)
        hop_feats = make_hop_features(features, adj, k=args.k, alpha=args.alpha)
        ndc = compute_ndc(hop_feats, adj)
        anr = compute_anr(adj, labels)
        results[ds] = {
            'path': path,
            'n_nodes': int(len(labels)),
            'anomaly_rate': float(labels.mean()),
            'ndc': summarize_metric(ndc, labels, anomaly_high=True),
            'anr': summarize_metric(anr, labels, anomaly_high=True),
            'anr_leakage_warning': True,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
