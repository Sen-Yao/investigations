#!/usr/bin/env python3
"""
Try original delta-based NDC inside unified score framework.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score


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


def build_train_normal_idx(labels, train_rate, seed):
    rng = np.random.default_rng(seed)
    normal_idx = np.where(labels == 0)[0]
    rng.shuffle(normal_idx)
    n_train = max(1, int(len(labels) * train_rate))
    n_train = min(n_train, len(normal_idx))
    return np.sort(normal_idx[:n_train])


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
    return np.stack(hops, axis=1)


def make_representation(features, adj, alpha, pca_dim=128):
    x = row_normalize(features.copy())
    neigh = adj.dot(x)
    z = (1 - alpha) * x + alpha * neigh
    if z.shape[1] > pca_dim:
        z = PCA(n_components=pca_dim, random_state=0).fit_transform(z)
    norms = np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
    return z / norms


def compute_delta_ndc(hop_feats, adj):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--train-rate', type=float, default=0.05)
    ap.add_argument('--alpha', type=float, default=0.1)
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    adj, features, labels = load_mat_dataset(args.data)
    train_normal_idx = build_train_normal_idx(labels, args.train_rate, args.seed)
    z = make_representation(features, adj, args.alpha)
    center = z[train_normal_idx].mean(axis=0)
    center = center / (np.linalg.norm(center) + 1e-12)
    center_alignment = z @ center
    hop_feats = make_hop_features(features, adj, k=args.k, alpha=args.alpha)
    delta_ndc = compute_delta_ndc(hop_feats, adj)

    center_plus_delta_ndc = 0.5 * center_alignment - 0.5 * delta_ndc
    delta_ndc_only = -delta_ndc  # lower ndc => more anomaly-like if needed can compare by auc

    result = {
        'data': args.data,
        'seed': args.seed,
        'train_rate': args.train_rate,
        'alpha': args.alpha,
        'k': args.k,
        'center_only': eval_score(center_alignment, labels, anomaly_high=False),
        'delta_ndc_only_assume_anomaly_high': eval_score(delta_ndc, labels, anomaly_high=True),
        'delta_ndc_only_assume_anomaly_low': eval_score(delta_ndc, labels, anomaly_high=False),
        'center_plus_delta_ndc': eval_score(center_plus_delta_ndc, labels, anomaly_high=False),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
