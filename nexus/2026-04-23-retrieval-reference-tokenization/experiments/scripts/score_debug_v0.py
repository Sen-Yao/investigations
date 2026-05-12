#!/usr/bin/env python3
"""
Debug score components on a dataset: center-only / residual-only / hybrid.
"""

import argparse
import json
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
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


def make_representation(features, adj, alpha, pca_dim=128):
    x = row_normalize(features.copy())
    neigh = adj.dot(x)
    z = (1 - alpha) * x + alpha * neigh
    if z.shape[1] > pca_dim:
        z = PCA(n_components=pca_dim, random_state=0).fit_transform(z)
    norms = np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
    return z / norms


def eval_score(score, labels):
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    ks = ks_2samp(score[normal_mask], score[anomaly_mask])
    return {
        'auc_neg_score': float(roc_auc_score(labels, -score)),
        'ap_neg_score': float(average_precision_score(labels, -score)),
        'normal_mean': float(score[normal_mask].mean()),
        'anomaly_mean': float(score[anomaly_mask].mean()),
        'ks_stat': float(ks.statistic),
        'ks_pvalue': float(ks.pvalue),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--train-rate', type=float, default=0.05)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    adj, features, labels = load_mat_dataset(args.data)
    train_normal_idx = build_train_normal_idx(labels, args.train_rate, args.seed)
    z = make_representation(features, adj, args.alpha)
    center = z[train_normal_idx].mean(axis=0)
    center = center / (np.linalg.norm(center) + 1e-12)
    alignment = z @ center
    normal_bank = z[train_normal_idx]
    top1 = (z @ normal_bank.T).max(axis=1)
    residual = 1.0 - top1
    hybrid = 0.5 * alignment - 0.5 * residual

    result = {
        'data': args.data,
        'seed': args.seed,
        'train_rate': args.train_rate,
        'alpha': args.alpha,
        'center_only': eval_score(-alignment, labels),
        'residual_only': eval_score(residual, labels),
        'hybrid': eval_score(hybrid, labels),
    }

    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
