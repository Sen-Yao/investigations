#!/usr/bin/env python3
"""
Lightweight probe for retrieval-based reference tokenization.
Builds a coupled hybrid score from normal-center alignment and residual,
then evaluates score separation and top-k reference purity.
"""

import argparse
import json
from pathlib import Path

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


def compute_hybrid_score(z, train_normal_idx):
    center = z[train_normal_idx].mean(axis=0)
    center = center / (np.linalg.norm(center) + 1e-12)
    alignment = z @ center

    normal_bank = z[train_normal_idx]
    sims = z @ normal_bank.T
    top1 = sims.max(axis=1)
    residual = 1.0 - top1

    score = 0.5 * alignment - 0.5 * residual
    return score, alignment, residual


def topk_purity(score, labels, k):
    order_hi = np.argsort(score)[::-1][:k]
    order_lo = np.argsort(score)[:k]
    return {
        'normal_topk_normal_ratio': float(np.mean(labels[order_hi] == 0)),
        'anomaly_topk_anomaly_ratio': float(np.mean(labels[order_lo] == 1)),
        'normal_topk_indices': order_hi.tolist(),
        'anomaly_topk_indices': order_lo.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--train-rate', type=float, default=0.05)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--topk', type=int, default=32)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    adj, features, labels = load_mat_dataset(args.data)
    train_normal_idx = build_train_normal_idx(labels, args.train_rate, args.seed)
    z = make_representation(features, adj, args.alpha)
    score, alignment, residual = compute_hybrid_score(z, train_normal_idx)

    normal_mask = labels == 0
    anomaly_mask = labels == 1
    ks = ks_2samp(score[normal_mask], score[anomaly_mask])
    purity = topk_purity(score, labels, min(args.topk, len(labels)))

    result = {
        'data': args.data,
        'train_rate': args.train_rate,
        'alpha': args.alpha,
        'seed': args.seed,
        'n_train_normal': int(len(train_normal_idx)),
        'score_probe': {
            'auc_using_negative_score_as_anomaly': float(roc_auc_score(labels, -score)),
            'ap_using_negative_score_as_anomaly': float(average_precision_score(labels, -score)),
            'ks_stat': float(ks.statistic),
            'ks_pvalue': float(ks.pvalue),
            'normal_score_mean': float(score[normal_mask].mean()),
            'anomaly_score_mean': float(score[anomaly_mask].mean()),
            'alignment_normal_mean': float(alignment[normal_mask].mean()),
            'alignment_anomaly_mean': float(alignment[anomaly_mask].mean()),
            'residual_normal_mean': float(residual[normal_mask].mean()),
            'residual_anomaly_mean': float(residual[anomaly_mask].mean()),
        },
        'topk_probe': purity,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
