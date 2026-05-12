#!/usr/bin/env python3
"""
Family-level retrieval audit for three score families:
1. center family
2. original delta-NDC family
3. ANR-inspired proxy family

No anomaly labels are used in constructing the families.
Labels are only for evaluation.
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


def build_train_normal_idx(labels, train_rate, seed):
    rng = np.random.default_rng(seed)
    normal_idx = np.where(labels == 0)[0]
    rng.shuffle(normal_idx)
    n_train = max(1, int(len(labels) * train_rate))
    n_train = min(n_train, len(normal_idx))
    return np.sort(normal_idx[:n_train])


def make_representation(features, adj, alpha, pca_dim=128):
    x = row_normalize(features.copy())
    neigh = adj.dot(x)
    z = (1 - alpha) * x + alpha * neigh
    if z.shape[1] > pca_dim:
        z = PCA(n_components=pca_dim, random_state=0).fit_transform(z)
    norms = np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
    return z / norms


def cosine(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


def compute_center_family(z, train_normal_idx):
    center = z[train_normal_idx].mean(axis=0)
    center = center / (np.linalg.norm(center) + 1e-12)
    return z @ center


def compute_delta_ndc_family(features, adj, k=6):
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
        neigh = np.where(adj[i] > 0)[0]
        if len(neigh) == 0:
            ndc[i] = 0.0
            continue
        neigh_delta_mean = np.mean(delta_vectors[neigh], axis=0)
        a = delta_vectors[i].reshape(-1)
        b = neigh_delta_mean.reshape(-1)
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            ndc[i] = 0.0
        else:
            ndc[i] = np.corrcoef(a, b)[0, 1]
    return ndc


def compute_anr_proxy_family(z, adj):
    n = z.shape[0]
    proxy = np.zeros(n, dtype=np.float32)
    for i in range(n):
        neigh = np.where(adj[i] > 0)[0]
        patch = np.unique(np.concatenate(([i], neigh[:64])))
        patch_z = z[patch]
        patch_center = patch_z.mean(axis=0)
        target_patch_mismatch = 1.0 - cosine(z[i], patch_center)
        if len(patch) > 1:
            sample = patch_z[:min(len(patch_z), 16)]
            sims = []
            for a in range(len(sample)):
                for b in range(a + 1, len(sample)):
                    sims.append(cosine(sample[a], sample[b]))
            patch_consistency = float(np.mean(sims)) if sims else 1.0
        else:
            patch_consistency = 1.0
        proxy[i] = 0.5 * (target_patch_mismatch + (1.0 - patch_consistency))
    return proxy


def eval_family(values, labels, anomaly_high, topk):
    score = values if anomaly_high else -values
    ks = ks_2samp(values[labels == 0], values[labels == 1])
    order = np.argsort(values)
    if anomaly_high:
        anomaly_idx = order[::-1][:topk]
        normal_idx = order[:topk]
    else:
        anomaly_idx = order[:topk]
        normal_idx = order[::-1][:topk]
    return {
        'auc': float(roc_auc_score(labels, score)),
        'ap': float(average_precision_score(labels, score)),
        'normal_mean': float(values[labels == 0].mean()),
        'anomaly_mean': float(values[labels == 1].mean()),
        'difference': float(values[labels == 1].mean() - values[labels == 0].mean()),
        'ks_stat': float(ks.statistic),
        'ks_pvalue': float(ks.pvalue),
        'anomaly_high': anomaly_high,
        'normal_topk_normal_ratio': float(np.mean(labels[normal_idx] == 0)),
        'anomaly_topk_anomaly_ratio': float(np.mean(labels[anomaly_idx] == 1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--train-rate', type=float, default=0.05)
    ap.add_argument('--alpha', type=float, default=0.1)
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--topk', type=int, default=32)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    adj, features, labels = load_mat_dataset(args.data)
    train_normal_idx = build_train_normal_idx(labels, args.train_rate, args.seed)
    z = make_representation(features, adj, args.alpha)

    center_family = compute_center_family(z, train_normal_idx)
    delta_ndc_family = compute_delta_ndc_family(features, adj, k=args.k)
    anr_proxy_family = compute_anr_proxy_family(z, adj)

    result = {
        'data': args.data,
        'seed': args.seed,
        'train_rate': args.train_rate,
        'alpha': args.alpha,
        'k': args.k,
        'topk': args.topk,
        'families': {
            'center_family': eval_family(center_family, labels, anomaly_high=False, topk=args.topk),
            'delta_ndc_family': eval_family(delta_ndc_family, labels, anomaly_high=True, topk=args.topk),
            'anr_proxy_family': eval_family(anr_proxy_family, labels, anomaly_high=True, topk=args.topk),
        }
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
