#!/usr/bin/env python3
"""
Audit NDC-family and ANR-family proxies on Elliptic.
Labels are used only for evaluation, never for constructing proxies.
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


def eval_metric(name, values, labels, anomaly_high=True):
    score = values if anomaly_high else -values
    ks = ks_2samp(values[labels == 0], values[labels == 1])
    return {
        'name': name,
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
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    adj, features, labels = load_mat_dataset(args.data)
    train_normal_idx = build_train_normal_idx(labels, args.train_rate, args.seed)
    z = make_representation(features, adj, args.alpha)
    center = z[train_normal_idx].mean(axis=0)
    center = center / (np.linalg.norm(center) + 1e-12)

    n = z.shape[0]
    metrics = {
        'center_alignment': np.zeros(n, dtype=np.float32),
        'node_to_neigh_mismatch': np.zeros(n, dtype=np.float32),
        'node_to_patch_mismatch': np.zeros(n, dtype=np.float32),
        'patch_internal_inconsistency': np.zeros(n, dtype=np.float32),
        'patch_density': np.zeros(n, dtype=np.float32),
        'patch_boundary_contrast': np.zeros(n, dtype=np.float32),
    }

    for i in range(n):
        metrics['center_alignment'][i] = cosine(z[i], center)
        neigh = adj.indices[adj.indptr[i]:adj.indptr[i+1]]
        if len(neigh) > 0:
            neigh_center = z[neigh].mean(axis=0)
            metrics['node_to_neigh_mismatch'][i] = 1.0 - cosine(z[i], neigh_center)
        else:
            metrics['node_to_neigh_mismatch'][i] = 1.0

        patch = np.unique(np.concatenate(([i], neigh[:64])))
        patch_z = z[patch]
        patch_center = patch_z.mean(axis=0)
        metrics['node_to_patch_mismatch'][i] = 1.0 - cosine(z[i], patch_center)

        if len(patch) > 1:
            sub_adj = adj[patch][:, patch]
            edges = sub_adj.nnz / 2.0
            metrics['patch_density'][i] = float(edges / (len(patch) * (len(patch) - 1) / 2.0))
            sample = patch_z[:min(len(patch_z), 16)]
            sims = []
            for a in range(len(sample)):
                for b in range(a + 1, len(sample)):
                    sims.append(cosine(sample[a], sample[b]))
            consistency = float(np.mean(sims)) if sims else 1.0
        else:
            metrics['patch_density'][i] = 0.0
            consistency = 1.0
        metrics['patch_internal_inconsistency'][i] = 1.0 - consistency

        boundary_mask = np.asarray(adj[patch].sum(axis=0)).ravel() > 0
        boundary = np.where(boundary_mask)[0]
        boundary = np.setdiff1d(boundary, patch, assume_unique=False)
        if len(boundary) > 0:
            boundary_center = z[boundary[:64]].mean(axis=0)
            metrics['patch_boundary_contrast'][i] = float(np.linalg.norm(patch_center - boundary_center))
        else:
            metrics['patch_boundary_contrast'][i] = 0.0

    ndc_family = [
        eval_metric('center_alignment', metrics['center_alignment'], labels, anomaly_high=False),
        eval_metric('node_to_neigh_mismatch', metrics['node_to_neigh_mismatch'], labels, anomaly_high=True),
    ]
    anr_family = [
        eval_metric('node_to_patch_mismatch', metrics['node_to_patch_mismatch'], labels, anomaly_high=True),
        eval_metric('patch_internal_inconsistency', metrics['patch_internal_inconsistency'], labels, anomaly_high=True),
        eval_metric('patch_density', metrics['patch_density'], labels, anomaly_high=True),
        eval_metric('patch_boundary_contrast', metrics['patch_boundary_contrast'], labels, anomaly_high=True),
    ]

    ndc_family.sort(key=lambda x: x['auc'], reverse=True)
    anr_family.sort(key=lambda x: x['auc'], reverse=True)

    result = {
        'data': args.data,
        'seed': args.seed,
        'train_rate': args.train_rate,
        'alpha': args.alpha,
        'ndc_family': ndc_family,
        'anr_family': anr_family,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
