#!/usr/bin/env python3
"""
Score V1 probe with center / NDC / ANR components.
Strictly normal-only for construction; labels are only used for evaluation.
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


def cosine(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


def build_components(z, adj, train_normal_idx, max_patch_neighbors=64):
    center = z[train_normal_idx].mean(axis=0)
    center = center / (np.linalg.norm(center) + 1e-12)
    center_term = z @ center

    n = z.shape[0]
    ndc = np.zeros(n, dtype=np.float32)
    anr = np.zeros(n, dtype=np.float32)
    for i in range(n):
        neigh = adj.indices[adj.indptr[i]:adj.indptr[i+1]]
        if len(neigh) > 0:
            neigh_center = z[neigh].mean(axis=0)
            ndc[i] = 1.0 - cosine(z[i], neigh_center)
        else:
            ndc[i] = 1.0

        if len(neigh) > max_patch_neighbors:
            neigh = neigh[:max_patch_neighbors]
        patch = np.unique(np.concatenate(([i], neigh)))
        patch_z = z[patch]
        patch_center = patch_z.mean(axis=0)
        target_patch_mismatch = 1.0 - cosine(z[i], patch_center)
        if len(patch) > 1:
            sims = []
            m = min(len(patch), 16)
            sample = patch_z[:m]
            for a in range(m):
                for b in range(a + 1, m):
                    sims.append(cosine(sample[a], sample[b]))
            patch_consistency = float(np.mean(sims)) if sims else 1.0
        else:
            patch_consistency = 1.0
        anr[i] = 0.5 * (target_patch_mismatch + (1.0 - patch_consistency))
    return center_term, ndc, anr


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
    center_term, ndc_term, anr_term = build_components(z, adj, train_normal_idx)

    scores = {
        'center_only': center_term,
        'center_plus_ndc': 0.5 * center_term - 0.5 * ndc_term,
        'center_plus_anr': 0.5 * center_term - 0.5 * anr_term,
        'center_plus_ndc_plus_anr': 0.5 * center_term - 0.25 * ndc_term - 0.25 * anr_term,
    }

    result = {
        'data': args.data,
        'seed': args.seed,
        'train_rate': args.train_rate,
        'alpha': args.alpha,
        'components': {
            'center_term': eval_score(center_term, labels),
            'ndc_term': eval_score(ndc_term, labels),
            'anr_term': eval_score(anr_term, labels),
        },
        'scores': {k: eval_score(v, labels) for k, v in scores.items()},
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
