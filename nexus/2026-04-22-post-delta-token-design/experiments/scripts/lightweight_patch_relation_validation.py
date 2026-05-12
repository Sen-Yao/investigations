#!/usr/bin/env python3
"""
Memory-safe lightweight patch/relation validation for post-delta investigation.
"""

import argparse
import json

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RNG = np.random.default_rng(0)


def load_dataset(path):
    data = sio.loadmat(path)
    adj = data.get("Network", data.get("A"))
    feat = data.get("Attributes", data.get("X"))
    label = data.get("Label", data.get("Y")).flatten()
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    else:
        adj = adj.tocsr()
    if sp.issparse(feat):
        feat = feat.toarray()
    feat = np.asarray(feat, dtype=np.float32)
    label = np.asarray(label).astype(int)
    label = label - label.min()
    return adj, feat, label


def cosine(a, b, eps=1e-12):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def sample_indices(idx, max_items):
    if len(idx) <= max_items:
        return np.asarray(idx, dtype=int)
    return np.sort(RNG.choice(np.asarray(idx, dtype=int), size=max_items, replace=False))


def mean_pairwise_cos(vectors, max_points=24, max_pairs=48):
    m = len(vectors)
    if m < 2:
        return 0.0
    if m > max_points:
        sel = sample_indices(np.arange(m), max_points)
        vectors = vectors[sel]
        m = len(vectors)
    vals = []
    for _ in range(min(max_pairs, m * (m - 1) // 2)):
        i, j = RNG.integers(0, m, size=2)
        if i == j:
            continue
        vals.append(cosine(vectors[i], vectors[j]))
    return float(np.mean(vals)) if vals else 0.0


def patch_density_sparse(sub_adj):
    n = sub_adj.shape[0]
    if n <= 1:
        return 0.0
    edges = sub_adj.nnz / 2.0
    return float(edges / (n * (n - 1) / 2.0))


def get_neighbors(adj, i, max_neighbors):
    start, end = adj.indptr[i], adj.indptr[i + 1]
    neigh = adj.indices[start:end]
    return sample_indices(neigh, max_neighbors)


def compute_features(adj, x, max_neighbors=64, max_boundary=64):
    n = adj.shape[0]
    patch_features = {k: np.zeros(n, dtype=np.float32) for k in [
        "patch_size", "patch_density", "patch_feature_mean_norm", "patch_feature_var",
        "patch_internal_mean_pairwise_cos", "target_to_patch_center_cos", "target_to_patch_center_l2",
        "patch_boundary_contrast", "patch_internal_external_gap"
    ]}
    relation_features = {k: np.zeros(n, dtype=np.float32) for k in [
        "node_to_neigh_cos", "node_to_neigh_l2", "node_to_neigh_corr", "node_to_patch_cos",
        "node_to_patch_l2", "neigh_internal_mean_pairwise_cos", "patch_internal_mean_pairwise_cos",
        "node_context_contrast_gap"
    ]}

    for i in range(n):
        neigh = get_neighbors(adj, i, max_neighbors)
        patch = np.unique(np.concatenate(([i], neigh)))
        patch_x = x[patch]
        patch_center = patch_x.mean(axis=0)
        patch_features["patch_size"][i] = len(patch)
        patch_features["patch_density"][i] = patch_density_sparse(adj[patch][:, patch])
        patch_features["patch_feature_mean_norm"][i] = float(np.linalg.norm(patch_center))
        patch_features["patch_feature_var"][i] = float(np.mean(np.var(patch_x, axis=0)))
        patch_ipc = mean_pairwise_cos(patch_x)
        patch_features["patch_internal_mean_pairwise_cos"][i] = patch_ipc
        patch_features["target_to_patch_center_cos"][i] = cosine(x[i], patch_center)
        patch_features["target_to_patch_center_l2"][i] = float(np.linalg.norm(x[i] - patch_center))

        boundary_mask = np.asarray(adj[patch].sum(axis=0)).ravel() > 0
        boundary = np.where(boundary_mask)[0]
        boundary = np.setdiff1d(boundary, patch, assume_unique=False)
        boundary = sample_indices(boundary, max_boundary)
        if len(boundary) > 0:
            boundary_center = x[boundary].mean(axis=0)
            pbc = float(np.linalg.norm(patch_center - boundary_center))
        else:
            pbc = 0.0
        patch_features["patch_boundary_contrast"][i] = pbc
        patch_features["patch_internal_external_gap"][i] = patch_ipc - pbc

        if len(neigh) > 0:
            neigh_x = x[neigh]
            neigh_center = neigh_x.mean(axis=0)
            relation_features["node_to_neigh_cos"][i] = cosine(x[i], neigh_center)
            relation_features["node_to_neigh_l2"][i] = float(np.linalg.norm(x[i] - neigh_center))
            relation_features["node_to_neigh_corr"][i] = corr(x[i], neigh_center)
            relation_features["neigh_internal_mean_pairwise_cos"][i] = mean_pairwise_cos(neigh_x)
        relation_features["node_to_patch_cos"][i] = cosine(x[i], patch_center)
        relation_features["node_to_patch_l2"][i] = float(np.linalg.norm(x[i] - patch_center))
        relation_features["patch_internal_mean_pairwise_cos"][i] = patch_ipc
        relation_features["node_context_contrast_gap"][i] = relation_features["node_to_patch_cos"][i] - patch_ipc

    return patch_features, relation_features


def summarize_stats(features, y):
    out = {}
    normal = y == 0
    anomaly = y == 1
    for name, vals in features.items():
        ks_stat, ks_p = ks_2samp(vals[normal], vals[anomaly])
        out[name] = {
            "normal_mean": float(np.mean(vals[normal])),
            "anomaly_mean": float(np.mean(vals[anomaly])),
            "difference": float(np.mean(vals[anomaly]) - np.mean(vals[normal])),
            "ks_stat": float(ks_stat),
            "ks_p": float(ks_p),
        }
    return out


def probe_auc(x, y, seed=0):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=seed, stratify=y)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    clf = LogisticRegression(max_iter=1000, random_state=seed, solver="liblinear")
    clf.fit(x_train, y_train)
    prob = clf.predict_proba(x_test)[:, 1]
    return {
        "auc": float(roc_auc_score(y_test, prob)),
        "ap": float(average_precision_score(y_test, prob)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-neighbors", type=int, default=64)
    parser.add_argument("--max-boundary", type=int, default=64)
    args = parser.parse_args()

    adj, x, y = load_dataset(args.data)
    patch_features, relation_features = compute_features(adj, x, args.max_neighbors, args.max_boundary)
    patch_mat = np.column_stack(list(patch_features.values())).astype(np.float32)
    relation_mat = np.column_stack(list(relation_features.values())).astype(np.float32)

    result = {
        "data": args.data,
        "config": {"max_neighbors": args.max_neighbors, "max_boundary": args.max_boundary},
        "patch_feature_stats": summarize_stats(patch_features, y),
        "relation_feature_stats": summarize_stats(relation_features, y),
        "probes": {
            "node_only": probe_auc(x, y),
            "patch_only": probe_auc(patch_mat, y),
            "relation_only": probe_auc(relation_mat, y),
            "node_plus_patch": probe_auc(np.concatenate([x, patch_mat], axis=1), y),
            "node_plus_relation": probe_auc(np.concatenate([x, relation_mat], axis=1), y),
            "node_plus_patch_plus_relation": probe_auc(np.concatenate([x, patch_mat, relation_mat], axis=1), y),
        },
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"data": result["data"], "config": result["config"], "probes": result["probes"]}, indent=2))


if __name__ == "__main__":
    main()
