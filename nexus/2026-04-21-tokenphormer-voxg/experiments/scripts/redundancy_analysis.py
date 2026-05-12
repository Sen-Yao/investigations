#!/usr/bin/env python3
"""
Redundancy and complementarity analysis for hop/delta/prototype/consistency.
"""

import argparse
import json
from itertools import combinations

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(path):
    data = sio.loadmat(path)
    adj = data.get("Network", data.get("A"))
    feat = data.get("Attributes", data.get("X"))
    label = data.get("Label", data.get("Y")).flatten()
    if sp.issparse(adj):
        adj = adj.toarray()
    if sp.issparse(feat):
        feat = feat.toarray()
    label = np.asarray(label).astype(int)
    label = label - label.min()
    return np.asarray(adj), np.asarray(feat, dtype=float), label


def normalize_adj(adj):
    deg = adj.sum(axis=1)
    inv_sqrt = np.zeros_like(deg, dtype=float)
    mask = deg > 0
    inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
    d = np.diag(inv_sqrt)
    return d @ adj @ d


def compute_hop_and_delta(features, adj_norm, k):
    n, d = features.shape
    hops = np.zeros((n, k + 1, d), dtype=float)
    hops[:, 0, :] = features
    cur = features.copy()
    for i in range(1, k + 1):
        cur = adj_norm @ cur
        hops[:, i, :] = cur
    delta = hops[:, 1:, :] - hops[:, :-1, :]
    return hops, delta


def cosine(a, b, eps=1e-12):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mean_pairwise_cos(vectors, max_pairs=64):
    m = len(vectors)
    if m < 2:
        return 0.0
    idx = list(combinations(range(m), 2))
    if len(idx) > max_pairs:
        step = max(1, len(idx) // max_pairs)
        idx = idx[::step][:max_pairs]
    vals = [cosine(vectors[i].ravel(), vectors[j].ravel()) for i, j in idx]
    return float(np.mean(vals)) if vals else 0.0


def compute_extra_features(adj, delta):
    n = adj.shape[0]
    flat_delta = delta.reshape(n, -1)
    features = {
        "delta_corr_to_proto": np.zeros(n),
        "delta_cos_to_proto": np.zeros(n),
        "delta_l2_to_proto": np.zeros(n),
        "proto_delta_norm": np.zeros(n),
        "neigh_delta_var": np.zeros(n),
        "neigh_delta_mean_pairwise_cos": np.zeros(n),
        "node_to_proto_l2": np.zeros(n),
        "node_to_proto_cos": np.zeros(n),
    }
    proto_store = np.zeros_like(flat_delta)
    for i in range(n):
        neigh = np.where(adj[i] > 0)[0]
        if len(neigh) == 0:
            continue
        neigh_delta = flat_delta[neigh]
        proto = neigh_delta.mean(axis=0)
        proto_store[i] = proto
        node = flat_delta[i]
        node_std = np.std(node)
        proto_std = np.std(proto)
        if node_std > 1e-12 and proto_std > 1e-12:
            features["delta_corr_to_proto"][i] = float(np.corrcoef(node, proto)[0, 1])
        features["delta_cos_to_proto"][i] = cosine(node, proto)
        features["delta_l2_to_proto"][i] = float(np.linalg.norm(node - proto))
        features["proto_delta_norm"][i] = float(np.linalg.norm(proto))
        features["neigh_delta_var"][i] = float(np.mean(np.var(neigh_delta, axis=0)))
        features["neigh_delta_mean_pairwise_cos"][i] = mean_pairwise_cos(neigh_delta)
        features["node_to_proto_l2"][i] = float(np.linalg.norm(node - proto))
        features["node_to_proto_cos"][i] = cosine(node, proto)
    proto_only = np.column_stack([
        features["delta_corr_to_proto"],
        features["delta_cos_to_proto"],
        features["delta_l2_to_proto"],
        features["proto_delta_norm"],
    ])
    consistency_only = np.column_stack([
        features["neigh_delta_var"],
        features["neigh_delta_mean_pairwise_cos"],
        features["node_to_proto_l2"],
        features["node_to_proto_cos"],
    ])
    return flat_delta, proto_store, proto_only, consistency_only


def fit_linear_recover(x, y, seed=0):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=seed)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train)
    x_test = x_scaler.transform(x_test)
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)
    model = Ridge(alpha=1.0)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return {
        "r2": float(r2_score(y_test, pred, multioutput="variance_weighted")),
        "mse": float(np.mean((pred - y_test) ** 2)),
    }


def logistic_with_residuals(x, y, seed=0):
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        x, y, np.arange(len(y)), test_size=0.5, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(x_train, y_train)
    prob = clf.predict_proba(x_test)[:, 1]
    residual = np.abs(y_test - prob)
    return {
        "auc": float(roc_auc_score(y_test, prob)),
        "ap": float(average_precision_score(y_test, prob)),
        "prob": prob,
        "residual": residual,
        "y_test": y_test,
        "idx_test": idx_test,
    }


def residual_correlation(residual, feature_matrix):
    out = []
    for j in range(feature_matrix.shape[1]):
        col = feature_matrix[:, j]
        if np.std(col) < 1e-12 or np.std(residual) < 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(residual, col)[0, 1])
        out.append(corr)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    adj, x, y = load_dataset(args.data)
    adj_norm = normalize_adj(adj)
    hops, delta = compute_hop_and_delta(x, adj_norm, args.k)

    hop_flat = hops.reshape(hops.shape[0], -1)
    delta_flat, proto_seq, proto_only, consistency_only = compute_extra_features(adj, delta)

    linear_recovery = {
        "hop_to_delta": fit_linear_recover(hop_flat, delta_flat),
        "delta_to_hop": fit_linear_recover(delta_flat, hop_flat),
        "delta_to_proto_seq": fit_linear_recover(delta_flat, proto_seq),
        "delta_to_prototype_only": fit_linear_recover(delta_flat, proto_only),
        "delta_to_consistency_only": fit_linear_recover(delta_flat, consistency_only),
    }

    delta_probe = logistic_with_residuals(delta_flat, y)
    residual = delta_probe["residual"]
    test_idx = delta_probe["idx_test"]

    residual_analysis = {
        "prototype_feature_residual_corr": residual_correlation(residual, proto_only[test_idx]),
        "consistency_feature_residual_corr": residual_correlation(residual, consistency_only[test_idx]),
    }

    result = {
        "data": args.data,
        "k": args.k,
        "shapes": {
            "hop_flat": list(hop_flat.shape),
            "delta_flat": list(delta_flat.shape),
            "prototype_only": list(proto_only.shape),
            "consistency_only": list(consistency_only.shape),
        },
        "delta_probe": {
            "auc": delta_probe["auc"],
            "ap": delta_probe["ap"],
        },
        "linear_recovery": linear_recovery,
        "residual_analysis": residual_analysis,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
