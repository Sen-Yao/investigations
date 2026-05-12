#!/usr/bin/env python3
"""
Lightweight validation for anomaly-aware token features.

Purpose:
- compute delta prototype features
- compute local consistency features
- compare normal vs anomaly distributions
- run lightweight logistic probes

Note:
This script is for exploratory analysis on labeled benchmarks.
It should not be interpreted as a deployable semi-supervised detector.
"""

import argparse
import json
import os
from itertools import combinations

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
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


def corrcoef_safe(a, b, eps=1e-12):
    if np.std(a) < eps or np.std(b) < eps:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


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


def compute_features(adj, delta):
    n = adj.shape[0]
    feat = {
        "delta_corr_to_proto": np.zeros(n),
        "delta_cos_to_proto": np.zeros(n),
        "delta_l2_to_proto": np.zeros(n),
        "proto_delta_norm": np.zeros(n),
        "neigh_delta_var": np.zeros(n),
        "neigh_delta_mean_pairwise_cos": np.zeros(n),
        "node_to_proto_l2": np.zeros(n),
        "node_to_proto_cos": np.zeros(n),
    }
    flat_delta = delta.reshape(n, -1)
    for i in range(n):
        neigh = np.where(adj[i] > 0)[0]
        if len(neigh) == 0:
            continue
        neigh_delta = flat_delta[neigh]
        proto = neigh_delta.mean(axis=0)
        node = flat_delta[i]
        feat["delta_corr_to_proto"][i] = corrcoef_safe(node, proto)
        feat["delta_cos_to_proto"][i] = cosine(node, proto)
        feat["delta_l2_to_proto"][i] = float(np.linalg.norm(node - proto))
        feat["proto_delta_norm"][i] = float(np.linalg.norm(proto))
        feat["neigh_delta_var"][i] = float(np.mean(np.var(neigh_delta, axis=0)))
        feat["neigh_delta_mean_pairwise_cos"][i] = mean_pairwise_cos(neigh_delta)
        feat["node_to_proto_l2"][i] = float(np.linalg.norm(node - proto))
        feat["node_to_proto_cos"][i] = cosine(node, proto)
    return feat, flat_delta


def ks_report(values, labels):
    normal = values[labels == 0]
    anomaly = values[labels == 1]
    ks_stat, ks_p = stats.ks_2samp(normal, anomaly)
    return {
        "normal_mean": float(np.mean(normal)),
        "anomaly_mean": float(np.mean(anomaly)),
        "difference": float(np.mean(anomaly) - np.mean(normal)),
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
    }


def run_probe(x, y, seed=0):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(x_train, y_train)
    prob = clf.predict_proba(x_test)[:, 1]
    return {
        "auc": float(roc_auc_score(y_test, prob)),
        "ap": float(average_precision_score(y_test, prob)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    adj, x, y = load_dataset(args.data)
    adj_norm = normalize_adj(adj)
    _, delta = compute_hop_and_delta(x, adj_norm, args.k)
    extra_feat, flat_delta = compute_features(adj, delta)

    stats_out = {name: ks_report(vals, y) for name, vals in extra_feat.items()}

    proto_only = np.column_stack([
        extra_feat["delta_corr_to_proto"],
        extra_feat["delta_cos_to_proto"],
        extra_feat["delta_l2_to_proto"],
        extra_feat["proto_delta_norm"],
    ])
    consistency_only = np.column_stack([
        extra_feat["neigh_delta_var"],
        extra_feat["neigh_delta_mean_pairwise_cos"],
        extra_feat["node_to_proto_l2"],
        extra_feat["node_to_proto_cos"],
    ])

    probe_inputs = {
        "delta_flatten": flat_delta,
        "prototype_only": proto_only,
        "consistency_only": consistency_only,
        "delta_plus_prototype": np.concatenate([flat_delta, proto_only], axis=1),
        "delta_plus_consistency": np.concatenate([flat_delta, consistency_only], axis=1),
        "delta_plus_prototype_plus_consistency": np.concatenate([flat_delta, proto_only, consistency_only], axis=1),
    }
    probes = {name: run_probe(arr, y) for name, arr in probe_inputs.items()}

    result = {
        "data": os.path.basename(args.data),
        "k": args.k,
        "feature_stats": stats_out,
        "probes": probes,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
