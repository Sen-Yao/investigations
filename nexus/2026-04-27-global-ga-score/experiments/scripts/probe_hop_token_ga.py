#!/usr/bin/env python3
import argparse, json, time, sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path.home() / "VoxG"
sys.path.insert(0, str(ROOT))
from utils import load_mat, preprocess_features, normalize_adj


def to_dense_features(dataset, features):
    if dataset in ["Amazon", "tf_finace", "reddit", "elliptic"]:
        features, _ = preprocess_features(features)
        return np.asarray(features, dtype=np.float32)
    return np.asarray(features.todense(), dtype=np.float32)


def build_hop_tokens(features, adj, hops=2):
    adj_norm = normalize_adj(adj)
    x = features.astype(np.float32)
    outs = [x]
    cur = x
    for _ in range(hops):
        cur = adj_norm.dot(cur)
        cur = np.asarray(cur, dtype=np.float32)
        outs.append(cur)
    return np.concatenate(outs, axis=1).astype(np.float32)


def rank_percentile(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x)
    r = np.empty_like(order, dtype=np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    return r / max(1, len(x) - 1)


def pca_reconstruction_error(z, train_idx, max_components=64):
    zn = z[train_idx]
    scaler = StandardScaler(with_mean=True, with_std=True)
    zn_s = scaler.fit_transform(zn)
    z_s = scaler.transform(z)
    ncomp = int(min(max_components, zn_s.shape[0] - 1, zn_s.shape[1]))
    if ncomp < 1:
        return np.zeros(z.shape[0], dtype=np.float64), {"components": 0}
    pca = PCA(n_components=ncomp, svd_solver="randomized", random_state=0)
    pca.fit(zn_s)
    rec = pca.inverse_transform(pca.transform(z_s))
    err = np.mean((z_s - rec) ** 2, axis=1)
    return err, {"components": ncomp, "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_))}


def diag_mahalanobis(z, train_idx, eps=1e-6):
    zn = z[train_idx]
    mu = zn.mean(axis=0, keepdims=True)
    std = zn.std(axis=0, keepdims=True) + eps
    return np.mean(((z - mu) / std) ** 2, axis=1)


def ego_struct_deviation(features, adj, train_idx):
    # lightweight structural stats, normal-calibrated with diagonal z-score
    csr = adj.tocsr()
    deg = np.asarray(csr.sum(axis=1)).reshape(-1).astype(np.float32)
    adj2 = csr @ csr
    deg2 = np.asarray(adj2.sum(axis=1)).reshape(-1).astype(np.float32)
    adj_norm = normalize_adj(adj)
    neigh_mean = np.asarray(adj_norm.dot(features), dtype=np.float32)
    neigh_res = np.mean((features - neigh_mean) ** 2, axis=1)
    # feature neighborhood variance approx: E[x^2] - E[x]^2
    neigh_x2 = np.asarray(adj_norm.dot(features ** 2), dtype=np.float32)
    neigh_var = np.mean(np.maximum(neigh_x2 - neigh_mean ** 2, 0), axis=1)
    stats = np.stack([np.log1p(deg), np.log1p(deg2), neigh_res, neigh_var], axis=1)
    return diag_mahalanobis(stats, train_idx), {"stats": ["log_degree", "log_2hop_degree", "neighbor_residual", "neighbor_variance"]}


def metrics(score, labels, ks=(16,32,64,128,256,512)):
    y = labels.astype(int)
    s = np.asarray(score, dtype=np.float64)
    out = {
        "auc": float(roc_auc_score(y, s)),
        "ap": float(average_precision_score(y, s)),
    }
    order = np.argsort(-s)
    for k in ks:
        kk = min(k, len(y))
        out[f"top{k}_density"] = float(np.mean(y[order[:kk]] == 1))
    return out


def run_dataset(dataset, train_rate=0.05, val_rate=0.1, hops=2, pca_components=64):
    t0 = time.time()
    dummy = argparse.Namespace(dataset=dataset, sample_rate=0.15)
    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat(dataset, train_rate, val_rate, args=dummy)
    features = to_dense_features(dataset, features)
    labels_np = np.asarray(ano_label).reshape(-1).astype(int)
    normal_train = np.asarray(normal_for_train_idx, dtype=int)

    z = build_hop_tokens(features, adj, hops=hops)
    scores = {}
    meta = {"dataset": dataset, "n": int(len(labels_np)), "feature_dim": int(features.shape[1]), "hop_token_dim": int(z.shape[1]), "normal_train": int(len(normal_train)), "anomaly_rate": float(labels_np.mean())}

    s, m = pca_reconstruction_error(z, normal_train, pca_components)
    scores["hop_pca_rec"] = {"raw": s, "meta": m}
    s = diag_mahalanobis(z, normal_train)
    scores["hop_diag_mahal"] = {"raw": s, "meta": {}}
    s, m = ego_struct_deviation(features, adj, normal_train)
    scores["ego_struct_dev"] = {"raw": s, "meta": m}

    # simple hybrid diagnostics without extra weights
    hp = rank_percentile(scores["hop_pca_rec"]["raw"])
    ep = rank_percentile(scores["ego_struct_dev"]["raw"])
    scores["min_hop_pca_ego"] = {"raw": np.minimum(hp, ep), "meta": {"combine": "min(percentile(hop_pca), percentile(ego_struct))"}}
    scores["prod_hop_pca_ego"] = {"raw": hp * ep, "meta": {"combine": "product(percentile(hop_pca), percentile(ego_struct))"}}

    result = {"meta": meta, "scores": {}, "elapsed_sec": None}
    for name, obj in scores.items():
        result["scores"][name] = {"metrics": metrics(obj["raw"], labels_np), "meta": obj["meta"]}
    result["elapsed_sec"] = time.time() - t0
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["photo", "elliptic"])
    ap.add_argument("--outdir", default=str(Path.home() / "VoxG/nexus/investigations/2026-04-27-global-ga-score/experiments/outputs/hop_token_ga"))
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--pca-components", type=int, default=64)
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for ds in args.datasets:
        print(f"DATASET_START {ds}", flush=True)
        res = run_dataset(ds, hops=args.hops, pca_components=args.pca_components)
        (outdir / f"{ds}_hop_token_ga.json").write_text(json.dumps(res, indent=2))
        summary[ds] = res
        print("DATASET_DONE", ds, json.dumps({k:v["metrics"] for k,v in res["scores"].items()}, ensure_ascii=False), flush=True)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("ALL_DONE", outdir / "summary.json")

if __name__ == "__main__":
    main()
