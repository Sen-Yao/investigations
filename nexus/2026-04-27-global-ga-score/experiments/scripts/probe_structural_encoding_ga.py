#!/usr/bin/env python3
import argparse, json, sys, time, math, hashlib
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path.home() / "VoxG"
sys.path.insert(0, str(ROOT))
from utils import load_mat, preprocess_features, normalize_adj


def load_dataset(dataset, train_rate=0.05, val_rate=0.1):
    dummy = argparse.Namespace(dataset=dataset, sample_rate=0.15)
    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat(dataset, train_rate, val_rate, args=dummy)
    if dataset in ["Amazon", "tf_finace", "reddit", "elliptic"]:
        features, _ = preprocess_features(features)
        features = np.asarray(features, dtype=np.float32)
    else:
        features = np.asarray(features.todense(), dtype=np.float32)
    labels_np = np.asarray(ano_label).reshape(-1).astype(int)
    return adj.tocsr(), features, labels_np, np.asarray(normal_for_train_idx, dtype=int)


def metrics(score, labels, ks=(16,32,64,128,256,512)):
    y = labels.astype(int)
    s = np.asarray(score, dtype=np.float64)
    # If score is reversed, report as-is; caller decides. AUC<0.5 is useful diagnostic.
    out = {"auc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s))}
    order = np.argsort(-s)
    for k in ks:
        kk = min(k, len(y))
        out[f"top{k}_density"] = float(np.mean(y[order[:kk]] == 1))
    return out


def diag_mahal(z, train_idx, eps=1e-6):
    zn = z[train_idx]
    mu = zn.mean(axis=0, keepdims=True)
    sd = zn.std(axis=0, keepdims=True) + eps
    return np.mean(((z - mu) / sd) ** 2, axis=1)


def pca_rec(z, train_idx, max_components=8):
    zn = z[train_idx]
    scaler = StandardScaler(with_mean=True, with_std=True)
    zn_s = scaler.fit_transform(zn)
    z_s = scaler.transform(z)
    ncomp = int(min(max_components, zn_s.shape[0] - 1, zn_s.shape[1]))
    if ncomp < 1:
        return np.zeros(z.shape[0]), {"components": 0}
    pca = PCA(n_components=ncomp, random_state=0)
    pca.fit(zn_s)
    rec = pca.inverse_transform(pca.transform(z_s))
    return np.mean((z_s - rec) ** 2, axis=1), {"components": ncomp, "evr_sum": float(pca.explained_variance_ratio_.sum())}


def rwse(adj, steps=8):
    # Random-walk structural encoding: diag(P^k), k=1..K.
    csr = adj.tocsr().astype(np.float64)
    deg = np.asarray(csr.sum(axis=1)).reshape(-1)
    inv_deg = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    P = sp.diags(inv_deg).dot(csr).tocsr()
    cur = P.copy()
    feats = []
    for k in range(1, steps + 1):
        feats.append(cur.diagonal().astype(np.float64))
        if k < steps:
            cur = cur.dot(P).tocsr()
    return np.stack(feats, axis=1)


def distance_profile(adj, max_radius=4):
    # Reachability shell profile via boolean sparse powers, normalized by n.
    # shell_r = nodes reachable within r minus reachable within r-1.
    n = adj.shape[0]
    A = adj.copy().astype(bool).astype(np.int8).tocsr()
    reach_prev = sp.eye(n, format="csr", dtype=np.int8)
    frontier = sp.eye(n, format="csr", dtype=np.int8)
    profiles = []
    for r in range(1, max_radius + 1):
        frontier = (frontier.dot(A) > 0).astype(np.int8).tocsr()
        reach_cur = ((reach_prev + frontier) > 0).astype(np.int8).tocsr()
        shell = (reach_cur.astype(np.int16) - reach_prev.astype(np.int16)).maximum(0).tocsr()
        profiles.append(np.asarray(shell.sum(axis=1)).reshape(-1) / max(1, n - 1))
        reach_prev = reach_cur
    return np.stack(profiles, axis=1).astype(np.float64)


def stable_hash(items):
    s = "|".join(map(str, items))
    return hashlib.md5(s.encode()).hexdigest()[:12]


def wl_role_nll(adj, normal_idx, iters=2, degree_bins=16, alpha=1.0):
    deg = np.asarray(adj.sum(axis=1)).reshape(-1)
    # Quantile/bucket degree into finite roles.
    qs = np.quantile(deg, np.linspace(0, 1, degree_bins + 1))
    colors = np.searchsorted(qs[1:-1], deg, side="right").astype(str)
    csr = adj.tocsr()
    for _ in range(iters):
        new = []
        for u in range(csr.shape[0]):
            neigh = csr.indices[csr.indptr[u]:csr.indptr[u+1]]
            cnt = Counter(colors[neigh])
            sig = [colors[u]] + [f"{k}:{v}" for k, v in sorted(cnt.items())]
            new.append(stable_hash(sig))
        colors = np.asarray(new)
    normal_colors = colors[normal_idx]
    cnt = Counter(normal_colors)
    total = len(normal_colors)
    vocab = len(set(colors))
    score = np.empty(len(colors), dtype=np.float64)
    for i, c in enumerate(colors):
        # smoothed negative log likelihood under normal role distribution
        p = (cnt.get(c, 0) + alpha) / (total + alpha * max(1, vocab))
        score[i] = -math.log(p)
    return score, {"iters": iters, "degree_bins": degree_bins, "normal_role_count": len(cnt), "all_role_count": vocab}


def rank_percentile(x):
    x = np.asarray(x)
    order = np.argsort(x)
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x))
    return r / max(1, len(x)-1)


def run_dataset(dataset, rw_steps=8, sp_radius=4, wl_iters=2):
    t0 = time.time()
    adj, features, labels, normal_idx = load_dataset(dataset)
    result = {"meta": {"dataset": dataset, "n": int(adj.shape[0]), "m": int(adj.nnz//2), "normal_train": int(len(normal_idx)), "anomaly_rate": float(labels.mean())}, "scores": {}}

    print(f"  RWSE steps={rw_steps}", flush=True)
    r = rwse(adj, rw_steps)
    s = diag_mahal(r, normal_idx)
    result["scores"]["rwse_mahal"] = {"metrics": metrics(s, labels), "meta": {"steps": rw_steps, "dim": int(r.shape[1])}}
    s2, meta = pca_rec(r, normal_idx, max_components=min(8, rw_steps))
    result["scores"]["rwse_pca"] = {"metrics": metrics(s2, labels), "meta": meta | {"steps": rw_steps, "dim": int(r.shape[1])}}

    print(f"  distance profile radius={sp_radius}", flush=True)
    dp = distance_profile(adj, sp_radius)
    s = diag_mahal(dp, normal_idx)
    result["scores"]["spath_profile_mahal"] = {"metrics": metrics(s, labels), "meta": {"radius": sp_radius, "dim": int(dp.shape[1])}}
    s2, meta = pca_rec(dp, normal_idx, max_components=min(4, sp_radius))
    result["scores"]["spath_profile_pca"] = {"metrics": metrics(s2, labels), "meta": meta | {"radius": sp_radius, "dim": int(dp.shape[1])}}

    print(f"  WL role iters={wl_iters}", flush=True)
    s, meta = wl_role_nll(adj, normal_idx, iters=wl_iters)
    result["scores"]["wl_role_nll"] = {"metrics": metrics(s, labels), "meta": meta}

    # no-weight structural consensus diagnostics
    rp = rank_percentile(result_score := s)  # wl last
    rw = rank_percentile(diag_mahal(r, normal_idx))
    sd = rank_percentile(diag_mahal(dp, normal_idx))
    combo_min = np.minimum(np.minimum(rw, sd), rp)
    combo_prod = rw * sd * rp
    result["scores"]["struct_consensus_min"] = {"metrics": metrics(combo_min, labels), "meta": {"combine": "min(percentile(rwse), percentile(spath), percentile(wl))"}}
    result["scores"]["struct_consensus_prod"] = {"metrics": metrics(combo_prod, labels), "meta": {"combine": "product(percentile(rwse), percentile(spath), percentile(wl))"}}

    result["elapsed_sec"] = time.time() - t0
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["photo", "elliptic"])
    ap.add_argument("--rw-steps", type=int, default=8)
    ap.add_argument("--sp-radius", type=int, default=4)
    ap.add_argument("--wl-iters", type=int, default=2)
    ap.add_argument("--outdir", default=str(Path.home() / "VoxG/nexus/investigations/2026-04-27-global-ga-score/experiments/outputs/structural_encoding_ga"))
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for ds in args.datasets:
        print(f"DATASET_START {ds}", flush=True)
        res = run_dataset(ds, args.rw_steps, args.sp_radius, args.wl_iters)
        (outdir / f"{ds}_structural_encoding_ga.json").write_text(json.dumps(res, indent=2))
        summary[ds] = res
        print("DATASET_DONE", ds, json.dumps({k:v["metrics"] for k,v in res["scores"].items()}, ensure_ascii=False), flush=True)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("ALL_DONE", outdir / "summary.json")

if __name__ == "__main__":
    main()
