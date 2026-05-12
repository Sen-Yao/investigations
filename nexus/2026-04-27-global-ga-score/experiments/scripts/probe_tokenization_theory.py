#!/usr/bin/env python3
"""Offline probe for normal-calibrated dual-reference tokenization.

This script does not train a model. It evaluates whether the proposed
G_n/L_n/G_a/L_a decomposition produces interpretable reference sequences.
"""
import argparse, json, sys, time, random
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

ROOT = Path.home() / "VoxG"
sys.path.insert(0, str(ROOT))
from utils import load_mat, preprocess_features, normalize_adj


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)


def to_dense_features(dataset, features):
    if dataset in ["Amazon", "tf_finace", "reddit", "elliptic"]:
        features, _ = preprocess_features(features)
        return np.asarray(features, dtype=np.float32)
    return np.asarray(features.todense(), dtype=np.float32)


def l2_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def rank_percentile(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x)
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    return r / max(1, len(x) - 1)


def build_hop_attr(features, adj, hops=2):
    adj_norm = normalize_adj(adj)
    x = features.astype(np.float32)
    outs = [x]
    cur = x
    for _ in range(hops):
        cur = np.asarray(adj_norm.dot(cur), dtype=np.float32)
        outs.append(cur)
    return np.concatenate(outs, axis=1).astype(np.float32)


def rwse(adj, steps=8):
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
    return np.stack(feats, axis=1).astype(np.float32)


def build_descriptor(mode, features, adj, hops=2, rw_steps=8):
    if mode == "hop_attr":
        return build_hop_attr(features, adj, hops)
    if mode == "rwse":
        return rwse(adj, rw_steps)
    if mode == "hop_attr_rwse":
        return np.concatenate([build_hop_attr(features, adj, hops), rwse(adj, rw_steps)], axis=1).astype(np.float32)
    raise ValueError(f"unknown descriptor_mode: {mode}")


class NormalModel:
    def __init__(self, estimator: str, z: np.ndarray, normal_idx: np.ndarray, pca_components: int = 32):
        self.estimator = estimator
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.zs = self.scaler.fit_transform(z[normal_idx])
        self.z_all = self.scaler.transform(z)
        self.mu = self.zs.mean(axis=0, keepdims=True)
        self.std = self.zs.std(axis=0, keepdims=True) + 1e-6
        self.pca = None
        if estimator == "pca_residual":
            ncomp = int(min(pca_components, self.zs.shape[0] - 1, self.zs.shape[1]))
            if ncomp > 0:
                self.pca = PCA(n_components=ncomp, svd_solver="randomized", random_state=0)
                self.pca.fit(self.zs)
        elif estimator != "diag_gaussian":
            raise ValueError(f"unknown pn_estimator: {estimator}")

    def rejection(self) -> np.ndarray:
        if self.estimator == "diag_gaussian" or self.pca is None:
            return np.mean(((self.z_all - self.mu) / self.std) ** 2, axis=1)
        rec = self.pca.inverse_transform(self.pca.transform(self.z_all))
        return np.mean((self.z_all - rec) ** 2, axis=1)

    def residual(self) -> np.ndarray:
        if self.estimator == "diag_gaussian" or self.pca is None:
            return ((self.z_all - self.mu) / self.std).astype(np.float32)
        rec = self.pca.inverse_transform(self.pca.transform(self.z_all))
        return (self.z_all - rec).astype(np.float32)

    def density_score(self) -> np.ndarray:
        return -self.rejection()


def normal_soft_or_score(features, adj, normal_idx):
    hop = build_hop_attr(features, adj, hops=2)
    nm = NormalModel("diag_gaussian", hop, normal_idx)
    hop_rej = rank_percentile(nm.rejection())
    struct = rwse(adj, steps=8)
    sm = NormalModel("diag_gaussian", struct, normal_idx)
    struct_rej = rank_percentile(sm.rejection())
    return 1.0 - (1.0 - hop_rej) * (1.0 - struct_rej)


def topk_metrics(score, labels, ks=(16,32,64,128,256,512)):
    y = labels.astype(int)
    s = np.asarray(score, dtype=np.float64)
    out = {"auc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s))}
    order = np.argsort(-s)
    for k in ks:
        kk = min(k, len(y)); out[f"top{k}_density"] = float(np.mean(y[order[:kk]] == 1))
    return out


def cosine_rows_to_matrix(a, b, block=1024):
    an = l2_rows(a.astype(np.float32)); bn = l2_rows(b.astype(np.float32))
    out = []
    for st in range(0, an.shape[0], block):
        out.append(an[st:st+block] @ bn.T)
    return np.vstack(out)


def select_refs(z, residual, labels, normal_idx, normal_model, features, adj, args):
    n = z.shape[0]
    rejection = normal_model.rejection()
    density = normal_model.density_score()
    residual_norm = np.linalg.norm(residual, axis=1)

    if args.gn_mode == "label_gate":
        normal_pool = np.asarray(normal_idx)
        gn = np.zeros(n, dtype=np.float32); gn[normal_pool] = 1.0
    elif args.gn_mode == "normal_density":
        normal_pool = np.arange(n); gn = rank_percentile(density).astype(np.float32)
    elif args.gn_mode == "label_gate_density":
        normal_pool = np.asarray(normal_idx); gn = rank_percentile(density).astype(np.float32); gn[np.setdiff1d(np.arange(n), normal_pool)] = -1e9
    else:
        raise ValueError(args.gn_mode)

    if args.ga_mode == "normal_rejection":
        ga = rank_percentile(rejection).astype(np.float32)
    elif args.ga_mode == "residual_norm":
        ga = rank_percentile(residual_norm).astype(np.float32)
    elif args.ga_mode == "normal_soft_or":
        ga = normal_soft_or_score(features, adj, normal_idx).astype(np.float32)
    else:
        raise ValueError(args.ga_mode)

    # L_n candidates: descriptor similarity or single-reference reconstruction gain approximation.
    sim_n = cosine_rows_to_matrix(z, z[normal_pool])
    if args.ln_mode == "descriptor_similarity":
        ln_mat = sim_n
    elif args.ln_mode == "reconstruction_gain":
        # For a single normalized basis vector, lower reconstruction error is proportional to squared cosine.
        ln_mat = sim_n ** 2
    else:
        raise ValueError(args.ln_mode)
    n_scores = ln_mat + gn[normal_pool][None, :]
    n_order = np.argsort(-n_scores, axis=1)[:, :args.normal_k]
    normal_refs = normal_pool[n_order]

    if args.la_mode == "residual_cosine":
        l_a = cosine_rows_to_matrix(residual, residual)
    elif args.la_mode == "descriptor_similarity":
        l_a = cosine_rows_to_matrix(z, z)
    else:
        raise ValueError(args.la_mode)
    a_scores = l_a + ga[None, :]
    np.fill_diagonal(a_scores, -1e9)
    a_order = np.argsort(-a_scores, axis=1)[:, :args.anom_k]
    anom_refs = a_order.astype(np.int64)

    return normal_refs, anom_refs, {"gn": gn, "ga": ga, "rejection": rejection, "density": density, "residual_norm": residual_norm}


def projection_error(vectors, refs, ridge=1e-4):
    errs = np.zeros(vectors.shape[0], dtype=np.float64)
    Xall = vectors.astype(np.float64)
    for i in range(Xall.shape[0]):
        B = Xall[refs[i]].T  # d x k
        y = Xall[i]
        # ridge least squares; stable even when d >> k
        A = B.T @ B + ridge * np.eye(B.shape[1])
        coef = np.linalg.solve(A, B.T @ y)
        rec = B @ coef
        errs[i] = np.mean((y - rec) ** 2)
    return errs


def run_one(args) -> Dict:
    dummy = argparse.Namespace(dataset=args.dataset, sample_rate=0.15)
    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat(args.dataset, args.train_rate, 0.1, args=dummy)
    features = to_dense_features(args.dataset, features)
    labels_np = np.asarray(ano_label).reshape(-1).astype(int)
    normal_idx = np.asarray(normal_for_train_idx, dtype=int)
    assert np.sum(labels_np[normal_idx]) == 0, "Data leakage: normal_for_train_idx contains anomalies"

    z = build_descriptor(args.descriptor_mode, features, adj, args.hops, args.rw_steps)
    nm = NormalModel(args.pn_estimator, z, normal_idx, args.pca_components)
    residual = nm.residual()
    normal_refs, anom_refs, scores = select_refs(z, residual, labels_np, normal_idx, nm, features, adj, args)

    rn_err = projection_error(nm.z_all, normal_refs)
    rnorm = l2_rows(residual)
    align = np.mean(np.sum(rnorm[:, None, :] * rnorm[anom_refs], axis=2), axis=1)
    normal_sim = np.mean(cosine_rows_to_matrix(z, z[normal_idx])[:, :min(args.normal_k, len(normal_idx))], axis=1) if False else np.mean(np.sum(l2_rows(z)[:, None, :] * l2_rows(z)[normal_refs], axis=2), axis=1)
    contrast = align - normal_sim

    result = {
        "config": vars(args),
        "meta": {"dataset": args.dataset, "n": int(len(labels_np)), "normal_train": int(len(normal_idx)), "anomaly_rate": float(labels_np.mean()), "descriptor_dim": int(z.shape[1])},
        "scores": {
            "ga": topk_metrics(scores["ga"], labels_np),
            "residual_norm": topk_metrics(scores["residual_norm"], labels_np),
            "rn_explanation_error": topk_metrics(rn_err, labels_np),
            "ra_residual_alignment": topk_metrics(align, labels_np),
            "contrast_margin": topk_metrics(contrast, labels_np),
        },
        "reference": {
            "normal_ref_normal_ratio": float(np.mean(labels_np[normal_refs] == 0)),
            "anom_ref_anom_ratio": float(np.mean(labels_np[anom_refs] == 1)),
            "anom_ref_anom_ratio_on_anom_nodes": float(np.mean(labels_np[anom_refs[labels_np == 1]] == 1)) if np.any(labels_np == 1) else 0.0,
            "rn_err_mean_normal": float(np.mean(rn_err[labels_np == 0])),
            "rn_err_mean_anom": float(np.mean(rn_err[labels_np == 1])) if np.any(labels_np == 1) else 0.0,
            "align_mean_normal": float(np.mean(align[labels_np == 0])),
            "align_mean_anom": float(np.mean(align[labels_np == 1])) if np.any(labels_np == 1) else 0.0,
            "contrast_mean_normal": float(np.mean(contrast[labels_np == 0])),
            "contrast_mean_anom": float(np.mean(contrast[labels_np == 1])) if np.any(labels_np == 1) else 0.0,
        },
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_rate", type=float, default=0.05)
    ap.add_argument("--descriptor_mode", choices=["hop_attr","rwse","hop_attr_rwse"], default="hop_attr_rwse")
    ap.add_argument("--pn_estimator", choices=["diag_gaussian","pca_residual"], default="diag_gaussian")
    ap.add_argument("--gn_mode", choices=["label_gate","normal_density","label_gate_density"], default="label_gate_density")
    ap.add_argument("--ln_mode", choices=["descriptor_similarity","reconstruction_gain"], default="descriptor_similarity")
    ap.add_argument("--ga_mode", choices=["normal_rejection","residual_norm","normal_soft_or"], default="normal_rejection")
    ap.add_argument("--la_mode", choices=["residual_cosine","descriptor_similarity"], default="residual_cosine")
    ap.add_argument("--reference_mode", default="dual_reference")
    ap.add_argument("--normal_k", type=int, default=4)
    ap.add_argument("--anom_k", type=int, default=16)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--rw_steps", type=int, default=8)
    ap.add_argument("--pca_components", type=int, default=32)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    set_seed(args.seed)
    t0 = time.time()
    res = run_one(args)
    res["elapsed_sec"] = time.time() - t0
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
