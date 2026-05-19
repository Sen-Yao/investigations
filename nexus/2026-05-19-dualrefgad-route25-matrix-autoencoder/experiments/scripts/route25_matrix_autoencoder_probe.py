#!/usr/bin/env python3
"""Route2.5 normal-only Matrix Autoencoder probe for DualRefGAD.

Diagnostic question: does the full DualRef response matrix M(v) contain
normal-only anomaly signal beyond scalar margin / simple matrix summaries?

Protocol:
- Frozen VecGAD encoder.
- AE trains only on labeled-normal training nodes.
- Labels are used only for AUC/AP/autopsy metrics.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def l2_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def rank_percentile(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x)
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x))
    return r / max(1, len(x) - 1)


def safe_auc_ap(labels, score, idx):
    idx = np.asarray(idx, dtype=np.int64)
    return float(roc_auc_score(labels[idx], score[idx])), float(average_precision_score(labels[idx], score[idx]))


def safe_spearman(a, b):
    try:
        v = spearmanr(np.asarray(a), np.asarray(b)).correlation
        return float(0.0 if np.isnan(v) else v)
    except Exception:
        return 0.0


def top_indices(scores, frac):
    scores = np.asarray(scores)
    return np.argsort(-scores)[:max(1, int(len(scores) * frac))]


def jaccard_top(a, b, frac=0.05):
    ia = set(top_indices(a, frac).tolist())
    ib = set(top_indices(b, frac).tolist())
    return float(len(ia & ib) / max(1, len(ia | ib)))


def metric_block(labels, idx, arrays, base_name="margin"):
    labels_i = labels[idx]
    base = arrays[base_name][idx]
    out = {}
    for name, vals in arrays.items():
        v = np.asarray(vals)[idx]
        auc, ap = safe_auc_ap(labels, vals, idx)
        out[name] = {
            "auc": auc,
            "ap": ap,
            "spearman_with_margin": safe_spearman(v, base),
            "top1_jaccard_with_margin": jaccard_top(v, base, 0.01),
            "top5_jaccard_with_margin": jaccard_top(v, base, 0.05),
            "normal_mean": float(np.mean(v[labels_i == 0])) if np.any(labels_i == 0) else 0.0,
            "anom_mean": float(np.mean(v[labels_i == 1])) if np.any(labels_i == 1) else 0.0,
        }
    return out


def build_hop_attr(features, adj, normalize_adj, hops=2):
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
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    P = sp.diags(inv).dot(csr).tocsr()
    cur = P.copy()
    feats = []
    for k in range(1, steps + 1):
        feats.append(cur.diagonal().astype(np.float64))
        if k < steps:
            cur = cur.dot(P).tocsr()
    return np.stack(feats, axis=1).astype(np.float32)


def build_descriptor(mode, features, adj, normalize_adj, hops=2, rw_steps=8):
    if mode == "hop_attr":
        return build_hop_attr(features, adj, normalize_adj, hops)
    if mode == "rwse":
        return rwse(adj, rw_steps)
    if mode == "hop_attr_rwse":
        return np.concatenate([build_hop_attr(features, adj, normalize_adj, hops), rwse(adj, rw_steps)], axis=1).astype(np.float32)
    raise ValueError(mode)


class NormalModel:
    def __init__(self, estimator, z, normal_idx, pca_components=32):
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
            raise ValueError(estimator)

    def rejection(self):
        if self.estimator == "diag_gaussian" or self.pca is None:
            return np.mean(((self.z_all - self.mu) / self.std) ** 2, axis=1)
        rec = self.pca.inverse_transform(self.pca.transform(self.z_all))
        return np.mean((self.z_all - rec) ** 2, axis=1)

    def residual(self):
        if self.estimator == "diag_gaussian" or self.pca is None:
            return ((self.z_all - self.mu) / self.std).astype(np.float32)
        rec = self.pca.inverse_transform(self.pca.transform(self.z_all))
        return (self.z_all - rec).astype(np.float32)

    def density_score(self):
        return -self.rejection()


def normal_soft_or_score(features, adj, normal_idx, normalize_adj):
    hop = build_hop_attr(features, adj, normalize_adj, hops=2)
    hm = NormalModel("diag_gaussian", hop, normal_idx)
    h = rank_percentile(hm.rejection())
    st = rwse(adj, steps=8)
    sm = NormalModel("diag_gaussian", st, normal_idx)
    s = rank_percentile(sm.rejection())
    return (1.0 - (1.0 - h) * (1.0 - s)).astype(np.float32)


def cosine_rows_to_matrix(a, b, block=1024):
    an = l2_rows(a.astype(np.float32))
    bn = l2_rows(b.astype(np.float32))
    outs = []
    for st in range(0, an.shape[0], block):
        outs.append(an[st: st + block] @ bn.T)
    return np.vstack(outs)


def topk_refs_candidate_pool(a, b, score_b, candidate_idx, k, block=1024, exclude_self=True):
    an = l2_rows(a.astype(np.float32))
    bn = l2_rows(b[candidate_idx].astype(np.float32))
    score_b = np.asarray(score_b, dtype=np.float32)
    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
    refs = np.empty((a.shape[0], k), dtype=np.int64)
    for st in range(0, a.shape[0], block):
        ed = min(st + block, a.shape[0])
        scores = an[st:ed] @ bn.T
        scores += score_b[candidate_idx][None, :]
        if exclude_self:
            rows = np.arange(st, ed)
            hit = np.where(candidate_idx[None, :] == rows[:, None])
            if len(hit[0]):
                scores[hit] = -1e9
        part = np.argpartition(-scores, kth=min(k - 1, scores.shape[1] - 1), axis=1)[:, :k]
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        refs[st:ed] = candidate_idx[np.take_along_axis(part, order, axis=1)]
    return refs


def select_refs(z, residual, normal_idx, nm, features, adj, args, normalize_adj):
    n = z.shape[0]
    rejection = nm.rejection()
    density = nm.density_score()
    residual_norm = np.linalg.norm(residual, axis=1)
    if args.gn_mode == "label_gate":
        normal_pool = np.asarray(normal_idx)
        gn = np.zeros(n, dtype=np.float32)
        gn[normal_pool] = 1.0
    elif args.gn_mode == "normal_density":
        normal_pool = np.arange(n)
        gn = rank_percentile(density).astype(np.float32)
    elif args.gn_mode == "label_gate_density":
        normal_pool = np.asarray(normal_idx)
        gn = rank_percentile(density).astype(np.float32)
        mask = np.ones(n, bool)
        mask[normal_pool] = False
        gn[mask] = -1e9
    else:
        raise ValueError(args.gn_mode)

    if args.ga_mode == "normal_rejection":
        ga = rank_percentile(rejection).astype(np.float32)
    elif args.ga_mode == "residual_norm":
        ga = rank_percentile(residual_norm).astype(np.float32)
    elif args.ga_mode == "normal_soft_or":
        ga = normal_soft_or_score(features, adj, normal_idx, normalize_adj).astype(np.float32)
    else:
        raise ValueError(args.ga_mode)

    sim_n = cosine_rows_to_matrix(z, z[normal_pool], block=args.ref_block_size)
    n_scores = sim_n + gn[normal_pool][None, :]
    part_n = np.argpartition(-n_scores, kth=min(args.normal_k - 1, n_scores.shape[1] - 1), axis=1)[:, :args.normal_k]
    part_n_scores = np.take_along_axis(n_scores, part_n, axis=1)
    normal_refs = normal_pool[np.take_along_axis(part_n, np.argsort(-part_n_scores, axis=1), axis=1)]

    cand_global = np.argsort(-ga)[:min(args.anom_approx_k, n)] if args.use_approx_anom_refs else np.arange(n)
    if args.la_mode == "residual_cosine":
        anom_refs = topk_refs_candidate_pool(residual, residual, ga, cand_global, args.anom_k, block=args.ref_block_size)
    elif args.la_mode == "descriptor_similarity":
        anom_refs = topk_refs_candidate_pool(z, z, ga, cand_global, args.anom_k, block=args.ref_block_size)
    else:
        raise ValueError(args.la_mode)
    return normal_refs, anom_refs, {"ga": ga, "rejection": rejection, "residual_norm": residual_norm, "degree": np.asarray(adj.sum(axis=1)).reshape(-1)}


def build_tokens(features, normal_refs, anom_refs):
    toks = []
    for i in range(features.shape[0]):
        toks.append(np.concatenate([features[i:i + 1], features[normal_refs[i]], features[anom_refs[i]]], axis=0))
    return torch.from_numpy(np.stack(toks).astype(np.float32))


def encode_tokens_batched(model, token_tensor_cpu, device, batch_size: int):
    n = token_tensor_cpu.shape[0]
    chunks = []
    for st in range(0, n, batch_size):
        chunks.append(model.TransformerEncoder(token_tensor_cpu[st: st + batch_size].to(device)).squeeze(0).detach().cpu())
    return torch.cat(chunks, dim=0).numpy()


def response_matrix_from_embeddings(emb, normal_refs, anom_refs):
    h = emb.astype(np.float32)
    rn = h[normal_refs]
    ra = h[anom_refs]
    rn_mean = rn.mean(axis=1)
    ra_mean = ra.mean(axis=1)
    u = h - rn_mean
    d = ra_mean - rn_mean
    margin = np.sum(l2_rows(u) * l2_rows(d), axis=1)

    un = h[:, None, :] - rn
    dn = ra[:, None, :, :] - rn[:, :, None, :]
    un_n = un / (np.linalg.norm(un, axis=2, keepdims=True) + 1e-12)
    dn_n = dn / (np.linalg.norm(dn, axis=3, keepdims=True) + 1e-12)
    mat = np.einsum("nid,nijd->nij", un_n, dn_n).astype(np.float32)
    return mat, margin


class MatrixAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=32, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_ae(X, normal_idx, labels, idx_test, args, latent_dim, device):
    nidx = np.asarray(normal_idx, dtype=np.int64)
    rng = np.random.default_rng(args.seed + latent_dim * 1009)
    nidx = rng.permutation(nidx)
    val_n = max(1, int(len(nidx) * args.normal_val_frac))
    val_idx = nidx[:val_n]
    train_idx = nidx[val_n:]
    mu = X[train_idx].mean(axis=0, keepdims=True)
    std = X[train_idx].std(axis=0, keepdims=True) + 1e-6
    Xs = ((X - mu) / std).astype(np.float32)

    x_all = torch.tensor(Xs, dtype=torch.float32, device=device)
    train_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_t = torch.tensor(val_idx, dtype=torch.long, device=device)
    model = MatrixAE(X.shape[1], latent_dim, args.ae_hidden_dim, args.ae_dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.ae_lr, weight_decay=args.ae_weight_decay)
    best_state = None
    best_val = float("inf")
    last_train = None
    for epoch in range(1, args.num_epoch + 1):
        model.train()
        perm = train_t[torch.randperm(train_t.numel(), device=device)]
        losses = []
        for st in range(0, perm.numel(), args.batch_size):
            b = perm[st:st + args.batch_size]
            xb = x_all[b]
            if args.denoise_std > 0:
                xin = xb + torch.randn_like(xb) * args.denoise_std
            else:
                xin = xb
            rec = model(xin)
            loss = F.mse_loss(rec, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        last_train = float(np.mean(losses))
        if epoch % args.eval_every == 0 or epoch == args.num_epoch:
            model.eval()
            with torch.no_grad():
                rec_val = model(x_all[val_t])
                val_loss = float(F.mse_loss(rec_val, x_all[val_t]).detach().cpu())
            print(json.dumps({"latent_dim": latent_dim, "epoch": epoch, "train_loss": last_train, "val_loss": val_loss}, ensure_ascii=False), flush=True)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    scores = []
    with torch.no_grad():
        for st in range(0, x_all.shape[0], args.batch_size):
            xb = x_all[st:st + args.batch_size]
            rec = model(xb)
            scores.append(torch.mean((rec - xb) ** 2, dim=1).detach().cpu().numpy())
    score = np.concatenate(scores, axis=0)
    auc, ap = safe_auc_ap(labels, score, idx_test)
    return {
        "latent_dim": latent_dim,
        "score": score,
        "auc": auc,
        "ap": ap,
        "best_val_loss": float(best_val),
        "last_train_loss": float(last_train if last_train is not None else 0.0),
        "train_normal_count": int(len(train_idx)),
        "val_normal_count": int(len(val_idx)),
    }


def decision(best_ae_auc, scalar_best_auc, best_spearman_margin, best_degree_corr, drop_threshold=0.58):
    if best_degree_corr is not None and abs(best_degree_corr) > 0.5:
        return "DROP_OR_REPAIR_degree_correlated"
    if best_ae_auc > scalar_best_auc + 0.02:
        return "PROMOTE"
    if abs(best_ae_auc - scalar_best_auc) <= 0.02 and abs(best_spearman_margin) < 0.4:
        return "PROMOTE_CAUTION"
    if best_ae_auc < drop_threshold:
        return "DROP"
    return "INCONCLUSIVE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(Path.home() / "DualRefGAD"))
    ap.add_argument("--dataset", default="elliptic")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_rate", type=float, default=0.05)
    ap.add_argument("--val_rate", type=float, default=0.0)
    ap.add_argument("--descriptor_mode", choices=["hop_attr", "rwse", "hop_attr_rwse"], default="hop_attr_rwse")
    ap.add_argument("--pn_estimator", choices=["diag_gaussian", "pca_residual"], default="diag_gaussian")
    ap.add_argument("--gn_mode", choices=["label_gate", "normal_density", "label_gate_density"], default="label_gate_density")
    ap.add_argument("--ln_mode", default="descriptor_similarity")
    ap.add_argument("--ga_mode", choices=["normal_rejection", "residual_norm", "normal_soft_or"], default="normal_rejection")
    ap.add_argument("--la_mode", choices=["residual_cosine", "descriptor_similarity"], default="residual_cosine")
    ap.add_argument("--normal_k", type=int, default=4)
    ap.add_argument("--anom_k", type=int, default=16)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--rw_steps", type=int, default=8)
    ap.add_argument("--pca_components", type=int, default=32)
    ap.add_argument("--embedding_dim", type=int, default=256)
    ap.add_argument("--GT_ffn_dim", type=int, default=256)
    ap.add_argument("--GT_dropout", type=float, default=0.4)
    ap.add_argument("--GT_attention_dropout", type=float, default=0.4)
    ap.add_argument("--GT_num_heads", type=int, default=2)
    ap.add_argument("--GT_num_layers", type=int, default=1)
    ap.add_argument("--pp_k", type=int, default=6)
    ap.add_argument("--sample_rate", type=float, default=0.15)
    ap.add_argument("--mean", type=float, default=0.02)
    ap.add_argument("--var", type=float, default=0.01)
    ap.add_argument("--outlier_beta", type=float, default=0.3)
    ap.add_argument("--ring_R_max", type=float, default=1.0)
    ap.add_argument("--ring_R_min", type=float, default=0.3)
    ap.add_argument("--lambda_rec_tok", type=float, default=1.0)
    ap.add_argument("--lambda_rec_emb", type=float, default=0.1)
    ap.add_argument("--encode_batch_size", type=int, default=2048)
    ap.add_argument("--ref_block_size", type=int, default=1024)
    ap.add_argument("--use_approx_anom_refs", action="store_true")
    ap.add_argument("--anom_approx_k", type=int, default=1000)
    ap.add_argument("--latent_dims", default="4,8,16")
    ap.add_argument("--num_epoch", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--ae_hidden_dim", type=int, default=32)
    ap.add_argument("--ae_lr", type=float, default=1e-3)
    ap.add_argument("--ae_weight_decay", type=float, default=1e-5)
    ap.add_argument("--ae_dropout", type=float, default=0.0)
    ap.add_argument("--denoise_std", type=float, default=0.0)
    ap.add_argument("--normal_val_frac", type=float, default=0.2)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    start = time.time()
    set_seed(args.seed)
    root = Path(args.project_root).expanduser().resolve()
    sys.path.insert(0, str(root))
    os.chdir(str(root))
    from utils import load_mat, preprocess_features, normalize_adj  # noqa: E402
    from VecGAD import VecGAD  # noqa: E402

    def to_dense_features(dataset, features):
        if dataset in ["Amazon", "tf_finace", "t_finance", "reddit", "elliptic"]:
            features, _ = preprocess_features(features)
            return np.asarray(features, dtype=np.float32)
        return np.asarray(features.todense(), dtype=np.float32)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")
    print(json.dumps({"stage": "load_data", "project_root": str(root), "device": str(device)}, ensure_ascii=False), flush=True)
    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat(args.dataset, args.train_rate, args.val_rate, args=args)
    features_np = to_dense_features(args.dataset, features)
    labels_np = np.asarray(ano_label).reshape(-1).astype(int)
    normal_idx = np.asarray(normal_for_train_idx, dtype=np.int64)
    idx_test = np.asarray(idx_test, dtype=np.int64)
    assert np.sum(labels_np[normal_idx]) == 0, "Data leakage: normal_for_train_idx contains anomalies"

    print(json.dumps({"stage": "select_refs", "num_nodes": int(len(labels_np)), "normal_train": int(len(normal_idx)), "test": int(len(idx_test))}, ensure_ascii=False), flush=True)
    z = build_descriptor(args.descriptor_mode, features_np, adj, normalize_adj, args.hops, args.rw_steps)
    nm = NormalModel(args.pn_estimator, z, normal_idx, args.pca_components)
    residual = nm.residual()
    normal_refs, anom_refs, meta = select_refs(z, residual, normal_idx, nm, features_np, adj, args, normalize_adj)
    print(json.dumps({"stage": "encode", "normal_refs": list(normal_refs.shape), "anom_refs": list(anom_refs.shape), "anom_ref_anom_ratio_diagnostic": float(np.mean(labels_np[anom_refs] == 1))}, ensure_ascii=False), flush=True)

    token_tensor = build_tokens(features_np, normal_refs, anom_refs)
    model = VecGAD(features_np.shape[1], args.embedding_dim, "prelu", args).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    with torch.no_grad():
        emb = encode_tokens_batched(model, token_tensor, device, args.encode_batch_size)
    del token_tensor, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(json.dumps({"stage": "build_matrix"}, ensure_ascii=False), flush=True)
    mat, margin = response_matrix_from_embeddings(emb, normal_refs, anom_refs)
    X = mat.reshape(mat.shape[0], -1)
    arrays = {
        "margin": margin,
        "neg_margin": -margin,
        "mat_mean": mat.mean(axis=(1, 2)),
        "neg_mat_mean": -mat.mean(axis=(1, 2)),
        "mat_std": mat.std(axis=(1, 2)),
        "mat_top5_mean": np.sort(X, axis=1)[:, -5:].mean(axis=1),
        "mat_high08_ratio": (mat > 0.8).mean(axis=(1, 2)),
        "rejection": meta["rejection"],
        "residual_norm": meta["residual_norm"],
        "degree": meta["degree"],
    }
    baseline_metrics = metric_block(labels_np, idx_test, arrays, base_name="margin")
    scalar_best_name, scalar_best = max(baseline_metrics.items(), key=lambda kv: kv[1]["auc"])

    print(json.dumps({"stage": "train_ae", "scalar_best_name": scalar_best_name, "scalar_best_auc": scalar_best["auc"]}, ensure_ascii=False), flush=True)
    latent_dims = [int(x.strip()) for x in args.latent_dims.split(",") if x.strip()]
    ae_runs = []
    ae_arrays = {}
    for ld in latent_dims:
        run = train_ae(X, normal_idx, labels_np, idx_test, args, ld, device)
        ae_arrays[f"ae_mse_latent{ld}"] = run.pop("score")
        run["spearman_with_margin"] = safe_spearman(ae_arrays[f"ae_mse_latent{ld}"], margin)
        run["spearman_with_degree"] = safe_spearman(ae_arrays[f"ae_mse_latent{ld}"], meta["degree"])
        run["top5_jaccard_with_margin"] = jaccard_top(ae_arrays[f"ae_mse_latent{ld}"][idx_test], margin[idx_test], 0.05)
        print(json.dumps({"stage": "ae_result", **run}, ensure_ascii=False), flush=True)
        ae_runs.append(run)

    all_arrays = {**arrays, **ae_arrays}
    all_metrics = metric_block(labels_np, idx_test, all_arrays, base_name="margin")
    best_ae_name = max(ae_arrays.keys(), key=lambda k: all_metrics[k]["auc"])
    best_ae = all_metrics[best_ae_name]
    verdict = decision(best_ae["auc"], scalar_best["auc"], best_ae["spearman_with_margin"], best_ae.get("spearman_with_degree"))

    result = {
        "status": "finished",
        "probe": "route25_matrix_autoencoder",
        "protocol": "Frozen encoder; AE trained on labeled-normal training nodes only; labels diagnostic-only for AUC/AP/autopsy.",
        "dataset": args.dataset,
        "seed": args.seed,
        "config": vars(args),
        "counts": {
            "num_nodes": int(len(labels_np)),
            "num_test": int(len(idx_test)),
            "num_labeled_normals": int(len(normal_idx)),
            "matrix_shape": list(mat.shape),
            "flatten_dim": int(X.shape[1]),
        },
        "reference_diagnostics": {
            "anom_ref_anom_ratio_diagnostic": float(np.mean(labels_np[anom_refs] == 1)),
            "normal_ref_anom_ratio_diagnostic": float(np.mean(labels_np[normal_refs] == 1)),
        },
        "baseline_metrics": baseline_metrics,
        "ae_runs": ae_runs,
        "all_metrics": all_metrics,
        "scalar_best": {"name": scalar_best_name, **scalar_best},
        "best_ae": {"name": best_ae_name, **best_ae},
        "decision": verdict,
        "time_sec": float(time.time() - start),
    }
    print("FINAL " + json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
