#!/usr/bin/env python3
"""DualRefGAD A/C/D loss ablation probe.

A = labeled-normal low score.
C = normal-reference ranking over selected normal references using only normal-label-safe reliability targets.
D = reference entropy target + anti-hub penalty over effective attention mass.

No anomaly labels are used in the loss. Validation/test labels are report-only diagnostics.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))
from utils import load_mat, normalize_adj, preprocess_features  # noqa: E402
from VecGAD import VecGAD  # noqa: E402


def atomic_write_json(path, payload):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_dense_features(dataset, features):
    if dataset in ["Amazon", "tf_finace", "reddit", "elliptic"]:
        features, _ = preprocess_features(features)
        return np.asarray(features, dtype=np.float32)
    return np.asarray(features.todense(), dtype=np.float32)


def l2_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def rank_percentile(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x)
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x))
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
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    P = sp.diags(inv).dot(csr).tocsr()
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


def cosine_rows_to_matrix(a, b, block=1024):
    an = l2_rows(a.astype(np.float32))
    bn = l2_rows(b.astype(np.float32))
    outs = []
    for st in range(0, an.shape[0], block):
        outs.append(an[st:st + block] @ bn.T)
    return np.vstack(outs)


def select_normal_refs(z, normal_idx, nm, args):
    n = z.shape[0]
    normal_pool = np.asarray(normal_idx)
    density = rank_percentile(nm.density_score()).astype(np.float32)
    gn = np.full(n, -1e9, dtype=np.float32)
    gn[normal_pool] = density[normal_pool]
    sim_n = cosine_rows_to_matrix(z, z[normal_pool])
    scores = sim_n + gn[normal_pool][None, :]
    return normal_pool[np.argsort(-scores, axis=1)[:, :args.normal_k]].astype(np.int64)


def build_tokens(features, normal_refs):
    toks = []
    for i in range(features.shape[0]):
        toks.append(np.concatenate([features[i:i + 1], features[normal_refs[i]]], axis=0))
    return torch.from_numpy(np.stack(toks).astype(np.float32))


def encode_tokens_batched(model, token_tensor_cpu, device, batch_size):
    n = token_tensor_cpu.shape[0]
    chunks = []
    for st in range(0, n, batch_size):
        chunks.append(model.TransformerEncoder(token_tensor_cpu[st:st + batch_size].to(device, non_blocking=True)).squeeze(0))
    return torch.cat(chunks, dim=0)


def scorer(model, emb):
    return model.fc3(model.act(model.fc2(model.act(model.fc1(emb))))).squeeze(-1)


def encode_tokens_with_attention(model, token_tensor_cpu, device, batch_size):
    """Encode tokens and expose final-layer self-token attention over token positions.

    This mirrors VecGAD.TransformerEncoder but returns attention_scores so C/D losses
    can supervise effective normal-reference use without changing the model class.
    """
    chunks = []
    attn_chunks = []
    for st in range(0, token_tensor_cpu.shape[0], batch_size):
        tokens = token_tensor_cpu[st:st + batch_size].to(device, non_blocking=True)
        emb = model.token_projection(tokens)
        attention_weights = None
        for i, layer in enumerate(model.layers):
            emb, current_attention_weights = layer(emb)
            if i == len(model.layers) - 1:
                attention_weights = current_attention_weights
        emb = model.final_ln(emb)
        if attention_weights is None:
            # GT_num_layers should be >=1 in this probe. Keep a safe fallback.
            attention_scores = torch.full((emb.shape[0], emb.shape[1]), 1.0 / emb.shape[1], device=device, dtype=emb.dtype)
        else:
            attention_scores = torch.mean(attention_weights, dim=1)[:, 0, :]
        pooled = torch.bmm(attention_scores.unsqueeze(1), emb).squeeze(1).unsqueeze(0)
        chunks.append(pooled.squeeze(0))
        attn_chunks.append(attention_scores)
    return torch.cat(chunks, dim=0), torch.cat(attn_chunks, dim=0)


def entropy_from_probs(p, eps=1e-12):
    return -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(dim=1)


def safe_auc_ap(scores, labels, idx):
    try:
        return float(roc_auc_score(labels[idx], scores[idx])), float(average_precision_score(labels[idx], scores[idx]))
    except Exception:
        return None, None


def safe_spearman(a, b):
    try:
        c = spearmanr(np.asarray(a), np.asarray(b)).correlation
        return float(0.0 if np.isnan(c) else c)
    except Exception:
        return 0.0




def evaluate_snapshot(model, token_tensor, normal_idx, labels_np, idx_val, idx_test, ref_density_mean, args, device, encode_with_ref_dropout):
    model.eval()
    with torch.no_grad():
        emb_eval, attn_eval_t = encode_tokens_with_attention(model, token_tensor, device, args.encode_batch_size)
        logits_eval_t = scorer(model, emb_eval)
        emb_drop_eval = encode_with_ref_dropout()
        logits_drop_eval_t = scorer(model, emb_drop_eval)
    logits_eval = logits_eval_t.detach().cpu().numpy()
    logits_drop_eval = logits_drop_eval_t.detach().cpu().numpy()
    val_auc, val_ap = safe_auc_ap(logits_eval, labels_np, idx_val)
    test_auc, test_ap = safe_auc_ap(logits_eval, labels_np, idx_test)
    attn_eval = attn_eval_t.detach().cpu().numpy()
    return logits_eval, logits_drop_eval, attn_eval, val_auc, val_ap, test_auc, test_ap


def run_variant(base_args, variant_name, seed, device_id, progress_cb=None):
    args = argparse.Namespace(**vars(base_args))
    args.seed = int(seed)
    args.device = int(device_id)
    variant_defs = {
        "A_normal_low": {"lambda_normal_low": 1.0, "lambda_ref_rank": 0.0, "lambda_entropy": 0.0, "lambda_antihub": 0.0, "train": True, "label": "A: labeled-normal low score"},
        "A_C_ref_rank": {"lambda_normal_low": 1.0, "lambda_ref_rank": 1.0, "lambda_entropy": 0.0, "lambda_antihub": 0.0, "train": True, "label": "A + C: normal-reference ranking"},
        "A_D_entropy_antihub": {"lambda_normal_low": 1.0, "lambda_ref_rank": 0.0, "lambda_entropy": 1.0, "lambda_antihub": 1.0, "train": True, "label": "A + D: entropy target + anti-hub"},
        "A_C_D": {"lambda_normal_low": 1.0, "lambda_ref_rank": 1.0, "lambda_entropy": 1.0, "lambda_antihub": 1.0, "train": True, "label": "A + C + D"},
    }
    if variant_name not in variant_defs:
        raise ValueError(f"unknown variant {variant_name}")
    spec = variant_defs[variant_name]
    args.lambda_normal_low = spec["lambda_normal_low"]
    args.lambda_ref_rank = spec["lambda_ref_rank"]
    args.lambda_entropy = spec["lambda_entropy"]
    args.lambda_antihub = spec["lambda_antihub"]
    args.lambda_ref_drop = 0.0
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")

    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, *_rest = load_mat(args.dataset, args.train_rate, 0.1, args=args)
    normal_idx = np.asarray(_rest[-2], dtype=int)
    features_np = to_dense_features(args.dataset, features)
    labels_np = np.asarray(ano_label).reshape(-1).astype(int)
    assert np.sum(labels_np[normal_idx]) == 0, "Data leakage: normal_for_train_idx contains anomalies"

    z = build_descriptor(args.descriptor_mode, features_np, adj, args.hops, args.rw_steps)
    nm = NormalModel(args.pn_estimator, z, normal_idx, args.pca_components)
    normal_refs = select_normal_refs(z, normal_idx, nm, args)
    token_tensor = build_tokens(features_np, normal_refs)
    ref_density = rank_percentile(nm.density_score())
    ref_density_mean = ref_density[normal_refs].mean(axis=1)
    sim_selected = np.sum(l2_rows(z)[:, None, :] * l2_rows(z)[normal_refs], axis=2).astype(np.float32)
    ref_quality_np = (args.ref_rank_sim_weight * sim_selected + args.ref_rank_density_weight * ref_density[normal_refs]).astype(np.float32)
    normal_refs_t = torch.tensor(normal_refs, dtype=torch.long, device=device)
    ref_quality_t = torch.tensor(ref_quality_np, dtype=torch.float32, device=device)

    model = VecGAD(features_np.shape[1], args.embedding_dim, "prelu", args).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    normal_t = torch.tensor(normal_idx, dtype=torch.long, device=device)
    rows = []
    best = {"val_auc": -1.0, "val_ap": -1.0, "test_auc": -1.0, "test_ap": -1.0, "epoch": -1}

    def encode_with_ref_dropout():
        tok = token_tensor.clone()
        if args.ref_dropout_rate > 0:
            ref_mask = (torch.rand(tok.shape[0], tok.shape[1] - 1) < args.ref_dropout_rate)
            tok[:, 1:, :][ref_mask] = tok[:, 0:1, :].expand(-1, tok.shape[1] - 1, -1)[ref_mask]
        return encode_tokens_batched(model, tok, device, args.encode_batch_size)

    epochs = [0] if not spec["train"] else list(range(args.num_epoch + 1))
    started = time.time()
    for epoch in epochs:
        if spec["train"]:
            model.train()
            opt.zero_grad()
            emb_full, attn_full = encode_tokens_with_attention(model, token_tensor, device, args.encode_batch_size)
            logits_full = scorer(model, emb_full)
            ref_attn = attn_full[:, 1:] / (attn_full[:, 1:].sum(dim=1, keepdim=True) + 1e-12)
            l_normal_low = F.softplus(logits_full[normal_t]).mean()
            target_ref = torch.softmax(ref_quality_t[normal_t] / max(args.ref_rank_temperature, 1e-6), dim=1)
            l_ref_rank = F.kl_div((ref_attn[normal_t] + 1e-12).log(), target_ref, reduction="batchmean")
            norm_entropy = entropy_from_probs(ref_attn) / np.log(max(2, args.normal_k))
            l_entropy = ((norm_entropy - args.entropy_target) ** 2).mean()
            mass = torch.zeros(token_tensor.shape[0], device=device, dtype=ref_attn.dtype)
            mass.scatter_add_(0, normal_refs_t.reshape(-1), ref_attn.reshape(-1))
            usage = mass / float(token_tensor.shape[0])
            l_antihub = F.relu(usage - args.antihub_cap).pow(2).sum()
            l_ref_drop = torch.zeros((), device=device)
            loss = (args.lambda_normal_low * l_normal_low + args.lambda_ref_rank * l_ref_rank +
                    args.lambda_entropy * l_entropy + args.lambda_antihub * l_antihub)
            loss.backward()
            opt.step()
        else:
            with torch.no_grad():
                emb_full, attn_full = encode_tokens_with_attention(model, token_tensor, device, args.encode_batch_size)
                logits_full = scorer(model, emb_full)
                ref_attn = attn_full[:, 1:] / (attn_full[:, 1:].sum(dim=1, keepdim=True) + 1e-12)
                l_normal_low = F.softplus(logits_full[normal_t]).mean()
                emb_drop = encode_with_ref_dropout()
                logits_drop = scorer(model, emb_drop)
                l_ref_drop = F.mse_loss(logits_drop[normal_t], logits_full[normal_t])
                target_ref = torch.softmax(ref_quality_t[normal_t] / max(args.ref_rank_temperature, 1e-6), dim=1)
                l_ref_rank = F.kl_div((ref_attn[normal_t] + 1e-12).log(), target_ref, reduction="batchmean")
                norm_entropy = entropy_from_probs(ref_attn) / np.log(max(2, args.normal_k))
                l_entropy = ((norm_entropy - args.entropy_target) ** 2).mean()
                mass = torch.zeros(token_tensor.shape[0], device=device, dtype=ref_attn.dtype)
                mass.scatter_add_(0, normal_refs_t.reshape(-1), ref_attn.reshape(-1))
                usage = mass / float(token_tensor.shape[0])
                l_antihub = F.relu(usage - args.antihub_cap).pow(2).sum()
                loss = args.lambda_normal_low * l_normal_low + args.lambda_ref_rank * l_ref_rank + args.lambda_entropy * l_entropy + args.lambda_antihub * l_antihub

        if epoch % args.eval_every == 0 or epoch == args.num_epoch or not spec["train"]:
            logits_eval, logits_drop_eval, attn_eval, val_auc, val_ap, test_auc, test_ap = evaluate_snapshot(
                model, token_tensor, normal_idx, labels_np, idx_val, idx_test, ref_density_mean, args, device, encode_with_ref_dropout
            )
            if val_auc is not None and val_ap is not None and val_auc + val_ap > best["val_auc"] + best["val_ap"]:
                best = {"val_auc": val_auc, "val_ap": val_ap, "test_auc": test_auc, "test_ap": test_ap, "epoch": epoch}
            row = {
                "variant": variant_name,
                "variant_label": spec["label"],
                "seed": int(args.seed),
                "device": int(args.device),
                "epoch": int(epoch),
                "loss": float(loss.item()),
                "L_normal_low": float(l_normal_low.item()),
                "L_ref_drop": float(l_ref_drop.item()),
                "L_ref_rank": float(l_ref_rank.item()),
                "L_entropy": float(l_entropy.item()),
                "L_antihub": float(l_antihub.item()),
                "normal_score_mean": float(logits_eval[normal_idx].mean()),
                "all_score_std": float(logits_eval.std()),
                "dropout_score_mse_all": float(np.mean((logits_drop_eval - logits_eval) ** 2)),
                "dropout_score_mse_normal": float(np.mean((logits_drop_eval[normal_idx] - logits_eval[normal_idx]) ** 2)),
                "spearman_score_ref_density": safe_spearman(logits_eval, ref_density_mean),
                "ref_attn_entropy_norm_mean": float((-(attn_eval[:,1:] / (attn_eval[:,1:].sum(axis=1, keepdims=True)+1e-12)) * np.log((attn_eval[:,1:] / (attn_eval[:,1:].sum(axis=1, keepdims=True)+1e-12)) + 1e-12)).sum(axis=1).mean() / np.log(max(2, args.normal_k))),
                "ref_usage_top1_report_only": float(np.bincount(normal_refs.reshape(-1), weights=(attn_eval[:,1:] / (attn_eval[:,1:].sum(axis=1, keepdims=True)+1e-12)).reshape(-1), minlength=features_np.shape[0]).max() / features_np.shape[0]),
                "val_auc_report_only": val_auc,
                "val_ap_report_only": val_ap,
                "test_auc_report_only": test_auc,
                "test_ap_report_only": test_ap,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if progress_cb:
                progress_cb(variant_name, int(args.seed), int(args.device), int(epoch), row, best)

    return {
        "variant": variant_name,
        "variant_label": spec["label"],
        "seed": int(args.seed),
        "device": int(args.device),
        "config": vars(args),
        "loss_protocol": {
            "train_losses": ([] if not spec["train"] else [name for name, lam in [("A:L_normal-low", args.lambda_normal_low), ("C:L_ref-rank", args.lambda_ref_rank), ("D:L_entropy", args.lambda_entropy), ("D:L_antihub", args.lambda_antihub)] if lam > 0]),
            "uses_anomaly_labels_in_loss": False,
            "uses_pseudo_anomalies": False,
            "validation_test_labels": "report-only diagnostics",
        },
        "data": {"num_nodes": int(features_np.shape[0]), "num_features": int(features_np.shape[1]), "num_labeled_normal_train": int(len(normal_idx)), "token_shape": list(token_tensor.shape)},
        "reference": {"normal_k": args.normal_k, "normal_ref_normal_ratio_report_only": float(np.mean(labels_np[normal_refs] == 0)), "static_ref_usage_top1": float(np.bincount(normal_refs.reshape(-1), minlength=features_np.shape[0]).max() / normal_refs.shape[0])},
        "rows": rows,
        "best_report_only": best,
        "time_sec": time.time() - started,
    }


def parse_csv(s, cast=str):
    return [cast(x.strip()) for x in str(s).split(',') if x.strip()]


def run_task_worker(args_dict, variant, seed, device_id):
    return run_variant(argparse.Namespace(**args_dict), variant, seed, device_id, progress_cb=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="elliptic")
    ap.add_argument("--devices", default="0,1,2,3")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--variants", default="A_normal_low,A_D_entropy_antihub,A_C_ref_rank,A_C_D")
    ap.add_argument("--parallel_workers", type=int, default=4)
    ap.add_argument("--train_rate", type=float, default=0.05)
    ap.add_argument("--num_epoch", type=int, default=120)
    ap.add_argument("--descriptor_mode", choices=["hop_attr", "rwse", "hop_attr_rwse"], default="hop_attr_rwse")
    ap.add_argument("--pn_estimator", choices=["diag_gaussian", "pca_residual"], default="diag_gaussian")
    ap.add_argument("--normal_k", type=int, default=8)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--rw_steps", type=int, default=8)
    ap.add_argument("--pca_components", type=int, default=32)
    ap.add_argument("--embedding_dim", type=int, default=128)
    ap.add_argument("--GT_ffn_dim", type=int, default=128)
    ap.add_argument("--GT_dropout", type=float, default=0.2)
    ap.add_argument("--GT_attention_dropout", type=float, default=0.2)
    ap.add_argument("--GT_num_heads", type=int, default=2)
    ap.add_argument("--GT_num_layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--lambda_normal_low", type=float, default=1.0)  # accepted for validator; variant overrides
    ap.add_argument("--lambda_ref_drop", type=float, default=0.0)    # accepted for validator; unused in A/C/D variants
    ap.add_argument("--lambda_ref_rank", type=float, default=1.0)   # accepted for validator; variant overrides
    ap.add_argument("--lambda_entropy", type=float, default=1.0)    # accepted for validator; variant overrides
    ap.add_argument("--lambda_antihub", type=float, default=1.0)    # accepted for validator; variant overrides
    ap.add_argument("--ref_rank_temperature", type=float, default=0.2)
    ap.add_argument("--ref_rank_sim_weight", type=float, default=0.5)
    ap.add_argument("--ref_rank_density_weight", type=float, default=0.5)
    ap.add_argument("--entropy_target", type=float, default=0.72)
    ap.add_argument("--antihub_cap", type=float, default=0.03)
    ap.add_argument("--ref_dropout_rate", type=float, default=0.0)
    ap.add_argument("--encode_batch_size", type=int, default=2048)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress_out", required=True)
    ap.add_argument("--sample_rate", type=float, default=0.15)
    ap.add_argument("--pp_k", type=int, default=None)
    ap.add_argument("--ablation_mode", default="long_epoch_r0_r3")
    args = ap.parse_args()
    args.pp_k = args.normal_k if args.pp_k is None else args.pp_k
    assert args.pp_k == args.normal_k, "VecGAD token_decoder shape requires pp_k == normal_k for this probe"

    start = time.time()
    devices = parse_csv(args.devices, int)
    seeds = parse_csv(args.seeds, int)
    variants = parse_csv(args.variants, str)
    tasks = [(v, s, devices[i % len(devices)]) for i, (v, s) in enumerate((v, s) for s in seeds for v in variants)]
    progress = {
        "status": "running", "done": 0, "total": len(tasks), "current": "start",
        "start_time": start, "config": vars(args), "tasks": [{"variant": v, "seed": s, "device": d} for v, s, d in tasks],
        "design_note": "A/C/D ablation: keep labeled-normal low-score anchor A, test normal-reference ranking C and entropy/anti-hub D. Labels remain report-only.",
    }
    atomic_write_json(args.progress_out, progress)
    results = []
    flat_rows = []
    completed = []

    try:
        if args.parallel_workers > 1 and len(tasks) > 1:
            max_workers = min(int(args.parallel_workers), len(tasks), len(devices))
            futures = {}
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                for task_i, (variant, seed, device_id) in enumerate(tasks, 1):
                    fut = ex.submit(run_task_worker, vars(args), variant, seed, device_id)
                    futures[fut] = (task_i, variant, seed, device_id)
                for fut in as_completed(futures):
                    task_i, variant, seed, device_id = futures[fut]
                    res = fut.result()
                    results.append(res)
                    flat_rows.extend(res["rows"])
                    completed.append({"variant": variant, "seed": seed, "device": device_id, "best_report_only": res["best_report_only"]})
                    p = dict(progress)
                    p.update({
                        "done": len(completed),
                        "current": {"task_index": task_i, "variant": variant, "seed": seed, "device": device_id, "state": "task_finished"},
                        "elapsed_sec": time.time() - start,
                        "completed": completed,
                    })
                    atomic_write_json(args.progress_out, p)
        else:
            for task_i, (variant, seed, device_id) in enumerate(tasks, 1):
                def cb(v, s, d, epoch, row, best):
                    p = dict(progress)
                    p.update({
                        "done": len(completed),
                        "current": {"task_index": task_i, "variant": v, "seed": s, "device": d, "epoch": epoch},
                        "elapsed_sec": time.time() - start,
                        "latest": row,
                        "partial_best": {f"{r['variant']}|seed{r['seed']}": r.get('best_report_only') for r in completed},
                    })
                    atomic_write_json(args.progress_out, p)
                res = run_variant(args, variant, seed, device_id, cb)
                results.append(res)
                flat_rows.extend(res["rows"])
                completed.append({"variant": variant, "seed": seed, "device": device_id, "best_report_only": res["best_report_only"]})
                p = dict(progress)
                p.update({"done": len(completed), "current": "task_finished", "elapsed_sec": time.time() - start, "completed": completed})
                atomic_write_json(args.progress_out, p)

        leaderboard = sorted([
            {"variant": r["variant"], "seed": r["seed"], **r["best_report_only"]} for r in results
        ], key=lambda x: (x.get("val_auc", -1) or -1, x.get("val_ap", -1) or -1), reverse=True)
        out = {
            "status": "finished",
            "dataset": args.dataset,
            "seeds": seeds,
            "variants": variants,
            "devices": devices,
            "config": vars(args),
            "protocol": {
                "purpose": "A/C/D reference-loss ablation under normal-only supervision",
                "epoch_rationale": "keep the prior long-epoch budget while changing only the loss family: A normal-low, C reference ranking, D entropy/anti-hub.",
                "uses_anomaly_labels_in_loss": False,
                "uses_pseudo_anomalies": False,
                "validation_test_labels": "report-only diagnostics",
            },
            "results": results,
            "rows": flat_rows,
            "leaderboard_report_only": leaderboard,
            "time_sec": time.time() - start,
        }
        atomic_write_json(args.out, out)
        final = dict(progress)
        final.update({"status": "finished", "done": len(tasks), "current": "finished", "elapsed_sec": time.time() - start, "leaderboard_report_only": leaderboard})
        atomic_write_json(args.progress_out, final)
        print("FINAL", json.dumps(out, ensure_ascii=False), flush=True)
    except Exception as exc:
        fail = dict(progress)
        fail.update({"status": "failed", "current": "exception", "elapsed_sec": time.time() - start, "errors": [repr(exc)], "completed": completed})
        atomic_write_json(args.progress_out, fail)
        atomic_write_json(args.out, fail)
        raise


if __name__ == "__main__":
    main()
