#!/usr/bin/env python3
"""MatGuardGT single-run prototype for DualRefGAD response matrices.

Design intent:
- response-matrix token is a vector profile, not a scalar entry;
- mat_mean is used as label-free high-confidence pairwise ranking teacher, not as a residual base;
- known-normal nodes impose a serious low-score constraint;
- barrier and reference-dropout losses are optional ablations, not defaults.

This script is self-contained enough for HCCS-25 smoke: it uses DualRefGAD's
utils.py for dataset/split loading, then builds a deterministic label-free
response matrix from feature/hop descriptors and reference sets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    if not path:
        return
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def sha1_ints(xs) -> str:
    arr = np.asarray(xs, dtype=np.int64).reshape(-1)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def safe_spearman(a, b) -> float:
    try:
        v = spearmanr(np.asarray(a), np.asarray(b)).correlation
        return float(0.0 if np.isnan(v) else v)
    except Exception:
        return 0.0


def safe_auc_ap(labels, score, idx) -> Tuple[Any, Any]:
    idx = np.asarray(idx, dtype=np.int64)
    y = np.asarray(labels).reshape(-1).astype(int)[idx]
    s = np.asarray(score, dtype=np.float64)[idx]
    if len(np.unique(y)) < 2:
        return None, None
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def rank01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x)
    r = np.empty_like(order, dtype=np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    return r / max(1.0, float(len(x) - 1))


def metric_block(labels, idx_test, scores: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in scores.items():
        auc, ap = safe_auc_ap(labels, v, idx_test)
        out[k] = {"auc": auc, "ap": ap}
    return out


def top_jaccard(a, b, frac=0.05) -> float:
    a = np.asarray(a); b = np.asarray(b)
    k = max(1, int(len(a) * frac))
    ia = set(np.argsort(-a)[:k].tolist())
    ib = set(np.argsort(-b)[:k].tolist())
    return float(len(ia & ib) / max(1, len(ia | ib)))


def load_dualrefgad_data(project_root: Path, dataset: str, train_rate: float, val_rate: float, seed: int, sample_rate: float):
    sys.path.insert(0, str(project_root))
    os.chdir(str(project_root))
    from utils import load_mat, normalize_adj, preprocess_features  # type: ignore

    class A:
        pass
    args = A()
    args.data_split_seed = int(seed)
    args.sample_rate = float(sample_rate)
    adj, feat, labels, all_idx, idx_train, idx_val, idx_test, _, _, _, normal_idx, gen_idx = load_mat(dataset, train_rate=train_rate, val_rate=val_rate, args=args)
    if dataset in ["Amazon", "tf_finace", "t_finance", "reddit", "elliptic"]:
        feat_dense, _ = preprocess_features(feat)
        x0 = np.asarray(feat_dense, dtype=np.float32)
    else:
        x0 = np.asarray(feat.todense(), dtype=np.float32)
    adj_norm = normalize_adj(adj)
    return adj_norm, x0, np.asarray(labels).reshape(-1).astype(int), np.asarray(idx_train), np.asarray(idx_val), np.asarray(idx_test), np.asarray(normal_idx, dtype=np.int64), np.asarray(gen_idx, dtype=np.int64)


def build_descriptor(adj_norm, x0: np.ndarray, hops: int) -> np.ndarray:
    outs = [x0.astype(np.float32)]
    cur = x0.astype(np.float32)
    for _ in range(int(hops)):
        cur = np.asarray(adj_norm.dot(cur), dtype=np.float32)
        outs.append(cur)
    desc = np.concatenate(outs, axis=1).astype(np.float32)
    mean = desc.mean(axis=0, keepdims=True)
    std = desc.std(axis=0, keepdims=True) + 1e-6
    desc = (desc - mean) / std
    norm = np.linalg.norm(desc, axis=1, keepdims=True) + 1e-8
    return (desc / norm).astype(np.float32)


def choose_refs(desc: np.ndarray, normal_idx: np.ndarray, labels: np.ndarray, normal_k: int, anom_k: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed) + 17)
    normal_idx = np.asarray(normal_idx, dtype=np.int64)
    if len(normal_idx) < normal_k:
        raise ValueError(f"not enough known-normal refs: {len(normal_idx)} < {normal_k}")
    normal_refs = rng.choice(normal_idx, size=int(normal_k), replace=False)
    center = desc[normal_refs].mean(axis=0, keepdims=True)
    center = center / (np.linalg.norm(center, axis=1, keepdims=True) + 1e-8)
    teacher = 1.0 - np.squeeze(desc @ center.T)
    mask = np.ones(desc.shape[0], dtype=bool)
    mask[normal_idx] = False
    cand = np.where(mask)[0]
    # label-free pseudo anomaly refs: largest distance from the known-normal center among non-known-normal nodes
    order = cand[np.argsort(-teacher[cand])]
    anom_refs = order[:int(anom_k)]
    return normal_refs.astype(np.int64), anom_refs.astype(np.int64), teacher.astype(np.float64)


def build_response_matrix(desc: np.ndarray, normal_refs: np.ndarray, anom_refs: np.ndarray, block_size: int = 4096) -> np.ndarray:
    n = desc.shape[0]
    nr = desc[normal_refs].astype(np.float32)   # [Kn, D]
    ar = desc[anom_refs].astype(np.float32)     # [Ka, D]
    out = np.empty((n, len(anom_refs), len(normal_refs)), dtype=np.float32)
    for st in range(0, n, int(block_size)):
        xb = desc[st:st + int(block_size)].astype(np.float32)
        sim_a = xb @ ar.T  # [B, Ka]
        sim_n = xb @ nr.T  # [B, Kn]
        out[st:st + int(block_size)] = sim_a[:, :, None] - sim_n[:, None, :]
    return out


def standardize_matrix(mat: np.ndarray, normal_idx: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    ref = mat[np.asarray(normal_idx, dtype=np.int64)]
    mean = ref.mean(axis=(0, 1, 2), keepdims=True)
    std = ref.std(axis=(0, 1, 2), keepdims=True) + 1e-6
    return ((mat - mean) / std).astype(np.float32), {"mode": "global_known_normal", "mean": float(mean.squeeze()), "std": float(std.squeeze())}


class MatGuardGT(nn.Module):
    def __init__(self, normal_k: int, anom_k: int, tokenization: str, d_model: int, heads: int, layers: int, ffn_dim: int, dropout: float, pooling: str, identity: str):
        super().__init__()
        self.normal_k = int(normal_k)
        self.anom_k = int(anom_k)
        self.tokenization = tokenization
        self.pooling = pooling
        self.identity = identity
        self.row_proj = nn.Sequential(nn.Linear(self.normal_k, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.col_proj = nn.Sequential(nn.Linear(self.anom_k, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.row_embed = nn.Embedding(max(1, self.anom_k), d_model)
        self.col_embed = nn.Embedding(max(1, self.normal_k), d_model)
        self.type_embed = nn.Embedding(2, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=ffn_dim, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        if pooling == "attn":
            self.pool = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1))
        elif pooling != "mean":
            raise ValueError(pooling)
        self.norm = nn.LayerNorm(d_model)
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, ref_drop_prob: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, Ka, Kn]
        b = x.shape[0]
        toks: List[torch.Tensor] = []
        if self.tokenization in ("row", "rowcol"):
            row = self.row_proj(x.float())
            if self.identity in ("index", "index_type"):
                row_ids = torch.arange(self.anom_k, device=x.device).view(1, self.anom_k).expand(b, -1)
                row = row + self.row_embed(row_ids)
            if self.identity == "index_type":
                row = row + self.type_embed(torch.zeros((b, self.anom_k), dtype=torch.long, device=x.device))
            toks.append(row)
        if self.tokenization in ("col", "rowcol"):
            col = self.col_proj(x.transpose(1, 2).float())
            if self.identity in ("index", "index_type"):
                col_ids = torch.arange(self.normal_k, device=x.device).view(1, self.normal_k).expand(b, -1)
                col = col + self.col_embed(col_ids)
            if self.identity == "index_type":
                col = col + self.type_embed(torch.ones((b, self.normal_k), dtype=torch.long, device=x.device))
            toks.append(col)
        tok = torch.cat(toks, dim=1)
        if self.training and ref_drop_prob > 0:
            keep = (torch.rand(tok.shape[:2], device=tok.device) >= ref_drop_prob).float().unsqueeze(-1)
            tok = tok * keep / max(1e-6, 1.0 - ref_drop_prob)
        h = self.encoder(tok)
        if self.pooling == "mean":
            z = h.mean(dim=1)
        else:
            w = torch.softmax(self.pool(h).squeeze(-1), dim=1)
            z = torch.sum(h * w.unsqueeze(-1), dim=1)
        z = self.norm(z)
        s = self.score(z).squeeze(-1)
        return s, z


def variant_grid() -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = []
    for tokenization in ["row", "rowcol"]:
        for pooling in ["mean", "attn"]:
            for guide in ["pairwise", "pairwise_center"]:
                for lam_norm in [0.25, 0.5, 1.0]:
                    variants.append({"tokenization": tokenization, "pooling": pooling, "guide_mode": guide, "lambda_known_normal": lam_norm, "lambda_pairwise": 1.0, "lambda_barrier": 0.0, "ref_drop_prob": 0.0})
    # explicit sidecar ablations; not default mainline
    for base in list(variants[:8]):
        v = dict(base); v["lambda_barrier"] = 0.05; variants.append(v)
        v = dict(base); v["ref_drop_prob"] = 0.10; variants.append(v)
    return variants


def apply_variant(args: argparse.Namespace) -> argparse.Namespace:
    grid = variant_grid()
    vid = int(args.variant_id)
    if vid < 0 or vid >= len(grid):
        raise SystemExit(f"variant_id out of range: {vid}; expected 0..{len(grid)-1}")
    cfg = vars(args).copy(); cfg.update(grid[vid]); cfg["variant_definition"] = grid[vid]
    return argparse.Namespace(**cfg)


def train_matguard(mat: np.ndarray, mat_mean: np.ndarray, normal_idx: np.ndarray, labels: np.ndarray, idx_test: np.ndarray, args: argparse.Namespace, device: torch.device):
    x_all = torch.tensor(mat, dtype=torch.float32)
    teacher = np.asarray(mat_mean, dtype=np.float64)
    teacher_rank = rank01(teacher)
    n = x_all.shape[0]
    normal_idx = np.asarray(normal_idx, dtype=np.int64)
    normal_mask = np.zeros(n, dtype=bool); normal_mask[normal_idx] = True
    train_pool = np.arange(n, dtype=np.int64)
    model = MatGuardGT(mat.shape[2], mat.shape[1], args.tokenization, args.d_model, args.heads, args.layers, args.ffn_dim, args.dropout, args.pooling, args.identity).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x_dev = x_all.to(device)
    best_state = None; best_monitor = float("inf"); history = []
    rng = np.random.default_rng(int(args.seed) + 9001)
    normal_t = torch.tensor(normal_idx, dtype=torch.long, device=device)

    for epoch in range(1, int(args.num_epoch) + 1):
        model.train(); losses = []
        for _ in range(max(1, int(args.steps_per_epoch))):
            bsz = min(int(args.batch_size), n)
            batch = rng.choice(train_pool, size=bsz, replace=False if bsz < n else True)
            xb = x_dev[torch.tensor(batch, dtype=torch.long, device=device)]
            s, z = model(xb, ref_drop_prob=float(args.ref_drop_prob))
            # high-confidence pairwise distillation from mat_mean teacher
            pair_count = min(int(args.pairs_per_batch), bsz * 2)
            ia = rng.integers(0, bsz, size=pair_count)
            ib = rng.integers(0, bsz, size=pair_count)
            ga = teacher_rank[batch[ia]]; gb = teacher_rank[batch[ib]]
            keep = np.abs(ga - gb) >= float(args.pair_gap)
            if keep.any():
                ia_t = torch.tensor(ia[keep], dtype=torch.long, device=device)
                ib_t = torch.tensor(ib[keep], dtype=torch.long, device=device)
                target = torch.tensor((ga[keep] > gb[keep]).astype(np.float32), device=device)
                loss_pair = F.binary_cross_entropy_with_logits(s[ia_t] - s[ib_t], target)
            else:
                loss_pair = s.mean() * 0.0
            is_norm = torch.tensor(normal_mask[batch], dtype=torch.bool, device=device)
            if bool(is_norm.any().item()):
                # known normals should remain below a conservative score margin
                loss_norm = F.softplus(s[is_norm] - float(args.normal_score_margin)).mean()
            else:
                loss_norm = s.mean() * 0.0
            if args.guide_mode == "pairwise_center":
                with torch.no_grad():
                    s_norm, _ = model(x_dev[normal_t], ref_drop_prob=0.0)
                    c = s_norm.mean()
                loss_center = (s[is_norm].mean() - c).pow(2) if bool(is_norm.any().item()) else s.mean() * 0.0
            else:
                loss_center = s.mean() * 0.0
            if float(args.lambda_barrier) > 0:
                # weak score spread barrier; ablation only
                loss_barrier = F.relu(float(args.score_std_floor) - s.std()).pow(2)
            else:
                loss_barrier = s.mean() * 0.0
            loss = float(args.lambda_pairwise) * loss_pair + float(args.lambda_known_normal) * loss_norm + float(args.lambda_center) * loss_center + float(args.lambda_barrier) * loss_barrier
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip)); opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % int(args.eval_every) == 0 or epoch == int(args.num_epoch):
            model.eval(); chunks = []
            with torch.no_grad():
                for st in range(0, n, int(args.eval_batch_size)):
                    ss, _ = model(x_dev[st:st + int(args.eval_batch_size)], ref_drop_prob=0.0)
                    chunks.append(ss.detach().cpu())
            score = torch.cat(chunks).numpy().astype(np.float64)
            normal_q90 = float(np.quantile(score[normal_idx], 0.90))
            corr = safe_spearman(score, mat_mean)
            monitor = float(normal_q90 - 0.05 * corr)
            rec = {"epoch": int(epoch), "train_loss": float(np.mean(losses)) if losses else None, "label_free_monitor": monitor, "known_normal_score_q90": normal_q90, "spearman_score_mat_mean": corr, "score_std": float(np.std(score))}
            history.append(rec); print(json.dumps({"stage": "epoch", **rec}, ensure_ascii=False), flush=True)
            if monitor < best_monitor:
                best_monitor = monitor; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval(); chunks = []
    with torch.no_grad():
        for st in range(0, n, int(args.eval_batch_size)):
            ss, _ = model(x_dev[st:st + int(args.eval_batch_size)], ref_drop_prob=0.0)
            chunks.append(ss.detach().cpu())
    score = torch.cat(chunks).numpy().astype(np.float64)
    auc, ap = safe_auc_ap(labels, score, idx_test)
    return {"score": score, "history": history, "best_label_free_monitor": best_monitor, "diagnostic_auc": auc, "diagnostic_ap": ap}


def run_one(args: argparse.Namespace) -> Dict[str, Any]:
    random.seed(int(args.seed)); np.random.seed(int(args.seed)); torch.manual_seed(int(args.seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(args.seed))
    project_root = Path(args.project_root).expanduser().resolve()
    adj_norm, x0, labels, idx_train, idx_val, idx_test, normal_idx, gen_idx = load_dualrefgad_data(project_root, args.dataset, args.train_rate, args.val_rate, args.seed, args.sample_rate)
    desc = build_descriptor(adj_norm, x0, args.hops)
    normal_refs, anom_refs, center_teacher = choose_refs(desc, normal_idx, labels, args.normal_k, args.anom_k, args.seed)
    mat = build_response_matrix(desc, normal_refs, anom_refs, args.ref_block_size)
    mat_std, standardization = standardize_matrix(mat, normal_idx)
    mat_mean = mat_std.mean(axis=(1, 2)).astype(np.float64)
    device = torch.device(f"cuda:{int(args.device)}" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    train = train_matguard(mat_std, mat_mean, normal_idx, labels, idx_test, args, device)
    score = train.pop("score")
    metrics = metric_block(labels, idx_test, {"matguard_score": score, "mat_mean": mat_mean, "center_teacher": center_teacher})
    return {
        "status": "finished",
        "probe": "matguardgt_single_run",
        "method_family": "MatGuardGT",
        "seed": int(args.seed),
        "variant_id": int(args.variant_id),
        "effective_config": vars(args),
        "protocol": {
            "token_definition": f"{args.tokenization} vector-profile tokens over response matrix; no scalar-entry tokens",
            "mat_mean_role": "label-free high-confidence pairwise ranking teacher only; no residual-over-mat_mean",
            "known_normal_constraint": "known-normal low-score softplus constraint retained as a primary loss term",
            "diagnostic_labels": "AUC/AP use true labels after training only; not used for optimization or checkpoint selection",
        },
        "matrix_shape": list(mat_std.shape),
        "split_fingerprint": {"idx_train_sha1": sha1_ints(idx_train), "idx_val_sha1": sha1_ints(idx_val), "idx_test_sha1": sha1_ints(idx_test), "normal_idx_sha1": sha1_ints(normal_idx), "train_count": int(len(idx_train)), "test_count": int(len(idx_test)), "normal_count": int(len(normal_idx)), "train_anom_count_diagnostic": int(np.sum(labels[idx_train] == 1)), "test_anom_count_diagnostic": int(np.sum(labels[idx_test] == 1))},
        "reference_global": {"normal_refs": normal_refs.tolist(), "anom_refs": anom_refs.tolist(), "normal_ref_anom_count_diagnostic": int(np.sum(labels[normal_refs] == 1)), "anom_ref_anom_count_diagnostic": int(np.sum(labels[anom_refs] == 1))},
        "standardization": standardization,
        "train": train,
        "metrics": metrics,
        "comparisons": {"matguard_minus_mat_mean_auc": None if metrics["matguard_score"]["auc"] is None else float(metrics["matguard_score"]["auc"] - metrics["mat_mean"]["auc"]), "spearman_matguard_mat_mean": safe_spearman(score, mat_mean), "top5_jaccard_matguard_mat_mean": top_jaccard(score[idx_test], mat_mean[idx_test], 0.05)},
    }


def try_wandb_init(enabled: bool):
    if not enabled:
        return None
    try:
        import wandb  # type: ignore
        return wandb.init()
    except Exception as exc:
        print(json.dumps({"stage": "wandb_init_failed", "error": repr(exc)}), flush=True)
        return None


def merge_wandb_config(args, run):
    if run is None:
        return args
    cfg = vars(args).copy()
    for k, v in dict(run.config).items():
        if k in cfg:
            cfg[k] = v
    return argparse.Namespace(**cfg)


def log_wandb(run, payload):
    if run is None:
        return
    try:
        import wandb  # type: ignore
        m = payload["metrics"]["matguard_score"]
        summary = {"final_test_auc": m.get("auc"), "final_test_ap": m.get("ap"), "mat_mean_auc": payload["metrics"]["mat_mean"].get("auc"), "matguard_minus_mat_mean_auc": payload["comparisons"].get("matguard_minus_mat_mean_auc"), "spearman_matguard_mat_mean": payload["comparisons"].get("spearman_matguard_mat_mean"), "best_label_free_monitor": payload["train"].get("best_label_free_monitor")}
        wandb.log({k: v for k, v in summary.items() if v is not None})
        for k, v in summary.items():
            if v is not None: run.summary[k] = v
    except Exception as exc:
        print(json.dumps({"stage": "wandb_log_failed", "error": repr(exc)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(Path.home() / "DualRefGAD"))
    ap.add_argument("--dataset", default="elliptic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--wandb", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False)
    ap.add_argument("--variant_id", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--progress_out", default="")
    ap.add_argument("--train_rate", type=float, default=0.05)
    ap.add_argument("--val_rate", type=float, default=0.0)
    ap.add_argument("--sample_rate", type=float, default=0.15)
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--normal_k", type=int, default=4)
    ap.add_argument("--anom_k", type=int, default=16)
    ap.add_argument("--ref_block_size", type=int, default=4096)
    ap.add_argument("--tokenization", choices=["row", "col", "rowcol"], default="rowcol")
    ap.add_argument("--identity", choices=["none", "index", "index_type"], default="index_type")
    ap.add_argument("--d_model", type=int, default=32)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--ffn_dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--pooling", choices=["mean", "attn"], default="mean")
    ap.add_argument("--guide_mode", choices=["pairwise", "pairwise_center"], default="pairwise")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_epoch", type=int, default=20)
    ap.add_argument("--steps_per_epoch", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--eval_batch_size", type=int, default=4096)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--pairs_per_batch", type=int, default=512)
    ap.add_argument("--pair_gap", type=float, default=0.25)
    ap.add_argument("--normal_score_margin", type=float, default=-0.5)
    ap.add_argument("--lambda_pairwise", type=float, default=1.0)
    ap.add_argument("--lambda_known_normal", type=float, default=0.5)
    ap.add_argument("--lambda_center", type=float, default=0.05)
    ap.add_argument("--lambda_barrier", type=float, default=0.0)
    ap.add_argument("--score_std_floor", type=float, default=0.25)
    ap.add_argument("--ref_drop_prob", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    args = ap.parse_args()
    run = try_wandb_init(bool(args.wandb))
    args = merge_wandb_config(args, run)
    args = apply_variant(args)
    started = time.time()
    atomic_write_json(args.progress_out, {"status": "running", "started_at": started, "seed": int(args.seed), "variant_id": int(args.variant_id)})
    print(json.dumps({"stage": "matguardgt_start", "config": vars(args)}, ensure_ascii=False), flush=True)
    try:
        payload = run_one(args)
        payload["elapsed_sec"] = time.time() - started
        atomic_write_json(args.out, payload)
        atomic_write_json(args.progress_out, {"status": "finished", "out": args.out, "elapsed_sec": payload["elapsed_sec"], "auc": payload["metrics"]["matguard_score"].get("auc"), "ap": payload["metrics"]["matguard_score"].get("ap")})
        log_wandb(run, payload)
        print(json.dumps({"stage": "matguardgt_done", "auc": payload["metrics"]["matguard_score"].get("auc"), "ap": payload["metrics"]["matguard_score"].get("ap"), "out": args.out}, ensure_ascii=False), flush=True)
    except Exception as exc:
        atomic_write_json(args.progress_out, {"status": "failed", "error": repr(exc), "elapsed_sec": time.time() - started})
        raise
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
