#!/usr/bin/env python3
"""Profile-token row/column readout sweep for C-LEG3 response matrices.

Single WandB-run semantics:
- One process evaluates one seed and one profile-token reader configuration.
- C-LEG3 / old_exact_080_regime response-matrix construction is reused from the
  RIFT-GT R1 probe.
- Training labels are known-normal only; diagnostic anomaly labels are used only
  after label-free checkpoint selection.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
RIFT_DIR = Path("/home/openclawvm/investigations/nexus/2026-05-27-dualrefgad-rift-gt-response-manifold/experiments/scripts")
if RIFT_DIR.exists():
    sys.path.insert(0, str(RIFT_DIR))

from rift_gt_r1_pos_rocc_probe import (  # noqa: E402
    build_cleg3_matrix,
    jaccard_top,
    metric_block,
    parse_ints,
    safe_spearman,
    standardize_matrix_global,
)


def make_profile_variant_grid():
    """224 fixed profile-token reader configs for the formal 1120-run sweep.

    1120 = 5 seeds × 224 profile_variant_id values.
    The grid is intentionally reader-side only; C-LEG3 response construction,
    dataset, split protocol, and reference sizes stay fixed.
    """
    variants = []
    lambda_grid = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0, 1.5]
    for tokenization in ["row_profile", "rowcol_profile"]:
        for token_identity in ["none", "index_type"]:
            for d_model in [16, 32, 48, 64]:
                for pooling in ["mean", "attn"]:
                    for lambda_u in lambda_grid:
                        variants.append({
                            "tokenization": tokenization,
                            "token_identity": token_identity,
                            "d_model": d_model,
                            "heads": 2 if d_model < 64 else 4,
                            "layers": 1,
                            "ffn_dim": d_model * 2,
                            "pooling": pooling,
                            "lambda_trimmed_unlabeled": lambda_u,
                        })
    assert len(variants) == 224, len(variants)
    return variants


PROFILE_VARIANTS = make_profile_variant_grid()


def apply_profile_variant(args: argparse.Namespace) -> argparse.Namespace:
    vid = int(args.profile_variant_id)
    if vid < 0 or vid >= len(PROFILE_VARIANTS):
        raise SystemExit(f"profile_variant_id out of range: {vid}; expected 0..{len(PROFILE_VARIANTS)-1}")
    cfg = vars(args).copy()
    cfg.update(PROFILE_VARIANTS[vid])
    cfg["profile_variant_id"] = vid
    cfg["profile_variant_definition"] = PROFILE_VARIANTS[vid]
    return argparse.Namespace(**cfg)


def try_wandb_init(enabled: bool):
    if not enabled:
        return None
    try:
        import wandb  # type: ignore
        return wandb.init()
    except Exception as exc:  # wandb missing/offline should not hide local smoke errors
        print(json.dumps({"stage": "wandb_init_failed", "error": repr(exc)}), flush=True)
        return None


def merge_wandb_config(args: argparse.Namespace, run) -> argparse.Namespace:
    if run is None:
        return args
    cfg = vars(args).copy()
    for k, v in dict(run.config).items():
        if k in cfg:
            cfg[k] = v
    return argparse.Namespace(**cfg)


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    if not path:
        return
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def safe_auc_ap(labels, score, idx):
    idx = np.asarray(idx, dtype=np.int64)
    y = np.asarray(labels).reshape(-1).astype(int)[idx]
    s = np.asarray(score, dtype=np.float64)[idx]
    if len(np.unique(y)) < 2:
        return None, None
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def safe_corr(a, b):
    try:
        v = spearmanr(np.asarray(a), np.asarray(b)).correlation
        return float(0.0 if np.isnan(v) else v)
    except Exception:
        return 0.0


def effective_rank_np(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        return 0.0
    xc = x - x.mean(axis=0, keepdims=True)
    cov = (xc.T @ xc) / max(1, x.shape[0] - 1)
    eig = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    total = float(eig.sum())
    if total <= 1e-12:
        return 0.0
    p = eig / total
    ent = -float(np.sum(p * np.log(p + 1e-12)))
    return float(np.exp(ent))


class ProfileTokenReader(nn.Module):
    """Vector-token reader for response profiles.

    Tokenization modes:
    - row_profile: each normal-reference row is one token with raw dim K_a.
    - col_profile: each anomaly-reference column is one token with raw dim K_n.
    - rowcol_profile: concatenate row tokens and column tokens after separate input projections.

    This intentionally fixes the scalar-token philosophical bug: each token is a
    response profile vector, not a single response entry.
    """

    def __init__(self, tokenization: str, normal_k: int, anom_k: int, d_model: int, heads: int, layers: int,
                 ffn_dim: int, dropout: float, pooling: str, token_identity: str):
        super().__init__()
        self.tokenization = tokenization
        self.normal_k = int(normal_k)
        self.anom_k = int(anom_k)
        self.pooling = pooling
        self.token_identity = token_identity
        self.row_proj = nn.Sequential(nn.Linear(self.anom_k, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.col_proj = nn.Sequential(nn.Linear(self.normal_k, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.row_embed = nn.Embedding(max(1, self.normal_k), d_model)
        self.col_embed = nn.Embedding(max(1, self.anom_k), d_model)
        self.type_embed = nn.Embedding(2, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.out_norm = nn.LayerNorm(d_model)
        if pooling == "attn":
            self.attn_pool = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1))
        elif pooling != "mean":
            raise ValueError(f"unknown pooling={pooling}")

    def forward(self, x, token_drop_prob: float = 0.0):
        # x: [B, K_n, K_a]
        b = x.shape[0]
        toks = []
        if self.tokenization in ("row_profile", "rowcol_profile"):
            row_tok = self.row_proj(x.float())
            if self.token_identity in ("index", "index_type"):
                row_ids = torch.arange(self.normal_k, device=x.device).view(1, self.normal_k).expand(b, -1)
                row_tok = row_tok + self.row_embed(row_ids)
            if self.token_identity == "index_type":
                row_tok = row_tok + self.type_embed(torch.zeros((b, self.normal_k), dtype=torch.long, device=x.device))
            toks.append(row_tok)
        if self.tokenization in ("col_profile", "rowcol_profile"):
            col_x = x.transpose(1, 2).float()
            col_tok = self.col_proj(col_x)
            if self.token_identity in ("index", "index_type"):
                col_ids = torch.arange(self.anom_k, device=x.device).view(1, self.anom_k).expand(b, -1)
                col_tok = col_tok + self.col_embed(col_ids)
            if self.token_identity == "index_type":
                col_tok = col_tok + self.type_embed(torch.ones((b, self.anom_k), dtype=torch.long, device=x.device))
            toks.append(col_tok)
        if not toks:
            raise ValueError(f"unsupported tokenization={self.tokenization}")
        tok = torch.cat(toks, dim=1)
        if self.training and token_drop_prob > 0:
            keep = (torch.rand(tok.shape[:2], device=tok.device) >= token_drop_prob).float().unsqueeze(-1)
            tok = tok * keep / max(1e-6, 1.0 - token_drop_prob)
        h = self.encoder(tok)
        if self.pooling == "mean":
            z = h.mean(dim=1)
        else:
            w = torch.softmax(self.attn_pool(h).squeeze(-1), dim=1)
            z = torch.sum(h * w.unsqueeze(-1), dim=1)
        return self.out_norm(z)


def compute_scores(model, x_all, normal_idx, batch_size, device, center=None, token_drop_prob: float = 0.0):
    model.eval()
    z_chunks = []
    with torch.no_grad():
        for st in range(0, x_all.shape[0], batch_size):
            xb = x_all[st:st + batch_size].to(device)
            z_chunks.append(model(xb, token_drop_prob=token_drop_prob).detach().cpu())
    z = torch.cat(z_chunks, dim=0)
    if center is None:
        center = z[np.asarray(normal_idx, dtype=np.int64)].mean(dim=0, keepdim=True)
    energy = torch.sum((z - center) ** 2, dim=1).numpy()
    return z.numpy(), center.numpy(), energy.astype(np.float64)


def rocc_train(mat_std, normal_idx, labels_np, idx_test, args, device):
    x_all = torch.tensor(mat_std, dtype=torch.float32)
    n = x_all.shape[0]
    normal_idx = np.asarray(normal_idx, dtype=np.int64)
    normal_mask = np.zeros(n, dtype=bool)
    normal_mask[normal_idx] = True
    unlabeled_idx = np.where(~normal_mask)[0].astype(np.int64)
    if len(unlabeled_idx) == 0:
        raise ValueError("No unlabeled nodes available for ROCC trimming")

    model = ProfileTokenReader(
        tokenization=args.tokenization,
        normal_k=mat_std.shape[1],
        anom_k=mat_std.shape[2],
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        pooling=args.pooling,
        token_identity=args.token_identity,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x_all_dev = x_all.to(device)
    normal_t = torch.tensor(normal_idx, dtype=torch.long, device=device)
    best_state = None
    best_monitor = float("inf")
    prev_center_np = None
    prev_trim = None
    history = []

    for epoch in range(1, args.num_epoch + 1):
        model.train()
        with torch.no_grad():
            _, _, energy = compute_scores(model, x_all, normal_idx, args.eval_batch_size, device)
            k_trim = max(1, int(len(unlabeled_idx) * args.trim_fraction))
            trim_idx_np = unlabeled_idx[np.argsort(energy[unlabeled_idx])[:k_trim]]
        selected_np = np.concatenate([normal_idx, trim_idx_np])
        rng = np.random.default_rng(int(args.seed) * 1000003 + epoch)
        selected_np = rng.permutation(selected_np)
        losses = []
        for st in range(0, len(selected_np), args.batch_size):
            batch_np = selected_np[st:st + args.batch_size]
            batch_t = torch.tensor(batch_np, dtype=torch.long, device=device)
            xb = x_all_dev[batch_t]
            z_batch = model(xb, token_drop_prob=args.train_token_drop)
            with torch.no_grad():
                z_norm = model(x_all_dev[normal_t], token_drop_prob=0.0)
                c_det = z_norm.mean(dim=0, keepdim=True)
            e = torch.sum((z_batch - c_det) ** 2, dim=1)
            is_norm = torch.tensor(normal_mask[batch_np], dtype=torch.bool, device=device)
            loss_norm = e[is_norm].mean() if bool(is_norm.any().item()) else e.mean() * 0.0
            loss_unlab = e[~is_norm].mean() if bool((~is_norm).any().item()) else e.mean() * 0.0
            std = z_batch.std(dim=0)
            collapse_penalty = F.relu(args.var_floor - std).mean()
            loss = loss_norm + args.lambda_trimmed_unlabeled * loss_unlab + args.lambda_var * collapse_penalty
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        if epoch % args.eval_every == 0 or epoch == args.num_epoch or epoch == 1:
            z_np, center_np, energy = compute_scores(model, x_all, normal_idx, args.eval_batch_size, device)
            k_trim = max(1, int(len(unlabeled_idx) * args.trim_fraction))
            trim_idx = unlabeled_idx[np.argsort(energy[unlabeled_idx])[:k_trim]]
            center_drift = None if prev_center_np is None else float(np.linalg.norm(center_np - prev_center_np) / (np.linalg.norm(prev_center_np) + 1e-12))
            trim_jaccard = None
            if prev_trim is not None:
                a, b = set(map(int, trim_idx)), set(map(int, prev_trim))
                trim_jaccard = float(len(a & b) / max(1, len(a | b)))
            normal_energy = energy[normal_idx]
            z_sel = z_np[np.concatenate([normal_idx, trim_idx])]
            collapse_penalty_np = float(np.maximum(args.var_floor - z_sel.std(axis=0), 0.0).mean())
            monitor = float(normal_energy.mean() + args.lambda_trimmed_unlabeled * energy[trim_idx].mean() + args.lambda_var * collapse_penalty_np)
            rec = {
                "epoch": int(epoch),
                "train_loss": float(np.mean(losses)) if losses else None,
                "label_free_monitor": monitor,
                "center_drift": center_drift,
                "trimmed_unlabeled_jaccard": trim_jaccard,
                "known_normal_energy_mean": float(normal_energy.mean()),
                "known_normal_energy_q90": float(np.quantile(normal_energy, 0.90)),
                "trimmed_unlabeled_energy_mean": float(energy[trim_idx].mean()),
                "collapse_effective_rank": effective_rank_np(z_sel),
                "embedding_var_mean": float(np.var(z_sel, axis=0).mean()),
                "trimmed_unlabeled_count": int(len(trim_idx)),
            }
            history.append(rec)
            print(json.dumps({"stage": "epoch", **rec}, ensure_ascii=False), flush=True)
            if monitor < best_monitor:
                best_monitor = monitor
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            prev_center_np = center_np.copy()
            prev_trim = trim_idx.copy()

    if best_state is not None:
        model.load_state_dict(best_state)
    z_np, center_np, energy = compute_scores(model, x_all, normal_idx, args.eval_batch_size, device)
    auc, ap = safe_auc_ap(labels_np, energy, idx_test)
    return {
        "score": energy,
        "embedding_shape": list(z_np.shape),
        "center_shape": list(center_np.shape),
        "history": history,
        "best_label_free_monitor": best_monitor,
        "diagnostic_auc": auc,
        "diagnostic_ap": ap,
        "final_no_leakage": history[-1] if history else {},
    }


def run_one(args):
    random.seed(int(args.seed)); np.random.seed(int(args.seed)); torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device_idx = int(args.device)
    device_obj = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() and device_idx >= 0 else "cpu")
    pack = build_cleg3_matrix(args, int(args.seed), device_idx)
    mat_std, standardization = standardize_matrix_global(pack["mat"], pack["normal_idx"])
    train = rocc_train(mat_std, pack["normal_idx"], pack["labels_np"], pack["idx_test"], args, device_obj)
    profile_score = train.pop("score")
    profile_metrics = metric_block(pack["labels_np"], pack["idx_test"], {
        "profile_rocc_energy": profile_score,
        "margin": pack["margin"],
        "mat_mean": pack["mat_mean"],
    }, base_name="margin")
    return {
        "status": "finished",
        "probe": "profile_token_rocc_single_run",
        "method_family": "Profile-token response-matrix reader",
        "seed": int(args.seed),
        "effective_config": vars(args),
        "protocol": {
            "cleg3_regime": "old_exact_080_regime",
            "token_definition": f"{args.tokenization}: vector response-profile tokens; no scalar-entry tokenization",
            "training_labels": "known-normal nodes only; unlabeled nodes enter only trimmed majority-normal compactness",
            "selection_rule": "best checkpoint by label-free ROCC monitor; diagnostic AUC/AP never used for training selection",
            "diagnostic_labels": "AUC/AP, Spearman/top-K overlap, and reference purity computed after training only",
        },
        "matrix_shape": list(pack["mat"].shape),
        "split_fingerprint": pack["split_fingerprint"],
        "standardization": standardization,
        "baseline_metrics": pack["baseline_metrics"],
        "reference_global": pack["reference_global"],
        "profile_train": train,
        "profile_metrics": profile_metrics,
        "comparisons": {
            "profile_minus_mat_mean_auc": None if profile_metrics["profile_rocc_energy"]["auc"] is None else float(profile_metrics["profile_rocc_energy"]["auc"] - pack["baseline_metrics"]["mat_mean"]["auc"]),
            "profile_minus_margin_auc": None if profile_metrics["profile_rocc_energy"]["auc"] is None else float(profile_metrics["profile_rocc_energy"]["auc"] - pack["baseline_metrics"]["margin"]["auc"]),
            "spearman_profile_with_mat_mean": safe_spearman(profile_score, pack["mat_mean"]),
            "spearman_profile_with_margin": safe_spearman(profile_score, pack["margin"]),
            "top5_jaccard_profile_with_mat_mean": jaccard_top(profile_score[pack["idx_test"]], pack["mat_mean"][pack["idx_test"]], 0.05),
        },
    }


def log_wandb(run, payload):
    if run is None:
        return
    try:
        import wandb  # type: ignore
        metrics = payload["profile_metrics"]["profile_rocc_energy"]
        summary = {
            "final_test_auc": metrics.get("auc"),
            "final_test_ap": metrics.get("ap"),
            "mat_mean_auc": payload["baseline_metrics"]["mat_mean"].get("auc"),
            "margin_auc": payload["baseline_metrics"]["margin"].get("auc"),
            "profile_minus_mat_mean_auc": payload["comparisons"].get("profile_minus_mat_mean_auc"),
            "spearman_profile_with_mat_mean": payload["comparisons"].get("spearman_profile_with_mat_mean"),
            "best_label_free_monitor": payload["profile_train"].get("best_label_free_monitor"),
        }
        wandb.log({k: v for k, v in summary.items() if v is not None})
        for k, v in summary.items():
            if v is not None:
                run.summary[k] = v
    except Exception as exc:
        print(json.dumps({"stage": "wandb_log_failed", "error": repr(exc)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(Path.home() / "DualRefGAD"))
    ap.add_argument("--dataset", default="elliptic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--wandb", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False)
    ap.add_argument("--profile_variant_id", type=int, default=0)
    ap.add_argument("--out", default="")
    # C-LEG3 fixed response-matrix construction knobs.
    ap.add_argument("--train_rate", type=float, default=0.05)
    ap.add_argument("--val_rate", type=float, default=0.0)
    ap.add_argument("--ln_mode", default="descriptor_similarity")
    ap.add_argument("--normal_k", type=int, default=4)
    ap.add_argument("--anom_k", type=int, default=16)
    ap.add_argument("--encode_batch_size", type=int, default=1024)
    ap.add_argument("--ref_block_size", type=int, default=1024)
    ap.add_argument("--top_k", type=int, default=96)
    ap.add_argument("--n_bins", type=int, default=3)
    ap.add_argument("--pca_dims", default="2,4,8,12,24")
    # Profile-token reader / ROCC knobs.
    ap.add_argument("--tokenization", choices=["row_profile", "col_profile", "rowcol_profile"], default="row_profile")
    ap.add_argument("--token_identity", choices=["none", "index", "index_type"], default="index_type")
    ap.add_argument("--d_model", type=int, default=32)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--ffn_dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--pooling", choices=["mean", "attn"], default="mean")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_epoch", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--eval_batch_size", type=int, default=2048)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--trim_fraction", type=float, default=0.70)
    ap.add_argument("--lambda_trimmed_unlabeled", type=float, default=0.25)
    ap.add_argument("--lambda_var", type=float, default=0.05)
    ap.add_argument("--var_floor", type=float, default=0.05)
    ap.add_argument("--train_token_drop", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    args = ap.parse_args()

    run = try_wandb_init(bool(args.wandb))
    args = merge_wandb_config(args, run)
    args = apply_profile_variant(args)
    started = time.time()
    print(json.dumps({"stage": "profile_token_start", "config": vars(args)}, ensure_ascii=False), flush=True)
    payload = run_one(args)
    payload["elapsed_sec"] = time.time() - started
    atomic_write_json(args.out, payload)
    log_wandb(run, payload)
    print(json.dumps({"stage": "profile_token_done", "auc": payload["profile_metrics"]["profile_rocc_energy"].get("auc"), "out": args.out}, ensure_ascii=False), flush=True)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
