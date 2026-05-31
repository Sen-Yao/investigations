#!/usr/bin/env python3
"""RIFT-GT R0 no-position ROCC probe for C-LEG3 response matrices.

Protocol:
- Freeze C-LEG3 / old_exact_080_regime response matrix construction.
- Tokenize each response-matrix entry as a scalar value token only: no row/column/rank/role PE.
- Train a small Set/Transformer reader with ROCC (robust one-class compactness):
  known-normal compactness + trimmed majority-unlabeled compactness + light anti-collapse regularizer.
- Use only known-normal labels and unlabeled majority-normal assumption for training/selection.
- AUC/AP and true anomaly labels are diagnostic-only and computed after training.
"""
import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import queue
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
# Same fixed C-LEG3 source used by the strict reproduction audit.
from route25_leg3_response_matrix_decomposition_probe import (  # noqa: E402
    BASE_DEFAULTS,
    VARIANTS as CLEG_VARIANTS,
    parse_ints,
)
from route25_matrix_autoencoder_probe import (  # noqa: E402
    NormalModel,
    build_descriptor,
    build_tokens,
    encode_tokens_batched,
    jaccard_top,
    metric_block,
    response_matrix_from_embeddings,
    safe_spearman,
    select_refs,
    set_seed,
)

RIFT_VARIANTS = {
    "r0_mean_lr1e3": {
        "report_codename": "RIFT-R0-mean-lr1e3",
        "definition": "No-PE scalar-entry Set/Transformer reader, mean pooling, ROCC, AdamW lr=1e-3.",
        "changes": {"pooling": "mean", "rift_lr": 1e-3, "rift_weight_decay": 1e-4},
    },
    "r0_mean_lr3e4": {
        "report_codename": "RIFT-R0-mean-lr3e4",
        "definition": "No-PE scalar-entry Set/Transformer reader, mean pooling, ROCC, lower AdamW lr=3e-4.",
        "changes": {"pooling": "mean", "rift_lr": 3e-4, "rift_weight_decay": 1e-4},
    },
    "r0_attn_lr1e3": {
        "report_codename": "RIFT-R0-attn-lr1e3",
        "definition": "No-PE scalar-entry Set/Transformer reader, attention pooling without positional identity, ROCC, AdamW lr=1e-3.",
        "changes": {"pooling": "attn", "rift_lr": 1e-3, "rift_weight_decay": 1e-4},
    },
}


def atomic_write_json(path, payload):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    x = np.asarray(vals, dtype=np.float64)
    return {"mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()), "max": float(x.max())}


def sha1_ints(xs):
    arr = np.asarray(xs, dtype=np.int64).reshape(-1)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def split_fingerprint(labels_np, idx_train, idx_val, idx_test, normal_idx):
    idx_train = np.asarray(idx_train, dtype=np.int64)
    idx_val = np.asarray(idx_val, dtype=np.int64)
    idx_test = np.asarray(idx_test, dtype=np.int64)
    normal_idx = np.asarray(normal_idx, dtype=np.int64)
    return {
        "idx_train_sha1": sha1_ints(idx_train),
        "idx_val_sha1": sha1_ints(idx_val),
        "idx_test_sha1": sha1_ints(idx_test),
        "normal_for_train_sha1": sha1_ints(normal_idx),
        "train_count": int(len(idx_train)),
        "val_count": int(len(idx_val)),
        "test_count": int(len(idx_test)),
        "train_anom_count_diagnostic": int(np.sum(labels_np[idx_train] == 1)),
        "test_anom_count_diagnostic": int(np.sum(labels_np[idx_test] == 1)),
        "normal_for_train_count": int(len(normal_idx)),
        "normal_for_train_anom_count": int(np.sum(labels_np[normal_idx] == 1)),
    }


def to_dense_features(dataset, features, preprocess_features):
    if dataset in ["Amazon", "tf_finace", "t_finance", "reddit", "elliptic"]:
        features, _ = preprocess_features(features)
        return np.asarray(features, dtype=np.float32)
    return np.asarray(features.todense(), dtype=np.float32)


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
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0.0, None)
    total = float(eig.sum())
    if total <= 1e-12:
        return 0.0
    p = eig / total
    ent = -float(np.sum(p * np.log(p + 1e-12)))
    return float(np.exp(ent))


class ResponseSetReader(nn.Module):
    """No-position response-entry reader.

    The only input is scalar response value per matrix entry. Row/column identity is
    deliberately unavailable in R0; self-attention is permutation-equivariant and
    mean/attention pooling is permutation-invariant up to content scores.
    """

    def __init__(self, d_model=64, nhead=4, num_layers=1, ffn_dim=128, dropout=0.1, pooling="mean"):
        super().__init__()
        self.pooling = pooling
        self.value_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)
        if pooling == "attn":
            self.attn_pool = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1))
        elif pooling != "mean":
            raise ValueError(pooling)

    def forward(self, x, token_drop_prob=0.0):
        # x: [B, T] or [B, K_n, K_a]
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        tok = self.value_embed(x.unsqueeze(-1).float())
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


def standardize_matrix_global(mat, normal_idx):
    # Global scalar standardization, not entry-wise, to avoid introducing row/col identity in R0.
    vals = np.asarray(mat[normal_idx], dtype=np.float32).reshape(-1)
    mu = float(vals.mean())
    std = float(vals.std() + 1e-6)
    return ((mat.astype(np.float32) - mu) / std).astype(np.float32), {"global_mu": mu, "global_std": std}


def compute_scores(model, x_all, normal_idx, batch_size, device, center=None, token_drop_prob=0.0):
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
    x_all = torch.tensor(mat_std.reshape(mat_std.shape[0], -1), dtype=torch.float32)
    n = x_all.shape[0]
    normal_idx = np.asarray(normal_idx, dtype=np.int64)
    normal_mask = np.zeros(n, dtype=bool)
    normal_mask[normal_idx] = True
    unlabeled_idx = np.where(~normal_mask)[0].astype(np.int64)
    if len(unlabeled_idx) == 0:
        raise ValueError("No unlabeled nodes available for ROCC trimming")

    model = ResponseSetReader(
        d_model=args.rift_dim,
        nhead=args.rift_heads,
        num_layers=args.rift_layers,
        ffn_dim=args.rift_ffn_dim,
        dropout=args.rift_dropout,
        pooling=args.pooling,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.rift_lr, weight_decay=args.rift_weight_decay)
    normal_t = torch.tensor(normal_idx, dtype=torch.long, device=device)
    # x_all_dev is only used for indexed mini-batches / known-normal batches.
    # Full-dataset reader passes are evaluated in batches via compute_scores();
    # doing `model(x_all_dev)` OOMs on 11GB GPUs with parallel workers.
    x_all_dev = x_all.to(device)
    prev_center_np = None
    prev_trim = None
    best_state = None
    best_monitor = float("inf")
    history = []

    for epoch in range(1, args.num_epoch + 1):
        model.train()
        with torch.no_grad():
            # Full-dataset ROCC trimming is inference-only and must be batched;
            # parallel workers on 11GB GPUs cannot hold Transformer activations
            # for all nodes at once.
            _, _, energy = compute_scores(model, x_all, normal_idx, args.eval_batch_size, device)
            k_trim = max(1, int(len(unlabeled_idx) * args.trim_fraction))
            trim_idx_np = unlabeled_idx[np.argsort(energy[unlabeled_idx])[:k_trim]]
        selected_np = np.concatenate([normal_idx, trim_idx_np])
        rng = np.random.default_rng(args.seed * 1000003 + epoch)
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
            center_drift = None
            if prev_center_np is not None:
                center_drift = float(np.linalg.norm(center_np - prev_center_np) / (np.linalg.norm(prev_center_np) + 1e-12))
            trim_jaccard = None
            if prev_trim is not None:
                a, b = set(map(int, trim_idx)), set(map(int, prev_trim))
                trim_jaccard = float(len(a & b) / max(1, len(a | b)))
            rsc = None
            if args.rsc_token_drop > 0:
                _, c1, e1 = compute_scores(model, x_all, normal_idx, args.eval_batch_size, device, token_drop_prob=args.rsc_token_drop)
                _, c2, e2 = compute_scores(model, x_all, normal_idx, args.eval_batch_size, device, token_drop_prob=args.rsc_token_drop)
                rsc = safe_corr(e1[unlabeled_idx], e2[unlabeled_idx])
            normal_energy = energy[normal_idx]
            unlabeled_energy = energy[unlabeled_idx]
            z_sel = z_np[np.concatenate([normal_idx, trim_idx])]
            eff_rank = effective_rank_np(z_sel)
            z_var_mean = float(np.var(z_sel, axis=0).mean())
            high_thr = float(np.quantile(normal_energy, 0.95))
            tail_mass = float(np.mean(unlabeled_energy > high_thr))
            collapse_penalty_np = float(np.maximum(args.var_floor - z_sel.std(axis=0), 0.0).mean())
            monitor = float(normal_energy.mean() + args.lambda_trimmed_unlabeled * energy[trim_idx].mean() + args.lambda_var * collapse_penalty_np)
            rec = {
                "epoch": int(epoch),
                "train_loss": float(np.mean(losses)) if losses else None,
                "label_free_monitor": monitor,
                "center_drift": center_drift,
                "trimmed_unlabeled_jaccard": trim_jaccard,
                "rsc_token_dropout_spearman": rsc,
                "known_normal_energy_mean": float(normal_energy.mean()),
                "known_normal_energy_q90": float(np.quantile(normal_energy, 0.90)),
                "trimmed_unlabeled_energy_mean": float(energy[trim_idx].mean()),
                "collapse_effective_rank": eff_rank,
                "embedding_var_mean": z_var_mean,
                "score_tail_mass_vs_normal_q95": tail_mass,
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
        "embedding": z_np,
        "center": center_np.reshape(-1),
        "history": history,
        "best_label_free_monitor": best_monitor,
        "diagnostic_auc": auc,
        "diagnostic_ap": ap,
        "final_no_leakage": history[-1] if history else {},
    }


def build_cleg3_matrix(cli_args, seed, device):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = copy.deepcopy(BASE_DEFAULTS)
    cfg.update(CLEG_VARIANTS["old_exact_080_regime"]["changes"])
    cfg.update(vars(cli_args))
    cfg["variant"] = "old_exact_080_regime"
    cfg["device"] = int(device)
    cfg["seed"] = int(seed)
    cfg["data_split_seed"] = int(seed)
    cfg["strict_sequential"] = True
    v_args = argparse.Namespace(**cfg)

    root = Path(v_args.project_root).expanduser().resolve()
    sys.path.insert(0, str(root))
    os.chdir(str(root))
    from utils import load_mat, preprocess_features, normalize_adj  # noqa: E402
    from VecGAD import VecGAD  # noqa: E402

    device_obj = torch.device(f"cuda:{device}" if torch.cuda.is_available() and int(device) >= 0 else "cpu")
    print(json.dumps({"stage": "cleg3_seed_start", "seed": seed, "device": str(device_obj)}, ensure_ascii=False), flush=True)
    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat(v_args.dataset, v_args.train_rate, v_args.val_rate, args=v_args)
    features_np = to_dense_features(v_args.dataset, features, preprocess_features)
    labels_np = np.asarray(ano_label).reshape(-1).astype(int)
    normal_idx = np.asarray(normal_for_train_idx, dtype=np.int64)
    idx_test = np.asarray(idx_test, dtype=np.int64)
    fp = split_fingerprint(labels_np, idx_train, idx_val, idx_test, normal_idx)
    assert fp["normal_for_train_anom_count"] == 0, "Data leakage: normal_for_train_idx contains anomalies"

    z = build_descriptor(v_args.descriptor_mode, features_np, adj, normalize_adj, v_args.hops, v_args.rw_steps)
    nm = NormalModel(v_args.pn_estimator, z, normal_idx, v_args.pca_components)
    residual = nm.residual()
    normal_refs, anom_refs, meta = select_refs(z, residual, normal_idx, nm, features_np, adj, v_args, normalize_adj)
    token_tensor = build_tokens(features_np, normal_refs, anom_refs)
    model = VecGAD(features_np.shape[1], v_args.embedding_dim, "prelu", v_args).to(device_obj)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    with torch.no_grad():
        emb = encode_tokens_batched(model, token_tensor, device_obj, v_args.encode_batch_size)
    del token_tensor, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mat, margin = response_matrix_from_embeddings(emb, normal_refs, anom_refs)
    mat_mean = mat.mean(axis=(1, 2))
    baseline_metrics = metric_block(labels_np, idx_test, {"margin": margin, "mat_mean": mat_mean}, base_name="margin")
    return {
        "mat": mat,
        "margin": margin,
        "mat_mean": mat_mean,
        "labels_np": labels_np,
        "idx_test": idx_test,
        "normal_idx": normal_idx,
        "split_fingerprint": fp,
        "baseline_metrics": baseline_metrics,
        "reference_global": {
            "normal_ref_anom_ratio_diagnostic": float(np.mean(labels_np[normal_refs] == 1)),
            "anom_ref_anom_ratio_diagnostic": float(np.mean(labels_np[anom_refs] == 1)),
            "normal_ref_dup_rate": float(np.mean([1.0 - len(set(map(int, r))) / max(1, len(r)) for r in normal_refs])),
            "anom_ref_dup_rate": float(np.mean([1.0 - len(set(map(int, r))) / max(1, len(r)) for r in anom_refs])),
        },
    }


def run_one(cli_args, rift_variant, seed, device):
    cfg = copy.deepcopy(vars(cli_args))
    cfg.update(RIFT_VARIANTS[rift_variant]["changes"])
    cfg["seed"] = int(seed)
    cfg["device"] = int(device)
    args = argparse.Namespace(**cfg)
    pack = build_cleg3_matrix(args, seed, device)
    mat_std, standardization = standardize_matrix_global(pack["mat"], pack["normal_idx"])
    device_obj = torch.device(f"cuda:{device}" if torch.cuda.is_available() and int(device) >= 0 else "cpu")
    print(json.dumps({"stage": "rift_train_start", "variant": rift_variant, "seed": seed, "mat_shape": list(pack["mat"].shape)}, ensure_ascii=False), flush=True)
    train = rocc_train(mat_std, pack["normal_idx"], pack["labels_np"], pack["idx_test"], args, device_obj)
    rift_score = train.pop("score")
    train.pop("embedding", None)
    train.pop("center", None)
    rift_metrics = metric_block(pack["labels_np"], pack["idx_test"], {
        "rift_rocc_energy": rift_score,
        "margin": pack["margin"],
        "mat_mean": pack["mat_mean"],
    }, base_name="margin")
    row = {
        "variant": rift_variant,
        "report_codename": RIFT_VARIANTS[rift_variant]["report_codename"],
        "definition": RIFT_VARIANTS[rift_variant]["definition"],
        "seed": int(seed),
        "device": int(device),
        "effective_config": cfg,
        "protocol": {
            "cleg3_regime": "old_exact_080_regime",
            "position_encoding": "none; scalar response-entry value only; global scalar normalization from known normals",
            "training_labels": "known-normal nodes only; all non-known-normal nodes treated as unlabeled for trimmed majority-normal ROCC",
            "selection_rule": "best checkpoint by label-free ROCC monitor, never by diagnostic AUC/AP",
            "diagnostic_labels": "AUC/AP, reference purity, and baseline comparisons are computed after training only",
        },
        "split_fingerprint": pack["split_fingerprint"],
        "standardization": standardization,
        "matrix_shape": list(pack["mat"].shape),
        "baseline_metrics": pack["baseline_metrics"],
        "reference_global": pack["reference_global"],
        "rift_train": train,
        "rift_metrics": rift_metrics,
        "comparisons": {
            "rift_minus_mat_mean_auc": None if rift_metrics["rift_rocc_energy"]["auc"] is None else float(rift_metrics["rift_rocc_energy"]["auc"] - pack["baseline_metrics"]["mat_mean"]["auc"]),
            "rift_minus_margin_auc": None if rift_metrics["rift_rocc_energy"]["auc"] is None else float(rift_metrics["rift_rocc_energy"]["auc"] - pack["baseline_metrics"]["margin"]["auc"]),
            "spearman_rift_with_mat_mean": safe_spearman(rift_score, pack["mat_mean"]),
            "spearman_rift_with_margin": safe_spearman(rift_score, pack["margin"]),
            "top5_jaccard_rift_with_mat_mean": jaccard_top(rift_score[pack["idx_test"]], pack["mat_mean"][pack["idx_test"]], 0.05),
        },
    }
    print(json.dumps({"stage": "rift_seed_done", "variant": rift_variant, "seed": seed, "rift_auc": rift_metrics["rift_rocc_energy"]["auc"], "mat_mean_auc": pack["baseline_metrics"]["mat_mean"]["auc"]}, ensure_ascii=False), flush=True)
    return row


def summarize(rows):
    return {
        "n_rows": len(rows),
        "rift_auc": mean_std([r["rift_metrics"]["rift_rocc_energy"]["auc"] for r in rows]),
        "rift_ap": mean_std([r["rift_metrics"]["rift_rocc_energy"]["ap"] for r in rows]),
        "mat_mean_auc": mean_std([r["baseline_metrics"]["mat_mean"]["auc"] for r in rows]),
        "mat_mean_ap": mean_std([r["baseline_metrics"]["mat_mean"]["ap"] for r in rows]),
        "margin_auc": mean_std([r["baseline_metrics"]["margin"]["auc"] for r in rows]),
        "rift_minus_mat_mean_auc": mean_std([r["comparisons"]["rift_minus_mat_mean_auc"] for r in rows]),
        "spearman_rift_with_mat_mean": mean_std([r["comparisons"]["spearman_rift_with_mat_mean"] for r in rows]),
        "final_center_drift": mean_std([(r["rift_train"].get("final_no_leakage") or {}).get("center_drift") for r in rows]),
        "final_trimmed_jaccard": mean_std([(r["rift_train"].get("final_no_leakage") or {}).get("trimmed_unlabeled_jaccard") for r in rows]),
        "final_effective_rank": mean_std([(r["rift_train"].get("final_no_leakage") or {}).get("collapse_effective_rank") for r in rows]),
        "split_fingerprints": {str(r["seed"]): r["split_fingerprint"] for r in rows},
    }



def worker_loop(worker_id, physical_device, args_dict, task_queue, result_queue):
    """One spawned worker owns one physical GPU and processes tasks sequentially."""
    worker_args = argparse.Namespace(**args_dict)
    while True:
        task = task_queue.get()
        if task is None:
            result_queue.put({"type": "worker_exit", "worker_id": int(worker_id), "device": int(physical_device)})
            return
        variant, seed = task
        current = {"variant": variant, "seed": int(seed), "device": int(physical_device), "worker_id": int(worker_id)}
        result_queue.put({"type": "task_start", "current": current})
        print(json.dumps({"stage": "task_start", **current}, ensure_ascii=False), flush=True)
        try:
            row = run_one(worker_args, variant, seed, int(physical_device))
            row["worker_id"] = int(worker_id)
            row["device"] = int(physical_device)
            result_queue.put({"type": "task_done", "current": current, "row": row})
            print(json.dumps({"stage": "task_done", **current}, ensure_ascii=False), flush=True)
        except Exception as e:
            tb = traceback.format_exc()
            print(tb, flush=True)
            result_queue.put({
                "type": "task_error",
                "current": current,
                "error": repr(e),
                "traceback": tb[-4000:],
            })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(Path.home() / "DualRefGAD"))
    ap.add_argument("--dataset", default="elliptic")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--devices", default="")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--variants", default="r0_mean_lr1e3,r0_mean_lr3e4,r0_attn_lr1e3")
    ap.add_argument("--train_rate", type=float, default=0.05)
    ap.add_argument("--val_rate", type=float, default=0.0)
    # C-LEG3 fixed response-matrix construction knobs; defaults match strict audit.
    ap.add_argument("--ln_mode", default="descriptor_similarity")
    ap.add_argument("--normal_k", type=int, default=4)
    ap.add_argument("--anom_k", type=int, default=16)
    ap.add_argument("--encode_batch_size", type=int, default=1024)
    ap.add_argument("--ref_block_size", type=int, default=1024)
    ap.add_argument("--top_k", type=int, default=96)
    ap.add_argument("--n_bins", type=int, default=3)
    ap.add_argument("--pca_dims", default="2,4,8,12,24")
    # RIFT-R0 reader / ROCC knobs.
    ap.add_argument("--rift_dim", type=int, default=48)
    ap.add_argument("--rift_heads", type=int, default=4)
    ap.add_argument("--rift_layers", type=int, default=1)
    ap.add_argument("--rift_ffn_dim", type=int, default=96)
    ap.add_argument("--rift_dropout", type=float, default=0.10)
    ap.add_argument("--pooling", choices=["mean", "attn"], default="mean")
    ap.add_argument("--rift_lr", type=float, default=1e-3)
    ap.add_argument("--rift_weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_epoch", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--eval_batch_size", type=int, default=2048)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--trim_fraction", type=float, default=0.70)
    ap.add_argument("--lambda_trimmed_unlabeled", type=float, default=0.25)
    ap.add_argument("--lambda_var", type=float, default=0.05)
    ap.add_argument("--var_floor", type=float, default=0.05)
    ap.add_argument("--train_token_drop", type=float, default=0.0)
    ap.add_argument("--rsc_token_drop", type=float, default=0.10)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress_out", default="")
    args = ap.parse_args()

    seeds = parse_ints(args.seeds)
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    devices = parse_ints(args.devices) if args.devices.strip() else [int(args.device)]
    unknown = [v for v in variants if v not in RIFT_VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown RIFT variants: {unknown}; available={list(RIFT_VARIANTS)}")

    start = time.time()
    rows_by_variant = {v: [] for v in variants}
    errors = []
    total = len(seeds) * len(variants)
    done = 0

    def snapshot(status="running", current=None):
        atomic_write_json(args.progress_out, {
            "status": status,
            "probe": "rift_gt_r0_rocc_probe",
            "done": done,
            "total": total,
            "variants": variants,
            "variant_definitions": {k: RIFT_VARIANTS[k] for k in variants},
            "seeds": seeds,
            "current": current,
            "partial": {v: summarize(rows_by_variant[v]) if rows_by_variant[v] else {"n_rows": 0} for v in variants},
            "errors": errors[-5:],
            "elapsed_sec": time.time() - start,
        })

    snapshot("running")
    tasks = [(variant, seed) for variant in variants for seed in seeds]
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    for task in tasks:
        task_queue.put(task)
    for _ in devices:
        task_queue.put(None)

    args_dict = vars(args).copy()
    workers = []
    for worker_id, physical_device in enumerate(devices):
        proc = ctx.Process(target=worker_loop, args=(worker_id, int(physical_device), args_dict, task_queue, result_queue), daemon=False)
        proc.start()
        workers.append(proc)
        print(json.dumps({"stage": "worker_start", "worker_id": worker_id, "device": int(physical_device), "pid": proc.pid}, ensure_ascii=False), flush=True)

    current_by_worker = {}
    exited_workers = 0
    while done < total:
        try:
            msg = result_queue.get(timeout=30)
        except queue.Empty:
            live = [p.is_alive() for p in workers]
            snapshot("running", current={"active": list(current_by_worker.values()), "worker_alive": live})
            if not any(live):
                errors.append({"error": "all workers exited before all tasks completed", "done": done, "total": total})
                break
            continue
        mtype = msg.get("type")
        if mtype == "task_start":
            cur = msg["current"]
            current_by_worker[cur["worker_id"]] = cur
            snapshot("running", current={"active": list(current_by_worker.values())})
        elif mtype == "task_done":
            cur = msg["current"]
            rows_by_variant[cur["variant"]].append(msg["row"])
            done += 1
            current_by_worker.pop(cur["worker_id"], None)
            snapshot("running", current={"active": list(current_by_worker.values()), "last_done": cur})
        elif mtype == "task_error":
            cur = msg["current"]
            errors.append({"variant": cur["variant"], "seed": int(cur["seed"]), "device": int(cur["device"]), "worker_id": int(cur["worker_id"]), "error": msg.get("error"), "traceback": msg.get("traceback")})
            done += 1
            current_by_worker.pop(cur["worker_id"], None)
            snapshot("running", current={"active": list(current_by_worker.values()), "last_error": cur})
        elif mtype == "worker_exit":
            exited_workers += 1
            snapshot("running", current={"active": list(current_by_worker.values()), "exited_workers": exited_workers})

    for proc in workers:
        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)
            errors.append({"error": "worker terminated after timeout", "pid": proc.pid})
        elif proc.exitcode not in (0, None):
            errors.append({"error": "worker nonzero exit", "pid": proc.pid, "exitcode": proc.exitcode})

    variant_summaries = []
    for variant in variants:
        rows = sorted(rows_by_variant[variant], key=lambda r: r["seed"])
        if len(rows) != len(seeds):
            errors.append({"variant": variant, "error": f"completed {len(rows)}/{len(seeds)} seeds"})
        variant_summaries.append({
            "variant": variant,
            "report_codename": RIFT_VARIANTS[variant]["report_codename"],
            "definition": RIFT_VARIANTS[variant]["definition"],
            "changes_from_base": RIFT_VARIANTS[variant]["changes"],
            "aggregate": summarize(rows),
            "rows": rows,
        })
    payload = {
        "status": "finished" if not errors else "finished_with_errors",
        "probe": "rift_gt_r0_rocc_probe",
        "method_family": "RIFT-GT / RIFT-R0",
        "protocol": {
            "input": "fixed C-LEG3 old_exact_080_regime response matrix, normal_k=4, anom_k=16 unless overridden",
            "architecture": "no-position scalar response-entry Set/Transformer reader; z_v is pooled contextual token representation",
            "loss": "ROCC = known-normal compactness + trimmed majority-unlabeled compactness + light anti-collapse variance floor",
            "selection": "label-free monitor only; diagnostic AUC/AP forbidden for checkpoint or hyperparameter selection",
            "diagnostics": "AUC/AP, Spearman/top-K overlap with mat_mean/margin, and reference purity are post-training diagnostic-only",
        },
        "config": vars(args),
        "variant_definitions": {k: RIFT_VARIANTS[k] for k in variants},
        "cleg3_reference_definition": CLEG_VARIANTS["old_exact_080_regime"],
        "results": variant_summaries,
        "errors": errors,
        "elapsed_sec": time.time() - start,
    }
    atomic_write_json(args.out, payload)
    snapshot(payload["status"])
    print(json.dumps({"stage": "probe_done", "status": payload["status"], "done": done, "total": total, "out": args.out}, ensure_ascii=False), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
