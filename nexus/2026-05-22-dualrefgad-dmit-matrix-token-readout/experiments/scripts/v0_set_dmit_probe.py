#!/usr/bin/env python3
"""V0 / Set-D-MiT probe for DualRefGAD.

Scientific boundary:
- Fixed C-LEG3 / old_exact response-matrix regime.
- 64 response entries are treated as an unordered set: no row-id, col-id,
  absolute 2D position, or reference identity embedding.
- Training uses known-normal matrices plus pseudo matrices generated from known
  normals. True anomaly labels are diagnostic-only for AUC/AP/autopsy.
"""
import argparse
import copy
import json
import os
import queue
import random
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from route25_leg3_response_matrix_decomposition_probe import (  # noqa: E402
    BASE_DEFAULTS,
    VARIANTS,
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


def atomic_write_json(path, payload):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    x = np.asarray(vals, dtype=np.float64)
    return {"mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()), "max": float(x.max())}


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


class SetDMiT(nn.Module):
    """Permutation-invariant matrix-entry token reader.

    Each scalar response entry is embedded independently. No row/column/id/2D
    information is provided. Transformer self-attention can only compare entry
    values and their learned scalar embeddings; mean pooling keeps the reader
    permutation invariant up to Transformer numerical effects.
    """

    def __init__(self, d_model=32, nhead=4, num_layers=1, ffn_dim=64, dropout=0.1):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
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
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        # x: [B, 64]
        tok = self.entry(x.unsqueeze(-1))
        tok = self.encoder(tok)
        pooled = tok.mean(dim=1)
        return self.head(pooled).squeeze(-1)


def build_readout_scores(mat, margin):
    x = mat.reshape(mat.shape[0], -1).astype(np.float64)
    row_mean = mat.mean(axis=2)
    col_mean = mat.mean(axis=1)
    sorted_x = np.sort(x, axis=1)
    q10 = np.quantile(x, 0.10, axis=1)
    q25 = np.quantile(x, 0.25, axis=1)
    q50 = np.quantile(x, 0.50, axis=1)
    q75 = np.quantile(x, 0.75, axis=1)
    q90 = np.quantile(x, 0.90, axis=1)
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    top8 = sorted_x[:, -8:].mean(axis=1)
    top16 = sorted_x[:, -16:].mean(axis=1)
    trim10 = sorted_x[:, 6:-6].mean(axis=1) if x.shape[1] > 12 else mean
    winsor10 = np.clip(x, q10[:, None], q90[:, None]).mean(axis=1)
    return {
        "margin": margin.astype(np.float64),
        "mat_mean": mean,
        "mat_std": std,
        "trim10_mean": trim10,
        "winsor10_mean": winsor10,
        "q50_median": q50,
        "q75": q75,
        "q90": q90,
        "top8_mean": top8,
        "top16_mean": top16,
        "mean_q75_blend": 0.5 * mean + 0.5 * q75,
        "mean_top16_blend": 0.5 * mean + 0.5 * top16,
        "row_top2_mean": np.sort(row_mean, axis=1)[:, -2:].mean(axis=1),
        "col_top4_mean": np.sort(col_mean, axis=1)[:, -4:].mean(axis=1),
    }


def make_pseudo(norm_x, mu, sd, rng, noise_scale=1.0, tail_boost=1.0, strategy="v0_mixed", alpha_low=0.65, alpha_high=1.25):
    """Generate pseudo matrices from known normals without anomaly labels.

    Supported strategies:
    - v0_mixed: historical three-family mixture from the first Set-D-MiT probe.
    - dense_hard_shift: calibrated dense positive shift selected by the G2 probe.

    For dense_hard_shift, alpha_low/alpha_high are fixed internal generator
    hyperparameters unless explicitly swept by a later calibration.
    """
    n, d = norm_x.shape
    base = norm_x.copy()
    z_sd = sd.reshape(1, d)
    center = mu.reshape(1, d)
    if strategy == "dense_hard_shift":
        amp = rng.uniform(float(alpha_low), float(alpha_high), size=(n, 1)).astype(np.float32) * float(noise_scale)
        return np.clip(base + amp * z_sd, -1.0, 1.0).astype(np.float32)
    if strategy != "v0_mixed":
        raise ValueError(f"Unsupported pseudo_strategy for trainable probe: {strategy}")
    typ = rng.integers(0, 3, size=n)
    pseudo = base.copy()
    for i in range(n):
        if typ[i] == 0:
            pseudo[i] = base[i] + rng.uniform(0.35, 0.90) * noise_scale * z_sd.reshape(-1)
        elif typ[i] == 1:
            k = int(rng.integers(max(4, d // 8), max(5, d // 3)))
            idx = rng.choice(d, size=k, replace=False)
            pseudo[i, idx] = base[i, idx] + rng.uniform(0.8, 1.8) * tail_boost * z_sd.reshape(-1)[idx]
        else:
            k_hi = int(rng.integers(max(4, d // 10), max(5, d // 4)))
            k_lo = int(rng.integers(max(4, d // 10), max(5, d // 4)))
            hi = rng.choice(d, size=k_hi, replace=False)
            lo = rng.choice(d, size=k_lo, replace=False)
            pseudo[i, hi] = base[i, hi] + rng.uniform(0.8, 1.6) * z_sd.reshape(-1)[hi]
            pseudo[i, lo] = center.reshape(-1)[lo] - rng.uniform(0.8, 1.6) * z_sd.reshape(-1)[lo]
    return np.clip(pseudo, -1.0, 1.0).astype(np.float32)


def train_set_dmit(x_all, normal_idx, idx_test, seed, device_obj, args):
    rng = np.random.default_rng(seed + 20260522)
    normals = np.asarray(normal_idx, dtype=np.int64)
    tr_norm, va_norm = train_test_split(normals, test_size=args.normal_val_frac, random_state=seed, shuffle=True)
    mu = x_all[tr_norm].mean(axis=0)
    sd = x_all[tr_norm].std(axis=0) + 1e-6
    x_std = ((x_all - mu) / sd).astype(np.float32)

    train_norm_x = x_all[tr_norm].astype(np.float32)
    val_norm_x = x_all[va_norm].astype(np.float32)
    model = SetDMiT(args.d_model, args.nhead, args.reader_layers, args.reader_ffn_dim, args.reader_dropout).to(device_obj)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state, best_val, best_epoch = None, float("inf"), -1
    history = []
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train_norm_x))
        losses = []
        pair_accs = []
        for st in range(0, len(order), args.batch_size):
            ids = order[st: st + args.batch_size]
            n_raw = train_norm_x[ids]
            p_raw = make_pseudo(
                n_raw,
                mu,
                sd,
                rng,
                args.pseudo_noise_scale,
                args.pseudo_tail_boost,
                args.pseudo_strategy,
                args.pseudo_alpha_low,
                args.pseudo_alpha_high,
            )
            n = ((n_raw - mu) / sd).astype(np.float32)
            p = ((p_raw - mu) / sd).astype(np.float32)
            xb = torch.tensor(np.vstack([n, p]), dtype=torch.float32, device=device_obj)
            yb = torch.tensor(np.concatenate([np.zeros(len(n)), np.ones(len(p))]), dtype=torch.float32, device=device_obj)
            logits = model(xb)
            bce = F.binary_cross_entropy_with_logits(logits, yb)
            n_logit = logits[:len(n)]
            p_logit = logits[len(n):]
            rank_loss = F.softplus(args.pair_margin - (p_logit - n_logit)).mean()
            loss = bce + args.lambda_pair * rank_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            pair_accs.append(float((p_logit.detach() > n_logit.detach()).float().mean().cpu()))

        model.eval()
        with torch.no_grad():
            vp_raw = make_pseudo(
                val_norm_x,
                mu,
                sd,
                rng,
                args.pseudo_noise_scale,
                args.pseudo_tail_boost,
                args.pseudo_strategy,
                args.pseudo_alpha_low,
                args.pseudo_alpha_high,
            )
            vn = torch.tensor(((val_norm_x - mu) / sd).astype(np.float32), device=device_obj)
            vp = torch.tensor(((vp_raw - mu) / sd).astype(np.float32), device=device_obj)
            val_n = model(vn)
            val_p = model(vp)
            val_loss = float(F.softplus(args.pair_margin - (val_p - val_n)).mean().cpu())
            val_pair_acc = float((val_p > val_n).float().mean().cpu())
            val_gap = float((val_p - val_n).mean().cpu())
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "pair_acc": float(np.mean(pair_accs)), "val_pair_loss": val_loss, "val_pair_acc": val_pair_acc, "val_gap": val_gap})
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    outs = []
    with torch.no_grad():
        for st in range(0, len(x_std), args.eval_batch_size):
            xb = torch.tensor(x_std[st: st + args.eval_batch_size], dtype=torch.float32, device=device_obj)
            outs.append(model(xb).detach().cpu().numpy())
    score = np.concatenate(outs).astype(np.float64)

    # Constraint diagnostics on held-out normal/pseudo pairs; no anomaly labels.
    with torch.no_grad():
        vp_raw = make_pseudo(
            val_norm_x,
            mu,
            sd,
            rng,
            args.pseudo_noise_scale,
            args.pseudo_tail_boost,
            args.pseudo_strategy,
            args.pseudo_alpha_low,
            args.pseudo_alpha_high,
        )
        vn = torch.tensor(((val_norm_x - mu) / sd).astype(np.float32), device=device_obj)
        vp = torch.tensor(((vp_raw - mu) / sd).astype(np.float32), device=device_obj)
        sn = model(vn).detach().cpu().numpy()
        sp = model(vp).detach().cpu().numpy()
    constraint = {
        "known_normal_score_mean": float(np.mean(sn)),
        "pseudo_score_mean": float(np.mean(sp)),
        "pseudo_minus_source_mean": float(np.mean(sp - sn)),
        "pseudo_pair_satisfaction_rate": float(np.mean(sp > sn)),
        "best_epoch": int(best_epoch),
        "best_val_pair_loss": float(best_val),
        "history_tail": history[-5:],
        "train_normals": int(len(tr_norm)),
        "val_normals": int(len(va_norm)),
    }
    return score, constraint


def top_set(idx_test, score, k):
    idx_test = np.asarray(idx_test, dtype=np.int64)
    order = np.argsort(-np.asarray(score)[idx_test])[:k]
    return set(map(int, idx_test[order]))


def autopsy_vs(labels, idx_test, score, base_score, mat, margin, normal_refs, anom_refs, meta):
    n_anom = int(np.sum(labels[idx_test] == 1))
    k = max(1, n_anom)
    top = top_set(idx_test, score, k)
    base = top_set(idx_test, base_score, k)
    groups = {
        "rescued_anomalies": [n for n in (top - base) if labels[n] == 1],
        "introduced_false_positives": [n for n in (top - base) if labels[n] == 0],
        "lost_anomalies": [n for n in (base - top) if labels[n] == 1],
        "removed_false_positives": [n for n in (base - top) if labels[n] == 0],
    }
    def summary(nodes):
        if not nodes:
            return {"count": 0}
        arr = np.asarray(nodes, dtype=np.int64)
        flat = mat[arr].reshape(len(arr), -1)
        return {
            "count": int(len(arr)),
            "score": mean_std(score[arr]),
            "base_score": mean_std(base_score[arr]),
            "margin": mean_std(margin[arr]),
            "mat_mean": mean_std(flat.mean(axis=1)),
            "mat_std": mean_std(flat.std(axis=1)),
            "anom_ref_anom_ratio": mean_std(np.mean(labels[anom_refs[arr]] == 1, axis=1)),
            "normal_ref_anom_ratio": mean_std(np.mean(labels[normal_refs[arr]] == 1, axis=1)),
            "degree": mean_std(meta["degree"][arr]),
            "rejection": mean_std(meta["rejection"][arr]),
            "residual_norm": mean_std(meta["residual_norm"][arr]),
        }
    return {name: summary(nodes) for name, nodes in groups.items()}


def run_one(args, variant, seed, device):
    set_seed(seed)
    cfg = copy.deepcopy(BASE_DEFAULTS)
    cfg.update(VARIANTS[variant]["changes"])
    cfg.update(vars(args))
    cfg["variant"] = variant
    cfg["device"] = int(device)
    cfg["seed"] = int(seed)
    v_args = argparse.Namespace(**cfg)

    root = Path(v_args.project_root).expanduser().resolve()
    sys.path.insert(0, str(root))
    os.chdir(str(root))
    from utils import load_mat, preprocess_features, normalize_adj  # noqa: E402
    from VecGAD import VecGAD  # noqa: E402

    def to_dense_features(dataset, features):
        if dataset in ["Amazon", "tf_finace", "t_finance", "reddit", "elliptic"]:
            features, _ = preprocess_features(features)
            return np.asarray(features, dtype=np.float32)
        return np.asarray(features.todense(), dtype=np.float32)

    device_obj = torch.device(f"cuda:{device}" if torch.cuda.is_available() and int(device) >= 0 else "cpu")
    print(json.dumps({"stage": "seed_start", "variant": variant, "seed": seed, "device": str(device_obj)}, ensure_ascii=False), flush=True)
    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat(v_args.dataset, v_args.train_rate, v_args.val_rate, args=v_args)
    features_np = to_dense_features(v_args.dataset, features)
    labels_np = np.asarray(ano_label).reshape(-1).astype(int)
    normal_idx = np.asarray(normal_for_train_idx, dtype=np.int64)
    idx_test = np.asarray(idx_test, dtype=np.int64)
    assert np.sum(labels_np[normal_idx]) == 0, "Data leakage: normal_for_train_idx contains anomalies"

    z = build_descriptor(v_args.descriptor_mode, features_np, adj, normalize_adj, v_args.hops, v_args.rw_steps)
    nm = NormalModel(v_args.pn_estimator, z, normal_idx, v_args.pca_components)
    residual = nm.residual()
    normal_refs, anom_refs, meta = select_refs(z, residual, normal_idx, nm, features_np, adj, v_args, normalize_adj)
    token_tensor = build_tokens(features_np, normal_refs, anom_refs)
    encoder = VecGAD(features_np.shape[1], v_args.embedding_dim, "prelu", v_args).to(device_obj)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    with torch.no_grad():
        emb = encode_tokens_batched(encoder, token_tensor, device_obj, v_args.encode_batch_size)
    del token_tensor, encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mat, margin = response_matrix_from_embeddings(emb, normal_refs, anom_refs)
    x_all = mat.reshape(mat.shape[0], -1).astype(np.float32)
    scores = build_readout_scores(mat, margin)
    dmit_score, constraint_diag = train_set_dmit(x_all, normal_idx, idx_test, seed, device_obj, args)
    scores["v0_set_dmit"] = dmit_score
    metrics = metric_block(labels_np, idx_test, scores, base_name="mat_mean")
    ranked = sorted(metrics.items(), key=lambda kv: kv[1]["auc"], reverse=True)
    n_anom_test = int(np.sum(labels_np[idx_test] == 1))
    row = {
        "variant": variant,
        "report_codename": "C-MiT/V0-Set-D-MiT",
        "seed": int(seed),
        "device": int(device),
        "position_semantics": "no row/column/reference/2D position encoding; response entries are an unordered set",
        "best_metric": {"name": ranked[0][0], **ranked[0][1]},
        "metrics": metrics,
        "constraint_diagnostics": constraint_diag,
        "score_relationship": {
            "v0_spearman_with_mat_mean": safe_corr(dmit_score[idx_test], scores["mat_mean"][idx_test]),
            "v0_spearman_with_margin": safe_corr(dmit_score[idx_test], margin[idx_test]),
            "v0_top5_jaccard_with_mat_mean": jaccard_top(dmit_score[idx_test], scores["mat_mean"][idx_test], 0.05),
            "v0_top5_jaccard_with_margin": jaccard_top(dmit_score[idx_test], margin[idx_test], 0.05),
        },
        "autopsy_vs_mat_mean": autopsy_vs(labels_np, idx_test, dmit_score, scores["mat_mean"], mat, margin, normal_refs, anom_refs, meta),
        "reference_diagnostics": {
            "anom_ref_anom_ratio_diagnostic": float(np.mean(labels_np[anom_refs] == 1)),
            "normal_ref_anom_ratio_diagnostic": float(np.mean(labels_np[normal_refs] == 1)),
            "test_anom_rate": float(np.mean(labels_np[idx_test] == 1)),
        },
        "counts": {
            "num_nodes": int(len(labels_np)),
            "num_test": int(len(idx_test)),
            "num_test_anom": int(n_anom_test),
            "num_labeled_normals": int(len(normal_idx)),
            "matrix_shape": list(mat.shape),
        },
    }
    print(json.dumps({"stage": "seed_done", "variant": variant, "seed": seed, "best": ranked[0][0], "v0_auc": metrics["v0_set_dmit"]["auc"], "mat_auc": metrics["mat_mean"]["auc"]}, ensure_ascii=False), flush=True)
    return row


def summarize(rows):
    names = sorted({name for r in rows for name in r["metrics"]})
    out = {
        "n_rows": len(rows),
        "score_summary": {},
        "best_auc_votes": {},
        "constraint_summary": {
            "pseudo_pair_satisfaction_rate": mean_std([r["constraint_diagnostics"]["pseudo_pair_satisfaction_rate"] for r in rows]),
            "pseudo_minus_source_mean": mean_std([r["constraint_diagnostics"]["pseudo_minus_source_mean"] for r in rows]),
            "known_normal_score_mean": mean_std([r["constraint_diagnostics"]["known_normal_score_mean"] for r in rows]),
            "pseudo_score_mean": mean_std([r["constraint_diagnostics"]["pseudo_score_mean"] for r in rows]),
        },
        "relationship_summary": {
            "v0_spearman_with_mat_mean": mean_std([r["score_relationship"]["v0_spearman_with_mat_mean"] for r in rows]),
            "v0_spearman_with_margin": mean_std([r["score_relationship"]["v0_spearman_with_margin"] for r in rows]),
            "v0_top5_jaccard_with_mat_mean": mean_std([r["score_relationship"]["v0_top5_jaccard_with_mat_mean"] for r in rows]),
            "v0_top5_jaccard_with_margin": mean_std([r["score_relationship"]["v0_top5_jaccard_with_margin"] for r in rows]),
        },
    }
    for name in names:
        out["score_summary"][name] = {
            "auc": mean_std([r["metrics"][name]["auc"] for r in rows]),
            "ap": mean_std([r["metrics"][name]["ap"] for r in rows]),
            "spearman_with_mat_mean": mean_std([r["metrics"][name].get("spearman_with_margin") for r in rows]),
        }
    for r in rows:
        n = r["best_metric"]["name"]
        out["best_auc_votes"][n] = out["best_auc_votes"].get(n, 0) + 1
    mat_auc = out["score_summary"]["mat_mean"]["auc"]["mean"]
    v0_auc = out["score_summary"]["v0_set_dmit"]["auc"]["mean"]
    v0_ap = out["score_summary"]["v0_set_dmit"]["ap"]["mean"]
    mat_ap = out["score_summary"]["mat_mean"]["ap"]["mean"]
    out["v0_vs_mat_mean"] = {"auc_delta": float(v0_auc - mat_auc), "ap_delta": float(v0_ap - mat_ap)}
    spear = out["relationship_summary"]["v0_spearman_with_mat_mean"]["mean"]
    jacc = out["relationship_summary"]["v0_top5_jaccard_with_mat_mean"]["mean"]
    if v0_auc >= mat_auc + 0.002:
        out["decision"] = "V0_SET_DMIT_BEATS_MAT_MEAN_GATE"
    elif v0_auc >= mat_auc - 0.01 and (spear < 0.90 or jacc < 0.70):
        out["decision"] = "V0_SET_DMIT_COMPLEMENTARY_SOFT_SIGNAL"
    elif v0_auc >= mat_auc - 0.02:
        out["decision"] = "V0_SET_DMIT_CLOSE_BUT_MAY_TRACK_SCALAR_BASELINE"
    else:
        out["decision"] = "V0_SET_DMIT_UNDERPERFORMS_STRONG_SCALAR_BASELINE"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(Path.home() / "DualRefGAD"))
    ap.add_argument("--dataset", default="elliptic")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--devices", default="")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--variants", default="old_exact_080_regime")
    ap.add_argument("--train_rate", type=float, default=0.05)
    ap.add_argument("--val_rate", type=float, default=0.0)
    ap.add_argument("--ln_mode", default="descriptor_similarity")
    ap.add_argument("--normal_k", type=int, default=4)
    ap.add_argument("--anom_k", type=int, default=16)
    ap.add_argument("--encode_batch_size", type=int, default=1024)
    ap.add_argument("--ref_block_size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--eval_batch_size", type=int, default=4096)
    ap.add_argument("--normal_val_frac", type=float, default=0.25)
    ap.add_argument("--d_model", type=int, default=32)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--reader_layers", type=int, default=1)
    ap.add_argument("--reader_ffn_dim", type=int, default=64)
    ap.add_argument("--reader_dropout", type=float, default=0.10)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--weight_decay", type=float, default=0.0001)
    ap.add_argument("--lambda_pair", type=float, default=0.5)
    ap.add_argument("--pair_margin", type=float, default=1.0)
    ap.add_argument("--pseudo_noise_scale", type=float, default=1.0)
    ap.add_argument("--pseudo_tail_boost", type=float, default=1.0)
    ap.add_argument("--pseudo_strategy", default="v0_mixed", choices=["v0_mixed", "dense_hard_shift"])
    ap.add_argument("--pseudo_alpha_low", type=float, default=0.65)
    ap.add_argument("--pseudo_alpha_high", type=float, default=1.25)
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress_out", default="")
    args = ap.parse_args()

    seeds = parse_ints(args.seeds)
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}; available={list(VARIANTS)}")
    devices = parse_ints(args.devices) if args.devices.strip() else [int(args.device)]
    if not devices:
        devices = [int(args.device)]

    start = time.time()
    task_q = queue.Queue()
    for v in variants:
        for s in seeds:
            task_q.put((v, s))
    total = task_q.qsize()
    rows_by_variant = {v: [] for v in variants}
    errors = []
    done = 0
    lock = threading.Lock()

    def snapshot(status="running", current=None):
        partial = []
        for v in variants:
            if rows_by_variant[v]:
                partial.append({"variant": v, "summary": summarize(rows_by_variant[v])})
        return {
            "status": status,
            "done": done,
            "total": total,
            "devices": devices,
            "current": current,
            "errors": errors[-5:],
            "partial": partial,
            "elapsed_sec": round(time.time() - start, 2),
        }

    def worker(device):
        nonlocal done
        while True:
            try:
                variant, seed = task_q.get_nowait()
            except queue.Empty:
                return
            current = {"variant": variant, "seed": seed, "device": int(device)}
            with lock:
                atomic_write_json(args.progress_out, snapshot("running", current))
            try:
                row = run_one(args, variant, seed, int(device))
                with lock:
                    rows_by_variant[variant].append(row)
                    done += 1
                    atomic_write_json(args.progress_out, snapshot("running", current))
            except Exception as e:
                tb = traceback.format_exc()
                with lock:
                    errors.append({"variant": variant, "seed": int(seed), "device": int(device), "error": repr(e), "traceback": tb})
                    done += 1
                    atomic_write_json(args.progress_out, snapshot("running", current))
                print(tb, flush=True)
            finally:
                task_q.task_done()

    threads = [threading.Thread(target=worker, args=(d,), daemon=False) for d in devices]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    status = "failed" if errors else "finished"
    payload = {
        "probe": "v0_set_dmit_probe",
        "codename": "C-MiT / V0 Set-D-MiT",
        "protocol": {
            "position_encoding": "none",
            "token_identity": "none",
            "training_labels": "known normal vs pseudo matrices generated from known normals",
            "diagnostic_labels": "true anomaly labels used only for AUC/AP/autopsy",
            "pseudo_strategy": args.pseudo_strategy,
            "pseudo_noise_scale": float(args.pseudo_noise_scale),
            "pseudo_alpha_low": float(args.pseudo_alpha_low),
            "pseudo_alpha_high": float(args.pseudo_alpha_high),
            "pseudo_alpha_note": "alpha_low/alpha_high are fixed internal generator hyperparameters in this run, not swept",
            "variant": variants,
            "seeds": seeds,
            "devices": devices,
        },
        "status": status,
        "elapsed_sec": round(time.time() - start, 2),
        "errors": errors,
        "by_variant": {v: {"rows": rows_by_variant[v], "summary": summarize(rows_by_variant[v]) if rows_by_variant[v] else None} for v in variants},
    }
    atomic_write_json(args.out, payload)
    atomic_write_json(args.progress_out, snapshot(status, None))
    print(json.dumps({"stage": "done", "status": status, "out": args.out, "elapsed_sec": payload["elapsed_sec"]}, ensure_ascii=False), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
