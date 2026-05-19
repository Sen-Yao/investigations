#!/usr/bin/env python3
"""Phase 1 audit for Route2.5 Matrix AE instability.

Question: is the observed 5-seed Matrix AE instability mostly split/reference
instability, or can repeated AE initializations on the same split recover a stable
normal-only reconstruction signal?

This script intentionally reuses the Route2.5 Matrix AE feature construction and
runs repeated AE initializations per split seed. Labels remain diagnostic-only.
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

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

# Import the original registered probe implementation as the protocol source.
from route25_matrix_autoencoder_probe import (  # noqa: E402
    MatrixAE,
    NormalModel,
    build_descriptor,
    build_tokens,
    encode_tokens_batched,
    metric_block,
    response_matrix_from_embeddings,
    safe_spearman,
    select_refs,
    set_seed,
)


def safe_auc_ap(labels, score, idx):
    idx = np.asarray(idx, dtype=np.int64)
    return float(roc_auc_score(labels[idx], score[idx])), float(average_precision_score(labels[idx], score[idx]))


def train_ae_once(X, normal_idx, labels, idx_test, args, latent_dim, ae_seed, device):
    """Train one AE with explicit init/split seed, keeping the same protocol."""
    random.seed(ae_seed)
    np.random.seed(ae_seed)
    torch.manual_seed(ae_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(ae_seed)

    nidx = np.asarray(normal_idx, dtype=np.int64)
    rng = np.random.default_rng(ae_seed + latent_dim * 1009)
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
            xin = xb + torch.randn_like(xb) * args.denoise_std if args.denoise_std > 0 else xb
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
                val_loss = float(F.mse_loss(model(x_all[val_t]), x_all[val_t]).detach().cpu())
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
            scores.append(torch.mean((model(xb) - xb) ** 2, dim=1).detach().cpu().numpy())
    score = np.concatenate(scores, axis=0)
    auc, ap = safe_auc_ap(labels, score, idx_test)
    return {
        "ae_seed": int(ae_seed),
        "latent_dim": int(latent_dim),
        "auc": auc,
        "ap": ap,
        "best_val_loss": float(best_val),
        "last_train_loss": float(last_train if last_train is not None else 0.0),
        "spearman_with_margin": None,  # filled by caller
        "spearman_with_degree": None,
    }, score


def mean_std(xs):
    xs = np.asarray(xs, dtype=float)
    return {"mean": float(xs.mean()), "std": float(xs.std(ddof=0)), "min": float(xs.min()), "max": float(xs.max())}


def corr_or_zero(a, b):
    if len(a) < 3:
        return 0.0
    return safe_spearman(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(Path.home() / "DualRefGAD"))
    ap.add_argument("--dataset", default="elliptic")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--split_seeds", default="0,1,2,3,4")
    ap.add_argument("--ae_seeds", default="0,1,2,3,4")
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
    ap.add_argument("--encode_batch_size", type=int, default=1024)
    ap.add_argument("--ref_block_size", type=int, default=1024)
    ap.add_argument("--use_approx_anom_refs", action="store_true")
    ap.add_argument("--anom_approx_k", type=int, default=1000)
    ap.add_argument("--latent_dims", default="8,16")
    ap.add_argument("--num_epoch", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--ae_hidden_dim", type=int, default=32)
    ap.add_argument("--ae_lr", type=float, default=1e-3)
    ap.add_argument("--ae_weight_decay", type=float, default=1e-5)
    ap.add_argument("--ae_dropout", type=float, default=0.0)
    ap.add_argument("--denoise_std", type=float, default=0.0)
    ap.add_argument("--normal_val_frac", type=float, default=0.2)
    ap.add_argument("--eval_every", type=int, default=20)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    start = time.time()
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
    split_seeds = [int(x) for x in args.split_seeds.split(",") if x.strip()]
    ae_seeds = [int(x) for x in args.ae_seeds.split(",") if x.strip()]
    latent_dims = [int(x) for x in args.latent_dims.split(",") if x.strip()]
    print(json.dumps({"stage": "phase1_start", "device": str(device), "split_seeds": split_seeds, "ae_seeds": ae_seeds, "latent_dims": latent_dims}, ensure_ascii=False), flush=True)

    split_results = []
    for split_seed in split_seeds:
        sargs = copy.copy(args)
        sargs.seed = split_seed
        set_seed(split_seed)
        print(json.dumps({"stage": "load_data", "split_seed": split_seed}, ensure_ascii=False), flush=True)
        adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat(args.dataset, args.train_rate, args.val_rate, args=sargs)
        features_np = to_dense_features(args.dataset, features)
        labels_np = np.asarray(ano_label).reshape(-1).astype(int)
        normal_idx = np.asarray(normal_for_train_idx, dtype=np.int64)
        idx_test = np.asarray(idx_test, dtype=np.int64)
        assert np.sum(labels_np[normal_idx]) == 0, "Data leakage: normal_for_train_idx contains anomalies"

        z = build_descriptor(args.descriptor_mode, features_np, adj, normalize_adj, args.hops, args.rw_steps)
        nm = NormalModel(args.pn_estimator, z, normal_idx, args.pca_components)
        residual = nm.residual()
        normal_refs, anom_refs, meta = select_refs(z, residual, normal_idx, nm, features_np, adj, sargs, normalize_adj)
        token_tensor = build_tokens(features_np, normal_refs, anom_refs)
        model = VecGAD(features_np.shape[1], args.embedding_dim, "prelu", sargs).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        with torch.no_grad():
            emb = encode_tokens_batched(model, token_tensor, device, args.encode_batch_size)
        del token_tensor, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
        runs = []
        for ae_seed in ae_seeds:
            for ld in latent_dims:
                run, score = train_ae_once(X, normal_idx, labels_np, idx_test, args, ld, ae_seed, device)
                run["spearman_with_margin"] = safe_spearman(score, margin)
                run["spearman_with_degree"] = safe_spearman(score, meta["degree"])
                run["delta_auc_vs_scalar"] = float(run["auc"] - scalar_best["auc"])
                run["delta_ap_vs_scalar"] = float(run["ap"] - scalar_best["ap"])
                runs.append(run)
                print(json.dumps({"stage": "ae_repeat", "split_seed": split_seed, "scalar_best": scalar_best_name, **run}, ensure_ascii=False), flush=True)
        by_latent = {}
        for ld in latent_dims:
            rr = [r for r in runs if r["latent_dim"] == ld]
            by_latent[str(ld)] = {
                "auc": mean_std([r["auc"] for r in rr]),
                "ap": mean_std([r["ap"] for r in rr]),
                "delta_auc_vs_scalar": mean_std([r["delta_auc_vs_scalar"] for r in rr]),
                "best_val_loss": mean_std([r["best_val_loss"] for r in rr]),
                "spearman_with_margin": mean_std([r["spearman_with_margin"] for r in rr]),
                "spearman_with_degree": mean_std([r["spearman_with_degree"] for r in rr]),
            }
        best_run = max(runs, key=lambda r: r["auc"])
        split_result = {
            "split_seed": int(split_seed),
            "scalar_best": {"name": scalar_best_name, **scalar_best},
            "reference_diagnostics": {
                "anom_ref_anom_ratio_diagnostic": float(np.mean(labels_np[anom_refs] == 1)),
                "normal_ref_anom_ratio_diagnostic": float(np.mean(labels_np[normal_refs] == 1)),
            },
            "by_latent": by_latent,
            "best_repeat": best_run,
            "runs": runs,
        }
        print(json.dumps({"stage": "split_done", "split_seed": split_seed, "best_repeat": best_run, "scalar_auc": scalar_best["auc"]}, ensure_ascii=False), flush=True)
        split_results.append(split_result)

    best_repeat_auc = [s["best_repeat"]["auc"] for s in split_results]
    best_repeat_delta = [s["best_repeat"]["delta_auc_vs_scalar"] for s in split_results]
    scalar_auc = [s["scalar_best"]["auc"] for s in split_results]
    ref_anom_ratio = [s["reference_diagnostics"]["anom_ref_anom_ratio_diagnostic"] for s in split_results]
    within_split_auc_std = []
    for s in split_results:
        within_split_auc_std.append(float(np.mean([s["by_latent"][str(ld)]["auc"]["std"] for ld in latent_dims])))

    result = {
        "status": "finished",
        "probe": "route25_matrix_ae_phase1_instability_audit",
        "protocol": "Frozen encoder; Route2.5 response matrix; repeated AE initializations per split; labels diagnostic-only.",
        "dataset": args.dataset,
        "config": vars(args),
        "split_results": split_results,
        "aggregate": {
            "best_repeat_auc": mean_std(best_repeat_auc),
            "best_repeat_delta_auc_vs_scalar": mean_std(best_repeat_delta),
            "scalar_auc": mean_std(scalar_auc),
            "within_split_ae_auc_std_mean": mean_std(within_split_auc_std),
            "corr_best_delta_with_anom_ref_ratio": corr_or_zero(best_repeat_delta, ref_anom_ratio),
            "corr_best_delta_with_scalar_auc": corr_or_zero(best_repeat_delta, scalar_auc),
            "promote_repeat_count": int(sum(x > 0.02 for x in best_repeat_delta)),
            "drop_repeat_count": int(sum(a < 0.58 for a in best_repeat_auc)),
        },
        "decision": None,
        "time_sec": float(time.time() - start),
    }
    agg = result["aggregate"]
    if agg["best_repeat_delta_auc_vs_scalar"]["mean"] > 0.02 and agg["promote_repeat_count"] >= max(3, len(split_seeds) - 1):
        result["decision"] = "AE_INIT_REPAIR_PROMISING"
    elif agg["within_split_ae_auc_std_mean"]["mean"] < 0.015 and agg["best_repeat_delta_auc_vs_scalar"]["mean"] < 0:
        result["decision"] = "SPLIT_REFERENCE_INSTABILITY_DOMINATES__DO_NOT_PROMOTE"
    elif agg["best_repeat_auc"]["mean"] < 0.58:
        result["decision"] = "DROP"
    else:
        result["decision"] = "INCONCLUSIVE_BUT_NOT_PROMOTED"
    print("FINAL " + json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
