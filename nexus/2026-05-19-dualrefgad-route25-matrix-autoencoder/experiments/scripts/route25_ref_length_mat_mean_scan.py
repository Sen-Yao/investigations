#!/usr/bin/env python3
"""Route2.5 normal/anomaly reference length scan for mat_mean performance.

Protocol:
- Frozen VecGAD encoder / response matrix construction reused from Route2.5.
- No AE training.
- Sweep normal_k x anom_k, 5 seeds each.
- Labels diagnostic-only for AUC/AP.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from route25_matrix_autoencoder_probe import (  # noqa: E402
    NormalModel,
    build_descriptor,
    build_tokens,
    encode_tokens_batched,
    metric_block,
    response_matrix_from_embeddings,
    safe_spearman,
    set_seed,
)
from route25_matrix_autoencoder_probe import select_refs  # noqa: E402


def parse_int_list(s):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def mean_std(vals):
    vals = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def to_dense_features(dataset, features, preprocess_features):
    if dataset in ["Amazon", "tf_finace", "t_finance", "reddit", "elliptic"]:
        features, _ = preprocess_features(features)
        return np.asarray(features, dtype=np.float32)
    return np.asarray(features.todense(), dtype=np.float32)


def run_one(args, base_args, seed, normal_k, anom_k):
    set_seed(seed)
    base_args.seed = seed
    base_args.normal_k = int(normal_k)
    base_args.anom_k = int(anom_k)

    root = Path(args.project_root).expanduser().resolve()
    sys.path.insert(0, str(root))
    os.chdir(str(root))
    from utils import load_mat, preprocess_features, normalize_adj  # noqa: E402
    from VecGAD import VecGAD  # noqa: E402

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")
    t0 = time.time()
    print(json.dumps({
        "stage": "combo_start",
        "seed": seed,
        "normal_k": normal_k,
        "anom_k": anom_k,
        "device": str(device),
    }, ensure_ascii=False), flush=True)

    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat(args.dataset, args.train_rate, args.val_rate, args=base_args)
    features_np = to_dense_features(args.dataset, features, preprocess_features)
    labels_np = np.asarray(ano_label).reshape(-1).astype(int)
    normal_idx = np.asarray(normal_for_train_idx, dtype=np.int64)
    idx_test = np.asarray(idx_test, dtype=np.int64)
    assert np.sum(labels_np[normal_idx]) == 0, "Data leakage: normal_for_train_idx contains anomalies"

    z = build_descriptor(args.descriptor_mode, features_np, adj, normalize_adj, args.hops, args.rw_steps)
    nm = NormalModel(args.pn_estimator, z, normal_idx, args.pca_components)
    residual = nm.residual()
    normal_refs, anom_refs, meta = select_refs(z, residual, normal_idx, nm, features_np, adj, base_args, normalize_adj)

    token_tensor = build_tokens(features_np, normal_refs, anom_refs)
    model = VecGAD(features_np.shape[1], args.embedding_dim, "prelu", base_args).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    with torch.no_grad():
        emb = encode_tokens_batched(model, token_tensor, device, args.encode_batch_size)
    del token_tensor, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mat, margin = response_matrix_from_embeddings(emb, normal_refs, anom_refs)
    mat_mean = mat.mean(axis=(1, 2))
    arrays = {
        "mat_mean": mat_mean,
        "neg_mat_mean": -mat_mean,
        "margin": margin,
        "neg_margin": -margin,
        "degree": meta["degree"],
        "rejection": meta["rejection"],
    }
    metrics = metric_block(labels_np, idx_test, arrays, base_name="margin")
    row = {
        "seed": int(seed),
        "normal_k": int(normal_k),
        "anom_k": int(anom_k),
        "matrix_shape": list(mat.shape),
        "flatten_dim": int(mat.shape[1] * mat.shape[2]),
        "metrics": metrics,
        "reference_diagnostics": {
            "anom_ref_anom_ratio_diagnostic": float(np.mean(labels_np[anom_refs] == 1)),
            "normal_ref_anom_ratio_diagnostic": float(np.mean(labels_np[normal_refs] == 1)),
            "test_anom_rate": float(np.mean(labels_np[idx_test] == 1)),
        },
        "correlations": {
            "mat_mean_spearman_with_margin": safe_spearman(mat_mean[idx_test], margin[idx_test]),
            "mat_mean_spearman_with_degree": safe_spearman(mat_mean[idx_test], meta["degree"][idx_test]),
            "neg_mat_mean_spearman_with_degree": safe_spearman((-mat_mean)[idx_test], meta["degree"][idx_test]),
        },
        "time_sec": float(time.time() - t0),
    }
    best_name, best_metric = max(metrics.items(), key=lambda kv: kv[1]["auc"])
    row["best"] = {"name": best_name, **best_metric}
    print(json.dumps({
        "stage": "combo_done",
        "seed": seed,
        "normal_k": normal_k,
        "anom_k": anom_k,
        "mat_mean_auc": metrics["mat_mean"]["auc"],
        "neg_mat_mean_auc": metrics["neg_mat_mean"]["auc"],
        "best": best_name,
        "time_sec": row["time_sec"],
    }, ensure_ascii=False), flush=True)
    return row


def summarize(rows):
    by_pair = {}
    for r in rows:
        key = (r["normal_k"], r["anom_k"])
        by_pair.setdefault(key, []).append(r)
    pair_summaries = []
    for (nk, ak), vals in sorted(by_pair.items()):
        mat_auc = [v["metrics"]["mat_mean"]["auc"] for v in vals]
        neg_auc = [v["metrics"]["neg_mat_mean"]["auc"] for v in vals]
        mat_ap = [v["metrics"]["mat_mean"]["ap"] for v in vals]
        neg_ap = [v["metrics"]["neg_mat_mean"]["ap"] for v in vals]
        ref_ratio = [v["reference_diagnostics"]["anom_ref_anom_ratio_diagnostic"] for v in vals]
        degree_corr = [v["correlations"]["mat_mean_spearman_with_degree"] for v in vals]
        pair_summaries.append({
            "normal_k": nk,
            "anom_k": ak,
            "flatten_dim": int(nk * ak),
            "num_seeds": len(vals),
            "mat_mean_auc": mean_std(mat_auc),
            "neg_mat_mean_auc": mean_std(neg_auc),
            "mat_mean_ap": mean_std(mat_ap),
            "neg_mat_mean_ap": mean_std(neg_ap),
            "mat_mean_minus_neg_mat_mean_auc": mean_std([a - b for a, b in zip(mat_auc, neg_auc)]),
            "anom_ref_anom_ratio": mean_std(ref_ratio),
            "mat_mean_degree_spearman": mean_std(degree_corr),
            "negative_orientation_wins": int(sum(1 for v in vals if v["metrics"]["neg_mat_mean"]["auc"] > v["metrics"]["mat_mean"]["auc"])),
        })
    best_mat = max(pair_summaries, key=lambda x: x["mat_mean_auc"]["mean"])
    best_neg = max(pair_summaries, key=lambda x: x["neg_mat_mean_auc"]["mean"])
    best_either = max(pair_summaries, key=lambda x: max(x["mat_mean_auc"]["mean"], x["neg_mat_mean_auc"]["mean"]))
    return {
        "pairs": pair_summaries,
        "best_mat_mean_pair": best_mat,
        "best_neg_mat_mean_pair": best_neg,
        "best_either_orientation_pair": best_either,
        "grid_count": len(pair_summaries),
        "run_count": len(rows),
    }


def make_base_args(args):
    # Recreate the argparse namespace shape expected by Route2.5 helper functions / VecGAD.
    return argparse.Namespace(**vars(args))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(Path.home() / "DualRefGAD"))
    ap.add_argument("--dataset", default="elliptic")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--normal_ks", default="2,4,8,16")
    ap.add_argument("--anom_ks", default="4,8,16,32")
    ap.add_argument("--train_rate", type=float, default=0.05)
    ap.add_argument("--val_rate", type=float, default=0.0)
    ap.add_argument("--descriptor_mode", choices=["hop_attr", "rwse", "hop_attr_rwse"], default="hop_attr_rwse")
    ap.add_argument("--pn_estimator", choices=["diag_gaussian", "pca_residual"], default="diag_gaussian")
    ap.add_argument("--gn_mode", choices=["label_gate", "normal_density", "label_gate_density"], default="label_gate_density")
    ap.add_argument("--ln_mode", default="descriptor_similarity")
    ap.add_argument("--ga_mode", choices=["normal_rejection", "residual_norm", "normal_soft_or"], default="normal_rejection")
    ap.add_argument("--la_mode", choices=["residual_cosine", "descriptor_similarity"], default="residual_cosine")
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
    ap.add_argument("--progress_out", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    start = time.time()
    seeds = parse_int_list(args.seeds)
    normal_ks = parse_int_list(args.normal_ks)
    anom_ks = parse_int_list(args.anom_ks)
    rows = []
    total = len(seeds) * len(normal_ks) * len(anom_ks)
    done = 0
    for nk in normal_ks:
        for ak in anom_ks:
            for seed in seeds:
                base_args = make_base_args(args)
                row = run_one(args, base_args, seed, nk, ak)
                rows.append(row)
                done += 1
                progress = {
                    "status": "running",
                    "done": done,
                    "total": total,
                    "last": {"normal_k": nk, "anom_k": ak, "seed": seed},
                    "elapsed_sec": float(time.time() - start),
                }
                if args.progress_out:
                    Path(args.progress_out).parent.mkdir(parents=True, exist_ok=True)
                    Path(args.progress_out).write_text(json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8")
    result = {
        "status": "finished",
        "probe": "route25_ref_length_mat_mean_scan",
        "protocol": "Frozen encoder; no AE training; normal_k/anom_k grid; 5 seeds; labels diagnostic-only for AUC/AP.",
        "dataset": args.dataset,
        "seeds": seeds,
        "normal_ks": normal_ks,
        "anom_ks": anom_ks,
        "config": vars(args),
        "rows": rows,
        "aggregate": summarize(rows),
        "time_sec": float(time.time() - start),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.progress_out:
        Path(args.progress_out).write_text(json.dumps({
            "status": "finished",
            "done": total,
            "total": total,
            "elapsed_sec": result["time_sec"],
            "out": str(out),
            "best_either_orientation_pair": result["aggregate"]["best_either_orientation_pair"],
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("FINAL " + json.dumps(result["aggregate"]["best_either_orientation_pair"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
