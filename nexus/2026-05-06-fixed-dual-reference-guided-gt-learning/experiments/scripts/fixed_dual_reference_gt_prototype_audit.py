#!/usr/bin/env python3
"""Frozen fixed dual-reference GT normal prototype audit.

Non-intrusive diagnostic script. It reuses the existing fixed dual-reference
construction and GT encoder, but does not train. It evaluates whether 5% labeled
normal nodes can calibrate anomaly judgment in frozen GT embedding space.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path.home() / "VoxG"
sys.path.insert(0, str(ROOT))
SRC = ROOT / "nexus/investigations/2026-05-05-elliptic-training-degradation/experiments/scripts"
sys.path.insert(0, str(SRC))

from run_training_degradation_diagnosis import (  # noqa: E402
    set_seed, to_dense_features, build_descriptor, NormalModel, select_refs,
    apply_ablation, reference_purity, build_tokens, encode_tokens_batched,
    scorer, cosine_rows_to_matrix
)
from utils import load_mat
from VecGAD import VecGAD


def safe_auc(y, s):
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def zscore_by_train_normal(scores, train_idx):
    mu = float(np.mean(scores[train_idx]))
    sd = float(np.std(scores[train_idx]) + 1e-12)
    return (scores - mu) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="elliptic")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_rate", type=float, default=0.05)
    ap.add_argument("--descriptor_mode", choices=["hop_attr","rwse","hop_attr_rwse"], default="hop_attr")
    ap.add_argument("--pn_estimator", choices=["diag_gaussian","pca_residual"], default="pca_residual")
    ap.add_argument("--gn_mode", choices=["label_gate","normal_density","label_gate_density"], default="label_gate")
    ap.add_argument("--ln_mode", choices=["descriptor_similarity","reconstruction_gain"], default="descriptor_similarity")
    ap.add_argument("--ga_mode", choices=["normal_rejection","residual_norm","normal_soft_or"], default="normal_soft_or")
    ap.add_argument("--la_mode", choices=["residual_cosine","descriptor_similarity"], default="descriptor_similarity")
    ap.add_argument("--reference_mode", default="dual_reference")
    ap.add_argument("--ablation_mode", choices=["full","no_ra","shuffled_ra","fixed_labeled_normal"], default="full")
    ap.add_argument("--normal_k", type=int, default=4)
    ap.add_argument("--anom_k", type=int, default=16)
    ap.add_argument("--pp_k", type=int, default=6)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--rw_steps", type=int, default=8)
    ap.add_argument("--pca_components", type=int, default=32)
    ap.add_argument("--embedding_dim", type=int, default=256)
    ap.add_argument("--GT_ffn_dim", type=int, default=256)
    ap.add_argument("--GT_dropout", type=float, default=0.4)
    ap.add_argument("--GT_attention_dropout", type=float, default=0.4)
    ap.add_argument("--GT_num_heads", type=int, default=2)
    ap.add_argument("--GT_num_layers", type=int, default=3)
    ap.add_argument("--encode_batch_size", type=int, default=256)
    ap.add_argument("--proto_k", type=int, default=4)
    ap.add_argument("--knn_k", type=int, default=16)
    ap.add_argument("--wandb", type=lambda x: str(x).lower() in ["1","true","yes"], default=False)
    ap.add_argument("--out", default="")
    # Compatibility args expected by VecGAD internals
    ap.add_argument("--sample_rate", type=float, default=0.15)
    ap.add_argument("--mean", type=float, default=0.02)
    ap.add_argument("--var", type=float, default=0.01)
    ap.add_argument("--outlier_beta", type=float, default=0.3)
    ap.add_argument("--ring_R_max", type=float, default=1.0)
    ap.add_argument("--ring_R_min", type=float, default=0.3)
    ap.add_argument("--lambda_rec_tok", type=float, default=1.0)
    ap.add_argument("--lambda_rec_emb", type=float, default=0.1)
    args = ap.parse_args()

    t0 = time.time()
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")
    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat(args.dataset, args.train_rate, 0.1, args=args)
    features_np = to_dense_features(args.dataset, features)
    labels_np = np.asarray(ano_label).reshape(-1).astype(int)
    normal_idx = np.asarray(normal_for_train_idx, dtype=int)
    idx_test = np.asarray(idx_test, dtype=int)
    assert np.sum(labels_np[normal_idx]) == 0, "Data leakage: normal_for_train_idx contains anomalies"

    z = build_descriptor(args.descriptor_mode, features_np, adj, args.hops, args.rw_steps)
    nm = NormalModel(args.pn_estimator, z, normal_idx, args.pca_components)
    residual = nm.residual()
    normal_refs, anom_refs, score_meta = select_refs(z, residual, normal_idx, nm, features_np, adj, args, labels_np)
    normal_refs, anom_refs = apply_ablation(normal_refs, anom_refs, normal_idx, labels_np, args)
    pur = reference_purity(normal_refs, anom_refs, labels_np)

    token_tensor = build_tokens(features_np, normal_refs, anom_refs)
    model = VecGAD(features_np.shape[1], args.embedding_dim, "prelu", args).to(device)
    model.eval()
    with torch.no_grad():
        emb = encode_tokens_batched(model, token_tensor, device, args.encode_batch_size).detach()
        logits0 = scorer(model, emb).detach().cpu().numpy()
    emb_np = emb.detach().cpu().numpy().astype(np.float32)
    normal_emb = emb_np[normal_idx]

    # S1: single normal center
    center = normal_emb.mean(axis=0, keepdims=True)
    s_single = np.linalg.norm(emb_np - center, axis=1)

    # S2: multi-prototype kmeans over labeled normals
    k_eff = min(args.proto_k, len(normal_idx))
    km = KMeans(n_clusters=k_eff, random_state=args.seed, n_init=10).fit(normal_emb)
    centers = km.cluster_centers_.astype(np.float32)
    s_multi = np.min(np.linalg.norm(emb_np[:, None, :] - centers[None, :, :], axis=2), axis=1)

    # S3: local kNN to labeled normals
    knn_eff = min(args.knn_k, len(normal_idx))
    nbr = NearestNeighbors(n_neighbors=knn_eff, metric="euclidean").fit(normal_emb)
    d_knn, _ = nbr.kneighbors(emb_np, return_distance=True)
    s_knn = d_knn.mean(axis=1)

    # S4: fixed reference profile probes only
    rn_mean = emb_np[normal_refs].mean(axis=1)
    ra_mean = emb_np[anom_refs].mean(axis=1)
    s_dist_rn = np.linalg.norm(emb_np - rn_mean, axis=1)
    s_dist_ra = np.linalg.norm(emb_np - ra_mean, axis=1)
    s_delta_dist = s_dist_rn - s_dist_ra
    # normal-calibrated profile fusion: no learned weights, just zscore probes by labeled normals
    s_profile_fusion = zscore_by_train_normal(s_single, normal_idx) + zscore_by_train_normal(s_delta_dist, normal_idx)

    metrics = {}
    for name, score in [
        ("single_center_l2", s_single),
        ("multi_proto_l2", s_multi),
        ("local_knn_normal_l2", s_knn),
        ("ref_delta_dist", s_delta_dist),
        ("profile_fusion_probe", s_profile_fusion),
        ("epoch0_random_head", logits0),
    ]:
        auc, apv = safe_auc(labels_np[idx_test], score[idx_test])
        metrics[f"{name}_auc"] = auc
        metrics[f"{name}_ap"] = apv
        metrics[f"{name}_normal_mean"] = float(np.mean(score[normal_idx]))
        metrics[f"{name}_test_mean"] = float(np.mean(score[idx_test]))

    row = {
        "seed": args.seed,
        "dataset": args.dataset,
        "train_rate": args.train_rate,
        "proto_k": args.proto_k,
        "proto_k_eff": k_eff,
        "knn_k": args.knn_k,
        "knn_k_eff": knn_eff,
        "n_nodes": int(len(labels_np)),
        "n_labeled_normal": int(len(normal_idx)),
        "test_anomaly_rate": float(labels_np[idx_test].mean()),
        "normal_ref_normal_ratio": pur["normal_ref_normal_ratio"],
        "anom_ref_anom_ratio": pur["anom_ref_anom_ratio"],
        "anom_ref_anom_ratio_on_anom_nodes": pur["anom_ref_anom_ratio_on_anom_nodes"],
        "time_sec": time.time() - t0,
    }
    row.update(metrics)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    if args.wandb:
        import wandb
        run = wandb.init(project="VoxG", entity="HCCS", config=vars(args), name=f"fixed_ref_gt_proto_{args.dataset}_k{args.proto_k}_s{args.seed}")
        wandb.log(row)
        wandb.summary.update(row)
        run.finish()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(row, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
