#!/usr/bin/env python3
"""Reference geometry anatomy for DualRefGAD.

No-training diagnostic. Reuses the existing DualRefGAD reference construction and
frozen/random-init VecGAD token encoder path, then decomposes the margin signal.
Anomaly labels are used only for post-hoc diagnostics.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_auc(y, s):
    try:
        return float(roc_auc_score(y, s)), float(average_precision_score(y, s))
    except Exception:
        return 0.0, 0.0


def safe_spearman(a, b):
    try:
        v = spearmanr(np.asarray(a), np.asarray(b)).correlation
        return float(0.0 if np.isnan(v) else v)
    except Exception:
        return 0.0


def safe_pearson(a, b):
    try:
        v = pearsonr(np.asarray(a), np.asarray(b))[0]
        return float(0.0 if np.isnan(v) else v)
    except Exception:
        return 0.0


def top_indices(scores, frac):
    n = max(1, int(len(scores) * frac))
    return np.argsort(-np.asarray(scores))[:n]


def top_ratio(labels, scores, frac):
    idx = top_indices(scores, frac)
    return float(np.mean(np.asarray(labels)[idx]))


def summarize_by_label(labels, values):
    labels = np.asarray(labels).astype(int)
    values = np.asarray(values, dtype=float)
    out = {}
    for lab, name in [(0, "normal"), (1, "anom")]:
        m = labels == lab
        if np.any(m):
            out[f"{name}_mean"] = float(np.mean(values[m]))
            out[f"{name}_std"] = float(np.std(values[m]))
            out[f"{name}_q05"] = float(np.quantile(values[m], 0.05))
            out[f"{name}_q50"] = float(np.quantile(values[m], 0.50))
            out[f"{name}_q95"] = float(np.quantile(values[m], 0.95))
        else:
            out[f"{name}_mean"] = out[f"{name}_std"] = 0.0
            out[f"{name}_q05"] = out[f"{name}_q50"] = out[f"{name}_q95"] = 0.0
    if np.any(labels == 0) and np.any(labels == 1):
        out["anom_minus_normal_mean"] = out["anom_mean"] - out["normal_mean"]
    else:
        out["anom_minus_normal_mean"] = 0.0
    return out


def metric_block(labels, idx, arrays):
    labels_i = labels[idx]
    out = {}
    for name, values in arrays.items():
        v = np.asarray(values)[idx]
        auc, ap = safe_auc(labels_i, v)
        out[name] = {
            "auc": auc,
            "ap": ap,
            "top1_ratio": top_ratio(labels_i, v, 0.01),
            "top5_ratio": top_ratio(labels_i, v, 0.05),
            **summarize_by_label(labels_i, v),
        }
    return out


def cosine_np(a, b):
    return np.sum(a * b, axis=1) / ((np.linalg.norm(a, axis=1) + 1e-12) * (np.linalg.norm(b, axis=1) + 1e-12))


def hop_slices(descriptor_mode, feat_dim, hops, rw_steps):
    slices = []
    if descriptor_mode in ("hop_attr", "hop_attr_rwse"):
        for h in range(hops + 1):
            slices.append((f"hop{h}_attr", h * feat_dim, (h + 1) * feat_dim))
    if descriptor_mode in ("rwse", "hop_attr_rwse"):
        start = (hops + 1) * feat_dim if descriptor_mode == "hop_attr_rwse" else 0
        slices.append(("rwse", start, start + rw_steps))
    return slices


def mean_ref_distance(matrix, refs, metric="l2"):
    ref_mean = matrix[refs].mean(axis=1)
    diff = matrix - ref_mean
    if metric == "l2":
        return np.linalg.norm(diff, axis=1)
    if metric == "cosine_distance":
        return 1.0 - cosine_np(matrix, ref_mean)
    raise ValueError(metric)


def ref_pair_distance(matrix, normal_refs, anom_refs, metric="l2"):
    rn = matrix[normal_refs].mean(axis=1)
    ra = matrix[anom_refs].mean(axis=1)
    if metric == "l2":
        return np.linalg.norm(ra - rn, axis=1)
    if metric == "cosine_distance":
        return 1.0 - cosine_np(rn, ra)
    raise ValueError(metric)


def subset_reference_purity(normal_refs_subset, anom_refs_subset, labels, row_mask=None):
    """Purity for an arbitrary subset of rows.

    The project helper reference_purity() assumes refs are full-node arrays because
    it indexes anom_refs[labels == 1]. For subset rows, use row_mask to identify
    which subset rows correspond to anomaly targets.
    """
    labels = np.asarray(labels).astype(int)
    out = {
        "normal_ref_normal_ratio": float(np.mean(labels[normal_refs_subset] == 0)) if len(normal_refs_subset) else 0.0,
        "anom_ref_anom_ratio": float(np.mean(labels[anom_refs_subset] == 1)) if len(anom_refs_subset) else 0.0,
    }
    if row_mask is not None and np.any(row_mask):
        out["anom_ref_anom_ratio_on_anom_nodes"] = float(np.mean(labels[anom_refs_subset[row_mask]] == 1))
    else:
        out["anom_ref_anom_ratio_on_anom_nodes"] = 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default="/data/linziyao/DualRefGAD")
    ap.add_argument("--dataset", default="elliptic")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_rate", type=float, default=0.05)
    ap.add_argument("--val_rate", type=float, default=0.0)
    ap.add_argument("--descriptor_mode", choices=["hop_attr", "rwse", "hop_attr_rwse"], default="hop_attr")
    ap.add_argument("--pn_estimator", choices=["diag_gaussian", "pca_residual"], default="pca_residual")
    ap.add_argument("--gn_mode", choices=["label_gate", "normal_density", "label_gate_density"], default="label_gate")
    ap.add_argument("--ln_mode", choices=["descriptor_similarity", "reconstruction_gain"], default="descriptor_similarity")
    ap.add_argument("--ga_mode", choices=["normal_rejection", "residual_norm", "normal_soft_or"], default="normal_soft_or")
    ap.add_argument("--la_mode", choices=["residual_cosine", "descriptor_similarity"], default="descriptor_similarity")
    ap.add_argument("--ablation_mode", choices=["full", "no_ra", "shuffled_ra", "fixed_labeled_normal"], default="full")
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
    ap.add_argument("--sample_rate", type=float, default=0.15)
    ap.add_argument("--mean", type=float, default=0.02)
    ap.add_argument("--var", type=float, default=0.01)
    ap.add_argument("--outlier_beta", type=float, default=0.3)
    ap.add_argument("--ring_R_max", type=float, default=1.0)
    ap.add_argument("--ring_R_min", type=float, default=0.3)
    ap.add_argument("--lambda_rec_tok", type=float, default=1.0)
    ap.add_argument("--lambda_rec_emb", type=float, default=0.1)
    ap.add_argument("--encode_batch_size", type=int, default=512)
    ap.add_argument("--top_frac", type=float, default=0.05)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    t0 = time.time()
    set_seed(args.seed)
    root = Path(args.project_root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts"))
    os.chdir(str(root))

    from utils import load_mat  # noqa
    from VecGAD import VecGAD  # noqa
    from run_training_degradation_diagnosis import (  # noqa
        to_dense_features, build_descriptor, NormalModel, select_refs,
        apply_ablation, reference_purity, build_tokens, encode_tokens_batched,
    )

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")
    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat(args.dataset, args.train_rate, args.val_rate, args=args)
    features_np = to_dense_features(args.dataset, features)
    labels_np = np.asarray(ano_label).reshape(-1).astype(int)
    idx_train = np.asarray(idx_train, dtype=np.int64)
    idx_test = np.asarray(idx_test, dtype=np.int64)
    normal_idx = np.asarray(normal_for_train_idx, dtype=np.int64)
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
        normal_refs_t = torch.as_tensor(normal_refs, dtype=torch.long, device=device)
        anom_refs_t = torch.as_tensor(anom_refs, dtype=torch.long, device=device)
        all_t = torch.arange(len(labels_np), dtype=torch.long, device=device)
        h = emb[all_t]
        rn = emb[normal_refs_t].mean(dim=1)
        ra = emb[anom_refs_t].mean(dim=1)
        u = h - rn
        d = ra - rn
        h_minus_ra = h - ra
        u_norm = F.normalize(u, p=2, dim=1, eps=1e-12)
        d_norm = F.normalize(d, p=2, dim=1, eps=1e-12)
        margin = torch.sum(u_norm * d_norm, dim=1)
        normal_dist_emb = torch.linalg.norm(u, dim=1)
        anomaly_dist_emb = torch.linalg.norm(h_minus_ra, dim=1)
        ref_gap_emb = torch.linalg.norm(d, dim=1)
        orth = torch.linalg.norm(u_norm - margin[:, None] * d_norm, dim=1)

    arrays = {
        "margin": margin.cpu().numpy(),
        "neg_normal_dist_emb": (-normal_dist_emb).cpu().numpy(),
        "normal_dist_emb": normal_dist_emb.cpu().numpy(),
        "neg_anomaly_dist_emb": (-anomaly_dist_emb).cpu().numpy(),
        "anomaly_dist_emb": anomaly_dist_emb.cpu().numpy(),
        "ref_gap_emb": ref_gap_emb.cpu().numpy(),
        "orthogonal_residual_emb": orth.cpu().numpy(),
        "normal_model_rejection": np.asarray(score_meta["rejection"]),
        "residual_norm_descriptor": np.asarray(score_meta["residual_norm"]),
        "ga_score": np.asarray(score_meta["ga"]),
    }

    # Descriptor-space anatomy.
    arrays["normal_dist_desc_l2"] = mean_ref_distance(z, normal_refs, "l2")
    arrays["anom_dist_desc_l2"] = mean_ref_distance(z, anom_refs, "l2")
    arrays["ref_gap_desc_l2"] = ref_pair_distance(z, normal_refs, anom_refs, "l2")
    arrays["normal_dist_rawfeat_l2"] = mean_ref_distance(features_np, normal_refs, "l2")
    arrays["anom_dist_rawfeat_l2"] = mean_ref_distance(features_np, anom_refs, "l2")
    arrays["ref_gap_rawfeat_l2"] = ref_pair_distance(features_np, normal_refs, anom_refs, "l2")

    hop_blocks = {}
    for name, s, e in hop_slices(args.descriptor_mode, features_np.shape[1], args.hops, args.rw_steps):
        if e <= z.shape[1]:
            hop_blocks[f"{name}_normal_dist_l2"] = mean_ref_distance(z[:, s:e], normal_refs, "l2")
            hop_blocks[f"{name}_anom_dist_l2"] = mean_ref_distance(z[:, s:e], anom_refs, "l2")
            hop_blocks[f"{name}_ref_gap_l2"] = ref_pair_distance(z[:, s:e], normal_refs, anom_refs, "l2")
    arrays.update(hop_blocks)

    # Reference purity by node label/split.
    normal_row_mask = labels_np == 0
    anom_row_mask = labels_np == 1
    ref_stats = {
        "overall": pur,
        "normal_nodes": subset_reference_purity(normal_refs[normal_row_mask], anom_refs[normal_row_mask], labels_np, row_mask=anom_row_mask[normal_row_mask]) if np.any(normal_row_mask) else {},
        "anom_nodes": subset_reference_purity(normal_refs[anom_row_mask], anom_refs[anom_row_mask], labels_np, row_mask=anom_row_mask[anom_row_mask]) if np.any(anom_row_mask) else {},
        "test_nodes": subset_reference_purity(normal_refs[idx_test], anom_refs[idx_test], labels_np, row_mask=anom_row_mask[idx_test]),
    }

    # Correlation/decomposition of margin.
    corr_names = [
        "normal_dist_emb", "anomaly_dist_emb", "ref_gap_emb", "orthogonal_residual_emb",
        "normal_dist_desc_l2", "anom_dist_desc_l2", "ref_gap_desc_l2",
        "normal_model_rejection", "residual_norm_descriptor", "ga_score",
    ] + list(hop_blocks.keys())
    correlations = {}
    for name in corr_names:
        if name in arrays:
            correlations[name] = {
                "spearman_with_margin_test": safe_spearman(arrays[name][idx_test], arrays["margin"][idx_test]),
                "pearson_with_margin_test": safe_pearson(arrays[name][idx_test], arrays["margin"][idx_test]),
            }

    # Top-k failure cases on test split.
    test_scores = arrays["margin"][idx_test]
    top_local = top_indices(test_scores, args.top_frac)
    top_nodes = idx_test[top_local]
    top_false_positive = [int(x) for x in top_nodes[labels_np[top_nodes] == 0][:50]]
    missed_anom_order = idx_test[np.argsort(test_scores)]
    missed_anomalies_low_margin = [int(x) for x in missed_anom_order[labels_np[missed_anom_order] == 1][:50]]

    per_node_fields = [
        "node", "label", "split", "margin", "normal_dist_emb", "anomaly_dist_emb", "ref_gap_emb",
        "orthogonal_residual_emb", "normal_dist_desc_l2", "anom_dist_desc_l2", "ref_gap_desc_l2",
        "normal_ref_normal_ratio_node", "anom_ref_anom_ratio_node",
    ]
    csv_lines = [",".join(per_node_fields)]
    split = np.full(len(labels_np), "other", dtype=object)
    split[idx_train] = "train"
    split[idx_test] = "test"
    for i in range(len(labels_np)):
        vals = {
            "node": i,
            "label": int(labels_np[i]),
            "split": split[i],
            "margin": arrays["margin"][i],
            "normal_dist_emb": arrays["normal_dist_emb"][i],
            "anomaly_dist_emb": arrays["anomaly_dist_emb"][i],
            "ref_gap_emb": arrays["ref_gap_emb"][i],
            "orthogonal_residual_emb": arrays["orthogonal_residual_emb"][i],
            "normal_dist_desc_l2": arrays["normal_dist_desc_l2"][i],
            "anom_dist_desc_l2": arrays["anom_dist_desc_l2"][i],
            "ref_gap_desc_l2": arrays["ref_gap_desc_l2"][i],
            "normal_ref_normal_ratio_node": float(np.mean(labels_np[normal_refs[i]] == 0)),
            "anom_ref_anom_ratio_node": float(np.mean(labels_np[anom_refs[i]] == 1)),
        }
        csv_lines.append(",".join(str(vals[f]) for f in per_node_fields))

    summary = {
        "status": "reference_geometry_anatomy_no_training",
        "protocol": "No training; VecGAD token encoder is eval-only/random-init as in existing margin diagnostics; labels diagnostic-only.",
        "dataset": args.dataset,
        "seed": args.seed,
        "config": vars(args),
        "counts": {
            "num_nodes": int(len(labels_np)),
            "num_test": int(len(idx_test)),
            "num_train": int(len(idx_train)),
            "num_labeled_normals": int(len(normal_idx)),
            "num_anomalies_total": int(np.sum(labels_np == 1)),
            "num_anomalies_test": int(np.sum(labels_np[idx_test] == 1)),
        },
        "reference_purity": ref_stats,
        "test_metrics": metric_block(labels_np, idx_test, arrays),
        "train_metrics": metric_block(labels_np, idx_train, arrays),
        "correlations": correlations,
        "topk_failure_cases": {
            "top_frac": args.top_frac,
            "top_margin_anomaly_ratio": top_ratio(labels_np[idx_test], test_scores, args.top_frac),
            "top_margin_false_positive_nodes_first50": top_false_positive,
            "missed_anomaly_low_margin_nodes_first50": missed_anomalies_low_margin,
        },
        "time_sec": time.time() - t0,
    }

    out_base = Path(args.out) if args.out else root / f"outputs/reference_geometry_anatomy/reference_geometry_anatomy_s{args.seed}"
    out_base.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_base.with_suffix(".summary.json")
    csv_path = out_base.with_suffix(".per_node.csv")
    npz_path = out_base.with_suffix(".arrays.npz")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    np.savez_compressed(npz_path, labels=labels_np, idx_test=idx_test, normal_refs=normal_refs, anom_refs=anom_refs, **arrays)
    print(json.dumps({"summary": str(summary_path), "csv": str(csv_path), "npz": str(npz_path), "margin_test": summary["test_metrics"]["margin"], "purity": pur, "time_sec": summary["time_sec"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
