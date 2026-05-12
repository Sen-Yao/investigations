#!/usr/bin/env python3
"""Pseudo anomaly quality audit.

Non-intrusive diagnostic script. Reuses dual-reference construction logic from
training degradation diagnosis, but does not train. It evaluates whether current
pseudo generation creates separable and real-anomaly-aligned positives.
"""
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path.home() / "VoxG"
sys.path.insert(0, str(ROOT))
SRC = ROOT / "nexus/investigations/2026-05-05-elliptic-training-degradation/experiments/scripts"
sys.path.insert(0, str(SRC))

from run_training_degradation_diagnosis import (  # noqa: E402
    set_seed, to_dense_features, build_descriptor, NormalModel, select_refs,
    apply_ablation, reference_purity, build_tokens, encode_tokens_batched,
    scorer, eval_logits, cosine_rows_to_matrix
)
from utils import load_mat
from VecGAD import VecGAD


def l2_normalize(x, axis=1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


def safe_auc(y, s):
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def mean_pairwise_cos(x):
    if len(x) < 2:
        return 1.0
    xn = l2_normalize(x.astype(np.float32))
    sim = xn @ xn.T
    iu = np.triu_indices(len(x), k=1)
    return float(sim[iu].mean())


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
    ap.add_argument("--pseudo_beta", type=float, default=0.2)
    ap.add_argument("--pseudo_strategy", choices=["local_displacement", "ra_mean_positive", "ra_individual_positive"], default="local_displacement")
    ap.add_argument("--pseudo_noise", type=float, default=0.0)
    ap.add_argument("--embedding_dim", type=int, default=256)
    ap.add_argument("--GT_ffn_dim", type=int, default=256)
    ap.add_argument("--GT_dropout", type=float, default=0.4)
    ap.add_argument("--GT_attention_dropout", type=float, default=0.4)
    ap.add_argument("--GT_num_heads", type=int, default=2)
    ap.add_argument("--GT_num_layers", type=int, default=3)
    ap.add_argument("--encode_batch_size", type=int, default=256)
    ap.add_argument("--wandb", type=lambda x: str(x).lower() in ["1","true","yes"], default=False)
    ap.add_argument("--out", default="")
    ap.add_argument("--dry_run", action="store_true")
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
    assert np.sum(labels_np[normal_idx]) == 0, "Data leakage: normal_for_train_idx contains anomalies"

    z = build_descriptor(args.descriptor_mode, features_np, adj, args.hops, args.rw_steps)
    nm = NormalModel(args.pn_estimator, z, normal_idx, args.pca_components)
    residual = nm.residual()
    normal_refs, anom_refs, score_meta = select_refs(z, residual, normal_idx, nm, features_np, adj, args, labels_np)
    normal_refs, anom_refs = apply_ablation(normal_refs, anom_refs, normal_idx, labels_np, args)
    pur = reference_purity(normal_refs, anom_refs, labels_np)
    if args.dry_run:
        print(json.dumps({"config": vars(args), "purity": pur, "n": len(labels_np)}, indent=2, ensure_ascii=False)); return

    token_tensor = build_tokens(features_np, normal_refs, anom_refs)
    model = VecGAD(features_np.shape[1], args.embedding_dim, "prelu", args).to(device)
    model.eval()
    with torch.no_grad():
        emb = encode_tokens_batched(model, token_tensor, device, args.encode_batch_size).detach()
        logits0 = scorer(model, emb).detach().cpu().numpy()
    emb_np = emb.detach().cpu().numpy().astype(np.float32)

    normal_t = torch.tensor(normal_idx, dtype=torch.long, device=device)
    with torch.no_grad():
        nr_t = torch.tensor(normal_refs[normal_idx], dtype=torch.long, device=device)
        ar_t = torch.tensor(anom_refs[normal_idx], dtype=torch.long, device=device)
        pool_rn = emb[nr_t].mean(dim=1)
        pool_ra = emb[ar_t].mean(dim=1)
        direction = F.normalize(pool_ra - pool_rn, dim=1)
        normal_emb = emb[normal_t]
        if args.pseudo_strategy == "local_displacement":
            pseudo_emb = normal_emb + args.pseudo_beta * direction
        elif args.pseudo_strategy == "ra_mean_positive":
            pseudo_emb = pool_ra
        elif args.pseudo_strategy == "ra_individual_positive":
            # Use all individual R_a refs as positives; repeat source normals to match.
            pseudo_emb = emb[ar_t.reshape(-1)]
            normal_emb = normal_emb.repeat_interleave(args.anom_k, dim=0)
            direction = F.normalize(pseudo_emb - normal_emb, dim=1)
        else:
            raise ValueError(args.pseudo_strategy)
        logits_train = scorer(model, torch.cat([normal_emb, pseudo_emb], dim=0)).detach().cpu().numpy()
    pseudo_y = np.r_[np.zeros(len(normal_emb)), np.ones(len(pseudo_emb))]
    pseudo_auc, pseudo_ap = eval_logits(logits_train, pseudo_y, np.arange(len(pseudo_y)))
    normal_logits = logits_train[:len(normal_emb)]; pseudo_logits = logits_train[len(normal_emb):]

    pseudo_np = pseudo_emb.detach().cpu().numpy().astype(np.float32)
    normal_emb_np = normal_emb.detach().cpu().numpy().astype(np.float32)
    direction_np = direction.detach().cpu().numpy().astype(np.float32)

    anom_idx = np.where(labels_np == 1)[0]
    all_normal_idx = np.where(labels_np == 0)[0]
    nbr_a = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(emb_np[anom_idx])
    nbr_n = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(emb_np[all_normal_idx])
    dist_a, idx_a = nbr_a.kneighbors(pseudo_np, return_distance=True)
    dist_n, idx_n = nbr_n.kneighbors(pseudo_np, return_distance=True)
    nearest_anom = anom_idx[idx_a[:,0]]
    real_dir = l2_normalize(emb_np[nearest_anom] - normal_emb_np)
    align = np.sum(l2_normalize(direction_np) * real_dir, axis=1)

    nbr_all = NearestNeighbors(n_neighbors=16, metric="euclidean").fit(emb_np)
    _, knn = nbr_all.kneighbors(pseudo_np, return_distance=True)
    knn_anom_ratio = labels_np[knn].mean(axis=1)

    rn_mean = emb_np[normal_refs].mean(axis=1)
    ra_mean = emb_np[anom_refs].mean(axis=1)
    s_dist_rn = np.linalg.norm(emb_np - rn_mean, axis=1)
    s_dist_ra = np.linalg.norm(emb_np - ra_mean, axis=1)
    s_delta_dist = s_dist_rn - s_dist_ra
    cos_t_ra = np.mean(np.sum(l2_normalize(emb_np[:,None,:]) * l2_normalize(emb_np[anom_refs]), axis=2), axis=1)
    cos_t_rn = np.mean(np.sum(l2_normalize(emb_np[:,None,:]) * l2_normalize(emb_np[normal_refs]), axis=2), axis=1)
    s_cos_delta = cos_t_ra - cos_t_rn
    ref_auc_dist, ref_ap_dist = safe_auc(labels_np[idx_test], s_delta_dist[idx_test])
    ref_auc_cos, ref_ap_cos = safe_auc(labels_np[idx_test], s_cos_delta[idx_test])
    epoch0_auc, epoch0_ap = safe_auc(labels_np[idx_test], logits0[idx_test])

    # R_a diversity on a bounded subset for speed
    rng = np.random.default_rng(args.seed)
    sample_nodes = rng.choice(np.arange(len(labels_np)), size=min(2000, len(labels_np)), replace=False)
    ra_pair_cos = [mean_pairwise_cos(emb_np[anom_refs[i]]) for i in sample_nodes]
    rn_pair_cos = [mean_pairwise_cos(emb_np[normal_refs[i]]) for i in sample_nodes]
    mean_dir = l2_normalize((ra_mean - rn_mean).astype(np.float32))
    indiv_cos = []
    for i in sample_nodes[:1000]:
        dirs = l2_normalize((emb_np[anom_refs[i]] - rn_mean[i]).astype(np.float32))
        indiv_cos.extend((dirs @ mean_dir[i]).tolist())

    row = {
        "seed": args.seed,
        "pseudo_strategy": args.pseudo_strategy,
        "pseudo_beta": args.pseudo_beta,
        "pseudo_auc": pseudo_auc,
        "pseudo_ap": pseudo_ap,
        "pseudo_margin": float(pseudo_logits.mean() - normal_logits.mean()),
        "normal_pseudo_l2": float(np.linalg.norm(pseudo_np - normal_emb_np, axis=1).mean()),
        "normal_pseudo_cos": float(np.sum(l2_normalize(pseudo_np)*l2_normalize(normal_emb_np), axis=1).mean()),
        "pseudo_dist_to_anomaly": float(dist_a[:,0].mean()),
        "pseudo_dist_to_normal": float(dist_n[:,0].mean()),
        "pseudo_normal_anom_dist_ratio": float((dist_n[:,0] / (dist_a[:,0] + 1e-12)).mean()),
        "pseudo_knn16_anom_ratio": float(knn_anom_ratio.mean()),
        "pseudo_real_direction_cos": float(align.mean()),
        "pseudo_real_direction_cos_std": float(align.std()),
        "ra_pairwise_cos": float(np.mean(ra_pair_cos)),
        "rn_pairwise_cos": float(np.mean(rn_pair_cos)),
        "mean_dir_individual_dir_cos": float(np.mean(indiv_cos)),
        "mean_dir_norm": float(np.linalg.norm(ra_mean - rn_mean, axis=1).mean()),
        "ref_delta_dist_auc": ref_auc_dist,
        "ref_delta_dist_ap": ref_ap_dist,
        "ref_cos_delta_auc": ref_auc_cos,
        "ref_cos_delta_ap": ref_ap_cos,
        "epoch0_auc": epoch0_auc,
        "epoch0_ap": epoch0_ap,
        "normal_ref_normal_ratio": pur["normal_ref_normal_ratio"],
        "anom_ref_anom_ratio": pur["anom_ref_anom_ratio"],
        "anom_ref_anom_ratio_on_anom_nodes": pur["anom_ref_anom_ratio_on_anom_nodes"],
        "time_sec": time.time() - t0,
    }
    print(json.dumps(row, ensure_ascii=False), flush=True)
    if args.wandb:
        import wandb
        run = wandb.init(project="VoxG", entity="HCCS", config=vars(args), name=f"pseudo_quality_{args.dataset}_{args.pseudo_strategy}_b{args.pseudo_beta}_s{args.seed}")
        wandb.log(row)
        wandb.summary.update(row)
        run.finish()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(row, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
