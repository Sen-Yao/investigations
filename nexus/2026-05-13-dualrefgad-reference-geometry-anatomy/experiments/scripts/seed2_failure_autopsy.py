#!/usr/bin/env python3
"""Seed-2 failure autopsy for DualRefGAD route-2 response matrix.

Diagnostic-only, no training. Labels are used only for post-hoc autopsy and node
case selection. The goal is to distinguish whether seed2 degradation comes from
coarse matrix summaries (mat_mean) or unstable reference selection.
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dense_adj_neighbors(adj):
    """Return list-of-neighbor-arrays for scipy sparse / numpy / torch adjacency."""
    if hasattr(adj, "tocsr"):
        csr = adj.tocsr()
        return [csr.indices[csr.indptr[i]:csr.indptr[i + 1]].astype(np.int64) for i in range(csr.shape[0])]
    if torch.is_tensor(adj):
        a = adj.detach().cpu()
        if a.is_sparse:
            a = a.coalesce()
            rows, cols = a.indices().numpy()
            n = int(a.shape[0])
            out = [[] for _ in range(n)]
            for r, c in zip(rows, cols):
                out[int(r)].append(int(c))
            return [np.asarray(sorted(set(x)), dtype=np.int64) for x in out]
        arr = a.numpy()
    else:
        arr = np.asarray(adj)
    return [np.flatnonzero(arr[i]).astype(np.int64) for i in range(arr.shape[0])]


def local_context(node, neighbors, labels, normal_idx_set, max_refs=20):
    n1 = neighbors[node]
    n1_no_self = n1[n1 != node]
    two = set()
    for v in n1_no_self[:2000]:  # guard against pathological dense rows
        two.update(map(int, neighbors[int(v)]))
    two.discard(int(node))
    for v in n1_no_self:
        two.discard(int(v))
    n2 = np.fromiter(two, dtype=np.int64) if two else np.asarray([], dtype=np.int64)

    def ratio_anom(nodes):
        return float(np.mean(labels[nodes] == 1)) if len(nodes) else 0.0

    def ratio_labeled_normal(nodes):
        if not len(nodes):
            return 0.0
        return float(np.mean([int(x) in normal_idx_set for x in nodes]))

    return {
        "degree_1hop": int(len(n1_no_self)),
        "anom_ratio_1hop_diagnostic": ratio_anom(n1_no_self),
        "labeled_normal_ratio_1hop": ratio_labeled_normal(n1_no_self),
        "degree_2hop_exclusive": int(len(n2)),
        "anom_ratio_2hop_exclusive_diagnostic": ratio_anom(n2),
        "labeled_normal_ratio_2hop_exclusive": ratio_labeled_normal(n2),
        "sample_1hop": [int(x) for x in n1_no_self[:max_refs]],
        "sample_2hop_exclusive": [int(x) for x in n2[:max_refs]],
    }


def quantiles(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "q05": float(np.quantile(x, 0.05)),
        "q25": float(np.quantile(x, 0.25)),
        "q50": float(np.quantile(x, 0.50)),
        "q75": float(np.quantile(x, 0.75)),
        "q90": float(np.quantile(x, 0.90)),
        "q95": float(np.quantile(x, 0.95)),
        "max": float(np.max(x)),
        "positive_ratio": float(np.mean(x > 0)),
        "high08_ratio": float(np.mean(x > 0.8)),
    }


def rank_desc(values, idx_test):
    vals = np.asarray(values)
    order = idx_test[np.argsort(-vals[idx_test])]
    rank = np.empty(vals.shape[0], dtype=np.int64)
    rank.fill(-1)
    rank[order] = np.arange(1, len(order) + 1)
    return rank


def top_ref_rows(refs, labels, scores, k=8):
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores)[:k]
    return [
        {"ref": int(refs[j]), "ref_label_diagnostic": int(labels[refs[j]]), "response": float(scores[j]), "slot": int(j)}
        for j in order
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default="/data/linziyao/DualRefGAD")
    ap.add_argument("--dataset", default="elliptic")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--device", type=int, default=0)
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
    ap.add_argument("--select_n", type=int, default=25)
    ap.add_argument("--out", default="experiments/outputs/seed2_failure_autopsy")
    args = ap.parse_args()

    t0 = time.time()
    set_seed(args.seed)
    root = Path(args.project_root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts"))
    os.chdir(str(root))

    from utils import load_mat
    from VecGAD import VecGAD
    from run_training_degradation_diagnosis import (
        to_dense_features,
        build_descriptor,
        NormalModel,
        select_refs,
        apply_ablation,
        reference_purity,
        build_tokens,
        encode_tokens_batched,
    )

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")
    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat(args.dataset, args.train_rate, args.val_rate, args=args)
    features_np = to_dense_features(args.dataset, features)
    labels_np = np.asarray(ano_label).reshape(-1).astype(int)
    idx_test = np.asarray(idx_test, dtype=np.int64)
    normal_idx = np.asarray(normal_for_train_idx, dtype=np.int64)
    normal_idx_set = set(map(int, normal_idx))

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
        emb = encode_tokens_batched(model, token_tensor, device, args.encode_batch_size).detach().cpu().numpy()

    h = emb
    rn_set = emb[normal_refs]
    ra_set = emb[anom_refs]
    rn_mean = rn_set.mean(axis=1)
    ra_mean = ra_set.mean(axis=1)
    u = h - rn_mean
    d = ra_mean - rn_mean
    margin = np.sum((u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)) * (d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)), axis=1)
    u_n = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
    da = ra_set - rn_mean[:, None, :]
    da_n = da / (np.linalg.norm(da, axis=2, keepdims=True) + 1e-12)
    ra_resp = np.einsum("nd,nkd->nk", u_n, da_n)
    un = h[:, None, :] - rn_set
    dn = ra_set[:, None, :, :] - rn_set[:, :, None, :]
    un_n = un / (np.linalg.norm(un, axis=2, keepdims=True) + 1e-12)
    dn_n = dn / (np.linalg.norm(dn, axis=3, keepdims=True) + 1e-12)
    mat = np.einsum("nid,nijd->nij", un_n, dn_n)
    mat_mean = mat.mean(axis=(1, 2))

    margin_rank = rank_desc(margin, idx_test)
    mat_rank = rank_desc(mat_mean, idx_test)
    test_set = set(map(int, idx_test))

    q_margin_high = np.quantile(margin[idx_test], 0.90)
    q_mat_low = np.quantile(mat_mean[idx_test], 0.50)
    margin_high_mat_low_pool = [int(n) for n in idx_test if margin[n] >= q_margin_high and mat_mean[n] <= q_mat_low]
    # prioritize true anomalies: nodes margin finds but mat_mean suppresses
    margin_high_mat_low_pool = sorted(
        margin_high_mat_low_pool,
        key=lambda n: (int(labels_np[n] == 1), margin[n] - mat_mean[n], -mat_rank[n]),
        reverse=True,
    )[: args.select_n]

    n_top_mat = max(1, int(len(idx_test) * 0.05))
    mat_top = idx_test[np.argsort(-mat_mean[idx_test])[:n_top_mat]]
    mat_fp_pool = [int(n) for n in mat_top if labels_np[n] == 0]
    mat_fp_pool = sorted(mat_fp_pool, key=lambda n: (mat_mean[n], margin[n]), reverse=True)[: args.select_n]

    # Also include true anomaly nodes won by margin but not mat, if the quantile gate is too sparse.
    anomaly_margin_top = [int(n) for n in idx_test[np.argsort(-margin[idx_test])] if labels_np[n] == 1]
    anomaly_margin_not_mat = [n for n in anomaly_margin_top if mat_rank[n] > margin_rank[n] * 2][: args.select_n]

    selected = []
    for case, pool in [
        ("margin_high_mat_low", margin_high_mat_low_pool),
        ("mat_mean_false_positive", mat_fp_pool),
        ("anomaly_margin_wins_mat_loses", anomaly_margin_not_mat),
    ]:
        for n in pool:
            selected.append((case, int(n)))
    # preserve first case label, avoid duplicates
    seen = {}
    for case, n in selected:
        seen.setdefault(n, case)
    selected = [(case, n) for n, case in seen.items()]

    neighbors = dense_adj_neighbors(adj)
    rows = []
    for case, n in selected:
        m = mat[n]
        ra = ra_resp[n]
        # matrix column/row summaries identify whether a few refs/anchors dominate.
        col_mean = m.mean(axis=0)
        row_mean = m.mean(axis=1)
        rows.append({
            "case": case,
            "node": int(n),
            "label_diagnostic": int(labels_np[n]),
            "split": "test" if n in test_set else "other",
            "scores": {
                "margin": float(margin[n]),
                "mat_mean": float(mat_mean[n]),
                "margin_rank_test_desc": int(margin_rank[n]),
                "mat_mean_rank_test_desc": int(mat_rank[n]),
                "rank_gap_mat_minus_margin": int(mat_rank[n] - margin_rank[n]),
            },
            "reference_purity": {
                "ra_anom_ratio_diagnostic": float(np.mean(labels_np[anom_refs[n]] == 1)),
                "rn_normal_ratio_diagnostic": float(np.mean(labels_np[normal_refs[n]] == 0)),
                "ra_labels_diagnostic": [int(labels_np[x]) for x in anom_refs[n]],
                "rn_labels_diagnostic": [int(labels_np[x]) for x in normal_refs[n]],
            },
            "response_distribution_shape": {
                "ra_resp": quantiles(ra),
                "mat_full": quantiles(m.reshape(-1)),
                "mat_col_mean_over_ra_refs": quantiles(col_mean),
                "mat_row_mean_over_rn_refs": quantiles(row_mean),
            },
            "top_references": {
                "anom_refs_by_ra_resp": top_ref_rows(anom_refs[n], labels_np, ra, k=min(8, len(ra))),
                "anom_refs_by_mat_col_mean": top_ref_rows(anom_refs[n], labels_np, col_mean, k=min(8, len(col_mean))),
                "normal_refs_by_mat_row_mean": top_ref_rows(normal_refs[n], labels_np, row_mean, k=min(4, len(row_mean))),
                "anom_refs_all": [int(x) for x in anom_refs[n]],
                "normal_refs_all": [int(x) for x in normal_refs[n]],
            },
            "local_graph_context": local_context(int(n), neighbors, labels_np, normal_idx_set),
        })

    def case_summary(case):
        sub = [r for r in rows if r["case"] == case]
        if not sub:
            return {"n": 0}
        return {
            "n": len(sub),
            "label_anom_ratio_diagnostic": float(np.mean([r["label_diagnostic"] for r in sub])),
            "mean_margin": float(np.mean([r["scores"]["margin"] for r in sub])),
            "mean_mat_mean": float(np.mean([r["scores"]["mat_mean"] for r in sub])),
            "mean_ra_anom_ratio_diagnostic": float(np.mean([r["reference_purity"]["ra_anom_ratio_diagnostic"] for r in sub])),
            "mean_degree_1hop": float(np.mean([r["local_graph_context"]["degree_1hop"] for r in sub])),
            "mean_anom_ratio_1hop_diagnostic": float(np.mean([r["local_graph_context"]["anom_ratio_1hop_diagnostic"] for r in sub])),
            "mean_ra_resp_std": float(np.mean([r["response_distribution_shape"]["ra_resp"]["std"] for r in sub])),
            "mean_mat_std": float(np.mean([r["response_distribution_shape"]["mat_full"]["std"] for r in sub])),
        }

    summary = {
        "status": "seed2_failure_autopsy_no_training",
        "protocol": "experiment-runner registered probe; no training; labels diagnostic-only for autopsy selection/interpretation",
        "dataset": args.dataset,
        "seed": args.seed,
        "config": vars(args),
        "runtime_sec": time.time() - t0,
        "global_reference_purity": pur,
        "selection_thresholds": {
            "margin_test_q90": float(q_margin_high),
            "mat_mean_test_q50": float(q_mat_low),
            "mat_top5_percent_count": int(n_top_mat),
        },
        "case_summaries": {
            "margin_high_mat_low": case_summary("margin_high_mat_low"),
            "mat_mean_false_positive": case_summary("mat_mean_false_positive"),
            "anomaly_margin_wins_mat_loses": case_summary("anomaly_margin_wins_mat_loses"),
        },
        "rows": rows,
    }

    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_base.with_suffix(".json")
    md_path = out_base.with_suffix(".md")
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Seed2 Failure Autopsy — margin vs mat_mean",
        "",
        f"Runtime: {summary['runtime_sec']:.1f}s",
        "",
        "## Case summaries",
        "",
        "| case | n | anom ratio | mean margin | mean mat_mean | mean R_a purity | mean 1-hop anomaly ratio | mean mat std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case, s in summary["case_summaries"].items():
        if s.get("n", 0):
            lines.append(f"| {case} | {s['n']} | {s['label_anom_ratio_diagnostic']:.3f} | {s['mean_margin']:.4f} | {s['mean_mat_mean']:.4f} | {s['mean_ra_anom_ratio_diagnostic']:.3f} | {s['mean_anom_ratio_1hop_diagnostic']:.3f} | {s['mean_mat_std']:.4f} |")
        else:
            lines.append(f"| {case} | 0 | - | - | - | - | - | - |")
    lines += ["", "## Selected nodes", ""]
    for r in rows:
        sc = r["scores"]
        pur_r = r["reference_purity"]
        ctx = r["local_graph_context"]
        shp = r["response_distribution_shape"]
        lines += [
            f"### {r['case']} — node {r['node']} (label={r['label_diagnostic']})",
            f"- scores: margin={sc['margin']:.4f} rank={sc['margin_rank_test_desc']}; mat_mean={sc['mat_mean']:.4f} rank={sc['mat_mean_rank_test_desc']}; rank_gap={sc['rank_gap_mat_minus_margin']}",
            f"- R_a purity diagnostic: {pur_r['ra_anom_ratio_diagnostic']:.3f}; R_n normal ratio: {pur_r['rn_normal_ratio_diagnostic']:.3f}",
            f"- response shape: ra_std={shp['ra_resp']['std']:.4f}, ra_q90={shp['ra_resp']['q90']:.4f}, mat_std={shp['mat_full']['std']:.4f}, mat_q90={shp['mat_full']['q90']:.4f}, mat_high08={shp['mat_full']['high08_ratio']:.3f}",
            f"- local graph: degree={ctx['degree_1hop']}, 1hop_anom_ratio={ctx['anom_ratio_1hop_diagnostic']:.3f}, 2hop_anom_ratio={ctx['anom_ratio_2hop_exclusive_diagnostic']:.3f}",
            "- top anomaly refs by matrix column mean: " + ", ".join([f"{x['ref']}(y={x['ref_label_diagnostic']},v={x['response']:.3f})" for x in r['top_references']['anom_refs_by_mat_col_mean'][:5]]),
            "",
        ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "md": str(md_path), "runtime_sec": summary["runtime_sec"], "case_summaries": summary["case_summaries"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
