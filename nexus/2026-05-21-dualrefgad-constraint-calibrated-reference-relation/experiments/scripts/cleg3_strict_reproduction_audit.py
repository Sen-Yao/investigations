#!/usr/bin/env python3
"""Strict C-LEG3 margin/mat_mean reproducibility audit.

Purpose:
- Freeze the C-LEG3 / old_exact_080_regime formula and reference construction.
- Remove the multi-threaded global-RNG split drift observed between Step-1/Step-2/LSD.
- Run seeds sequentially by default and force data_split_seed=seed before load_mat.
- Save per-seed split fingerprints, effective config, margin/mat_mean metrics, and
  reference purity diagnostics so future reports can distinguish formula drift from
  split/protocol drift.

Labels are diagnostic-only for AUC/AP and reference-purity audit. No training.
"""
import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
# Prefer same-investigation scripts first, then Route2.5 source scripts.
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parents[2] / "2026-05-19-dualrefgad-route25-matrix-autoencoder" / "experiments" / "scripts"))

from route25_leg3_response_matrix_decomposition_probe import (  # noqa: E402
    BASE_DEFAULTS,
    VARIANTS,
    build_decomposition_arrays,
    parse_ints,
)
from route25_matrix_autoencoder_probe import (  # noqa: E402
    NormalModel,
    build_descriptor,
    build_tokens,
    encode_tokens_batched,
    metric_block,
    response_matrix_from_embeddings,
    select_refs,
    set_seed,
)


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
        "train_anom_count": int(np.sum(labels_np[idx_train] == 1)),
        "train_normal_count": int(np.sum(labels_np[idx_train] == 0)),
        "test_anom_count": int(np.sum(labels_np[idx_test] == 1)),
        "test_normal_count": int(np.sum(labels_np[idx_test] == 0)),
        "normal_for_train_count": int(len(normal_idx)),
        "normal_for_train_anom_count": int(np.sum(labels_np[normal_idx] == 1)),
    }


def to_dense_features(dataset, features, preprocess_features):
    if dataset in ["Amazon", "tf_finace", "t_finance", "reddit", "elliptic"]:
        features, _ = preprocess_features(features)
        return np.asarray(features, dtype=np.float32)
    return np.asarray(features.todense(), dtype=np.float32)


def run_one(cli_args, variant, seed, device):
    # Hard reset all known RNGs before each seed. The critical additional lock is
    # data_split_seed=seed, because load_mat uses global random.shuffle unless this
    # attribute exists.
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = copy.deepcopy(BASE_DEFAULTS)
    cfg.update(VARIANTS[variant]["changes"])
    cfg.update(vars(cli_args))
    cfg["variant"] = variant
    cfg["device"] = int(device)
    cfg["seed"] = int(seed)
    cfg["data_split_seed"] = int(seed)
    cfg["strict_sequential"] = True
    cfg["audit_formula"] = {
        "margin": "response_matrix_from_embeddings(...)[1] evaluated directly on idx_test",
        "mat_mean": "build_decomposition_arrays(mat, margin, ...)[\"mat_mean\"] = mat.mean(axis=(1,2))",
    }
    v_args = argparse.Namespace(**cfg)

    root = Path(v_args.project_root).expanduser().resolve()
    sys.path.insert(0, str(root))
    os.chdir(str(root))
    from utils import load_mat, preprocess_features, normalize_adj  # noqa: E402
    from VecGAD import VecGAD  # noqa: E402

    device_obj = torch.device(f"cuda:{device}" if torch.cuda.is_available() and int(device) >= 0 else "cpu")
    print(json.dumps({"stage": "seed_start", "variant": variant, "seed": seed, "device": str(device_obj), "data_split_seed": seed}, ensure_ascii=False), flush=True)
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
    arrays, _ = build_decomposition_arrays(mat, margin, meta, normal_idx, seed, parse_ints(v_args.pca_dims))
    mat_mean = arrays["mat_mean"]
    direct_mat_mean = mat.mean(axis=(1, 2))
    max_abs_diff = float(np.max(np.abs(np.asarray(mat_mean) - np.asarray(direct_mat_mean))))
    metrics = metric_block(labels_np, idx_test, {"margin": margin, "mat_mean": mat_mean}, base_name="margin")
    row = {
        "variant": variant,
        "report_codename": VARIANTS[variant]["report_codename"],
        "seed": int(seed),
        "device": int(device),
        "effective_config": cfg,
        "split_fingerprint": fp,
        "formula_check": {
            "mat_mean_equals_direct_matrix_mean_max_abs_diff": max_abs_diff,
            "margin_score_shape": list(np.asarray(margin).shape),
            "mat_shape": list(np.asarray(mat).shape),
        },
        "metrics": metrics,
        "reference_global": {
            "normal_ref_anom_ratio_diagnostic": float(np.mean(labels_np[normal_refs] == 1)),
            "anom_ref_anom_ratio_diagnostic": float(np.mean(labels_np[anom_refs] == 1)),
            "normal_ref_dup_rate": float(np.mean([1.0 - len(set(map(int, r))) / max(1, len(r)) for r in normal_refs])),
            "anom_ref_dup_rate": float(np.mean([1.0 - len(set(map(int, r))) / max(1, len(r)) for r in anom_refs])),
        },
    }
    print(json.dumps({"stage": "seed_done", "seed": seed, "margin_auc": metrics["margin"]["auc"], "mat_mean_auc": metrics["mat_mean"]["auc"], "split": fp}, ensure_ascii=False), flush=True)
    return row


def summarize(rows):
    return {
        "n_rows": len(rows),
        "margin_auc": mean_std([r["metrics"]["margin"]["auc"] for r in rows]),
        "margin_ap": mean_std([r["metrics"]["margin"]["ap"] for r in rows]),
        "mat_mean_auc": mean_std([r["metrics"]["mat_mean"]["auc"] for r in rows]),
        "mat_mean_ap": mean_std([r["metrics"]["mat_mean"]["ap"] for r in rows]),
        "mat_minus_margin_auc_delta": mean_std([r["metrics"]["mat_mean"]["auc"] - r["metrics"]["margin"]["auc"] for r in rows]),
        "mat_mean_formula_max_abs_diff": mean_std([r["formula_check"]["mat_mean_equals_direct_matrix_mean_max_abs_diff"] for r in rows]),
        "split_fingerprints": {str(r["seed"]): r["split_fingerprint"] for r in rows},
    }


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
    ap.add_argument("--top_k", type=int, default=96)
    ap.add_argument("--n_bins", type=int, default=3)
    ap.add_argument("--pca_dims", default="2,4,8,12,24")
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress_out", default="")
    args = ap.parse_args()

    seeds = parse_ints(args.seeds)
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    devices = parse_ints(args.devices) if args.devices.strip() else [int(args.device)]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}; available={list(VARIANTS)}")

    start = time.time()
    rows_by_variant = {v: [] for v in variants}
    errors = []
    total = len(seeds) * len(variants)
    done = 0

    def snapshot(status="running"):
        atomic_write_json(args.progress_out, {
            "status": status,
            "probe": "cleg3_strict_reproduction_audit",
            "done": done,
            "total": total,
            "variants": variants,
            "seeds": seeds,
            "sequential": True,
            "partial": {v: summarize(rows_by_variant[v]) if rows_by_variant[v] else {"n_rows": 0} for v in variants},
            "errors": errors[-5:],
            "elapsed_sec": time.time() - start,
        })

    snapshot("running")
    for variant in variants:
        for i, seed in enumerate(seeds):
            device = devices[i % len(devices)]
            try:
                row = run_one(args, variant, seed, device)
                rows_by_variant[variant].append(row)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(tb, flush=True)
                errors.append({"variant": variant, "seed": seed, "device": device, "error": repr(e), "traceback": tb[-4000:]})
            finally:
                done += 1
                snapshot("running")

    variant_summaries = []
    for variant in variants:
        rows = sorted(rows_by_variant[variant], key=lambda r: r["seed"])
        variant_summaries.append({
            "variant": variant,
            "report_codename": VARIANTS[variant]["report_codename"],
            "definition": VARIANTS[variant]["definition"],
            "changes_from_base": VARIANTS[variant]["changes"],
            "aggregate": summarize(rows),
            "rows": rows,
        })
    payload = {
        "status": "finished" if not errors else "finished_with_errors",
        "probe": "cleg3_strict_reproduction_audit",
        "protocol": {
            "type": "runner-registered pure probe; sequential strict reproduction audit; no training",
            "critical_fix": "force data_split_seed=seed and run seeds sequentially to avoid global random.shuffle cross-thread drift in load_mat",
            "label_boundary": "labels diagnostic-only for AUC/AP and reference-purity audit",
            "formula_boundary": "margin from response_matrix_from_embeddings; mat_mean = mat.mean(axis=(1,2)) via build_decomposition_arrays and direct equality check",
        },
        "dataset": args.dataset,
        "seeds": seeds,
        "variants": variants,
        "devices": devices,
        "config": vars(args),
        "variant_summaries": variant_summaries,
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
