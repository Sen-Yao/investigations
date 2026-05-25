#!/usr/bin/env python3
"""DualRefGAD R2-bounded + self-stop probe.

This runner-registered pure probe tests whether the previous R2_ref_drop_only
signal survives when score scale is bounded and checkpoint selection is based on
label-free self-stop criteria. Anomaly labels are used only for report-only
AUC/AP diagnostics after checkpoints are chosen.
"""
import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))
from utils import load_mat, normalize_adj, preprocess_features  # noqa: E402
from VecGAD import VecGAD  # noqa: E402


def atomic_write_json(path, payload):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_dense_features(dataset, features):
    if dataset in ["Amazon", "tf_finace", "reddit", "elliptic"]:
        features, _ = preprocess_features(features)
        return np.asarray(features, dtype=np.float32)
    return np.asarray(features.todense(), dtype=np.float32)


def l2_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def rank_percentile(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x)
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x))
    return r / max(1, len(x) - 1)


def build_hop_attr(features, adj, hops=2):
    adj_norm = normalize_adj(adj)
    x = features.astype(np.float32)
    outs = [x]
    cur = x
    for _ in range(hops):
        cur = np.asarray(adj_norm.dot(cur), dtype=np.float32)
        outs.append(cur)
    return np.concatenate(outs, axis=1).astype(np.float32)


def rwse(adj, steps=8):
    csr = adj.tocsr().astype(np.float64)
    deg = np.asarray(csr.sum(axis=1)).reshape(-1)
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    P = sp.diags(inv).dot(csr).tocsr()
    cur = P.copy()
    feats = []
    for k in range(1, steps + 1):
        feats.append(cur.diagonal().astype(np.float64))
        if k < steps:
            cur = cur.dot(P).tocsr()
    return np.stack(feats, axis=1).astype(np.float32)


def build_descriptor(mode, features, adj, hops=2, rw_steps=8):
    if mode == "hop_attr":
        return build_hop_attr(features, adj, hops)
    if mode == "rwse":
        return rwse(adj, rw_steps)
    if mode == "hop_attr_rwse":
        return np.concatenate([build_hop_attr(features, adj, hops), rwse(adj, rw_steps)], axis=1).astype(np.float32)
    raise ValueError(mode)


class NormalModel:
    def __init__(self, estimator, z, normal_idx, pca_components=32):
        self.estimator = estimator
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.zs = self.scaler.fit_transform(z[normal_idx])
        self.z_all = self.scaler.transform(z)
        self.mu = self.zs.mean(axis=0, keepdims=True)
        self.std = self.zs.std(axis=0, keepdims=True) + 1e-6
        self.pca = None
        if estimator == "pca_residual":
            ncomp = int(min(pca_components, self.zs.shape[0] - 1, self.zs.shape[1]))
            if ncomp > 0:
                self.pca = PCA(n_components=ncomp, svd_solver="randomized", random_state=0)
                self.pca.fit(self.zs)
        elif estimator != "diag_gaussian":
            raise ValueError(estimator)

    def rejection(self):
        if self.estimator == "diag_gaussian" or self.pca is None:
            return np.mean(((self.z_all - self.mu) / self.std) ** 2, axis=1)
        rec = self.pca.inverse_transform(self.pca.transform(self.z_all))
        return np.mean((self.z_all - rec) ** 2, axis=1)

    def density_score(self):
        return -self.rejection()


def cosine_rows_to_matrix(a, b, block=1024):
    an = l2_rows(a.astype(np.float32))
    bn = l2_rows(b.astype(np.float32))
    outs = []
    for st in range(0, an.shape[0], block):
        outs.append(an[st:st + block] @ bn.T)
    return np.vstack(outs)


def select_normal_refs(z, normal_idx, nm, args):
    n = z.shape[0]
    normal_pool = np.asarray(normal_idx)
    density = rank_percentile(nm.density_score()).astype(np.float32)
    gn = np.full(n, -1e9, dtype=np.float32)
    gn[normal_pool] = density[normal_pool]
    sim_n = cosine_rows_to_matrix(z, z[normal_pool])
    scores = sim_n + gn[normal_pool][None, :]
    return normal_pool[np.argsort(-scores, axis=1)[:, :args.normal_k]].astype(np.int64)


def build_tokens(features, normal_refs):
    toks = []
    for i in range(features.shape[0]):
        toks.append(np.concatenate([features[i:i + 1], features[normal_refs[i]]], axis=0))
    return torch.from_numpy(np.stack(toks).astype(np.float32))


def encode_tokens_batched(model, token_tensor_cpu, device, batch_size):
    chunks = []
    for st in range(0, token_tensor_cpu.shape[0], batch_size):
        chunks.append(model.TransformerEncoder(token_tensor_cpu[st:st + batch_size].to(device, non_blocking=True)).squeeze(0))
    return torch.cat(chunks, dim=0)


def raw_scorer(model, emb):
    return model.fc3(model.act(model.fc2(model.act(model.fc1(emb))))).squeeze(-1)


def bounded_score(raw, score_bound):
    return float(score_bound) * torch.tanh(raw / float(score_bound))


def safe_auc_ap(scores, labels, idx):
    try:
        return float(roc_auc_score(labels[idx], scores[idx])), float(average_precision_score(labels[idx], scores[idx]))
    except Exception:
        return None, None


def safe_spearman(a, b):
    try:
        c = spearmanr(np.asarray(a), np.asarray(b)).correlation
        return float(0.0 if np.isnan(c) else c)
    except Exception:
        return 0.0


def self_stop_health(row, args):
    # Label-free health: prefer reference-dropout stability while rejecting
    # trivial/collapsed and saturated score regimes. Uses all-node and
    # train-normal score distribution only; no val/test labels.
    std_floor_penalty = max(0.0, float(args.min_score_std) - row["all_score_std"])
    return float(
        - row["dropout_score_mse_normal"]
        - args.saturation_penalty * row["saturation_frac_all"]
        - args.mean_drift_penalty * abs(row["normal_score_mean_delta"])
        - args.collapse_penalty * std_floor_penalty
    )


def run_seed(base_args, seed, device_id):
    args = argparse.Namespace(**vars(base_args))
    args.seed = int(seed)
    args.device = int(device_id)
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")

    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, *_rest = load_mat(args.dataset, args.train_rate, 0.1, args=args)
    normal_idx = np.asarray(_rest[-2], dtype=int)
    features_np = to_dense_features(args.dataset, features)
    labels_np = np.asarray(ano_label).reshape(-1).astype(int)
    assert np.sum(labels_np[normal_idx]) == 0, "Data leakage: normal_for_train_idx contains anomalies"

    z = build_descriptor(args.descriptor_mode, features_np, adj, args.hops, args.rw_steps)
    nm = NormalModel(args.pn_estimator, z, normal_idx, args.pca_components)
    normal_refs = select_normal_refs(z, normal_idx, nm, args)
    token_tensor = build_tokens(features_np, normal_refs)
    ref_density = rank_percentile(nm.density_score())
    ref_density_mean = ref_density[normal_refs].mean(axis=1)

    model = VecGAD(features_np.shape[1], args.embedding_dim, "prelu", args).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    normal_t = torch.tensor(normal_idx, dtype=torch.long, device=device)
    rows = []
    best_report_only = {"val_auc": -1.0, "val_ap": -1.0, "test_auc": -1.0, "test_ap": -1.0, "epoch": -1}
    best_self = {"health": -1e18, "epoch": -1, "row": None}
    no_improve = 0
    stop_reason = "max_epoch"
    baseline_normal_mean = None
    started = time.time()

    def encode_with_ref_dropout():
        tok = token_tensor.clone()
        if args.ref_dropout_rate > 0:
            ref_mask = (torch.rand(tok.shape[0], tok.shape[1] - 1) < args.ref_dropout_rate)
            tok[:, 1:, :][ref_mask] = tok[:, 0:1, :].expand(-1, tok.shape[1] - 1, -1)[ref_mask]
        return encode_tokens_batched(model, tok, device, args.encode_batch_size)

    for epoch in range(args.num_epoch + 1):
        model.train()
        opt.zero_grad()
        emb_full = encode_tokens_batched(model, token_tensor, device, args.encode_batch_size)
        raw_full = raw_scorer(model, emb_full)
        score_full = bounded_score(raw_full, args.score_bound)
        emb_drop = encode_with_ref_dropout()
        raw_drop = raw_scorer(model, emb_drop)
        score_drop = bounded_score(raw_drop, args.score_bound)
        l_ref_drop = F.mse_loss(score_drop[normal_t], score_full[normal_t].detach())
        loss = l_ref_drop
        loss.backward()
        opt.step()

        if epoch % args.eval_every == 0 or epoch == args.num_epoch:
            model.eval()
            with torch.no_grad():
                emb_eval = encode_tokens_batched(model, token_tensor, device, args.encode_batch_size)
                raw_eval_t = raw_scorer(model, emb_eval)
                score_eval_t = bounded_score(raw_eval_t, args.score_bound)
                emb_drop_eval = encode_with_ref_dropout()
                raw_drop_eval_t = raw_scorer(model, emb_drop_eval)
                score_drop_eval_t = bounded_score(raw_drop_eval_t, args.score_bound)
            raw_eval = raw_eval_t.detach().cpu().numpy()
            score_eval = score_eval_t.detach().cpu().numpy()
            score_drop_eval = score_drop_eval_t.detach().cpu().numpy()
            val_auc, val_ap = safe_auc_ap(score_eval, labels_np, idx_val)
            test_auc, test_ap = safe_auc_ap(score_eval, labels_np, idx_test)
            normal_mean = float(score_eval[normal_idx].mean())
            if baseline_normal_mean is None:
                baseline_normal_mean = normal_mean
            row = {
                "variant": "R2_bounded_refdrop_selfstop",
                "seed": int(args.seed),
                "device": int(args.device),
                "epoch": int(epoch),
                "loss": float(loss.item()),
                "L_ref_drop": float(l_ref_drop.item()),
                "normal_score_mean": normal_mean,
                "normal_score_mean_delta": float(normal_mean - baseline_normal_mean),
                "all_score_mean": float(score_eval.mean()),
                "all_score_std": float(score_eval.std()),
                "raw_score_std": float(raw_eval.std()),
                "score_min": float(score_eval.min()),
                "score_max": float(score_eval.max()),
                "saturation_frac_all": float(np.mean(np.abs(score_eval) > args.saturation_frac_threshold * args.score_bound)),
                "dropout_score_mse_all": float(np.mean((score_drop_eval - score_eval) ** 2)),
                "dropout_score_mse_normal": float(np.mean((score_drop_eval[normal_idx] - score_eval[normal_idx]) ** 2)),
                "spearman_score_ref_density": safe_spearman(score_eval, ref_density_mean),
                "val_auc_report_only": val_auc,
                "val_ap_report_only": val_ap,
                "test_auc_report_only": test_auc,
                "test_ap_report_only": test_ap,
            }
            row["self_stop_health"] = self_stop_health(row, args)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

            if val_auc is not None and val_ap is not None and val_auc + val_ap > best_report_only["val_auc"] + best_report_only["val_ap"]:
                best_report_only = {"val_auc": val_auc, "val_ap": val_ap, "test_auc": test_auc, "test_ap": test_ap, "epoch": epoch}

            if epoch >= args.self_stop_min_epoch:
                if row["self_stop_health"] > best_self["health"] + args.self_stop_min_delta:
                    best_self = {"health": row["self_stop_health"], "epoch": epoch, "row": row}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= args.self_stop_patience:
                    stop_reason = f"self_stop_patience_{args.self_stop_patience}"
                    break

    if best_self["row"] is None:
        # Fallback remains label-free: choose the highest health among evaluated rows.
        chosen = max(rows, key=lambda r: r["self_stop_health"])
        best_self = {"health": chosen["self_stop_health"], "epoch": chosen["epoch"], "row": chosen}

    return {
        "variant": "R2_bounded_refdrop_selfstop",
        "seed": int(args.seed),
        "device": int(args.device),
        "config": vars(args),
        "loss_protocol": {
            "train_losses": ["L_ref-drop"],
            "bounded_score": f"score = {args.score_bound} * tanh(raw / {args.score_bound})",
            "self_stop": "label-free health over ref-drop stability, saturation, score drift, and collapse penalties",
            "uses_anomaly_labels_in_loss": False,
            "uses_pseudo_anomalies": False,
            "validation_test_labels": "report-only diagnostics after label-free checkpoint selection",
        },
        "data": {"num_nodes": int(features_np.shape[0]), "num_features": int(features_np.shape[1]), "num_labeled_normal_train": int(len(normal_idx)), "token_shape": list(token_tensor.shape)},
        "reference": {"normal_k": args.normal_k, "normal_ref_normal_ratio_report_only": float(np.mean(labels_np[normal_refs] == 0))},
        "rows": rows,
        "self_stop_selected": best_self,
        "best_report_only": best_report_only,
        "stop_reason": stop_reason,
        "time_sec": time.time() - started,
    }


def parse_csv(s, cast=str):
    return [cast(x.strip()) for x in str(s).split(',') if x.strip()]


def run_task_worker(args_dict, seed, device_id):
    return run_seed(argparse.Namespace(**args_dict), seed, device_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="elliptic")
    ap.add_argument("--devices", default="0,1,2,3")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--parallel_workers", type=int, default=4)
    ap.add_argument("--train_rate", type=float, default=0.05)
    ap.add_argument("--num_epoch", type=int, default=120)
    ap.add_argument("--descriptor_mode", choices=["hop_attr", "rwse", "hop_attr_rwse"], default="hop_attr_rwse")
    ap.add_argument("--pn_estimator", choices=["diag_gaussian", "pca_residual"], default="diag_gaussian")
    ap.add_argument("--normal_k", type=int, default=8)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--rw_steps", type=int, default=8)
    ap.add_argument("--pca_components", type=int, default=32)
    ap.add_argument("--embedding_dim", type=int, default=128)
    ap.add_argument("--GT_ffn_dim", type=int, default=128)
    ap.add_argument("--GT_dropout", type=float, default=0.2)
    ap.add_argument("--GT_attention_dropout", type=float, default=0.2)
    ap.add_argument("--GT_num_heads", type=int, default=2)
    ap.add_argument("--GT_num_layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--ref_dropout_rate", type=float, default=0.5)
    ap.add_argument("--score_bound", type=float, default=5.0)
    ap.add_argument("--saturation_frac_threshold", type=float, default=0.95)
    ap.add_argument("--saturation_penalty", type=float, default=0.05)
    ap.add_argument("--mean_drift_penalty", type=float, default=0.01)
    ap.add_argument("--collapse_penalty", type=float, default=0.10)
    ap.add_argument("--min_score_std", type=float, default=0.05)
    ap.add_argument("--self_stop_min_epoch", type=int, default=20)
    ap.add_argument("--self_stop_patience", type=int, default=8)
    ap.add_argument("--self_stop_min_delta", type=float, default=1e-5)
    ap.add_argument("--encode_batch_size", type=int, default=2048)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress_out", required=True)
    ap.add_argument("--sample_rate", type=float, default=0.15)
    ap.add_argument("--pp_k", type=int, default=None)
    ap.add_argument("--ablation_mode", default="r2_bounded_self_stop")
    args = ap.parse_args()
    args.pp_k = args.normal_k if args.pp_k is None else args.pp_k
    assert args.pp_k == args.normal_k, "VecGAD token_decoder shape requires pp_k == normal_k for this probe"

    start = time.time()
    devices = parse_csv(args.devices, int)
    seeds = parse_csv(args.seeds, int)
    tasks = [(s, devices[i % len(devices)]) for i, s in enumerate(seeds)]
    progress = {
        "status": "running", "done": 0, "total": len(tasks), "current": "start",
        "start_time": start, "config": vars(args), "tasks": [{"seed": s, "device": d} for s, d in tasks],
        "design_note": "R2 ref-drop-only with bounded tanh score and label-free self-stop; labels are report-only.",
    }
    atomic_write_json(args.progress_out, progress)
    results = []
    flat_rows = []
    completed = []

    try:
        max_workers = min(int(args.parallel_workers), len(tasks), len(devices)) if args.parallel_workers > 1 else 1
        if max_workers > 1 and len(tasks) > 1:
            futures = {}
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                for task_i, (seed, device_id) in enumerate(tasks, 1):
                    fut = ex.submit(run_task_worker, vars(args), seed, device_id)
                    futures[fut] = (task_i, seed, device_id)
                for fut in as_completed(futures):
                    task_i, seed, device_id = futures[fut]
                    res = fut.result()
                    results.append(res)
                    flat_rows.extend(res["rows"])
                    completed.append({"seed": seed, "device": device_id, "self_stop_selected": res["self_stop_selected"], "best_report_only": res["best_report_only"], "stop_reason": res["stop_reason"]})
                    p = dict(progress)
                    p.update({"done": len(completed), "current": {"task_index": task_i, "seed": seed, "device": device_id, "state": "task_finished"}, "elapsed_sec": time.time() - start, "completed": completed})
                    atomic_write_json(args.progress_out, p)
        else:
            for task_i, (seed, device_id) in enumerate(tasks, 1):
                res = run_seed(args, seed, device_id)
                results.append(res)
                flat_rows.extend(res["rows"])
                completed.append({"seed": seed, "device": device_id, "self_stop_selected": res["self_stop_selected"], "best_report_only": res["best_report_only"], "stop_reason": res["stop_reason"]})
                p = dict(progress)
                p.update({"done": len(completed), "current": {"task_index": task_i, "seed": seed, "device": device_id, "state": "task_finished"}, "elapsed_sec": time.time() - start, "completed": completed})
                atomic_write_json(args.progress_out, p)

        self_stop_rows = []
        best_rows = []
        for r in results:
            ss = dict(r["self_stop_selected"]["row"])
            ss["selected_by"] = "self_stop_label_free"
            self_stop_rows.append(ss)
            br = {"seed": r["seed"], **r["best_report_only"]}
            br["selected_by"] = "best_val_report_only_oracle"
            best_rows.append(br)

        def mean_std(rows, keys):
            out = {}
            for k in keys:
                vals = [x.get(k) for x in rows if x.get(k) is not None]
                if vals:
                    out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=0)), "n": len(vals)}
            return out

        aggregate = {
            "self_stop_report_only": mean_std(self_stop_rows, ["val_auc_report_only", "val_ap_report_only", "test_auc_report_only", "test_ap_report_only", "epoch", "all_score_std", "saturation_frac_all", "dropout_score_mse_normal"]),
            "best_val_oracle_report_only": mean_std(best_rows, ["val_auc", "val_ap", "test_auc", "test_ap", "epoch"]),
            "self_stop_rows": self_stop_rows,
            "best_val_oracle_rows": best_rows,
        }
        out = {
            "status": "finished",
            "dataset": args.dataset,
            "seeds": seeds,
            "variant": "R2_bounded_refdrop_selfstop",
            "devices": devices,
            "config": vars(args),
            "protocol": {
                "purpose": "diagnose whether R2_ref_drop_only survives bounded score scale and label-free self-stop checkpointing",
                "bounded_score": f"score = {args.score_bound} * tanh(raw / {args.score_bound})",
                "self_stop": "label-free health = -MSE_refdrop_normal - saturation penalty - normal mean drift penalty - collapse penalty",
                "uses_anomaly_labels_in_loss": False,
                "uses_pseudo_anomalies": False,
                "validation_test_labels": "report-only diagnostics; not used by loss or self-stop",
                "comparison_note": "previous seed-0 unbounded R2 best report-only test AUC/AP was about 0.6955/0.2596, but with severe scale drift",
            },
            "results": results,
            "rows": flat_rows,
            "aggregate": aggregate,
            "time_sec": time.time() - start,
        }
        atomic_write_json(args.out, out)
        final = dict(progress)
        final.update({"status": "finished", "done": len(tasks), "current": "finished", "elapsed_sec": time.time() - start, "aggregate": aggregate})
        atomic_write_json(args.progress_out, final)
        print("FINAL", json.dumps({"status": "finished", "aggregate": aggregate}, ensure_ascii=False), flush=True)
    except Exception as exc:
        fail = dict(progress)
        fail.update({"status": "failed", "current": "exception", "elapsed_sec": time.time() - start, "errors": [repr(exc)], "completed": completed})
        atomic_write_json(args.progress_out, fail)
        atomic_write_json(args.out, fail)
        raise


if __name__ == "__main__":
    main()
