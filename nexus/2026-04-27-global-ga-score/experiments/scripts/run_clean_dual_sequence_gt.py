#!/usr/bin/env python3
"""Clean Dual-Sequence GT classifier.

Non-intrusive experimental entry point:
  dual-sequence reference tokens -> existing VecGAD TransformerEncoder -> classifier

Differences from original VecGAD training:
  - no token_decoder / reconstruction loss
  - no ring loss
  - no modification to run.py / VecGAD.py
  - normal-only labels are used only for normal train set and normal calibration
  - anomaly labels are used only for val/test evaluation

Training objective:
  normal embeddings from V_train^n are class 0;
  synthetic pseudo-outlier embeddings generated from normal embeddings are class 1.
"""
import argparse, json, time, random
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

# Allow running from investigation/scripts while importing VoxG modules.
sys.path.insert(0, str(Path.home() / 'VoxG'))

from utils import load_mat, preprocess_features, normalize_adj
from VecGAD import VecGAD


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def row_norm(x):
    s = x.sum(axis=1, keepdims=True); s[s == 0] = 1.0
    return x / s


def l2_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def minmax(x):
    x = np.asarray(x, dtype=np.float64)
    return ((x - x.min()) / (x.max() - x.min() + 1e-12)).astype(np.float32)


def norm_adj_np(adj_np):
    deg = np.asarray(adj_np.sum(axis=1)).reshape(-1)
    inv = np.zeros_like(deg, dtype=np.float32)
    m = deg > 0
    inv[m] = 1.0 / np.sqrt(deg[m])
    return inv[:, None] * adj_np * inv[None, :]


def build_mps(features_np, adj_np, k=6):
    a = norm_adj_np(adj_np)
    hops = [features_np]
    cur = features_np.copy()
    for _ in range(k):
        cur = a @ cur
        hops.append(cur.copy())
    return np.concatenate(hops, axis=1).astype(np.float32)


def cosine_matrix(a, b):
    return l2_rows(a) @ l2_rows(b).T


def build_ndc(features_np, adj_np, k=6):
    feats = row_norm(features_np.copy())
    a = norm_adj_np(adj_np)
    n, d = feats.shape
    hops = np.zeros((n, k + 1, d), dtype=np.float32)
    hops[:, 0] = feats
    cur = feats.copy()
    for h in range(1, k + 1):
        cur = a @ cur
        hops[:, h] = cur
    delta = hops[:, 1:] - hops[:, :-1]
    ndc = np.zeros(n, dtype=np.float32)
    for i in range(n):
        neigh = np.where(adj_np[i] > 0)[0]
        if len(neigh) == 0:
            continue
        x = delta[i].reshape(-1)
        y = delta[neigh].mean(axis=0).reshape(-1)
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            ndc[i] = 0.0
        else:
            ndc[i] = np.corrcoef(x, y)[0, 1]
    return ndc


def local_mean(v, adj_np):
    deg = adj_np.sum(axis=1)
    out = adj_np @ v
    res = np.zeros_like(v, dtype=np.float64)
    m = deg > 0
    res[m] = out[m] / deg[m]
    res[~m] = v[~m]
    return res.astype(np.float32)


def topk_mean(x, bank, k=32, block=2048):
    vals = []
    k = min(k, bank.shape[0])
    for st in range(0, x.shape[0], block):
        sims = x[st:st + block] @ bank.T
        vals.append(np.partition(sims, -k, axis=1)[:, -k:].mean(axis=1))
    return np.concatenate(vals).astype(np.float32)


def normal_percentile(x, normal_idx):
    base = np.sort(x[normal_idx])
    return (np.searchsorted(base, x, side='right') / (len(base) + 1e-12)).astype(np.float32)


def compute_global_scores(features_np, adj_np, normal_train_idx, pp_k, ga_mode):
    z = l2_rows(build_mps(features_np, adj_np, pp_k))
    x_norm = l2_rows(row_norm(features_np.copy()).astype(np.float32))

    ndc = build_ndc(features_np, adj_np, pp_k)
    q = 1.0 / (1.0 + np.exp(-(ndc - ndc.mean()) / (ndc.std() + 1e-8)))
    c = local_mean(q, adj_np)
    support = minmax(q * c)
    h = np.stack([q, c], axis=1).astype(np.float32)

    center = z[normal_train_idx].mean(axis=0, keepdims=True)
    center = center / (np.linalg.norm(center) + 1e-12)
    d_center = minmax(np.linalg.norm(z - center, axis=1))
    d_density = minmax(1.0 - topk_mean(z, z[normal_train_idx], 32))
    d_attr = minmax(1.0 - topk_mean(x_norm, x_norm[normal_train_idx], 32))
    d_max = np.maximum(d_center, d_density)

    r_center = normal_percentile(d_center, normal_train_idx)
    r_density = normal_percentile(d_density, normal_train_idx)
    r_attr = normal_percentile(d_attr, normal_train_idx)
    r_max = normal_percentile(d_max, normal_train_idx)
    soft_or = 1.0 - (1.0 - r_center) * (1.0 - r_density) * (1.0 - r_attr)

    if ga_mode == 'baseline_qc':
        g_anom = support
    elif ga_mode == 'normal_attr':
        g_anom = r_attr
    elif ga_mode == 'normal_max_dev':
        g_anom = r_max
    elif ga_mode == 'normal_soft_or':
        g_anom = soft_or.astype(np.float32)
    else:
        raise ValueError(f'Unknown ga_mode: {ga_mode}')

    return {
        'z': z,
        'h': h,
        'g_anom': minmax(g_anom),
        'support': support,
        'r_center': r_center,
        'r_density': r_density,
        'r_attr': r_attr,
        'soft_or': soft_or.astype(np.float32),
    }


def build_dual_sequence_tokens(features_np, adj_np, normal_train_idx, pp_k, normal_k, anom_k, ga_mode, gl_combine):
    scores = compute_global_scores(features_np, adj_np, normal_train_idx, pp_k, ga_mode)
    z = scores['z']
    h = scores['h']
    g_anom = scores['g_anom']

    # Normal reference: hard eligibility on normal_train_idx + local representation similarity.
    local_normal_bank = cosine_matrix(z, z[normal_train_idx])
    local_anom = cosine_matrix(h, h)

    tokens = []
    normal_refs = []
    anom_refs = []
    for i in range(len(features_np)):
        self_tok = features_np[i:i + 1]

        n_scores = local_normal_bank[i]
        n_order = np.argsort(n_scores)[::-1][:normal_k]
        n_idx = np.asarray(normal_train_idx)[n_order]

        if gl_combine == 'add':
            l = minmax(local_anom[i])
            a_scores = g_anom + l
        elif gl_combine == 'multiply':
            a_scores = g_anom * local_anom[i]
        else:
            raise ValueError(f'Unknown gl_combine: {gl_combine}')
        a_scores[i] = -1e9
        a_idx = np.argsort(a_scores)[::-1][:anom_k]

        seq = np.concatenate([self_tok, features_np[n_idx], features_np[a_idx]], axis=0)
        tokens.append(seq)
        normal_refs.append(n_idx)
        anom_refs.append(a_idx)

    meta = {
        'token_len': int(1 + normal_k + anom_k),
        'normal_k': int(normal_k),
        'anom_k': int(anom_k),
        'ga_mode': ga_mode,
        'gl_combine': gl_combine,
        'normal_refs': np.stack(normal_refs).astype(np.int64),
        'anom_refs': np.stack(anom_refs).astype(np.int64),
    }
    return torch.from_numpy(np.stack(tokens).astype(np.float32)), meta


def reference_purity(meta, labels):
    nref = meta['normal_refs']
    aref = meta['anom_refs']
    return {
        'normal_ref_normal_ratio': float(np.mean(labels[nref] == 0)),
        'anom_ref_anom_ratio': float(np.mean(labels[aref] == 1)),
        'anom_ref_anom_ratio_on_anom_nodes': float(np.mean(labels[aref[labels == 1]] == 1)) if np.any(labels == 1) else 0.0,
    }


def eval_logits(logits, labels, idx):
    s = logits[idx]
    y = labels[idx]
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def encode_tokens_batched(model, token_tensor_cpu, device, batch_size: int):
    """Encode node-token sequences in node batches.

    token_tensor_cpu: [N, T, F] kept on CPU to avoid Elliptic full-token GPU OOM.
    VecGAD.TransformerEncoder attends only within each node token sequence, not across nodes,
    so splitting along N is semantically equivalent to full encoding.
    Returns: [N, d]
    """
    n = token_tensor_cpu.shape[0]
    if batch_size <= 0 or batch_size >= n:
        return model.TransformerEncoder(token_tensor_cpu.to(device)).squeeze(0)
    chunks = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = token_tensor_cpu[start:end].to(device, non_blocking=True)
        chunks.append(model.TransformerEncoder(chunk).squeeze(0))
    return torch.cat(chunks, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train_rate', type=float, default=0.05)
    ap.add_argument('--num_epoch', type=int, default=200)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--pp_k', type=int, default=6)
    ap.add_argument('--normal_k', type=int, default=4)
    ap.add_argument('--anom_k', type=int, default=16)
    ap.add_argument('--ga_mode', choices=['baseline_qc','normal_attr','normal_max_dev','normal_soft_or'], default='normal_soft_or')
    ap.add_argument('--gl_combine', choices=['add','multiply'], default='add')
    ap.add_argument('--pseudo_beta', type=float, default=0.3)
    ap.add_argument('--pseudo_noise', type=float, default=0.01)
    ap.add_argument('--embedding_dim', type=int, default=256)
    ap.add_argument('--GT_ffn_dim', type=int, default=256)
    ap.add_argument('--GT_dropout', type=float, default=0.4)
    ap.add_argument('--GT_attention_dropout', type=float, default=0.4)
    ap.add_argument('--GT_num_heads', type=int, default=2)
    ap.add_argument('--GT_num_layers', type=int, default=3)
    # Args required by VecGAD constructor but not used in this clean runner.
    ap.add_argument('--sample_rate', type=float, default=0.15)
    ap.add_argument('--mean', type=float, default=0.02)
    ap.add_argument('--var', type=float, default=0.01)
    ap.add_argument('--outlier_beta', type=float, default=0.3)
    ap.add_argument('--ring_R_max', type=float, default=1.0)
    ap.add_argument('--ring_R_min', type=float, default=0.3)
    ap.add_argument('--lambda_rec_tok', type=float, default=1.0)
    ap.add_argument('--lambda_rec_emb', type=float, default=0.1)
    ap.add_argument('--ablation_mode', type=str, default='none')
    ap.add_argument('--encode_batch_size', type=int, default=2048)
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--out', type=str, default='')
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')

    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat(args.dataset, args.train_rate, 0.1, args=args)
    if args.dataset in ['Amazon', 'tf_finace', 'reddit', 'elliptic']:
        features, _ = preprocess_features(features)
    else:
        features = features.todense()
    features_np = np.asarray(features, dtype=np.float32)
    adj_norm = normalize_adj(adj)
    adj_with_self = (adj_norm + sp.eye(adj.shape[0])).todense()
    adj_np = np.asarray(adj_with_self, dtype=np.float32)
    labels_np = np.asarray(ano_label).reshape(-1).astype(int)

    token_tensor, meta = build_dual_sequence_tokens(features_np, adj_np, np.asarray(normal_for_train_idx), args.pp_k, args.normal_k, args.anom_k, args.ga_mode, args.gl_combine)
    pur = reference_purity(meta, labels_np)

    # Keep token_tensor on CPU; move node batches to GPU inside encode_tokens_batched.
    model = VecGAD(features_np.shape[1], args.embedding_dim, 'prelu', args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    normal_idx = torch.tensor(normal_for_train_idx, dtype=torch.long, device=device)

    if args.dry_run:
        with torch.no_grad():
            emb = encode_tokens_batched(model, token_tensor, device, args.encode_batch_size)
        result = {'dataset': args.dataset, 'token_shape': list(token_tensor.shape), 'embedding_shape': list(emb.shape), 'purity': pur, 'ga_mode': args.ga_mode, 'gl_combine': args.gl_combine, 'encode_batch_size': args.encode_batch_size}
        print(json.dumps(result, indent=2))
        return

    best = {'val_auc': -1, 'val_ap': -1, 'test_auc': -1, 'test_ap': -1, 'epoch': -1}
    start = time.time()
    for epoch in range(args.num_epoch + 1):
        model.train(); optimizer.zero_grad()
        emb = encode_tokens_batched(model, token_tensor, device, args.encode_batch_size)  # [N, d]
        normal_emb = emb[normal_idx]
        center = normal_emb.mean(dim=0, keepdim=True)
        direction = F.normalize(normal_emb - center, dim=1)
        noise = torch.randn_like(normal_emb) * args.pseudo_noise
        outlier_emb = normal_emb + args.pseudo_beta * direction + noise
        emb_combine = torch.cat([normal_emb, outlier_emb], dim=0)
        logits_train = model.fc3(model.act(model.fc2(model.act(model.fc1(emb_combine))))).squeeze(-1)
        y_train = torch.cat([torch.zeros(len(normal_emb), device=device), torch.ones(len(outlier_emb), device=device)])
        loss = bce(logits_train, y_train)
        loss.backward(); optimizer.step()

        if epoch % 10 == 0 or epoch == args.num_epoch:
            model.eval()
            with torch.no_grad():
                emb_eval = encode_tokens_batched(model, token_tensor, device, args.encode_batch_size)
                logits = model.fc3(model.act(model.fc2(model.act(model.fc1(emb_eval))))).squeeze(-1).detach().cpu().numpy()
            val_auc, val_ap = eval_logits(logits, labels_np, idx_val)
            test_auc, test_ap = eval_logits(logits, labels_np, idx_test)
            if val_auc + val_ap > best['val_auc'] + best['val_ap']:
                best.update({'val_auc': val_auc, 'val_ap': val_ap, 'test_auc': test_auc, 'test_ap': test_ap, 'epoch': epoch})
            print(json.dumps({'epoch': epoch, 'loss': float(loss.item()), 'val_auc': val_auc, 'val_ap': val_ap, 'test_auc': test_auc, 'test_ap': test_ap, 'best': best}, ensure_ascii=False))

    result = {'dataset': args.dataset, 'seed': args.seed, 'ga_mode': args.ga_mode, 'gl_combine': args.gl_combine, 'encode_batch_size': args.encode_batch_size, 'best': best, 'purity': pur, 'time_sec': time.time() - start}
    print('FINAL', json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
