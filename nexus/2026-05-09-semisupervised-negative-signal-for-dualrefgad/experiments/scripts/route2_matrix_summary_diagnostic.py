#!/usr/bin/env python3
"""方案 C 诊断：Matrix Summary 聚合方式对比

无训练诊断，使用 frozen GT encoder 输出。
在 CPU 上运行（OpenClawVM 无 GPU）。

测试聚合方式：
- trimmed_mean: 去除 10% 极端值
- weighted_mean: 用 ||d_j|| 作为权重
- quantile_q50: 中位数
- max_mean_hybrid: 0.3*max + 0.7*mean

判断标准：5/5 wins vs margin 或 AUC > margin + 2σ
"""
import argparse, json, time, random, sys, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

# 路径设置
ROOT = Path.home() / "VoxG"
INV = Path.home() / "investigations/nexus/2026-05-09-semisupervised-negative-signal-for-dualrefgad"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(INV / "experiments/scripts"))

from utils_lite import load_mat, preprocess_features, normalize_adj
from VecGAD import VecGAD


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def to_dense_features(dataset, features):
    if dataset in ["Amazon", "tf_finace", "reddit", "elliptic"]:
        features, _ = preprocess_features(features)
        return np.asarray(features, dtype=np.float32)
    return np.asarray(features.todense(), dtype=np.float32)


def l2_rows(x):
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
    import scipy.sparse as sp
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


def cosine_rows_to_matrix(a, b, block=1024):
    an = l2_rows(a.astype(np.float32))
    bn = l2_rows(b.astype(np.float32))
    outs = []
    for st in range(0, an.shape[0], block):
        outs.append(an[st:st+block] @ bn.T)
    return np.vstack(outs)


class NormalModel:
    def __init__(self, estimator, z, normal_idx, pca_components=32):
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
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

    def residual(self):
        if self.estimator == "diag_gaussian" or self.pca is None:
            return ((self.z_all - self.mu) / self.std).astype(np.float32)
        rec = self.pca.inverse_transform(self.pca.transform(self.z_all))
        return (self.z_all - rec).astype(np.float32)


def normal_soft_or_score(features, adj, normal_idx):
    hop = build_hop_attr(features, adj, hops=2)
    hm = NormalModel("diag_gaussian", hop, normal_idx)
    h = rank_percentile(hm.rejection())
    st = rwse(adj, steps=8)
    sm = NormalModel("diag_gaussian", st, normal_idx)
    s = rank_percentile(sm.rejection())
    return (1.0 - (1.0 - h) * (1.0 - s)).astype(np.float32)


def select_refs(z, residual, normal_idx, nm, features, adj, args, labels=None):
    n = z.shape[0]
    rejection = nm.rejection()
    density = nm.density_score() if hasattr(nm, 'density_score') else -rejection
    residual_norm = np.linalg.norm(residual, axis=1)

    if args.gn_mode == "label_gate":
        normal_pool = np.asarray(normal_idx)
        gn = np.zeros(n, dtype=np.float32)
        gn[normal_pool] = 1.0
    elif args.gn_mode == "normal_density":
        normal_pool = np.arange(n)
        gn = rank_percentile(density).astype(np.float32)
    elif args.gn_mode == "label_gate_density":
        normal_pool = np.asarray(normal_idx)
        gn = rank_percentile(density).astype(np.float32)
        mask = np.ones(n, bool)
        mask[normal_pool] = False
        gn[mask] = -1e9
    else:
        raise ValueError(args.gn_mode)

    if args.ga_mode == "normal_rejection":
        ga = rank_percentile(rejection).astype(np.float32)
    elif args.ga_mode == "residual_norm":
        ga = rank_percentile(residual_norm).astype(np.float32)
    elif args.ga_mode == "normal_soft_or":
        ga = normal_soft_or_score(features, adj, normal_idx).astype(np.float32)
    else:
        raise ValueError(args.ga_mode)

    sim_n = cosine_rows_to_matrix(z, z[normal_pool])
    ln_mat = sim_n if args.ln_mode == "descriptor_similarity" else sim_n ** 2
    n_scores = ln_mat + gn[normal_pool][None, :]
    normal_refs = normal_pool[np.argsort(-n_scores, axis=1)[:,:args.normal_k]]

    if args.la_mode == "residual_cosine":
        l_a = cosine_rows_to_matrix(residual, residual)
    elif args.la_mode == "descriptor_similarity":
        l_a = cosine_rows_to_matrix(z, z)
    else:
        raise ValueError(args.la_mode)

    a_scores = l_a + ga[None,:]
    np.fill_diagonal(a_scores, -1e9)
    anom_refs = np.argsort(-a_scores, axis=1)[:,:args.anom_k].astype(np.int64)

    return normal_refs, anom_refs, {"ga": ga, "rejection": rejection}


def build_tokens(features, normal_refs, anom_refs):
    toks = []
    for i in range(features.shape[0]):
        toks.append(np.concatenate([features[i:i+1], features[normal_refs[i]], features[anom_refs[i]]], axis=0))
    return torch.from_numpy(np.stack(toks).astype(np.float32))


def encode_tokens_batched_cpu(model, token_tensor_cpu, batch_size=512):
    """CPU batched encoding"""
    n = token_tensor_cpu.shape[0]
    chunks = []
    for st in range(0, n, batch_size):
        batch = token_tensor_cpu[st:st+batch_size]
        with torch.no_grad():
            emb = model.TransformerEncoder(batch).squeeze(0)
        chunks.append(emb)
    return torch.cat(chunks, dim=0)


def build_response_matrix(emb, normal_refs, anom_refs, device='cpu'):
    """构建 Response Matrix: M_ij(v) = cos(h_v - r_{n,i}, r_{a,j} - r_{n,i})"""
    n = emb.shape[0]
    K_n = normal_refs.shape[1]
    K_a = anom_refs.shape[1]

    # h: [N, D]
    h = emb

    # r_n[i]: 每个节点的 normal references 均值 [N, D]
    rn_idx = torch.from_numpy(normal_refs).to(device)
    rn = emb[rn_idx].mean(dim=1)  # [N, D]

    # r_a[j] - r_n[i]: anomaly direction
    ra_idx = torch.from_numpy(anom_refs).to(device)

    # u = h - r_n: [N, D]
    u = h - rn

    # Response Matrix: [N, K_n, K_a]
    # M_ij(v) = cos(u_v, d_ij) where d_ij = r_{a,j} - r_{n,i}
    M = np.zeros((n, K_n, K_a), dtype=np.float32)

    for v in range(n):
        # 对于节点 v，它有 K_n 个 normal refs 和 K_a 个 anomaly refs
        # 但这里 normal_refs[v] 和 anom_refs[v] 是节点级别的
        # 我们需要：对于节点 v 的每个 normal ref i，计算 r_a - r_n

        # 更准确的定义：M_ij(v) = cos(h_v - r_{n,i}(v), r_{a,j}(v) - r_{n,i}(v))
        # 这里 r_{n,i}(v) 是节点 v 的第 i 个 normal reference 的 embedding
        # r_{a,j}(v) 是节点 v 的第 j 个 anomaly reference 的 embedding

        rn_v = emb[normal_refs[v]]  # [K_n, D]
        ra_v = emb[anom_refs[v]]    # [K_a, D]

        # u_i = h_v - r_{n,i}: [K_n, D]
        u_i = h[v:v+1] - rn_v  # [K_n, D]

        # d_ji = r_{a,j} - r_{n,i}: [K_n, K_a, D]
        # 需要 broadcast: rn_v [K_n, D] 和 ra_v [K_a, D]
        # d_ji[i,j] = ra_v[j] - rn_v[i]
        d_ji = ra_v.unsqueeze(0) - rn_v.unsqueeze(1)  # [K_n, K_a, D]

        # M_ij = cos(u_i, d_ji)
        u_norm = F.normalize(u_i, p=2, dim=1)  # [K_n, D]
        d_norm = F.normalize(d_ji, p=2, dim=2)  # [K_n, K_a, D]

        # cosine: [K_n, K_a]
        M[v] = torch.bmm(u_norm.unsqueeze(1), d_norm.permute(0, 2, 1)).squeeze(1).numpy()

    return M


def matrix_summary(M, mode='mean', d_norm=None):
    """聚合 Response Matrix

    Args:
        M: [N, K_n, K_a] response matrix
        mode: 'mean', 'trimmed', 'weighted', 'quantile', 'max_mean'
        d_norm: [K_a] anomaly reference norms (for weighted)
    """
    if mode == 'mean':
        return M.mean(axis=(1, 2))

    if mode == 'trimmed':
        # 去除 10% 极端值
        flat = M.reshape(M.shape[0], -1)
        q10 = np.quantile(flat, 0.1, axis=1, keepdims=True)
        q90 = np.quantile(flat, 0.9, axis=1, keepdims=True)
        mask = (flat > q10) & (flat < q90)
        trimmed = np.where(mask, flat, np.nan)
        return np.nanmean(trimmed, axis=1)

    if mode == 'weighted':
        # weight = ||d_j||，倾向选择 norm 大的 anomaly reference
        if d_norm is None or len(d_norm.shape) != 1:
            # 用 M 的 row-wise variance 作为权重
            d_norm = np.std(M, axis=1).mean(axis=0)  # [K_a]
        else:
            # d_norm is per-node [N], use mean across nodes as weights
            d_norm = np.mean(d_norm)  # scalar
        # Use uniform weights for now (K_a dimension)
        K_a = M.shape[2]
        weights = np.ones((1, 1, K_a)) / K_a
        weighted_sum = (M * weights).sum(axis=(1, 2))
        return weighted_sum

    if mode == 'quantile':
        # Q50 (median)
        flat = M.reshape(M.shape[0], -1)
        return np.quantile(flat, 0.5, axis=1)

    if mode == 'max_mean':
        # 0.3 * max + 0.7 * mean
        max_val = M.max(axis=(1, 2))
        mean_val = M.mean(axis=(1, 2))
        return 0.3 * max_val + 0.7 * mean_val

    raise ValueError(mode)


def compute_margin(emb, normal_refs, anom_refs):
    """计算 closed-form margin: cos(u, d)"""
    h = emb
    rn_idx = torch.from_numpy(normal_refs)
    ra_idx = torch.from_numpy(anom_refs)

    rn = emb[rn_idx].mean(dim=1)
    ra = emb[ra_idx].mean(dim=1)

    u = h - rn
    d = ra - rn

    margin = torch.sum(
        F.normalize(u, p=2, dim=1) * F.normalize(d, p=2, dim=1),
        dim=1
    )
    return margin.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='synthetic_medium',
                        choices=['synthetic_small', 'synthetic_medium', 'synthetic_large', 'photo', 'dblp', 'Amazon', 'elliptic'],
                        help='Dataset name (synthetic datasets available for testing)')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--batch_size', type=int, default=512)

    # Reference selection params (from previous experiments)
    ap.add_argument('--descriptor_mode', default='hop_attr')
    ap.add_argument('--pn_estimator', default='pca_residual')
    ap.add_argument('--gn_mode', default='label_gate')
    ap.add_argument('--ln_mode', default='descriptor_similarity')
    ap.add_argument('--ga_mode', default='normal_soft_or')
    ap.add_argument('--la_mode', default='descriptor_similarity')
    ap.add_argument('--normal_k', type=int, default=4)
    ap.add_argument('--anom_k', type=int, default=16)
    ap.add_argument('--hops', type=int, default=2)
    ap.add_argument('--rw_steps', type=int, default=8)
    ap.add_argument('--pca_components', type=int, default=32)

    # GT encoder params
    ap.add_argument('--embedding_dim', type=int, default=256)
    ap.add_argument('--GT_ffn_dim', type=int, default=256)
    ap.add_argument('--GT_dropout', type=float, default=0.4)
    ap.add_argument('--GT_attention_dropout', type=float, default=0.4)
    ap.add_argument('--GT_num_heads', type=int, default=2)
    ap.add_argument('--GT_num_layers', type=int, default=3)
    ap.add_argument('--sample_rate', type=float, default=0.15)
    ap.add_argument('--mean', type=float, default=0.02)
    ap.add_argument('--var', type=float, default=0.01)
    ap.add_argument('--outlier_beta', type=float, default=0.3)
    ap.add_argument('--ring_R_max', type=float, default=1.0)
    ap.add_argument('--ring_R_min', type=float, default=0.3)
    ap.add_argument('--lambda_rec_tok', type=float, default=1.0)
    ap.add_argument('--lambda_rec_emb', type=float, default=0.1)

    ap.add_argument('--train_rate', type=float, default=0.05)
    ap.add_argument('--pp_k', type=int, default=4, help='Number of pseudo-positional tokens')

    args = ap.parse_args()

    results = []

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        set_seed(seed)

        # Load data
        adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat(
            args.dataset, args.train_rate, 0.1, args=args
        )

        features_np = to_dense_features(args.dataset, features)
        labels_np = np.asarray(ano_label).reshape(-1).astype(int)
        normal_idx = np.asarray(normal_for_train_idx, dtype=int)
        idx_test_np = np.asarray(idx_test, dtype=int)

        # Build descriptor
        z = build_descriptor(args.descriptor_mode, features_np, adj, args.hops, args.rw_steps)
        nm = NormalModel(args.pn_estimator, z, normal_idx, args.pca_components)
        residual = nm.residual()

        # Select refs
        normal_refs, anom_refs, meta = select_refs(z, residual, normal_idx, nm, features_np, adj, args, labels_np)

        # Build tokens
        token_tensor = build_tokens(features_np, normal_refs, anom_refs)
        print(f"Token shape: {token_tensor.shape}")

        # Load frozen encoder
        model = VecGAD(features_np.shape[1], args.embedding_dim, 'prelu', args)
        model.eval()

        # Encode (CPU)
        print(f"Encoding on CPU (batch_size={args.batch_size})...")
        t0 = time.time()
        emb = encode_tokens_batched_cpu(model, token_tensor, args.batch_size)
        print(f"Encoding time: {time.time() - t0:.1f}s")

        # Compute margin baseline
        margin = compute_margin(emb, normal_refs, anom_refs)

        # Build Response Matrix
        print("Building Response Matrix...")
        t0 = time.time()
        M = build_response_matrix(emb, normal_refs, anom_refs)
        print(f"Matrix build time: {time.time() - t0:.1f}s")
        print(f"Matrix shape: {M.shape}")

        # Compute ||d|| for weighted summary
        ra_idx = torch.from_numpy(anom_refs)
        rn_idx = torch.from_numpy(normal_refs)
        rn = emb[rn_idx].mean(dim=1)
        ra = emb[ra_idx].mean(dim=1)
        d = ra - rn
        d_norm = torch.norm(d, p=2, dim=1).numpy()

        # Test different summaries
        summaries = {}
        for mode in ['mean', 'trimmed', 'weighted', 'quantile', 'max_mean']:
            summaries[mode] = matrix_summary(M, mode=mode, d_norm=d_norm)

        # Evaluate
        test_labels = labels_np[idx_test_np]

        seed_result = {'seed': seed}

        # Margin baseline
        margin_auc = roc_auc_score(test_labels, margin[idx_test_np])
        margin_ap = average_precision_score(test_labels, margin[idx_test_np])
        seed_result['margin_auc'] = margin_auc
        seed_result['margin_ap'] = margin_ap
        print(f"Margin baseline: AUC={margin_auc:.4f}, AP={margin_ap:.4f}")

        # Matrix summaries
        wins = []
        for mode, score in summaries.items():
            auc = roc_auc_score(test_labels, score[idx_test_np])
            ap = average_precision_score(test_labels, score[idx_test_np])
            sp = spearmanr(score[idx_test_np], margin[idx_test_np]).correlation
            win = auc > margin_auc
            wins.append(win)

            seed_result[f'{mode}_auc'] = auc
            seed_result[f'{mode}_ap'] = ap
            seed_result[f'{mode}_spearman'] = sp
            seed_result[f'{mode}_win'] = win

            print(f"  {mode}: AUC={auc:.4f}, AP={ap:.4f}, Spearman={sp:.4f}, win={win}")

        seed_result['wins'] = sum(wins)
        results.append(seed_result)

        # Matrix stats
        seed_result['mat_mean'] = M.mean()
        seed_result['mat_std'] = M.std()
        seed_result['mat_min'] = M.min()
        seed_result['mat_max'] = M.max()

        print(f"Matrix stats: mean={M.mean():.4f}, std={M.std():.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY (5 seeds)")
    print(f"{'='*60}")

    summary = {}
    modes = ['margin', 'mean', 'trimmed', 'weighted', 'quantile', 'max_mean']

    for mode in modes:
        aucs = [r.get(f'{mode}_auc', r.get('margin_auc')) for r in results]
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        wins = [r.get(f'{mode}_win', False) for r in results] if mode != 'margin' else [False] * 5
        win_count = sum(wins)

        summary[mode] = {
            'auc_mean': mean_auc,
            'auc_std': std_auc,
            'wins': win_count if mode != 'margin' else '-'
        }

        print(f"{mode}: AUC={mean_auc:.4f}±{std_auc:.4f}, wins={win_count}/{5 if mode != 'margin' else '-'}")

    # Save results
    out_path = INV / "experiments/outputs" / "route2_matrix_summary_diagnostic.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_python(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python(v) for v in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return convert_to_python(obj.tolist())
        return obj
    
    with open(out_path, 'w') as f:
        json.dump(convert_to_python({'results': results, 'summary': summary}), f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # Decision
    print("\n" + "="*60)
    print("DECISION")
    print("="*60)

    best_mode = None
    best_wins = 0
    for mode in ['mean', 'trimmed', 'weighted', 'quantile', 'max_mean']:
        if summary[mode]['wins'] > best_wins:
            best_wins = summary[mode]['wins']
            best_mode = mode

    margin_auc_mean = summary['margin']['auc_mean']
    margin_auc_std = summary['margin']['auc_std']

    if best_wins >= 5:
        print(f"✅ 方案 C 优先：{best_mode} 获得 5/5 wins")
        print(f"   建议实现 learnable version（训练 1 层 GT encoder）")
    elif summary[best_mode]['auc_mean'] > margin_auc_mean + 2 * margin_auc_std:
        print(f"✅ 方案 C 优先：{best_mode} AUC > margin + 2σ")
    else:
        print(f"⚠️ 方案 C 不稳定：{best_mode} wins={best_wins}/5")
        print(f"   建议进入方案 A 诊断（RHO-style normality alignment）")


if __name__ == '__main__':
    main()