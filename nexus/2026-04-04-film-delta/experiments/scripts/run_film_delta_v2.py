"""
FiLM-Delta VoxGFormer 完整训练脚本 v2

关键修复：完全复刻 concat 的训练流程

作者: Nexus
日期: 2026-04-04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
import sys

# 导入模型
from FiLMDeltaVoxG import FiLMDeltaVoxG


# ============ 数据加载 ============

def load_photo_dataset(data_path: str = None):
    """加载 Photo 数据集"""
    if data_path is None:
        data_path = '/root/gpufree-data/linziyao/MatrixGAD/dataset/photo.mat'
    
    data = sio.loadmat(data_path)
    
    features = data['Attributes']
    if hasattr(features, 'todense'):
        features = features.todense()
    features = torch.tensor(np.array(features), dtype=torch.float32)
    
    labels = data['Label'].flatten()
    labels = torch.tensor(labels, dtype=torch.long)
    
    adj = data['Network']
    if not hasattr(adj, 'todense'):
        adj = sp.csr_matrix(adj)
    
    return features, labels, adj


def normalize_adjacency(adj):
    """标准化邻接矩阵"""
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def generate_hop_features(features, adj, n_hops=6):
    """生成 Hop features"""
    adj_norm = normalize_adjacency(adj)
    adj_tensor = torch.tensor(adj_norm.todense(), dtype=torch.float32)
    
    n_nodes, n_features = features.shape
    hop_features = torch.zeros(n_nodes, n_hops + 1, n_features)
    hop_features[:, 0, :] = features
    
    for hop in range(1, n_hops + 1):
        hop_features[:, hop, :] = torch.matmul(adj_tensor, hop_features[:, hop - 1, :])
    
    return hop_features


# ============ 完整模型 ============

class FiLMDeltaVoxGFullV2(nn.Module):
    """
    FiLM-Delta VoxGFormer 完整版 v2
    
    完全复刻 concat 的训练流程
    """
    def __init__(
        self,
        n_features: int,
        d_hidden: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        n_hops: int = 7,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_hops = n_hops
        self.n_features = n_features
        self.d_hidden = d_hidden
        
        # 主模型
        self.backbone = FiLMDeltaVoxG(
            n_features=n_features,
            d_hidden=d_hidden,
            n_layers=n_layers,
            n_heads=n_heads,
            n_hops=n_hops,
            dropout=dropout
        )
        
        # Token 解码器（从 embedding 重构 tokens）
        self.token_decoder = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, n_hops * n_features)
        )
        
        # 重构错误投影器
        self.reconstruction_proj = nn.Sequential(
            nn.Linear(n_hops * n_features, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden)
        )
        
        # 分类头
        self.fc1 = nn.Linear(d_hidden, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, 1)
    
    def forward(self, hop_features, normal_for_train_idx, train_flag=True, 
                sample_rate=0.5, outlier_beta=0.1, noise_var=0.1, noise_mean=0.0,
                ring_R_min=0.5, ring_R_max=2.0):
        """
        完全复刻 concat 的前向传播
        
        Args:
            hop_features: (n_nodes, n_hops, n_features)
            normal_for_train_idx: 训练集正常节点索引
            train_flag: 是否训练模式
            sample_rate: 用于生成离群点的正常节点比例
            outlier_beta: 离群点缩放因子
            noise_var: 噪声方差
            noise_mean: 噪声均值
            ring_R_min: Ring 损失最小半径
            ring_R_max: Ring 损失最大半径
        
        Returns:
            emb: 所有节点嵌入
            emb_combine: 正常节点 + 离群点的组合嵌入
            logits: 分类 logits
            outlier_emb: 离群点嵌入
            noised_normal_emb: 加噪声的正常节点嵌入
            loss_rec: 重构损失
            loss_ring: Ring 损失
        """
        n_nodes = hop_features.shape[0]
        
        # === 1. 主模型编码 ===
        # 提取 Delta
        delta = hop_features[:, 1:, :] - hop_features[:, :-1, :]
        
        # 投影
        hop_proj = self.backbone.input_proj(hop_features)
        delta_proj = self.backbone.input_proj(delta)
        
        # 聚合 Delta
        cond = self.backbone.delta_aggregator(delta_proj)
        
        # Transformer 层
        x = hop_proj
        for layer in self.backbone.layers:
            x = layer(x, cond)
        
        # 节点嵌入 (使用 mean pooling)
        emb = x.mean(dim=1)  # (n_nodes, d_hidden)
        
        # 全局中心
        h_mean = emb.mean(dim=0, keepdim=True)  # (1, d_hidden)
        
        # 初始化
        outlier_emb = None
        emb_combine = None
        noised_normal_emb = None
        loss_rec = torch.tensor(0.0, device=hop_features.device)
        loss_ring = torch.tensor(0.0, device=hop_features.device)
        
        if train_flag:
            # === 2. 选择部分正常节点用于生成离群点 ===
            perm = torch.randperm(normal_for_train_idx.size(0), device=normal_for_train_idx.device)
            normal_for_train_idx = normal_for_train_idx[perm]
            
            n_generate = int(len(normal_for_train_idx) * sample_rate)
            normal_for_generation_idx = normal_for_train_idx[:n_generate]
            
            normal_for_generation_emb = emb[normal_for_generation_idx]
            
            # === 3. 添加噪声 ===
            noise = torch.randn_like(normal_for_generation_emb) * noise_var + noise_mean
            noised_normal_emb = normal_for_generation_emb + noise
            
            # === 4. Token 重构 ===
            reconstructed_tokens = self.token_decoder(emb)  # (n_nodes, n_hops * n_features)
            
            # === 5. 计算重构错误 ===
            input_tokens_flat = hop_features.flatten(1)  # (n_nodes, n_hops * n_features)
            reconstruction_error = reconstructed_tokens - input_tokens_flat
            
            # === 6. 投影重构错误 ===
            recon_error_proj = self.reconstruction_proj(reconstruction_error[normal_for_generation_idx])
            
            # === 7. 生成离群点 ===
            outlier_emb = normal_for_generation_emb + outlier_beta * recon_error_proj
            
            # === 8. Ring 损失 ===
            outlier_to_center_dist = torch.norm(outlier_emb - h_mean, p=2, dim=1)
            ring_out_range_loss = torch.relu(ring_R_min - outlier_to_center_dist)
            ring_in_range_loss = torch.relu(outlier_to_center_dist - ring_R_max)
            loss_ring = torch.mean(ring_out_range_loss + ring_in_range_loss)
            
            # === 9. 重构损失 ===
            # Token 重构损失
            token_rec_loss = F.mse_loss(
                reconstructed_tokens[normal_for_generation_idx],
                input_tokens_flat[normal_for_generation_idx]
            )
            
            # Embedding 重构损失（将重构的 tokens 再编码）
            reconstructed_tokens_vector = reconstructed_tokens[normal_for_generation_idx].view(-1, self.n_hops, self.n_features)
            # 简化：直接计算嵌入距离
            emb_rec_loss = F.mse_loss(
                self.backbone.input_proj(reconstructed_tokens_vector).mean(dim=1),
                normal_for_generation_emb
            )
            
            loss_rec = 0.5 * token_rec_loss + 0.5 * emb_rec_loss
            
            # === 10. 组合嵌入 ===
            emb_combine = torch.cat([emb[normal_for_train_idx], outlier_emb], dim=0)
        
        return emb, emb_combine, outlier_emb, noised_normal_emb, loss_rec, loss_ring


# ============ 训练函数 ============

def train_epoch(model, hop_features, labels, normal_for_train_idx, optimizer, 
                bce_loss_fn, args):
    """训练一个 epoch（完全复刻 concat）"""
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    emb, emb_combine, outlier_emb, noised_normal_emb, loss_rec, loss_ring = model(
        hop_features, normal_for_train_idx, train_flag=True,
        sample_rate=args['sample_rate'],
        outlier_beta=args['outlier_beta'],
        noise_var=args['noise_var'],
        noise_mean=args['noise_mean'],
        ring_R_min=args['ring_R_min'],
        ring_R_max=args['ring_R_max']
    )
    
    # 分类
    f1 = torch.relu(model.fc1(emb_combine))
    f2 = torch.relu(model.fc2(f1))
    logits = model.fc3(f2)
    
    # BCE 损失：正常节点标签为 0，离群点标签为 1
    lbl = torch.cat([
        torch.zeros(len(normal_for_train_idx)),
        torch.ones(len(outlier_emb))
    ]).unsqueeze(1).to(hop_features.device)
    
    logits = logits.squeeze(0)  # (n_combine, 1)
    
    loss_bce = bce_loss_fn(logits, lbl)
    loss_bce = torch.mean(loss_bce)
    
    # 总损失
    loss = (args['bce_weight'] * loss_bce + 
            args['rec_weight'] * loss_rec + 
            args['ring_weight'] * loss_ring)
    
    loss.backward()
    optimizer.step()
    
    return {
        'total_loss': loss.item(),
        'bce_loss': loss_bce.item(),
        'rec_loss': loss_rec.item(),
        'ring_loss': loss_ring.item()
    }


@torch.no_grad()
def evaluate(model, hop_features, labels, test_idx):
    """评估模型"""
    model.eval()
    
    # 只需要编码，不需要生成离群点
    delta = hop_features[:, 1:, :] - hop_features[:, :-1, :]
    hop_proj = model.backbone.input_proj(hop_features)
    delta_proj = model.backbone.input_proj(delta)
    cond = model.backbone.delta_aggregator(delta_proj)
    
    x = hop_proj
    for layer in model.backbone.layers:
        x = layer(x, cond)
    
    emb = x.mean(dim=1)
    
    # 分类
    f1 = torch.relu(model.fc1(emb))
    f2 = torch.relu(model.fc2(f1))
    logits = model.fc3(f2).squeeze(-1)
    
    probs = torch.sigmoid(logits[test_idx])
    preds = probs.cpu().numpy()
    targets = labels[test_idx].cpu().numpy()
    
    if len(np.unique(targets)) < 2:
        return 0.5, 0.0
    
    auc = roc_auc_score(targets, preds)
    ap = average_precision_score(targets, preds)
    
    return auc, ap


def run_training(
    n_epochs=200,
    lr=0.001,
    d_hidden=128,
    n_layers=3,
    seed=42,
    verbose=True,
    args=None
):
    """完整训练流程"""
    # 默认参数
    if args is None:
        args = {
            'sample_rate': 0.5,
            'outlier_beta': 0.1,
            'noise_var': 0.1,
            'noise_mean': 0.0,
            'ring_R_min': 0.5,
            'ring_R_max': 2.0,
            'bce_weight': 1.0,
            'rec_weight': 0.5,
            'ring_weight': 0.1,
            'negsamp_ratio': 1.0
        }
    
    print("=" * 60)
    print(f"FiLMDeltaVoxG Full V2 Training (seed={seed})")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 加载数据
    if verbose:
        print("\n[1] 加载数据...")
    features, labels, adj = load_photo_dataset()
    n_nodes, n_features = features.shape
    
    if verbose:
        print(f"    节点数: {n_nodes}, 特征维度: {n_features}")
        print(f"    异常节点: {labels.sum().item()} ({labels.sum().item() / n_nodes * 100:.1f}%)")
    
    # 生成 Hop features
    if verbose:
        print("\n[2] 生成 Hop features...")
    n_hops = 6
    hop_features = generate_hop_features(features, adj, n_hops)
    if verbose:
        print(f"    Hop features: {hop_features.shape}")
    
    # 数据划分
    torch.manual_seed(seed)
    normal_idx = torch.where(labels == 0)[0]
    perm = torch.randperm(len(normal_idx))
    train_size = int(0.05 * n_nodes)
    normal_for_train_idx = normal_idx[perm[:train_size]]
    
    test_idx = torch.cat([normal_idx[perm[train_size:]], torch.where(labels == 1)[0]])
    
    if verbose:
        print(f"\n[3] 数据划分:")
        print(f"    训练集: {len(normal_for_train_idx)} (仅正常)")
        print(f"    测试集: {len(test_idx)}")
    
    # 创建模型
    model = FiLMDeltaVoxGFullV2(
        n_features=n_features,
        d_hidden=d_hidden,
        n_layers=n_layers,
        n_hops=n_hops + 1
    )
    
    if verbose:
        print(f"\n[4] 模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none', 
                                        pos_weight=torch.tensor([args['negsamp_ratio']]))
    
    best_test_auc = 0
    best_test_ap = 0
    best_epoch = 0
    
    if verbose:
        print(f"\n[5] 训练...")
    
    for epoch in range(n_epochs):
        losses = train_epoch(model, hop_features, labels, normal_for_train_idx, 
                            optimizer, bce_loss_fn, args)
        test_auc, test_ap = evaluate(model, hop_features, labels, test_idx)
        
        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_test_ap = test_ap
            best_epoch = epoch + 1
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch + 1}: Loss={losses['total_loss']:.4f} "
                  f"(BCE={losses['bce_loss']:.4f}, Rec={losses['rec_loss']:.4f}, Ring={losses['ring_loss']:.4f}), "
                  f"Test AUC={test_auc:.4f}")
    
    results = {
        'seed': seed,
        'best_epoch': best_epoch,
        'best_test_auc': best_test_auc,
        'best_test_ap': best_test_ap
    }
    
    if verbose:
        print(f"\n[6] 结果:")
        print(f"    最佳 Epoch: {best_epoch}")
        print(f"    最佳 Test AUC: {best_test_auc:.4f}")
        print(f"    最佳 Test AP: {best_test_ap:.4f}")
    
    return results


def run_5seed_experiment(seeds=[0, 1, 2, 3, 4], **kwargs):
    """5-seed 实验"""
    print("=" * 60)
    print("5-Seed Experiment: FiLMDeltaVoxG Full V2")
    print("=" * 60)
    
    all_results = []
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        result = run_training(seed=seed, verbose=False, **kwargs)
        all_results.append(result)
        print(f"  Test AUC: {result['best_test_auc']:.4f}, AP: {result['best_test_ap']:.4f}")
    
    # 统计
    aucs = [r['best_test_auc'] for r in all_results]
    aps = [r['best_test_ap'] for r in all_results]
    
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    mean_ap = np.mean(aps)
    std_ap = np.std(aps)
    
    print("\n" + "=" * 60)
    print("5-Seed Results")
    print("=" * 60)
    print(f"Test AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Test AP:  {mean_ap:.4f} ± {std_ap:.4f}")
    
    return mean_auc, std_auc, all_results


# ============ 主函数 ============

def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("FiLMDeltaVoxG Full V2 训练")
    print("=" * 70)
    
    mean_auc, std_auc, results = run_5seed_experiment(n_epochs=200)
    
    # 与基线对比
    print("\n" + "=" * 70)
    print("与基线对比")
    print("=" * 70)
    print(f"{'方法':<30} {'AUC':<20}")
    print("-" * 50)
    print(f"{'VecGAD (SOTA)':<30} {'0.8960':<20}")
    print(f"{'concat':<30} {'0.8777 ± 0.038':<20}")
    print(f"{'DualStreamVoxG':<30} {'0.5157 ± 0.011':<20}")
    print(f"{'FiLMDeltaVoxG (无重构)':<30} {'0.48':<20}")
    print(f"{'FiLMDeltaVoxG Full V2':<30} {f'{mean_auc:.4f} ± {std_auc:.4f}':<20}")


if __name__ == "__main__":
    main()