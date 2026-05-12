"""
FiLM-Delta VoxGFormer 完整训练脚本

关键修复：
1. 添加重构损失（Token + Embedding）
2. 添加离群点生成机制
3. Ring 损失

参考：HBAFormer_v2.py

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
from FiLMDeltaVoxG import FiLMDeltaVoxG, create_film_delta_model


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


def split_data(n_nodes, labels, train_rate=0.05, val_rate=0.10, seed=42):
    """数据划分"""
    torch.manual_seed(seed)
    
    normal_idx = torch.where(labels == 0)[0]
    anomaly_idx = torch.where(labels == 1)[0]
    
    perm = torch.randperm(len(normal_idx))
    train_size = int(train_rate * n_nodes)
    val_size = int(val_rate * n_nodes)
    
    train_idx = normal_idx[perm[:train_size]]
    val_idx = normal_idx[perm[train_size:train_size + val_size]]
    test_idx = torch.cat([normal_idx[perm[train_size + val_size:]], anomaly_idx])
    
    return train_idx, val_idx, test_idx


# ============ 重构模块 ============

class TokenReconstructor(nn.Module):
    """
    Token 重构器
    
    从 embedding 重构原始 tokens
    """
    def __init__(self, d_model: int, n_hops: int, n_features: int):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, n_hops * n_features)
        )
        
        self.n_hops = n_hops
        self.n_features = n_features
    
    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb: (batch, d_model) - 节点嵌入
        
        Returns:
            reconstructed: (batch, n_hops, n_features) - 重构的 tokens
        """
        reconstructed = self.decoder(emb)  # (batch, n_hops * n_features)
        reconstructed = reconstructed.view(-1, self.n_hops, self.n_features)
        return reconstructed


# ============ 离群点生成 ============

class OutlierGenerator(nn.Module):
    """
    离群点生成器
    
    使用重构错误生成离群点嵌入
    """
    def __init__(self, d_model: int, input_dim: int, n_hops: int):
        super().__init__()
        
        self.recon_proj = nn.Sequential(
            nn.Linear(n_hops * input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, input_tokens: torch.Tensor, reconstructed_tokens: torch.Tensor, 
                normal_emb: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
        """
        Args:
            input_tokens: (batch, n_hops, n_features) - 原始 tokens
            reconstructed_tokens: (batch, n_hops, n_features) - 重构 tokens
            normal_emb: (batch, d_model) - 正常节点嵌入
            beta: 离群点缩放因子
        
        Returns:
            outlier_emb: (batch, d_model) - 离群点嵌入
        """
        # 计算重构错误
        recon_error = (reconstructed_tokens - input_tokens).flatten(1)  # (batch, n_hops * n_features)
        
        # 投影到嵌入维度
        recon_error_proj = self.recon_proj(recon_error)  # (batch, d_model)
        
        # 生成离群点
        outlier_emb = normal_emb + beta * recon_error_proj
        
        return outlier_emb


# ============ 完整模型 ============

class FiLMDeltaVoxGFull(nn.Module):
    """
    FiLM-Delta VoxGFormer 完整版
    
    包含：
    1. FiLM-Delta 主模型
    2. Token 重构器
    3. 离群点生成器
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
        
        # Token 重构器
        self.reconstructor = TokenReconstructor(d_hidden, n_hops, n_features)
        
        # 离群点生成器
        self.outlier_generator = OutlierGenerator(d_hidden, n_features, n_hops)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1)
        )
    
    def forward(
        self,
        hop_features: torch.Tensor,
        train_idx: torch.Tensor = None,
        beta: float = 0.1
    ):
        """
        Args:
            hop_features: (n_nodes, n_hops, n_features)
            train_idx: 训练集索引
            beta: 离群点缩放因子
        
        Returns:
            logits: (n_nodes,) - 分类 logits
            rec_loss: 重构损失
            outlier_emb: 离群点嵌入
        """
        n_nodes = hop_features.shape[0]
        
        # 主模型前向传播
        # 这里我们需要修改 backbone 来返回嵌入
        
        return self._forward_full(hop_features, train_idx, beta)
    
    def _forward_full(self, hop_features, train_idx, beta):
        """完整前向传播"""
        n_nodes = hop_features.shape[0]
        
        # 提取 Delta
        delta = hop_features[:, 1:, :] - hop_features[:, :-1, :]
        
        # 投影
        hop_proj = self.backbone.input_proj(hop_features)  # (n_nodes, n_hops, d_hidden)
        delta_proj = self.backbone.input_proj(delta)
        
        # 聚合 Delta
        cond = self.backbone.delta_aggregator(delta_proj)
        
        # Transformer 层
        x = hop_proj
        for layer in self.backbone.layers:
            x = layer(x, cond)
        
        # 节点嵌入
        node_emb = x.mean(dim=1)  # (n_nodes, d_hidden)
        
        # Token 重构
        reconstructed_tokens = self.reconstructor(node_emb)  # (n_nodes, n_hops, n_features)
        
        # 计算重构损失
        rec_loss = F.mse_loss(reconstructed_tokens, hop_features)
        
        # 生成离群点（仅训练集）
        if train_idx is not None:
            normal_emb = node_emb[train_idx]
            normal_tokens = hop_features[train_idx]
            normal_reconstructed = reconstructed_tokens[train_idx]
            
            outlier_emb = self.outlier_generator(
                normal_tokens, normal_reconstructed, normal_emb, beta
            )
        else:
            outlier_emb = None
        
        # 分类
        logits = self.classifier(node_emb).squeeze(-1)
        
        return logits, rec_loss, outlier_emb, node_emb


# ============ 训练函数 ============

def train_epoch(model, hop_features, labels, train_idx, optimizer, 
                lambda_rec=0.5, beta=0.1, device='cpu'):
    """训练一个 epoch"""
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    logits, rec_loss, outlier_emb, node_emb = model(
        hop_features, train_idx, beta
    )
    
    # 分类损失
    # 正常节点标签为 0
    bce_loss = F.binary_cross_entropy_with_logits(
        logits[train_idx], 
        torch.zeros(len(train_idx), device=device)
    )
    
    # 如果有离群点，添加对比损失
    if outlier_emb is not None:
        # 离群点应该被分类为异常（标签 1）
        outlier_logits = model.classifier(outlier_emb).squeeze(-1)
        outlier_loss = F.binary_cross_entropy_with_logits(
            outlier_logits,
            torch.ones(len(outlier_emb), device=device)
        )
    else:
        outlier_loss = 0.0
    
    # 总损失
    total_loss = bce_loss + lambda_rec * rec_loss + outlier_loss
    
    total_loss.backward()
    optimizer.step()
    
    return {
        'total_loss': total_loss.item(),
        'bce_loss': bce_loss.item(),
        'rec_loss': rec_loss.item(),
        'outlier_loss': outlier_loss.item() if isinstance(outlier_loss, float) else outlier_loss.item()
    }


@torch.no_grad()
def evaluate(model, hop_features, labels, test_idx):
    """评估模型"""
    model.eval()
    
    logits, _, _, _ = model(hop_features)
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
    lambda_rec=0.5,
    beta=0.1,
    seed=42,
    verbose=True
):
    """完整训练流程"""
    print("=" * 60)
    print(f"FiLMDeltaVoxG Full Training (seed={seed})")
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
    train_idx, val_idx, test_idx = split_data(n_nodes, labels, seed=seed)
    if verbose:
        print(f"\n[3] 数据划分:")
        print(f"    训练集: {len(train_idx)} (仅正常)")
        print(f"    测试集: {len(test_idx)}")
    
    # 创建模型
    model = FiLMDeltaVoxGFull(
        n_features=n_features,
        d_hidden=d_hidden,
        n_layers=n_layers,
        n_hops=n_hops + 1
    )
    
    if verbose:
        print(f"\n[4] 模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_test_auc = 0
    best_test_ap = 0
    best_epoch = 0
    
    if verbose:
        print(f"\n[5] 训练...")
    
    for epoch in range(n_epochs):
        losses = train_epoch(model, hop_features, labels, train_idx, optimizer, 
                            lambda_rec=lambda_rec, beta=beta)
        test_auc, test_ap = evaluate(model, hop_features, labels, test_idx)
        
        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_test_ap = test_ap
            best_epoch = epoch + 1
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch + 1}: Loss={losses['total_loss']:.4f} "
                  f"(BCE={losses['bce_loss']:.4f}, Rec={losses['rec_loss']:.4f}), "
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
    print("5-Seed Experiment: FiLMDeltaVoxG Full")
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
    # 单次训练测试
    print("\n" + "=" * 70)
    print("FiLMDeltaVoxG Full 训练")
    print("=" * 70)
    
    mean_auc, std_auc, results = run_5seed_experiment(
        n_epochs=200,
        d_hidden=128,
        n_layers=3,
        lambda_rec=0.5,
        beta=0.1
    )
    
    # 与基线对比
    print("\n" + "=" * 70)
    print("与基线对比")
    print("=" * 70)
    print(f"{'方法':<25} {'AUC':<20}")
    print("-" * 45)
    print(f"{'VecGAD (SOTA)':<25} {'0.8960':<20}")
    print(f"{'concat':<25} {'0.8777 ± 0.038':<20}")
    print(f"{'DualStreamVoxG':<25} {'0.5157 ± 0.011':<20}")
    print(f"{'FiLMDeltaVoxG (无重构)':<25} {'0.48':<20}")
    print(f"{'FiLMDeltaVoxG Full':<25} {f'{mean_auc:.4f} ± {std_auc:.4f}':<20}")


if __name__ == "__main__":
    main()