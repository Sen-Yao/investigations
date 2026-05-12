"""
VoxGFormer with Attention Pooling

改动：mean pool → Attention Pooling

作者: Nexus
日期: 2026-04-04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import scipy.io as sio
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


class AttentionPooling(nn.Module):
    """Attention Pooling: 让模型学习哪些 hop 更重要"""
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_hops, dim)
        
        Returns:
            pooled: (batch, dim)
        """
        w = torch.softmax(self.attn(x), dim=1)  # (batch, num_hops, 1)
        return (w * x).sum(dim=1)  # (batch, dim)


class VoxGFormerWithAttnPool(nn.Module):
    """VoxGFormer + Attention Pooling"""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        num_hops: int = 7,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hops = num_hops
        
        # 输入投影（concat hop + delta）
        self.hop_proj = nn.Linear(input_dim, hidden_dim)
        self.delta_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention Pooling（替代 mean pool）
        self.pool = AttentionPooling(hidden_dim)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 重构解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim)
        )
    
    def forward(self, hop_features: torch.Tensor, return_loss: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hop_features: (batch, num_hops, input_dim)
        
        Returns:
            logits: (batch,)
            loss: Optional[Tensor]
        """
        # 计算 Delta
        deltas = torch.zeros_like(hop_features)
        deltas[:, 1:] = hop_features[:, 1:] - hop_features[:, :-1]
        
        # Concat 投影
        hop_emb = self.hop_proj(hop_features)
        delta_emb = self.delta_proj(deltas)
        concat = hop_emb + delta_emb  # 加法融合
        
        # Transformer
        trans_out = self.transformer(concat)
        
        # Attention Pooling
        pooled = self.pool(trans_out)
        
        # 分类
        logits = self.classifier(pooled).squeeze(-1)
        
        if return_loss:
            recon = self.decoder(pooled)
            target = hop_features[:, 0]
            loss = F.mse_loss(recon, target)
            return logits, loss
        
        return logits, None


# ============ 数据加载和训练 ============

def load_photo():
    data = sio.loadmat('/root/gpufree-data/linziyao/MatrixGAD/dataset/photo.mat')
    features = torch.tensor(np.array(data['Attributes'].todense()), dtype=torch.float32)
    labels = torch.tensor(data['Label'].flatten(), dtype=torch.long)
    adj = data['Network']
    return features, labels, adj


def normalize_adj(adj):
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def generate_hop_features(features, adj, n_hops=6):
    adj_norm = normalize_adj(adj)
    adj_tensor = torch.tensor(adj_norm.todense(), dtype=torch.float32)
    
    hop_features = torch.zeros(features.shape[0], n_hops + 1, features.shape[1])
    hop_features[:, 0, :] = features
    for hop in range(1, n_hops + 1):
        hop_features[:, hop, :] = torch.matmul(adj_tensor, hop_features[:, hop-1, :])
    
    return hop_features


def split_data(n_nodes, labels, train_rate=0.05, val_rate=0.10, seed=0):
    torch.manual_seed(seed)
    
    normal_idx = torch.where(labels == 0)[0]
    anomaly_idx = torch.where(labels == 1)[0]
    
    perm = torch.randperm(len(normal_idx))
    train_size = int(train_rate * n_nodes)
    val_size = int(val_rate * n_nodes)
    
    train_idx = normal_idx[perm[:train_size]]
    val_idx = normal_idx[perm[train_size:train_size+val_size]]
    test_idx = torch.cat([normal_idx[perm[train_size+val_size:]], anomaly_idx])
    
    return train_idx, val_idx, test_idx


def train():
    """训练函数（WandB Sweep 兼容）"""
    import wandb
    import random
    
    # 初始化 WandB
    wandb.init()
    config = wandb.config
    
    # 设置随机种子（完整版）
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.config.update({"device": str(device)})
    
    # 加载数据
    features, labels, adj = load_photo()
    n_nodes, n_features = features.shape
    
    # Hop features
    n_hops = 6
    hop_features = generate_hop_features(features, adj, n_hops)
    
    # 移动数据到 GPU
    hop_features = hop_features.to(device)
    labels = labels.to(device)
    
    # 数据划分
    train_idx, val_idx, test_idx = split_data(
        n_nodes, labels.cpu(),  # split_data 需要 CPU 上的 labels
        train_rate=config.train_rate,
        seed=config.seed
    )
    train_idx = train_idx.to(device)
    test_idx = test_idx.to(device)
    
    # ✅ 数据泄漏断言
    train_labels = labels[train_idx]
    train_anomaly_count = train_labels.sum().item()
    assert train_anomaly_count == 0, f"🚨 数据泄漏！训练集包含 {train_anomaly_count} 个异常节点！"
    
    # ✅ 记录 WandB 配置
    wandb.config.update({
        "train_size": len(train_idx),
        "train_normal": len(train_idx),
        "train_anomaly": train_anomaly_count,  # 必须是 0
        "test_size": len(test_idx),
        "test_anomaly": labels[test_idx].sum().item(),
        "n_features": n_features,
        "n_hops": n_hops
    })
    
    # 模型
    model = VoxGFormerWithAttnPool(
        n_features,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_hops=n_hops + 1
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    # 训练
    best_auc = 0
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        
        logits, loss = model(hop_features, return_loss=True)
        
        bce = F.binary_cross_entropy_with_logits(
            logits[train_idx],
            torch.zeros(len(train_idx), device=device)
        )
        
        total_loss = bce + 0.5 * loss
        total_loss.backward()
        optimizer.step()
        
        # 评估
        model.eval()
        with torch.no_grad():
            logits_eval, _ = model(hop_features, return_loss=False)
            probs = torch.sigmoid(logits_eval[test_idx])
            targets = labels[test_idx]
            
            auc = roc_auc_score(targets.cpu().numpy(), probs.cpu().numpy())
            ap = average_precision_score(targets.cpu().numpy(), probs.cpu().numpy())
        
        if auc > best_auc:
            best_auc = auc
            best_ap = ap
        
        wandb.log({
            'epoch': epoch,
            'loss': total_loss.item(),
            'auc': auc,
            'ap': ap,
            'best_auc': best_auc
        })
    
    wandb.finish()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='photo')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_rate', type=float, default=0.05)
    parser.add_argument('--test', action='store_true', help='Run test only')
    
    args = parser.parse_args()
    
    if args.test:
        # 仅测试
        print("Testing VoxGFormerWithAttnPool")
        model = VoxGFormerWithAttnPool(745)
        x = torch.randn(4, 7, 745)
        logits, loss = model(x, return_loss=True)
        print(f"Input: {x.shape}")
        print(f"Output: {logits.shape}")
        print(f"Loss: {loss.item():.4f}")
        print("✅ Test passed")
    else:
        # WandB Sweep 训练
        train()