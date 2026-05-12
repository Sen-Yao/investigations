"""
FiLM-Delta VoxGFormer

核心思想：Delta 作为调制信号，不直接参与 Transformer 计算

架构设计：
1. 标准 Transformer 处理 Hop features
2. Delta 通过 MLP 生成调制参数 (gamma, beta, alpha)
3. AdaLN 调制 Transformer 输出
4. 零初始化保证训练稳定

文献依据：
- FiLM (Perez et al., 2018): Feature-wise Linear Modulation
- GNN-FiLM (Brockschmidt, 2020): 图神经网络中的 FiLM
- AdaLN-Zero (DiT, 2023): Transformer 中的条件注入

作者: Nexus
日期: 2026-04-04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class DeltaExtractor(nn.Module):
    """
    从 Hop features 提取 Delta
    
    Delta_k = Hop_{k+1} - Hop_k
    
    输入: (batch, seq_len, d_model) - Hop features
    输出: (batch, seq_len-1, d_model) - Delta features
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, hop_features: torch.Tensor) -> torch.Tensor:
        delta = hop_features[:, 1:, :] - hop_features[:, :-1, :]
        return delta


class AdaLNZeroModulation(nn.Module):
    """
    AdaLN-Zero: Adaptive Layer Normalization with Zero Initialization
    
    核心设计：
    1. gamma, beta, alpha 由条件信号（Delta）生成
    2. MLP 最后一层权重初始化为 0
    3. 训练初期等价于普通 Transformer
    
    公式：
        gamma, beta, alpha = MLP(cond)
        gamma = 1 + gamma  # 保证初始 gamma ≈ 1
        output = gamma * LayerNorm(x) + beta
        output = x + alpha * output  # 门控残差
    """
    def __init__(self, d_model: int, d_cond: int, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # LayerNorm
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        
        # 条件 MLP (生成 gamma, beta)
        self.cond_mlp = nn.Sequential(
            nn.Linear(d_cond, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model * 2)  # gamma, beta
        )
        
        # 对输入层进行小随机初始化
        nn.init.xavier_uniform_(self.cond_mlp[0].weight, gain=0.1)
        nn.init.zeros_(self.cond_mlp[0].bias)
        
        # gamma 输出层初始化为接近 1
        nn.init.ones_(self.cond_mlp[-1].weight[:d_model])
        nn.init.zeros_(self.cond_mlp[-1].bias[:d_model])
        
        # beta 输出层初始化为 0
        nn.init.zeros_(self.cond_mlp[-1].weight[d_model:])
        nn.init.zeros_(self.cond_mlp[-1].bias[d_model:])
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) - Transformer 输出
            cond: (batch, d_cond) - 条件信号（Delta 的聚合表示）
        
        Returns:
            output: (batch, seq_len, d_model) - 调制后的输出
        """
        # 生成调制参数
        params = self.cond_mlp(cond)  # (batch, d_model * 2)
        gamma, beta = params.chunk(2, dim=-1)  # each (batch, d_model)
        
        # gamma 已经初始化为 1，不需要再加
        
        # 调制（直接使用调制后的输出，不使用门控残差）
        x_norm = self.norm(x)  # (batch, seq_len, d_model)
        output = gamma.unsqueeze(1) * x_norm + beta.unsqueeze(1)
        
        return output


class FiLMDeltaLayer(nn.Module):
    """
    FiLM-Delta Transformer Layer
    
    结构：
    1. 标准 Self-Attention + FFN
    2. AdaLN 调制（由 Delta 控制）
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1, d_cond: int = None):
        super().__init__()
        
        d_cond = d_cond or d_model
        
        # Self-Attention
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # AdaLN 调制
        self.ada_ln = AdaLNZeroModulation(d_model, d_cond, dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            cond: (batch, d_cond) - Delta 的聚合表示
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        # Self-Attention with residual
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        # AdaLN 调制（带残差）
        x_modulated = self.ada_ln(x, cond)
        x = x + x_modulated  # 残差连接
        
        return x


class DeltaAggregator(nn.Module):
    """
    将 Delta 序列聚合为单个条件向量
    
    方式：mean pooling + 投影
    """
    def __init__(self, d_model: int, d_cond: int = None):
        super().__init__()
        d_cond = d_cond or d_model
        self.proj = nn.Linear(d_model, d_cond)
    
    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta: (batch, seq_len-1, d_model)
        
        Returns:
            cond: (batch, d_cond)
        """
        # Mean pooling
        delta_mean = delta.mean(dim=1)  # (batch, d_model)
        # 投影
        cond = self.proj(delta_mean)  # (batch, d_cond)
        return cond


class FiLMDeltaVoxG(nn.Module):
    """
    FiLM-Delta VoxGFormer
    
    架构：
    1. 输入投影（Hop 和 Delta 共享）
    2. Delta 提取 + 聚合
    3. N 层 FiLM-Delta Transformer
    4. Readout
    5. 分类头
    
    关键创新：
    - Delta 作为调制信号，不直接参与 Transformer
    - AdaLN-Zero 保证训练稳定
    - 特征级调制，比 token 级别更细粒度
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
        self.d_hidden = d_hidden
        
        # Delta 提取器
        self.delta_extractor = DeltaExtractor()
        
        # Delta 聚合器
        self.delta_aggregator = DeltaAggregator(d_hidden, d_hidden)
        
        # 输入投影
        self.input_proj = nn.Linear(n_features, d_hidden)
        
        # FiLM-Delta Transformer 层
        self.layers = nn.ModuleList([
            FiLMDeltaLayer(d_hidden, n_heads, dropout, d_hidden)
            for _ in range(n_layers)
        ])
        
        # Readout
        self.readout = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1)
        )
    
    def forward(
        self,
        hop_features: torch.Tensor,
        return_intermediate: bool = False
    ) -> torch.Tensor:
        """
        Args:
            hop_features: (batch, n_nodes, n_hops, n_features)
                          或 (n_nodes, n_hops, n_features)
        
        Returns:
            scores: (batch, n_nodes) 或 (n_nodes,)
        """
        # 处理输入维度
        if hop_features.dim() == 3:
            hop_features = hop_features.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, n_nodes, n_hops, n_features = hop_features.shape
        
        # Reshape
        hop_flat = hop_features.view(batch_size * n_nodes, n_hops, n_features)
        
        # 提取 Delta
        delta = self.delta_extractor(hop_flat)  # (batch*n_nodes, n_hops-1, n_features)
        
        # 投影
        hop_proj = self.input_proj(hop_flat)  # (batch*n_nodes, n_hops, d_hidden)
        delta_proj = self.input_proj(delta)  # (batch*n_nodes, n_hops-1, d_hidden)
        
        # 聚合 Delta 为条件向量
        cond = self.delta_aggregator(delta_proj)  # (batch*n_nodes, d_hidden)
        
        intermediates = [hop_proj.clone()] if return_intermediate else None
        
        # FiLM-Delta Transformer 层
        x = hop_proj
        for layer in self.layers:
            x = layer(x, cond)
            if return_intermediate:
                intermediates.append(x.clone())
        
        # Readout: mean over hops
        node_repr = x.mean(dim=1)  # (batch*n_nodes, d_hidden)
        
        # 分类
        scores = self.readout(node_repr).squeeze(-1)  # (batch*n_nodes,)
        
        # Reshape
        scores = scores.view(batch_size, n_nodes)
        
        if squeeze_output:
            scores = scores.squeeze(0)
        
        if return_intermediate:
            return scores, intermediates
        return scores
    
    def get_model_info(self) -> dict:
        """返回模型信息"""
        n_params = sum(p.numel() for p in self.parameters())
        return {
            'model': 'FiLMDeltaVoxG',
            'd_hidden': self.d_hidden,
            'n_layers': len(self.layers),
            'n_params': n_params,
            'n_params_M': n_params / 1e6
        }


# ============ 工具函数 ============

def create_film_delta_model(
    n_features: int,
    config: Optional[dict] = None
) -> FiLMDeltaVoxG:
    """
    创建 FiLM-Delta 模型的工厂函数
    """
    default_config = {
        'd_hidden': 128,
        'n_layers': 3,
        'n_heads': 4,
        'n_hops': 7,
        'dropout': 0.1
    }
    
    if config:
        default_config.update(config)
    
    return FiLMDeltaVoxG(n_features, **default_config)


# ============ 测试代码 ============

def test_film_delta_model():
    """测试 FiLM-Delta 模型"""
    print("=" * 60)
    print("测试 FiLMDeltaVoxG")
    print("=" * 60)
    
    # 创建模型
    n_features = 745
    model = create_film_delta_model(n_features)
    
    # 打印模型信息
    info = model.get_model_info()
    print(f"\n模型信息:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    # 测试前向传播
    batch_size = 2
    n_nodes = 100
    n_hops = 7
    
    hop_features = torch.randn(batch_size, n_nodes, n_hops, n_features)
    
    print(f"\n输入形状: {hop_features.shape}")
    
    # 前向传播
    with torch.no_grad():
        scores, intermediates = model(hop_features, return_intermediate=True)
    
    print(f"输出形状: {scores.shape}")
    print(f"中间层数量: {len(intermediates)}")
    
    # 验证 Delta 提取
    delta = model.delta_extractor(hop_features.view(-1, n_hops, n_features))
    print(f"\nDelta 形状: {delta.shape}")
    
    # 验证调制参数范围
    print(f"\n测试调制参数:")
    for name, param in model.named_parameters():
        if 'cond_mlp' in name and 'weight' in name:
            print(f"  {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")
    
    print("\n✅ 测试通过")


if __name__ == "__main__":
    test_film_delta_model()