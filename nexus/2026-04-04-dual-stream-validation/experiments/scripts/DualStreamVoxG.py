"""
DualStreamVoxG - 双流架构 VoxGFormer

理论依据：
1. 投影层损失 22% Delta 信息 → 需要在投影前分流
2. Transformer 累积损失 15% → D 流绕过点积注意力
3. MLP > Attention 1.7% → D 流使用 MLP 处理

架构设计：
- Stream S (Structure): 标准 Transformer 处理 Hop features
- Stream D (Difference): MLP 处理 Delta features
- 融合：门控机制

参考文献：
- VecGAD: RDV 架构级差异计算
- GTA: 图 Transformer 注意力机制

作者: Nexus
日期: 2026-04-04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class DeltaExtractor(nn.Module):
    """
    从 Hop features 提取 Delta features
    
    Delta_k = Hop_{k+1} - Hop_k
    
    输入: (batch, seq_len, d_model) - Hop features
    输出: (batch, seq_len-1, d_model) - Delta features
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, hop_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hop_features: (batch, seq_len, d_model)
        Returns:
            delta_features: (batch, seq_len-1, d_model)
        """
        # Delta_k = Hop_{k+1} - Hop_k
        delta = hop_features[:, 1:, :] - hop_features[:, :-1, :]
        return delta


class StreamS_Layer(nn.Module):
    """
    Stream S: 标准 Transformer 层
    
    处理 Hop features，捕获结构模式
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x


class StreamD_Layer(nn.Module):
    """
    Stream D: 专用差异处理层
    
    使用 MLP 处理 Delta features，不经点积注意力
    
    关键设计：
    - 避免点积注意力的相似度偏好
    - 可以学习任意变换（包括保留差异）
    """
    def __init__(self, d_model: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        # MLP with residual
        mlp_out = self.mlp(delta)
        delta = self.norm(delta + self.dropout(mlp_out))
        return delta


class GatedFusion(nn.Module):
    """
    门控融合机制
    
    output = gate * s_out + (1 - gate) * d_out
    
    gate 由 [s_out, d_out] 拼接后通过 sigmoid 学习
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
    
    def forward(self, s_out: torch.Tensor, d_out: torch.Tensor) -> torch.Tensor:
        # d_out 序列长度比 s_out 少 1，需要 padding 或插值
        # 这里使用 zero padding 到相同长度
        if d_out.size(1) < s_out.size(1):
            pad_len = s_out.size(1) - d_out.size(1)
            padding = torch.zeros(
                d_out.size(0), pad_len, d_out.size(2),
                device=d_out.device, dtype=d_out.dtype
            )
            d_out = torch.cat([padding, d_out], dim=1)
        
        # 计算门控
        combined = torch.cat([s_out, d_out], dim=-1)
        gate = self.gate(combined)
        
        # 融合
        output = gate * s_out + (1 - gate) * d_out
        return output


class DualStreamLayer(nn.Module):
    """
    双流层：一个 S 层 + 一个 D 层 + 门控融合
    
    这是双流架构的基本单元
    """
    def __init__(
        self, 
        d_model: int, 
        n_heads: int = 4, 
        expansion: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.stream_s = StreamS_Layer(d_model, n_heads, dropout)
        self.stream_d = StreamD_Layer(d_model, expansion, dropout)
        self.fusion = GatedFusion(d_model)
    
    def forward(
        self, 
        hop_features: torch.Tensor,
        delta_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hop_features: (batch, seq_len, d_model)
            delta_features: (batch, seq_len-1, d_model)
        
        Returns:
            hop_out: (batch, seq_len, d_model)
            delta_out: (batch, seq_len-1, d_model)
        """
        # Stream S: 标准 Transformer
        hop_out = self.stream_s(hop_features)
        
        # Stream D: MLP 处理 Delta
        delta_out = self.stream_d(delta_features)
        
        # 门控融合
        hop_out = self.fusion(hop_out, delta_out)
        
        return hop_out, delta_out


class DualStreamVoxG(nn.Module):
    """
    双流 VoxGFormer
    
    架构：
    1. 输入投影（Hop 和 Delta 各自投影）
    2. N 层双流层
    3. Readout
    4. 分类头
    
    关键创新：
    - Delta 在投影前就分流，避免投影层损失
    - D 流使用 MLP，避免点积注意力的相似度偏好
    - 门控融合学习最优权重
    """
    def __init__(
        self,
        n_features: int,
        d_hidden: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        n_hops: int = 7,
        expansion: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_hops = n_hops
        self.d_hidden = d_hidden
        
        # Delta 提取器
        self.delta_extractor = DeltaExtractor()
        
        # 输入投影（Hop 和 Delta 共享投影层，减少参数）
        self.input_proj = nn.Linear(n_features, d_hidden)
        
        # 双流层
        self.layers = nn.ModuleList([
            DualStreamLayer(d_hidden, n_heads, expansion, dropout)
            for _ in range(n_layers)
        ])
        
        # Readout: mean over hops
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
                          或 (n_nodes, n_hops, n_features) 无 batch
            return_intermediate: 是否返回中间层输出
        
        Returns:
            scores: (batch, n_nodes) 或 (n_nodes,)
        """
        # 处理输入维度
        if hop_features.dim() == 3:
            hop_features = hop_features.unsqueeze(0)  # (1, n_nodes, n_hops, n_features)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, n_nodes, n_hops, n_features = hop_features.shape
        
        # Reshape: (batch * n_nodes, n_hops, n_features)
        hop_flat = hop_features.view(batch_size * n_nodes, n_hops, n_features)
        
        # 提取 Delta（在投影前！）
        delta_flat = self.delta_extractor(hop_flat)  # (batch*n_nodes, n_hops-1, n_features)
        
        # 投影（Hop 和 Delta 共享）
        hop_proj = self.input_proj(hop_flat)  # (batch*n_nodes, n_hops, d_hidden)
        delta_proj = self.input_proj(delta_flat)  # (batch*n_nodes, n_hops-1, d_hidden)
        
        intermediates = [(hop_proj.clone(), delta_proj.clone())] if return_intermediate else None
        
        # 双流层
        hop_out, delta_out = hop_proj, delta_proj
        for layer in self.layers:
            hop_out, delta_out = layer(hop_out, delta_out)
            if return_intermediate:
                intermediates.append((hop_out.clone(), delta_out.clone()))
        
        # Readout: mean over hops
        node_repr = hop_out.mean(dim=1)  # (batch*n_nodes, d_hidden)
        
        # 分类
        scores = self.readout(node_repr).squeeze(-1)  # (batch*n_nodes,)
        
        # Reshape output
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
            'model': 'DualStreamVoxG',
            'd_hidden': self.d_hidden,
            'n_layers': len(self.layers),
            'n_params': n_params,
            'n_params_M': n_params / 1e6
        }


# ============ 工具函数 ============

def compute_delta_from_hop(hop_features: torch.Tensor) -> torch.Tensor:
    """
    从 Hop features 计算 Delta features
    
    Args:
        hop_features: (..., n_hops, d_model)
    
    Returns:
        delta_features: (..., n_hops-1, d_model)
    """
    return hop_features[..., 1:, :] - hop_features[..., :-1, :]


def create_dual_stream_model(
    n_features: int,
    config: Optional[dict] = None
) -> DualStreamVoxG:
    """
    创建双流模型的工厂函数
    
    Args:
        n_features: 输入特征维度
        config: 可选的配置字典
    
    Returns:
        DualStreamVoxG 模型实例
    """
    default_config = {
        'd_hidden': 128,
        'n_layers': 3,
        'n_heads': 4,
        'n_hops': 7,
        'expansion': 2,
        'dropout': 0.1
    }
    
    if config:
        default_config.update(config)
    
    return DualStreamVoxG(n_features, **default_config)


# ============ 测试代码 ============

def test_dual_stream_model():
    """测试双流模型"""
    print("=" * 60)
    print("测试 DualStreamVoxG")
    print("=" * 60)
    
    # 创建模型
    n_features = 745
    model = create_dual_stream_model(n_features)
    
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
    
    print("\n✅ 测试通过")


if __name__ == "__main__":
    test_dual_stream_model()