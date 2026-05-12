"""
BypassVoxG: 高维 Bypass 异常评分架构

核心创新：
1. Transformer 通道：处理低频全局语义（128维）
2. Bypass 通道：保留高频局部细节（745维直接计算）
3. 门控融合：自适应组合两通道
4. 多尺度重构：原始特征 + Delta + 统计量

文献依据：
- ResNet (2016): Skip Connection
- HRNet (2019): 高分辨率保持
- FAGCN (2021): 高/低频分离
- PatchCore (2022): 高维特征对比

作者: Nexus
日期: 2026-04-04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


# ============ Bypass 通道组件 ============

class LearnableProbes(nn.Module):
    """
    可学习探针：在高维空间中寻找判别方向
    
    原理：在 745 维空间中投影到低维，但投影矩阵是可学习的，
    可以学习到最具判别力的方向，而不是 PCA 的最大方差方向。
    """
    def __init__(self, input_dim: int, num_probes: int = 8):
        super().__init__()
        
        self.num_probes = num_probes
        self.input_dim = input_dim
        
        # 可学习探针向量
        self.probes = nn.Parameter(torch.randn(num_probes, input_dim))
        
        # 正交初始化
        with torch.no_grad():
            nn.init.orthogonal_(self.probes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        
        Returns:
            projections: (batch, seq_len, num_probes)
        """
        # 归一化探针
        probes_norm = F.normalize(self.probes, dim=-1)
        
        # 投影
        projections = torch.einsum('bnh,ph->bnp', x, probes_norm)
        
        return projections


class HighDimStatisticsExtractor(nn.Module):
    """
    高维统计量提取器
    
    在原始 745 维空间直接计算统计量，零信息损失
    """
    def __init__(self, num_hops: int = 7):
        super().__init__()
        
        self.num_hops = num_hops
    
    def forward(self, hop_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hop_features: (batch, num_hops, input_dim)
        
        Returns:
            hop_stats: (batch, num_hops * 4) - mean, std, min, max per hop
            delta_stats: (batch, (num_hops-1) * 4) - delta statistics
        """
        # Hop 统计量
        hop_mean = hop_features.mean(dim=-1)           # (batch, num_hops)
        hop_std = hop_features.std(dim=-1)             # (batch, num_hops)
        hop_min = hop_features.min(dim=-1).values      # (batch, num_hops)
        hop_max = hop_features.max(dim=-1).values      # (batch, num_hops)
        
        hop_stats = torch.cat([hop_mean, hop_std, hop_min, hop_max], dim=-1)
        
        # Delta 统计量
        deltas = hop_features[:, 1:, :] - hop_features[:, :-1, :]
        
        delta_mean = deltas.mean(dim=-1)               # (batch, num_hops-1)
        delta_std = deltas.std(dim=-1)                 # (batch, num_hops-1)
        delta_min = deltas.min(dim=-1).values          # (batch, num_hops-1)
        delta_max = deltas.max(dim=-1).values          # (batch, num_hops-1)
        
        delta_stats = torch.cat([delta_mean, delta_std, delta_min, delta_max], dim=-1)
        
        return hop_stats, delta_stats


class BypassChannel(nn.Module):
    """
    Bypass 通道
    
    在高维空间直接操作，保留 Transformer 丢失的高频信息
    """
    def __init__(self, input_dim: int, num_hops: int = 7, num_probes: int = 8, output_dim: int = 64):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_hops = num_hops
        self.num_probes = num_probes
        
        # 可学习探针
        self.hop_probes = LearnableProbes(input_dim, num_probes)
        self.delta_probes = LearnableProbes(input_dim, num_probes)
        
        # 统计量提取
        self.stats_extractor = HighDimStatisticsExtractor(num_hops)
        
        # 统计量维度
        # hop_stats: num_hops * 4
        # delta_stats: (num_hops-1) * 4
        # probes: num_hops * num_probes + (num_hops-1) * num_probes
        stats_dim = num_hops * 4 + (num_hops - 1) * 4
        probes_dim = num_hops * num_probes + (num_hops - 1) * num_probes
        
        # 编码器
        # 关键改进：压缩到与 Transformer 同等维度
        self.stats_encoder = nn.Sequential(
            nn.Linear(stats_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.probes_encoder = nn.Sequential(
            nn.Linear(probes_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
    
    def forward(self, hop_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hop_features: (batch, num_hops, input_dim)
        
        Returns:
            bypass_emb: (batch, output_dim)
        """
        batch_size = hop_features.shape[0]
        
        # 1. 提取统计量
        hop_stats, delta_stats = self.stats_extractor(hop_features)
        all_stats = torch.cat([hop_stats, delta_stats], dim=-1)
        stats_emb = self.stats_encoder(all_stats)
        
        # 2. 探针投影
        hop_proj = self.hop_probes(hop_features)  # (batch, num_hops, num_probes)
        
        deltas = hop_features[:, 1:, :] - hop_features[:, :-1, :]
        delta_proj = self.delta_probes(deltas)     # (batch, num_hops-1, num_probes)
        
        all_probes = torch.cat([
            hop_proj.reshape(batch_size, -1),
            delta_proj.reshape(batch_size, -1)
        ], dim=-1)
        probes_emb = self.probes_encoder(all_probes)
        
        # 3. 融合
        bypass_emb = self.fusion(torch.cat([stats_emb, probes_emb], dim=-1))
        
        return bypass_emb


# ============ Transformer 通道 ============

class TransformerChannel(nn.Module):
    """
    Transformer 通道
    
    标准 VoxGFormer 架构，处理低频全局语义
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3, 
                 num_heads: int = 4, num_hops: int = 7, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 输入投影（concat hop + delta）
        self.input_proj = nn.Linear(input_dim * 2, hidden_dim)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出池化
        self.pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, hop_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hop_features: (batch, num_hops, input_dim)
        
        Returns:
            trans_emb: (batch, hidden_dim)
        """
        # 计算 Delta
        deltas = torch.zeros_like(hop_features)
        deltas[:, 1:] = hop_features[:, 1:] - hop_features[:, :-1]
        
        # Concat
        concat_input = torch.cat([hop_features, deltas], dim=-1)
        
        # 投影
        tokens = self.input_proj(concat_input)
        
        # Transformer
        trans_out = self.transformer(tokens)
        
        # 池化
        trans_emb = trans_out.mean(dim=1)
        
        return trans_emb


# ============ 门控融合 ============

class GatedFusion(nn.Module):
    """
    门控融合
    
    自适应组合 Transformer 和 Bypass 特征
    
    公式：
        G = σ(W1 * H_trans + W2 * H_bypass + b)
        H_out = G ⊙ H_trans + (1-G) ⊙ H_bypass
    """
    def __init__(self, trans_dim: int, bypass_dim: int, output_dim: int):
        super().__init__()
        
        # 先对齐维度
        self.trans_proj = nn.Linear(trans_dim, output_dim)
        self.bypass_proj = nn.Linear(bypass_dim, output_dim)
        
        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, trans_emb: torch.Tensor, bypass_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trans_emb: (batch, trans_dim)
            bypass_emb: (batch, bypass_dim)
        
        Returns:
            fused: (batch, output_dim)
        """
        # 对齐维度
        h_trans = self.trans_proj(trans_emb)
        h_bypass = self.bypass_proj(bypass_emb)
        
        # 门控
        gate = self.gate(torch.cat([h_trans, h_bypass], dim=-1))
        
        # 融合
        fused = gate * h_trans + (1 - gate) * h_bypass
        
        return fused


# ============ 多尺度重构 ============

class MultiScaleDecoder(nn.Module):
    """
    多尺度重构解码器
    
    重构目标：
    1. 原始 ego 特征
    2. Delta 均值
    3. Hop 统计量
    """
    def __init__(self, hidden_dim: int, input_dim: int, num_hops: int = 7):
        super().__init__()
        
        # Ego 特征重构
        self.ego_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim)
        )
        
        # Delta 重构
        self.delta_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 统计量重构
        stats_dim = num_hops * 4  # mean, std, min, max per hop
        self.stats_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stats_dim)
        )
    
    def forward(self, z: torch.Tensor, hop_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (batch, hidden_dim) - 融合后的嵌入
            hop_features: (batch, num_hops, input_dim) - 原始输入
        
        Returns:
            ego_recon: (batch, input_dim)
            delta_recon: (batch, input_dim)
            stats_recon: (batch, num_hops * 4)
        """
        ego_recon = self.ego_decoder(z)
        delta_recon = self.delta_decoder(z)
        stats_recon = self.stats_decoder(z)
        
        return ego_recon, delta_recon, stats_recon


# ============ 离群点生成 ============

class OutlierGenerator(nn.Module):
    """
    离群点生成器
    
    方式：Mixup + Noise
    """
    def __init__(self, noise_std: float = 0.1, mixup_alpha: float = 0.2):
        super().__init__()
        self.noise_std = noise_std
        self.mixup_alpha = mixup_alpha
    
    def forward(self, x: torch.Tensor, method: str = 'noise') -> torch.Tensor:
        """
        Args:
            x: (batch, dim) - 正常样本特征
            method: 'noise' or 'mixup'
        
        Returns:
            outliers: (batch, dim) - 生成的离群点
        """
        if method == 'noise':
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        elif method == 'mixup':
            batch_size = x.size(0)
            indices = torch.randperm(batch_size, device=x.device)
            lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample(
                (batch_size,)
            ).to(x.device)
            lam = lam.view(-1, 1)
            return lam * x + (1 - lam) * x[indices]
        else:
            return x + torch.randn_like(x) * self.noise_std


# ============ 完整模型 ============

class BypassVoxG(nn.Module):
    """
    BypassVoxG: 高维 Bypass 异常评分架构
    
    架构：
    1. Transformer 通道：处理低频全局语义
    2. Bypass 通道：保留高频局部细节
    3. 门控融合：自适应组合
    4. 多尺度重构：增强判别力
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        bypass_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        num_hops: int = 7,
        num_probes: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hops = num_hops
        
        # Transformer 通道
        self.transformer_channel = TransformerChannel(
            input_dim, hidden_dim, num_layers, num_heads, num_hops, dropout
        )
        
        # Bypass 通道
        self.bypass_channel = BypassChannel(
            input_dim, num_hops, num_probes, bypass_dim
        )
        
        # 门控融合
        self.fusion = GatedFusion(hidden_dim, bypass_dim, hidden_dim)
        
        # 多尺度解码器
        self.decoder = MultiScaleDecoder(hidden_dim, input_dim, num_hops)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 离群点生成器
        self.outlier_generator = OutlierGenerator(noise_std=0.1, mixup_alpha=0.2)
        
        # 去噪 dropout
        self.input_dropout = nn.Dropout(0.2)
    
    def forward(
        self, 
        hop_features: torch.Tensor,
        return_loss: bool = False,
        generate_outliers: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hop_features: (batch, num_hops, input_dim)
            return_loss: 是否返回重构损失
            generate_outliers: 是否生成离群点
        
        Returns:
            logits: (batch,) - 异常分数
            loss: Optional[Tensor] - 重构损失
        """
        # 训练时：添加噪声（去噪自编码器）
        if self.training:
            hop_noisy = self.input_dropout(hop_features)
        else:
            hop_noisy = hop_features
        
        # Transformer 通道
        trans_emb = self.transformer_channel(hop_noisy)
        
        # Bypass 通道
        bypass_emb = self.bypass_channel(hop_noisy)
        
        # 门控融合
        fused = self.fusion(trans_emb, bypass_emb)
        
        # 分类
        logits = self.classifier(fused).squeeze(-1)
        
        if return_loss:
            # 多尺度重构（重构干净输入，而非噪声输入）
            ego_recon, delta_recon, stats_recon = self.decoder(fused, hop_features)
            
            # 目标
            ego_target = hop_features[:, 0]
            delta_target = (hop_features[:, 1:] - hop_features[:, :-1]).mean(dim=1)
            stats_target = torch.cat([
                hop_features.mean(dim=-1),
                hop_features.std(dim=-1),
                hop_features.min(dim=-1).values,
                hop_features.max(dim=-1).values
            ], dim=-1)
            
            # 正常样本损失
            loss_ego = F.mse_loss(ego_recon, ego_target)
            loss_delta = F.mse_loss(delta_recon, delta_target)
            loss_stats = F.mse_loss(stats_recon, stats_target)
            
            loss_normal = loss_ego + 0.5 * loss_delta + 0.3 * loss_stats
            
            # 离群点生成
            if generate_outliers:
                # 生成离群点嵌入
                outlier_emb = self.outlier_generator(fused, method='noise')
                outlier_logits = self.classifier(outlier_emb).squeeze(-1)
                
                # 离群点应该有高异常分数
                loss_outlier = F.binary_cross_entropy_with_logits(
                    outlier_logits, 
                    torch.ones_like(outlier_logits)
                )
                
                loss = loss_normal + 0.5 * loss_outlier
            else:
                loss = loss_normal
            
            return logits, loss
        
        return logits, None
    
    def get_model_info(self) -> dict:
        """返回模型信息"""
        n_params = sum(p.numel() for p in self.parameters())
        return {
            'model': 'BypassVoxG',
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'n_params': n_params,
            'n_params_M': n_params / 1e6
        }


# ============ 工厂函数 ============

def create_bypass_voxg(
    input_dim: int,
    config: Optional[dict] = None
) -> BypassVoxG:
    """创建 BypassVoxG 模型"""
    default_config = {
        'hidden_dim': 128,
        'bypass_dim': 64,
        'num_layers': 3,
        'num_heads': 4,
        'num_hops': 7,
        'num_probes': 8,
        'dropout': 0.1
    }
    
    if config:
        default_config.update(config)
    
    return BypassVoxG(input_dim, **default_config)


# ============ 测试 ============

def test_bypass_voxg():
    """测试 BypassVoxG"""
    print("=" * 60)
    print("测试 BypassVoxG")
    print("=" * 60)
    
    # 创建模型
    input_dim = 745
    model = create_bypass_voxg(input_dim)
    
    # 打印信息
    info = model.get_model_info()
    print(f"\n模型信息:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    # 测试前向传播
    batch_size = 4
    num_hops = 7
    
    hop_features = torch.randn(batch_size, num_hops, input_dim)
    
    print(f"\n输入: {hop_features.shape}")
    
    # 不带损失
    logits, _ = model(hop_features, return_loss=False)
    print(f"输出 (logits): {logits.shape}")
    
    # 带损失
    logits, loss = model(hop_features, return_loss=True)
    print(f"输出 (with loss): logits={logits.shape}, loss={loss.item():.4f}")
    
    print("\n✅ 测试通过")


if __name__ == "__main__":
    test_bypass_voxg()