#!/usr/bin/env python3
"""
Hop-Aware Attention 轻量化验证实验

验证两个方案：
1. 方案 A：相对距离偏置 (hop_bias = B[hop[j] - hop[i]])
2. 方案 B：Hop-Aware Cross-Attention

时间：~2 小时
目标：验证可行性，不追求最终性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import sys
import os

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# 方案 A：相对距离偏置
# ============================================================

class RelativeHopBiasAttention(nn.Module):
    """
    方案 A：相对距离偏置
    
    核心思想：hop_bias 基于 Hop 相对距离 (hop[j] - hop[i])
    
    物理意义：
    - hop[j] - hop[i] = 1 表示"比较相邻层"
    - 异常检测时，重点看相邻层的差异
    """
    
    def __init__(self, d_model: int, n_head: int, max_hop: int = 7):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.max_hop = max_hop
        
        # 标准 Q, K, V 投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 相对距离偏置（可学习）
        # B[k] 表示 hop 差值为 k 时的偏置
        # k 的范围是 [-(max_hop-1), max_hop-1]
        self.hop_bias = nn.Parameter(torch.zeros(2 * max_hop - 1))
        
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor, hop_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            hop_indices: [seq_len] 每个 token 的 hop 索引
            
        Returns:
            [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V 投影
        Q = self.W_q(x).view(batch_size, seq_len, self.n_head, -1).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_head, -1).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_head, -1).transpose(1, 2)
        
        # 标准注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_model ** 0.5)
        
        # 构建相对距离偏置矩阵
        # hop_diff[i, j] = hop[j] - hop[i]
        hop_i = hop_indices.unsqueeze(1)  # [seq_len, 1]
        hop_j = hop_indices.unsqueeze(0)  # [1, seq_len]
        hop_diff = hop_j - hop_i  # [seq_len, seq_len]
        
        # 将 hop_diff 映射到 bias 索引
        # hop_diff 范围：[-(max_hop-1), max_hop-1]
        # 索引范围：[0, 2*max_hop-2]
        bias_idx = hop_diff + (self.max_hop - 1)  # 偏移到非负索引
        bias_idx = bias_idx.clamp(0, 2 * self.max_hop - 2)
        
        # 获取偏置
        bias_matrix = self.hop_bias[bias_idx]  # [seq_len, seq_len]
        
        # 添加偏置到注意力分数
        scores = scores + bias_matrix.unsqueeze(0).unsqueeze(0)
        
        # Softmax 和输出
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.out_proj(out), attn, self.hop_bias.data


# ============================================================
# 方案 B：Hop-Aware Cross-Attention
# ============================================================

class HopCrossAttention(nn.Module):
    """
    方案 B：Hop-Aware Cross-Attention
    
    核心思想：以 hop_0 为 Query，其他 hop 为 Key/Value
    
    物理意义："以自我为中心，审视周围环境是否与我一致"
    """
    
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, hop_0: torch.Tensor, other_hops: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hop_0: [batch, d_model] - 自身特征
            other_hops: [batch, n_hops, d_model] - 其他 hop 特征
            
        Returns:
            output: [batch, d_model]
            attn: [batch, n_head, 1, n_hops] 注意力权重
        """
        batch_size = hop_0.shape[0]
        n_hops = other_hops.shape[1]
        
        # Q = hop_0, K/V = other_hops
        Q = self.W_q(hop_0).view(batch_size, 1, self.n_head, -1).transpose(1, 2)
        K = self.W_k(other_hops).view(batch_size, n_hops, self.n_head, -1).transpose(1, 2)
        V = self.W_v(other_hops).view(batch_size, n_hops, self.n_head, -1).transpose(1, 2)
        
        # Cross-Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_model ** 0.5)
        attn = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, 1, -1)
        
        return self.out_proj(out.squeeze(1)), attn


# ============================================================
# 验证实验 1：探测 hop_bias 是否有意义
# ============================================================

def experiment_1_probe_hop_bias():
    """
    实验 1：探测 hop_bias 能否学到有意义的模式
    
    构造任务：预测节点是否异常
    异常节点的定义：hop_0 与 hop_1 差异大
    
    预期：模型应该学会让 B[1] 更重要（相邻层比较）
    """
    print("\n" + "="*60)
    print("实验 1：探测 hop_bias 是否有意义")
    print("="*60)
    
    # 参数
    n_samples = 1000
    d_model = 64
    n_hops = 7
    n_epochs = 200
    
    # 生成数据
    # hop_0: 自身特征
    # hop_k: hop_0 + k * delta（模拟传播）
    hop_features = torch.randn(n_samples, n_hops, d_model)
    hop_indices = torch.arange(n_hops)
    
    # 定义异常：hop_0 与 hop_1 差异大
    delta_01 = hop_features[:, 1, :] - hop_features[:, 0, :]
    anomaly_scores = delta_01.norm(dim=-1)
    labels = (anomaly_scores > anomaly_scores.median()).long()
    
    # 模型
    model = RelativeHopBiasAttention(d_model, n_head=4, max_hop=n_hops)
    classifier = nn.Linear(d_model, 2)
    
    # 优化器
    optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 训练
    losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # 前向传播
        out, attn, hop_bias = model(hop_features, hop_indices)
        out = out.mean(dim=1)  # [batch, d_model]
        logits = classifier(out)
        
        loss = criterion(logits, labels)
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
    
    # 分析 hop_bias
    print("\n--- hop_bias 分析 ---")
    print(f"hop_bias = {hop_bias}")
    
    # 可视化 hop_bias
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    
    plt.subplot(1, 2, 2)
    x_labels = [f"B[{i-(n_hops-1)}]" for i in range(2*n_hops-1)]
    plt.bar(range(2*n_hops-1), hop_bias.numpy())
    plt.xticks(range(2*n_hops-1), x_labels, rotation=45)
    plt.xlabel('Hop Difference')
    plt.ylabel('Bias Value')
    plt.title('Learned Hop Bias')
    
    plt.tight_layout()
    plt.savefig('experiment_1_hop_bias.png', dpi=150)
    print(f"\n✅ 图表已保存: experiment_1_hop_bias.png")
    
    # 分析结论
    b_plus_1 = hop_bias[n_hops]  # B[1] - 相邻层比较
    b_0 = hop_bias[n_hops-1]      # B[0] - 同层比较
    print(f"\n关键发现:")
    print(f"  B[0] (同层) = {b_0:.4f}")
    print(f"  B[1] (相邻层) = {b_plus_1:.4f}")
    if b_plus_1 > b_0:
        print(f"  ✅ B[1] > B[0]：模型学会了相邻层比较更重要！")
    else:
        print(f"  ⚠️ B[1] <= B[0]：需要进一步分析")
    
    return hop_bias


# ============================================================
# 验证实验 2：Cross-Attention 区分正常/异常
# ============================================================

def experiment_2_cross_attention():
    """
    实验 2：Cross-Attention 能否区分正常/异常节点
    
    构造数据：
    - 正常节点：hop_0 ≈ hop_1 ≈ hop_2
    - 异常节点：hop_0 与 hop_1/hop_2 差异大
    
    预期：异常节点的注意力权重更分散
    """
    print("\n" + "="*60)
    print("实验 2：Cross-Attention 区分正常/异常")
    print("="*60)
    
    # 参数
    n_samples = 500
    d_model = 64
    n_other_hops = 6  # hop_1 到 hop_6
    
    # 生成数据
    hop_0 = torch.randn(n_samples, d_model)
    other_hops = torch.randn(n_samples, n_other_hops, d_model)
    
    # 定义正常/异常
    is_anomaly = torch.zeros(n_samples, dtype=torch.long)
    for i in range(n_samples):
        if i < n_samples // 2:
            # 正常：其他 hop 接近 hop_0
            other_hops[i] = hop_0[i].unsqueeze(0) + torch.randn(n_other_hops, d_model) * 0.1
            is_anomaly[i] = 0
        else:
            # 异常：其他 hop 远离 hop_0
            other_hops[i] = hop_0[i].unsqueeze(0) + torch.randn(n_other_hops, d_model) * 2.0
            is_anomaly[i] = 1
    
    # 模型
    model = HopCrossAttention(d_model, n_head=4)
    classifier = nn.Linear(d_model, 2)
    
    # 优化器
    optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 训练
    n_epochs = 200
    losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        out, attn = model(hop_0, other_hops)
        logits = classifier(out)
        
        loss = criterion(logits, is_anomaly)
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
    
    # 分析注意力模式
    print("\n--- 注意力模式分析 ---")
    with torch.no_grad():
        _, attn = model(hop_0, other_hops)
        attn = attn.squeeze(1)  # [batch, n_head, n_hops]
        
        # 正常 vs 异常的注意力差异
        normal_attn = attn[:n_samples//2].mean(dim=(0, 1))  # [n_hops]
        anomaly_attn = attn[n_samples//2:].mean(dim=(0, 1))
        
        print(f"\n正常节点注意力: {normal_attn}")
        print(f"异常节点注意力: {anomaly_attn}")
        
        # 熵分析
        normal_entropy = -(normal_attn * torch.log(normal_attn + 1e-10)).sum()
        anomaly_entropy = -(anomaly_attn * torch.log(anomaly_attn + 1e-10)).sum()
        
        print(f"\n正常节点注意力熵: {normal_entropy:.4f}")
        print(f"异常节点注意力熵: {anomaly_entropy:.4f}")
        
        if anomaly_entropy > normal_entropy:
            print(f"✅ 异常节点注意力更分散（熵更高）！")
        else:
            print(f"⚠️ 需要进一步分析")
    
    # 可视化
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    
    plt.subplot(1, 3, 2)
    plt.bar(range(n_other_hops), normal_attn.numpy())
    plt.xlabel('Hop Index')
    plt.ylabel('Attention Weight')
    plt.title('Normal Node Attention')
    
    plt.subplot(1, 3, 3)
    plt.bar(range(n_other_hops), anomaly_attn.numpy())
    plt.xlabel('Hop Index')
    plt.ylabel('Attention Weight')
    plt.title('Anomaly Node Attention')
    
    plt.tight_layout()
    plt.savefig('experiment_2_cross_attention.png', dpi=150)
    print(f"\n✅ 图表已保存: experiment_2_cross_attention.png")
    
    return normal_attn, anomaly_attn


# ============================================================
# 验证实验 3：真实数据快速验证
# ============================================================

def experiment_3_real_data():
    """
    实验 3：使用真实 Hop 数据快速验证
    
    从之前保存的 hop_features 加载，验证 hop_bias 是否能学到有意义的东西
    """
    print("\n" + "="*60)
    print("实验 3：真实数据快速验证")
    print("="*60)
    
    # 这里可以加载真实的 hop_features
    # 为了演示，使用模拟数据
    
    print("\n⚠️ 实验需要真实数据，请确保有 hop_features 文件")
    print("演示模式：使用模拟数据")
    
    # 模拟真实数据的统计特性
    n_samples = 500
    d_model = 745  # Photo 数据集的特征维度
    n_hops = 7
    
    # 生成更真实的数据（考虑 Photo 的特性）
    hop_features = torch.randn(n_samples, n_hops, d_model) * 0.5
    
    # 模拟异常：hop 特征方差大
    variances = hop_features.var(dim=(1, 2))
    labels = (variances > variances.median()).long()
    
    # 模型
    model = RelativeHopBiasAttention(d_model, n_head=8, max_hop=n_hops)
    classifier = nn.Linear(d_model, 2)
    
    # 快速训练
    optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    hop_indices = torch.arange(n_hops)
    n_epochs = 100
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        out, _, _ = model(hop_features, hop_indices)
        logits = classifier(out.mean(dim=1))
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
    
    print("\n✅ 实验 3 完成")
    return model.hop_bias.data


# ============================================================
# 主函数
# ============================================================

def main():
    print("="*60)
    print("Hop-Aware Attention 轻量化验证实验")
    print("="*60)
    print("\n目标：验证方案 A 和 B 的可行性")
    print("时间：~2 小时")
    
    # 实验 1
    hop_bias = experiment_1_probe_hop_bias()
    
    # 实验 2
    normal_attn, anomaly_attn = experiment_2_cross_attention()
    
    # 实验 3
    real_hop_bias = experiment_3_real_data()
    
    # 总结
    print("\n" + "="*60)
    print("实验总结")
    print("="*60)
    print("\n✅ 所有实验完成！")
    print("\n生成的文件：")
    print("  - experiment_1_hop_bias.png")
    print("  - experiment_2_cross_attention.png")
    
    print("\n下一步：")
    print("  1. 分析实验结果")
    print("  2. 决定采用哪个方案")
    print("  3. 集成到 VoxGFormer")


if __name__ == "__main__":
    main()