# RHO 与 VoxG 结合方案

## 一、现有架构分析

### 1. VoxGFormer 核心组件

```
节点特征 → Token 序列 → Prompt 提取 → Transformer → 对比学习
           [H^0...H^K]   [P1...Pm]
```

**关键组件**：
- **Token 序列**：多跳传播结果 `[H^0, H^1, ..., H^K]`
- **Prompt Token**：可学习的查询向量，提取多视角特征
- **Transformer**：多层自注意力编码器
- **对比学习**：通过 Prompt 特征区分正常/异常

### 2. RHO 核心组件

```
特征 → Channel-wise 滤波 → 对比学习
     → Cross-channel 滤波 →
```

**关键组件**：
- **Channel-wise k**：每个特征维度有独立的滤波强度 `[k_1, k_2, ..., k_D]`
- **Cross-channel k**：所有维度共享滤波强度 `k`
- **对比学习**：对齐两个视图

---

## 二、结合方案

### 方案 1：Channel-wise Token（推荐）

**核心思想**：将每个通道的传播序列作为 Token

```
原始：节点 Token = [H^0, H^1, ..., H^K]  # 所有通道一起

改进：通道 Token = [[H^0_1, H^1_1, ..., H^K_1],  # 通道 1
                   [H^0_2, H^1_2, ..., H^K_2],  # 通道 2
                   ...
                   [H^0_D, H^1_D, ..., H^K_D]]  # 通道 D
```

**Transformer 输入**：
- `[CLS, Channel_1, Channel_2, ..., Channel_D]`
- 每个 Channel_i 是其传播序列的聚合

**优势**：
- Transformer 自动学习通道重要性 → 等价于学习 Channel-wise k
- 无需手动设计 k 参数
- 与现有 Prompt 系统完全兼容

**实现**：
```python
class ChannelWiseTokenizer(nn.Module):
    """Channel-wise Token 生成器"""
    
    def __init__(self, num_hops, num_channels, hidden_dim):
        super().__init__()
        # 每个通道的 Token 序列
        self.channel_proj = nn.Linear(num_hops + 1, hidden_dim)
        # CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
    
    def forward(self, tokens):
        """
        tokens: [N, K+1, D] - 多跳传播结果
        
        Returns: [N, D+1, hidden_dim] - Channel-wise Token 序列
        """
        N, K, D = tokens.shape
        
        # 转置：[N, D, K+1] - 每个通道的传播序列
        channel_tokens = tokens.permute(0, 2, 1)
        
        # 投影到隐藏维度
        channel_tokens = self.channel_proj(channel_tokens)  # [N, D, hidden_dim]
        
        # 添加 CLS Token
        cls = self.cls_token.expand(N, -1, -1)
        channel_tokens = torch.cat([cls, channel_tokens], dim=1)
        
        return channel_tokens
```

---

### 方案 2：双视图 Transformer

**核心思想**：两个 Transformer 分别处理两个视图

```
视图 1 (Channel-wise)：每个通道独立 Token → Transformer_1
视图 2 (Cross-channel)：所有通道一起 Token → Transformer_2
                                       ↓
                                   特征融合
```

**实现**：
```python
class DualViewTransformer(nn.Module):
    """双视图 Transformer"""
    
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super().__init__()
        
        # Channel-wise 视图
        self.cw_tokenizer = ChannelWiseTokenizer(num_hops, input_dim, hidden_dim)
        self.cw_transformer = TransformerEncoder(hidden_dim, num_heads, num_layers)
        
        # Cross-channel 视图（原有方式）
        self.cc_transformer = TransformerEncoder(input_dim, num_heads, num_layers)
        
        # 融合层
        self.fusion = nn.Linear(hidden_dim + input_dim, hidden_dim)
    
    def forward(self, tokens):
        # Channel-wise 视图
        cw_tokens = self.cw_tokenizer(tokens)
        cw_feat = self.cw_transformer(cw_tokens)[:, 0, :]  # CLS Token
        
        # Cross-channel 视图
        cc_feat = self.cc_transformer(tokens).mean(dim=1)
        
        # 融合
        fused = self.fusion(torch.cat([cw_feat, cc_feat], dim=-1))
        
        return fused
```

---

### 方案 3：Prompt 驱动的 k 学习

**核心思想**：用 Prompt 来学习 Channel-wise k

```
Prompt: "这个通道对异常检测有多重要？"
Input: 通道特征序列
Output: k 值
```

**实现**：
```python
class PromptDrivenK(nn.Module):
    """Prompt 驱动的 k 学习"""
    
    def __init__(self, num_channels, hidden_dim, num_prompts=4):
        super().__init__()
        
        # Prompt Token
        self.prompts = nn.Parameter(torch.randn(num_prompts, hidden_dim))
        
        # 通道特征投影
        self.channel_proj = nn.Linear(num_channels, hidden_dim)
        
        # k 预测头
        self.k_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, channel_features):
        """
        channel_features: [N, D] - 通道特征
        
        Returns: k: [D] - 每个通道的滤波强度
        """
        # 用 Prompt 提取通道特征
        # prompts: [P, H]
        # channel_features: [N, D] → [D, H]
        channel_proj = self.channel_proj(channel_features.T)  # [D, H]
        
        # Prompt 注意力
        attn = torch.matmul(self.prompts, channel_proj.T)  # [P, D]
        attn = F.softmax(attn, dim=0)
        
        # 加权聚合
        channel_repr = torch.matmul(attn.T, self.prompts)  # [D, H]
        
        # 预测 k
        k = self.k_head(channel_repr).squeeze(-1)  # [D]
        k = torch.sigmoid(k)  # 归一化到 [0, 1]
        
        return k
```

---

## 三、推荐实现路径

### 阶段 1：Channel-wise Token（最简单）

1. 实现 `ChannelWiseTokenizer`
2. 替换现有 Token 输入
3. 验证效果

**优势**：
- 改动最小
- 与现有 Prompt 系统完全兼容
- Transformer 自动学习通道重要性

### 阶段 2：双视图融合

1. 实现双 Transformer 架构
2. 添加对比学习损失
3. 验证融合效果

### 阶段 3：Prompt 驱动（最完整）

1. 实现 Prompt 驱动的 k 学习
2. 与 RHO 的 Channel-wise 滤波结合
3. 端到端训练

---

## 四、理论分析

### Channel-wise Token 的优势

| 方法 | k 学习方式 | 可解释性 |
|------|-----------|----------|
| RHO 原版 | 直接参数 | 低 |
| Channel-wise Token | Transformer 注意力 | 高 |
| Prompt 驱动 | Prompt 查询 | 最高 |

**Channel-wise Token 的理论保证**：

1. **注意力权重 ≈ k 值**
   - Transformer 的注意力权重决定每个通道的重要性
   - 这等价于 RHO 的 Channel-wise k

2. **可解释性**
   - 可视化注意力权重 → 知道哪些通道重要
   - 可用于特征选择

3. **端到端学习**
   - 不需要手动设计 k 参数
   - Transformer 自动学习最优权重

---

## 五、实验计划

### 实验 1：Channel-wise Token vs Baseline

| 方法 | Token 类型 | 预期效果 |
|------|-----------|----------|
| Baseline | Cross-channel Token | — |
| Ours | Channel-wise Token | 在低维图上提升 |

### 实验 2：双视图融合

| 方法 | 视图 | 预期效果 |
|------|------|----------|
| Single | Cross-channel | — |
| Dual | Channel-wise + Cross-channel | 在中维图上提升 |

### 实验 3：Prompt 驱动 k

| 方法 | k 学习 | 预期效果 |
|------|--------|----------|
| RHO | 参数学习 | — |
| Ours | Prompt 驱动 | 更好的可解释性 |

---

_文档创建时间: 2026-03-27 14:10_