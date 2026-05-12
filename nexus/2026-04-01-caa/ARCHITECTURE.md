# CAA 架构详细说明

**设计日期**: 2026-04-01  
**代码实现**: `experiments/scripts/convergence_aware_attention.py`

---

## 数学公式

### 核心注意力机制

标准注意力:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

CAA增强注意力:
```
Attention(Q, K, V) = softmax((QK^T / sqrt(d_k) + bias_depth(depth)) / temp_scale) * V
```

### 收敛偏置公式

```
bias_depth(depth) = learnable_weight[depth] * convergence_score
```

其中:
- `learnable_weight[depth]`: 每个Token深度位置的权重参数
- `convergence_score`: 基于Delta特征计算的收敛指示器

### 收敛分数计算

```
convergence_score = ConvergenceProj(x)
                   = sigmoid(Linear(GELU(Linear(x))))
```

这是一个可学习的收敛分数估计器：
1. 输入: Token特征 `x [B, T, D]`
2. 降维: `D → D/2`
3. 激活: GELU
4. 输出: `D/2 → 1`
5. 范围: Sigmoid约束到[0, 1]

### 温度缩放

```
temp_scale = temperature * (1.0 - 0.3 * position_scale)
```

其中:
- `temperature`: 可学习的基础温度参数
- `position_scale = depth / (K-1)`: 位置比例
- 深层Token温度更低 → 更锐利的注意力分布

---

## 代码实现详解

### ConvergenceScore模块

```python
class ConvergenceScore(nn.Module):
    def __init__(self, hidden_dim, num_tokens, eps=1e-8):
        # 可学习深度权重
        self.convergence_weights = nn.Parameter(
            torch.ones(num_tokens) / num_tokens
        )
        # 收敛分数投影
        self.convergence_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
```

**设计理由**:
- 初始化均匀权重，训练中自适应
- 两层MLP足够捕获收敛特征
- Sigmoid确保分数范围稳定

### ConvergenceAwareAttention模块

```python
class ConvergenceAwareAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_tokens):
        # 标准Q/K/V投影
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # 收敛组件
        self.convergence_score = ConvergenceScore(hidden_size, num_tokens)
        self.depth_bias = nn.Parameter(torch.zeros(num_tokens))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # 深层Token强调（指数加权）
        self.register_buffer(depth_emphasis, 
            torch.logspace(0, 1, num_tokens))  # [1.0, ..., 10.0]
```

**设计理由**:
- `depth_bias`: 额外的可学习偏置
- `temperature`: 全局温度控制
- `depth_emphasis`: 固定的指数加权，深层Token权重更大

---

## 关键组件说明

### 1. 收敛分数估计器

| 层 | 输入维度 | 输出维度 | 作用 |
|----|----------|----------|------|
| Linear | D | D/2 | 降维 |
| GELU | D/2 | D/2 | 非线性激活 |
| Linear | D/2 | 1 | 输出分数 |
| Sigmoid | 1 | 1 | 范围约束 |

**参数量**: `D*(D/2) + (D/2)*1 = D^2/2 + D/2`

### 2. 深度权重

| 参数 | 形状 | 初始值 | 作用 |
|------|------|--------|------|
| convergence_weights | [K] | 1/K | 可学习Token权重 |
| depth_bias | [K] | 0 | 额外偏置 |
| depth_emphasis | [K] | logspace(0,1,K) | 固定指数加权 |

### 3. 温度缩放

| 参数 | 形状 | 初始值 | 作用 |
|------|------|--------|------|
| temperature | scalar | 1.0 | 全局温度 |
| position_scale | [K] | depth/(K-1) | 位置比例 |

---

## 参数设计理由

### 隐藏维度 (hidden_dim)

推荐值: 64-256

| 值 | 参数量 | 适用场景 |
|----|--------|----------|
| 64 | ~50K | 小数据集 |
| 128 | ~200K | 中数据集 (Photo) |
| 256 | ~800K | 大数据集 |

### Token数量 (K)

推荐值: 6-7

基于Phase 1-3发现:
- K=6时，Token 4-6 MI最高
- 更大K增加深层Token数量，但信息增益递减

### 注意力头数 (num_heads)

推荐值: 4

理由:
- 4头足够捕获多种收敛模式
- 更多头增加计算开销但增益有限

### 温度初始值

推荐值: 1.0

理由:
- 1.0是标准注意力温度
- 训练中自适应调节
- 深层Token通过position_scale自动降温

---

## 计算复杂度

| 操作 | 复杂度 |
|------|--------|
| Q/K/V投影 | O(T * D^2) |
| 注意力计算 | O(T^2 * D) |
| 收敛分数 | O(T * D^2/2) |
| 偏置添加 | O(T^2) |

**总复杂度**: O(T^2 * D + T * D^2) ≈ 标准Transformer

---

_CAA通过轻量级修改增强深层Token信息传递，计算开销可控。_
