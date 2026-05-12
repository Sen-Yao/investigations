# Delta Vector 半监督异常检测策略设计

**日期**: 2026-03-23
**主题**: 基于现有架构的 Delta Vector 半监督学习方案

---

## 一、背景与动机

### 1.1 有监督实验回顾

今天早些时候，我们进行了 Delta Vector 的有监督异常检测实验，核心发现：

| 数据集 | Delta 向量 AUC | Delta 范数 AUC | 提升 |
|--------|----------------|----------------|------|
| tolokers | 0.7070 | 0.5342 | +0.17 |
| elliptic | 0.5036 | - | 有效 |

**关键观察**：
1. Delta 向量（保留维度信息）比 Delta 范数（L2 归一化）效果好很多
2. 不同数据集需要不同的检测方向（正向/反向）
3. 有监督设定下可以通过标签自适应选择方向

### 1.2 半监督设定的挑战

**有监督 vs 半监督的核心差异**：

| 设定 | 训练时可访问 | 方向选择 |
|------|-------------|----------|
| 有监督 | 真实异常标签 | 可自适应选择正向/反向 |
| 半监督 | 仅正常节点 | 无法知道异常 delta 更大还是更小 |

**核心问题**：半监督设定下，如何让模型**自动发现** delta 的异常模式？

### 1.3 与现有架构的关系

**PromptGAD 的核心机制**：
- 可学习 Prompt Token 查询传播特征序列
- 通过注意力机制提取频率域特征
- 高温幻觉生成伪异常

**GGADFormer 的核心机制**：
- Transformer 编码传播特征序列
- 重构误差作为伪异常生成方向
- Ring Loss 约束伪异常分布

**我们的目标**：设计能复用这些架构的 Delta 利用策略

---

## 二、方案一：Delta 建模 + 自监督预测

### 2.1 核心思想

**不是用 delta 直接做检测，而是学习 delta 的正常模式，检测偏离正常模式的节点。**

理论基础：
- 正常节点的 delta 应该遵循某种可预测的模式
- 异常节点的 delta 偏离正常模式，难以预测
- 自监督预测任务：预测下一个 hop 的 delta

### 2.2 架构设计

```python
class DeltaModeler(nn.Module):
    """
    建模 delta 序列的正常模式，通过预测误差检测异常
    
    核心假设：
    - 正常节点的 delta 序列有规律，可以预测
    - 异常节点的 delta 序列无规律，难以预测
    """
    
    def __init__(self, input_dim, hidden_dim, num_hops, num_heads=4):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_hops = num_hops
        
        # Delta 序列编码器（Transformer）
        # 输入: delta 序列 [N, K, D]
        # 输出: delta 隐藏表示 [N, K, hidden_dim]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.delta_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 预测头：预测下一个 delta
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 分布建模：学习正常 delta 的分布
        self.mu_net = nn.Linear(input_dim, input_dim)
        self.logvar_net = nn.Linear(input_dim, input_dim)
        
        # 与主干网络的融合
        self.fusion = nn.Linear(input_dim * 2, input_dim)
    
    def compute_delta_sequence(self, node_tokens):
        """
        从传播 token 序列计算 delta 序列
        
        Args:
            node_tokens: [N, K+1, D] 传播后的特征序列
        
        Returns:
            delta_seq: [N, K, D] delta 序列
        """
        # δₖ = X⁽ᵏ⁾ - X⁽ᵏ⁻¹⁾
        delta_seq = node_tokens[:, 1:] - node_tokens[:, :-1]
        return delta_seq
    
    def forward(self, node_tokens, normal_idx=None, train_mode=True):
        """
        前向传播
        
        Args:
            node_tokens: [N, K+1, D] 传播特征序列
            normal_idx: 正常节点索引（训练时使用）
            train_mode: 是否训练模式
        
        Returns:
            delta_emb: delta 编码表示
            pred_loss: 预测损失
            dist_loss: 分布损失
            anomaly_score: 异常分数（推理时）
        """
        # 计算 delta 序列
        delta_seq = self.compute_delta_sequence(node_tokens)  # [N, K, D]
        
        # 编码 delta 序列
        delta_emb = self.delta_encoder(delta_seq)  # [N, K, D]
        
        pred_loss = torch.tensor(0.0, device=node_tokens.device)
        dist_loss = torch.tensor(0.0, device=node_tokens.device)
        
        if train_mode and normal_idx is not None:
            # === 训练模式：计算两个自监督损失 ===
            
            # 1. 预测损失：预测下一个 delta
            pred_loss = self.compute_prediction_loss(delta_seq, delta_emb, normal_idx)
            
            # 2. 分布损失：正常节点的 delta 应该聚集
            dist_loss = self.compute_distribution_loss(delta_emb, normal_idx)
            
        else:
            # === 推理模式：计算异常分数 ===
            # 异常分数 = 预测误差 + 分布偏离
            anomaly_score = self.compute_anomaly_score(delta_seq, delta_emb)
            return delta_emb, anomaly_score
        
        return delta_emb, pred_loss, dist_loss
    
    def compute_prediction_loss(self, delta_seq, delta_emb, normal_idx):
        """
        自监督预测损失
        
        任务：给定前 t 个 delta，预测第 t+1 个 delta
        正常节点：delta 序列有规律，预测误差小
        异常节点：delta 序列无规律，预测误差大
        """
        K = delta_seq.size(1)
        total_loss = torch.tensor(0.0, device=delta_seq.device)
        
        for t in range(K - 1):
            # 用第 t 个 delta 的编码预测第 t+1 个 delta
            pred_next = self.predictor(delta_emb[:, t, :])  # [N, D]
            
            # 只计算正常节点的损失
            target = delta_seq[normal_idx, t + 1, :]
            pred = pred_next[normal_idx, :]
            
            loss = F.mse_loss(pred, target)
            total_loss = total_loss + loss
        
        return total_loss / (K - 1)
    
    def compute_distribution_loss(self, delta_emb, normal_idx):
        """
        分布建模损失
        
        目标：让正常节点的 delta 在隐空间中聚集
        使用变分自编码器的正则化思想
        """
        # 取正常节点在所有 hop 的平均 delta 表示
        normal_delta_repr = delta_emb[normal_idx, :, :].mean(dim=1)  # [num_normal, D]
        
        # 建模高斯分布
        mu = self.mu_net(normal_delta_repr)  # [num_normal, D]
        logvar = self.logvar_net(normal_delta_repr)  # [num_normal, D]
        
        # KL 散度损失：鼓励分布接近标准高斯
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 重构损失：鼓励同一节点在不同 hop 的 delta 表示一致
        recon_loss = torch.tensor(0.0, device=delta_emb.device)
        for h in range(delta_emb.size(1) - 1):
            recon_loss = recon_loss + F.mse_loss(
                delta_emb[normal_idx, h, :],
                delta_emb[normal_idx, h + 1, :]
            )
        recon_loss = recon_loss / (delta_emb.size(1) - 1)
        
        return kl_loss + 0.1 * recon_loss
    
    def compute_anomaly_score(self, delta_seq, delta_emb):
        """
        推理时计算异常分数
        
        异常分数 = 预测误差（难以预测） + 分布偏离（偏离正常模式）
        """
        K = delta_seq.size(1)
        N = delta_seq.size(0)
        
        # 1. 预测误差分数
        pred_errors = torch.zeros(N, device=delta_seq.device)
        for t in range(K - 1):
            pred_next = self.predictor(delta_emb[:, t, :])
            error = torch.norm(pred_next - delta_seq[:, t + 1, :], p=2, dim=-1)
            pred_errors = pred_errors + error
        pred_score = pred_errors / (K - 1)
        
        # 2. 分布偏离分数（使用学习的分布参数）
        delta_repr = delta_emb.mean(dim=1)  # [N, D]
        mu = self.mu_net(delta_repr)
        logvar = self.logvar_net(delta_repr)
        
        # 马氏距离作为偏离分数
        dist_score = torch.mean((delta_repr - mu).pow(2) / (logvar.exp() + 1e-8), dim=-1)
        
        # 综合异常分数
        anomaly_score = pred_score + 0.5 * dist_score
        
        return anomaly_score
```

### 2.3 与现有架构的融合

**与 PromptGAD 融合**：

```python
class PromptGAD_WithDelta(PromptGAD):
    def __init__(self, input_dim, hidden_dim, activation, args):
        super().__init__(input_dim, hidden_dim, activation, args)
        
        # 新增 Delta Modeler
        self.delta_modeler = DeltaModeler(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_hops=args.pp_k
        )
        
        # 融合层
        self.delta_fusion = nn.Linear(input_dim, input_dim)
    
    def forward(self, input_tokens, adj, _, normal_for_train_idx, train_flag, args, sparse=False):
        # === 原有 PromptGAD 流程 ===
        prompt_features, attn_weights = self.extract_prompt_features(input_tokens)
        ortho_loss = self.compute_orthogonal_loss(attn_weights)
        embeddings = self.encode_with_cls_token(prompt_features)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        # === 新增 Delta Modeler 流程 ===
        delta_emb, pred_loss, dist_loss = self.delta_modeler(
            input_tokens, normal_for_train_idx, train_flag
        )
        
        # 融合两种表示
        delta_repr = delta_emb.mean(dim=1)  # [N, D]
        combined_emb = embeddings + self.delta_fusion(delta_repr).unsqueeze(0)
        
        # === 后续流程 ===
        # ...（伪异常生成、分类等）
        
        return combined_emb, ..., pred_loss, dist_loss
```

### 2.4 训练目标

总损失函数：
$$
\mathcal{L} = \mathcal{L}_{\text{BCE}} + \lambda_1 \mathcal{L}_{\text{pred}} + \lambda_2 \mathcal{L}_{\text{dist}} + \lambda_3 \mathcal{L}_{\text{ortho}}
$$

其中：
- $\mathcal{L}_{\text{BCE}}$：分类损失（正常 vs 伪异常）
- $\mathcal{L}_{\text{pred}}$：Delta 预测损失
- $\mathcal{L}_{\text{dist}}$：Delta 分布损失
- $\mathcal{L}_{\text{ortho}}$：Prompt 正交损失

### 2.5 优势与局限

**优势**：
1. 自监督学习，无需异常标签
2. 模型自动学习 delta 的正常模式
3. 异常分数可解释（预测误差 + 分布偏离）
4. 与现有架构无缝融合

**局限**：
1. 假设正常 delta 可预测，可能不适用于所有数据集
2. 需要调节多个损失权重

---

## 三、方案二：Delta 对比学习

### 3.1 核心思想

**通过对比学习，让模型自动发现 delta 的多模式结构。**

理论基础：
- 正常节点可能属于多个正常模式（对应不同的 delta 分布）
- 同一模式内的节点 delta 相似
- 异常节点的 delta 不属于任何正常模式

### 3.2 架构设计

```python
class DeltaContrastive(nn.Module):
    """
    Delta 对比学习模块
    
    核心思想：
    - 通过对比学习发现 delta 的多模式结构
    - 正常节点聚类到多个模式中心
    - 异常节点不属于任何模式
    """
    
    def __init__(self, input_dim, hidden_dim, num_hops, num_prototypes=8, temperature=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        
        # Delta 编码器
        self.delta_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 可学习的模式原型（类似 Prompt）
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, hidden_dim))
        
        # 投影头（用于对比学习）
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def compute_delta_sequence(self, node_tokens):
        """计算 delta 序列"""
        delta_seq = node_tokens[:, 1:] - node_tokens[:, :-1]
        return delta_seq
    
    def encode_delta(self, delta_seq):
        """
        编码 delta 序列为单个向量
        
        Args:
            delta_seq: [N, K, D]
        
        Returns:
            delta_emb: [N, hidden_dim]
        """
        # 方案1: 取平均
        delta_mean = delta_seq.mean(dim=1)  # [N, D]
        
        # 方案2: 加权平均（学习权重）
        # weights = F.softmax(self.hop_weights, dim=0)
        # delta_mean = (delta_seq * weights.view(1, -1, 1)).sum(dim=1)
        
        # 编码
        delta_emb = self.delta_encoder(delta_mean)  # [N, hidden_dim]
        return delta_emb
    
    def forward(self, node_tokens, normal_idx=None, train_mode=True):
        """
        前向传播
        """
        # 计算 delta 序列
        delta_seq = self.compute_delta_sequence(node_tokens)
        
        # 编码 delta
        delta_emb = self.encode_delta(delta_seq)  # [N, hidden_dim]
        
        # 投影
        delta_proj = self.projector(delta_emb)  # [N, hidden_dim]
        delta_proj = F.normalize(delta_proj, dim=-1)
        
        contrastive_loss = torch.tensor(0.0, device=node_tokens.device)
        
        if train_mode and normal_idx is not None:
            # 对比学习损失
            contrastive_loss = self.compute_contrastive_loss(delta_proj, normal_idx)
        
        return delta_emb, contrastive_loss
    
    def compute_contrastive_loss(self, delta_proj, normal_idx):
        """
        Delta 对比学习损失
        
        目标：
        - 同一原型下的正常节点拉近
        - 不同原型之间推远
        """
        normal_proj = delta_proj[normal_idx]  # [num_normal, hidden_dim]
        num_normal = normal_proj.size(0)
        
        # 归一化原型
        prototypes_norm = F.normalize(self.prototypes, dim=-1)  # [P, hidden_dim]
        
        # 计算每个节点到每个原型的相似度
        sim_to_prototypes = torch.mm(normal_proj, prototypes_norm.t())  # [num_normal, P]
        sim_to_prototypes = sim_to_prototypes / self.temperature
        
        # 软分配：每个节点属于各原型的概率
        assignment = F.softmax(sim_to_prototypes, dim=-1)  # [num_normal, P]
        
        # 找每个节点的主原型
        dominant_prototype = torch.argmax(assignment, dim=-1)  # [num_normal]
        
        # === 模式内聚合损失 ===
        # 同一原型下的节点应该相似
        intra_loss = torch.tensor(0.0, device=delta_proj.device)
        for p in range(self.num_prototypes):
            mask = (dominant_prototype == p)
            if mask.sum() < 2:
                continue
            
            same_proto_nodes = normal_proj[mask]  # [num_in_proto, hidden_dim]
            sim_matrix = torch.mm(same_proto_nodes, same_proto_nodes.t()) / self.temperature
            
            # 排除对角线
            sim_matrix = sim_matrix.masked_fill(
                torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device),
                float('-inf')
            )
            
            # InfoNCE: 同模式内节点应该相似
            intra_loss = intra_loss + torch.logsumexp(sim_matrix, dim=1).mean()
        
        intra_loss = intra_loss / max(1, (dominant_prototype.unique().size(0)))
        
        # === 模式间分散损失 ===
        # 不同原型之间应该不同
        inter_loss = torch.tensor(0.0, device=delta_proj.device)
        proto_sim = torch.mm(prototypes_norm, prototypes_norm.t()) / self.temperature
        
        # 取上三角
        upper_mask = torch.triu(torch.ones_like(proto_sim, dtype=torch.bool), diagonal=1)
        inter_loss = torch.exp(proto_sim[upper_mask]).mean()
        
        # 总对比学习损失
        contrastive_loss = intra_loss + 0.1 * inter_loss
        
        return contrastive_loss
    
    def compute_anomaly_score(self, node_tokens):
        """
        推理时计算异常分数
        
        异常分数 = 到最近原型的距离
        """
        delta_seq = self.compute_delta_sequence(node_tokens)
        delta_emb = self.encode_delta(delta_seq)
        delta_proj = self.projector(delta_emb)
        delta_proj = F.normalize(delta_proj, dim=-1)
        
        # 计算到每个原型的相似度
        prototypes_norm = F.normalize(self.prototypes, dim=-1)
        sim_to_prototypes = torch.mm(delta_proj, prototypes_norm.t())  # [N, P]
        
        # 异常分数 = 1 - 最大相似度（距离最远的原型）
        max_sim, _ = sim_to_prototypes.max(dim=-1)
        anomaly_score = 1 - max_sim
        
        return anomaly_score
```

### 3.3 与现有架构的融合

**与 PromptGAD 融合**：

```python
class PromptGAD_WithDeltaContrastive(PromptGAD):
    def __init__(self, input_dim, hidden_dim, activation, args):
        super().__init__(input_dim, hidden_dim, activation, args)
        
        # 新增 Delta 对比学习模块
        self.delta_contrastive = DeltaContrastive(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_hops=args.pp_k,
            num_prototypes=getattr(args, 'num_delta_prototypes', 8),
            temperature=getattr(args, 'delta_contrast_temp', 0.1)
        )
        
        # 融合权重
        self.delta_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, input_tokens, adj, _, normal_for_train_idx, train_flag, args, sparse=False):
        # === 原有 PromptGAD 流程 ===
        prompt_features, attn_weights = self.extract_prompt_features(input_tokens)
        ortho_loss = self.compute_orthogonal_loss(attn_weights)
        embeddings = self.encode_with_cls_token(prompt_features)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        # === 新增 Delta 对比学习流程 ===
        delta_emb, contrastive_loss = self.delta_contrastive(
            input_tokens, normal_for_train_idx, train_flag
        )
        
        # === 训练时：生成伪异常 ===
        if train_flag:
            # 方法1：利用原型生成伪异常
            pseudo_anomaly_emb = self.generate_pseudo_anomaly_via_prototypes(
                embeddings, normal_for_train_idx
            )
            # ...
        
        return embeddings, ..., contrastive_loss
    
    def generate_pseudo_anomaly_via_prototypes(self, embeddings, normal_idx):
        """
        利用原型生成伪异常
        
        方法：将正常节点推向"错误"的原型
        """
        normal_emb = embeddings[0, normal_idx, :]  # [num_normal, D]
        
        # 找到每个节点的最近原型
        proto_assign = self.delta_contrastive.compute_prototype_assignment(normal_emb)
        
        # 选择一个不同的原型作为目标
        wrong_proto_idx = (proto_assign + torch.randint(
            1, self.delta_contrastive.num_prototypes, proto_assign.shape, device=proto_assign.device
        )) % self.delta_contrastive.num_prototypes
        
        wrong_proto = self.delta_contrastive.prototypes[wrong_proto_idx]
        
        # 生成伪异常：向错误原型方向移动
        pseudo_anomaly = normal_emb + 0.5 * (wrong_proto - normal_emb)
        
        return pseudo_anomaly
```

### 3.4 训练目标

总损失函数：
$$
\mathcal{L} = \mathcal{L}_{\text{BCE}} + \lambda_1 \mathcal{L}_{\text{contrast}} + \lambda_2 \mathcal{L}_{\text{ortho}}
$$

其中：
- $\mathcal{L}_{\text{BCE}}$：分类损失
- $\mathcal{L}_{\text{contrast}}$：Delta 对比学习损失
- $\mathcal{L}_{\text{ortho}}$：Prompt 正交损失

### 3.5 与 PromptGAD 的关系

**相似之处**：
- 都使用可学习的原型/Prompt
- 都通过注意力/相似度分配节点到模式
- 都有多模式建模的思想

**关键区别**：
- PromptGAD 的 Prompt 查询的是**传播特征** $X^{(k)}$
- Delta 对比学习的原型查询的是**delta 向量** $\delta_k$
- Delta 原型直接编码"异常偏离模式"

### 3.6 优势与局限

**优势**：
1. 无需预测任务，对比学习更稳定
2. 多原型设计自然支持多模式正常数据
3. 异常分数有直观解释（到最近原型的距离）
4. 可与 PromptGAD 的 Prompt 机制协同

**局限**：
1. 原型数量需要调参
2. 对于异常比例很高的数据集可能不适用

---

## 四、两个方案对比

| 维度 | 方案一：Delta 建模 | 方案二：Delta 对比学习 |
|------|-------------------|----------------------|
| **核心机制** | 自监督预测 | 对比学习 |
| **异常假设** | 异常 delta 难以预测 | 异常 delta 不属于正常模式 |
| **可解释性** | 预测误差 + 分布偏离 | 到原型的距离 |
| **训练稳定性** | 依赖预测任务设计 | 通常更稳定 |
| **与 PromptGAD 融合** | 作为额外模块 | 与 Prompt 机制类似 |
| **调参复杂度** | 多个损失权重 | 原型数量 + 温度 |

---

## 五、实现优先级建议

### 5.1 推荐先尝试方案二（Delta 对比学习）

理由：
1. 对比学习通常更稳定
2. 与 PromptGAD 的 Prompt 机制类似，易于融合
3. 异常分数解释性更好

### 5.2 实现步骤

1. **Step 1**: 实现 `DeltaContrastive` 模块
2. **Step 2**: 在 `PromptGAD` 中集成
3. **Step 3**: 在 tolokers/elliptic 数据集上验证
4. **Step 4**: 如果效果好，尝试方案一的组合

### 5.3 验证指标

- 主实验：AUROC, AUPRC
- 辅助指标：
  - 原型分配分布（是否有多模式）
  - 正常/异常节点到原型的距离分布
  - 对比学习损失曲线

---

## 六、风险与后续工作

### 6.1 潜在风险

1. **数据集特异性**：不同数据集的 delta 模式可能差异很大
2. **超参敏感**：原型数量、温度等可能需要针对数据集调参
3. **计算开销**：额外的对比学习模块会增加训练时间

### 6.2 后续工作

1. 在更多数据集上验证 delta 的统计特性
2. 探索两种方案的组合
3. 研究自适应确定原型数量的方法

---

*文档完成于 2026-03-23 21:10*