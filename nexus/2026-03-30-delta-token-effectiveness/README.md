# Delta Token 有效性探究

## 状态
🔄 进行中

## 背景

Delta 向量（相邻 hop 特征的差值）在有监督学习 + 线性回归场景下展现了卓越的性能。然而，将其直接作为 VoxGFormer 的 token 输入时，效果并不理想。

## 核心问题

**为什么 Delta 向量在线性回归上有效，但在 VoxGFormer 上效果不佳？**

## 当前发现

### 有监督 + 线性回归（已知）
- Delta 向量表现优异
- 可能原因：Delta 捕捉了节点特征的变化趋势，对异常检测有判别力

### VoxGFormer + Delta Token Mode（本次实验）

| 数据集 | original AUC | delta AUC | 差距 |
|--------|-------------|-----------|------|
| Photo | 0.8546 | 0.8653 | +1.1% ✅ |
| Amazon | 0.8977 | 0.7842 | -12.6% ❌ |
| Elliptic | 0.7881 | 0.5524 | -29.9% ❌ |

**观察**：Delta 只在 Photo 上略有提升，在 Amazon 和 Elliptic 上效果显著下降。

## 探究方向

1. **数据集特性分析**
   - Photo vs Amazon/Elliptic 的图结构差异
   - 特征维度、稀疏性、异常比例的影响

2. **Delta 向量性质分析**
   - Delta 是否丢失了重要信息？
   - Delta 的分布特性（均值、方差、稀疏性）

3. **VoxGFormer 架构适配**
   - Attention 机制是否适合处理 Delta？
   - 是否需要修改模型结构来更好地利用 Delta？

4. **混合策略**
   - Original + Delta 的 concat 模式
   - 加权组合策略

## 文件结构



## 时间线

- 2026-03-30: 启动探究，完成 token_mode 对比实验
