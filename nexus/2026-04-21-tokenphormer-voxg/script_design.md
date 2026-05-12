# 统计分析脚本设计说明

## 目标

实现一个轻量分析脚本，用于在不启动大规模训练的前提下，验证以下两个新特征是否携带额外区分力：

1. **Delta Prototype 特征**
2. **Local Consistency 特征**

## 输入

- 数据集 `.mat` 文件
- 节点特征 `X`
- 邻接矩阵 `A`
- 标签 `y`（仅用于统计分析与 probe，不用于半监督训练结论）

## 输出

- Prototype / Consistency 相关统计量
- Normal vs Anomaly 分布对比
- KS 检验结果
- 轻量 logistic probe 结果

## 拟计算特征

### A. Delta Prototype 特征
- delta_corr_to_proto
- delta_cos_to_proto
- delta_l2_to_proto
- proto_delta_norm

### B. Local Consistency 特征
- neigh_delta_var
- neigh_delta_mean_pairwise_cos
- node_to_proto_l2
- node_to_proto_cos

## Probe 组合
1. Delta flatten
2. Prototype-only
3. Consistency-only
4. Delta + Prototype
5. Delta + Consistency
6. Delta + Prototype + Consistency

## 数据集优先级
1. Photo
2. Amazon
3. Tolokers
4. Elliptic

## 说明
当前脚本用于理论验证与特征可分性分析，不直接作为最终半监督方法结论。