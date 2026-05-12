# README

**探究名称**: 2026-04-22-post-delta-token-design
**主题**: 放弃一阶差分后，面向 VoxG 的新 token 来源设计
**状态**: 已启动

## 背景

在 `2026-04-21-tokenphormer-voxg/` 的连续探究中，我们发现：

- 一阶 Delta token 与传播主干信息重合度较高
- Prototype / Consistency 虽有统计信号，但未表现出稳定新增量
- Tokenphormer 对 VoxG 的最大启发不在于差分，而在于 multi-token family 的方法论

因此，本探究不再围绕一阶差分修补，而是重新定义：

> 对图异常检测而言，哪些 token 才代表真正不同的信息来源？

## 核心目标

围绕三个新的 token 来源展开系统探究：

1. **Community / Patch Token**
2. **Prototype Assignment / Contrast Token**
3. **Relation Token**

## 核心问题

1. 这三类 token 分别承载什么异常检测语义？
2. 它们与原有 hop / delta 路线相比，新增量在哪里？
3. 哪些设计最有希望与 NDC / ANR 现象形成闭环？
4. 后续最小验证路径该如何设计？

## 探究边界

- 当前阶段以方法设计、概念收敛、验证路线设计为主
- 暂不直接启动大规模实验
- 需要把最近 NDC / ANR 发现纳入叙事中心

## 预期产出

- 三类 token 来源的系统分析
- 每类 token 的候选定义
- 与 NDC / ANR 的对应关系
- 后续轻量验证路线
