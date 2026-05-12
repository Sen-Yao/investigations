# README

**探究名称**: 2026-04-21-tokenphormer-voxg
**子主题**: 基于 NDC / ANR 现象设计 anomaly-aware multi-token for VoxG
**状态**: 已获用户批准，进入进一步探究

## 探究目标

沿着 Tokenphormer 对 graph tokenization 的启发，结合我们在 VoxG 中观察到的 NDC / ANR 现象，探索是否可以形成一套面向异常检测的 multi-token 设计框架。

## 这轮进一步探究要回答的问题

1. NDC / ANR 现象在表示学习层面分别对应什么语义？
2. 除了 hop token / Delta token，是否需要 neighborhood prototype token 或 community-aware token？
3. 能否形成一个 anomaly-aware multi-token 设计草案？
4. 后续最小验证路径是什么？

## 研究边界

- 当前以理论收敛与方法草图设计为主
- 暂不直接启动大规模实验
- 如需轻量验证，再单独设计低成本分析

## 预期产出

- 新增方法设计文档
- 明确 token family 候选结构
- 形成后续轻量验证方案
