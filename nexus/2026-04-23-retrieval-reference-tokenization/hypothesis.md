# Hypotheses

## H1: 全图 reference retrieval 提供局部邻域之外的新增量
如果为每个节点检索一组远程 normal / anomaly-like references，那么这些 reference token 应该在统计上提供超出局部 hop tokenization 的额外信息。

## H2: 弱指标即可支持有价值的 retrieval
用于构造 reference sequence 的 normal / anomaly score 不需要本身完成异常检测任务，只要它作为 weak retrieval prior 具备显著信息量，就可能支持后续 Transformer 交互学习。

## H3: reference token 的价值来自 interaction，而不是 score 本身
如果方法成立，其核心收益应主要来自目标节点与 reference tokens 的交互，而不是仅靠 score 排名本身完成分类。

## H4: 该方向的新增量来自“全图 reference tokenization”，而不是简单增加 token 数量
若只是多加若干无信息 token，则不会带来稳定收益；真正的收益应来自 reference selection 的语义质量。
