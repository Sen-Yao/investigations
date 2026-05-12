# Post-Delta 当前工作梳理

## 1. 今天的新探究主题
今天已正式开启 `2026-04-22-post-delta-token-design`。核心不是继续修 `delta token`，而是转向 post-delta 时代的 anomaly-aware multi-token family 设计。

当前三条主线已明确：
- `Patch Token`
- `Relation Token`
- `Prototype Assignment / Contrast Token`

## 2. 当前方向收敛
目前优先级已经基本收紧为：
1. `Patch Token`（第一主线）
2. `Relation Token`（第二视角）
3. `Prototype Assignment / Contrast Token`（暂作参考层）

其中 `Patch Token` 是当前最值得押注的方向，因为它改变了建模粒度，最自然承接 `ANR`；`Relation Token` 主要承接 `NDC`，但目前更像补充视角。

## 3. 已完成的核心文档
目前 investigation 下已形成一套比较完整的方法骨架，核心文件包括：
- `patch_token_design.md`
- `relation_token_design.md`
- `prototype_assignment_token_design.md`
- `post_delta_multi_token_framework.md`
- `patch_relation_first_validation_plan.md`
- `cross_dataset_patch_relation_report.md`
- `patch_first_method_draft.md`

## 4. 第一轮实验进展
已完成 Patch / Relation 轻量验证，数据集包括：
- `Amazon`
- `Elliptic`
- `Photo`
- `Reddit`
- `Tolokers`

当前阶段结论：
- `Patch` 比 `Relation` 更稳；
- `Elliptic` 与 `Photo` 对这条路线支持最强；
- `Reddit` 有小幅正反馈；
- `Amazon` 基本持平；
- `Tolokers` 是边界例子。

因此，当前可以把结论收成：
> post-delta 路线已经获得跨数据集初步支持，其中 `Patch Token` 是当前最有前景的主方向，`Relation Token` 是有价值的补充关系视角。

## 5. 当前方法收敛
刚刚已进一步整理 `patch_first_method_draft.md`。
当前建议：
- 第一版优先采用 `P-C1: Local Ego-Patch Token`
- 先做 `Node Token + Patch Token` 的最小双 token 结构
- 暂不直接上完整 unified multi-token Transformer

## 6. 当前最合理的下一步
1. 把 `Patch-first` 推成第一版正式方法实现；
2. 再收紧 `Relation` 的角色（并行 token / 辅助支路 / residual explainer）；
3. `Prototype` 暂时不进入第一轮主实验。

如果需要，我下一步可以继续把 `Relation` 的角色判断也单独整理成决策文档。
