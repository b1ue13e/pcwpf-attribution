# Task C2 — D1/D2/D3 按 v4 修订 schema 重编（请杜老师再判一轮）

> 材料：`ConstraintLifecycle_Taxonomy_schema_v4.md`（新定义）+ `TaskC2_D1D3_重编_空白表.csv`。
> 说明：第一轮 Task C 的 D1/D2/D3 三方 κ 只到 0.19–0.37，根因是 **schema 定义有歧义**
> （D1 混入了"计算输入"这层意思）。我们改了 schema（v4），请你不用管上一轮的判断，
> **按 v4 定义对 21 篇的 D1/D2/D3 重新判一遍**。

## 你只需要做三件事

1. 读 `ConstraintLifecycle_Taxonomy_schema_v4.md`，重点看开头的**铁律**和 D1/D2/D3 三节。
2. 对照你已有的 21 篇全文/摘要（不用重新检索，你上一轮材料还在），在空白表里填
   `D1_provenance`、`D2_stage`、`D3_inference_req` 三列（取值用 v4 的英文）。
3. 有把握不稳的，在 `coderD_notes` 写一句。

## v4 相比 v3 的关键变化（提醒你注意）

| 变化 | 你要怎么判 |
|---|---|
| D1 **删除了 `forecast-derived`** | 判断只问"参数值从哪来"：解析物理/常数 → `intrinsic`；历史拟合 → `historical-static`；训练中更新 → `learned-online`。**"算它需要预报"不再影响 D1**，交给 D3 |
| D2 加了三条判定原则 | loss 里通过梯度起效 → `training-loss`；前向图不可分割一环 → `inference-architecture`；产出额外训练数据 → `data-generation` |
| D3 明确 contemporaneous（可观测）vs latent（不可观测）| 实时桨距角/上游实测风速/限电状态 = `contemporaneous-state`；被遮蔽的"本可发出功率"= `latent-state` |

## 交付

- 填好的 `TaskC2_D1D3_重编_空白表.csv`（改名 `TaskC2_D1D3_杜俊涛.csv`）
- 一段说明：哪些篇这次判定和上次不同、为什么（可选，但很有价值）

多谢！
