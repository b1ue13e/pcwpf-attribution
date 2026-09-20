# 48 项语料扩展编码任务书（给杜老师）

> 材料：`corpus48_编码空白表.csv`（48 行）+ `ConstraintLifecycle_Taxonomy_schema_v4.md`。
> 背景：上一轮 Task D 您按宽标准筛出 53 篇，合并 5 组预印本/正刊重复后是 **48 项
> 独立研究**。我们决定把审计语料从 21 篇扩展到这 48 项。请您对 48 项做编码。

## 编码内容（48 项 × 以下字段）

### 0. inclusion_check（纳入复核）
48 项是"宽标准"候选，请逐篇确认它**有没有显式物理约束**：
- `include`：有显式物理约束/先验（功率曲线、Betz/切入切出边界、KDE-JS/MMD 分布
  匹配、PDE、Jensen 尾流、气动解析方程作为 loss 项或架构部件等）
- `exclude`：标题带 physics 但实则纯数据驱动 / 仅物理启发设计（无任何显式约束）
- 若 exclude，在 `exclusion_reason` 写一句（如 `no-explicit-physics-constraint`）

### 1. D1/D2/D3（按 v4，读 `schema_v4.md`）
先读铁律：**D1 只问"先验参数从哪来"，D3 只问"算它要什么输入"，两者独立**。
取值域以 v4 为准（D1 无 `forecast-derived`）。

### 2. D4 fidelity
- `code-aligned` / `code-mismatch`（有代码且一致/不一致）
- `not-found`（检索后未找到）
- `unverified`（未做检索）

### 3. D5 harm —— **拆成 4 个字段**（本次新增，替代单一 D5_harm）
请对每篇填四个是/否：

| 字段 | 问什么 | 取值 |
|---|---|---|
| `d5_ablation_present` | 是否做了"约束开 vs 关"的消融实验？| `yes` / `no` / `unknown`（摘要看不出）|
| `d5_negative_effect_reported` | 是否报告了约束的任何负面效应？| `yes` / `no` |
| `d5_harm_metric_reported` | 是否报告了伤害的量化指标？| `yes` / `no` |
| `d5_prior_substitution_tested` | 是否做了先验替换/敏感性测试？| `yes` / `no` |

> `D5_harm` 单列可以留空或填您认为最贴切的总括（我们主要用四个 binary 字段做统计）。

## 工作量建议（分两批，先做核心）

- **第一批（本次先交）**：`inclusion_check` + `D4` + `D5 四个字段`。
  每篇读摘要通常够（D4 需搜一下官方代码），约 1–1.5 天。
- **第二批**：`D1/D2/D3`（需全文，约 1–1.5 天）。可以等第一批确认语料范围后再做。

## 交付

- 填好的 `corpus48_编码空白表.csv`（改名 `corpus48_编码_杜俊涛.csv`）
- 一段说明（可选）：哪些篇您判 exclude、哪些篇 D5 四个字段有把握不稳。

多谢！
