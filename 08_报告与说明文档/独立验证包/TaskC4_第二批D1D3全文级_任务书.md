# 第二批编码任务书（D1/D2/D3 全文级证据链 + ablation 语义边界复核）

> 材料：`corpus48_编码_第二批空白表.csv`（46 行，只有 study_id/title/doi）+ `schema_v4.md`。
> 背景：第一批（inclusion + D4 + D5 四字段）已验收，46 篇纳入。现在推进第二批：
> 把您第一批的 D1/D2/D3 **前置预编**升级为**全文级证据链**，并顺带复核
> `d5_ablation_present` 的语义边界。

## 任务内容（对 46 篇纳入文献）

### 1. D1/D2/D3 全文级编码
按 `schema_v4.md`（铁律：D1 只问"先验参数从哪来"，D3 只问"算它要什么输入"）。
- 每篇给出判定 + 一句**证据链**（论文哪一节 / 哪个公式 / 哪个架构部件支撑你的判定）。
- 若与您第一批的预编不同，请在 `v1_to_v2_change` 列写"改了 + 原因"。

### 2. ablation 语义边界复核（配合第一批的 d5_ablation_present）
第一批把 46 篇的 `d5_ablation_present` 都判为 `yes`（约束开/关消融）。请对每篇确认：
- `ablation_type`：`constraint-on-off`（真的开关物理约束）/ `module-ablation`（消的是
  网络模块，非约束本身）/ `both` / `unclear`
- 若发现某篇其实是 `module-ablation`（不含约束开关），请在 `ablation_note` 说明，
  我们会把该篇的 d5_ablation_present 从 yes 改为 no（语义口径收紧）。

## 工作量

46 篇全文级，预计 1.5–2 天。可以按 study_id 顺序分批交（例如每 15 篇一批）。

## 交付

- 填好的 `corpus48_编码_第二批空白表.csv`（改名 `corpus48_编码_第二批_杜俊涛.csv`）
- 一段说明：D1-D3 预编→全文级有多少处变动、ablation 复核里发现多少篇实为 module-ablation

多谢！
