# 独立编码员指引（给第二标注者，P2 第 3 项）

> 本文件是**可直接交给独立标注者**的编码包。目标是获得与 coder_A 真正独立的
> 人工编码，替代当前"规则 coder_B"（κ=1.0 是上限，非真实双人工证据）。

## 任务

对 26 篇 physics-constrained wind power forecasting 论文，按 5 个维度编码。

## 输入

- `literature_coding_table_v3_arbitrated.csv` 的 `paper_id`、`constraint_math_form`、
  `input_variable`、`notes`（供定位原文，但**编码时以论文原文为准**，不看已有 D1-D5 列）。
- 每篇论文的标题/摘要/全文（标注者自行检索 DOI/标题获取）。

## 五个编码字段与取值（协议 v3，见 `ConstraintLifecycle_Taxonomy_schema_v3.md`）

### D1. Provenance（约束来源）
物理先验用什么数据估计？
- `intrinsic`：物理常数（Betz、额定、切入切出）——不需估计
- `historical-static`：训练前用历史观测一次性估计，训练期冻结（如 KDE power curve）
- `forecast-derived`：用 NWP/预报输入 + 物理模型的输出（Jensen wake、理想功率曲线、Cp estimator）
- `learned-online`：训练中随参数更新的先验
- `unreported`：未说明

### D2. Application stage（应用阶段）
- `training-loss`：进训练损失（soft/hard regularizer）
- `post-hoc-correction`：推理后修正
- `inference-architecture`：作为推理期模型结构
- `data-generation`：用于生成训练数据
- `unreported`

### D3. Inference requirement（推理期依赖）
- `none`：推理完全自足
- `forecast-input`：需要预报输入（NWP 等）
- `contemporaneous-state`：需要实时 SCADA/机型状态
- `latent-state`：需要不可直接观测的隐状态（限电/维护状态）
- `unreported`

### D4. Implementation fidelity（实现忠实度）
- `code-aligned`：官方代码与论文公式一致
- `code-mismatch`：不一致（有证据）
- `no-code`：无公开代码
- `parity-only`：只能从论文公式重建
- `unverified`：未核验

### D5. Harm reporting（伤害报告）
- `none-reported`：未报告负面结果
- `ablation-absent`：无约束有无的消融
- `harm-quantified`：报告了约束在某些设定的负面结果
- `substitution-tested`：做了替换测试

## 核心区分（防击穿条款）

> 训练期合法历史先验（D1=historical-static）≠ 推理期不可观测约束（D3）。
> 必须分开编码，不可把 historical-static prior 写成"不可观测"。

## 交付

把编码结果填入 `literature_coding_table_coderC_blank.csv`（每篇一行，
填 `coderC_D1`..`coderC_D5` 五列），不要看或修改 `D1_provenance`..`D5_harm`
（那是 coder_A 的列）。

## 一致性

完成后由协作者运行 `interrater_agreement_final.py` 计算 coder_A vs coder_C 的
Cohen's κ，报告 5 个维度的真实双人工一致率。
