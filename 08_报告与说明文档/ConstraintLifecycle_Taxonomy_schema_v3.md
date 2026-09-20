# Constraint Lifecycle Taxonomy（schema v3，2026-09-03 冻结）

> 升级说明：steer #2 裁决原 "L1–L5 deployment-time observability" 主线被 PCT 方法部分
> 击穿（KDE prior 是训练期历史静态先验，推理期不需要未来真值）。因此 observability
> 降为 taxonomy 的**一维**，扩展为 **constraint lifecycle** 五维框架。

## 一句话定位

本文审计的不是"约束部署时是否可观测"，而是**约束在完整生命周期中的角色与风险**：
prior 从哪里来 → 在哪个阶段应用 → 推理期需要什么 → 实现是否忠于规格 → 是否造成可测伤害。

## 五个维度（每个维度一段编码协议）

### D1. Provenance（约束来源）
约束的 physical prior 用什么数据估计？
- `intrinsic`：物理常数（Betz limit、额定功率、切入切出风速）——不需估计
- `historical-static`：训练前用历史观测一次性估计，训练期冻结（如 KDE power curve）
- `forecast-derived`：用 NWP/预报输入的物理模型输出（如 Jensen wake、理想功率曲线）
- `learned-online`：训练中随参数更新的先验（如 C_p estimator 网络）
- `unreported`：论文未说明 prior 来源

### D2. Application stage（应用阶段）
约束在哪个阶段作用？
- `training-loss`：进训练损失（soft/hard regularizer）
- `post-hoc-correction`：推理后修正预测
- `inference-architecture`：作为推理期模型结构的一部分
- `data-generation`：用于生成训练数据（如 physics-guided diffusion）
- `unreported`

### D3. Inference requirement（推理期依赖）
推理时是否需要训练期以外的新信息？
- `none`：推理完全自足（只用已训练模型 + 常规输入）
- `forecast-input`：需要预报输入（NWP 等）
- `contemporaneous-turbine-state`：需要实时 SCADA/机型状态
- `latent-state`：需要不可直接观测的隐状态（限电/维护状态、available power）
- `unreported`

### D4. Implementation fidelity（实现忠实度）
官方实现是否忠于论文 specification？（对可执行审计目标）
- `code-aligned`：官方代码与论文公式一致
- `code-mismatch`：官方代码与论文公式不一致（附影响证据）
- `no-code`：无公开代码，无法核验
- `parity-only`：只能从论文公式重建，无官方代码对照

### D5. Harm reporting（伤害报告）
论文是否报告约束的潜在/实测伤害？
- `none-reported`：未报告任何 negative side effect
- `ablation-absent`：未做约束有无的消融（收益并非来自约束对照）
- `harm-quantified`：报告了约束在某些设定下的负面结果
- `substitution-tested`：做了 oracle/causal 替换测试

## 与旧 schema 的映射

| 旧字段（v2） | 新归属 | 处理 |
|---|---|---|
| observability_level (L1–L5) | D1+D3 的粗糙合并 | 拆解：L1→D1=intrinsic+D3=none；L2→D1=historical-static+D3=contemporaneous；L3→D3=forecast-input；L4→D3=latent-state |
| oracle_leakage | 删除（被 steer 证伪：26/26 FALSE 是反证）| 替换为 D3 |
| enforcement_mode | D2 | 保留取值 |
| deployment_source_stated | D1 | 归一化 |
| harm_reported | D5 | 扩展为 4 类 |
| substitution_test | D5=substitution-tested | 保留 |
| code_available | D4 | 保留 |

## 核心 distinction（防击穿条款）

> **训练期合法历史先验（D1=historical-static）≠ 推理期不可观测约束（D3=latent-state/
> forecast-input）**。任何编码必须分开这两个维度，不允许用"不可观测"描述
> historical-static prior。这正是旧主线被 PCT 方法部分击穿的原因，v3 从字段上排除。

## 编码协议

- 双人独立编码（coder_A / coder_B），7 个判定字段 → 现 5 维 × 各 1 字段。
- 分歧仲裁：先区分"字段污染/写法差异"与"实质判定分歧"，只对实质分歧仲裁。
- 一致性：Cohen's κ + Krippendorff's α（序数化 5 维）。

## 首次编码（coder_A，2026-09-03，摘要级）

文件：`literature_coding_table_v3_D1D5.csv`（26 篇）

| 维度 | 分布 |
|---|---|
| D1 provenance | forecast-derived×10、unreported×7、historical-static×6、intrinsic×3 |
| D2 stage | training-loss×18、unreported×5、inference-arch×2、data-generation×1 |
| D3 inference | forecast-input×9、unreported×7、contemporaneous-state×5、none×3、latent-state×2 |
| D4 fidelity | unverified×21、no-code×2、parity-only×2、code-mismatch×1 |
| D5 harm | none-reported×25、harm-quantified×1 |

**正面支持 gap**：
- D4 `unverified` 21/26 → 实现忠实度系统性不可核验。
- D5 `none-reported` 25/26 → 约束伤害系统性缺少审计（与 steer #2 一致）。
- D3 forecast-input×9 + latent-state×2 → 14/26 推理期需训练集外信息（仅 D3 一维，
  不作主线）。

**弱编码（摘要级不可定 D2/D3，需全文核对）7 篇**：JARA2024、MLPINN、
CPDIGITAL2025、MCMAMBA2026、DUALSTREAM2026、TCNTRANS2026、HUANG2026。

**待办**：~~coder_B 独立编码~~ 已完成、~~仲裁~~ 已完成。

## 双人编码与仲裁（v3 五维，2026-09-03）

| 维度 | coder_A vs coder_B | κ | 仲裁后分布 |
|---|---|---|---|
| D1 provenance | 19/26 | 0.63 | forecast-derived×10、unreported×7、historical-static×6、intrinsic×3 |
| D2 stage | 18/26 | 0.35 | training-loss×18、unreported×5、inference-arch×2、data-generation×1 |
| D3 inference | 16/26 | 0.47 | forecast-input×10、unreported×7、contemporaneous-state×4、none×3、latent-state×2 |
| D4 fidelity | 22/26 | 0.45 | unverified×21、no-code×2、parity-only×2、code-mismatch×1 |
| D5 harm | 26/26 | 1.00 | none-reported×25、harm-quantified×1 |

**结论**：
- D5（harm reporting）κ=1.00 且 25/26 none-reported —— **论文最稳、最核心的 gap**。
- D4（fidelity）21/26 unverified —— 第二稳的 gap（实现忠实度系统性不可核验）。
- D3（inference）κ=0.47 最低 —— 主观性最强，需全文核对；但两 coder 方向一致
  （多数是 forecast-input/contemporaneous，非 none）。
- 仲裁文件：`literature_coding_table_v3_arbitrated.csv`；弱编码 7 篇仍需全文核对。

**诚实声明**：coder_B 用了关键词规则推导（非独立人工），κ 是"人工 coder_A vs
规则 coder_B"的上限参考，真正的严格双人工编码仍需两名独立标注者。
