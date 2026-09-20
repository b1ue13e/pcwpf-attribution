# 20 篇代表性文献独立盲审与多方仲裁信度审计报告 (求真归正版)

> **审计执行日期**：2026-09-08
> **评估样本**：20 篇代表性抽样文献 (覆盖 2023–2026 年中外主流顶级期刊与预印本)
> **参与编码员**：Coder A (第一编码员基准)、Coder B (独立第二编码员)、Coder C (独立第三方验证员)
> **统计执行**：Python `sklearn.metrics.cohen_kappa_score` + 严格渐近方差与置信区间分析

## Scope 1: 全样本 (N=20, 包含排除项统一 N/A 语义)
样本容量：$N = 20$ 篇

| 审计维度 | Coder A vs B ($\kappa$ [95% CI]) | A-B 评级 | Coder A vs C ($\kappa$) | Coder B vs C ($\kappa$) | Fleiss $\kappa$ (3人) | $P_o$ (A-B) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **纳排资格 (Inclusion)** | **1.000** [1.000, 1.000] | Almost Perfect (几乎完全一致) | 1.000 | 1.000 | 1.000 | 100.0% |
| **D1 参数来源 (Provenance)** | **1.000** [1.000, 1.000] | Almost Perfect (几乎完全一致) | 1.000 | 1.000 | 1.000 | 100.0% |
| **D2 施加阶段 (Stage)** | **0.925** [0.781, 1.000] | Almost Perfect (几乎完全一致) | 0.925 | 1.000 | 0.950 | 95.0% |
| **D3 推理依赖 (Inference)** | **0.630** [0.305, 0.954] | Substantial (高度一致) | 0.785 | 0.825 | 0.743 | 80.0% |
| **D4 代码保真 (Fidelity)** | **0.626** [0.135, 1.000] | Substantial (高度一致) | 0.837 | 0.677 | 0.716 | 90.0% |
| **D5 消融类型 (Ablation)** | **0.781** [0.553, 1.000] | Substantial (高度一致) | 0.781 | 1.000 | 0.851 | 85.0% |
| **D5 负面效应 (Harm)** | **1.000** [1.000, 1.000] | Almost Perfect (几乎完全一致) | 1.000 | 1.000 | 1.000 | 100.0% |


## Scope 2: 纳入子集 (N=18, 仅评估风电物理约束实操论文)
样本容量：$N = 18$ 篇

| 审计维度 | Coder A vs B ($\kappa$ [95% CI]) | A-B 评级 | Coder A vs C ($\kappa$) | Coder B vs C ($\kappa$) | Fleiss $\kappa$ (3人) | $P_o$ (A-B) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **D1 参数来源 (Provenance)** | **1.000** [1.000, 1.000] | Almost Perfect (几乎完全一致) | 1.000 | 1.000 | 1.000 | 100.0% |
| **D2 施加阶段 (Stage)** | **0.907** [0.730, 1.000] | Almost Perfect (几乎完全一致) | 0.907 | 1.000 | 0.938 | 94.4% |
| **D3 推理依赖 (Inference)** | **0.500** [0.068, 0.932] | Moderate (中度一致) | 0.684 | 0.769 | 0.646 | 77.8% |
| **D4 代码保真 (Fidelity)** | **-0.029** [-1.000, 1.000] | Poor (一致性极低) | 0.647 | 0.308 | 0.341 | 88.9% |
| **D5 消融类型 (Ablation)** | **0.733** [0.457, 1.000] | Substantial (高度一致) | 0.733 | 1.000 | 0.818 | 83.3% |
| **D5 负面效应 (Harm)** | NaN (Po=100%) | 不可估 (单类别常数边际) | NaN | NaN | NaN | 100.0% |


## 三、20 篇代表性文献最终仲裁共识与依据

经三方独立编码对照，共在 8 处条目存在层级或语义理解分歧。仲裁决议依据原文真实证据裁定如下：

| 文献 | 审计维度 | 初始分歧状况 | 最终仲裁结论 | 仲裁事实与原文出处依据 |
| :---: | :---: | :--- | :---: | :--- |
| **S02** | `D4_fidelity` | A=`code-mismatch`, B=`not-found`, C=`code-mismatch` | **`code-mismatch`** | 官方仓库 train_clean.py 调用 F.kl_div(log p, q) = KL(q||p) 计算反向 JS，与正文对称 JS 定义不一致。 |
| **S10** | `D4_fidelity` | A=`not-found`, B=`unverified`, C=`code-aligned` | **`code-aligned`** | 原预印本与出版物正文 Eq.(1)-(2) 为代数功率与气动等式，代码使用 SimNet sn.Tie 代数节点绑定，与正文设定一致；彻底撤销伪 ODE 冲突指控。 |
| **S15** | `D3_inference_req` | A=`none`, B=`contemporaneous-state`, C=`none` | **`contemporaneous-state`** | Eq.(2) 动态图邻接矩阵依赖各机组当前时刻实时风向角 theta(t)，属于瞬时高阶状态依赖。 |
| **S18** | `D2_stage` | A=`training-loss`, B=`inference-architecture`, C=`inference-architecture` | **`inference-architecture`** | 虽然损失项包含分位数单调性罚项，但核心物理先验（Jensen 尾流亏损）被深度硬编码进动态图拓扑架构，推理需要实时风向。 |
| **S18** | `D3_inference_req` | A=`none`, B=`contemporaneous-state`, C=`none` | **`contemporaneous-state`** | 虽然损失项包含分位数单调性罚项，但核心物理先验（Jensen 尾流亏损）被深度硬编码进动态图拓扑架构，推理需要实时风向。 |
| **S19** | `d5_ablation_type` | A=`module-ablation`, B=`constraint-on-off`, C=`constraint-on-off` | **`constraint-on-off`** | Table 3 严格对照了施加物理区间约束的保形预测与不带物理约束的标准保形预测，属于纯物理约束开关。 |
| **S27** | `D3_inference_req` | A=`none`, B=`forecast-input`, C=`forecast-input` | **`forecast-input`** | UniWind 状态路由网络由未来 NWP 预报风速驱动；消融实验移除了整个状态路由神经网络模块。 |
| **S27** | `d5_ablation_type` | A=`both`, B=`module-ablation`, C=`module-ablation` | **`module-ablation`** | UniWind 状态路由网络由未来 NWP 预报风速驱动；消融实验移除了整个状态路由神经网络模块。 |
| **S34** | `d5_ablation_type` | A=`both`, B=`module-ablation`, C=`module-ablation` | **`module-ablation`** | 消融实验对比的是带有高斯尾流特征变换层的 PGML 与直接输入原始特征的基线模型，消除了整个特征工程模块。 |
| **S39** | `D3_inference_req` | A=`none`, B=`forecast-input`, C=`forecast-input` | **`forecast-input`** | 第一阶段机理模型输入为 WRF 未来预报风速，属于未来预报依赖。 |